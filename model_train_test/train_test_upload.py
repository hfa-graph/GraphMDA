#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle, random, itertools, time, json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RGATConv, global_mean_pool
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score

ROOT = Path(r"/XXXXXX/XXX")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIXED_FEAT_DIM = 46
FIXED_REL_MAP = {"same_src": 0, "same_dst": 1}
EPOCHS = 100
LR_FC = 1e-3
LR_D = 1e-3
HIDDEN = 64
HEADS = 2
LAYERS = 2
NUM_CLASS = 2
ADV_WEIGHT = 0.001
LDG_WEIGHT = 0.1
BUDGETS = [200, 1000, 5000]
VAL_FRAC = 0.10
PATIENCE = 10
MIN_DELTA = 1e-4

try:
    from tqdm import tqdm
    def iter_progress(it, desc):
        return tqdm(it, desc=desc, unit="step", leave=False)
    _HAS_TQDM = True
except Exception:
    def iter_progress(it, desc): return it
    _HAS_TQDM = False

def device_str(dev: torch.device) -> str:
    if dev.type == "cuda":
        try:
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            return f"cuda:{idx} - {name}"
        except Exception:
            return "cuda"
    return "cpu"

def format_secs(sec: float) -> str:
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def list_graphs(root_dir: Path, domain: str, split: str) -> List[Path]:
    root = root_dir / domain / split
    if not root.exists(): return []
    out = []
    for win in sorted([p for p in root.glob("win_*_*_*") if p.is_dir()],
                      key=lambda x: int(x.name.split("_")[1])):
        for name in ("source_graph.pkl","target_graph.pkl","graph.pkl"):
            p = win / name
            if p.exists(): out.append(p)
    return out

def to_long(v):
    try: return int(v)
    except Exception:
        return int(np.int64(v)) if isinstance(v, np.generic) else 0

def nx_to_pyg(G: nx.Graph, feat_dim: int, rel_map: Dict[str,int], dom_default: int) -> Data:
    nodes = list(G.nodes())
    idx = {n:i for i,n in enumerate(nodes)}
    N = len(nodes)
    X = np.zeros((N, feat_dim), dtype=np.float32)
    Y = np.full((N,), -1, dtype=np.int64)
    D = np.full((N,), dom_default, dtype=np.int64)
    for n in nodes:
        i = idx[n]
        a = G.nodes[n]
        fv = a.get("feat", None)
        if fv is not None:
            fv = np.asarray(fv, dtype=np.float32).reshape(-1)
            X[i, :min(len(fv), feat_dim)] = fv[:feat_dim]
        if "Label" in a:
            y = to_long(a["Label"])
            Y[i] = y if y in (0,1) else -1
        if "domain_label" in a:
            D[i] = to_long(a["domain_label"])
    e_src, e_dst, e_rel = [], [], []
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        et = d.get("etype", "same_src")
        r = rel_map.get(et, rel_map.get("same_src", 0))
        e_src += [i, j]; e_dst += [j, i]; e_rel += [r, r]
    edge_index = torch.tensor([e_src, e_dst], dtype=torch.long)
    edge_type  = torch.tensor(e_rel, dtype=torch.long)
    data = Data(
        x=torch.tensor(X, dtype=torch.float32),
        y=torch.tensor(Y, dtype=torch.long),
        domain=torch.tensor(D, dtype=torch.long),
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=N,
    )
    data.batch = torch.zeros(N, dtype=torch.long)
    data.graph_domain = torch.tensor([1 if D.mean()>=0.5 else 0], dtype=torch.long)
    return data

class RGATBackbone(nn.Module):
    def __init__(self, in_dim, hid=128, heads=4, layers=2, num_rels=2):
        super().__init__()
        self.blocks = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            conv = RGATConv(d, hid // heads, heads=heads, concat=True,
                            num_relations=num_rels, bias=True)
            proj = nn.Linear(hid, hid)
            self.blocks.append(nn.ModuleDict(dict(conv=conv, proj=proj)))
            d = hid
    def forward(self, x, eidx, etype):
        h = x
        for b in self.blocks:
            h = b["conv"](h, eidx, etype)
            h = F.elu(b["proj"](h))
        return h

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

class DiscMLP(nn.Module):
    def __init__(self, in_dim, hidden=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

class DA_RGAT(nn.Module):
    def __init__(self, in_dim, num_rels, num_classes=2, hid=128, heads=4, layers=2):
        super().__init__()
        self.backbone   = RGATBackbone(in_dim, hid, heads, layers, num_rels)
        self.cls        = MLP(hid, 256, num_classes)
        self.node_disc  = DiscMLP(hid, 256, 2)
        self.graph_disc = DiscMLP(hid, 256, 2)
    def forward(self, data: Data):
        h = self.backbone(data.x, data.edge_index, data.edge_type)
        logits = self.cls(h)
        return h, logits

def save_checkpoint(split_root: Path, budget: int, model: nn.Module, metrics: dict):
    ckpt_dir = split_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_class": "DA_RGAT",
        "state_dict": model.state_dict(),
        "in_dim": FIXED_FEAT_DIM,
        "num_rels": len(FIXED_REL_MAP),
        "rel_map": FIXED_REL_MAP,
        "num_classes": NUM_CLASS,
        "hid": HIDDEN,
        "heads": HEADS,
        "layers": LAYERS,
        "adv_weight": ADV_WEIGHT,
        "ldg_weight": LDG_WEIGHT,
        "epochs": EPOCHS,
        "lr_fc": LR_FC,
        "lr_d": LR_D,
        "device": str(DEVICE),
        "budget": budget,
    }
    torch.save(ckpt, ckpt_dir / f"rgat_budget{budget}.pt")
    with open(ckpt_dir / f"rgat_budget{budget}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

def load_checkpoint(ckpt_path: Path, map_location: str = "cpu") -> nn.Module:
    blob = torch.load(ckpt_path, map_location=map_location)
    model = DA_RGAT(
        in_dim=blob["in_dim"],
        num_rels=blob["num_rels"],
        num_classes=blob["num_classes"],
        hid=blob["hid"],
        heads=blob["heads"],
        layers=blob["layers"],
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model

def build_all_sets(split_root: Path):
    src_tr = list_graphs(split_root, "source", "train")
    tgt_tr = list_graphs(split_root, "target", "train")
    src_te = list_graphs(split_root, "source", "test")
    tgt_te = list_graphs(split_root, "target", "test")
    all_paths = src_tr + tgt_tr + src_te + tgt_te
    if not all_paths:
        raise RuntimeError(f"No subgraph PKL found: {split_root}")
    feat_dim = FIXED_FEAT_DIM
    rel_map  = FIXED_REL_MAP
    def make(paths, def_dom):
        ds=[]
        for p in paths:
            with open(p, "rb") as f: G = pickle.load(f)
            ds.append(nx_to_pyg(G, feat_dim, rel_map, def_dom))
        return ds
    ds_src_tr = make(src_tr, 0)
    ds_tgt_tr = make(tgt_tr, 1)
    ds_src_te = make(src_te, 0)
    ds_tgt_te = make(tgt_te, 1)
    return ds_src_tr, ds_tgt_tr, ds_src_te, ds_tgt_te, feat_dim, len(rel_map)

def split_target_train_for_val(ds_tgt_tr: List[Data], val_frac: float, seed: int = 42) -> Tuple[List[Data], List[Data]]:
    if len(ds_tgt_tr) <= 1 or val_frac <= 0.0:
        return ds_tgt_tr, []
    rng = random.Random(seed)
    idx = list(range(len(ds_tgt_tr)))
    rng.shuffle(idx)
    val_n = max(1, int(round(len(idx) * val_frac)))
    val_ids = set(idx[:val_n])
    tr_ids  = [i for i in idx if i not in val_ids]
    ds_val = [ds_tgt_tr[i] for i in val_ids]
    ds_tr  = [ds_tgt_tr[i] for i in tr_ids]
    return ds_tr, ds_val

def build_target_labeled_masks(ds_tgt_tr: List[Data], budget: int, seed: int = 42):
    rng = random.Random(seed)
    pos, neg = [], []
    for gi, data in enumerate(ds_tgt_tr):
        y = data.y.numpy()
        for ni, yy in enumerate(y):
            if yy == 1: pos.append((gi, ni))
            elif yy == 0: neg.append((gi, ni))
    rng.shuffle(pos); rng.shuffle(neg)
    half = budget // 2
    sel_pos = pos[:half]
    sel_neg = neg[:half]
    if len(sel_pos) < half:
        need = half - len(sel_pos)
        sel_neg += neg[half:half+need]
    if len(sel_neg) < half:
        need = half - len(sel_neg)
        sel_pos += pos[half:half+need]
    selected = sel_pos + sel_neg
    for data in ds_tgt_tr:
        data.labeled_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    for gi, ni in selected:
        if gi < len(ds_tgt_tr):
            ds_tgt_tr[gi].labeled_mask[ni] = True
    for data in ds_tgt_tr:
        if not hasattr(data, "labeled_mask"):
            data.labeled_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

@torch.no_grad()
def evaluate_target_dataset(model: nn.Module, dataset: List[Data]):
    model.eval()
    y_true_all, y_pred_all, y_score_all = [], [], []
    for base in dataset:
        batch = base.to(DEVICE)
        batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=DEVICE)
        if not hasattr(batch, "graph_domain"):
            batch.graph_domain = base.graph_domain.to(DEVICE)
        h, logits = model(batch)
        m = (batch.y >= 0)
        if not m.any():
            continue
        probs = F.softmax(logits[m], dim=-1)[:, 1].detach().cpu().numpy()
        pred  = logits[m].argmax(-1).detach().cpu().numpy()
        y     = batch.y[m].detach().cpu().numpy()
        y_true_all.append(y); y_pred_all.append(pred); y_score_all.append(probs)
    if not y_true_all:
        return dict(
            accuracy=np.nan,
            macro_precision=np.nan, macro_recall=np.nan, macro_f1=np.nan,
            weighted_precision=np.nan, weighted_recall=np.nan, weighted_f1=np.nan,
            auc=np.nan
        )
    y_true  = np.concatenate(y_true_all, axis=0)
    y_pred  = np.concatenate(y_pred_all, axis=0)
    y_score = np.concatenate(y_score_all, axis=0)
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    return dict(
        accuracy=acc,
        macro_precision=mp, macro_recall=mr, macro_f1=mf1,
        weighted_precision=wp, weighted_recall=wr, weighted_f1=wf1,
        auc=auc
    )

def pairwise_epoch_iter(ds_src_tr: List[Data], ds_tgt_tr: List[Data], shuffle=True, seed=42):
    rng = random.Random(seed)
    src_idx = list(range(len(ds_src_tr)))
    tgt_idx = list(range(len(ds_tgt_tr)))
    if shuffle:
        rng.shuffle(src_idx)
        rng.shuffle(tgt_idx)
    L = max(len(src_idx), len(tgt_idx))
    src_cycle = itertools.cycle(src_idx)
    tgt_cycle = itertools.cycle(tgt_idx)
    for _ in range(L):
        yield ds_src_tr[next(src_cycle)], ds_tgt_tr[next(tgt_cycle)]

def toggle_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def _prep_on_device(base: Data) -> Data:
    batch = base.to(DEVICE)
    batch.batch = torch.zeros(batch.num_nodes, dtype=torch.long, device=DEVICE)
    if not hasattr(batch, "graph_domain"):
        batch.graph_domain = base.graph_domain.to(DEVICE)
    if not hasattr(batch, "labeled_mask"):
        batch.labeled_mask = torch.zeros(batch.num_nodes, dtype=torch.bool, device=DEVICE)
    return batch

class EarlyStopper:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.count = 0
        self.best_state = None
        self.best_epoch = 0
    def step(self, value: float, model: nn.Module, epoch: int) -> bool:
        improved = value > self.best + self.min_delta
        if improved:
            self.best = value
            self.count = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
            return False
        else:
            self.count += 1
            return self.count >= self.patience
    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

def train_once(split_root: Path, budget: int):
    print(f"\n========== [{split_root.name}] Start training with target labeled budget = {budget} ==========")
    t0 = time.perf_counter()
    ds_src_tr, ds_tgt_tr_all, ds_src_te, ds_tgt_te, in_dim, num_rels = build_all_sets(split_root)
    for d in ds_src_tr:
        d.labeled_mask = (d.y >= 0)
    ds_tgt_tr, ds_tgt_val = split_target_train_for_val(ds_tgt_tr_all, VAL_FRAC, seed=budget+1234)
    build_target_labeled_masks(ds_tgt_tr, budget)
    model = DA_RGAT(in_dim, num_rels, NUM_CLASS, HIDDEN, HEADS, LAYERS).to(DEVICE)
    opt_FC = torch.optim.AdamW(list(model.backbone.parameters())+list(model.cls.parameters()),
                               lr=LR_FC, weight_decay=1e-4)
    opt_D  = torch.optim.AdamW(list(model.node_disc.parameters())+list(model.graph_disc.parameters()),
                               lr=LR_D, weight_decay=1e-4)
    ce_cls = nn.CrossEntropyLoss()
    ce_dom = nn.CrossEntropyLoss()
    stopper = EarlyStopper(patience=PATIENCE, min_delta=MIN_DELTA)
    for ep in range(1, EPOCHS+1):
        model.train()
        s_loss = s_cls = s_dn = s_dg = s_g = 0.0
        steps = 0
        it = list(pairwise_epoch_iter(ds_src_tr, ds_tgt_tr, shuffle=True, seed=ep+123))
        progress = iter_progress(it, desc=f"{split_root.name} Ep {ep}/{EPOCHS}")
        for src_base, tgt_base in progress:
            src = _prep_on_device(src_base)
            tgt = _prep_on_device(tgt_base)
            toggle_grad(model.node_disc, True)
            toggle_grad(model.graph_disc, True)
            toggle_grad(model.backbone,  False)
            toggle_grad(model.cls,       False)
            with torch.no_grad():
                h_src, _ = model(src)
                h_tgt, _ = model(tgt)
                hg_src = global_mean_pool(h_src, src.batch)
                hg_tgt = global_mean_pool(h_tgt, tgt.batch)
            d_node_src = model.node_disc(h_src.detach())
            d_node_tgt = model.node_disc(h_tgt.detach())
            d_graph_src = model.graph_disc(hg_src.detach())
            d_graph_tgt = model.graph_disc(hg_tgt.detach())
            L_dn = ce_dom(d_node_src, src.domain.long()) + ce_dom(d_node_tgt, tgt.domain.long())
            L_dg = ce_dom(d_graph_src, src.graph_domain.long()) + ce_dom(d_graph_tgt, tgt.graph_domain.long())
            loss_D = L_dn + LDG_WEIGHT * L_dg
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            toggle_grad(model.node_disc, False)
            toggle_grad(model.graph_disc, False)
            toggle_grad(model.backbone,  True)
            toggle_grad(model.cls,       True)
            h_src, logits_src = model(src)
            h_tgt, logits_tgt = model(tgt)
            hg_src = global_mean_pool(h_src, src.batch)
            hg_tgt = global_mean_pool(h_tgt, tgt.batch)
            m_src = src.labeled_mask
            m_tgt = tgt.labeled_mask
            L_cls_src = ce_cls(logits_src[m_src], src.y[m_src]) if m_src.any() else torch.tensor(0., device=DEVICE)
            L_cls_tgt = ce_cls(logits_tgt[m_tgt], tgt.y[m_tgt]) if m_tgt.any() else torch.tensor(0., device=DEVICE)
            L_cls = 0.1 * L_cls_src + 1.0 * L_cls_tgt
            inv_src_dom = 1 - src.domain.long()
            inv_tgt_dom = 1 - tgt.domain.long()
            inv_src_g   = 1 - src.graph_domain.long()
            inv_tgt_g   = 1 - tgt.graph_domain.long()
            d_node_src_g = model.node_disc(h_src)
            d_node_tgt_g = model.node_disc(h_tgt)
            d_graph_src_g = model.graph_disc(hg_src)
            d_graph_tgt_g = model.graph_disc(hg_tgt)
            L_g = (
                ce_dom(d_node_src_g, inv_src_dom) + ce_dom(d_node_tgt_g, inv_tgt_dom) +
                ce_dom(d_graph_src_g, inv_src_g)   + ce_dom(d_graph_tgt_g, inv_tgt_g)
            )
            loss_G = L_cls + ADV_WEIGHT * L_g
            opt_FC.zero_grad()
            loss_G.backward()
            opt_FC.step()
            s_loss += float(loss_G.item())
            s_cls  += float(L_cls.item())
            s_dn   += float(L_dn.item())
            s_dg   += float(L_dg.item())
            s_g    += float(L_g.item())
            steps  += 1
            if _HAS_TQDM:
                progress.set_postfix(loss=f"{loss_G.item():.4f}",
                                     cls=f"{L_cls.item():.4f}",
                                     d_node=f"{L_dn.item():.4f}",
                                     d_graph=f"{L_dg.item():.4f}",
                                     Lg=f"{L_g.item():.4f}")
        model.eval()
        val_metrics = {"macro_f1": float("nan")}
        if ds_tgt_val:
            val_metrics = evaluate_target_dataset(model, ds_tgt_val)
            print(f"[{split_root.name} Ep {ep:02d}] TrainAvg: lossG={s_loss/max(steps,1):.4f} | cls={s_cls/max(steps,1):.4f} | d_node={s_dn/max(steps,1):.4f} | d_graph={s_dg/max(steps,1):.4f} | Lg={s_g/max(steps,1):.4f} || VAL MacroF1={val_metrics['macro_f1']:.4f}")
        else:
            print(f"[{split_root.name} Ep {ep:02d}] TrainAvg: lossG={s_loss/max(steps,1):.4f} | cls={s_cls/max(steps,1):.4f} | d_node={s_dn/max(steps,1):.4f} | d_graph={s_dg/max(steps,1):.4f} | Lg={s_g/max(steps,1):.4f} || VAL MacroF1=nan (no val set)")
        if ds_tgt_val:
            should_stop = stopper.step(val_metrics["macro_f1"], model, ep)
            if should_stop:
                print(f"[{split_root.name}] Early stopped at epoch {ep} (best VAL MacroF1={stopper.best:.4f} @ epoch {stopper.best_epoch})")
                break
    if ds_tgt_val and stopper.best_state is not None:
        stopper.restore_best(model)
    train_dur = time.perf_counter() - t0
    if not ds_tgt_te:
        print(f"[{split_root.name}] No target-test data found. Skipping evaluation.")
        res = dict(
            budget=budget,
            accuracy=np.nan,
            macro_precision=np.nan, macro_recall=np.nan, macro_f1=np.nan,
            weighted_precision=np.nan, weighted_recall=np.nan, weighted_f1=np.nan,
            auc=np.nan,
            train_time_sec=train_dur
        )
        save_checkpoint(split_root, budget, model, res)
        print(f"[{split_root.name}] Saved checkpoint (no test): {(split_root/'checkpoints'/f'rgat_budget{budget}.pt').as_posix()}")
        print(f"[{split_root.name} budget={budget}] Train time = {format_secs(train_dur)} ({train_dur:.2f}s)")
        return res
    metrics = evaluate_target_dataset(model, ds_tgt_te)
    res = dict(budget=budget, **metrics, train_time_sec=train_dur)
    print(
        f"[{split_root.name} Target-Test @ budget={budget}] "
        f"ACC={metrics['accuracy']:.4f} | "
        f"Macro(P/R/F1)={metrics['macro_precision']:.4f}/{metrics['macro_recall']:.4f}/{metrics['macro_f1']:.4f} | "
        f"Weighted(P/R/F1)={metrics['weighted_precision']:.4f}/{metrics['weighted_recall']:.4f}/{metrics['weighted_f1']:.4f} | "
        f"AUC={metrics['auc']:.4f}"
    )
    print(f"[{split_root.name} budget={budget}] Train time = {format_secs(train_dur)} ({train_dur:.2f}s)")
    save_checkpoint(split_root, budget, model, res)
    print(f"[{split_root.name}] Saved checkpoint: { (split_root/'checkpoints'/f'rgat_budget{budget}.pt').as_posix() }")
    return res

if __name__ == "__main__":
    print(f"Using device: {device_str(DEVICE)}")
    print(f"Fixed feature dim = {FIXED_FEAT_DIM}, relation map = {FIXED_REL_MAP}")
    base_dir = ROOT.parent
    split_dirs = sorted(
        [d for d in base_dir.glob("k*_preprocessed_ce") if d.is_dir() and d.name[1:].split('_')[0].isdigit()],
        key=lambda p: int(p.name.split("_")[0][1:])
    )
    if not split_dirs:
        raise SystemExit(f"No split directories found under: {base_dir}/k*_preprocessed_ce")
    all_summaries: Dict[str, List[Dict]] = {}
    grand_t0 = time.perf_counter()
    grand_train_seconds = 0.0
    for split_root in split_dirs:
        per_split_results = []
        print(f"\n===== Running split: {split_root.name} =====")
        for K in BUDGETS:
            res = train_once(split_root, K)
            per_split_results.append(res)
            if isinstance(res.get("train_time_sec"), (int, float)):
                grand_train_seconds += float(res["train_time_sec"])
        all_summaries[split_root.name] = per_split_results
        print(f"\n--- Summary on Target-Test [{split_root.name}] ---")
        print("budget\tACC\tMacroF1\tWeightedF1\tAUC\tTrainTime(s)")
        for r in per_split_results:
            def _fmt(v):
                try: return f"{v:.4f}"
                except Exception: return "nan"
            tsec = r.get("train_time_sec", float("nan"))
            tsec_str = f"{tsec:.2f}" if isinstance(tsec, (int,float)) else "nan"
            print(
                f"{r['budget']}\t"
                f"{_fmt(r.get('accuracy'))}\t"
                f"{_fmt(r.get('macro_f1'))}\t"
                f"{_fmt(r.get('weighted_f1'))}\t"
                f"{_fmt(r.get('auc'))}\t"
                f"{tsec_str}"
            )
    grand_wall_seconds = time.perf_counter() - grand_t0
    print("\n===== Cross-Split Summary (MacroF1 / WeightedF1 / AUC) =====")
    headers = ["split"] + [f"MacroF1@{b}" for b in BUDGETS] + [f"WtdF1@{b}" for b in BUDGETS] + [f"AUC@{b}" for b in BUDGETS]
    print("\t".join(headers))
    for split_name, rows in all_summaries.items():
        macro_cols, wtd_cols, auc_cols = [], [], []
        for b in BUDGETS:
            rec = next((r for r in rows if r["budget"] == b), None)
            macro_cols.append(f"{rec['macro_f1']:.4f}" if rec and isinstance(rec.get('macro_f1'), (int,float)) else "nan")
            wtd_cols.append(f"{rec['weighted_f1']:.4f}" if rec and isinstance(rec.get('weighted_f1'), (int,float)) else "nan")
            auc_cols.append(f"{rec['auc']:.4f}" if rec and isinstance(rec.get('auc'), (int,float)) else "nan")
        print("\t".join([split_name] + macro_cols + wtd_cols + auc_cols))
    print("\n===== Cross-Split Averages by Budget (ACC / MacroP / MacroR / MacroF1 / WtdP / WtdR / WtdF1 / AUC) =====")
    print("budget\tACC\tMacroP\tMacroR\tMacroF1\tWtdP\tWtdR\tWtdF1\tAUC")
    def _fmt_avg(v):
        try:
            if np.isnan(v): return "nan"
            return f"{v:.4f}"
        except Exception:
            return "nan"
    for b in BUDGETS:
        accs, mp, mr, mf1, wp, wr, wf1, aucs = [], [], [], [], [], [], [], []
        for rows in all_summaries.values():
            rec = next((r for r in rows if r["budget"] == b), None)
            if rec is None:
                continue
            accs.append(rec.get("accuracy", np.nan))
            mp.append(rec.get("macro_precision", np.nan))
            mr.append(rec.get("macro_recall", np.nan))
            mf1.append(rec.get("macro_f1", np.nan))
            wp.append(rec.get("weighted_precision", np.nan))
            wr.append(rec.get("weighted_recall", np.nan))
            wf1.append(rec.get("weighted_f1", np.nan))
            aucs.append(rec.get("auc", np.nan))
        print(
            f"{b}\t"
            f"{_fmt_avg(np.nanmean(accs) if accs else np.nan)}\t"
            f"{_fmt_avg(np.nanmean(mp)   if mp   else np.nan)}\t"
            f"{_fmt_avg(np.nanmean(mr)   if mr   else np.nan)}\t"
            f"{_fmt_avg(np.nanmean(mf1)  if mf1  else np.nan)}\t"
            f"{_fmt_avg(np.nanmean(wp)   if wp   else np.nan)}\t"
            f"{_fmt_avg(np.nanmean(wr)   if wr   else np.nan)}\t"
            f"{_fmt_avg(np.nanmean(wf1)  if wf1  else np.nan)}\t"
            f"{_fmt_avg(np.nanmean(aucs) if aucs else np.nan)}"
        )
    acc_all, mp_all, mr_all, mf1_all, wp_all, wr_all, wf1_all, auc_all = [], [], [], [], [], [], [], []
    for rows in all_summaries.values():
        for rec in rows:
            acc_all.append(rec.get("accuracy", np.nan))
            mp_all.append(rec.get("macro_precision", np.nan))
            mr_all.append(rec.get("macro_recall", np.nan))
            mf1_all.append(rec.get("macro_f1", np.nan))
            wp_all.append(rec.get("weighted_precision", np.nan))
            wr_all.append(rec.get("weighted_recall", np.nan))
            wf1_all.append(rec.get("weighted_f1", np.nan))
            auc_all.append(rec.get("auc", np.nan))
    print("\n===== Overall Averages across all k & budgets =====")
    print("ACC\tMacroP\tMacroR\tMacroF1\tWtdP\tWtdR\tWtdF1\tAUC")
    print(
        f"{_fmt_avg(np.nanmean(acc_all))}\t"
        f"{_fmt_avg(np.nanmean(mp_all))}\t"
        f"{_fmt_avg(np.nanmean(mr_all))}\t"
        f"{_fmt_avg(np.nanmean(mf1_all))}\t"
        f"{_fmt_avg(np.nanmean(wp_all))}\t"
        f"{_fmt_avg(np.nanmean(wr_all))}\t"
        f"{_fmt_avg(np.nanmean(wf1_all))}\t"
        f"{_fmt_avg(np.nanmean(auc_all))}"
    )
    print("\n===== Training Time Summary =====")
    grand_wall_seconds = time.perf_counter() - grand_t0
    print(f"Total TRAIN time (sum of all split+budget runs): {format_secs(grand_train_seconds)} ({grand_train_seconds:.2f}s)")
    print(f"Total WALL time (train + eval + logging):       {format_secs(grand_wall_seconds)} ({grand_wall_seconds:.2f}s)")
