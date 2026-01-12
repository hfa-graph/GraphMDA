#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import shutil, pickle, networkx as nx

OUT_BASE = Path(r"XXX/XXXXX")
CLASSES  = [1,2,3,4,5,6]
DOMAINS  = ["target", "source"]
# =================================

TRAIN_FRAC = 0.75
NODE_TOL   = 0.03
RAND_SEED  = 20250923
COPY_FILES = True  
DELETE_ORIGINAL_WINS = True 
VERBOSE    = True

ALPHA_TRAIN_DOM = 1.0
ALPHA_TEST_DOM  = 1.0
BETA_TRAIN_TEST = 0.8

np.random.seed(RAND_SEED)


def _prop_vec(counts):
    tot = counts.sum()
    return counts / tot if tot > 0 else counts

def _save_with_domain_label(src_graph: Path, dst_graph: Path, domain: str):
    if not src_graph.exists(): 
        return
    with open(src_graph, "rb") as f:
        G = pickle.load(f)
    label = 1 if domain == "target" else 0
    nx.set_node_attributes(G, {n: label for n in G.nodes()}, "domain_label")
    dst_graph.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_graph, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

def _vprint(msg: str):
    if VERBOSE:
        print(msg)


def load_domain_windows(k_dir: Path, domain: str) -> pd.DataFrame:
    rows = []
    win_dirs = sorted([p for p in k_dir.glob("win_*_*_*") if p.is_dir()],
                      key=lambda x: int(x.name.split("_")[1]))
    for win_dir in win_dirs:
        graph_name = None
        if domain == "target" and (win_dir / "target_graph.pkl").exists():
            graph_name = "target_graph.pkl"
        elif domain == "source" and (win_dir / "source_graph.pkl").exists():
            graph_name = "source_graph.pkl"
        else:
            continue

        cnt_csv = win_dir / "label_counts.csv"
        if not cnt_csv.exists():
            continue

        df = pd.read_csv(cnt_csv)
        nodes_total = int(df["count"].sum())
        row = {
            "win_id": int(win_dir.name.split("_")[1]),
            "win_dir": str(win_dir),
            "graph_file": graph_name,
            "nodes_total": nodes_total
        }
        for _, r in df.iterrows():
            row[f"count_{int(r['Attack_encoded'])}"] = int(r["count"])
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["win_id","win_dir","graph_file","nodes_total"])

    out = pd.DataFrame(rows).sort_values("win_id").reset_index(drop=True)
    label_cols = [c for c in out.columns if c.startswith("count_")]
    for c in label_cols:
        out[c] = out[c].fillna(0).astype(int)
    return out

def domain_targets(df: pd.DataFrame):
    label_cols = [c for c in df.columns if c.startswith("count_")]
    totals = df[label_cols].sum(axis=0)
    total_nodes = int(totals.sum())
    if total_nodes == 0:
        return None
    p = (totals / total_nodes).fillna(0.0)
    train_nodes_target = int(round(TRAIN_FRAC * total_nodes))
    train_targets = (p * train_nodes_target).round().astype(int)
    return dict(
        label_cols=label_cols, totals=totals, total_nodes=total_nodes,
        p=p, train_nodes_target=train_nodes_target, train_targets=train_targets
    )


def greedy_split_windows(df: pd.DataFrame, meta: dict):
    n = len(df)
    selected = np.zeros(n, dtype=bool)
    label_cols = meta["label_cols"]; totals = meta["totals"]; 
    train_nodes_target = meta["train_nodes_target"]; train_targets = meta["train_targets"]

    cur_nodes = 0
    cur_counts = np.zeros(len(label_cols), dtype=float)
    label_idx = {c:i for i,c in enumerate(label_cols)}

    W_nodes = df["nodes_total"].to_numpy()
    W_counts = np.zeros((n, len(label_cols)), dtype=float)
    for i,c in enumerate(label_cols):
        W_counts[:,i] = df[c].to_numpy()

    order = [c for c in label_cols if totals[c] > 0]
    order.sort(key=lambda c: int(totals[c]))
    for c in order:
        ci = label_idx[c]
        target_c = int(train_targets[c])
        if target_c <= 0:
            continue
        while cur_counts[ci] < target_c and cur_nodes < train_nodes_target:
            best_j, best_gain = -1, -1e18
            rem_before = np.maximum(train_targets.values - cur_counts, 0)
            for j in range(n):
                if selected[j]:
                    continue
                new_counts = cur_counts + W_counts[j]
                rem_after = np.maximum(train_targets.values - new_counts, 0)
                gain = np.linalg.norm(rem_before) - np.linalg.norm(rem_after)
                pred_nodes = cur_nodes + W_nodes[j]
                over_penalty = max(0.0, (pred_nodes - train_nodes_target)/max(1.0, train_nodes_target))
                gain -= 0.1 * over_penalty
                gain += 0.05 * W_counts[j, ci]
                if gain > best_gain:
                    best_gain, best_j = gain, j
            if best_j < 0:
                break
            selected[best_j] = True
            cur_nodes += int(W_nodes[best_j])
            cur_counts += W_counts[best_j]
            if cur_nodes >= train_nodes_target*(1-NODE_TOL):
                break

    return selected

def _score(train_counts, test_counts, p_domain, train_nodes, test_nodes, train_nodes_target):
    s = 0.0
    s += ALPHA_TRAIN_DOM * np.linalg.norm(_prop_vec(train_counts) - p_domain)
    s += ALPHA_TEST_DOM  * np.linalg.norm(_prop_vec(test_counts)  - p_domain)
    s += BETA_TRAIN_TEST * np.linalg.norm(_prop_vec(train_counts) - _prop_vec(test_counts))
    s += 0.3 * abs(train_nodes - train_nodes_target) / max(1, train_nodes_target)
    return float(s)

def refine_by_swaps(df, meta, train_mask, max_rounds=200, node_tol=NODE_TOL):
    label_cols = meta["label_cols"]; p_dom = meta["p"].values; tgt = meta["train_nodes_target"]
    low, high = int((1-node_tol)*tgt), int((1+node_tol)*tgt)
    W_nodes = df["nodes_total"].to_numpy()
    W_counts = df[label_cols].to_numpy(dtype=float)
    total_nodes_all = int(W_nodes.sum())
    tr = train_mask.copy(); te = ~tr

    def agg(mask):
        return (int(W_nodes[mask].sum()), W_counts[mask].sum(axis=0)) if mask.sum()>0 else (0, np.zeros(W_counts.shape[1],dtype=float))

    train_nodes, train_counts = agg(tr)
    test_nodes,  test_counts  = agg(te)
    best = _score(train_counts, test_counts, p_dom, train_nodes, test_nodes, tgt)

    improved, rounds = True, 0
    while improved and rounds < max_rounds:
        improved = False; rounds += 1
        train_idx = np.where(tr)[0]; test_idx = np.where(te)[0]

        for j in list(train_idx) + list(test_idx):
            move_to_train = (j in test_idx)
            new_tr_nodes = train_nodes + (W_nodes[j] if move_to_train else -W_nodes[j])
            if new_tr_nodes < low or new_tr_nodes > high:
                continue
            new_tr_counts = train_counts + (W_counts[j] if move_to_train else -W_counts[j])
            new_te_counts = test_counts  - (W_counts[j] if move_to_train else -W_counts[j])
            sc = _score(new_tr_counts, new_te_counts, p_dom, new_tr_nodes, int(total_nodes_all - new_tr_nodes), tgt)
            if sc + 1e-12 < best:
                best = sc; improved = True
                if move_to_train:
                    tr[j]=True; te[j]=False
                else:
                    tr[j]=False; te[j]=True
                train_nodes, train_counts = new_tr_nodes, new_tr_counts
                test_nodes,  test_counts  = int(total_nodes_all - new_tr_nodes), new_te_counts

    return tr, ~tr

def copy_split_files(k_dir: Path, domain: str, df_split: pd.DataFrame):
    for split in ["train","test"]:
        (k_dir / domain / split).mkdir(parents=True, exist_ok=True)

    for _, r in df_split.iterrows():
        split = r["set"]
        win_dir = Path(r["win_dir"])
        graph_file = r["graph_file"]
        src_graph = win_dir / graph_file
        dst_dir = k_dir / domain / split / win_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)

        _save_with_domain_label(src_graph, dst_dir / graph_file, domain)

        src_counts = win_dir / "label_counts.csv"
        if src_counts.exists():
            shutil.copy2(src_counts, dst_dir / "label_counts.csv")


def delete_original_windows(k_dir: Path):
    wins = [p for p in k_dir.glob("win_*_*_*") if p.is_dir()]
    if not wins:
        _vprint(f"[{k_dir.name}] No deleteable win_*_*_* directory found.")
        return
    for w in wins:
        try:
            shutil.rmtree(w)
            _vprint(f"[{k_dir.name}] Deleted {w.name}")
        except Exception as e:
            print(f"[{k_dir.name}] Delete {w} Fail: {e}")


def run_for_one_domain(k_dir: Path, k: int, domain: str):
    df = load_domain_windows(k_dir, domain)
    if df.empty:
        _vprint(f"[k={k}][{domain}] No windowed data, skip.")
        return
    meta = domain_targets(df)
    if meta is None:
        _vprint(f"[k={k}][{domain}] The total number of nodes is 0, so skip this step.")
        return
    
    # node-level label distribution and train size
    train_mask = greedy_split_windows(df, meta)
    train_mask, test_mask = refine_by_swaps(df, meta, train_mask)

    df_out = df.copy()
    df_out["set"] = np.where(train_mask, "train", "test")
    df_out.to_csv(k_dir / f"splits_{domain}_k{k}.csv", index=False)
    _vprint(f"[k={k}][{domain}] Splits CSV has been generated.")

    if COPY_FILES:
        copy_split_files(k_dir, domain, df_out)
        _vprint(f"[k={k}][{domain}] Train/test files have been copied.")

def main():
    for k in CLASSES:
        k_dir = OUT_BASE / f"k{k}"
        if not k_dir.exists():
            continue
        
        for domain in DOMAINS:
            run_for_one_domain(k_dir, k, domain)

        if DELETE_ORIGINAL_WINS:
            delete_original_windows(k_dir)

if __name__ == "__main__":
    main()
