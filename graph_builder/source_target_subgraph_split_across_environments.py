#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path
from collections import Counter
import networkx as nx
import pandas as pd

IN_CICIDS = Path(r"/XXX/out_graphs_cicids2018")
IN_BOT    = Path(r"/XXX/out_graphs_BoT")
IN_TON    = Path(r"/XXX/out_graphs_ToN")

OUT_ROOT  = Path(r"/XXX/differ_network")



def window_dirs_sorted(base: Path):
    wins = []
    for p in base.glob("win_*_*_*"):
        if not p.is_dir():
            continue
        try:
            wid = int(p.name.split("_")[1])
            wins.append((wid, p))
        except Exception:
            continue
    return [p for _, p in sorted(wins, key=lambda x: x[0])]

def load_graph(pkl_path: Path) -> nx.MultiGraph:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def save_graph_with_domain_label(G: nx.MultiGraph, out_path: Path, domain_label: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # domain_label: source=0, target=1
    nx.set_node_attributes(G, {n: domain_label for n in G.nodes()}, "domain_label")
    with open(out_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

def labels_and_counts(G: nx.MultiGraph):
    lbls = nx.get_node_attributes(G, "Attack_encoded")
    if not lbls:
        return set(), Counter()
    def to_int(v):
        try:
            return int(v)
        except Exception:
            return int(float(v))
    S = set(to_int(v) for v in lbls.values())
    C = Counter(to_int(v) for v in lbls.values())
    return S, C

def save_label_counts_csv(counter: Counter, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(sorted(counter.items()), columns=["Attack_encoded", "count"])
    df.to_csv(out_csv, index=False)

def ratio_df(counter: Counter) -> pd.DataFrame:
    total = sum(counter.values())
    if total == 0:
        return pd.DataFrame(columns=["label", "count", "ratio"])
    rows = [{"label": lab, "count": cnt, "ratio": cnt/total} for lab, cnt in sorted(counter.items())]
    return pd.DataFrame(rows)

def copy_dataset_as_domain(split_dir: Path, domain_name: str, in_base: Path, domain_label: int):
    assert domain_name in ("source", "target")
    rows = []
    label_counter = Counter()
    out_domain_dir = split_dir / domain_name
    out_domain_dir.mkdir(parents=True, exist_ok=True)

    if not in_base.exists():
        print(f"[WARN] The input directory does not exist:{in_base}")
        return rows, label_counter

    for win_dir in window_dirs_sorted(in_base):
        pkl = win_dir / "graph.pkl"
        if not pkl.exists():
            rows.append({
                "win_id": int(win_dir.name.split("_")[1]) if "_" in win_dir.name else -1,
                "ok": 0, "domain": domain_name, "action": "discard", "reason": "missing_graph",
                "src_dir": str(win_dir)
            })
            continue

        try:
            G = load_graph(pkl)
        except Exception as e:
            rows.append({
                "win_id": int(win_dir.name.split("_")[1]) if "_" in win_dir.name else -1,
                "ok": 0, "domain": domain_name, "action": "discard", "reason": f"load_error:{e}",
                "src_dir": str(win_dir)
            })
            continue

        S, C = labels_and_counts(G)
        dst_dir = out_domain_dir / win_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)

        out_name = "source_graph.pkl" if domain_name == "source" else "target_graph.pkl"
        save_graph_with_domain_label(G, dst_dir / out_name, domain_label)
        save_label_counts_csv(C, dst_dir / "label_counts.csv")
        label_counter.update(C)

        rows.append({
            "win_id": int(win_dir.name.split("_")[1]) if "_" in win_dir.name else -1,
            "ok": 1, "domain": domain_name, "action": "copy",
            "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "labels": ",".join(map(str, sorted(S))),
            "src_dir": str(win_dir)
        })

    return rows, label_counter

def process_split(split_name: str, src_base: Path, tgt_base: Path, out_root: Path):
    split_dir = out_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    rows_src, counter_src = copy_dataset_as_domain(split_dir, "source", src_base, domain_label=0)
    rows_tgt, counter_tgt = copy_dataset_as_domain(split_dir, "target", tgt_base, domain_label=1)

    # manifest
    manifest_df = pd.DataFrame(rows_src + rows_tgt).sort_values(["domain", "win_id"])
    manifest_df.to_csv(split_dir / f"manifest_{split_name}.csv", index=False)

    # summary
    src_df = ratio_df(counter_src).rename(columns={"label": "attack", "count": "source_count", "ratio": "source_ratio"})
    tgt_df = ratio_df(counter_tgt).rename(columns={"label": "attack", "count": "target_count", "ratio": "target_ratio"})

    attacks = sorted(set(src_df["attack"].tolist()) | set(tgt_df["attack"].tolist()))
    src_df = src_df.set_index("attack").reindex(attacks, fill_value=0).reset_index()
    tgt_df = tgt_df.set_index("attack").reindex(attacks, fill_value=0).reset_index()

    summary = pd.DataFrame({
        "attack": attacks,
        "source_count": [int(src_df.loc[src_df["attack"]==a, "source_count"].values[0]) for a in attacks],
        "source_ratio": [float(src_df.loc[src_df["attack"]==a, "source_ratio"].values[0]) for a in attacks],
        "target_count": [int(tgt_df.loc[tgt_df["attack"]==a, "target_count"].values[0]) for a in attacks],
        "target_ratio": [float(tgt_df.loc[tgt_df["attack"]==a, "target_ratio"].values[0]) for a in attacks],
    })

    meta_row = pd.DataFrame([{
        "attack": "___TOTAL___",
        "source_count": sum(counter_src.values()),
        "source_ratio": 1.0 if sum(counter_src.values())>0 else 0.0,
        "target_count": sum(counter_tgt.values()),
        "target_ratio": 1.0 if sum(counter_tgt.values())>0 else 0.0,
    }])

    pd.concat([summary, meta_row], ignore_index=True).to_csv(split_dir / f"summary_{split_name}.csv", index=False)

    print(f"[{split_name}] source({src_base.name}) windows={len(rows_src)} | "
          f"target({tgt_base.name}) windows={len(rows_tgt)} → {split_dir}")

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("k1", IN_CICIDS, IN_BOT),  # k1: CIC -> source, BoT -> target
        ("k2", IN_BOT,    IN_TON),  # k2: BoT -> source, ToN -> target
        ("k3", IN_TON,    IN_CICIDS) # k3: ToN -> source, CIC -> target
    ]

    for split_name, src_base, tgt_base in pairs:
        process_split(split_name, src_base, tgt_base, OUT_ROOT)

    print(f"Complete. Output located at:{OUT_ROOT}")

if __name__ == "__main__":
    main()
