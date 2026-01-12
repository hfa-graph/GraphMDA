#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path
import networkx as nx
import pandas as pd
from collections import Counter
import shutil
import random

IN_BASE  = Path(r"XXX/out_graphs_180")
OUT_BASE = Path(r"XXX/out_splits_strict_binary_src_tgt_180")

# Leave-one-out strategy:
# For each attack type k (0 means benign), target windows contain only labels in {0, k} and must include k;
# source windows contain no k (include other attacks). Mixed windows containing k and other attacks are discarded.
CLASSES  = [1, 2, 3, 4, 5, 6]
REQUIRE_BOTH_0_AND_K = False
PURE_BENIGN_TO_TARGET_RATIO = 0  
RANDOM_SEED = 42  


if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

def load_graph(win_dir: Path) -> nx.MultiGraph:
    with open(win_dir / "graph.pkl", "rb") as f:
        return pickle.load(f)

def save_graph(G: nx.MultiGraph, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

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

def labels_and_counts(G: nx.MultiGraph):
    lbls = nx.get_node_attributes(G, "Attack_encoded")
    if not lbls:
        return None, Counter()
    S = set(int(v) for v in lbls.values())
    C = Counter(int(v) for v in lbls.values())
    return S, C

def save_label_counts_csv(counter: Counter, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(sorted(counter.items()), columns=["Attack_encoded", "count"])
    df.to_csv(out_path, index=False)

def domain_ratio_df(counter: Counter):
    total = sum(counter.values())
    if total == 0:
        return pd.DataFrame(columns=["label", "count", "ratio"])
    rows = []
    for lab in sorted(counter.keys()):
        cnt = counter[lab]
        ratio = cnt / total
        rows.append({"label": lab, "count": cnt, "ratio": ratio})
    return pd.DataFrame(rows)

def process_for_k(k: int):
    out_k = OUT_BASE / f"k{k}"
    out_k.mkdir(parents=True, exist_ok=True)
    rows = []

    src_label_counter = Counter()
    tgt_label_counter = Counter()
    src_windows = 0
    tgt_windows = 0
    discard_windows = 0

    for win_dir in window_dirs_sorted(IN_BASE):
        try:
            wid = int(win_dir.name.split("_")[1])
        except Exception:
            continue

        pkl = win_dir / "graph.pkl"
        if not pkl.exists():
            rows.append({"win_id": wid, "ok": 0, "action": "discard", "reason": "missing_graph"})
            discard_windows += 1
            continue

        try:
            G = load_graph(win_dir)
        except Exception as e:
            rows.append({"win_id": wid, "ok": 0, "action": "discard", "reason": f"load_error:{e}"})
            discard_windows += 1
            continue

        S, C = labels_and_counts(G)
        if S is None:
            rows.append({"win_id": wid, "ok": 0, "action": "discard", "reason": "no_Attack_encoded"})
            discard_windows += 1
            continue

        has_k   = (k in S)
        has_0   = (0 in S)
        subset  = S.issubset({0, k})

        out_dir = out_k / win_dir.name

        if S == {0}:
            if random.random() < PURE_BENIGN_TO_TARGET_RATIO:
                save_graph(G, out_dir / "target_graph.pkl")
                save_label_counts_csv(C, out_dir / "label_counts.csv")
                tgt_label_counter.update(C)
                tgt_windows += 1

                row = {
                    "win_id": wid, "ok": 1, "action": "target_pure0",
                    "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                    "labels": "0", "count_0": C[0]
                }
                rows.append(row)
            else:
                save_graph(G, out_dir / "source_graph.pkl")
                save_label_counts_csv(C, out_dir / "label_counts.csv")
                src_label_counter.update(C)
                src_windows += 1

                row = {
                    "win_id": wid, "ok": 1, "action": "source_pure0",
                    "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                    "labels": "0", "count_0": C[0]
                }
                rows.append(row)
            continue

        if has_k and subset and (has_0 or not REQUIRE_BOTH_0_AND_K):
            save_graph(G, out_dir / "target_graph.pkl")
            save_label_counts_csv(C, out_dir / "label_counts.csv")
            tgt_label_counter.update(C)
            tgt_windows += 1

            row = {
                "win_id": wid, "ok": 1, "action": "target",
                "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                "labels": ",".join(map(str, sorted(S)))
            }
            for lab, cnt in C.items():
                row[f"count_{lab}"] = cnt
            row.setdefault("count_0", 0)
            row.setdefault(f"count_{k}", 0)
            rows.append(row)
            continue

        if not has_k:
            save_graph(G, out_dir / "source_graph.pkl")
            save_label_counts_csv(C, out_dir / "label_counts.csv")
            src_label_counter.update(C)
            src_windows += 1

            row = {
                "win_id": wid, "ok": 1, "action": "source",
                "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                "labels": ",".join(map(str, sorted(S)))
            }
            for lab, cnt in C.items():
                row[f"count_{lab}"] = cnt
            rows.append(row)
            continue

        reason = f"labels={sorted(S)} not subset of {{0,{k}}}"
        if REQUIRE_BOTH_0_AND_K and not has_0:
            reason += ";no_benign_0"
        rows.append({
            "win_id": wid, "ok": 0, "action": "discard",
            "reason": reason, "labels": ",".join(map(str, sorted(S)))
        })
        discard_windows += 1

    manifest_df = pd.DataFrame(rows).sort_values("win_id")
    manifest_df.to_csv(out_k / f"manifest_k{k}.csv", index=False)

    src_df = domain_ratio_df(src_label_counter)
    tgt_df = domain_ratio_df(tgt_label_counter)

    all_labels = sorted(set(src_df["label"].tolist()) | set(tgt_df["label"].tolist()))
    src_df = src_df.set_index("label").reindex(all_labels, fill_value=0).reset_index()
    tgt_df = tgt_df.set_index("label").reindex(all_labels, fill_value=0).reset_index()

    summary_df = pd.DataFrame({
        "label": all_labels,
        "source_count": [src_label_counter.get(l, 0) for l in all_labels],
        "source_ratio": [float(src_df[src_df["label"]==l]["ratio"]) if l in src_df["label"].values else 0.0 for l in all_labels],
        "target_count": [tgt_label_counter.get(l, 0) for l in all_labels],
        "target_ratio": [float(tgt_df[tgt_df["label"]==l]["ratio"]) if l in tgt_df["label"].values else 0.0 for l in all_labels],
    })

    meta_row = pd.DataFrame([{
        "label": "___TOTAL___",
        "source_count": sum(src_label_counter.values()),
        "source_ratio": 1.0 if sum(src_label_counter.values())>0 else 0.0,
        "target_count": sum(tgt_label_counter.values()),
        "target_ratio": 1.0 if sum(tgt_label_counter.values())>0 else 0.0,
    }])
    summary_out = pd.concat([summary_df, meta_row], ignore_index=True)
    summary_out.to_csv(out_k / f"summary_k{k}.csv", index=False)

    print(f"[k={k}] target_windows={tgt_windows}, source_windows={src_windows}, discard={discard_windows} → {out_k}")

    return {
        "k": k,
        "target_windows": tgt_windows,
        "source_windows": src_windows,
        "discard_windows": discard_windows,
        "target_labels_total": sum(tgt_label_counter.values()),
        "source_labels_total": sum(src_label_counter.values())
    }

def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    overview_rows = []
    for k in CLASSES:
        info = process_for_k(k)
        overview_rows.append(info)

    overview_df = pd.DataFrame(overview_rows, columns=[
        "k", "target_windows", "source_windows", "discard_windows",
        "target_labels_total", "source_labels_total"
    ])
    overview_df.to_csv(OUT_BASE / "manifest_summary_all.csv", index=False)

    print("All complete. Summary entry:", OUT_BASE / "manifest_summary_all.csv")

    if IN_BASE.exists():
        try:
            shutil.rmtree(IN_BASE)
            print(f"The original directory has been deleted: {IN_BASE}")
        except Exception as e:
            print(f"Delete {IN_BASE} Fail: {e}")

if __name__ == "__main__":
    main()
