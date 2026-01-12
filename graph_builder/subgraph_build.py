#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import networkx as nx
import pickle
import time                              
from itertools import combinations
from collections import defaultdict
from pathlib import Path

INPUT_CSV = r"XXX/XXX.csv"
OUTDIR    = r"XXX/out_graphs_180"

# Each flow becomes a node; edges encode same_src and same_dst relations within a window.
WINDOW_SECONDS     = 180
CHUNK_ROWS         = 1_000_000
CLIQUE_EDGES       = False
MAX_ACTIVE_WINDOWS = 200        
SAFE_GAP_WINS      = 2           


WIN_MS = WINDOW_SECONDS * 1000

NODE_ATTRS = [
    "PROTOCOL","L7_PROTO",
    "IN_BYTES","IN_PKTS","OUT_BYTES","OUT_PKTS",
    "TCP_FLAGS","CLIENT_TCP_FLAGS","SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS","DURATION_IN","DURATION_OUT",
    "MIN_TTL","MAX_TTL","LONGEST_FLOW_PKT","SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN","MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES","DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES","RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES","RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT","DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES","NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES","NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN","TCP_WIN_MAX_OUT",
    "ICMP_TYPE","ICMP_IPV4_TYPE",
    "DNS_QUERY_TYPE","DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN","SRC_TO_DST_IAT_MAX","SRC_TO_DST_IAT_AVG","SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN","DST_TO_SRC_IAT_MAX","DST_TO_SRC_IAT_AVG","DST_TO_SRC_IAT_STDDEV",
    "Label", "Attack_encoded"
]

USECOLS = sorted(set(
    ["FLOW_START_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT"] + NODE_ATTRS
))


def add_group_edges(G: nx.MultiGraph, ids, etype: str, clique: bool):
    ids = sorted(ids)
    if len(ids) < 2:
        return
    if clique:
        for u, v in combinations(ids, 2):
            G.add_edge(u, v, etype=etype)
    else:
        c = ids[0]
        for v in ids[1:]:
            u, w = (c, v) if c < v else (v, c)
            G.add_edge(u, w, etype=etype)


def save_graph_safe_merge(out_path: Path, G_new: nx.MultiGraph):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        try:
            with open(out_path, "rb") as f:
                G_old = pickle.load(f)
            G_merged = nx.disjoint_union(G_old, G_new)
            with open(out_path, "wb") as f:
                pickle.dump(G_merged, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            alt = out_path.with_name(out_path.stem + "_part.pkl")
            with open(alt, "wb") as f:
                pickle.dump(G_new, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(out_path, "wb") as f:
            pickle.dump(G_new, f, protocol=pickle.HIGHEST_PROTOCOL)


def flush_window(win_store, outdir: Path, wid: int, min_ts: int, manifest_rows: list):
    store = win_store.pop(wid, None)
    if not store:
        return
    G = store["graph"]
    if G is None or G.number_of_nodes() == 0:
        return
    
    if G.number_of_edges() == 0:
        for n in G.nodes():
            G.add_edge(n, n, etype="same_src")
            G.add_edge(n, n, etype="same_dst")
    else:
        for n, d in dict(G.degree()).items():
            if d == 0:
                G.add_edge(n, n, etype="same_src")
                G.add_edge(n, n, etype="same_dst")


    t0 = int(min_ts + wid * WIN_MS)
    t1 = int(t0 + WIN_MS)
    subdir = outdir / f"win_{wid}_{t0}_{t1}"
    subdir.mkdir(parents=True, exist_ok=True)

    save_graph_safe_merge(subdir / "graph.pkl", G)

    build_time = store.get("build_time", 0.0)  

    manifest_rows.append({
        "win_id": wid,
        "t0_ms": t0,
        "t1_ms": t1,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "build_time_seconds": build_time      
    })

    print(f"[Flush] win_id={wid}, nodes={G.number_of_nodes()}, "
          f"edges={G.number_of_edges()}, build_time={build_time:.4f}s")  


def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[Pass0] Scanning minimum FLOW_START_MILLISECONDS ...")
    min_ts, total_rows = None, 0
    for ch in pd.read_csv(INPUT_CSV, usecols=["FLOW_START_MILLISECONDS"],
                          chunksize=CHUNK_ROWS, low_memory=False):
        total_rows += len(ch)
        v = ch["FLOW_START_MILLISECONDS"].min()
        min_ts = v if min_ts is None else min(min_ts, v)
    if min_ts is None:
        print("Empty file.")
        return
    print(f"[Pass0] total rows ≈ {total_rows}, min_ts={min_ts}")

    win_store = {}
    manifest_rows = []
    max_seen_wid = -1  

    def ensure_window(wid: int):
        if wid not in win_store:
            win_store[wid] = {
                "graph": nx.MultiGraph(),
                "next_node_id": 0,
                "src_map": defaultdict(list),
                "dst_map": defaultdict(list),
                "build_time": 0.0           
            }
        return win_store[wid]

    def try_flush_completed_windows():

        if max_seen_wid < 0:
            return
        watermark = max_seen_wid - SAFE_GAP_WINS
        done_wids = [w for w in win_store.keys() if w <= watermark]
        for w in sorted(done_wids):
            flush_window(win_store, outdir, w, min_ts, manifest_rows)

    print("[Pass1] Building windowed graphs (safe) ...")
    processed = 0
    reader = pd.read_csv(INPUT_CSV, usecols=USECOLS,
                         chunksize=CHUNK_ROWS, low_memory=False)

    for cidx, df in enumerate(reader, 1):
        if df.empty:
            continue

        processed += len(df)
        df = df.copy()
        df["__win_id"] = ((df["FLOW_START_MILLISECONDS"] - min_ts) // WIN_MS).astype(np.int64)
        max_seen_wid = max(max_seen_wid, int(df["__win_id"].max()))

        for wid, g in df.groupby("__win_id", sort=True):
            wid = int(wid)
            store = ensure_window(wid)

            t_win_start = time.perf_counter()  

            G = store["graph"]
            base = store["next_node_id"]

            g = g.reset_index(drop=True)
            local_count = len(g)
            g["__node_id"] = np.arange(base, base + local_count, dtype=np.int64)
            store["next_node_id"] += local_count

            for _, row in g.iterrows():
                nid = int(row["__node_id"])
                attrs = {k: row[k] for k in NODE_ATTRS if k in row}
                G.add_node(nid, **attrs)

            for src_ip, sub in g.groupby("IPV4_SRC_ADDR")["__node_id"]:
                new_ids = list(map(int, sub.values.tolist()))
                add_group_edges(G, new_ids, "same_src", CLIQUE_EDGES)
                old_ids = store["src_map"][src_ip]
                if old_ids:
                    if CLIQUE_EDGES:
                        for u in new_ids:
                            for v in old_ids:
                                a, b = (u, v) if u < v else (v, u)
                                G.add_edge(a, b, etype="same_src")
                    else:
                        center = old_ids[0]
                        for u in new_ids:
                            a, b = (center, u) if center < u else (u, center)
                            G.add_edge(a, b, etype="same_src")
                store["src_map"][src_ip].extend(new_ids)

            for key, sub in g.groupby(["IPV4_DST_ADDR", "L4_DST_PORT"])["__node_id"]:
                new_ids = list(map(int, sub.values.tolist()))
                add_group_edges(G, new_ids, "same_dst", CLIQUE_EDGES)
                old_ids = store["dst_map"][key]
                if old_ids:
                    if CLIQUE_EDGES:
                        for u in new_ids:
                            for v in old_ids:
                                a, b = (u, v) if u < v else (v, u)
                                G.add_edge(a, b, etype="same_dst")
                    else:
                        center = old_ids[0]
                        for u in new_ids:
                            a, b = (center, u) if center < u else (u, center)
                            G.add_edge(a, b, etype="same_dst")
                store["dst_map"][key].extend(new_ids)

            store["build_time"] += time.perf_counter() - t_win_start  

        try_flush_completed_windows()

        if len(win_store) > MAX_ACTIVE_WINDOWS:
            try_flush_completed_windows()

        print(f"[Chunk {cidx}] processed={processed}, "
              f"active_windows={len(win_store)}, max_seen_wid={max_seen_wid}")

    for wid in sorted(list(win_store.keys())):
        flush_window(win_store, outdir, wid, min_ts, manifest_rows)

    pd.DataFrame(manifest_rows).sort_values("win_id").to_csv(
        Path(OUTDIR) / "manifest.csv", index=False
    )
    print("[Done] All windows saved. Manifest written to manifest.csv")


if __name__ == "__main__":
    main()
