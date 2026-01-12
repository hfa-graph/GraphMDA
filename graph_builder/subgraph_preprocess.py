#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import sys, pickle, shutil
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import category_encoders as ce
except Exception:
    raise SystemExit("Please install category_encoders: pip install category_encoders")
try:
    import networkx as nx
except Exception:
    raise SystemExit("Please install networkx: pip install networkx")

# =================== CONFIG ===================
ROOT = Path(r"XXX/XXXXXX")
SPLITS  = [f"k{i}" for i in range(1, 6+1)]
DOMAINS = ["source", "target"]
SETS    = ["train", "test"]
FILE_EXTS = (".gpickle", ".pkl", ".pickle")

CAT_COLS = [
    "PROTOCOL","L7_PROTO","TCP_FLAGS","CLIENT_TCP_FLAGS","SERVER_TCP_FLAGS",
    "ICMP_TYPE","ICMP_IPV4_TYPE","DNS_QUERY_TYPE","DNS_TTL_ANSWER","FTP_COMMAND_RET_CODE",
]
LABEL_COL = "Label"
DOMAIN_LABEL_COL = "domain_label"
OUT_SUFFIX = "_preprocessed_ce"
DROP_RAW_ATTRS = True  

DELETE_ORIGINAL_AFTER_SPLIT = True
VERBOSE = True

RANDOM_SEED = 20250924
np.random.seed(RANDOM_SEED)

CLIP_ABS = 1e9       
WARN_NONFINITE_TOPK = 20  

def vprint(*args):
    if VERBOSE:
        print(*args)

def discover_graph_files_under(dir_: Path) -> List[Path]:
    if not dir_.exists():
        return []
    files: List[Path] = []
    for ext in FILE_EXTS:
        files.extend(p for p in dir_.rglob(f"*{ext}") if p.is_file())
    return sorted(files)

def load_graph(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)

def save_graph(g, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)

def path_to_out(path: Path, split_root: Path) -> Path:
    rel = path.relative_to(split_root)
    new_root = Path(str(split_root) + OUT_SUFFIX)  
    return new_root / rel

def extract_nodes_df(g) -> pd.DataFrame:
    rows = []
    for nid, attrs in g.nodes(data=True):
        row = dict(attrs)
        row["__nid__"] = nid
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def collect_train_nodes_for_split(split_root: Path) -> pd.DataFrame:
    frames = []
    for domain in DOMAINS:
        train_dir = split_root / domain / "train"
        for f in discover_graph_files_under(train_dir):
            g = load_graph(f)
            df = extract_nodes_df(g)
            if df.empty:
                continue

            if LABEL_COL in df.columns:
                df[LABEL_COL] = df[LABEL_COL].astype(int)
            else:
                raise RuntimeError(f"{f} has no {LABEL_COL}")

            if DOMAIN_LABEL_COL in df.columns:
                df[DOMAIN_LABEL_COL] = df[DOMAIN_LABEL_COL].astype(int)
            else:
                df[DOMAIN_LABEL_COL] = pd.Series([np.nan] * len(df), dtype="Int64")

            if "Attack_encoded" in df.columns:
                df["Attack_encoded"] = df["Attack_encoded"].astype(int)
            else:
                raise RuntimeError(f"{f} has no Attack_encoded")

            for c in CAT_COLS:
                if c not in df.columns:
                    df[c] = np.nan

            df["__graph_path__"] = str(f)
            frames.append(df)

    if not frames:
        raise RuntimeError(f"[{split_root.name}] No training nodes found to fit encoders/scaler.")
    return pd.concat(frames, ignore_index=True)

def select_numeric_columns(df: pd.DataFrame) -> List[str]:
    helper = {LABEL_COL, DOMAIN_LABEL_COL, "__graph_path__", "__nid__", "feat", "Attack_encoded"}
    num_cols: List[str] = []
    for col, dtype in df.dtypes.items():
        if col in helper or col in CAT_COLS:
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            num_cols.append(col)
    return num_cols

def _diagnose_nonfinite(df: pd.DataFrame, tag: str):
    if not VERBOSE or df.empty:
        return
    bad_info: List[Tuple[str, int]] = []
    for c in df.columns:
        vals = df[c].values
        nonfinite = np.count_nonzero(~np.isfinite(vals))
        too_big = np.count_nonzero(np.abs(vals[np.isfinite(vals)]) > CLIP_ABS)
        bad = nonfinite + too_big
        if bad > 0:
            bad_info.append((c, bad))
    if bad_info:
        bad_info.sort(key=lambda x: -x[1])
        vprint(f"[DIAG-{tag}] Columns with non-finite or >|{CLIP_ABS}| values (top {WARN_NONFINITE_TOPK}):")
        for c, k in bad_info[:WARN_NONFINITE_TOPK]:
            vprint(f"  - {c}: {k} problematic entries")

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _clean_numeric_for_fit(X: pd.DataFrame, medians: Dict[str, float]) -> pd.DataFrame:
    if X.empty:
        return X
    X = _coerce_numeric(X, X.columns.tolist())
    _diagnose_nonfinite(X, "FIT-BEFORE")
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        X[c] = X[c].fillna(medians.get(c, 0.0))
    X = X.clip(lower=-CLIP_ABS, upper=CLIP_ABS)
    X = pd.DataFrame(np.nan_to_num(X.values, nan=0.0, posinf=CLIP_ABS, neginf=-CLIP_ABS),
                     columns=X.columns, index=X.index)
    _diagnose_nonfinite(X, "FIT-AFTER")
    return X

def _clean_numeric_for_transform(X: pd.DataFrame, medians: Dict[str, float]) -> pd.DataFrame:
    if X.empty:
        return X
    X = _coerce_numeric(X, X.columns.tolist())
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in X.columns:
        X[c] = X[c].fillna(medians.get(c, 0.0))
    X = X.clip(lower=-CLIP_ABS, upper=CLIP_ABS)
    X = pd.DataFrame(np.nan_to_num(X.values, nan=0.0, posinf=CLIP_ABS, neginf=-CLIP_ABS),
                     columns=X.columns, index=X.index)
    return X

# Fit encoders/scaler using TRAIN nodes of this split only (prevents leakage).
def fit_target_encoder_and_scaler(train_df: pd.DataFrame):
    te = ce.TargetEncoder(
        cols=CAT_COLS,
        min_samples_leaf=20,
        smoothing=10.0,
        handle_missing="value",
        handle_unknown="value",
    )
    te.fit(train_df[CAT_COLS], train_df[LABEL_COL])

    num_cols = sorted(select_numeric_columns(train_df))
    medians = {c: float(train_df[c].median()) if c in train_df.columns else 0.0 for c in num_cols}

    if num_cols:
        X = train_df[num_cols].copy()
        X = _clean_numeric_for_fit(X, medians)
        scaler = StandardScaler().fit(X.values.astype(np.float64))
    else:
        scaler = StandardScaler()  # placeholder
    return te, num_cols, medians, scaler

def build_feat_for_graph(g, te, num_cols, medians, scaler):
    df = extract_nodes_df(g)
    if df.empty:
        return g

    # Label
    if LABEL_COL in df.columns:
        df[LABEL_COL] = df[LABEL_COL].astype(int)
    else:
        raise RuntimeError("Graph has no Label")

    # domain_label
    if DOMAIN_LABEL_COL in df.columns:
        df[DOMAIN_LABEL_COL] = df[DOMAIN_LABEL_COL].astype(int)
    else:
        df[DOMAIN_LABEL_COL] = pd.Series([np.nan] * len(df), dtype="Int64")

    if "Attack_encoded" in df.columns:
        df["Attack_encoded"] = df["Attack_encoded"].astype(int)
    else:
        raise RuntimeError("Graph has no Attack_encoded")

    for c in CAT_COLS:
        if c not in df.columns:
            df[c] = np.nan


    te_vals = te.transform(df[CAT_COLS])

    if num_cols:
        X_num = df[num_cols].copy()
        X_num = _clean_numeric_for_transform(X_num, medians)
        try:
            Xs = scaler.transform(X_num.values.astype(np.float64))
        except Exception as e:
            vprint(f"[WARN] scaler.transform failed ({e}); fallback to zero-mean no-scale.")
            mu = getattr(scaler, "mean_", np.zeros(X_num.shape[1], dtype=np.float64))
            Xs = (X_num.values.astype(np.float64) - mu)
        Xs_df = pd.DataFrame(Xs, columns=num_cols, index=df.index)
    else:
        Xs_df = pd.DataFrame(index=df.index)

    feat_df = pd.concat([te_vals.reset_index(drop=True), Xs_df.reset_index(drop=True)], axis=1)
    feat_mat = feat_df.values.astype(np.float32) 

    idx_by_nid = {nid: i for i, nid in enumerate(df["__nid__"].tolist())}
    for nid, attrs in g.nodes(data=True):
        i = idx_by_nid[nid]
        attrs["feat"] = feat_mat[i]
        attrs[LABEL_COL] = int(df.iloc[i][LABEL_COL])
        dlv = df.iloc[i][DOMAIN_LABEL_COL]
        attrs[DOMAIN_LABEL_COL] = (int(dlv) if pd.notna(dlv) else None)
        attrs["Attack_encoded"] = int(df.iloc[i]["Attack_encoded"])

        if DROP_RAW_ATTRS:
            keep = {"feat", LABEL_COL, DOMAIN_LABEL_COL, "Attack_encoded"}
            for k in list(attrs.keys()):
                if k not in keep:
                    del attrs[k]
    return g

def process_one_split(split_name: str, root: Path):
    split_root = root / split_name
    if not split_root.exists():
        print(f"[{split_name}] skipped (not found).")
        return

    out_root = Path(str(split_root) + OUT_SUFFIX)

    print(f"\n=== Processing {split_name} ===")
    print("[1/4] Gathering TRAIN nodes (this split only) ...")
    train_df = collect_train_nodes_for_split(split_root)
    print(f"  - Train nodes: {len(train_df)} across {train_df['__graph_path__'].nunique()} graphs")

    print("[2/4] Fitting TargetEncoder & StandardScaler ...")
    te, num_cols, medians, scaler = fit_target_encoder_and_scaler(train_df)
    print(f"  - TE cols: {len(CAT_COLS)} | Numeric cols: {len(num_cols)}")
    print(f"  - Final feat dim for {split_name}: {len(CAT_COLS) + len(num_cols)}")

    print("[3/4] Transforming & saving graphs for this split ...")
    ok = 0
    for domain in DOMAINS:
        for subset in SETS:
            base = split_root / domain / subset
            for f in discover_graph_files_under(base):
                try:
                    g = load_graph(f)
                    g = build_feat_for_graph(g, te, num_cols, medians, scaler)
                    out_path = path_to_out(f, split_root)
                    save_graph(g, out_path)
                    ok += 1
                    print(f"  ✓ {f.relative_to(split_root)} → {out_path.relative_to(out_root)} | nodes: {g.number_of_nodes()}")
                except Exception as e:
                    print(f"  ! Failed {f}: {e}")
    print(f"[4/4] {split_name} done. Saved {ok} graphs under {out_root}")

    # Optional: delete original split to save disk space after successful export.
    if DELETE_ORIGINAL_AFTER_SPLIT:
        out_files = discover_graph_files_under(out_root)
        if ok > 0 and out_root.exists() and len(out_files) > 0:
            try:
                if out_root.resolve() == split_root.resolve():
                    raise RuntimeError("Out root equals split root (unexpected). Abort deletion.")
                print(f"[*] Removing original split directory: {split_root}")
                shutil.rmtree(split_root)
                print(f"[✓] Removed {split_root.name}")
            except Exception as e:
                print(f"[!] Failed to remove {split_root}: {e}")
        else:
            print(f"[!] Skip deletion for {split_name}: output not found or empty.")

def process_all(root: Path):
    for split in SPLITS:
        process_one_split(split, root)

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        ROOT = Path(sys.argv[1])
    if not ROOT.exists():
        raise SystemExit(f"Root directory not found: {ROOT}")
    process_all(ROOT)
