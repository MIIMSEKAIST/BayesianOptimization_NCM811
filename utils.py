# utils.py
# -*- coding: utf-8 -*-
"""General IO, CSV append, encoding helpers, iteration helpers, and plotting dirs."""

import os, pickle
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

from config import (
    LOGGER, TRAIN_DATA_FILE, DATA_SAVE_FILE, OPTIMIZER_FILE, ENCODER_FILE, ITERATION_FILE,
    PLOTS_DIR, TARGET_VARIABLE, NUMERICAL_BOUNDS, CAT_COLS, ALL_OPT_COLS
)


# ---------- Pickle & CSV helpers ----------
def save_pickle(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    LOGGER.info("Saved → %s", path)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def csv_header(path: str) -> List[str]:
    return list(pd.read_csv(path, nrows=0).columns)

def append_csv_aligned(df_new: pd.DataFrame, path: str) -> None:
    if not os.path.exists(path):
        df_new.to_csv(path, index=False, encoding="utf-8-sig")
        LOGGER.info("Created %s", path)
        return
    header = csv_header(path)
    df_new = df_new.reindex(columns=header, fill_value=np.nan)
    df_new.to_csv(path, mode="a", header=False, index=False, encoding="utf-8-sig")
    LOGGER.debug("Appended %d rows", len(df_new))

# ---------- Iteration counter ----------
def get_iter(path: str = ITERATION_FILE) -> int:
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write("1")
        return 1
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except (ValueError, OSError):
        with open(path, "w") as fh:
            fh.write("1")
        return 1

def next_iter(path: str = ITERATION_FILE) -> int:
    cur = get_iter(path) + 1
    with open(path, "w") as fh:
        fh.write(str(cur))
    return cur

# ---------- Encoding helpers ----------
def load_encode(csv_path: str, encoder: OrdinalEncoder | None = None):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "doi" in df.columns:
        df = df.drop(columns=["doi"])
    if encoder is None:
        encoder = OrdinalEncoder()
        df_enc = df.copy()
        df_enc[CAT_COLS] = encoder.fit_transform(df[CAT_COLS])
    else:
        df_enc = df.copy()
        df_enc[CAT_COLS] = encoder.transform(df[CAT_COLS])

    mask = np.ones(len(df), dtype=bool)
    for col, (lo, hi) in NUMERICAL_BOUNDS.items():
        mask &= df[col].between(lo, hi)
    return df[mask].reset_index(drop=True), df_enc[mask].reset_index(drop=True), encoder

def decode_cats(df_enc: pd.DataFrame, enc: OrdinalEncoder) -> pd.DataFrame:
    df = df_enc.copy()
    df[CAT_COLS] = enc.inverse_transform(df_enc[CAT_COLS])
    return df

# ---------- Plotting/output dirs ----------
def ensure_plots_dir(out_dir: str | None = None):
    d = out_dir or PLOTS_DIR
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ---------- Snapshot helpers ----------
from datetime import datetime

def df_upto_iter(df: pd.DataFrame, upto_it: int) -> pd.DataFrame:
    if "Iteration" not in df.columns:
        return df.copy()
    return df[df["Iteration"].fillna(0).astype(int) <= int(upto_it)].copy()

def iter_dir(it: int) -> str:
    d = os.path.join(PLOTS_DIR, f"iter_{int(it):03d}")
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    return d

def write_manifest_snapshot(df_snapshot: pd.DataFrame, upto_it: int, out_dir: str):
    comp = df_snapshot.dropna(subset=[TARGET_VARIABLE])
    best = None
    if not comp.empty:
        best_row = comp.loc[comp[TARGET_VARIABLE].idxmax()]
        best = float(best_row[TARGET_VARIABLE])
    with open(os.path.join(out_dir, "MANIFEST.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"Snapshot: data ≤ Iteration {int(upto_it)}\n")
        fh.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"Rows in snapshot: {df_snapshot.shape[0]}\n")
        fh.write(f"Completed experiments: {comp.shape[0]}\n")
        if best is not None:
            fh.write(f"Best measured Discharge_Capacity: {best:.2f} mAh/g\n")
