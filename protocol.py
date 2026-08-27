# protocol.py
# -*- coding: utf-8 -*-
"""Guards to enforce: train-once; iterate/update strictly sequential."""

import os
import numpy as np
import pandas as pd

from config import (
    LOGGER, DATA_SAVE_FILE, OPTIMIZER_FILE, ENCODER_FILE, ITERATION_FILE,
    TARGET_VARIABLE, BATCH_SIZE
)
from utils import get_iter

class ProtocolViolation(RuntimeError):
    pass

def _training_done() -> bool:
    return (
        os.path.exists(OPTIMIZER_FILE)
        and os.path.exists(ENCODER_FILE)
        and os.path.exists(DATA_SAVE_FILE)
    )

def _read_df_safe() -> pd.DataFrame:
    if not os.path.exists(DATA_SAVE_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(DATA_SAVE_FILE)
    except Exception:
        return pd.DataFrame()

def _iter_series(df: pd.DataFrame) -> pd.Series:
    if "Iteration" in df.columns:
        return df["Iteration"].fillna(0).astype(int)
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index)

def assert_can_train():
    if _training_done():
        LOGGER.error(
            "TRAIN blocked: already completed once. "
            "To reset, delete: %s, %s, %s, %s.",
            OPTIMIZER_FILE, ENCODER_FILE, DATA_SAVE_FILE, ITERATION_FILE
        )
        raise ProtocolViolation("Training already completed.")

def assert_can_iterate():
    if not _training_done():
        LOGGER.error("ITERATE blocked: training not completed. Run --mode train first.")
        raise ProtocolViolation("Training not done.")

    df = _read_df_safe()
    if df.empty:
        LOGGER.error("ITERATE blocked: data file missing/empty. Run --mode train first.")
        raise ProtocolViolation("No data available.")

    cur = get_iter()
    it = _iter_series(df)

    pend_prior = df[(it < cur) & (df[TARGET_VARIABLE].isna())]
    if not pend_prior.empty:
        max_prior = int(it[pend_prior.index].max())
        LOGGER.error(
            "ITERATE blocked: %d unmeasured rows exist for earlier iteration(s) ≤ %d. "
            "Fill '%s' and run --mode update first.",
            len(pend_prior), max_prior, TARGET_VARIABLE
        )
        raise ProtocolViolation("Pending from prior iterations.")

    cur_rows = df[it == cur]
    if not cur_rows.empty:
        if cur_rows[TARGET_VARIABLE].isna().any():
            LOGGER.error(
                "ITERATE blocked: Iteration %d already open with %d pending candidates. "
                "Measure and run --mode update first.",
                cur, int(cur_rows[TARGET_VARIABLE].isna().sum())
            )
        else:
            LOGGER.error(
                "ITERATE blocked: Iteration %d has measurements but is not closed. "
                "Run --mode update to advance.",
                cur
            )
        raise ProtocolViolation("Current iteration open/needs update.")

def assert_can_update():
    if not _training_done():
        LOGGER.error("UPDATE blocked: training not completed. Run --mode train first.")
        raise ProtocolViolation("Training not done.")

    df = _read_df_safe()
    if df.empty:
        LOGGER.error("UPDATE blocked: data file missing/empty. Run --mode iterate first.")
        raise ProtocolViolation("No data available.")

    cur = get_iter()
    it = _iter_series(df)

    pend_prior = df[(it < cur) & (df[TARGET_VARIABLE].isna())]
    if not pend_prior.empty:
        LOGGER.error(
            "UPDATE blocked: %d unmeasured rows exist for earlier iteration(s) ≤ %d.",
            len(pend_prior), int(it[pend_prior.index].max())
        )
        raise ProtocolViolation("Pending from prior iterations.")

    rows_cur = df[it == cur]
    if rows_cur.empty:
        LOGGER.error("UPDATE blocked: no candidate rows for Iteration %d. Run --mode iterate first.", cur)
        raise ProtocolViolation("No rows for current iteration.")

    miss = rows_cur[rows_cur[TARGET_VARIABLE].isna()]
    if not miss.empty:
        LOGGER.error(
            "UPDATE blocked: Iteration %d has %d/%d candidates without measured '%s'.",
            cur, len(miss), len(rows_cur), TARGET_VARIABLE
        )
        try:
            LOGGER.info("Pending candidate row indices: %s", list(map(int, miss.index)))
        except Exception:
            pass
        raise ProtocolViolation("Current iteration not fully measured.")

    if len(rows_cur) != BATCH_SIZE:
        LOGGER.warning(
            "Iteration %d has %d rows (BATCH_SIZE=%d). Proceeding; verify this is intended.",
            cur, len(rows_cur), BATCH_SIZE
        )
