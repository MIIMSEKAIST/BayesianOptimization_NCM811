# modes.py
# -*- coding: utf-8 -*-
"""Implements CLI modes: train, iterate, update, best, visualize, loop."""

import os, warnings, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor

from config import (
    LOGGER, TRAIN_DATA_FILE, DATA_SAVE_FILE, OPTIMIZER_FILE, ENCODER_FILE,
    ITERATION_FILE,
    TARGET_VARIABLE, PLOTS_DIR, SEED, ALL_OPT_COLS)




from config import log_t  # keep exact transform
from utils import (
    save_pickle, load_pickle, append_csv_aligned,
    load_encode, decode_cats, get_iter, next_iter,
    df_upto_iter, iter_dir, write_manifest_snapshot
)
from protocol import ProtocolViolation, assert_can_train, assert_can_iterate, assert_can_update
from optimizer_utils import build_space, init_optimizer, suggest_unique, optimizer_best
from viz import (
    plot_convergence, plot_learning_curve, plot_acquisition_efficiency, plot_feature_importance,
    plot_categorical_heatmap, plot_all_precursor_heatmaps, plot_reliability_suite
)

def mode_train():
    try:
        assert_can_train()
    except ProtocolViolation:
        return

    df_raw, df_enc, enc = load_encode(TRAIN_DATA_FILE)
    spc = build_space(enc)
    y0 = -log_t(df_raw[TARGET_VARIABLE].values)
    missing = [c for c in ALL_OPT_COLS if c not in df_enc.columns]
    if missing:
        LOGGER.error("Missing expected columns in encoded dataframe: %s", missing)
        return
    X0 = df_enc[ALL_OPT_COLS].values
    opt = init_optimizer(spc, X0, y0)
    save_pickle(opt, OPTIMIZER_FILE)
    save_pickle(enc, ENCODER_FILE)

    with open(ITERATION_FILE, "w") as fh:
        fh.write("1")

    df_raw.assign(Iteration=0).pipe(append_csv_aligned, DATA_SAVE_FILE)
    LOGGER.info("TRAIN complete – optimiser bootstrapped with %d entries", len(df_raw))

def mode_iterate():
    try:
        assert_can_iterate()
    except ProtocolViolation:
        return

    opt, enc = load_pickle(OPTIMIZER_FILE), load_pickle(ENCODER_FILE)
    df_exist, enc_exist, _ = load_encode(DATA_SAVE_FILE, encoder=enc)

    new_enc = suggest_unique(opt, enc_exist, batch=3)  # BATCH_SIZE=3 preserved
    if new_enc is None:
        return

    const_cols = [c for c in df_exist.columns if c not in list(new_enc.columns) + [TARGET_VARIABLE, "Iteration"]]
    const_vals = df_exist.iloc[0][const_cols] if not df_exist.empty else pd.Series(dtype=object)

    new_dec = decode_cats(new_enc, enc)
    for col in const_cols:
        new_dec[col] = const_vals.get(col, np.nan)

    itr = get_iter()
    new_dec["Iteration"] = itr
    new_dec[TARGET_VARIABLE] = np.nan
    append_csv_aligned(new_dec, DATA_SAVE_FILE)
    LOGGER.info("ITERATION %d – appended %d suggested experiments", itr, len(new_dec))

def mode_update():
    try:
        assert_can_update()
    except ProtocolViolation:
        return

    opt, enc = load_pickle(OPTIMIZER_FILE), load_pickle(ENCODER_FILE)
    df = pd.read_csv(DATA_SAVE_FILE)

    done = df.dropna(subset=[TARGET_VARIABLE])
    if done.empty:
        LOGGER.warning("No completed experiments – update skipped")
        return

    done_enc = done.copy()
    done_enc[["Li_Precursor","Ni_Precursor","Co_Precursor","Mn_Precursor"]] = \
        enc.transform(done[["Li_Precursor","Ni_Precursor","Co_Precursor","Mn_Precursor"]])

    y_new = -log_t(done_enc[TARGET_VARIABLE].values)
    opt.tell(done_enc[["Li_Precursor","Ni_Precursor","Co_Precursor","Mn_Precursor",
                       "Hydrothermal_Temperature","Hydrothermal_Time",
                       "First_Calcination_Temperature","First_Calcination_Time",
                       "Second_Calcination_Temperature","Second_Calcination_Time"]].values.tolist(),
             y_new.tolist())
    save_pickle(opt, OPTIMIZER_FILE)

    best = optimizer_best(opt, enc)
    LOGGER.info("Optimizer ingested %d points; best so far:", len(done_enc))
    for k, v in best.items():
        LOGGER.info("  %s: %s", k, v)

    next_iter()

def mode_best(top_n: int = 3):
    if not os.path.exists(DATA_SAVE_FILE) or not os.path.exists(ENCODER_FILE):
        LOGGER.error("Data or encoder missing – run iterations first")
        return

    enc = load_pickle(ENCODER_FILE)
    df = pd.read_csv(DATA_SAVE_FILE)

    comp = df.dropna(subset=[TARGET_VARIABLE])
    if comp.shape[0] == 0:
        LOGGER.error("No completed experiments available – cannot train surrogate to predict.")
        return
    if comp.shape[0] < 10:
        LOGGER.warning("Fewer than 10 completed experiments – predictions may be unstable.")

    comp_enc = comp.copy(); comp_enc[["Li_Precursor","Ni_Precursor","Co_Precursor","Mn_Precursor"]] = \
        enc.transform(comp[["Li_Precursor","Ni_Precursor","Co_Precursor","Mn_Precursor"]])
    X, y = comp_enc[["Li_Precursor","Ni_Precursor","Co_Precursor","Mn_Precursor",
                     "Hydrothermal_Temperature","Hydrothermal_Time",
                     "First_Calcination_Temperature","First_Calcination_Time",
                     "Second_Calcination_Temperature","Second_Calcination_Time"]].values, \
           comp_enc[TARGET_VARIABLE].values

    rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1).fit(X, y)

    pending = df[df[TARGET_VARIABLE].isna()].copy()
    if pending.empty:
        LOGGER.warning("No pending candidates (rows with NaN '%s') to score.", TARGET_VARIABLE)
        return

    if "Iteration" not in pending.columns or pending["Iteration"].isna().all():
        LOGGER.warning("No valid 'Iteration' info on pending rows – scoring all pending.")
        recent = pending.copy()
        latest_iter = "N/A"
    else:
        latest_iter = int(pending["Iteration"].max())
        recent = pending[pending["Iteration"] == latest_iter].copy()
        if recent.empty:
            LOGGER.warning("No pending candidates in the latest iteration – scoring all pending.")
            recent = pending.copy(); latest_iter = "N/A"

    recent_enc = recent.copy()
    recent_enc[["Li_Precursor","Ni_Precursor","Co_Precursor","Mn_Precursor"]] = \
        enc.transform(recent[["Li_Precursor","Ni_Precursor","Co_Precursor","Mn_Precursor"]])
    recent["_pred"] = rf.predict(recent_enc[["Li_Precursor","Ni_Precursor","Co_Precursor","Mn_Precursor",
                                             "Hydrothermal_Temperature","Hydrothermal_Time",
                                             "First_Calcination_Temperature","First_Calcination_Time",
                                             "Second_Calcination_Temperature","Second_Calcination_Time"]].values)

    top = recent.sort_values("_pred", ascending=False).head(top_n).reset_index(drop=False)
    hdr = f"Iteration {latest_iter}" if latest_iter != "N/A" else "latest pending candidates"
    LOGGER.info("=== PREDICTED DISCHARGE CAPACITY for %s ===", hdr)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        params = {k: row[k] for k in ["Li_Precursor","Ni_Precursor","Co_Precursor","Mn_Precursor",
                                      "Hydrothermal_Temperature","Hydrothermal_Time",
                                      "First_Calcination_Temperature","First_Calcination_Time",
                                      "Second_Calcination_Temperature","Second_Calcination_Time"]}
        itr = row.get("Iteration", "N/A")
        LOGGER.info("Rank %d | Pred %.2f mAh/g | Iter %s | Params %s",
                    rank, row["_pred"], itr, params)

def mode_visualize():
    if not os.path.exists(DATA_SAVE_FILE) or not os.path.exists(ENCODER_FILE):
        LOGGER.error("Need data & encoder – run some iterations first"); return

    enc = load_pickle(ENCODER_FILE)
    df = pd.read_csv(DATA_SAVE_FILE)
    if df.empty:
        LOGGER.warning("Data file empty"); return

    if "Iteration" in df.columns:
        iters = sorted(set(int(x) for x in df["Iteration"].dropna().astype(int).tolist()))
    else:
        iters = [0]

    for it in iters:
        df_snap = df_upto_iter(df, it)
        out_dir = iter_dir(it)
        it_tag = f"Data ≤ Iteration {it:03d}"

        plot_convergence(df_snap, out_dir=out_dir, it_label=it_tag)
        plot_learning_curve(df_snap, enc, out_dir=out_dir, it_label=it_tag)
        plot_acquisition_efficiency(df_snap, out_dir=out_dir, it_label=it_tag)
        plot_feature_importance(df_snap, enc, out_dir=out_dir, it_label=it_tag)
        plot_categorical_heatmap(df_snap, out_dir=out_dir, it_label=it_tag)
        plot_all_precursor_heatmaps(df_snap, out_dir=out_dir, it_label=it_tag)
        plot_reliability_suite(df_snap, enc, out_dir=out_dir, it_label=it_tag)

        write_manifest_snapshot(df_snap, it, out_dir)

    LOGGER.info("Saved per-iteration snapshots under '%s/iter_XXX'", PLOTS_DIR)

def mode_loop(n: int):
    for i in range(n):
        LOGGER.info("==== LOOP %d/%d : ITERATE ====", i + 1, n)
        mode_iterate(); input("→ run experiments, fill results, then press ENTER …")
        LOGGER.info("==== UPDATE ===="); mode_update()
