# viz.py
# -*- coding: utf-8 -*-
"""All visualizations, per-iteration snapshots, reliability diagnostics, heatmaps."""

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

from config import (
    LOGGER, PLOTS_DIR, TARGET_VARIABLE, CAT_COLS, ALL_OPT_COLS, SEED
)
from utils import ensure_plots_dir

# ---------- Core plots ----------
def plot_convergence(df: pd.DataFrame, out_dir: str | None = None, it_label: str | None = None):
    df_sorted = df.sort_values("Iteration")
    best_by_iter = df_sorted.groupby("Iteration")[TARGET_VARIABLE].max().cummax()

    ensure_plots_dir(out_dir)
    plt.figure(figsize=(6, 4))
    best_by_iter.plot(marker="o")
    plt.xlabel("Iteration"); plt.ylabel("Best capacity (mAh g⁻¹)")
    title = "Convergence of best Discharge Capacity"
    if it_label: title += f" — {it_label}"
    plt.title(title); plt.grid(True); plt.tight_layout()
    fname = os.path.join(out_dir or PLOTS_DIR, "convergence.png")
    plt.savefig(fname); LOGGER.info("Saved %s", fname); plt.close()

def plot_learning_curve(df: pd.DataFrame, enc, out_dir: str | None = None, it_label: str | None = None):
    ensure_plots_dir(out_dir)
    r2_scores = []
    iters = sorted(df["Iteration"].dropna().unique()) if "Iteration" in df.columns else [0]
    for it in iters:
        subset = df[df["Iteration"] <= it].dropna(subset=[TARGET_VARIABLE])
        if subset.shape[0] < 10:
            continue
        subset_enc = subset.copy(); subset_enc[CAT_COLS] = enc.transform(subset[CAT_COLS])
        X, y = subset_enc[ALL_OPT_COLS].values, subset_enc[TARGET_VARIABLE].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
        rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=SEED).fit(X_train, y_train)
        r2_scores.append((int(it), r2_score(y_test, rf.predict(X_test))))
    if not r2_scores:
        LOGGER.warning("Not enough data for learning curve"); return
    xs, ys = zip(*r2_scores)
    plt.figure(figsize=(6, 4)); plt.plot(xs, ys, marker="o")
    plt.xlabel("Iteration"); plt.ylabel("Hold-out R²")
    title = "Surrogate learning curve"
    if it_label: title += f" — {it_label}"
    plt.title(title); plt.grid(True); plt.tight_layout()
    fname = os.path.join(out_dir or PLOTS_DIR, "learning_curve.png")
    plt.savefig(fname); LOGGER.info("Saved %s", fname); plt.close()

def plot_acquisition_efficiency(df: pd.DataFrame, out_dir: str | None = None, it_label: str | None = None):
    ensure_plots_dir(out_dir)
    df_box = df.copy(); df_box["Set"] = np.where(df_box["Iteration"] == 0, "Train", "Suggested")
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Set", y=TARGET_VARIABLE, data=df_box)
    plt.ylabel("Discharge Capacity (mAh g⁻¹)")
    title = "Capacity distribution: training vs suggestions"
    if it_label: title += f" — {it_label}"
    plt.title(title); plt.tight_layout()
    fname = os.path.join(out_dir or PLOTS_DIR, "acquisition_efficiency.png")
    plt.savefig(fname); LOGGER.info("Saved %s", fname); plt.close()

def plot_feature_importance(df: pd.DataFrame, enc, out_dir: str | None = None, it_label: str | None = None):
    ensure_plots_dir(out_dir)
    comp = df.dropna(subset=[TARGET_VARIABLE])
    if comp.shape[0] < 10:
        LOGGER.warning("Not enough completed experiments to compute feature importance"); return
    comp_enc = comp.copy(); comp_enc[CAT_COLS] = enc.transform(comp[CAT_COLS])
    X, y = comp_enc[ALL_OPT_COLS].values, comp_enc[TARGET_VARIABLE].values
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1).fit(X, y)
    importances = rf.feature_importances_
    feat_df = pd.DataFrame({"Feature": ALL_OPT_COLS, "Importance": importances}).sort_values("Importance", ascending=False)
    plt.figure(figsize=(7, 4))
    sns.barplot(x="Importance", y="Feature", data=feat_df)
    title = "RF surrogate feature importance"
    if it_label: title += f" — {it_label}"
    plt.title(title); plt.tight_layout()
    fname = os.path.join(out_dir or PLOTS_DIR, "feature_importance.png")
    plt.savefig(fname); LOGGER.info("Saved %s", fname); plt.close()

def plot_categorical_heatmap(df: pd.DataFrame, out_dir: str | None = None, it_label: str | None = None):
    ensure_plots_dir(out_dir)
    comp = df.dropna(subset=[TARGET_VARIABLE])
    if comp.empty: return
    pivot = comp.pivot_table(index="Li_Precursor", columns="Ni_Precursor",
                             values=TARGET_VARIABLE, aggfunc="mean")
    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
    title = "Mean capacity by Li vs Ni precursor"
    if it_label: title += f" — {it_label}"
    plt.title(title); plt.tight_layout()
    fname = os.path.join(out_dir or PLOTS_DIR, "cat_heatmap_Li_vs_Ni.png")
    plt.savefig(fname); LOGGER.info("Saved %s", fname); plt.close()

def plot_all_precursor_heatmaps(df: pd.DataFrame, aggfunc: str = "mean",
                                out_dir: str | None = None, it_label: str | None = None):
    comp = df.dropna(subset=[TARGET_VARIABLE]).copy()
    if comp.empty:
        LOGGER.warning("No completed experiments to plot precursor heatmaps.")
        return
    ensure_plots_dir(out_dir)
    pairs = list(combinations(CAT_COLS, 2))
    label_suffix = f" — {it_label}" if it_label else ""
    pdf_name = "precursor_heatmaps.pdf"
    if it_label:
        pdf_name = f"precursor_heatmaps_{it_label.replace(' ', '_').replace('≤', 'le')}.pdf"
    pdf_path = os.path.join(out_dir or PLOTS_DIR, pdf_name)

    with PdfPages(pdf_path) as pdf:
        for a, b in pairs:
            pivot = comp.pivot_table(index=a, columns=b,
                                     values=TARGET_VARIABLE, aggfunc=aggfunc)
            counts = comp.pivot_table(index=a, columns=b,
                                      values=TARGET_VARIABLE, aggfunc="count")
            if pivot.empty:
                LOGGER.warning("Skipping empty heatmap for %s × %s", a, b)
                continue
            n_rows = max(1, len(pivot.index))
            n_cols = max(1, len(pivot.columns))
            fig_w = max(6, 1.2 * n_cols)
            fig_h = max(5, 0.8 * n_rows)

            plt.figure(figsize=(fig_w, fig_h))
            ax = sns.heatmap(
                pivot, annot=True, fmt=".1f", cmap="viridis",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": f"{TARGET_VARIABLE} (mAh g⁻¹)"}
            )
            ax.set_title(f"{aggfunc.title()} {TARGET_VARIABLE} by {a} × {b}{label_suffix}")
            ax.set_xlabel(b); ax.set_ylabel(a)

            for i in range(n_rows):
                for j in range(n_cols):
                    try:
                        n = int(counts.iloc[i, j])
                    except Exception:
                        n = 0
                    ax.text(j + 0.5, i + 0.85, f"n={n}",
                            ha="center", va="center", fontsize=7, color="black")

            plt.tight_layout()
            fname = os.path.join(out_dir or PLOTS_DIR, f"heatmap_{a}_vs_{b}.png")
            plt.savefig(fname, dpi=220); pdf.savefig(); plt.close()
            LOGGER.info("Saved %s", fname)

    LOGGER.info("Saved multi-page heatmap PDF → %s", pdf_path)

# ---------- Reliability & performance ----------
def plot_reliability_suite(df: pd.DataFrame, enc, n_splits: int = 5,
                           out_dir: str | None = None, it_label: str | None = None):
    ensure_plots_dir(out_dir)
    comp = df.dropna(subset=[TARGET_VARIABLE]).copy()
    if comp.shape[0] < 3:
        LOGGER.warning("Too few completed experiments (%d) for reliability plots.", comp.shape[0])
        return

    comp_enc = comp.copy()
    comp_enc[CAT_COLS] = enc.transform(comp[CAT_COLS])
    X = comp_enc[ALL_OPT_COLS].values
    y = comp_enc[TARGET_VARIABLE].values

    # Choose effective CV folds
    if len(y) >= 10:
        n_possible = max(2, min(5, len(y) // 5))
    else:
        n_possible = 2
    n_splits_eff = min(n_splits, n_possible)
    if len(y) < 30:
        LOGGER.info("Using %d-fold CV on %d samples; metrics may be a bit noisy.", n_splits_eff, len(y))

    # K-fold CV with tree-std uncertainty proxy
    kf = KFold(n_splits=n_splits_eff, shuffle=True, random_state=SEED)
    y_true_all, y_pred_all, y_std_all = [], [], []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        rf = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1, bootstrap=True)
        rf.fit(X_tr, y_tr)
        preds_trees = np.vstack([t.predict(X_te) for t in rf.estimators_])
        y_pred = preds_trees.mean(axis=0)
        y_std  = preds_trees.std(axis=0)
        y_true_all.append(y_te); y_pred_all.append(y_pred); y_std_all.append(y_std)

    if y_true_all:
        y_true = np.concatenate(y_true_all); y_pred = np.concatenate(y_pred_all); y_std = np.concatenate(y_std_all)
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))

        # Parity (CV)
        lim_lo = float(min(np.min(y_true), np.min(y_pred)))
        lim_hi = float(max(np.max(y_true), np.max(y_pred)))
        plt.figure(figsize=(5.2, 5.2))
        plt.scatter(y_true, y_pred, s=16, alpha=0.7)
        plt.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1)
        plt.xlabel("Measured (mAh g⁻¹)"); plt.ylabel("Predicted (mAh g⁻¹)")
        title = f"Parity (CV {n_splits_eff}-fold) — R²={r2:.3f}, MAE={mae:.1f}, RMSE={rmse:.1f}"
        if it_label: title += f" — {it_label}"
        plt.title(title); plt.tight_layout()
        fname = os.path.join(out_dir or PLOTS_DIR, "parity_cv.png")
        plt.savefig(fname, dpi=220); plt.close(); LOGGER.info("Saved %s", fname)

        # Residuals (CV)
        residuals = y_pred - y_true
        plt.figure(figsize=(6.4, 4.2))
        plt.hist(residuals, bins=30, alpha=0.85); plt.axvline(0, color="k", lw=1)
        plt.xlabel("Residual (Pred − Measured) [mAh g⁻¹]"); plt.ylabel("Count")
        title = "Residuals Distribution (CV)"
        if it_label: title += f" — {it_label}"
        plt.title(title); plt.tight_layout()
        fname = os.path.join(out_dir or PLOTS_DIR, "residual_hist_cv.png")
        plt.savefig(fname, dpi=220); plt.close(); LOGGER.info("Saved %s", fname)

        # Residual vs predicted (CV)
        plt.figure(figsize=(6.0, 4.6))
        plt.scatter(y_pred, residuals, s=12, alpha=0.7); plt.axhline(0, color="k", lw=1)
        plt.xlabel("Predicted (mAh g⁻¹)"); plt.ylabel("Residual (Pred − Measured) [mAh g⁻¹]")
        title = "Residuals vs Predicted (CV)"
        if it_label: title += f" — {it_label}"
        plt.title(title); plt.tight_layout()
        fname = os.path.join(out_dir or PLOTS_DIR, "residual_vs_pred_cv.png")
        plt.savefig(fname, dpi=220); plt.close(); LOGGER.info("Saved %s", fname)

        # Uncertainty reliability (CV)
        abs_err = np.abs(residuals)
        df_rel = pd.DataFrame({"std": y_std, "ae": abs_err})
        try:
            df_rel["bin"] = pd.qcut(df_rel["std"], q=5, duplicates="drop")
        except ValueError:
            edges = np.linspace(df_rel["std"].min(), df_rel["std"].max() + 1e-12, 6)
            df_rel["bin"] = pd.cut(df_rel["std"], bins=edges, include_lowest=True)
        grp = df_rel.groupby("bin", observed=True).agg(
            mean_std=("std", "mean"), mae=("ae", "mean"), n=("ae", "size")
        ).reset_index()
        rho = pd.Series(df_rel["std"]).corr(pd.Series(df_rel["ae"]), method="spearman")

        plt.figure(figsize=(6.4, 4.2))
        plt.plot(grp["mean_std"], grp["mae"], marker="o")
        for _, r in grp.iterrows():
            plt.text(r["mean_std"], r["mae"], f"n={int(r['n'])}", fontsize=8, ha="left", va="bottom")
        plt.xlabel("Predicted std across trees (relative uncertainty)")
        plt.ylabel("MAE in bin (mAh g⁻¹)")
        title = f"Uncertainty Reliability (CV) — Spearman ρ={rho:.2f}"
        if it_label: title += f" — {it_label}"
        plt.title(title); plt.tight_layout()
        fname = os.path.join(out_dir or PLOTS_DIR, "reliability_curve_cv.png")
        plt.savefig(fname, dpi=220); plt.close(); LOGGER.info("Saved %s", fname)

    # OOB (cross-check)
    rf_oob = RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1,
                                   bootstrap=True, oob_score=True)
    rf_oob.fit(X, y)
    if getattr(rf_oob, "oob_prediction_", None) is not None:
        y_pred_oob = rf_oob.oob_prediction_
        r2_oob = r2_score(y, y_pred_oob)
        mae_oob = mean_absolute_error(y, y_pred_oob)
        rmse_oob = math.sqrt(mean_squared_error(y, y_pred_oob))

        lim_lo = float(min(np.min(y), np.min(y_pred_oob)))
        lim_hi = float(max(np.max(y), np.max(y_pred_oob)))
        plt.figure(figsize=(5.2, 5.2))
        plt.scatter(y, y_pred_oob, s=16, alpha=0.7)
        plt.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1)
        plt.xlabel("Measured (mAh g⁻¹)"); plt.ylabel("Predicted (mAh g⁻¹)")
        title = f"Parity (OOB) — R²={r2_oob:.3f}, MAE={mae_oob:.1f}, RMSE={rmse_oob:.1f}"
        if it_label: title += f" — {it_label}"
        plt.title(title); plt.tight_layout()
        fname = os.path.join(out_dir or PLOTS_DIR, "parity_oob.png")
        plt.savefig(fname, dpi=220); plt.close(); LOGGER.info("Saved %s", fname)

        res_oob = y_pred_oob - y
        plt.figure(figsize=(6.4, 4.2))
        plt.hist(res_oob, bins=30, alpha=0.85); plt.axvline(0, color="k", lw=1)
        plt.xlabel("Residual (Pred − Measured) [mAh g⁻¹]"); plt.ylabel("Count")
        title = "Residuals Distribution (OOB)"
        if it_label: title += f" — {it_label}"
        plt.title(title); plt.tight_layout()
        fname = os.path.join(out_dir or PLOTS_DIR, "residual_hist_oob.png")
        plt.savefig(fname, dpi=220); plt.close(); LOGGER.info("Saved %s", fname)

        preds_trees_all = np.vstack([t.predict(X) for t in rf_oob.estimators_])
        coverages = [0.50, 0.70, 0.80, 0.90, 0.95]; actual = []
        for c in coverages:
            qlo = (1.0 - c) / 2.0 * 100.0; qhi = (1.0 + c) / 2.0 * 100.0
            lo = np.percentile(preds_trees_all, qlo, axis=0)
            hi = np.percentile(preds_trees_all, qhi, axis=0)
            covered = np.mean((y >= lo) & (y <= hi)); actual.append(covered)

        plt.figure(figsize=(5.4, 5.0))
        plt.plot(coverages, actual, marker="o", label="Actual coverage")
        plt.plot([min(coverages), max(coverages)],
                 [min(coverages), max(coverages)], "k--", lw=1, label="Ideal")
        plt.xlabel("Nominal coverage (tree percentiles)"); plt.ylabel("Actual coverage")
        title = "Interval Calibration (OOB)"
        if it_label: title += f" — {it_label}"
        plt.title(title); plt.legend()
        fname = os.path.join(out_dir or PLOTS_DIR, "interval_calibration_oob.png")
        plt.tight_layout(); plt.savefig(fname, dpi=220); plt.close()
        LOGGER.info("Saved %s", fname)
    else:
        LOGGER.info("OOB predictions not available; skipped OOB-based plots.")
