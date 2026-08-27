# config.py
# -*- coding: utf-8 -*-
"""Global configuration, logging, seeds, Matplotlib backend (Agg), and constants."""

import os, sys, logging
os.environ["MPLBACKEND"] = "Agg"         # must be set before importing pyplot
import matplotlib
matplotlib.use("Agg", force=True)        # belt & suspenders; headless plotting

# ------------- Logging -------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    force=True
)
LOGGER = logging.getLogger(__name__)

# ------------- Files / Paths -------------
TRAIN_DATA_FILE = "./dataset/extracted_data_NCM811_final_sanitized_final.csv"
DATA_SAVE_FILE  = "./dataset/data_with_optimized_condition.csv"
OPTIMIZER_FILE  = "rf_optimized_model.pkl"
ENCODER_FILE    = "rf_encoder.pkl"
ITERATION_FILE  = "iteration.txt"
PLOTS_DIR       = "plots"

# ------------- Target / Columns / Bounds -------------
TARGET_VARIABLE = "Discharge_Capacity"
BATCH_SIZE = 3
MAX_ATTEMPT = 10000

# Keep seeds exactly as in the original code
SEED = 0  # random_state=0 everywhere

import numpy as np
from typing import Tuple

# Variance-stabilising transform (kept exactly)
log_t  = np.log1p
ilog_t = np.expm1  # not used by optimizer

NUMERICAL_BOUNDS: dict[str, Tuple[int, int]] = {
    "Hydrothermal_Temperature": (170, 200),
    "Hydrothermal_Time": (9, 15),
    "First_Calcination_Temperature": (440, 600),
    "First_Calcination_Time": (3, 7),
    "Second_Calcination_Temperature": (700, 950),
    "Second_Calcination_Time": (5, 27),
}
CAT_COLS = ["Li_Precursor", "Ni_Precursor", "Co_Precursor", "Mn_Precursor"]
NUM_COLS = list(NUMERICAL_BOUNDS.keys())
ALL_OPT_COLS = CAT_COLS + NUM_COLS

# Optional cleanup if Tk ever sneaks in (defensive)
import atexit
def _cleanup_tk_on_exit():
    if "tkinter" in sys.modules:
        try:
            import tkinter as _tk
            root = getattr(_tk, "_default_root", None)
            if root is not None:
                root.destroy()
        except Exception:
            pass
atexit.register(_cleanup_tk_on_exit)
