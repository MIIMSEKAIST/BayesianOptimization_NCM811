# optimizer_utils.py
# -*- coding: utf-8 -*-
"""skopt space definition, optimizer init, candidate suggestion, best decode."""

import numpy as np
import pandas as pd
from skopt import Optimizer
from skopt.learning import RandomForestRegressor as SKO_RFR
from skopt.space import Categorical, Integer

from config import (
    LOGGER, SEED, NUMERICAL_BOUNDS, CAT_COLS, NUM_COLS, ALL_OPT_COLS
)

def build_space(enc):
    return [Categorical(list(range(len(enc.categories_[i]))), name=c)
            for i, c in enumerate(CAT_COLS)] + \
           [Integer(*NUMERICAL_BOUNDS[c], name=c) for c in NUM_COLS]

def init_optimizer(space, X0, y0):
    # Keep seeds/settings exactly as your original code
    rf = SKO_RFR(n_estimators=200, n_jobs=-1, random_state=SEED)
    opt = Optimizer(space, base_estimator=rf, acq_func="EI", random_state=SEED)
    opt.tell(X0.tolist(), y0.tolist())
    return opt

def suggest_unique(opt, enc_exist: pd.DataFrame, batch: int, max_attempt: int = 10000):
    seen = set(map(tuple, enc_exist[ALL_OPT_COLS].values.astype(float)))
    new, tries = [], 0
    while len(new) < batch and tries < max_attempt:
        need = batch - len(new)
        for row in opt.ask(n_points=need):
            tup = tuple(map(float, row))
            if tup not in seen:
                new.append(row); seen.add(tup)
        tries += 1
    if len(new) < batch:
        LOGGER.warning("Space exhausted – %d/%d", len(new), batch)
        return None
    return pd.DataFrame(new, columns=ALL_OPT_COLS)

def optimizer_best(opt, enc):
    idx = int(np.argmin(opt.yi))
    best_enc = opt.Xi[idx]
    best = {d.name: v for d, v in zip(opt.space, best_enc)}
    for col in CAT_COLS:
        best[col] = enc.categories_[CAT_COLS.index(col)][int(best[col])]
    return best
