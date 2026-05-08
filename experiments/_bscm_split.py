"""
Deterministic train/test/holdout split utilities for BSCM experiments.

Defines two disjoint sample pools over the 2000-sample So2Sat subsample:
    HOLDOUT  : 200 samples for hyperparameter search (Phase 2)
    EVAL     : 1800 samples for kernel-SVM evaluation (Phase 3)
both produced by a single deterministic stratified shuffle (seed 1234)
so every script that imports this module sees the same pools.

The split is stratified over the 17 LCZ classes when class labels are
available; otherwise it falls back to a uniform-random split.

This module also persists the index arrays to disk so the protocol is
auditable and reproducible across machines.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

SPLIT_SEED = 1234   # locked, do not change.
HOLDOUT_SIZE = 200
SPLIT_PATH = os.path.join(config.RESULTS_DIR, "bscm", "split_indices.npz")


@dataclass
class IndexSplit:
    holdout: np.ndarray  # shape (200,)
    eval_pool: np.ndarray  # shape (1800,)
    n_total: int

    def assert_disjoint(self) -> None:
        assert len(set(self.holdout.tolist()) & set(self.eval_pool.tolist())) == 0
        assert len(self.holdout) + len(self.eval_pool) == self.n_total


def make_or_load_split(y: np.ndarray) -> IndexSplit:
    """Return the deterministic split, creating and persisting it if absent.

    The split is stratified over `y` if y has at least 2 unique classes.
    """
    if os.path.exists(SPLIT_PATH):
        d = np.load(SPLIT_PATH)
        sp = IndexSplit(
            holdout=d["holdout"], eval_pool=d["eval_pool"],
            n_total=int(d["n_total"]),
        )
        if sp.n_total != len(y):
            raise RuntimeError(
                f"Stored split has n_total={sp.n_total}, current data has "
                f"len(y)={len(y)}.  Delete {SPLIT_PATH} to regenerate."
            )
        sp.assert_disjoint()
        return sp

    n_total = len(y)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=HOLDOUT_SIZE,
                                  random_state=SPLIT_SEED)
    (eval_pool, holdout), = sss.split(np.zeros(n_total), y)
    eval_pool = np.sort(eval_pool)
    holdout = np.sort(holdout)
    sp = IndexSplit(holdout=holdout, eval_pool=eval_pool, n_total=n_total)
    sp.assert_disjoint()

    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    np.savez(SPLIT_PATH,
             holdout=holdout, eval_pool=eval_pool,
             n_total=np.array(n_total))
    return sp
