"""
audit_pool_variance_rbf.py
==========================
Quantifies the fixed-pool sampling variance of tuned RBF-SVM on So2Sat
physics-16 features at the scale of the E26 / E32 pools.

Why
---
The paper (§3.1, "Within-pool versus cross-pool comparisons", and
Limitations (vii)) states that tuned RBF-SVM macro-F1 on independent
N~=1{,}800 stratified draws from the So2Sat physics-Fisher pool can
differ by ~0.05 F1, citing the E26 (Pool A, 0.518 +/- 0.020) vs E32
(Pool B, 0.465 +/- 0.028) difference as evidence.  That claim was
hand-waved.  This script substantiates it by directly resampling.

Protocol
--------
* Load the 2{,}000-sample So2Sat physics-16 pool (the E26 superset
  produced by ``scripts/extract_so2sat_10k.py`` / cached as
  ``physics_features_16.npz`` and exposed via
  ``experiments._e_common.load_physics_16``).
* For ``N_RESAMPLES`` distinct stratified sub-pools of size
  ``POOL_SIZE`` (drawn without replacement from the 2{,}000-sample
  superset, one fresh sub-pool per resample seed), do exactly what
  E26's classical baseline does: a single ``StratifiedShuffleSplit``
  (test_frac = 0.30, random_state = 42 inside that sub-pool), then
  ``GridSearchCV`` over the matched RBF $(C, \\gamma)$ grid with
  3-fold inner CV, ``scoring='f1_macro'``, ``class_weight='balanced'``.
* Record macro-F1 per resample.  Report mean / std / quantile spread
  across the ``N_RESAMPLES`` independent sub-pools.  The std of this
  distribution is the empirical estimate of the cross-pool variance.

This is a purely classical script -- no PennyLane required.

Output: ``results/pool_variance_rbf/rbf_pool_variance.json``
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import List

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._e_common import load_physics_16

# --------------------------------------------------------------------------- #
# Protocol constants -- match E26's classical RBF baseline exactly.
# --------------------------------------------------------------------------- #
POOL_SIZE = 1800            # match E32 fixed-pool size
N_RESAMPLES = 50
TEST_FRAC = 0.30
SPLIT_SEED = 42             # within-pool eval seed (same as E26)
INNER_CV = 3
C_GRID = [1, 10, 100]
GAMMA_GRID = ["scale", 0.01, 0.1, 1.0]
RESAMPLE_SEED_BASE = 1000   # disjoint from any per-experiment seed in repo

OUT_DIR = os.path.join(config.RESULTS_DIR, "pool_variance_rbf")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "rbf_pool_variance.json")
LOG_PATH = os.path.join(OUT_DIR, "audit_pool_variance.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
log = logging.getLogger("pool_var")


def _stratified_subsample(y: np.ndarray, pool_size: int,
                          rng: np.random.Generator) -> np.ndarray:
    """Stratified sub-pool of ``pool_size`` indices from ``y``, no replacement.

    Class quotas are proportional to class frequency in the full ``y``,
    floored, with the residual (pool_size - sum(quotas)) distributed by
    fractional remainder.
    """
    classes, counts = np.unique(y, return_counts=True)
    frac = counts / counts.sum()
    raw = frac * pool_size
    quotas = np.floor(raw).astype(int)
    deficit = pool_size - int(quotas.sum())
    if deficit > 0:
        # Assign the deficit by largest fractional remainder.
        order = np.argsort(-(raw - quotas))
        for i in range(deficit):
            quotas[order[i]] += 1
    chosen: List[int] = []
    for c, q in zip(classes, quotas):
        idx = np.flatnonzero(y == c)
        if q >= len(idx):
            chosen.extend(idx.tolist())
        else:
            pick = rng.choice(idx, size=q, replace=False)
            chosen.extend(pick.tolist())
    out = np.array(sorted(chosen), dtype=np.int64)
    assert len(out) == pool_size, (len(out), pool_size)
    return out


def _eval_rbf(X_raw: np.ndarray, y: np.ndarray, eval_seed: int) -> dict:
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRAC, random_state=eval_seed,
    )
    (tr, te), = sss.split(np.zeros(len(y)), y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_raw[tr])
    X_te = scaler.transform(X_raw[te])
    grid = {"C": C_GRID, "gamma": GAMMA_GRID}
    clf = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced"),
        grid, cv=INNER_CV, scoring="f1_macro", n_jobs=-1,
    )
    clf.fit(X_tr, y[tr])
    yp = clf.predict(X_te)
    return {
        "macro_f1": float(f1_score(y[te], yp, average="macro",
                                   zero_division=0)),
        "best_C": float(clf.best_params_["C"]),
        "best_gamma": (
            float(clf.best_params_["gamma"])
            if not isinstance(clf.best_params_["gamma"], str)
            else clf.best_params_["gamma"]
        ),
    }


def main() -> None:
    log.info("=" * 72)
    log.info("  RBF-SVM pool-variance bootstrap")
    log.info("  pool_size=%d, resamples=%d", POOL_SIZE, N_RESAMPLES)
    log.info("=" * 72)
    _, X_raw_all, y_all = load_physics_16()
    log.info("loaded physics-16 superset: X=%s y=%s (n_classes=%d)",
             X_raw_all.shape, y_all.shape, int(np.unique(y_all).size))

    f1s: List[float] = []
    records: List[dict] = []
    for k in range(N_RESAMPLES):
        rng = np.random.default_rng(RESAMPLE_SEED_BASE + k)
        idx = _stratified_subsample(y_all, POOL_SIZE, rng)
        Xk = X_raw_all[idx]
        yk = y_all[idx]
        t0 = time.time()
        res = _eval_rbf(Xk, yk, SPLIT_SEED)
        dt = time.time() - t0
        f1s.append(res["macro_f1"])
        records.append({
            "resample_idx": int(k),
            "resample_seed": int(RESAMPLE_SEED_BASE + k),
            "macro_f1": res["macro_f1"],
            "best_C": res["best_C"],
            "best_gamma": res["best_gamma"],
            "seconds": float(dt),
        })
        log.info("  [%2d/%2d] F1=%.4f  C=%s gamma=%s  (%.0fs)",
                 k + 1, N_RESAMPLES, res["macro_f1"],
                 res["best_C"], res["best_gamma"], dt)

    arr = np.asarray(f1s, dtype=float)
    summary = {
        "protocol": {
            "pool_size": POOL_SIZE,
            "n_resamples": N_RESAMPLES,
            "test_fraction": TEST_FRAC,
            "eval_split_seed": SPLIT_SEED,
            "inner_cv_folds": INNER_CV,
            "C_grid": C_GRID,
            "gamma_grid": GAMMA_GRID,
            "resample_seed_base": RESAMPLE_SEED_BASE,
            "feature_set": "physics_features_16 (E26/E32 superset)",
            "subsample_method": "stratified without replacement",
        },
        "summary": {
            "f1_mean": float(arr.mean()),
            "f1_std": float(arr.std(ddof=1)),
            "f1_min": float(arr.min()),
            "f1_max": float(arr.max()),
            "f1_q025": float(np.quantile(arr, 0.025)),
            "f1_q975": float(np.quantile(arr, 0.975)),
            "max_minus_min": float(arr.max() - arr.min()),
            "iqr_q75_minus_q25": float(
                np.quantile(arr, 0.75) - np.quantile(arr, 0.25)
            ),
        },
        "records": records,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    s = summary["summary"]
    log.info("")
    log.info("RBF macro-F1 across %d pool draws (N=%d each):",
             N_RESAMPLES, POOL_SIZE)
    log.info("  mean       = %.4f", s["f1_mean"])
    log.info("  std (s.d.) = %.4f", s["f1_std"])
    log.info("  range      = [%.4f, %.4f]   (max - min = %.4f)",
             s["f1_min"], s["f1_max"], s["max_minus_min"])
    log.info("  95%% band  = [%.4f, %.4f]", s["f1_q025"], s["f1_q975"])
    log.info("")
    log.info("Wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
