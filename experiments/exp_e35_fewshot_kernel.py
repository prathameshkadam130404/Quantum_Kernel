"""
E35 -- Few-Shot Kernel Adaptation: Learning Curves on Precomputed Gram Matrices.

Motivation
----------
A high-quality kernel captures domain structure in its similarity matrix:
same-class pairs have high K(x,x'), different-class pairs have low K(x,x').
If this structure is genuine, an SVM should achieve reasonable accuracy
even with *very few* labelled training samples, because the kernel has
already encoded the relevant relationships.

This experiment measures how quickly SVM performance grows as a function
of labelled training samples per class (the "few-shot learning curve"),
using the *precomputed* gram matrices from E33.  No new quantum circuits
are evaluated — the experiment is pure NumPy submatrix indexing.

Scientific questions:
  1. Does SG-BSCM produce a useful similarity structure on EuroSAT
     physics8, where BSCM-uniform catastrophically fails?
  2. How many labelled samples does each kernel need to reach 90% of
     its saturated performance?
  3. Does the quantum kernel learning curve dominate the classical
     RBF-SVM curve (better sample efficiency)?

Protocol
--------
For each (dataset, feature_set) condition from E33:
  Load the precomputed gram matrices K(N×N) for:
      SG-BSCM, BSCM-uniform, SRQFM-fid
  Load raw features for:
      RBF-SVM, RandomForest (retrained per shot-level)

  For each shot level k in {2, 5, 10, 20, 50}:
    For each of 20 random seeds:
      1. Sample k instances per class (stratified) as "train".
      2. Use all remaining instances as "test".
      3. For quantum kernels: extract K[train,train] and K[test,train]
         submatrices; train SVM(kernel='precomputed'); predict; measure F1.
      4. For RBF-SVM: fit StandardScaler + GridSearchCV on raw features.
      5. For RandomForest: fit on raw features.
      6. Record macro-F1 per (condition, method, k, seed).

  Aggregate: mean ± std over 20 seeds at each shot level.

Shot levels are capped per dataset:
    So2Sat:  {2, 5}             (smallest class = 13 in eval pool;
                                 must leave ≥ 3 for testing)
    EuroSAT: {2, 5, 10, 20, 50} (smallest class = 108)

Output
------
    results/sg_bscm/fewshot_curves.csv    (long-form per-seed results)
    results/sg_bscm/fewshot_summary.json  (aggregated means + stds)
    results/sg_bscm/exp_e35.log

Dependencies
------------
    Requires E33 gram matrices in results/sg_bscm/K_*.npy.
    If a kernel cache is missing, that method is skipped with a warning.

Author: Prathamesh Kadam et al.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Reuse E33 data loaders for identical data splits.
from experiments.exp_e33_sg_bscm import _load_eurosat, _load_so2sat

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAVE_DIR = os.path.join(config.RESULTS_DIR, "sg_bscm")
os.makedirs(SAVE_DIR, exist_ok=True)

# Gram matrix directory (written by E33).
KERNEL_DIR = SAVE_DIR

# Quantum kernel methods (must match E33 naming convention).
KERNEL_METHODS = ["SG-BSCM", "BSCM-uniform", "SRQFM-fid"]

# Shot levels per dataset. Capped so that k < (min_class_size - 3).
SHOT_LEVELS: Dict[str, List[int]] = {
    "so2sat": [2, 5],                  # min class = 13 → max train 10, keep ≥3 test
    "eurosat": [2, 5, 10, 20, 50],     # min class = 108 → ample headroom
}

# Number of random sampling seeds per shot level.
N_SEEDS = 20

# RBF-SVM hyperparameter grid (same as E33).
C_GRID = [1, 10, 100]
GAMMA_GRID = ["scale", 0.01, 0.1, 1.0]
CV_FOLDS = 3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "exp_e35.log"), mode="w"),
    ],
)
log = logging.getLogger("exp_e35")


# ============================================================================
# Stratified few-shot sampler
# ============================================================================

def _stratified_sample(
    y: np.ndarray,
    k_per_class: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample exactly k instances per class; return (train_idx, test_idx).

    If a class has fewer than k+1 instances, it is skipped entirely
    (removed from both train and test) and a warning is logged.

    Returns sorted index arrays into the original y array.
    """
    classes = np.unique(y)
    train_idx_list: List[int] = []
    test_idx_list: List[int] = []

    for c in classes:
        members = np.where(y == c)[0]
        if len(members) < k_per_class + 1:
            # Not enough samples to have k train + at least 1 test.
            log.debug(
                "Class %d has only %d samples, need %d+1. Skipping.",
                c, len(members), k_per_class,
            )
            continue
        chosen = rng.choice(members, size=k_per_class, replace=False)
        rest = np.setdiff1d(members, chosen)
        train_idx_list.extend(chosen.tolist())
        test_idx_list.extend(rest.tolist())

    return np.sort(train_idx_list), np.sort(test_idx_list)


# ============================================================================
# Evaluators
# ============================================================================

def _eval_precomputed_kernel(
    K: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
) -> float:
    """Train SVM on K[tr,tr], predict on K[te,tr], return macro-F1.

    Uses a fixed C=10 for speed (no inner CV at few-shot scale — too few
    samples for reliable 3-fold CV with k=2 per class).  Sensitivity to C
    is reported in E33 at full-data scale.
    """
    K_tr = K[np.ix_(tr, tr)]
    K_te = K[np.ix_(te, tr)]
    y_tr, y_te = y[tr], y[te]

    # Use C=10 as default; add C=1 and C=100 only when we have enough
    # training samples for inner CV (at least 3 per class).
    min_class_n = min(np.bincount(y_tr)) if len(np.bincount(y_tr)) > 0 else 0
    if min_class_n >= CV_FOLDS:
        # Enough samples for inner CV.
        best_c, best_f1 = 10, -1.0
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        for C in C_GRID:
            fold_f1: List[float] = []
            try:
                for cv_tr, cv_val in skf.split(np.zeros(len(y_tr)), y_tr):
                    clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
                    clf.fit(K_tr[np.ix_(cv_tr, cv_tr)], y_tr[cv_tr])
                    yp = clf.predict(K_tr[np.ix_(cv_val, cv_tr)])
                    fold_f1.append(float(
                        f1_score(y_tr[cv_val], yp, average="macro", zero_division=0)
                    ))
            except ValueError:
                continue
            m = float(np.mean(fold_f1))
            if m > best_f1:
                best_f1, best_c = m, C
    else:
        best_c = 10  # Default when CV is infeasible.

    clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    clf.fit(K_tr, y_tr)
    yp = clf.predict(K_te)
    return float(f1_score(y_te, yp, average="macro", zero_division=0))


def _eval_rbf_svm(
    X_raw: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
) -> float:
    """Train RBF-SVM on raw features, return macro-F1."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_raw[tr])
    X_te = scaler.transform(X_raw[te])
    y_tr, y_te = y[tr], y[te]

    min_class_n = min(np.bincount(y_tr)) if len(np.bincount(y_tr)) > 0 else 0
    if min_class_n >= CV_FOLDS:
        grid = {"C": C_GRID, "gamma": GAMMA_GRID}
        clf = GridSearchCV(
            SVC(kernel="rbf", class_weight="balanced"),
            grid, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1,
        )
    else:
        clf = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")

    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    return float(f1_score(y_te, yp, average="macro", zero_division=0))


def _eval_rf(
    X_raw: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    seed: int,
) -> float:
    """Train RandomForest on raw features, return macro-F1."""
    clf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    clf.fit(X_raw[tr], y[tr])
    yp = clf.predict(X_raw[te])
    return float(f1_score(y[te], yp, average="macro", zero_division=0))


# ============================================================================
# Per-condition evaluation
# ============================================================================

def _evaluate_condition(
    dataset: str,
    fset: str,
    y: np.ndarray,
    X_raw: np.ndarray,
    kernels: Dict[str, np.ndarray],
    rows: List[dict],
) -> None:
    """Run full few-shot learning curve for one (dataset, fset) condition."""
    cond = f"{dataset}/{fset}"
    n_classes = len(np.unique(y))
    shots = SHOT_LEVELS[dataset]

    log.info("\n" + "=" * 72)
    log.info("  Condition: %s   N=%d   classes=%d", cond, len(y), n_classes)
    log.info("  Shot levels: %s   Seeds/level: %d", shots, N_SEEDS)
    log.info("  Quantum kernels available: %s", list(kernels.keys()))
    log.info("=" * 72)

    for k in shots:
        f1_collector: Dict[str, List[float]] = {m: [] for m in list(kernels.keys()) + ["RBF-SVM", "RandomForest"]}

        for seed_idx in range(N_SEEDS):
            rng = np.random.default_rng(1000 * k + seed_idx)
            tr, te = _stratified_sample(y, k, rng)

            if len(tr) == 0 or len(te) == 0:
                log.warning("  k=%d seed=%d: empty split, skipping.", k, seed_idx)
                continue

            # Quantum kernels (precomputed gram submatrices).
            for method, K in kernels.items():
                f1 = _eval_precomputed_kernel(K, y, tr, te)
                f1_collector[method].append(f1)
                rows.append({
                    "dataset": dataset, "feature_set": fset,
                    "method": method, "k_per_class": k,
                    "seed": seed_idx, "macro_f1": f1,
                    "n_train": len(tr), "n_test": len(te),
                })

            # Classical baselines.
            f1_rbf = _eval_rbf_svm(X_raw, y, tr, te)
            f1_collector["RBF-SVM"].append(f1_rbf)
            rows.append({
                "dataset": dataset, "feature_set": fset,
                "method": "RBF-SVM", "k_per_class": k,
                "seed": seed_idx, "macro_f1": f1_rbf,
                "n_train": len(tr), "n_test": len(te),
            })

            f1_rf = _eval_rf(X_raw, y, tr, te, seed=seed_idx)
            f1_collector["RandomForest"].append(f1_rf)
            rows.append({
                "dataset": dataset, "feature_set": fset,
                "method": "RandomForest", "k_per_class": k,
                "seed": seed_idx, "macro_f1": f1_rf,
                "n_train": len(tr), "n_test": len(te),
            })

        # Log aggregated results for this shot level.
        log.info("\n  k=%d per class (%s):", k, cond)
        for method in f1_collector:
            vals = f1_collector[method]
            if vals:
                arr = np.array(vals)
                log.info(
                    "    %-15s  F1 = %.4f +/- %.4f  (n=%d)",
                    method, arr.mean(), arr.std(), len(arr),
                )


# ============================================================================
# Kernel loader
# ============================================================================

def _load_kernels(dataset: str, fset: str) -> Dict[str, np.ndarray]:
    """Load precomputed gram matrices for the given condition.

    Checks E33's output directory first, then falls back to prior
    experiments (E32 for So2Sat, EuroSAT fixed-pool for EuroSAT).
    Missing kernels are skipped with a warning.
    """
    # Fallback paths from prior experiments (identical kernel configs).
    _fallback_map = {
        ("BSCM-uniform", "so2sat", "physics8"):
            os.path.join(config.RESULTS_DIR, "bscm", "K_bscm_physics8_uniform.npy"),
        ("BSCM-uniform", "so2sat", "pca8"):
            os.path.join(config.RESULTS_DIR, "bscm", "K_bscm_pca8_uniform.npy"),
        ("BSCM-uniform", "eurosat", "physics8"):
            os.path.join(config.RESULTS_DIR, "eurosat_bscm_fixed", "kernels",
                         "K_pool_bscm_physics8.npy"),
        ("BSCM-uniform", "eurosat", "pca8"):
            os.path.join(config.RESULTS_DIR, "eurosat_bscm_fixed", "kernels",
                         "K_pool_bscm_pca8.npy"),
        ("SRQFM-fid", "so2sat", "physics8"):
            os.path.join(config.RESULTS_DIR, "bscm", "K_srqfm_fid_physics8.npy"),
        ("SRQFM-fid", "so2sat", "pca8"):
            os.path.join(config.RESULTS_DIR, "bscm", "K_srqfm_fid_pca8.npy"),
        ("SRQFM-fid", "eurosat", "physics8"):
            os.path.join(config.RESULTS_DIR, "eurosat_bscm_fixed", "kernels",
                         "K_pool_srqfm_linear_physics8.npy"),
        ("SRQFM-fid", "eurosat", "pca8"):
            os.path.join(config.RESULTS_DIR, "eurosat_bscm_fixed", "kernels",
                         "K_pool_srqfm_linear_pca8.npy"),
    }

    kernels: Dict[str, np.ndarray] = {}
    for method in KERNEL_METHODS:
        # Primary: E33 cache.
        path = os.path.join(KERNEL_DIR, f"K_{method}_{dataset}_{fset}.npy")
        if os.path.exists(path):
            K = np.load(path)
            kernels[method] = K
            log.info("  Loaded %s  shape=%s", path, K.shape)
            continue

        # Fallback: prior experiments.
        fb = _fallback_map.get((method, dataset, fset))
        if fb is not None and os.path.exists(fb):
            K = np.load(fb)
            kernels[method] = K
            log.info("  [FALLBACK] Loaded %s from %s  shape=%s", method, fb, K.shape)
            continue

        log.warning("  MISSING kernel: %s for %s/%s  (skipping)", method, dataset, fset)
    return kernels


# ============================================================================
# Summary builder
# ============================================================================

def _build_summary(df: pd.DataFrame) -> dict:
    """Aggregate per-seed results into mean/std per (condition, method, k)."""
    summary: Dict[str, dict] = {}
    for (ds, fs, method, k), grp in df.groupby(
        ["dataset", "feature_set", "method", "k_per_class"]
    ):
        key = f"{ds}/{fs}"
        if key not in summary:
            summary[key] = {}
        if method not in summary[key]:
            summary[key][method] = {}
        vals = grp["macro_f1"].values
        summary[key][method][int(k)] = {
            "f1_mean": round(float(vals.mean()), 4),
            "f1_std": round(float(vals.std()), 4),
            "n_seeds": len(vals),
        }
    return summary


# ============================================================================
# Main
# ============================================================================

def run() -> pd.DataFrame:
    """Execute the full E35 few-shot kernel adaptation experiment."""
    t_wall = time.time()

    log.info("=" * 72)
    log.info("  E35: Few-Shot Kernel Adaptation Learning Curves")
    log.info("=" * 72)
    log.info("  Date:        %s", datetime.now().isoformat())
    log.info("  N_SEEDS:     %d per shot level", N_SEEDS)
    log.info("  Shot levels: So2Sat %s  EuroSAT %s",
             SHOT_LEVELS["so2sat"], SHOT_LEVELS["eurosat"])

    rows: List[dict] = []

    # -- Iterate conditions ---------------------------------------------------
    conditions = [
        ("so2sat", "physics8"), ("so2sat", "pca8"),
        ("eurosat", "physics8"), ("eurosat", "pca8"),
    ]

    for dataset, fset in conditions:
        # Load data (same loaders as E33 — identical data splits).
        if dataset == "so2sat":
            X_enc, X_raw, y, names = _load_so2sat(fset)
        else:
            X_enc, X_raw, y, names = _load_eurosat(fset)

        # Load precomputed gram matrices.
        kernels = _load_kernels(dataset, fset)
        if not kernels:
            log.warning("  No kernels for %s/%s, skipping condition.", dataset, fset)
            continue

        _evaluate_condition(dataset, fset, y, X_raw, kernels, rows)

    # -- Save results ---------------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = os.path.join(SAVE_DIR, "fewshot_curves.csv")
    df.to_csv(csv_path, index=False)
    log.info("\n  Saved: %s  (%d rows)", csv_path, len(df))

    summary = {
        "experiment": "E35_fewshot_kernel",
        "n_seeds_per_level": N_SEEDS,
        "shot_levels": SHOT_LEVELS,
        "curves": _build_summary(df),
        "wall_clock_s": round(time.time() - t_wall, 1),
    }
    json_path = os.path.join(SAVE_DIR, "fewshot_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("  Saved: %s", json_path)

    log.info("\n  E35 complete.  Wall clock: %.1f min", (time.time() - t_wall) / 60)
    return df


if __name__ == "__main__":
    run()
