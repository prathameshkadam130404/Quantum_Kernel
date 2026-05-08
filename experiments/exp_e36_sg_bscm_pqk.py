"""
E36 -- SG-BSCM Projected Quantum Kernel (SG-BSCM-PQK) Evaluation.

Motivation
----------
PQK (Huang et al., 2021) extracts per-qubit Bloch vectors from the
feature-map circuit and builds a classical RBF kernel on them.  This
sidesteps the N^2 fidelity bottleneck (only N circuit evaluations) and
is immune to the concentration that plagues fidelity kernels.

The existing SRQFM-PQK uses the ZZ entangling channel.  SG-BSCM-PQK
uses the XX+YY channel with singlet gating — the same circuit as the
SG-BSCM fidelity kernel in E33, but with local Pauli measurements
instead of a fidelity overlap.  This tests whether the richer XX+YY
geometry encodes more discriminative information into the Bloch vectors
than ZZ alone.

Protocol
--------
Same 4 conditions as E33:
    So2Sat  {physics8, pca8}   N=1800, 17 classes
    EuroSAT {physics8, pca8}   N=1500, 10 classes

Methods per condition:
    SG-BSCM-PQK   XX+YY singlet-gated circuit, Bloch RBF  (NEW)
    BSCM-PQK       XX+YY uniform circuit, Bloch RBF        (NEW)
    SRQFM-PQK     ZZ circuit, Bloch RBF                    (reuse/compute)
    RBF-SVM        classical baseline
    RandomForest   classical baseline

Evaluation:
    5 seeds [42..46], stratified split, inner-CV C-tuning,
    macro-F1, spectral metrics, Wilcoxon + Holm, Cohen's d.

Output
------
    results/sg_bscm_pqk/exp_e36.log
    results/sg_bscm_pqk/metrics.csv
    results/sg_bscm_pqk/summary.json
    results/sg_bscm_pqk/bloch_*.npy   (cached Bloch vectors)
    results/sg_bscm_pqk/K_*.npy       (cached gram matrices)

Author: Prathamesh Kadam et al.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._e_common import cohens_d, holm_bonferroni, spectral_metrics
from experiments.exp_e33_sg_bscm import (
    _load_eurosat,
    _load_so2sat,
    _kta_class_centred,
)
from src.bscm_kernel import (
    BELL_WEIGHT_PRESETS,
    compute_bscm_pqk_gram,
    extract_bscm_bloch_vectors,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS           = config.SEED_LIST            # [42, 43, 44, 45, 46]
N_QUBITS        = config.N_QUBITS             # 8
REPS            = config.ZZ_REPS              # 2
TAU_LOCKED      = 0.25                        # matches E33
PQK_GAMMA       = config.PQK_GAMMA           # 0.67
C_GRID          = [1, 10, 100]
CV_FOLDS        = 3
SO2SAT_TEST_FRAC  = 0.30
EUROSAT_N_TRAIN   = 1000
EUROSAT_N_TEST    = 500

SAVE_DIR = os.path.join(config.RESULTS_DIR, "sg_bscm_pqk")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "exp_e36.log"), mode="w"),
    ],
)
log = logging.getLogger("exp_e36")


# ============================================================================
# SVM helpers (identical to E33)
# ============================================================================

def _tune_c_precomputed(
    K_train: np.ndarray,
    y_train: np.ndarray,
    c_grid: List[float] = C_GRID,
    n_folds: int = CV_FOLDS,
    seed: int = 42,
) -> float:
    """Select best C by inner stratified k-fold CV on precomputed gram."""
    best_c, best_f1 = c_grid[0], -1.0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for C in c_grid:
        fold_f1: List[float] = []
        for cv_tr, cv_val in skf.split(np.zeros(len(y_train)), y_train):
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_train[np.ix_(cv_tr, cv_tr)], y_train[cv_tr])
            yp = clf.predict(K_train[np.ix_(cv_val, cv_tr)])
            fold_f1.append(float(
                f1_score(y_train[cv_val], yp, average="macro", zero_division=0)
            ))
        m = float(np.mean(fold_f1))
        if m > best_f1:
            best_f1, best_c = m, C
    return best_c


def _eval_kernel(
    K_full: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    seed: int,
) -> dict:
    """Evaluate a precomputed kernel on a given train/test split."""
    K_tr = K_full[np.ix_(tr, tr)]
    K_te = K_full[np.ix_(te, tr)]
    best_c = _tune_c_precomputed(K_tr, y[tr], seed=seed)
    clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    yp = clf.predict(K_te)
    return {
        "macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0)),
        "best_C": best_c,
    }


def _eval_rbf(
    X_raw: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
) -> dict:
    """Evaluate RBF-SVM with GridSearchCV over C x gamma."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_raw[tr])
    X_te = scaler.transform(X_raw[te])
    grid = {"C": C_GRID, "gamma": ["scale", 0.01, 0.1, 1.0]}
    clf = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced"),
        grid, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1,
    )
    clf.fit(X_tr, y[tr])
    yp = clf.predict(X_te)
    return {
        "macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0)),
        "best_C": clf.best_params_["C"],
        "best_gamma": str(clf.best_params_["gamma"]),
    }


def _eval_rf(
    X_raw: np.ndarray,
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    seed: int,
) -> dict:
    """Evaluate Random Forest."""
    clf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced",
        random_state=seed, n_jobs=-1,
    )
    clf.fit(X_raw[tr], y[tr])
    yp = clf.predict(X_raw[te])
    return {
        "macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0)),
    }


# ============================================================================
# SRQFM-PQK Bloch extraction (ZZ channel)
# ============================================================================

def _extract_srqfm_bloch(
    X_enc: np.ndarray,
    n_qubits: int,
    cache_path: str,
) -> np.ndarray:
    """Extract Bloch vectors from the SRQFM (IsingZZ, linear) circuit.

    Uses the same circuit as the published SRQFM-PQK: H-RZ-IsingZZ(sin^2)
    with linear connectivity.  Reuses cached Bloch arrays from the
    FQK/PQK experiment when available.
    """
    import pennylane as qml

    if os.path.exists(cache_path):
        bloch = np.load(cache_path)
        if bloch.shape == (len(X_enc), 3 * n_qubits):
            log.info("  [CACHE] Loaded SRQFM Bloch from %s", cache_path)
            return bloch

    dev = config.get_device(n_qubits)
    pairs = [(i, i + 1) for i in range(n_qubits - 1)]

    def _apply_srqfm(x):
        for _ in range(REPS):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for i in range(n_qubits):
                qml.RZ(x[i], wires=i)
            for qi, qj in pairs:
                c = float(np.sin((x[qi] - x[qj]) / 2.0) ** 2)
                if c > 1e-8:
                    qml.IsingZZ(c, wires=[qi, qj])

    @qml.qnode(dev, diff_method=None)
    def mx(x):
        _apply_srqfm(x)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def my(x):
        _apply_srqfm(x)
        return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def mz(x):
        _apply_srqfm(x)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    from tqdm import tqdm
    N = len(X_enc)
    bloch = np.zeros((N, 3 * n_qubits), dtype=np.float64)
    t0 = time.time()
    for idx in tqdm(range(N), desc=f"SRQFM-PQK Bloch (N={N})", unit="sample"):
        bx = np.asarray(mx(X_enc[idx]), dtype=np.float64)
        by = np.asarray(my(X_enc[idx]), dtype=np.float64)
        bz = np.asarray(mz(X_enc[idx]), dtype=np.float64)
        for q in range(n_qubits):
            bloch[idx, 3 * q]     = bx[q]
            bloch[idx, 3 * q + 1] = by[q]
            bloch[idx, 3 * q + 2] = bz[q]

    elapsed = time.time() - t0
    log.info("  SRQFM Bloch: %d samples in %.0f s", N, elapsed)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, bloch)
    return bloch


# ============================================================================
# Kernel builder
# ============================================================================

def _build_pqk_kernel(
    X_enc: np.ndarray,
    dataset: str,
    fset: str,
    method: str,
) -> np.ndarray:
    """Build and cache a PQK gram matrix for the given method.

    Steps:
      1. Extract Bloch vectors (cached per method+dataset+fset).
      2. Build RBF gram from Bloch vectors (cached).
    """
    gram_cache = os.path.join(SAVE_DIR, f"K_{method}_{dataset}_{fset}.npy")
    bloch_cache = os.path.join(SAVE_DIR, f"bloch_{method}_{dataset}_{fset}.npy")
    desc = f"{method}({dataset}/{fset})"

    # Check gram cache first.
    if os.path.exists(gram_cache):
        log.info("  [CACHE] Loading %s gram from %s", desc, gram_cache)
        return np.load(gram_cache)

    n_q = X_enc.shape[1]

    # Extract Bloch vectors.
    if method == "SG-BSCM-PQK":
        bloch = extract_bscm_bloch_vectors(
            X_enc, n_qubits=n_q, reps=REPS, tau=TAU_LOCKED,
            coupling_threshold=1e-4, connectivity="all",
            bell_weights=BELL_WEIGHT_PRESETS["uniform"],
            singlet_gated=True,
            cache_path=bloch_cache, desc="SG-BSCM-PQK",
        )
    elif method == "BSCM-PQK":
        bloch = extract_bscm_bloch_vectors(
            X_enc, n_qubits=n_q, reps=REPS, tau=TAU_LOCKED,
            coupling_threshold=1e-4, connectivity="all",
            bell_weights=BELL_WEIGHT_PRESETS["uniform"],
            singlet_gated=False,
            cache_path=bloch_cache, desc="BSCM-PQK",
        )
    elif method == "SRQFM-PQK":
        # Try fallback from FQK/PQK experiment first.
        fb = os.path.join(
            config.RESULTS_DIR, "eurosat_fqk_pqk_fixed", "kernels",
            f"bloch_pqk_{fset}.npy",
        )
        if dataset == "eurosat" and os.path.exists(fb):
            fb_bloch = np.load(fb)
            if fb_bloch.shape == (len(X_enc), 3 * n_q):
                log.info("  [REUSE] Loaded SRQFM Bloch from %s", fb)
                bloch = fb_bloch
            else:
                bloch = _extract_srqfm_bloch(X_enc, n_q, bloch_cache)
        else:
            bloch = _extract_srqfm_bloch(X_enc, n_q, bloch_cache)

        # Save a local copy for reproducibility.
        if not os.path.exists(bloch_cache):
            os.makedirs(os.path.dirname(bloch_cache), exist_ok=True)
            np.save(bloch_cache, bloch)
    else:
        raise ValueError(f"Unknown PQK method: {method}")

    # Build RBF gram from Bloch vectors.
    log.info("  Building RBF gram from Bloch vectors (gamma=%.3f)...", PQK_GAMMA)
    K = compute_bscm_pqk_gram(bloch, n_qubits=n_q, gamma=PQK_GAMMA)

    # Validate gram matrix properties.
    diag_err = float(np.max(np.abs(np.diag(K) - 1.0)))
    asym = float(np.max(np.abs(K - K.T)))
    if diag_err > 1e-8 or asym > 1e-8:
        log.warning(
            "  %s gram quality: diag_err=%.2e asym=%.2e", desc, diag_err, asym,
        )

    os.makedirs(os.path.dirname(gram_cache), exist_ok=True)
    np.save(gram_cache, K)
    log.info("  Saved %s gram to %s  shape=%s", desc, gram_cache, K.shape)
    return K


# ============================================================================
# Per-condition evaluation
# ============================================================================

def _evaluate_condition(
    dataset: str,
    fset: str,
    X_enc: np.ndarray,
    X_raw: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    rows: List[dict],
) -> None:
    """Evaluate all methods on one (dataset, feature_set) condition."""
    cond = f"{dataset}/{fset}"
    log.info("\n" + "=" * 72)
    log.info("  Condition: %s   N=%d  classes=%d", cond, len(y), len(np.unique(y)))
    log.info("  Features: %s", feature_names)
    log.info("=" * 72)

    # -- Build PQK gram matrices ----------------------------------------------
    pqk_methods = ["SG-BSCM-PQK", "BSCM-PQK", "SRQFM-PQK"]
    kernels: Dict[str, np.ndarray] = {}
    for method in pqk_methods:
        t0 = time.time()
        kernels[method] = _build_pqk_kernel(X_enc, dataset, fset, method)
        log.info("  %-15s built  (%.1fs)", method, time.time() - t0)

    # -- Spectral diagnostics -------------------------------------------------
    for method, K in kernels.items():
        sm = spectral_metrics(K)
        kta = _kta_class_centred(K, y)
        log.info(
            "  %-15s off_mean=%.4f  eff_rank=%.1f  KTA(cc)=%.4f",
            method, sm["off_diag_mean"], sm["eff_rank_shannon"], kta,
        )

    # -- Seed-level evaluation ------------------------------------------------
    for seed in SEEDS:
        if dataset == "so2sat":
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=SO2SAT_TEST_FRAC, random_state=seed,
            )
            (tr, te), = sss.split(np.zeros(len(y)), y)
        else:
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=EUROSAT_N_TRAIN,
                test_size=EUROSAT_N_TEST, random_state=seed,
            )
            (tr, te), = sss.split(np.zeros(len(y)), y)

        log.info("\n  Seed %d (%s):", seed, cond)

        # PQK kernels
        for method, K in kernels.items():
            r = _eval_kernel(K, y, tr, te, seed)
            rows.append({
                "dataset": dataset, "feature_set": fset, "method": method,
                "seed": seed, "macro_f1": r["macro_f1"], "best_C": r["best_C"],
            })
            log.info("    %-15s F1=%.4f  C=%s", method, r["macro_f1"], r["best_C"])

        # Classical baselines
        r_rbf = _eval_rbf(X_raw, y, tr, te)
        rows.append({
            "dataset": dataset, "feature_set": fset, "method": "RBF-SVM",
            "seed": seed, "macro_f1": r_rbf["macro_f1"],
            "best_C": r_rbf["best_C"], "best_gamma": r_rbf.get("best_gamma"),
        })
        log.info(
            "    %-15s F1=%.4f  C=%s  gamma=%s",
            "RBF-SVM", r_rbf["macro_f1"], r_rbf["best_C"],
            r_rbf.get("best_gamma"),
        )

        r_rf = _eval_rf(X_raw, y, tr, te, seed)
        rows.append({
            "dataset": dataset, "feature_set": fset, "method": "RandomForest",
            "seed": seed, "macro_f1": r_rf["macro_f1"],
        })
        log.info("    %-15s F1=%.4f", "RandomForest", r_rf["macro_f1"])


# ============================================================================
# Statistical tests
# ============================================================================

def _stat_tests(
    df: pd.DataFrame,
    dataset: str,
    fset: str,
    primary: str = "SG-BSCM-PQK",
) -> dict:
    """Wilcoxon signed-rank + Holm-Bonferroni for one condition."""
    sub = df[(df.dataset == dataset) & (df.feature_set == fset)]
    methods = sorted(sub["method"].unique())
    f1_by = {m: sub[sub.method == m]["macro_f1"].values for m in methods}

    log.info("\n  --- Stats: %s/%s, primary=%s ---", dataset, fset, primary)
    for m in methods:
        v = f1_by[m]
        log.info("    %-15s mean=%.4f  std=%.4f  n=%d", m, v.mean(), v.std(), len(v))

    primary_f1 = f1_by.get(primary, np.array([]))
    if len(primary_f1) == 0:
        log.warning("    Primary method '%s' not found, skipping.", primary)
        return {}

    comparisons = [m for m in methods if m != primary]
    pvals, ds = [], []
    for m in comparisons:
        other = f1_by[m]
        n = min(len(primary_f1), len(other))
        try:
            _, p = wilcoxon(primary_f1[:n], other[:n], alternative="two-sided")
        except ValueError:
            p = 1.0
        d = cohens_d(primary_f1[:n], other[:n])
        pvals.append(p)
        ds.append(d)

    sig = holm_bonferroni(pvals)
    wilcox_results = []
    for m, p, d, s in zip(comparisons, pvals, ds, sig):
        log.info("    %-15s  p=%.4f  d=%.3f  %s", m, p, d, "*" if s else "")
        wilcox_results.append({
            "A": primary, "B": m,
            "mean_A": float(primary_f1.mean()),
            "mean_B": float(f1_by[m].mean()),
            "delta": float(primary_f1.mean() - f1_by[m].mean()),
            "p_raw": float(p), "cohens_d": float(d),
            "reject_holm_05": bool(s),
        })

    return {
        "condition": f"{dataset}/{fset}",
        "method_summary": {
            m: {
                "f1_mean": float(v.mean()),
                "f1_std": float(v.std()),
                "f1_values": [round(float(x), 4) for x in v],
            }
            for m, v in f1_by.items()
        },
        "wilcoxon": wilcox_results,
    }


# ============================================================================
# Main
# ============================================================================

def run() -> pd.DataFrame:
    """Execute the full E36 SG-BSCM-PQK experiment."""
    t_wall = time.time()

    log.info("=" * 72)
    log.info("  E36: SG-BSCM Projected Quantum Kernel (PQK) Evaluation")
    log.info("=" * 72)
    log.info("  Date:       %s", datetime.now().isoformat())
    log.info("  Seeds:      %s", SEEDS)
    log.info("  TAU_LOCKED: %.3f", TAU_LOCKED)
    log.info("  PQK_GAMMA:  %.3f", PQK_GAMMA)
    log.info("  N_QUBITS:   %d", N_QUBITS)
    log.info("  REPS:       %d", REPS)
    log.info("  C_GRID:     %s", C_GRID)

    rows: List[dict] = []

    # -- So2Sat conditions ----------------------------------------------------
    for fset in ["physics8", "pca8"]:
        X_enc, X_raw, y, names = _load_so2sat(fset)
        _evaluate_condition("so2sat", fset, X_enc, X_raw, y, names, rows)

    # -- EuroSAT conditions ---------------------------------------------------
    for fset in ["physics8", "pca8"]:
        X_enc, X_raw, y, names = _load_eurosat(fset)
        _evaluate_condition("eurosat", fset, X_enc, X_raw, y, names, rows)

    # -- Save raw metrics -----------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = os.path.join(SAVE_DIR, "metrics.csv")
    df.to_csv(csv_path, index=False)
    log.info("\n  Saved metrics: %s  (%d rows)", csv_path, len(df))

    # -- Statistical tests ----------------------------------------------------
    summary: Dict[str, dict] = {"experiment": "E36_SG-BSCM-PQK", "conditions": {}}
    for dataset in ["so2sat", "eurosat"]:
        for fset in ["physics8", "pca8"]:
            key = f"{dataset}/{fset}"
            summary["conditions"][key] = _stat_tests(df, dataset, fset)

    summary["pqk_gamma"] = PQK_GAMMA
    summary["tau"] = TAU_LOCKED
    summary["wall_clock_s"] = round(time.time() - t_wall, 1)

    json_path = os.path.join(SAVE_DIR, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("  Saved summary: %s", json_path)

    log.info("\n  E36 complete.  Wall clock: %.1f min", (time.time() - t_wall) / 60)
    return df


if __name__ == "__main__":
    run()
