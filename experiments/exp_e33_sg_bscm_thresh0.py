"""
E33 -- Singlet-Gated BSCM (SG-BSCM) Fidelity Kernel Evaluation.

Motivation
----------
Standard BSCM couples via XX+YY Pauli generators whose coefficients
(½ cos x_i cos x_j, ½ sin x_i sin x_j) are non-zero even when x_i = x_j.
On feature sets with high inter-feature correlation (e.g. EuroSAT optical
physics indices with 11/28 pairs at |r| > 0.85), the 112 entangling gates
(28 pairs × 2 gates × 2 reps, all-to-all) drive the quantum state to a
data-independent fixed point, collapsing fidelity K(x,x') → 0 for all
x ≠ x' and producing a near-identity kernel (off_diag_mean = 0.009).

SG-BSCM multiplies each pair's Pauli coefficients by the singlet fraction
w_{Ψ−} = sin²((x_i − x_j)/2), which is exactly zero when x_i = x_j and
maximal when features differ by π.  This:
  1. Suppresses entanglement between redundant (correlated) feature pairs.
  2. Preserves the XX+YY Pauli structure (orthogonal to SRQFM's ZZ).
  3. Requires no hyperparameter and no classical preprocessing.

Protocol
--------
Datasets:
    So2Sat LCZ42  -- physics8 (Fisher-8) + pca8, N_eval=1800, 17 classes
    EuroSAT       -- physics8 (8 optical indices) + pca8, N_pool=1500, 10 classes

Methods per condition:
    SG-BSCM        singlet_gated=True,  all-to-all, tau=0.25
    BSCM-uniform   singlet_gated=False, all-to-all, tau=0.25
    SRQFM-fid      fidelity kernel (linear connectivity for EuroSAT,
                    all-to-all for So2Sat, matching published references)
    RBF-SVM        GridSearchCV over C × gamma
    RandomForest   n_estimators=500

Evaluation:
    5 seeds [42..46], stratified split per seed.
    So2Sat:  70/30 stratified shuffle, C ∈ {1,10,100} 3-fold inner CV.
    EuroSAT: fixed 1500-sample pool, 1000/500 split, same C grid.
    Macro-F1, spectral metrics, KTA (class-centred), Wilcoxon + Holm,
    Cohen's d.

Output:
    results/sg_bscm/exp_e33.log
    results/sg_bscm/metrics.csv
    results/sg_bscm/summary.json
    results/sg_bscm/K_*.npy  (cached gram matrices)

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
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._bscm_split import make_or_load_split
from experiments._e_common import cohens_d, holm_bonferroni, load_physics_16, spectral_metrics
from src.attention_kernel import FEATURE_NAMES_16, select_features_by_fisher
from src.bscm_kernel import BELL_WEIGHT_PRESETS, compute_bscm_fidelity_kernel
from src.srqfm_fidelity_kernel import compute_srqfm_fidelity_kernel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS          = config.SEED_LIST            # [42, 43, 44, 45, 46]
N_QUBITS       = config.N_QUBITS             # 8
REPS           = config.ZZ_REPS              # 2
TAU_LOCKED     = 0.25                        # from So2Sat Phase-2 holdout
C_GRID         = [1, 10, 100]
CV_FOLDS       = 3
SO2SAT_TEST_FRAC = 0.30
EUROSAT_N_TRAIN  = 1000
EUROSAT_N_TEST   = 500
EUROSAT_POOL_SEED = 0                        # matches bscm_eurosat_fixed_pool

SAVE_DIR = os.path.join(config.RESULTS_DIR, "sg_bscm_thresh0")
os.makedirs(SAVE_DIR, exist_ok=True)
# THRESHOLD-ZERO ABLATION: every per-pair gate is emitted regardless of
# coefficient magnitude. Used to verify that SG-BSCM's rescue is driven
# by the singlet weight w_{Psi-} multiplier, not by gate-dropping at the
# default coupling_threshold=1e-4.
COUPLING_THRESHOLD_ABLATION = 0.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "exp_e33_thresh0.log"), mode="w"),
    ],
)
log = logging.getLogger("exp_e33")


# ============================================================================
# SVM helpers (mirrors E32 byte-for-byte)
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
    """Evaluate RBF-SVM with GridSearchCV over C × gamma."""
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
# KTA (class-centred)
# ============================================================================

def _kta_class_centred(K: np.ndarray, y: np.ndarray) -> float:
    """Kernel-Target Alignment with class-centred ideal kernel."""
    n = len(y)
    Y = (y[:, None] == y[None, :]).astype(np.float64)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Yc = H @ Y @ H
    num = np.sum(Kc * Yc)
    denom = np.sqrt(np.sum(Kc * Kc) * np.sum(Yc * Yc))
    return float(num / denom) if denom > 1e-12 else 0.0


# ============================================================================
# Feature set loaders
# ============================================================================

def _load_so2sat(fset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load So2Sat feature set restricted to the EVAL pool.

    Returns (X_enc, X_raw, y, feature_names).
    X_enc is in [0, pi] for quantum encoding.
    X_raw is the unscaled representation for classical baselines.
    """
    if fset == "physics8":
        X_norm, X_raw_all, y_all = load_physics_16()
        sel_idx, _ = select_features_by_fisher(X_raw_all, y_all)
        X_enc = X_norm[:, sel_idx]
        X_raw = X_raw_all[:, sel_idx]
        names = [FEATURE_NAMES_16[i] for i in sel_idx]
    elif fset == "pca8":
        p = os.path.join(config.PROCESSED_DIR, "subsample_2000.npz")
        d = np.load(p)
        X_enc = d["fused_X_train"]
        X_raw = X_enc / np.pi  # inverse-map to [0,1] for classical
        y_all = d["y_train"]
        names = [f"pc{i+1}" for i in range(8)]
    else:
        raise ValueError(f"Unknown So2Sat feature set: {fset}")

    split = make_or_load_split(y_all)
    return X_enc[split.eval_pool], X_raw[split.eval_pool], y_all[split.eval_pool], names


def _load_eurosat(fset: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load EuroSAT feature set from the fixed pool (N=1500).

    Returns (X_enc, X_raw, y, feature_names).
    """
    # Lazy import to avoid circular dependency
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from eurosat_data import (
        PHYSICS_FEATURE_NAMES,
        compute_physics_indices,
        load_eurosat_allbands,
    )

    bm, labels, _ = load_eurosat_allbands(
        cache_path=os.path.join(config.PROCESSED_DIR, "eurosat_band_means.npz"),
    )

    # Deterministic fixed pool
    pool_sss = StratifiedShuffleSplit(
        n_splits=1, train_size=EUROSAT_N_TRAIN, test_size=EUROSAT_N_TEST,
        random_state=EUROSAT_POOL_SEED,
    )
    tr, te = next(pool_sss.split(bm, labels))
    pool_idx = np.concatenate([tr, te])
    bm_pool = bm[pool_idx]
    y = labels[pool_idx]

    # Feature-fit indices: first N_TRAIN samples of pool (canonical train portion)
    fit_idx = np.arange(EUROSAT_N_TRAIN)

    if fset == "physics8":
        from sklearn.decomposition import PCA as _PCA  # unused here, but keep consistent

        phys_raw = compute_physics_indices(bm_pool)
        scaler = StandardScaler()
        scaler.fit(phys_raw[fit_idx])
        phys_scaled = scaler.transform(phys_raw)
        mm = MinMaxScaler(feature_range=(0, np.pi))
        mm.fit(phys_scaled[fit_idx])
        X_enc = mm.transform(phys_scaled).astype(np.float32)
        X_raw = phys_raw
        names = list(PHYSICS_FEATURE_NAMES)
    elif fset == "pca8":
        from sklearn.decomposition import PCA

        scaler = StandardScaler()
        bm_scaled = scaler.fit_transform(bm_pool[fit_idx])
        pca = PCA(n_components=8, random_state=config.RANDOM_SEED)
        pca.fit(bm_scaled)
        bm_all_scaled = scaler.transform(bm_pool)
        X_pca = pca.transform(bm_all_scaled)
        mm = MinMaxScaler(feature_range=(0, np.pi))
        mm.fit(X_pca[fit_idx])
        X_enc = mm.transform(X_pca).astype(np.float32)
        X_raw = X_pca
        names = [f"pc{i+1}" for i in range(8)]
    else:
        raise ValueError(f"Unknown EuroSAT feature set: {fset}")

    return X_enc, X_raw, y, names


# ============================================================================
# Kernel builders
# ============================================================================

def _build_kernel(
    X_enc: np.ndarray,
    dataset: str,
    fset: str,
    method: str,
) -> np.ndarray:
    """Build and cache a gram matrix for the given method.

    For BSCM-uniform and SRQFM-fid, checks previously-computed kernels
    from E32 (So2Sat) and the EuroSAT fixed-pool experiment before
    recomputing.  If a match is found it is copied into E33's cache
    directory so the provenance trail is self-contained.
    """
    cache = os.path.join(SAVE_DIR, f"K_{method}_{dataset}_{fset}.npy")
    desc = f"{method}({dataset}/{fset})"

    # -- Check E33's own cache first -----------------------------------------
    if os.path.exists(cache):
        log.info("  [CACHE] Loading %s from %s", desc, cache)
        return np.load(cache)

    # -- Fallback DISABLED for threshold-zero ablation -----------------------
    # The default fallback would reuse kernels built with coupling_threshold=1e-4,
    # which would defeat the purpose of this ablation.  We always recompute
    # fresh kernels here under coupling_threshold=COUPLING_THRESHOLD_ABLATION
    # (=0.0), emitting every per-pair gate regardless of coefficient magnitude.
    log.info("  [ABLATION] coupling_threshold=%.0e for %s",
             COUPLING_THRESHOLD_ABLATION, desc)

    # -- Compute fresh -------------------------------------------------------
    if method == "SG-BSCM":
        return compute_bscm_fidelity_kernel(
            X_enc, n_qubits=N_QUBITS, reps=REPS, tau=TAU_LOCKED,
            coupling_threshold=COUPLING_THRESHOLD_ABLATION, connectivity="all",
            bell_weights=BELL_WEIGHT_PRESETS["uniform"],
            kernel_save_path=cache, desc=desc,
            singlet_gated=True,
        )
    elif method == "BSCM-uniform":
        return compute_bscm_fidelity_kernel(
            X_enc, n_qubits=N_QUBITS, reps=REPS, tau=TAU_LOCKED,
            coupling_threshold=COUPLING_THRESHOLD_ABLATION, connectivity="all",
            bell_weights=BELL_WEIGHT_PRESETS["uniform"],
            kernel_save_path=cache, desc=desc,
            singlet_gated=False,
        )
    elif method == "SRQFM-fid":
        conn = "linear" if dataset == "eurosat" else "all"
        return compute_srqfm_fidelity_kernel(
            X_enc, n_qubits=N_QUBITS, reps=REPS,
            coupling_threshold=COUPLING_THRESHOLD_ABLATION,
            connectivity=conn,
            kernel_save_path=cache, desc=desc,
        )
    else:
        raise ValueError(f"Unknown kernel method: {method}")


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
    cond_tag = f"{dataset}/{fset}"
    log.info("\n" + "=" * 72)
    log.info("  Condition: %s   N=%d  n_classes=%d", cond_tag, len(y), int(y.max()) + 1)
    log.info("  Features: %s", feature_names)
    log.info("=" * 72)

    # -- Build kernels -------------------------------------------------------
    kernel_methods = ["SG-BSCM", "BSCM-uniform", "SRQFM-fid"]
    kernels: Dict[str, np.ndarray] = {}
    for method in kernel_methods:
        t0 = time.time()
        kernels[method] = _build_kernel(X_enc, dataset, fset, method)
        log.info("  %-15s built  (%.1fs)", method, time.time() - t0)

    # -- Spectral diagnostics ------------------------------------------------
    for method, K in kernels.items():
        sm = spectral_metrics(K)
        kta = _kta_class_centred(K, y)
        log.info(
            "  %-15s off_mean=%.4f  eff_rank=%.1f  KTA(cc)=%.4f",
            method, sm["off_diag_mean"], sm["eff_rank_shannon"], kta,
        )

    # -- Seed-level evaluation -----------------------------------------------
    for seed in SEEDS:
        # Determine split
        if dataset == "so2sat":
            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=SO2SAT_TEST_FRAC, random_state=seed,
            )
            (tr, te), = sss.split(np.zeros(len(y)), y)
        else:
            # EuroSAT fixed-pool: 1000/500 within-pool split
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=EUROSAT_N_TRAIN,
                test_size=EUROSAT_N_TEST, random_state=seed,
            )
            (tr, te), = sss.split(np.zeros(len(y)), y)

        log.info("\n  Seed %d (%s):", seed, cond_tag)

        # Quantum kernels
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
        log.info("    %-15s F1=%.4f  C=%s  gamma=%s",
                 "RBF-SVM", r_rbf["macro_f1"], r_rbf["best_C"],
                 r_rbf.get("best_gamma"))

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
    primary: str = "SG-BSCM",
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
    log.info("    %-15s %10s %8s %6s", "Method", "p-value", "Cohen-d", "sig*")
    wilcox_results = []
    for m, p, d, s in zip(comparisons, pvals, ds, sig):
        log.info("    %-15s %10.4f %8.3f %6s", m, p, d, "*" if s else "")
        wilcox_results.append({
            "A": primary, "B": m,
            "mean_A": float(primary_f1.mean()), "mean_B": float(f1_by[m].mean()),
            "delta": float(primary_f1.mean() - f1_by[m].mean()),
            "p_raw": float(p), "cohens_d": float(d),
            "reject_holm_05": bool(s),
        })

    return {
        "condition": f"{dataset}/{fset}",
        "method_summary": {
            m: {"f1_mean": float(v.mean()), "f1_std": float(v.std()),
                "f1_values": [round(float(x), 4) for x in v]}
            for m, v in f1_by.items()
        },
        "wilcoxon": wilcox_results,
    }


# ============================================================================
# Main
# ============================================================================

def run() -> pd.DataFrame:
    """Execute the full E33 SG-BSCM experiment."""
    t_wall = time.time()

    log.info("=" * 72)
    log.info("  E33: Singlet-Gated BSCM (SG-BSCM) Fidelity Kernel Evaluation")
    log.info("=" * 72)
    log.info("  Date:       %s", datetime.now().isoformat())
    log.info("  Seeds:      %s", SEEDS)
    log.info("  TAU_LOCKED: %.3f", TAU_LOCKED)
    log.info("  N_QUBITS:   %d", N_QUBITS)
    log.info("  REPS:       %d", REPS)
    log.info("  C_GRID:     %s", C_GRID)

    rows: List[dict] = []

    # -- ABLATION SCOPE: only EuroSAT physics-8 ------------------------------
    # The threshold ablation matters only where BSCM-uniform collapses, i.e.
    # EuroSAT physics-8 (the case demonstrated in the main paper).
    # Running the full 4-condition sweep would take ~10 h with no additional
    # mechanistic value. To extend the ablation to all 4 conditions, set
    # ABLATION_FULL_SWEEP=True below.
    ABLATION_FULL_SWEEP = False

    if ABLATION_FULL_SWEEP:
        for fset in ["physics8", "pca8"]:
            X_enc, X_raw, y, names = _load_so2sat(fset)
            _evaluate_condition("so2sat", fset, X_enc, X_raw, y, names, rows)
        for fset in ["physics8", "pca8"]:
            X_enc, X_raw, y, names = _load_eurosat(fset)
            _evaluate_condition("eurosat", fset, X_enc, X_raw, y, names, rows)
    else:
        X_enc, X_raw, y, names = _load_eurosat("physics8")
        _evaluate_condition("eurosat", "physics8", X_enc, X_raw, y, names, rows)

    # -- Save raw metrics ----------------------------------------------------
    df = pd.DataFrame(rows)
    csv_path = os.path.join(SAVE_DIR, "metrics.csv")
    df.to_csv(csv_path, index=False)
    log.info("\n  Saved metrics: %s", csv_path)

    # -- Statistical tests ---------------------------------------------------
    summary: Dict[str, dict] = {"experiment": "E33_SG-BSCM", "conditions": {}}
    for dataset in ["so2sat", "eurosat"]:
        for fset in ["physics8", "pca8"]:
            key = f"{dataset}/{fset}"
            summary["conditions"][key] = _stat_tests(df, dataset, fset)

    summary["wall_clock_s"] = round(time.time() - t_wall, 1)

    json_path = os.path.join(SAVE_DIR, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("  Saved summary: %s", json_path)

    log.info("\n  E33 complete.  Wall clock: %.1f min", (time.time() - t_wall) / 60)
    return df


if __name__ == "__main__":
    run()
