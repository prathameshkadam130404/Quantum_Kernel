#!/usr/bin/env python3
"""
EuroSAT BSCM Fixed-Pool Evaluation
====================================

Computes BSCM-uniform and SRQFM-fidelity kernels ONCE on a fixed stratified
pool of N_POOL=1500 EuroSAT samples.  Five evaluation seeds obtain different
n_train=1000 / n_test=500 splits within the same pool by slicing the pre-
computed gram matrix — quantum circuit evaluation occurs once per
(kernel, feature_set) rather than once per (kernel, feature_set, seed).

Speedup vs. per-seed computation
    Per-seed:    5 × (N_train² + N_test × N_train)  =  7.50 M circuits
    Fixed-pool:  N_pool × (N_pool + 1) / 2          =  1.13 M circuits  (~6.7×)

Protocol
--------
  Pool selection   : StratifiedShuffleSplit(n_train=1000, n_test=500,
                     random_state=POOL_SEED=0) on full EuroSAT (~27 k samples).
  Feature fitting  : StandardScaler → PCA (pca8) and StandardScaler →
                     MinMaxScaler[0,π] (physics8) fitted on the canonical
                     training portion (indices 0..999 within pool); applied
                     uniformly to all 1500 pool samples.
  Kernel caching   : Gram matrices saved to .npy; existing files are loaded
                     without recomputation.  Row-level progress file enables
                     crash-safe resumption.
  Eval splits      : StratifiedShuffleSplit(n_train=1000, n_test=500,
                     random_state=seed) within pool for seeds {42,43,44,45,46}.
  SVM tuning       : C ∈ {0.1, 1, 10, 100}, 3-fold StratifiedKFold inner CV on
                     K_tr.  No test-set leakage — final test K_te is touched
                     only once at the chosen best_C.  Note: project's earlier
                     EuroSAT scripts selected C by argmax test-F1; this script
                     uses the corrected protocol (documented in paper).
  KTA              : Class-frequency-normalised "class-centered" KTA matching
                     the So2Sat publication-grade analysis pack
                     (scripts/bscm_publication_analysis.py) for two-dataset
                     comparability.
  Statistics       : Macro-F1, balanced accuracy, per-class F1, KTA
                     (product + class-centered), spectral metrics
                     (off_diag_mean, off_diag_var, within-1σ,
                     eff_rank_shannon, eff_rank_huang), McNemar per seed
                     (vs RBF-SVM and vs RF), Wilcoxon + Holm-Bonferroni on 4
                     planned pairs, permutation-KTA p-value (n_perm=1000) on
                     full pool, pooled stratified bootstrap (B=1000) of
                     per-class F1 deltas across all 5 seeds (n_eval=2500).

Outputs
-------
  results/eurosat_bscm_fixed/
      kernels/K_pool_bscm_pca8.npy          (1500 × 1500, float64)
      kernels/K_pool_bscm_physics8.npy
      kernels/K_pool_srqfm_linear_pca8.npy
      kernels/K_pool_srqfm_linear_physics8.npy
      eurosat_bscm_fixed_results.json
      run.log

Usage
-----
  python scripts/bscm_eurosat_fixed_pool.py
  python scripts/bscm_eurosat_fixed_pool.py --skip-srqfm
  python scripts/bscm_eurosat_fixed_pool.py --pca-only
  python scripts/bscm_eurosat_fixed_pool.py --physics-only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2, wilcoxon
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
from eurosat_data import (  # noqa: E402
    EUROSAT_CLASSES,
    PHYSICS_FEATURE_NAMES,
    compute_physics_indices,
    load_eurosat_allbands,
)
from src.bscm_kernel import (  # noqa: E402
    BELL_WEIGHT_PRESETS,
    apply_bscm_feature_map,
    apply_bscm_feature_map_adjoint,
)
from src.srqfm_fidelity_kernel import _apply_srqfm, _apply_srqfm_adjoint  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ── Experiment constants ───────────────────────────────────────────────────────
POOL_SEED    = 0                        # seed for pool selection only
N_TRAIN      = 1000
N_TEST       = 500
N_POOL       = N_TRAIN + N_TEST         # 1500
EVAL_SEEDS   = [42, 43, 44, 45, 46]
C_GRID       = [0.1, 1.0, 10.0, 100.0]
ZZ_REPS      = config.ZZ_REPS          # 2
BSCM_CONNECTIVITY  = "all"              # BSCM locked-tau tuning used all-to-all
SRQFM_CONNECTIVITY = "linear"           # Matches original run_eurosat_experiment.py
                                        # SRQFM (CNOT–RZ(c)–CNOT chain on i,i+1).
                                        # Required for apples-to-apples comparison
                                        # with published EuroSAT SRQFM reference
                                        # numbers (pca8 0.7816, physics8 0.6933).
                                        # All-to-all causes severe kernel concen-
                                        # tration on n_q=8 → low F1.
BSCM_COUP_THRESH   = 1e-4               # BSCM threshold (matches phase2 lockdown)
SRQFM_COUP_THRESH  = 0.0                # No threshold gating in original SRQFM
PCA_DIM      = 8

RESULTS_DIR     = Path("results/eurosat_bscm_fixed")
KERNEL_DIR      = RESULTS_DIR / "kernels"
LOCKED_TAU_PATH = Path("results/bscm/locked_tau.json")

log = logging.getLogger(__name__)


# ── Tau loading ────────────────────────────────────────────────────────────────

def load_locked_tau() -> float:
    if not LOCKED_TAU_PATH.exists():
        raise FileNotFoundError(
            f"locked_tau.json not found at {LOCKED_TAU_PATH}. "
            "Run scripts/bscm_phase2_lockdown.py first."
        )
    return float(json.loads(LOCKED_TAU_PATH.read_text())["locked"]["tau"])


# ── Feature preparation ────────────────────────────────────────────────────────

def prepare_pool_features(
    band_means_pool: np.ndarray,
    canonical_train_local: np.ndarray,
) -> Dict[str, object]:
    """
    Compute PCA-8 and Physics-8 features for all N_POOL samples.

    Scaler and PCA are fitted on band_means_pool[canonical_train_local] only,
    then applied to all N_POOL samples.  The canonical training portion is the
    first N_TRAIN pool samples (indices 0..N_TRAIN-1).
    """
    bm_tr  = band_means_pool[canonical_train_local]
    bm_all = band_means_pool

    # PCA path: StandardScaler → PCA8 → MinMaxScaler[0, π]
    ss_pca = StandardScaler().fit(bm_tr)
    pca    = PCA(n_components=PCA_DIM, random_state=POOL_SEED).fit(
                 ss_pca.transform(bm_tr))
    mm_pca = MinMaxScaler(feature_range=(0, np.pi)).fit(
                 pca.transform(ss_pca.transform(bm_tr)))
    X_pca  = mm_pca.transform(
                 pca.transform(ss_pca.transform(bm_all))
             ).astype(np.float32)

    # Physics path: indices → StandardScaler → MinMaxScaler[0, π]
    phys_tr = compute_physics_indices(bm_tr)
    ss_ph   = StandardScaler().fit(phys_tr)
    mm_ph   = MinMaxScaler(feature_range=(0, np.pi)).fit(
                  ss_ph.transform(phys_tr))
    X_phys  = mm_ph.transform(
                  ss_ph.transform(compute_physics_indices(bm_all))
              ).astype(np.float32)

    return {
        "pca8":             X_pca,
        "physics8":         X_phys,
        "pca_explained_var": float(pca.explained_variance_ratio_.sum()),
    }


# ── PennyLane circuit builders ─────────────────────────────────────────────────

def _get_device(n_q: int):
    import pennylane as qml
    for name in ("lightning.qubit", "default.qubit"):
        try:
            dev = qml.device(name, wires=n_q)
            @qml.qnode(dev)
            def _smoke():
                return qml.probs(wires=range(n_q))
            _smoke()
            log.info("  Using PennyLane device: %s (%d wires)", name, n_q)
            return dev
        except Exception:
            continue
    raise RuntimeError("No PennyLane device available.")


def make_bscm_circuit(n_q: int, tau: float, prior: str = "uniform"):
    import pennylane as qml
    dev = _get_device(n_q)
    w   = BELL_WEIGHT_PRESETS[prior]

    @qml.qnode(dev, diff_method=None)
    def circuit(x1, x2):
        apply_bscm_feature_map(
            x1, n_qubits=n_q, reps=ZZ_REPS, tau=tau,
            coupling_threshold=BSCM_COUP_THRESH, connectivity=BSCM_CONNECTIVITY,
            bell_weights=w,
        )
        apply_bscm_feature_map_adjoint(
            x2, n_qubits=n_q, reps=ZZ_REPS, tau=tau,
            coupling_threshold=BSCM_COUP_THRESH, connectivity=BSCM_CONNECTIVITY,
            bell_weights=w,
        )
        return qml.probs(wires=range(n_q))

    return circuit


def make_srqfm_circuit(n_q: int):
    """
    SRQFM fidelity circuit configured to MATCH the original
    scripts/run_eurosat_experiment.py SRQFM (linear nearest-neighbour
    connectivity, no threshold gating).  Using all-to-all here drives the
    8-qubit SRQFM kernel into severe concentration and collapses F1 — that
    was the discrepancy versus the published reference numbers.
    """
    import pennylane as qml
    dev = _get_device(n_q)

    @qml.qnode(dev, diff_method=None)
    def circuit(x1, x2):
        _apply_srqfm(x1, n_q, ZZ_REPS, SRQFM_COUP_THRESH, SRQFM_CONNECTIVITY)
        _apply_srqfm_adjoint(x2, n_q, ZZ_REPS, SRQFM_COUP_THRESH, SRQFM_CONNECTIVITY)
        return qml.probs(wires=range(n_q))

    return circuit


# ── Gram matrix computation ────────────────────────────────────────────────────

def compute_gram_pool(
    circuit,
    X_pool: np.ndarray,
    cache_path: Path,
    label: str = "kernel",
) -> np.ndarray:
    """
    Compute or load the (N_POOL × N_POOL) gram matrix.

    Exploits symmetry K[i,j] = K[j,i] and K[i,i] = 1 (fidelity kernel),
    halving circuit evaluations vs naïve double-loop.

    Crash-safe: saves K to cache_path after every row; a companion
    .progress file stores the last completed row index so a restarted run
    resumes from where it left off.  Final .progress file is deleted on
    successful completion.
    """
    N = len(X_pool)
    progress_path = cache_path.with_suffix(".progress")

    if cache_path.exists():
        K = np.load(cache_path)
        if K.shape == (N, N):
            log.info("  Loaded cached gram (%d×%d) from %s", N, N, cache_path)
            return K
        log.warning("  Cached file has wrong shape %s — recomputing.", K.shape)

    start_row = 0
    K = np.zeros((N, N), dtype=np.float64)
    np.fill_diagonal(K, 1.0)  # K(x,x) = 1 for all fidelity kernels

    if progress_path.exists():
        try:
            start_row = int(progress_path.read_text().strip()) + 1
            if cache_path.exists():
                K_part = np.load(cache_path)
                if K_part.shape == (N, N):
                    K[:start_row] = K_part[:start_row]
                    log.info("  Resuming %s from row %d / %d", label, start_row, N)
        except Exception:
            start_row = 0

    n_total = N * (N + 1) // 2
    n_done  = sum(N - i for i in range(start_row))
    pbar    = tqdm(total=n_total, initial=n_done,
                   desc=f"  {label} ({N}×{N})", unit="eval")

    SAVE_EVERY_ROWS = 25  # crash-safe checkpoint every 25 rows (~1.7% of N=1500)
    t0 = time.time()
    for i in range(start_row, N):
        K[i, i] = 1.0
        for j in range(i + 1, N):
            k_val    = float(circuit(X_pool[i], X_pool[j])[0])
            K[i, j]  = k_val
            K[j, i]  = k_val
            pbar.update(1)
        if (i + 1) % SAVE_EVERY_ROWS == 0 or i == N - 1:
            np.save(cache_path, K)
            progress_path.write_text(str(i))

    pbar.close()
    elapsed = time.time() - t0
    log.info("  %s gram (%d×%d) computed in %.0f s (%.1f min)",
             label, N, N, elapsed, elapsed / 60)

    if progress_path.exists():
        progress_path.unlink()

    return K


# ── Classical baselines ────────────────────────────────────────────────────────

def run_rbf_svm(X_tr, y_tr, X_te, y_te) -> dict:
    grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", random_state=42),
        {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1.0]},
        cv=3, scoring="f1_macro", n_jobs=-1, refit=True,
    )
    grid.fit(X_tr, y_tr)
    yp = grid.predict(X_te)
    return {
        "f1_macro":     float(f1_score(y_te, yp, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_te, yp)),
        "best_params":  {k: str(v) for k, v in grid.best_params_.items()},
        "predictions":  yp.tolist(),
    }


def run_rf(X_tr, y_tr, X_te, y_te) -> dict:
    rf = RandomForestClassifier(
        n_estimators=200, min_samples_split=5,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    rf.fit(X_tr, y_tr)
    yp = rf.predict(X_te)
    return {
        "f1_macro":     float(f1_score(y_te, yp, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_te, yp)),
        "predictions":  yp.tolist(),
    }


def run_quantum_svm(K_tr, y_tr, K_te, y_te, cv_seed: int = 0) -> dict:
    """
    Precomputed-kernel SVM with C selected by 3-fold inner CV on K_tr only.

    NO TEST SET LEAKAGE: C is chosen by maximum mean F1 on inner folds of
    the training set; the test set is touched exactly once at the final
    evaluation step.  This differs from the project's earlier EuroSAT
    scripts (run_eurosat_experiment.py, run_eurosat_bscm.py) which
    selected C by argmax test-set F1 — a protocol revision documented in
    the paper.
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=cv_seed)
    fold_idx = list(skf.split(K_tr, y_tr))

    best_C, best_cv_f1 = None, -1.0
    cv_curve = {}
    for C in C_GRID:
        f1s = []
        for tr_in, va_in in fold_idx:
            K_in = K_tr[np.ix_(tr_in, tr_in)]
            K_va = K_tr[np.ix_(va_in, tr_in)]
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_in, y_tr[tr_in])
            yp_va = clf.predict(K_va)
            f1s.append(float(f1_score(y_tr[va_in], yp_va, average="macro")))
        mean_f1 = float(np.mean(f1s))
        cv_curve[float(C)] = mean_f1
        if mean_f1 > best_cv_f1:
            best_cv_f1 = mean_f1
            best_C = float(C)

    final = SVC(kernel="precomputed", C=best_C, class_weight="balanced")
    final.fit(K_tr, y_tr)
    yp = final.predict(K_te)
    return {
        "f1_macro":     float(f1_score(y_te, yp, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_te, yp)),
        "best_C":       best_C,
        "best_cv_f1":   best_cv_f1,
        "cv_curve":     cv_curve,
        "predictions":  yp.tolist(),
    }


# ── Kernel-target alignment ────────────────────────────────────────────────────
#
# Definitions match scripts/bscm_publication_analysis.py (the So2Sat publication
# pack) so that EuroSAT and So2Sat KTA values are directly comparable in the paper.
#
#   Y_product      : same-class indicator,                 Y_ij = 1[y_i == y_j]
#   Y_class_weight : class-frequency-normalised indicator,  Y_ij = 1[y_i == y_j] / n_{y_i}
#                    (rescales each class's contribution so imbalanced classes do
#                    not dominate the alignment — important for EuroSAT pool whose
#                    class counts vary roughly 100..200).
#
# K is centered (Cortes-style) before alignment with Y_class_weight.

def _build_Y_product(y: np.ndarray) -> np.ndarray:
    return (y[:, None] == y[None, :]).astype(np.float64)


def _build_Y_class_weighted(y: np.ndarray) -> np.ndarray:
    cls, cnt = np.unique(y, return_counts=True)
    inv      = np.zeros(int(cls.max()) + 1, dtype=np.float64)
    inv[cls] = 1.0 / cnt
    same     = (y[:, None] == y[None, :]).astype(np.float64)
    return same * inv[y][:, None]


def _center_kernel(K: np.ndarray) -> np.ndarray:
    return K - K.mean(1, keepdims=True) - K.mean(0, keepdims=True) + K.mean()


def kta_product(K: np.ndarray, y: np.ndarray) -> float:
    Y = _build_Y_product(y)
    return float(np.sum(K * Y) / (np.sqrt(np.sum(K * K) * np.sum(Y * Y)) + 1e-12))


def kta_class_centered(K: np.ndarray, y: np.ndarray,
                        Kc: Optional[np.ndarray] = None) -> float:
    """Centered kernel alignment against class-frequency-normalised Y."""
    if Kc is None:
        Kc = _center_kernel(K)
    Y = _build_Y_class_weighted(y)
    return float(
        np.sum(Kc * Y) /
        (np.sqrt(np.sum(Kc * Kc) * np.sum(Y * Y)) + 1e-12)
    )


def perm_kta_pvalue(
    K: np.ndarray, y: np.ndarray,
    n_perm: int = 1000, rng_seed: int = 0,
    sub_n: Optional[int] = None,
) -> dict:
    """
    Permutation test for class-centered KTA significance.

    Reports observed KTA, mean and std of label-permutation null, and
    one-sided p-value = (#{null >= observed} + 1) / (n_perm + 1).
    For tractability on large pools, optionally stratified-subsample to
    sub_n samples (defaults to min(800, N)).
    """
    rng = np.random.default_rng(rng_seed)
    N = K.shape[0]
    if sub_n is not None and sub_n < N:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=sub_n, random_state=rng_seed,
        )
        idx, _ = next(sss.split(np.zeros(N), y))
        K = K[np.ix_(idx, idx)]
        y = y[idx]

    Kc       = _center_kernel(K)
    denom_K  = np.sqrt(np.sum(Kc * Kc))
    obs      = kta_class_centered(K, y, Kc=Kc)

    nulls = np.empty(n_perm, dtype=np.float64)
    for r in range(n_perm):
        y_perm = rng.permutation(y)
        Y_p    = _build_Y_class_weighted(y_perm)
        nulls[r] = float(np.sum(Kc * Y_p) /
                         (denom_K * np.sqrt(np.sum(Y_p * Y_p)) + 1e-12))

    p = float((np.sum(nulls >= obs) + 1) / (n_perm + 1))
    return {
        "kta_cc_observed": obs,
        "null_mean":       float(nulls.mean()),
        "null_std":        float(nulls.std()),
        "p_value":         p,
        "n_perm":          int(n_perm),
        "sub_n":           int(K.shape[0]),
    }


def spectral_metrics(K: np.ndarray) -> dict:
    n    = K.shape[0]
    off  = K[~np.eye(n, dtype=bool)]
    eig  = np.clip(np.linalg.eigvalsh((K + K.T) / 2), 1e-12, None)
    p    = eig / eig.sum()
    σ    = off.std()
    return {
        "off_diag_mean":    float(off.mean()),
        "off_diag_var":     float(off.var()),
        "within_1sigma":    float(np.mean(np.abs(off - off.mean()) <= σ)),
        "eff_rank_shannon": float(np.exp(-np.sum(p * np.log(p + 1e-12)))),
        "eff_rank_huang":   float(eig.sum() / eig.max()),
    }


# ── Per-class F1 + bootstrap ──────────────────────────────────────────────────

def per_class_f1_array(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Returns length-n_classes float vector of F1 per class (0 if class absent)."""
    return f1_score(
        y_true, y_pred,
        labels=list(range(n_classes)),
        average=None, zero_division=0,
    )


def bootstrap_per_class_delta(
    y_true:  np.ndarray,
    y_a:     np.ndarray,
    y_b:     np.ndarray,
    n_classes: int,
    B:       int = 1000,
    rng_seed: int = 0,
) -> dict:
    """
    Stratified bootstrap of per-class F1 delta (method A − method B).

    For each of B bootstrap resamples, draw with replacement (stratified by
    y_true) and recompute per-class F1 for both methods.  Returns the
    point estimate (delta on the original sample), 95 % CI per class,
    and a flag for classes whose CI excludes zero.
    """
    rng = np.random.default_rng(rng_seed)
    n   = len(y_true)

    f1_a = per_class_f1_array(y_true, y_a, n_classes)
    f1_b = per_class_f1_array(y_true, y_b, n_classes)
    delta_point = f1_a - f1_b

    # Stratified resampling preserves class proportions
    by_class = [np.where(y_true == c)[0] for c in range(n_classes)]
    deltas = np.empty((B, n_classes), dtype=np.float64)
    for r in range(B):
        idx = np.concatenate([
            rng.choice(arr, size=len(arr), replace=True)
            for arr in by_class if len(arr) > 0
        ])
        f1a_r = per_class_f1_array(y_true[idx], y_a[idx], n_classes)
        f1b_r = per_class_f1_array(y_true[idx], y_b[idx], n_classes)
        deltas[r] = f1a_r - f1b_r

    lo = np.percentile(deltas, 2.5,  axis=0)
    hi = np.percentile(deltas, 97.5, axis=0)
    return {
        "delta_point": delta_point.tolist(),
        "ci_lo_95":    lo.tolist(),
        "ci_hi_95":    hi.tolist(),
        "ci_excludes_zero": [bool((l > 0) or (h < 0)) for l, h in zip(lo, hi)],
        "B":           int(B),
    }


# ── Statistical tests ──────────────────────────────────────────────────────────

def mcnemar_test(y_true, y_a, y_b) -> dict:
    ca, cb = (y_a == y_true), (y_b == y_true)
    b01 = int(np.sum(ca & ~cb))
    b10 = int(np.sum(~ca & cb))
    if b01 + b10 == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b01": b01, "b10": b10}
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    return {
        "chi2": float(stat),
        "p_value": float(1 - chi2.cdf(stat, df=1)),
        "b01": b01, "b10": b10,
    }


def holm_bonferroni(p_values: List[float]) -> List[float]:
    n   = len(p_values)
    idx = sorted(range(n), key=lambda i: p_values[i])
    adj = [1.0] * n
    running_max = 0.0
    for rank, i in enumerate(idx):
        corrected   = min(1.0, p_values[i] * (n - rank))
        running_max = max(running_max, corrected)
        adj[i]      = running_max
    return adj


def wilcoxon_holm_table(f1_dict: Dict[str, List[float]],
                         planned_pairs: List[Tuple[str, str]]) -> List[dict]:
    raw_ps, rows = [], []
    for (a, b) in planned_pairs:
        aa = np.array(f1_dict.get(a, []))
        bb = np.array(f1_dict.get(b, []))
        if len(aa) < 3 or len(bb) < 3:
            continue
        try:
            _, p = wilcoxon(aa, bb, alternative="two-sided")
        except Exception:
            p = 1.0
        raw_ps.append(p)
        rows.append({
            "A": a, "B": b,
            "mean_A": round(float(aa.mean()), 4),
            "mean_B": round(float(bb.mean()), 4),
            "delta":  round(float(aa.mean() - bb.mean()), 4),
            "p_raw":  float(p),
        })
    p_holm = holm_bonferroni(raw_ps)
    for i, row in enumerate(rows):
        row["p_holm"]         = float(p_holm[i])
        row["reject_holm_05"] = bool(p_holm[i] < 0.05)
    return rows


# ── Per-seed evaluation ────────────────────────────────────────────────────────

def evaluate_feature_set(
    feat_name:    str,
    X_pool:       np.ndarray,
    pool_labels:  np.ndarray,
    gram_matrices: Dict[str, np.ndarray],   # kernel_name → (N_POOL, N_POOL)
    run_srqfm:    bool,
    planned_pairs: Optional[List[Tuple[str, str]]] = None,
    bootstrap_pairs: Optional[List[Tuple[str, str]]] = None,
) -> dict:
    """
    Evaluate all methods on 5 within-pool splits.  Classical baselines (RF,
    RBF-SVM) are run once per split and shared across kernel comparisons.
    Predictions are kept in memory until McNemar tests are complete, then
    stripped before JSON serialisation.
    """
    n_q = X_pool.shape[1]
    n_classes = int(pool_labels.max()) + 1
    method_names = list(gram_matrices) + ["RBF-SVM", "RF"]

    f1_track:   Dict[str, List[float]]      = {m: []  for m in method_names}
    pc_track:   Dict[str, List[np.ndarray]] = {m: []  for m in method_names}  # per-class F1
    pred_pool:  Dict[str, List[np.ndarray]] = {m: []  for m in method_names}  # for pooled bootstrap
    y_te_pool:  List[np.ndarray]            = []
    seed_results: List[dict] = []

    for seed in EVAL_SEEDS:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=N_TRAIN, test_size=N_TEST,
            random_state=seed,
        )
        tr, te = next(sss.split(X_pool, pool_labels))
        y_tr, y_te = pool_labels[tr], pool_labels[te]
        X_tr, X_te = X_pool[tr], X_pool[te]
        y_te_pool.append(y_te)

        sr: dict = {"seed": int(seed)}

        # Classical baselines (fast — ~5 s each)
        sr["RBF-SVM"] = run_rbf_svm(X_tr, y_tr, X_te, y_te)
        sr["RF"]      = run_rf(X_tr, y_tr, X_te, y_te)
        for m in ("RBF-SVM", "RF"):
            yp_arr = np.array(sr[m]["predictions"])
            f1_track[m].append(sr[m]["f1_macro"])
            pc      = per_class_f1_array(y_te, yp_arr, n_classes)
            sr[m]["per_class_f1"] = pc.tolist()
            pc_track[m].append(pc)
            pred_pool[m].append(yp_arr)

        # Quantum kernel SVMs (gram slicing — fast)
        for kname, K_pool in gram_matrices.items():
            K_tr = K_pool[np.ix_(tr, tr)]
            K_te = K_pool[np.ix_(te, tr)]
            sr[kname] = run_quantum_svm(K_tr, y_tr, K_te, y_te, cv_seed=seed)
            sr[kname]["kta_product"] = kta_product(K_tr, y_tr)
            sr[kname]["kta_cc"]      = kta_class_centered(K_tr, y_tr)
            sr[kname]["spectral"]    = spectral_metrics(K_tr)
            yp_arr = np.array(sr[kname]["predictions"])
            f1_track[kname].append(sr[kname]["f1_macro"])
            pc      = per_class_f1_array(y_te, yp_arr, n_classes)
            sr[kname]["per_class_f1"] = pc.tolist()
            pc_track[kname].append(pc)
            pred_pool[kname].append(yp_arr)

        # McNemar: each quantum kernel vs RBF-SVM and vs RF (n=500 predictions)
        y_rbf = np.array(sr["RBF-SVM"]["predictions"])
        y_rf  = np.array(sr["RF"]["predictions"])
        for kname in gram_matrices:
            yq = np.array(sr[kname]["predictions"])
            sr[f"McNemar_{kname}_vs_RBF"] = mcnemar_test(y_te, yq, y_rbf)
            sr[f"McNemar_{kname}_vs_RF"]  = mcnemar_test(y_te, yq, y_rf)

        # Cross-kernel McNemar: BSCM-uniform vs SRQFM-fid (if both present)
        if run_srqfm and "BSCM-uniform" in gram_matrices and "SRQFM-fid" in gram_matrices:
            yb = np.array(sr["BSCM-uniform"]["predictions"])
            ys = np.array(sr["SRQFM-fid"]["predictions"])
            sr["McNemar_BSCM_vs_SRQFM"] = mcnemar_test(y_te, yb, ys)

        # Log all methods' F1 for this seed (generic — works for any kernel names)
        _log_parts = [f"[{feat_name}] seed={seed}"]
        for _m in method_names:
            if _m in sr:
                _log_parts.append(f"{_m}={sr[_m]['f1_macro']:.4f}")
        log.info("  %s", "  ".join(_log_parts))

        # Strip predictions from per-seed entries (kept in pred_pool for bootstrap)
        for m in method_names:
            if m in sr and "predictions" in sr[m]:
                del sr[m]["predictions"]

        seed_results.append(sr)

    # Pooled McNemar (would need predictions — computed per-seed above)
    # Build summary
    if planned_pairs is None:
        planned_pairs = [
            ("BSCM-uniform", "RBF-SVM"),
            ("BSCM-uniform", "RF"),
        ]
        if run_srqfm:
            planned_pairs += [
                ("SRQFM-fid", "RBF-SVM"),
                ("BSCM-uniform", "SRQFM-fid"),
            ]

    wilcox = wilcoxon_holm_table(f1_track, planned_pairs)

    # ── Aggregate macro-F1 + per-class F1 across seeds ─────────────────────────
    method_summary = {}
    for m in method_names:
        vals = f1_track[m]
        if not vals:
            continue
        a = np.array(vals)
        pc = np.stack(pc_track[m]) if pc_track[m] else np.zeros((0, n_classes))
        method_summary[m] = {
            "f1_mean":          round(float(a.mean()), 4),
            "f1_std":           round(float(a.std()), 4),
            "f1_values":        [round(v, 4) for v in a.tolist()],
            "n_seeds":          len(vals),
            "per_class_f1_mean": [round(float(v), 4) for v in pc.mean(axis=0)],
            "per_class_f1_std":  [round(float(v), 4) for v in pc.std(axis=0)],
        }

    # ── Pooled (cross-seed) bootstrap per-class F1 deltas ──────────────────────
    # Stack all 5 seeds' test predictions into one 2500-sample evaluation.
    bootstrap_blocks = {}
    if bootstrap_pairs is None:
        bootstrap_pairs = [
            ("BSCM-uniform", "RBF-SVM"),
            ("BSCM-uniform", "RF"),
        ]
        if run_srqfm:
            bootstrap_pairs += [
                ("SRQFM-fid", "RBF-SVM"),
                ("BSCM-uniform", "SRQFM-fid"),
            ]
    _has_preds = any(
        pred_pool.get(a) and pred_pool.get(b)
        for a, b in bootstrap_pairs
    )
    if _has_preds:
        y_pool_all = np.concatenate(y_te_pool)
        for a, b in bootstrap_pairs:
            if not pred_pool.get(a) or not pred_pool.get(b):
                continue
            ya = np.concatenate(pred_pool[a])
            yb = np.concatenate(pred_pool[b])
            bootstrap_blocks[f"{a}_vs_{b}"] = bootstrap_per_class_delta(
                y_pool_all, ya, yb, n_classes=n_classes, B=1000, rng_seed=0,
            )

    log.info("\n  Summary [%s]:", feat_name)
    for m, s in method_summary.items():
        log.info("    %-15s  F1 = %.4f ± %.4f", m, s["f1_mean"], s["f1_std"])
    log.info("  Wilcoxon (Holm-corrected) — n=5 power floor p_raw=0.0625:")
    for w in wilcox:
        log.info(
            "    %-15s vs %-15s  Δ=%+.4f  p_raw=%.4f  p_holm=%.4f  reject=%s",
            w["A"], w["B"], w["delta"], w["p_raw"], w["p_holm"],
            w["reject_holm_05"],
        )
    if bootstrap_blocks:
        log.info("  Pooled bootstrap per-class CIs (B=1000):")
        for tag, blk in bootstrap_blocks.items():
            n_sig = sum(blk["ci_excludes_zero"])
            log.info(
                "    %s : %d / %d classes with 95%% CI excluding zero",
                tag, n_sig, n_classes,
            )

    return {
        "seed_results":         seed_results,
        "method_summary":       method_summary,
        "wilcoxon_holm":        wilcox,
        "bootstrap_per_class":  bootstrap_blocks,
        "n_pool_test_total":    int(sum(len(y) for y in y_te_pool)),
        "n5_power_note":        (
            "n=5 seeds; two-sided Wilcoxon p_raw floor is 0.0625 (=2/32). "
            "No pair can clear Holm-corrected alpha=0.05 at this seed count. "
            "Per-seed McNemar (n=500) and pooled bootstrap (n=2500) provide "
            "higher-power complementary tests."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EuroSAT BSCM Fixed-Pool Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/raw/eurosat/EuroSATallBands")
    parser.add_argument("--skip-srqfm",   action="store_true",
                        help="Compute BSCM-uniform only")
    parser.add_argument("--pca-only",     action="store_true")
    parser.add_argument("--physics-only", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(RESULTS_DIR / "run.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), fh],
    )

    t_wall = time.time()
    log.info("=" * 72)
    log.info("  EuroSAT BSCM Fixed-Pool Evaluation")
    log.info("=" * 72)

    # Locked tau (from So2Sat holdout — genuine transfer evaluation)
    tau = load_locked_tau()
    log.info(
        "Locked tau = %.3f  (selected on So2Sat held-out pool; "
        "applied to EuroSAT without re-tuning)", tau,
    )

    # Load data
    band_means, labels, class_names = load_eurosat_allbands(
        args.data_dir,
        cache_path="data/processed/eurosat_band_means.npz",
    )
    log.info("EuroSAT: %d samples, %d classes", len(labels), len(class_names))

    # Draw fixed pool (POOL_SEED=0, independent of evaluation seeds)
    pool_sss = StratifiedShuffleSplit(
        n_splits=1, train_size=N_TRAIN, test_size=N_TEST,
        random_state=POOL_SEED,
    )
    canonical_tr_full, canonical_te_full = next(
        pool_sss.split(band_means, labels)
    )
    pool_idx        = np.concatenate([canonical_tr_full, canonical_te_full])
    pool_labels     = labels[pool_idx]
    bm_pool         = band_means[pool_idx]          # (1500, 13)
    canonical_tr_local = np.arange(N_TRAIN)         # pool indices 0..999

    dist = dict(zip(class_names, np.bincount(pool_labels).tolist()))
    log.info(
        "Fixed pool: %d samples (pool_seed=%d, n_train_capacity=%d, "
        "n_test_capacity=%d)\n  Class distribution: %s",
        N_POOL, POOL_SEED, N_TRAIN, N_TEST, dist,
    )

    # Precompute pool features (fit on canonical training portion only)
    log.info("\nPreparing pool features...")
    feats = prepare_pool_features(bm_pool, canonical_tr_local)
    log.info("  PCA-8 explained variance: %.3f", feats["pca_explained_var"])
    log.info("  Physics-8: %s", PHYSICS_FEATURE_NAMES)

    feat_sets: List[Tuple[str, np.ndarray]] = []
    if not args.physics_only:
        feat_sets.append(("pca8",     feats["pca8"]))
    if not args.pca_only:
        feat_sets.append(("physics8", feats["physics8"]))

    run_srqfm = not args.skip_srqfm

    all_results = {
        "experiment":    "EuroSAT BSCM Fixed-Pool Evaluation",
        "dataset":       "EuroSAT (Helber et al. 2019) — 13 Sentinel-2 bands",
        "n_classes":     len(class_names),
        "class_names":   class_names,
        "pool_config": {
            "pool_seed":       POOL_SEED,
            "n_pool":          N_POOL,
            "n_train":         N_TRAIN,
            "n_test":          N_TEST,
            "eval_seeds":      EVAL_SEEDS,
            "feature_fit":     "fitted on canonical training portion (pool[:1000])",
        },
        "kernel_config": {
            "locked_tau":         tau,
            "tau_source":         "So2Sat held-out pool (200 samples), KTA-argmax",
            "bscm_connectivity":  BSCM_CONNECTIVITY,
            "bscm_coup_thresh":   BSCM_COUP_THRESH,
            "srqfm_connectivity": SRQFM_CONNECTIVITY,
            "srqfm_coup_thresh":  SRQFM_COUP_THRESH,
            "srqfm_topology_note": (
                "SRQFM uses linear (i,i+1) connectivity to match the published "
                "EuroSAT SRQFM in scripts/run_eurosat_experiment.py.  All-to-all "
                "on n_q=8 causes severe kernel concentration and is not the "
                "reference SRQFM topology."
            ),
            "zz_reps":            ZZ_REPS,
            "prior":              "uniform",
            "c_grid":             C_GRID,
        },
        "physics_features": PHYSICS_FEATURE_NAMES,
        "feature_sets": {},
    }

    for feat_name, X_pool in feat_sets:
        n_q = X_pool.shape[1]
        log.info("\n%s\n  FEATURE SET: %s  (n_qubits=%d)\n%s",
                 "=" * 72, feat_name.upper(), n_q, "=" * 72)

        # ── Kernel computation (once per feat_name) ──────────────────────────
        gram_matrices: Dict[str, np.ndarray] = {}

        log.info("\n  [1/2] BSCM-uniform kernel...")
        circuit_bscm = make_bscm_circuit(n_q, tau, prior="uniform")
        K_bscm = compute_gram_pool(
            circuit_bscm, X_pool,
            cache_path=KERNEL_DIR / f"K_pool_bscm_{feat_name}.npy",
            label=f"BSCM-uniform|{feat_name}",
        )
        gram_matrices["BSCM-uniform"] = K_bscm

        log.info("  Global BSCM-uniform spectral metrics (N=%d):", N_POOL)
        gm_bscm = spectral_metrics(K_bscm)
        for k, v in gm_bscm.items():
            log.info("    %s = %.4f", k, v)

        if run_srqfm:
            log.info("\n  [2/2] SRQFM-fid kernel...")
            circuit_srqfm = make_srqfm_circuit(n_q)
            # Filename includes connectivity tag to invalidate any pre-fix
            # all-to-all gram cached on disk.
            K_srqfm = compute_gram_pool(
                circuit_srqfm, X_pool,
                cache_path=KERNEL_DIR /
                    f"K_pool_srqfm_{SRQFM_CONNECTIVITY}_{feat_name}.npy",
                label=f"SRQFM-fid|{feat_name}",
            )
            gram_matrices["SRQFM-fid"] = K_srqfm

            log.info("  Global SRQFM-fid spectral metrics (N=%d):", N_POOL)
            gm_srqfm = spectral_metrics(K_srqfm)
            for k, v in gm_srqfm.items():
                log.info("    %s = %.4f", k, v)

        # ── Permutation KTA test on full pool (n_perm=1000) ────────────────────
        log.info("\n  Permutation KTA test (n_perm=1000) on full pool...")
        perm_results: Dict[str, dict] = {}
        for kname, K_g in gram_matrices.items():
            t_p = time.time()
            perm = perm_kta_pvalue(
                K_g, pool_labels, n_perm=1000, rng_seed=0, sub_n=None,
            )
            log.info(
                "    %-15s  KTA(cc)=%.4f  null=%.4f±%.4f  p=%.4f  (%.1fs)",
                kname, perm["kta_cc_observed"],
                perm["null_mean"], perm["null_std"], perm["p_value"],
                time.time() - t_p,
            )
            perm_results[kname] = perm

        # ── Evaluation on 5 within-pool splits ───────────────────────────────
        log.info("\n  Evaluating on %d within-pool splits...", len(EVAL_SEEDS))
        feat_entry = evaluate_feature_set(
            feat_name, X_pool, pool_labels, gram_matrices, run_srqfm,
        )

        # Attach global metrics to feat_entry
        feat_entry["global_spectral_bscm"] = gm_bscm
        if run_srqfm:
            feat_entry["global_spectral_srqfm"] = gm_srqfm
        feat_entry["permutation_kta"] = perm_results

        all_results["feature_sets"][feat_name] = feat_entry
        all_results["wall_clock_s"] = round(time.time() - t_wall, 1)
        _save(all_results)

    all_results["wall_clock_s"] = round(time.time() - t_wall, 1)
    _save(all_results)

    log.info("\n%s", "=" * 72)
    log.info("  DONE.  Wall clock: %.1f min", (time.time() - t_wall) / 60)
    log.info("  Results: %s", RESULTS_DIR / "eurosat_bscm_fixed_results.json")
    log.info("%s", "=" * 72)


def _save(results: dict):
    out = RESULTS_DIR / "eurosat_bscm_fixed_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
