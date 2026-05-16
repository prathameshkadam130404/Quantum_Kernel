"""
BSCM 14-qubit test: all 14 features, N=800 samples, locked tau.

Goal: settle the 14q-vs-8q debate empirically with a single contained
experiment.  After the kernel is computed, runs every diagnostic that
matters for deciding whether to scale 14q to N=1500.

PROTOCOL
--------
Qubits     : 14
Features   : 14   (drop the 2 lowest-Fisher: VV/VH and VV_tex)
Samples    : 800  (stratified sub-sample of the 1800 EVAL pool, seed 1234)
Tau        : 0.25 (locked from results/bscm/locked_tau.json — same lockdown
                   that produced the 8q result, so 14q is held to the same
                   hyperparameter discipline)
Bell prior : uniform   (the 8q winner)
Connect    : all       (matches the 8q recipe)
REPS       : 2         (matches the 8q recipe; do NOT change here)

Expected wall clock: ~12-13 hours on lightning.qubit
                     (43 evals/sec * 319,600 pairs / 7400 sec safety margin).
Checkpointing : every 25 rows (built into compute_bscm_fidelity_kernel).

WHAT IS REPORTED
----------------
Phase A -- structural health (always reported)
    1. Diagonal min/max (must be 1.0)
    2. Symmetry max |K - K^T|
    3. Off-diagonal mean / variance / std
    4. Off-diagonal histogram (20 bins, 0..1)
    5. Within-1-sigma fraction        <-- KEY CONCENTRATION TEST
    6. Off-diagonal percentiles 1, 5, 50, 95, 99
    7. Eigenvalue spectrum
       - all-positive check (PD kernel)
       - top-10 eigenvalues
       - Shannon and Huang effective ranks
    8. Direct comparison vs 8q baseline (same N=800 sub-sample if cached)

Phase B -- alignment metrics
    9. KTA with the SAME-CLASS labelling (correct multi-class formula)
   10. KTA with the >0 product labelling (matches the 8q eval script)
   11. Class-pair conditional means (intra-class fidelity vs inter-class)

Phase C -- classification (5 seeds * C-tuning {1, 10, 100}, 3-fold CV)
   12. Macro F1 mean +- std
   13. Accuracy mean +- std
   14. Per-class F1 (focus on previously-failing classes)
   15. Confusion matrix (seed=42)
   16. Per-class F1 delta vs 8q baseline at N=800

Phase D -- decision rules
   17. PRINTED VERDICT applying these gates:
       (a) Concentration:  abort if within-1-sigma > 0.95
       (b) Eff rank Shannon:  abort if < 8 (worse than 8q)
       (c) F1 lift over 8q baseline:
                  >= +0.04 -> "STRONGLY recommend scaling to N=1500"
                  +0.02 to +0.04 -> "borderline; pre-flight pass"
                  0 to +0.02 -> "marginal; not worth 1.8 days"
                  < 0 -> "do not pursue 14q path"

OUTPUT
------
results/bscm_14q_test/K_bscm_14q_phys14_uniform_tau0.25_N800.npy
results/bscm_14q_test/summary.json
results/bscm_14q_test/per_class_f1.csv
results/bscm_14q_test/confusion_matrix.npy
results/bscm_14q_test/eigenvalues.npy
results/bscm_14q_test/run.log
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._bscm_split import make_or_load_split
from experiments._e_common import load_physics_16, spectral_metrics
from src.attention_kernel import FEATURE_NAMES_16, compute_fisher_ratio
from src.bscm_kernel import BELL_WEIGHT_PRESETS, compute_bscm_fidelity_kernel

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
N_QUBITS = 14
N_FEATURES = 14
N_SAMPLES = 800
SUBSAMPLE_SEED = 1234            # same family as the EVAL/HOLDOUT split seed
LOCKED_TAU_PATH = os.path.join(config.RESULTS_DIR, "bscm", "locked_tau.json")
SAVE_DIR = os.path.join(config.RESULTS_DIR, "bscm_14q_test")
os.makedirs(SAVE_DIR, exist_ok=True)

KERNEL_FILENAME = (
    f"K_bscm_14q_phys14_uniform_tau0.25_N{N_SAMPLES}.npy"
)
KERNEL_PATH = os.path.join(SAVE_DIR, KERNEL_FILENAME)

SEEDS = config.SEED_LIST          # [42, 43, 44, 45, 46]
C_GRID = [1, 10, 100]
CV_FOLDS = 3
TEST_FRAC = 0.30

LCZ = config.LCZ_CLASS_NAMES
PREVIOUSLY_FAILING = [9, 4, 0, 6, 11, 3, 1]   # F1<0.40 in 8q baseline
EIGHT_Q_PER_CLASS_F1 = {           # measured from eval_bscm_uniform/per_class_f1.csv
    0: 0.2980, 1: 0.3869, 2: 0.5498, 3: 0.3608, 4: 0.2809, 5: 0.5365,
    6: 0.3274, 7: 0.4734, 8: 0.4179, 9: 0.2265, 10: 0.8528, 11: 0.4024,
    12: 0.5059, 13: 0.6540, 14: 0.4143, 15: 0.5733, 16: 0.9489,
}
EIGHT_Q_MACRO_F1 = 0.4829          # measured at 8q + N=1800

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "run.log"), mode="w"),
    ],
)
log = logging.getLogger("bscm_14q_test")


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #

def _sep(c="=", w=78):
    return c * w


def select_14_features(X_raw: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, list]:
    """Select 14 of 16 features by dropping the 2 lowest-Fisher ones.
    On the canonical so2sat data these are VV/VH (idx 8) and VV_tex (idx 14)."""
    fisher = compute_fisher_ratio(X_raw, y)
    drop = list(np.argsort(fisher)[:2])
    keep = sorted(i for i in range(16) if i not in drop)
    names = [FEATURE_NAMES_16[i] for i in keep]
    log.info("  Fisher rank (full 16):")
    for r, idx in enumerate(np.argsort(fisher)[::-1]):
        used = "KEEP" if idx in keep else "DROP"
        log.info("    #%2d  [%2d] %-12s Fisher=%.4f  %s",
                 r + 1, idx, FEATURE_NAMES_16[idx], fisher[idx], used)
    return np.array(keep), names


def stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Take a stratified n-sample of the 1800-sample EVAL pool indices."""
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    (idx, _), = sss.split(np.zeros(len(y)), y)
    return np.sort(idx)


def kta_same_class(K: np.ndarray, y: np.ndarray) -> float:
    """Multi-class KTA: ideal kernel = +1 if same class else -1."""
    n = len(y)
    Y_id = (y[:, None] == y[None, :]).astype(np.float64) * 2.0 - 1.0
    Ksym = (K + K.T) / 2.0
    num = float(np.sum(Ksym * Y_id))
    den = np.sqrt(float(np.sum(Ksym * Ksym)) * float(np.sum(Y_id * Y_id)))
    return num / (den + 1e-12)


def kta_product_labels(K: np.ndarray, y: np.ndarray) -> float:
    """Reproduces the formula used in scripts/evaluate_bscm_uniform.py."""
    n = len(y)
    Y = np.outer(y, y)
    Y = (Y > 0).astype(float) * 2.0 - 1.0
    Ksym = (K + K.T) / 2.0
    num = float(np.sum(Ksym * Y)) / (n * n)
    den = np.sqrt(float(np.sum(Ksym * Ksym)) * float(np.sum(Y * Y))) / (n * n)
    return num / (den + 1e-12)


def tune_c(K_train: np.ndarray, y_train: np.ndarray, seed: int) -> int:
    best_c, best_f1 = C_GRID[0], -1.0
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    for C in C_GRID:
        folds = []
        for cv_tr, cv_val in skf.split(np.zeros(len(y_train)), y_train):
            K_cv = K_train[np.ix_(cv_tr, cv_tr)]
            K_cv_v = K_train[np.ix_(cv_val, cv_tr)]
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_cv, y_train[cv_tr])
            folds.append(f1_score(y_train[cv_val], clf.predict(K_cv_v),
                                   average="macro", zero_division=0))
        m = float(np.mean(folds))
        if m > best_f1:
            best_f1, best_c = m, C
    return best_c


def eval_one_seed(K_full: np.ndarray, y: np.ndarray, seed: int) -> dict:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)
    K_tr = K_full[np.ix_(tr, tr)]
    K_te = K_full[np.ix_(te, tr)]
    best_c = tune_c(K_tr, y[tr], seed)
    clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    yp = clf.predict(K_te)
    return {
        "seed": seed,
        "best_C": best_c,
        "macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y[te], yp)),
        "y_test": y[te],
        "y_pred": yp,
        "per_class_f1": f1_score(y[te], yp, average=None,
                                  labels=list(range(17)), zero_division=0),
    }


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main() -> int:
    log.info(_sep())
    log.info(" BSCM 14-qubit test")
    log.info(" Started: %s", datetime.now().isoformat())
    log.info(_sep())
    log.info(" Qubits        : %d", N_QUBITS)
    log.info(" Features      : %d (drop 2 lowest-Fisher of 16)", N_FEATURES)
    log.info(" Samples       : %d (stratified from 1800 EVAL pool)", N_SAMPLES)
    log.info(" Subsample seed: %d", SUBSAMPLE_SEED)
    log.info(" Bell prior    : uniform")
    log.info(" Connectivity  : all")
    log.info(" REPS          : %d", config.ZZ_REPS)
    log.info(_sep())

    # ----- Load locked tau ----------------------------------------------- #
    if not os.path.exists(LOCKED_TAU_PATH):
        raise RuntimeError(f"Missing locked tau at {LOCKED_TAU_PATH}")
    with open(LOCKED_TAU_PATH) as f:
        tau_locked = float(json.load(f)["locked"]["tau"])
    log.info(" Locked tau    : %.4f (from Phase 2 holdout pool)", tau_locked)
    if abs(tau_locked - 0.25) > 1e-9:
        log.warning(" tau_locked is not 0.25; output filename hard-codes 0.25")

    # ----- Load features and labels -------------------------------------- #
    log.info("\n[step 1] Load physics_16 features and labels")
    X_norm_full, X_raw_full, y_full = load_physics_16()
    sp = make_or_load_split(y_full)
    eval_pool = sp.eval_pool   # 1800 indices
    log.info("  Eval pool size: %d", len(eval_pool))

    # ----- Stratified sub-sample of N=800 from the eval pool ------------- #
    log.info("\n[step 2] Stratified sub-sample to N=%d", N_SAMPLES)
    sub_idx = stratified_subsample(y_full[eval_pool], N_SAMPLES,
                                    SUBSAMPLE_SEED)
    # Map sub-sample indices back to original eval pool indices
    sub_orig = eval_pool[sub_idx]
    y = y_full[sub_orig]
    log.info("  Sub-sample size: %d", len(y))
    cls, cnt = np.unique(y, return_counts=True)
    for c, n in zip(cls, cnt):
        log.info("    [%2d] %-22s %4d  (%.1f%%)", c, LCZ[c], n, 100*n/len(y))

    # ----- Feature selection (top 14 by Fisher, on TRAINING only) ------- #
    log.info("\n[step 3] Select 14 features (training-only Fisher, no leakage)")
    # Split N=800 into train/test using first seed to define the training set
    FIRST_SEED = SEEDS[0]  # 42
    sss_feat = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                                      random_state=FIRST_SEED)
    (train_sub, _), = sss_feat.split(np.zeros(N_SAMPLES), y)
    train_orig = sub_orig[train_sub]

    keep_idx, keep_names = select_14_features(
        X_raw_full[train_orig], y_full[train_orig])
    log.info("  Selected (14) from seed=%d training portion: %s",
             FIRST_SEED, keep_names)

    X_norm_eval = X_norm_full[eval_pool][:, keep_idx]
    X_raw_eval = X_raw_full[eval_pool][:, keep_idx]
    y_eval = y_full[eval_pool]

    X_enc = X_norm_full[sub_orig][:, keep_idx]
    X_raw = X_raw_full[sub_orig][:, keep_idx]
    log.info("  Sub-sample size: %d", len(y))
    cls, cnt = np.unique(y, return_counts=True)
    for c, n in zip(cls, cnt):
        log.info("    [%2d] %-22s %4d  (%.1f%%)", c, LCZ[c], n, 100*n/len(y))

    # ===================================================================== #
    # PHASE 1 -- BUILD THE KERNEL
    # ===================================================================== #
    log.info("\n%s", _sep())
    log.info(" Phase 1: build BSCM kernel (14 qubits, N=%d)", N_SAMPLES)
    log.info(" Expected wall clock: ~12-13 hours on lightning.qubit")
    log.info(" Checkpointing every 25 rows to %s.ckpt.npz", KERNEL_PATH)
    log.info(_sep())
    t0 = time.time()
    K = compute_bscm_fidelity_kernel(
        X=X_enc,
        n_qubits=N_QUBITS,
        reps=config.ZZ_REPS,
        tau=tau_locked,
        coupling_threshold=1e-4,
        connectivity="all",
        bell_weights=BELL_WEIGHT_PRESETS["uniform"],
        kernel_save_path=KERNEL_PATH,
        desc=f"BSCM-14q-uniform(N={N_SAMPLES})",
    )
    build_seconds = time.time() - t0
    log.info(" Kernel build done in %.2f hours", build_seconds / 3600)
    log.info(" Saved to %s", KERNEL_PATH)

    # ===================================================================== #
    # PHASE A -- STRUCTURAL HEALTH
    # ===================================================================== #
    log.info("\n%s", _sep())
    log.info(" Phase A: structural health")
    log.info(_sep())

    diag_min = float(K.diagonal().min())
    diag_max = float(K.diagonal().max())
    sym_err = float(np.max(np.abs(K - K.T)))
    log.info("  Diagonal min / max : %.6f / %.6f", diag_min, diag_max)
    log.info("  Max |K - K^T|      : %.2e", sym_err)
    diag_ok = (diag_min > 0.99 and diag_max < 1.01 and sym_err < 1e-6)
    log.info("  Health             : %s", "PASS" if diag_ok else "FAIL")

    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = K[mask]
    sm = spectral_metrics(K)
    Ksym = (K + K.T) / 2.0
    eig = np.linalg.eigvalsh(Ksym)
    np.save(os.path.join(SAVE_DIR, "eigenvalues.npy"), eig)

    log.info("\n  Off-diagonal stats:")
    log.info("    mean  : %.6f", sm["off_diag_mean"])
    log.info("    var   : %.6f", sm["off_diag_var"])
    log.info("    std   : %.6f", float(np.sqrt(sm["off_diag_var"])))
    log.info("    pct  1: %.6f", float(np.percentile(off, 1)))
    log.info("    pct  5: %.6f", float(np.percentile(off, 5)))
    log.info("    pct 50: %.6f", float(np.percentile(off, 50)))
    log.info("    pct 95: %.6f", float(np.percentile(off, 95)))
    log.info("    pct 99: %.6f", float(np.percentile(off, 99)))
    within_1sig = float(np.mean(np.abs(off - off.mean()) < off.std()))
    log.info("    within 1 sigma : %.4f  (>0.95 = exponentially concentrated)",
             within_1sig)

    log.info("\n  Off-diagonal histogram (20 bins, 0..1):")
    hist, edges = np.histogram(off, bins=20, range=(0.0, 1.0))
    bar_unit = max(1, hist.max() // 30)
    for i in range(20):
        bar = "#" * (hist[i] // bar_unit)
        log.info("    [%.2f-%.2f]  %8d  %s", edges[i], edges[i+1], hist[i], bar)

    log.info("\n  Spectrum:")
    log.info("    n_positive_eigs : %d / %d", int((eig > 0).sum()), n)
    log.info("    eig min / max   : %.4e / %.4e", eig.min(), eig.max())
    log.info("    top-10 eigs     : %s",
             np.array2string(np.sort(eig)[::-1][:10], precision=3))
    log.info("    eff rank Shannon: %.2f / %d", sm["eff_rank_shannon"], n)
    log.info("    eff rank Huang  : %.2f / %d", sm["eff_rank_huang"], n)

    # ----- Side-by-side vs 8q baseline (recall: 8q at N=1800) ----------- #
    log.info("\n  vs 8q baseline (BSCM-uniform N=1800):")
    log.info("    metric            8q(N=1800)        14q(N=%d)", N_SAMPLES)
    log.info("    off_diag_mean     0.214630          %.6f", sm["off_diag_mean"])
    log.info("    off_diag_var      0.074209          %.6f", sm["off_diag_var"])
    log.info("    within_1sigma     0.807             %.4f", within_1sig)
    log.info("    eff_rank_Shannon  16.7              %.2f", sm["eff_rank_shannon"])

    # ===================================================================== #
    # PHASE B -- ALIGNMENT
    # ===================================================================== #
    log.info("\n%s", _sep())
    log.info(" Phase B: kernel-target alignment")
    log.info(_sep())

    kta_correct = kta_same_class(K, y)
    kta_legacy = kta_product_labels(K, y)
    log.info("  KTA (same-class, correct multi-class) : %.6f", kta_correct)
    log.info("  KTA (product>0, matches 8q eval script): %.6f", kta_legacy)

    # Intra- vs inter-class fidelity means
    same = []
    diff = []
    for i in range(n):
        for j in range(i+1, n):
            if y[i] == y[j]:
                same.append(K[i, j])
            else:
                diff.append(K[i, j])
    same = np.array(same); diff = np.array(diff)
    log.info("  Intra-class fidelity mean : %.4f  (n_pairs=%d)",
             float(same.mean()), len(same))
    log.info("  Inter-class fidelity mean : %.4f  (n_pairs=%d)",
             float(diff.mean()), len(diff))
    log.info("  Separation (intra - inter): %.4f  (larger = better)",
             float(same.mean() - diff.mean()))

    # ===================================================================== #
    # PHASE C -- CLASSIFICATION
    # ===================================================================== #
    log.info("\n%s", _sep())
    log.info(" Phase C: SVM classification (5 seeds, C tuning)")
    log.info(_sep())

    seed_results = []
    for seed in SEEDS:
        r = eval_one_seed(K, y, seed)
        seed_results.append(r)
        log.info("  seed=%d  macro_F1=%.4f  acc=%.4f  best_C=%d",
                 seed, r["macro_f1"], r["accuracy"], r["best_C"])

    f1_vals = np.array([r["macro_f1"] for r in seed_results])
    acc_vals = np.array([r["accuracy"] for r in seed_results])
    log.info("\n  Macro F1 : %.4f +- %.4f  (min=%.4f max=%.4f)",
             f1_vals.mean(), f1_vals.std(ddof=1), f1_vals.min(), f1_vals.max())
    log.info("  Accuracy : %.4f +- %.4f",
             acc_vals.mean(), acc_vals.std(ddof=1))

    # ----- Per-class -------------------------------------------------- #
    log.info("\n  Per-class F1 (averaged over 5 seeds):")
    pc_stack = np.stack([r["per_class_f1"] for r in seed_results], axis=0)
    pc_mean = pc_stack.mean(axis=0)
    pc_std = pc_stack.std(axis=0, ddof=1)

    rows = []
    log.info("  %3s  %-22s  %8s  %8s  %12s  %s",
             "Cls", "Name", "F1_14q", "std", "F1_8q(N=1800)", "delta")
    log.info("  " + "-" * 76)
    for c in range(17):
        f1_8q = EIGHT_Q_PER_CLASS_F1.get(c, np.nan)
        delta = pc_mean[c] - f1_8q
        marker = "  <<-- previously failing" if c in PREVIOUSLY_FAILING else ""
        log.info("  %3d  %-22s  %.4f    %.4f    %.4f       %+.4f%s",
                 c, LCZ[c], pc_mean[c], pc_std[c], f1_8q, delta, marker)
        rows.append({"class": c, "name": LCZ[c],
                      "f1_14q_mean": float(pc_mean[c]),
                      "f1_14q_std":  float(pc_std[c]),
                      "f1_8q_baseline": float(f1_8q),
                      "delta": float(delta),
                      "previously_failing": c in PREVIOUSLY_FAILING})
    pd.DataFrame(rows).to_csv(
        os.path.join(SAVE_DIR, "per_class_f1.csv"), index=False
    )

    # ----- Lacking-class summary ------------------------------------- #
    log.info("\n  PREVIOUSLY-FAILING classes summary:")
    sum_8q = sum(EIGHT_Q_PER_CLASS_F1[c] for c in PREVIOUSLY_FAILING)
    sum_14q = sum(pc_mean[c] for c in PREVIOUSLY_FAILING)
    n_failing = len(PREVIOUSLY_FAILING)
    log.info("    Mean F1 over failing classes  8q: %.4f", sum_8q / n_failing)
    log.info("    Mean F1 over failing classes 14q: %.4f", sum_14q / n_failing)
    log.info("    Lift on failing classes        : %+.4f",
             (sum_14q - sum_8q) / n_failing)

    # ----- Confusion matrix (seed=42) -------------------------------- #
    cm = confusion_matrix(seed_results[0]["y_test"], seed_results[0]["y_pred"],
                           labels=list(range(17)))
    np.save(os.path.join(SAVE_DIR, "confusion_matrix.npy"), cm)
    log.info("\n  Confusion matrix (seed=42) saved.")

    # ===================================================================== #
    # PHASE D -- DECISION GATES
    # ===================================================================== #
    log.info("\n%s", _sep())
    log.info(" Phase D: decision verdict")
    log.info(_sep())

    abort_concentration = within_1sig > 0.95
    abort_rank = sm["eff_rank_shannon"] < 8.0
    f1_lift = float(f1_vals.mean()) - EIGHT_Q_MACRO_F1

    log.info("  Gate (a) Concentration  within_1sigma=%.4f -> %s",
             within_1sig, "ABORT" if abort_concentration else "PASS")
    log.info("  Gate (b) Effective rank Shannon=%.2f -> %s",
             sm["eff_rank_shannon"], "ABORT" if abort_rank else "PASS")
    log.info("  Gate (c) F1 lift over 8q baseline: %+.4f",
             f1_lift)

    if abort_concentration or abort_rank:
        verdict = "ABORT 14q path: kernel is degenerate at N=800 already; " \
                   "scaling to N=1500 will not recover."
    elif f1_lift >= 0.04:
        verdict = "STRONGLY recommend scaling to N=1500 -- 14q gives material " \
                   "lift over 8q."
    elif f1_lift >= 0.02:
        verdict = "Borderline -- N=1500 14q run is justified; expect modest gain."
    elif f1_lift >= 0.0:
        verdict = "Marginal -- N=1500 14q run probably not worth 1.8 days."
    else:
        verdict = "Do NOT pursue 14q path -- N=800 14q is worse than N=1800 8q. " \
                   "Stick with multi-kernel (option 2) or feature swap (option 1)."
    log.info("\n  VERDICT: %s", verdict)

    # ===================================================================== #
    # SAVE SUMMARY
    # ===================================================================== #
    summary = {
        "config": {
            "n_qubits": N_QUBITS,
            "n_features": N_FEATURES,
            "feature_names_kept": keep_names,
            "n_samples": N_SAMPLES,
            "subsample_seed": SUBSAMPLE_SEED,
            "tau_locked": tau_locked,
            "bell_prior": "uniform",
            "connectivity": "all",
            "reps": config.ZZ_REPS,
        },
        "build": {
            "wall_clock_seconds": float(build_seconds),
            "wall_clock_hours": float(build_seconds / 3600),
            "evals_per_sec": float(N_SAMPLES * (N_SAMPLES - 1) / 2 / build_seconds),
        },
        "structural": {
            "diag_min": diag_min, "diag_max": diag_max,
            "symmetry_max_err": sym_err,
            "off_diag_mean": float(sm["off_diag_mean"]),
            "off_diag_var": float(sm["off_diag_var"]),
            "off_diag_std": float(np.sqrt(sm["off_diag_var"])),
            "within_1sigma": within_1sig,
            "off_diag_pct": {
                "p1": float(np.percentile(off, 1)),
                "p5": float(np.percentile(off, 5)),
                "p50": float(np.percentile(off, 50)),
                "p95": float(np.percentile(off, 95)),
                "p99": float(np.percentile(off, 99)),
            },
            "eff_rank_shannon": float(sm["eff_rank_shannon"]),
            "eff_rank_huang": float(sm["eff_rank_huang"]),
            "n_positive_eigs": int((eig > 0).sum()),
            "eig_min": float(eig.min()),
            "eig_max": float(eig.max()),
            "top_10_eigs": [float(v) for v in np.sort(eig)[::-1][:10]],
        },
        "alignment": {
            "kta_same_class": float(kta_correct),
            "kta_product_labels": float(kta_legacy),
            "intra_class_mean": float(same.mean()),
            "inter_class_mean": float(diff.mean()),
            "separation": float(same.mean() - diff.mean()),
        },
        "classification": {
            "macro_f1_mean": float(f1_vals.mean()),
            "macro_f1_std": float(f1_vals.std(ddof=1)),
            "accuracy_mean": float(acc_vals.mean()),
            "accuracy_std": float(acc_vals.std(ddof=1)),
            "per_seed": [
                {"seed": r["seed"], "macro_f1": r["macro_f1"],
                 "accuracy": r["accuracy"], "best_C": r["best_C"]}
                for r in seed_results
            ],
            "per_class_mean_f1": {int(c): float(pc_mean[c]) for c in range(17)},
            "lacking_class_lift": float((sum_14q - sum_8q) / n_failing),
        },
        "comparison": {
            "f1_8q_baseline_N1800": EIGHT_Q_MACRO_F1,
            "f1_14q_lift_over_8q": f1_lift,
        },
        "decision": {
            "abort_concentration": bool(abort_concentration),
            "abort_rank": bool(abort_rank),
            "f1_lift": f1_lift,
            "verdict": verdict,
        },
    }
    with open(os.path.join(SAVE_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("\nSummary saved: %s/summary.json", SAVE_DIR)
    log.info("\nFINAL: %s", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
