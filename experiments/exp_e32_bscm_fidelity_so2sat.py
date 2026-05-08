"""
E32 -- Bell-Spectrum Coupling Map (BSCM) Fidelity Kernel on So2Sat.

Both feature sets:
    physics8  -- Fisher-selected 8 features from the physics_features_16.npz
                 file, scaled to [0, pi].  Same selection rule as E26.
    pca8      -- PCA-fused 8-D representation from subsample_2000.npz
                 (`fused_X_train`).  Already in [0, pi].

Methods compared on each feature set:

    BSCM-uniform(tau_locked)   uniform Bell-prior, tau locked by Phase 2
    BSCM-phi-only(tau_locked)  ablation: |Phi+>,|Phi-> sectors only
    BSCM-psi-only(tau_locked)  ablation: |Psi+>,|Psi-> sectors only
    SRQFM-fid                  fidelity variant of SRQFM v1, all-to-all
    SRQFM-PQK                  canonical PQK SRQFM v1 (E26 kernel for
                               physics8; freshly computed for pca8)
    RBF-SVM                    GridSearchCV over (C, gamma)
    RandomForest               n_estimators=500

Evaluation harness mirrors E26 byte-for-byte:
    SEED_LIST = [42, 43, 44, 45, 46]
    test_frac = 0.30, stratified
    C grid    = {1, 10, 100}, 3-fold CV on the train split

Statistical tests:
    Wilcoxon signed-rank, Cohen's d, Holm-Bonferroni over BSCM vs others.
    Computed *separately for each feature set*.

Hyperparameter discipline:
    All BSCM evaluations use the SAME locked tau (read from
    results/bscm/locked_tau.json) selected on the disjoint 200-sample
    HOLDOUT pool.  Evaluation runs only on the 1800-sample EVAL pool.

Output:
    results/bscm/metrics.csv               (per-seed F1 long-form)
    results/bscm/exp_e32.log
    results/bscm/K_bscm_{fset}_{prior}.npy (cached BSCM kernels)
    results/bscm/K_srqfm_fid_{fset}.npy
    results/bscm/K_srqfm_pqk_{fset}.npy
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime

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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._bscm_split import make_or_load_split
from experiments._e_common import (
    cohens_d,
    holm_bonferroni,
    load_physics_16,
    spectral_metrics,
)
from src.attention_kernel import select_features_by_fisher, FEATURE_NAMES_16
from src.bscm_kernel import BELL_WEIGHT_PRESETS, compute_bscm_fidelity_kernel
from src.srqfm_fidelity_kernel import compute_srqfm_fidelity_kernel
from src.srqfm_kernel import compute_srqfm_pqk_kernel

# ----- Configuration --------------------------------------------------------
SEEDS = config.SEED_LIST
N_QUBITS = config.N_QUBITS
REPS = config.ZZ_REPS
TEST_FRAC = 0.30
C_GRID = [1, 10, 100]
CV_FOLDS = 3
BELL_PRIORS = ["uniform", "phi_only", "psi_only"]

SAVE_DIR = os.path.join(config.RESULTS_DIR, "bscm")
os.makedirs(SAVE_DIR, exist_ok=True)
LOCKED_TAU_PATH = os.path.join(SAVE_DIR, "locked_tau.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "exp_e32.log"), mode="w"),
    ],
)
logger = logging.getLogger("exp_e32")


def _load_locked_tau() -> float:
    if not os.path.exists(LOCKED_TAU_PATH):
        raise RuntimeError(
            f"Locked tau not found at {LOCKED_TAU_PATH}.  "
            f"Run scripts/bscm_phase2_lockdown.py first."
        )
    with open(LOCKED_TAU_PATH) as f:
        return float(json.load(f)["locked"]["tau"])


# ----- Tuning helpers (mirror E26) ------------------------------------------

def tune_c_precomputed(K_train, y_train, c_grid=C_GRID, n_folds=CV_FOLDS, seed=42):
    best_c, best_f1 = c_grid[0], -1.0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for C in c_grid:
        fold_f1 = []
        for cv_tr, cv_val in skf.split(np.zeros(len(y_train)), y_train):
            K_cv = K_train[np.ix_(cv_tr, cv_tr)]
            K_cv_val = K_train[np.ix_(cv_val, cv_tr)]
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_cv, y_train[cv_tr])
            yp = clf.predict(K_cv_val)
            fold_f1.append(f1_score(y_train[cv_val], yp, average="macro",
                                     zero_division=0))
        m = float(np.mean(fold_f1))
        if m > best_f1:
            best_f1, best_c = m, C
    return best_c


def eval_kernel_tuned(K_full, y, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)
    K_tr = K_full[np.ix_(tr, tr)]
    K_te = K_full[np.ix_(te, tr)]
    best_c = tune_c_precomputed(K_tr, y[tr], seed=seed)
    clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    yp = clf.predict(K_te)
    return {"macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0)),
            "best_C": best_c}


def eval_rbf_tuned(X_raw, y, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_raw[tr])
    X_te = scaler.transform(X_raw[te])
    grid = {"C": C_GRID, "gamma": ["scale", 0.01, 0.1, 1.0]}
    clf = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"),
                        grid, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1)
    clf.fit(X_tr, y[tr])
    yp = clf.predict(X_te)
    return {"macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0)),
            "best_C": clf.best_params_["C"],
            "best_gamma": clf.best_params_["gamma"]}


def eval_rf(X_raw, y, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)
    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                  random_state=seed, n_jobs=-1)
    clf.fit(X_raw[tr], y[tr])
    yp = clf.predict(X_raw[te])
    return {"macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0))}


# ----- Feature-set loaders --------------------------------------------------

def load_feature_set(name: str):
    """Returns (X_encoded[N,8], X_raw[N,8], y[N]) all restricted to the
    EVAL pool defined by experiments/_bscm_split.

    name = 'physics8' -> Fisher-selected 8 of 16 physics features
    name = 'pca8'     -> 8-D fused PCA representation
    """
    if name == "physics8":
        X_norm, X_raw, y = load_physics_16()
        sel_idx, _ = select_features_by_fisher(X_raw, y)
        X_enc = X_norm[:, sel_idx]
        X_raw_sel = X_raw[:, sel_idx]
        sel_names = [FEATURE_NAMES_16[i] for i in sel_idx]
    elif name == "pca8":
        p = os.path.join(config.PROCESSED_DIR, "subsample_2000.npz")
        d = np.load(p)
        X_enc = d["fused_X_train"]            # already in [0, pi]
        # Build a "raw" feature matrix in the original (pre-pi) scale
        # by inverse-mapping x_enc / pi  in [0, 1] domain.  RBF/RF will
        # see the [0, 1]-scaled features.
        X_raw_sel = X_enc / np.pi
        y = d["y_train"]
        sel_names = [f"pc{i+1}" for i in range(8)]
    else:
        raise ValueError(f"Unknown feature set: {name}")

    split = make_or_load_split(y)
    return (
        X_enc[split.eval_pool],
        X_raw_sel[split.eval_pool],
        y[split.eval_pool],
        sel_names,
    )


# ----- Kernel builders ------------------------------------------------------

def build_bscm(X_enc, fset_name: str, prior: str, tau: float):
    cache = os.path.join(SAVE_DIR, f"K_bscm_{fset_name}_{prior}.npy")
    return compute_bscm_fidelity_kernel(
        X_enc, n_qubits=N_QUBITS, reps=REPS, tau=tau,
        coupling_threshold=1e-4, connectivity="all",
        bell_weights=BELL_WEIGHT_PRESETS[prior],
        kernel_save_path=cache, desc=f"BSCM-{prior}({fset_name})",
    )


def build_srqfm_fidelity(X_enc, fset_name: str):
    cache = os.path.join(SAVE_DIR, f"K_srqfm_fid_{fset_name}.npy")
    return compute_srqfm_fidelity_kernel(
        X_enc, n_qubits=N_QUBITS, reps=REPS,
        coupling_threshold=0.01, connectivity="all",
        kernel_save_path=cache, desc=f"SRQFM-fid({fset_name})",
    )


def build_srqfm_pqk(X_enc, fset_name: str):
    cache = os.path.join(SAVE_DIR, f"K_srqfm_pqk_{fset_name}.npy")
    if os.path.exists(cache):
        logger.info("[CACHE] Loading SRQFM-PQK from %s", cache)
        return np.load(cache)
    # Compute fresh.  Note: this is the canonical PQK; fast (Bloch + RBF).
    K = compute_srqfm_pqk_kernel(
        X_enc, gamma=config.PQK_GAMMA, n_qubits=N_QUBITS, reps=REPS,
        coupling_threshold=0.01, connectivity="all",
        kernel_save_path=cache,
    )
    return K


# ----- Per-feature-set evaluation -------------------------------------------

def evaluate_feature_set(name: str, tau_locked: float, rows_out: list):
    logger.info("\n" + "=" * 72)
    logger.info("  Feature set: %s  (tau_locked = %.3f)", name, tau_locked)
    logger.info("=" * 72)

    X_enc, X_raw, y, sel_names = load_feature_set(name)
    logger.info("  EVAL pool: N=%d  features=%s", len(y), sel_names)

    # Build kernels.
    kernels = {}
    for prior in BELL_PRIORS:
        t0 = time.time()
        kernels[f"BSCM-{prior}"] = build_bscm(X_enc, name, prior, tau_locked)
        logger.info("  BSCM-%s built  (%.1fs)", prior, time.time() - t0)
    t0 = time.time()
    kernels["SRQFM-fid"] = build_srqfm_fidelity(X_enc, name)
    logger.info("  SRQFM-fid built  (%.1fs)", time.time() - t0)
    t0 = time.time()
    kernels["SRQFM-PQK"] = build_srqfm_pqk(X_enc, name)
    logger.info("  SRQFM-PQK built  (%.1fs)", time.time() - t0)

    # Spectral diagnostics.
    for kname, K in kernels.items():
        sm = spectral_metrics(K)
        logger.info(
            "  %-18s off_mean=%.4f  off_var=%.5f  eff_rank=%.1f",
            kname, sm["off_diag_mean"], sm["off_diag_var"],
            sm["eff_rank_shannon"],
        )

    # Seed-level evaluation.
    for seed in SEEDS:
        logger.info("\n  Seed %d (feature_set=%s):", seed, name)
        for kname, K in kernels.items():
            r = eval_kernel_tuned(K, y, seed)
            rows_out.append({
                "feature_set": name, "method": kname,
                "seed": seed, "macro_f1": r["macro_f1"],
                "best_C": r["best_C"],
            })
            logger.info("    %-18s F1=%.4f  C=%s",
                         kname, r["macro_f1"], r["best_C"])
        r_rbf = eval_rbf_tuned(X_raw, y, seed)
        rows_out.append({"feature_set": name, "method": "RBF-SVM",
                          "seed": seed, **r_rbf})
        logger.info("    %-18s F1=%.4f  C=%s  gamma=%s",
                     "RBF-SVM", r_rbf["macro_f1"], r_rbf["best_C"],
                     r_rbf.get("best_gamma"))
        r_rf = eval_rf(X_raw, y, seed)
        rows_out.append({"feature_set": name, "method": "RandomForest",
                          "seed": seed, "macro_f1": r_rf["macro_f1"]})
        logger.info("    %-18s F1=%.4f", "RandomForest", r_rf["macro_f1"])


def stat_tests(df: pd.DataFrame, fset: str, primary: str = "BSCM-uniform"):
    sub = df[df.feature_set == fset]
    methods = sorted(sub["method"].unique())
    f1_by = {m: sub[sub.method == m]["macro_f1"].values for m in methods}

    logger.info("\n  --- Stats: feature_set=%s, primary=%s ---", fset, primary)
    for m in methods:
        v = f1_by[m]
        logger.info("    %-18s mean=%.4f  std=%.4f  n=%d",
                     m, v.mean(), v.std(), len(v))

    primary_f1 = f1_by[primary]
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
    logger.info("    %-18s %10s %8s %6s", "Method", "p-value", "Cohen-d", "sig*")
    for m, p, d, s in zip(comparisons, pvals, ds, sig):
        logger.info("    %-18s %10.4f %8.3f %6s", m, p, d, "*" if s else "")


def run():
    logger.info("=" * 72)
    logger.info("  E32: BSCM Fidelity Benchmark on So2Sat (physics8 + pca8)")
    logger.info("=" * 72)
    logger.info("  Date:  %s", datetime.now().isoformat())
    logger.info("  Seeds: %s", SEEDS)

    tau_locked = _load_locked_tau()
    logger.info("  Locked tau (from holdout-only Phase 2): %.3f", tau_locked)

    rows = []
    for fset in ["physics8", "pca8"]:
        evaluate_feature_set(fset, tau_locked, rows)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(SAVE_DIR, "metrics.csv")
    df.to_csv(out_csv, index=False)
    logger.info("\n  Saved metrics: %s", out_csv)

    for fset in ["physics8", "pca8"]:
        stat_tests(df, fset, primary="BSCM-uniform")

    logger.info("\n  E32 complete.")
    return df


if __name__ == "__main__":
    run()
