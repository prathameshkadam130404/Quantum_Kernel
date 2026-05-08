"""
Honest head-to-head comparison:

    A.  BSCM-uniform   8q   N=1800   (already-cached kernel)
    B.  BSCM-phi_only  8q   N=1800   (already-cached kernel)
    C.  BSCM-uniform  14q   N= 800   (already-cached kernel from bscm_14q_test)
    D.  Random Forest  on physics-8 features (Fisher-selected, same eval pool)
        - evaluated on N=1800 (matched against A, B)
        - evaluated on N= 800 (matched against C, same sub-sample seed)

For each method we report (5 seeds, stratified 70/30, C-tuned for SVMs):
    macro F1 mean +- std,  accuracy mean +- std,
    KTA (for the SVM kernels),
    eff rank Shannon, off-diag mean / var / within-1sigma,
    per-class F1 means.

Why this script exists:
    The 14q test verdict was "do not pursue 14q at N=800". This script
    reproduces the verdict in a single side-by-side table together with
    8q-phi_only (never previously evaluated end-to-end) and Random Forest
    on the IDENTICAL feature set the 8q quantum kernel sees, so the
    quantum-vs-classical gap is computed with no feature confound.

Output:
    results/bscm_compare_14q_8q_rf/summary.json
    results/bscm_compare_14q_8q_rf/comparison_table.csv
    results/bscm_compare_14q_8q_rf/run.log
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._bscm_split import make_or_load_split
from experiments._e_common import load_physics_16, spectral_metrics
from src.attention_kernel import (
    FEATURE_NAMES_16,
    compute_fisher_ratio,
    select_features_by_fisher,
)

# --------------------------------------------------------------------------- #
# Paths to already-built artefacts
# --------------------------------------------------------------------------- #
KERNEL_8Q_UNIFORM = os.path.join(
    config.RESULTS_DIR, "bscm", "K_bscm_physics8_uniform.npy"
)
KERNEL_8Q_PHI = os.path.join(
    config.RESULTS_DIR, "bscm", "K_bscm_physics8_phi_only.npy"
)
KERNEL_14Q = os.path.join(
    config.RESULTS_DIR, "bscm_14q_test",
    "K_bscm_14q_phys14_uniform_tau0.25_N800.npy",
)

SAVE_DIR = os.path.join(config.RESULTS_DIR, "bscm_compare_14q_8q_rf")
os.makedirs(SAVE_DIR, exist_ok=True)

SEEDS = config.SEED_LIST
C_GRID = [1, 10, 100]
CV_FOLDS = 3
TEST_FRAC = 0.30
N_QUBITS_8Q = 8
SUB_SAMPLE_SEED_14Q = 1234     # must match scripts/bscm_14q_test.py
N_SAMPLES_14Q = 800

LCZ = config.LCZ_CLASS_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "run.log"), mode="w"),
    ],
)
log = logging.getLogger("bscm_compare")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _sep(c="=", w=78):
    return c * w


def kta_product_labels(K: np.ndarray, y: np.ndarray) -> float:
    """Same KTA formula as evaluate_bscm_uniform.py (product>0 labels)."""
    n = len(y)
    Y = np.outer(y, y)
    Y = (Y > 0).astype(float) * 2.0 - 1.0
    Ksym = (K + K.T) / 2.0
    num = float(np.sum(Ksym * Y)) / (n * n)
    den = np.sqrt(float(np.sum(Ksym * Ksym)) * float(np.sum(Y * Y))) / (n * n)
    return num / (den + 1e-12)


def kta_same_class(K: np.ndarray, y: np.ndarray) -> float:
    Y_id = (y[:, None] == y[None, :]).astype(np.float64) * 2.0 - 1.0
    Ksym = (K + K.T) / 2.0
    num = float(np.sum(Ksym * Y_id))
    den = np.sqrt(float(np.sum(Ksym * Ksym)) * float(np.sum(Y_id * Y_id)))
    return num / (den + 1e-12)


def tune_c_svm(K_train, y_train, seed):
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


def eval_kernel_one_seed(K_full, y, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                                  random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)
    K_tr = K_full[np.ix_(tr, tr)]
    K_te = K_full[np.ix_(te, tr)]
    best_c = tune_c_svm(K_tr, y[tr], seed)
    clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    yp = clf.predict(K_te)
    return {
        "seed": seed, "best_C": best_c,
        "macro_f1": float(f1_score(y[te], yp, average="macro",
                                     zero_division=0)),
        "accuracy": float(accuracy_score(y[te], yp)),
        "per_class_f1": f1_score(y[te], yp, average=None,
                                  labels=list(range(17)), zero_division=0),
    }


def eval_rf_one_seed(X, y, seed):
    """RandomForest with the same 70/30 stratified split discipline."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                                  random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)
    clf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X[tr], y[tr])
    yp = clf.predict(X[te])
    return {
        "seed": seed,
        "macro_f1": float(f1_score(y[te], yp, average="macro",
                                     zero_division=0)),
        "accuracy": float(accuracy_score(y[te], yp)),
        "per_class_f1": f1_score(y[te], yp, average=None,
                                  labels=list(range(17)), zero_division=0),
    }


def stratified_subsample(y, n, seed):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    (idx, _), = sss.split(np.zeros(len(y)), y)
    return np.sort(idx)


def evaluate_kernel(name: str, K: np.ndarray, y: np.ndarray) -> Dict:
    log.info("\n--- %s ---  K shape=%s, N=%d", name, K.shape, len(y))

    diag_min = float(K.diagonal().min())
    diag_max = float(K.diagonal().max())
    sym = float(np.max(np.abs(K - K.T)))
    sm = spectral_metrics(K)
    n = K.shape[0]
    off = K[~np.eye(n, dtype=bool)]
    within_1s = float(np.mean(np.abs(off - off.mean()) < off.std()))
    log.info("    diag range : %.4f .. %.4f   sym_err=%.2e",
             diag_min, diag_max, sym)
    log.info("    off-diag   : mean=%.4f var=%.4f std=%.4f within1sig=%.4f",
             sm["off_diag_mean"], sm["off_diag_var"],
             float(np.sqrt(sm["off_diag_var"])), within_1s)
    log.info("    eff rank   : Shannon=%.2f  Huang=%.2f",
             sm["eff_rank_shannon"], sm["eff_rank_huang"])

    kta_sc = kta_same_class(K, y)
    kta_pl = kta_product_labels(K, y)
    log.info("    KTA        : same-class=%.4f  product>0=%.4f",
             kta_sc, kta_pl)

    seed_results = []
    for seed in SEEDS:
        r = eval_kernel_one_seed(K, y, seed)
        seed_results.append(r)
        log.info("    seed=%d   macro_F1=%.4f  acc=%.4f  best_C=%d",
                 seed, r["macro_f1"], r["accuracy"], r["best_C"])

    f1 = np.array([r["macro_f1"] for r in seed_results])
    acc = np.array([r["accuracy"] for r in seed_results])
    pc = np.stack([r["per_class_f1"] for r in seed_results], axis=0)

    log.info("    >>> macro_F1 = %.4f +- %.4f    acc = %.4f +- %.4f",
             f1.mean(), f1.std(ddof=1), acc.mean(), acc.std(ddof=1))

    return {
        "name": name,
        "n_samples": int(n),
        "diag_min": diag_min, "diag_max": diag_max, "sym_err": sym,
        "off_diag_mean": float(sm["off_diag_mean"]),
        "off_diag_var": float(sm["off_diag_var"]),
        "off_diag_std": float(np.sqrt(sm["off_diag_var"])),
        "within_1sigma": within_1s,
        "eff_rank_shannon": float(sm["eff_rank_shannon"]),
        "eff_rank_huang": float(sm["eff_rank_huang"]),
        "kta_same_class": float(kta_sc),
        "kta_product_labels": float(kta_pl),
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_std": float(f1.std(ddof=1)),
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1)),
        "per_seed": [
            {"seed": r["seed"], "macro_f1": r["macro_f1"],
             "accuracy": r["accuracy"], "best_C": r["best_C"]}
            for r in seed_results
        ],
        "per_class_f1_mean": pc.mean(axis=0).tolist(),
        "per_class_f1_std":  pc.std(axis=0, ddof=1).tolist(),
    }


def evaluate_rf(name: str, X: np.ndarray, y: np.ndarray) -> Dict:
    log.info("\n--- %s ---  X shape=%s, N=%d", name, X.shape, len(y))
    seed_results = []
    for seed in SEEDS:
        r = eval_rf_one_seed(X, y, seed)
        seed_results.append(r)
        log.info("    seed=%d   macro_F1=%.4f  acc=%.4f",
                 seed, r["macro_f1"], r["accuracy"])
    f1 = np.array([r["macro_f1"] for r in seed_results])
    acc = np.array([r["accuracy"] for r in seed_results])
    pc = np.stack([r["per_class_f1"] for r in seed_results], axis=0)
    log.info("    >>> macro_F1 = %.4f +- %.4f    acc = %.4f +- %.4f",
             f1.mean(), f1.std(ddof=1), acc.mean(), acc.std(ddof=1))
    return {
        "name": name,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_std": float(f1.std(ddof=1)),
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1)),
        "per_seed": [
            {"seed": r["seed"], "macro_f1": r["macro_f1"],
             "accuracy": r["accuracy"]}
            for r in seed_results
        ],
        "per_class_f1_mean": pc.mean(axis=0).tolist(),
        "per_class_f1_std":  pc.std(axis=0, ddof=1).tolist(),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    log.info(_sep())
    log.info(" BSCM 14q vs 8q-uniform vs 8q-phi_only vs Random Forest")
    log.info(" Started: %s", datetime.now().isoformat())
    log.info(_sep())

    # ----- Load features and eval pool ---------------------------------- #
    X_norm, X_raw, y_full = load_physics_16()
    sp = make_or_load_split(y_full)
    eval_pool = sp.eval_pool       # 1800 indices

    X_norm_eval = X_norm[eval_pool]
    X_raw_eval = X_raw[eval_pool]
    y_eval = y_full[eval_pool]
    log.info(" Eval pool: %d samples", len(y_eval))

    # ----- Fisher-select 8 features (matches the 8q kernel build) -------- #
    sel_idx, fisher = select_features_by_fisher(X_raw_eval, y_eval)
    sel_names = [FEATURE_NAMES_16[i] for i in sel_idx]
    log.info(" Physics-8 (Fisher) features: %s", sel_names)
    X_phys8_1800 = X_norm_eval[:, sel_idx]      # for RF on N=1800
    X_phys8_raw_1800 = X_raw_eval[:, sel_idx]   # raw, for RF (RF is scale-free)

    # ----- 14q test sub-sample (matches bscm_14q_test exactly) ----------- #
    sub_idx_800 = stratified_subsample(y_eval, N_SAMPLES_14Q,
                                        SUB_SAMPLE_SEED_14Q)
    y_800 = y_eval[sub_idx_800]
    # 14q used the top-14 Fisher features (drop 2 lowest); for an honest
    # RF baseline at N=800 we use the SAME 14 features the kernel saw.
    fisher_full = compute_fisher_ratio(X_raw_eval, y_eval)
    drop = list(np.argsort(fisher_full)[:2])
    keep14 = sorted(i for i in range(16) if i not in drop)
    X_phys14_raw_800 = X_raw_eval[sub_idx_800][:, keep14]
    # And ALSO RF on the same physics-8 sub-sample for an apples-to-apples
    # vs the 8q kernels at the same N=800 sample.
    X_phys8_raw_800 = X_raw_eval[sub_idx_800][:, sel_idx]

    log.info(_sep())
    log.info(" Loading kernels and evaluating")
    log.info(_sep())

    # ----- A. 8q uniform (N=1800) --------------------------------------- #
    K_8u = np.load(KERNEL_8Q_UNIFORM)
    res_8u = evaluate_kernel("BSCM-uniform 8q (N=1800)", K_8u, y_eval)

    # ----- B. 8q phi_only (N=1800) -------------------------------------- #
    K_8p = np.load(KERNEL_8Q_PHI)
    res_8p = evaluate_kernel("BSCM-phi_only 8q (N=1800)", K_8p, y_eval)

    # ----- C. 14q uniform (N=800) --------------------------------------- #
    K_14 = np.load(KERNEL_14Q)
    res_14 = evaluate_kernel("BSCM-uniform 14q (N=800)", K_14, y_800)

    # ----- D1. RF on physics-8, N=1800 ---------------------------------- #
    log.info(_sep())
    log.info(" Random Forest baselines (same features, same splits)")
    log.info(_sep())
    res_rf_1800 = evaluate_rf("RF physics-8 (N=1800)",
                                X_phys8_raw_1800, y_eval)

    # ----- D2. RF on physics-8, N=800 (matched to 14q sub-sample) ------- #
    res_rf_800 = evaluate_rf("RF physics-8 (N=800, same sub-sample as 14q)",
                                X_phys8_raw_800, y_800)

    # ----- D3. RF on physics-14, N=800 (matched to 14q kernel) ---------- #
    res_rf14_800 = evaluate_rf("RF physics-14 (N=800, same as 14q kernel)",
                                X_phys14_raw_800, y_800)

    # ====================================================================
    # COMPARISON TABLE
    # ====================================================================
    rows = []
    for r in (res_8u, res_8p, res_14):
        rows.append({
            "method": r["name"], "N": r["n_samples"],
            "macro_F1": f"{r['macro_f1_mean']:.4f} +- {r['macro_f1_std']:.4f}",
            "accuracy": f"{r['accuracy_mean']:.4f} +- {r['accuracy_std']:.4f}",
            "KTA_pl":   f"{r['kta_product_labels']:.4f}",
            "KTA_sc":   f"{r['kta_same_class']:.4f}",
            "off_mean": f"{r['off_diag_mean']:.4f}",
            "off_var":  f"{r['off_diag_var']:.4f}",
            "within1s": f"{r['within_1sigma']:.4f}",
            "eff_rank_S": f"{r['eff_rank_shannon']:.2f}",
        })
    for r in (res_rf_1800, res_rf_800, res_rf14_800):
        rows.append({
            "method": r["name"], "N": r["n_samples"],
            "macro_F1": f"{r['macro_f1_mean']:.4f} +- {r['macro_f1_std']:.4f}",
            "accuracy": f"{r['accuracy_mean']:.4f} +- {r['accuracy_std']:.4f}",
            "KTA_pl": "-", "KTA_sc": "-", "off_mean": "-",
            "off_var": "-", "within1s": "-", "eff_rank_S": "-",
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "comparison_table.csv"), index=False)

    log.info("\n%s", _sep())
    log.info(" COMPARISON TABLE")
    log.info(_sep())
    log.info("%s", df.to_string(index=False))

    # ----- F1 deltas / honest verdict ----------------------------------- #
    log.info("\n%s", _sep())
    log.info(" HEAD-TO-HEAD DELTAS (macro F1)")
    log.info(_sep())

    log.info(" 8q-phi_only - 8q-uniform = %+.4f   (do non-XX/YY sectors help?)",
             res_8p["macro_f1_mean"] - res_8u["macro_f1_mean"])
    log.info(" 14q@N=800   - 8q-uniform@N=1800 = %+.4f   (the 14q decision)",
             res_14["macro_f1_mean"] - res_8u["macro_f1_mean"])
    log.info(" 14q@N=800   - RF phys-14@N=800  = %+.4f   (14q vs RF, matched)",
             res_14["macro_f1_mean"] - res_rf14_800["macro_f1_mean"])
    log.info(" 8q-uniform  - RF phys-8 @N=1800 = %+.4f   (8q vs RF, matched)",
             res_8u["macro_f1_mean"] - res_rf_1800["macro_f1_mean"])
    log.info(" 8q-phi_only - RF phys-8 @N=1800 = %+.4f   (phi_only vs RF, matched)",
             res_8p["macro_f1_mean"] - res_rf_1800["macro_f1_mean"])
    log.info(" 14q@N=800   - RF phys-8 @N=800  = %+.4f   (14q vs RF on smaller feature set)",
             res_14["macro_f1_mean"] - res_rf_800["macro_f1_mean"])

    # ----- per-class side-by-side --------------------------------------- #
    log.info("\n%s", _sep())
    log.info(" PER-CLASS F1 (mean over 5 seeds)")
    log.info(_sep())
    log.info("  %3s  %-22s  %8s  %8s  %8s  %8s  %8s  %8s",
             "Cls", "Name", "8q_uni", "8q_phi", "14q",
             "RF_p8_1800", "RF_p8_800", "RF_p14_800")
    log.info("  " + "-" * 100)
    rows2 = []
    for c in range(17):
        log.info("  %3d  %-22s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f",
                 c, LCZ[c],
                 res_8u["per_class_f1_mean"][c],
                 res_8p["per_class_f1_mean"][c],
                 res_14["per_class_f1_mean"][c],
                 res_rf_1800["per_class_f1_mean"][c],
                 res_rf_800["per_class_f1_mean"][c],
                 res_rf14_800["per_class_f1_mean"][c])
        rows2.append({
            "class": c, "name": LCZ[c],
            "f1_8q_uniform": res_8u["per_class_f1_mean"][c],
            "f1_8q_phi_only": res_8p["per_class_f1_mean"][c],
            "f1_14q_N800": res_14["per_class_f1_mean"][c],
            "f1_RF_phys8_N1800": res_rf_1800["per_class_f1_mean"][c],
            "f1_RF_phys8_N800":  res_rf_800["per_class_f1_mean"][c],
            "f1_RF_phys14_N800": res_rf14_800["per_class_f1_mean"][c],
        })
    pd.DataFrame(rows2).to_csv(
        os.path.join(SAVE_DIR, "per_class_f1_compare.csv"), index=False
    )

    # ----- save full summary -------------------------------------------- #
    summary = {
        "date": datetime.now().isoformat(),
        "feature_set_8q": sel_names,
        "n_features_14q_keep": 14,
        "kernels": {
            "8q_uniform_N1800": res_8u,
            "8q_phi_only_N1800": res_8p,
            "14q_uniform_N800":  res_14,
        },
        "rf_baselines": {
            "rf_physics8_N1800": res_rf_1800,
            "rf_physics8_N800":  res_rf_800,
            "rf_physics14_N800": res_rf14_800,
        },
    }
    with open(os.path.join(SAVE_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("\nSaved: %s", SAVE_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
