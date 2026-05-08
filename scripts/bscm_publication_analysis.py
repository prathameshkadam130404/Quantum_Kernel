"""
Publication-grade analysis pack for the BSCM contribution.

Implements the MLST-targeted analyses requested by reviewer-anticipating
restructure:

  T1a  Wilcoxon signed-rank tests for BSCM-phi_only vs (RBF, RF, SRQFM,
       BSCM-uniform) over 5 seeds, with Holm-Bonferroni multiple-testing
       correction.
  T1b  Bootstrap (B=1000) per-class F1 confidence intervals for
       BSCM-phi_only and the matched RF baseline; per-class deltas with
       95% CI on the difference.
  T1c  Permutation KTA significance test (n=1000 random label shuffles)
       for BSCM-phi_only and BSCM-uniform.
  T1d  Concentration metrics for BSCM-uniform and BSCM-phi_only at n=8
       on N=500 sub-samples of the cached N=1800 Gram matrices, slotting
       into the E28 framework. The 14-qubit point is loaded from the
       existing bscm_14q_test cache.
  T2c  Holm-Bonferroni-corrected significance table for the full
       comparison (all method-pair Wilcoxon).

All analyses use already-cached kernels and seed predictions; no new
quantum simulations are run.

Outputs (under results/bscm_publication_analysis/):
  wilcoxon_table.csv          T1a + T2c
  perm_kta_table.csv          T1c
  per_class_bootstrap_ci.csv  T1b
  bscm_concentration.csv      T1d
  summary.json                consolidated
  run.log
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
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
# Paths / config
# --------------------------------------------------------------------------- #
SAVE_DIR = os.path.join(config.RESULTS_DIR, "bscm_publication_analysis")
os.makedirs(SAVE_DIR, exist_ok=True)

KERNEL_8Q_UNIFORM  = os.path.join(config.RESULTS_DIR, "bscm",
                                  "K_bscm_physics8_uniform.npy")
KERNEL_8Q_PHI      = os.path.join(config.RESULTS_DIR, "bscm",
                                  "K_bscm_physics8_phi_only.npy")
KERNEL_14Q         = os.path.join(config.RESULTS_DIR, "bscm_14q_test",
                                  "K_bscm_14q_phys14_uniform_tau0.25_N800.npy")
FAIR_SRQFM_CSV     = os.path.join(config.RESULTS_DIR, "fair_srqfm",
                                  "metrics.csv")

SEEDS    = config.SEED_LIST
C_GRID   = [1, 10, 100]
CV_FOLDS = 3
TEST_FRAC = 0.30
N_BOOT   = 1000
N_PERM   = 1000
N_SAMPLES_14Q = 800
SUB_SAMPLE_SEED_14Q = 1234
N_E28 = 500          # to slot into E28 sub-sample
SEED_E28 = 42        # E28 seed

LCZ = config.LCZ_CLASS_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "run.log"), mode="w"),
    ],
)
log = logging.getLogger("bscm_pub_analysis")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _sep(c="=", w=78):
    return c * w


def kta_product(K: np.ndarray, y: np.ndarray) -> float:
    """Product-label KTA (matches paper E22/E26 convention)."""
    n = len(y)
    Y = np.outer(y, y)
    Y = (Y > 0).astype(float) * 2.0 - 1.0
    Ksym = (K + K.T) / 2.0
    num = float(np.sum(Ksym * Y)) / (n * n)
    den = np.sqrt(float(np.sum(Ksym * Ksym))
                  * float(np.sum(Y * Y))) / (n * n)
    return num / (den + 1e-12)


def _build_Y_class_weighted(y: np.ndarray) -> np.ndarray:
    """Vectorised class-weighted ideal kernel: 1/n_c if same class else 0."""
    cls, cnt = np.unique(y, return_counts=True)
    inv = np.zeros(int(cls.max()) + 1, dtype=np.float64)
    inv[cls] = 1.0 / cnt
    same = (y[:, None] == y[None, :])
    return same.astype(np.float64) * inv[y][:, None]


def kta_class_centered(K: np.ndarray, y: np.ndarray,
                        Kc_pre: np.ndarray = None) -> float:
    """Class-weighted, centered KTA (the principled multi-class form).
    If Kc_pre is provided, treat it as already centered (saves H @ K @ H)."""
    n = len(y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = Kc_pre if Kc_pre is not None else H @ ((K + K.T) / 2.0) @ H
    Y_id = _build_Y_class_weighted(y)
    Yc = H @ Y_id @ H
    num = float(np.sum(Kc * Yc))
    den = float(np.linalg.norm(Kc, ord="fro") *
                np.linalg.norm(Yc, ord="fro")) + 1e-12
    return num / den


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


def eval_kernel_with_preds(K_full, y, seed):
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
        "y_test": y[te].copy(), "y_pred": yp.copy(),
        "macro_f1": float(f1_score(y[te], yp, average="macro",
                                     zero_division=0)),
        "accuracy": float(accuracy_score(y[te], yp)),
    }


def eval_rf_with_preds(X, y, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                                  random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)
    clf = RandomForestClassifier(
        n_estimators=500, max_features="sqrt",
        class_weight="balanced", n_jobs=-1, random_state=seed,
    )
    clf.fit(X[tr], y[tr])
    yp = clf.predict(X[te])
    return {
        "seed": seed,
        "y_test": y[te].copy(), "y_pred": yp.copy(),
        "macro_f1": float(f1_score(y[te], yp, average="macro",
                                     zero_division=0)),
        "accuracy": float(accuracy_score(y[te], yp)),
    }


def stratified_subsample(y, n, seed):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    (idx, _), = sss.split(np.zeros(len(y)), y)
    return np.sort(idx)


def holm_bonferroni(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Return list of booleans: True iff null is rejected after Holm correction."""
    m = len(p_values)
    order = np.argsort(p_values)
    reject = [False] * m
    for rank, idx in enumerate(order):
        thr = alpha / (m - rank)
        if p_values[idx] <= thr:
            reject[idx] = True
        else:
            break
    return reject


def perm_kta_pvalue(K: np.ndarray, y: np.ndarray, n_perm: int,
                     rng_seed: int = 0,
                     subsample_n: int = 500) -> Dict:
    """Compute permutation p-value for KTA(K, y) under random label shuffles.
    For tractability, computes the test on a stratified sub-sample of size
    `subsample_n` if K is larger; the sub-sample is fixed across all perms."""
    n_full = len(y)
    if subsample_n is not None and n_full > subsample_n:
        # Stratified sub-sample, deterministic
        sss = StratifiedShuffleSplit(n_splits=1, train_size=subsample_n,
                                      random_state=rng_seed)
        (idx, _), = sss.split(np.zeros(n_full), y)
        idx = np.sort(idx)
        K = K[np.ix_(idx, idx)]
        y = y[idx]
    n = len(y)
    Ksym = (K + K.T) / 2.0
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ Ksym @ H
    Kc_norm = np.linalg.norm(Kc, ord="fro") + 1e-12

    obs_pl = kta_product(K, y)
    obs_cc = kta_class_centered(K, y, Kc_pre=Kc)

    rng = np.random.default_rng(rng_seed)
    perm_pl = np.empty(n_perm)
    perm_cc = np.empty(n_perm)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        # Product KTA — fully vectorised
        Yp = np.outer(y_perm, y_perm)
        Yp = (Yp > 0).astype(np.float64) * 2.0 - 1.0
        num = float(np.sum(Ksym * Yp)) / (n * n)
        den = (np.sqrt(float(np.sum(Ksym * Ksym))
                        * float(np.sum(Yp * Yp)))) / (n * n)
        perm_pl[i] = num / (den + 1e-12)
        # Class-centered — vectorised
        Y_id = _build_Y_class_weighted(y_perm)
        Yc = H @ Y_id @ H
        perm_cc[i] = float(np.sum(Kc * Yc)) / (Kc_norm
                            * (np.linalg.norm(Yc, ord="fro") + 1e-12))

    p_pl = float((np.sum(perm_pl >= obs_pl) + 1) / (n_perm + 1))
    p_cc = float((np.sum(perm_cc >= obs_cc) + 1) / (n_perm + 1))
    return {
        "n_eval": int(n),
        "kta_product": obs_pl,
        "kta_product_pvalue": p_pl,
        "kta_product_null_mean": float(perm_pl.mean()),
        "kta_product_null_std":  float(perm_pl.std(ddof=1)),
        "kta_class_centered": obs_cc,
        "kta_class_centered_pvalue": p_cc,
        "kta_class_centered_null_mean": float(perm_cc.mean()),
        "kta_class_centered_null_std":  float(perm_cc.std(ddof=1)),
    }


def bootstrap_per_class_f1(y_test_per_seed: List[np.ndarray],
                            y_pred_per_seed: List[np.ndarray],
                            n_boot: int, n_classes: int,
                            seed: int = 0) -> Dict:
    """
    Pool predictions across seeds, then bootstrap (sample test points with
    replacement, n_boot times) to get 95% CI on per-class F1.
    """
    y_t = np.concatenate(y_test_per_seed)
    y_p = np.concatenate(y_pred_per_seed)
    n = len(y_t)
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, n_classes))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = f1_score(y_t[idx], y_p[idx], average=None,
                              labels=list(range(n_classes)), zero_division=0)
    point = f1_score(y_t, y_p, average=None,
                      labels=list(range(n_classes)), zero_division=0)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return {"point": point, "lo": lo, "hi": hi, "boots": boots}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    log.info(_sep())
    log.info(" BSCM Publication-Grade Analysis Pack")
    log.info(" %s", datetime.now().isoformat())
    log.info(_sep())

    # ----- Load features & labels --------------------------------------- #
    X_norm, X_raw, y_full = load_physics_16()
    sp = make_or_load_split(y_full)
    eval_pool = sp.eval_pool
    X_norm_eval = X_norm[eval_pool]
    X_raw_eval = X_raw[eval_pool]
    y_eval = y_full[eval_pool]
    sel_idx, fisher = select_features_by_fisher(X_raw_eval, y_eval)
    sel_names = [FEATURE_NAMES_16[i] for i in sel_idx]
    log.info(" Eval pool N=%d, physics-Fisher-8 features: %s",
             len(y_eval), sel_names)

    # ----- 14q sub-sample ---------------------------------------------- #
    sub_idx_800 = stratified_subsample(y_eval, N_SAMPLES_14Q,
                                        SUB_SAMPLE_SEED_14Q)
    y_800 = y_eval[sub_idx_800]
    fisher_full = compute_fisher_ratio(X_raw_eval, y_eval)
    drop2 = list(np.argsort(fisher_full)[:2])
    keep14 = sorted(i for i in range(16) if i not in drop2)
    X_phys14_raw_800 = X_raw_eval[sub_idx_800][:, keep14]
    X_phys8_raw_1800 = X_raw_eval[:, sel_idx]
    X_phys8_raw_800  = X_raw_eval[sub_idx_800][:, sel_idx]

    # =================================================================== #
    # Re-run kernels with predictions saved (T1a, T1b, T2c need preds)
    # =================================================================== #
    log.info("\n%s", _sep())
    log.info(" Re-evaluating kernels with predictions saved (5 seeds each)")
    log.info(_sep())

    log.info("\n[8q-uniform]")
    K_8u = np.load(KERNEL_8Q_UNIFORM)
    res_8u = [eval_kernel_with_preds(K_8u, y_eval, s) for s in SEEDS]

    log.info("\n[8q-phi_only]")
    K_8p = np.load(KERNEL_8Q_PHI)
    res_8p = [eval_kernel_with_preds(K_8p, y_eval, s) for s in SEEDS]

    log.info("\n[14q-uniform N=800]")
    K_14 = np.load(KERNEL_14Q)
    res_14 = [eval_kernel_with_preds(K_14, y_800, s) for s in SEEDS]

    log.info("\n[RF physics-8 N=1800]")
    res_rf_1800 = [eval_rf_with_preds(X_phys8_raw_1800, y_eval, s)
                    for s in SEEDS]

    log.info("\n[RF physics-14 N=800]")
    res_rf14_800 = [eval_rf_with_preds(X_phys14_raw_800, y_800, s)
                     for s in SEEDS]

    log.info("\n[RF physics-8 N=800]")
    res_rf8_800 = [eval_rf_with_preds(X_phys8_raw_800, y_800, s)
                    for s in SEEDS]

    # SRQFM seed values from cached fair_srqfm metrics ---------- #
    fair = pd.read_csv(FAIR_SRQFM_CSV)
    srqfm_seeds = fair[fair["method"] == "SRQFM-PQK"].sort_values("seed")
    srqfm_f1 = srqfm_seeds["macro_f1"].to_numpy()
    rbf_seeds = fair[fair["method"] == "RBF-SVM"].sort_values("seed")
    rbf_f1_e26 = rbf_seeds["macro_f1"].to_numpy()
    rf_seeds_e26 = fair[fair["method"] == "RandomForest"].sort_values("seed")
    rf_f1_e26 = rf_seeds_e26["macro_f1"].to_numpy()
    zz_seeds = fair[fair["method"] == "ZZ-PQK"].sort_values("seed")
    zz_f1 = zz_seeds["macro_f1"].to_numpy()

    f1 = {
        "BSCM-phi_only":  np.array([r["macro_f1"] for r in res_8p]),
        "BSCM-uniform":   np.array([r["macro_f1"] for r in res_8u]),
        "BSCM-14q":       np.array([r["macro_f1"] for r in res_14]),
        "RF_phys8_N1800": np.array([r["macro_f1"] for r in res_rf_1800]),
        "RF_phys14_N800": np.array([r["macro_f1"] for r in res_rf14_800]),
        "RF_phys8_N800":  np.array([r["macro_f1"] for r in res_rf8_800]),
        "SRQFM-PQK":      srqfm_f1,
        "RBF-SVM (E26)":  rbf_f1_e26,
        "RF (E26)":       rf_f1_e26,
        "ZZ-PQK":         zz_f1,
    }
    log.info("\n  Mean F1 over 5 seeds:")
    for k, v in f1.items():
        log.info("    %-22s %.4f +- %.4f", k, v.mean(), v.std(ddof=1))

    # =================================================================== #
    # T1a + T2c: Wilcoxon + Holm-Bonferroni
    # =================================================================== #
    log.info("\n%s", _sep())
    log.info(" T1a/T2c: Wilcoxon + Holm-Bonferroni")
    log.info(_sep())

    # Pairs to test (BSCM-phi_only as anchor + key cross-checks)
    pairs = [
        ("BSCM-phi_only", "RBF-SVM (E26)"),
        ("BSCM-phi_only", "RF (E26)"),
        ("BSCM-phi_only", "SRQFM-PQK"),
        ("BSCM-phi_only", "BSCM-uniform"),
        ("BSCM-phi_only", "ZZ-PQK"),
        ("BSCM-uniform",  "ZZ-PQK"),
        ("BSCM-uniform",  "RF_phys8_N1800"),
        ("BSCM-14q",      "RF_phys14_N800"),
        ("SRQFM-PQK",     "RBF-SVM (E26)"),
    ]

    rows = []
    raw_p = []
    for a, b in pairs:
        diff = f1[a] - f1[b]
        # zero_method='zsplit' handles ties; alternative two-sided
        try:
            stat, p = wilcoxon(f1[a], f1[b], zero_method="zsplit",
                                alternative="two-sided")
            p = float(p)
        except Exception:
            p = float("nan")
        rows.append({"A": a, "B": b,
                      "mean_A": float(f1[a].mean()),
                      "mean_B": float(f1[b].mean()),
                      "delta_mean": float(diff.mean()),
                      "p_raw": p})
        raw_p.append(p)

    # Holm-Bonferroni
    rejects = holm_bonferroni([r["p_raw"] for r in rows], alpha=0.05)
    # Adjusted p-values (Holm step-down): p_adj = max over ranks
    order = np.argsort([r["p_raw"] for r in rows])
    p_adj = [None] * len(rows)
    running = 0.0
    m = len(rows)
    for rank, idx in enumerate(order):
        adj = (m - rank) * rows[idx]["p_raw"]
        adj = min(1.0, max(running, adj))
        running = adj
        p_adj[idx] = adj
    for i, r in enumerate(rows):
        r["p_holm"] = float(p_adj[i])
        r["reject_holm_05"] = bool(rejects[i])

    df_w = pd.DataFrame(rows)
    df_w.to_csv(os.path.join(SAVE_DIR, "wilcoxon_table.csv"), index=False)
    log.info("\n%s", df_w.to_string(index=False))

    # =================================================================== #
    # T1c: Permutation KTA significance
    # =================================================================== #
    log.info("\n%s", _sep())
    log.info(" T1c: Permutation KTA test (n_perm=%d)", N_PERM)
    log.info(_sep())

    perm_rows = []
    for name, K, y in [
        ("BSCM-phi_only 8q (N=1800)", K_8p, y_eval),
        ("BSCM-uniform 8q  (N=1800)", K_8u, y_eval),
        ("BSCM-uniform 14q (N=800)",  K_14, y_800),
    ]:
        log.info("  %s ...", name)
        r = perm_kta_pvalue(K, y, N_PERM, rng_seed=0)
        r["name"] = name
        perm_rows.append(r)
        log.info("    KTA(prod)=%.4f  null=%.4f+-%.4f  p=%.4f",
                 r["kta_product"], r["kta_product_null_mean"],
                 r["kta_product_null_std"], r["kta_product_pvalue"])
        log.info("    KTA(class-centered)=%.4f  null=%.4f+-%.4f  p=%.4f",
                 r["kta_class_centered"], r["kta_class_centered_null_mean"],
                 r["kta_class_centered_null_std"],
                 r["kta_class_centered_pvalue"])

    df_p = pd.DataFrame(perm_rows)
    df_p.to_csv(os.path.join(SAVE_DIR, "perm_kta_table.csv"), index=False)

    # =================================================================== #
    # T1b: Bootstrap per-class F1 confidence intervals
    # =================================================================== #
    log.info("\n%s", _sep())
    log.info(" T1b: Bootstrap per-class F1 95%% CI (B=%d)", N_BOOT)
    log.info(_sep())

    methods_pred = [
        ("BSCM-phi_only_N1800",  res_8p,        17),
        ("BSCM-uniform_N1800",   res_8u,        17),
        ("BSCM-14q_N800",        res_14,        17),
        ("RF_phys8_N1800",       res_rf_1800,   17),
        ("RF_phys14_N800",       res_rf14_800,  17),
    ]
    boot_records = []
    for name, res, ncls in methods_pred:
        y_t_seeds = [r["y_test"] for r in res]
        y_p_seeds = [r["y_pred"] for r in res]
        b = bootstrap_per_class_f1(y_t_seeds, y_p_seeds, N_BOOT, ncls, seed=0)
        for c in range(ncls):
            boot_records.append({
                "method": name, "class": c, "name": LCZ[c],
                "f1_point": float(b["point"][c]),
                "f1_lo95":  float(b["lo"][c]),
                "f1_hi95":  float(b["hi"][c]),
            })
    df_b = pd.DataFrame(boot_records)
    df_b.to_csv(os.path.join(SAVE_DIR, "per_class_bootstrap_ci.csv"),
                 index=False)
    log.info("  Saved bootstrap CIs for %d (method, class) cells",
             len(boot_records))

    # Per-class delta CI: BSCM-phi_only vs RF_phys8_N1800 (paired bootstrap)
    log.info("\n  Paired delta bootstrap: BSCM-phi_only vs RF_phys8_N1800")
    rng = np.random.default_rng(123)
    y_t_phi = np.concatenate([r["y_test"] for r in res_8p])
    y_p_phi = np.concatenate([r["y_pred"] for r in res_8p])
    y_t_rf  = np.concatenate([r["y_test"] for r in res_rf_1800])
    y_p_rf  = np.concatenate([r["y_pred"] for r in res_rf_1800])
    # The two methods used the SAME StratifiedShuffleSplit seeds and so the
    # test-index sets are identical per seed. We therefore align by seed.
    deltas = np.empty((N_BOOT, 17))
    for b_i in range(N_BOOT):
        # Per-seed bootstrap, then concat — preserves pairing
        f1_phi_b = []
        f1_rf_b  = []
        for r_phi, r_rf in zip(res_8p, res_rf_1800):
            n = len(r_phi["y_test"])
            ix = rng.integers(0, n, n)
            f1_phi_b.append(f1_score(r_phi["y_test"][ix],
                                       r_phi["y_pred"][ix],
                                       average=None,
                                       labels=list(range(17)),
                                       zero_division=0))
            f1_rf_b.append(f1_score(r_rf["y_test"][ix],
                                      r_rf["y_pred"][ix],
                                      average=None,
                                      labels=list(range(17)),
                                      zero_division=0))
        deltas[b_i] = np.mean(f1_phi_b, axis=0) - np.mean(f1_rf_b, axis=0)
    delta_lo = np.percentile(deltas, 2.5, axis=0)
    delta_hi = np.percentile(deltas, 97.5, axis=0)
    delta_pt = deltas.mean(axis=0)
    delta_records = []
    for c in range(17):
        delta_records.append({
            "class": c, "name": LCZ[c],
            "delta_point": float(delta_pt[c]),
            "delta_lo95":  float(delta_lo[c]),
            "delta_hi95":  float(delta_hi[c]),
            "ci_excludes_zero":
                bool((delta_lo[c] > 0) or (delta_hi[c] < 0)),
        })
    df_d = pd.DataFrame(delta_records)
    df_d.to_csv(os.path.join(SAVE_DIR, "per_class_delta_phi_vs_rf.csv"),
                 index=False)
    log.info("\n  Per-class delta (phi_only - RF_p8_N1800), 95%% CI:")
    log.info("  %s", df_d.to_string(index=False))

    # =================================================================== #
    # T1d: BSCM concentration metrics at n=8 (free, from cached kernels)
    # =================================================================== #
    log.info("\n%s", _sep())
    log.info(" T1d: BSCM concentration metrics at n=8 (N=500 sub-sample)")
    log.info(_sep())

    # Match E28 sub-sampling: rng = np.random.default_rng(SEED=42), choose
    # N_SAMPLE=500 from len(y) where y = first SUBSAMPLE_TRAIN=2000 of full y.
    # The cached BSCM kernels are computed over the eval_pool indices (1800).
    # For comparability we extract a stratified N=500 from the cached pool
    # using a fixed seed; this matches the spirit of E28 (random N=500).
    sub_idx_500 = stratified_subsample(y_eval, N_E28, SEED_E28)
    y_500 = y_eval[sub_idx_500]
    K_8u_500 = K_8u[np.ix_(sub_idx_500, sub_idx_500)]
    K_8p_500 = K_8p[np.ix_(sub_idx_500, sub_idx_500)]

    def kernel_health(K, y):
        n = K.shape[0]
        off = K[~np.eye(n, dtype=bool)]
        sm = spectral_metrics(K)
        within_1s = float(np.mean(np.abs(off - off.mean()) < off.std()))
        return {
            "N": int(n),
            "off_diag_mean": float(sm["off_diag_mean"]),
            "off_diag_var":  float(sm["off_diag_var"]),
            "off_diag_std":  float(np.sqrt(sm["off_diag_var"])),
            "within_1sigma": within_1s,
            "eff_rank_shannon": float(sm["eff_rank_shannon"]),
            "eff_rank_huang":  float(sm["eff_rank_huang"]),
            "kta_product":     float(kta_product(K, y)),
            "kta_class_centered": float(kta_class_centered(K, y)),
        }

    conc = []
    for name, K, y in [
        ("BSCM-uniform   n=8  (N=500)",  K_8u_500, y_500),
        ("BSCM-phi_only  n=8  (N=500)",  K_8p_500, y_500),
        ("BSCM-uniform   n=8  (N=1800)", K_8u,     y_eval),
        ("BSCM-phi_only  n=8  (N=1800)", K_8p,     y_eval),
        ("BSCM-uniform   n=14 (N=800)",  K_14,     y_800),
    ]:
        h = kernel_health(K, y)
        h["kernel"] = name
        conc.append(h)
        log.info("  %s", name)
        log.info("    off_mean=%.4f var=%.4f within1s=%.4f eff_rk_S=%.2f "
                 "KTA(prod)=%.4f KTA(cc)=%.4f",
                 h["off_diag_mean"], h["off_diag_var"], h["within_1sigma"],
                 h["eff_rank_shannon"], h["kta_product"],
                 h["kta_class_centered"])
    df_c = pd.DataFrame(conc)
    df_c.to_csv(os.path.join(SAVE_DIR, "bscm_concentration.csv"), index=False)

    # =================================================================== #
    # Summary JSON
    # =================================================================== #
    summary = {
        "date": datetime.now().isoformat(),
        "n_seeds": len(SEEDS),
        "n_boot": N_BOOT,
        "n_perm": N_PERM,
        "f1_means": {k: float(v.mean()) for k, v in f1.items()},
        "f1_stds":  {k: float(v.std(ddof=1)) for k, v in f1.items()},
        "wilcoxon": rows,
        "perm_kta": perm_rows,
        "concentration": conc,
    }
    with open(os.path.join(SAVE_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info("\nAll outputs saved to %s", SAVE_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
