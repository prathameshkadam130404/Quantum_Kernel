"""
E26 — Fair SRQFM Benchmark: C-tuned comparison across all methods.

Addresses the confound in E22 where all SVM methods used a fixed C=1.0.
Here every SVM (kernel and classical) has its regularisation parameter C
selected by 3-fold stratified cross-validation on the training split,
over the grid C ∈ {1, 10, 100}.

Methods evaluated
-----------------
    SRQFM-PQK   : fidelity-coupling projected kernel, C-tuned
    ZZ-PQK      : standard Havlicek projected kernel, C-tuned
    AGPQK       : Fisher-weighted projected kernel, C-tuned
    RBF-SVM     : classical RBF, (C, gamma) jointly tuned by GridSearchCV
    RandomForest: n_estimators=500, no tuning (RF is largely insensitive)

Statistical tests
-----------------
    Wilcoxon signed-rank (paired) over SEED_LIST seeds.
    Cohen's d (pooled std) for effect sizes.
    Holm-Bonferroni correction over all pairwise tests vs SRQFM.

Output: results/fair_srqfm/metrics.csv  and  exp_e26.log
"""

import os
import sys
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedShuffleSplit,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._e_common import (
    load_physics_16,
    spectral_metrics,
    cohens_d,
    holm_bonferroni,
    build_agpqk_kernel,
)
from src.attention_kernel import select_features_by_fisher

# ---------------------------------------------------------------------------
SEEDS      = config.SEED_LIST          # [42, 43, 44, 45, 46]
N_QUBITS   = config.N_QUBITS           # 8
ZZ_REPS    = config.ZZ_REPS            # 2
PQK_GAMMA  = config.PQK_GAMMA          # 0.67
TEST_FRAC  = 0.30
C_GRID     = [1, 10, 100]
CV_FOLDS   = 3

SAVE_DIR = os.path.join(config.RESULTS_DIR, "fair_srqfm")
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "exp_e26.log"), mode="w"),
    ],
)
logger = logging.getLogger("exp_e26")


# ---------------------------------------------------------------------------
# Hyperparameter tuning utilities
# ---------------------------------------------------------------------------

def tune_c_precomputed(K_train: np.ndarray, y_train: np.ndarray,
                       c_grid=C_GRID, n_folds=CV_FOLDS, seed=42) -> int:
    """Select C by stratified k-fold CV on a precomputed kernel."""
    best_c, best_f1 = c_grid[0], -1.0
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for C in c_grid:
        fold_f1s = []
        for cv_tr, cv_val in skf.split(np.zeros(len(y_train)), y_train):
            K_cv = K_train[np.ix_(cv_tr, cv_tr)]
            K_cv_val = K_train[np.ix_(cv_val, cv_tr)]
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_cv, y_train[cv_tr])
            yp = clf.predict(K_cv_val)
            fold_f1s.append(f1_score(y_train[cv_val], yp, average="macro",
                                     zero_division=0))
        mean_f1 = float(np.mean(fold_f1s))
        if mean_f1 > best_f1:
            best_f1, best_c = mean_f1, C

    return best_c


def eval_kernel_tuned(K_full: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Split data, tune C, then evaluate.  Returns F1 and best C."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                                 random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)

    K_tr = K_full[np.ix_(tr, tr)]
    K_te = K_full[np.ix_(te, tr)]

    best_c = tune_c_precomputed(K_tr, y[tr], seed=seed)

    clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    yp = clf.predict(K_te)
    f1 = float(f1_score(y[te], yp, average="macro", zero_division=0))

    return {"macro_f1": f1, "best_C": best_c, "train_idx": tr, "test_idx": te}


def eval_rbf_tuned(X_raw: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """RBF-SVM with (C, gamma) tuned by GridSearchCV."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                                 random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)

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
    f1 = float(f1_score(y[te], yp, average="macro", zero_division=0))

    return {
        "macro_f1": f1,
        "best_C": clf.best_params_["C"],
        "best_gamma": clf.best_params_["gamma"],
    }


def eval_rf(X_raw: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Random Forest (n_estimators=500, balanced weights)."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                                 random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)

    clf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    clf.fit(X_raw[tr], y[tr])
    yp = clf.predict(X_raw[te])

    return {"macro_f1": float(f1_score(y[te], yp, average="macro",
                                       zero_division=0))}


# ---------------------------------------------------------------------------
# Kernel builders (cached per experiment run)
# ---------------------------------------------------------------------------

def build_srqfm_kernel(X_enc: np.ndarray) -> np.ndarray:
    from src.srqfm_kernel import compute_srqfm_pqk_kernel

    cache = os.path.join(SAVE_DIR, "K_srqfm.npy")
    if os.path.exists(cache):
        logger.info("[CACHE] Loading SRQFM-PQK kernel")
        return np.load(cache)

    K = compute_srqfm_pqk_kernel(
        X_enc, gamma=PQK_GAMMA, n_qubits=N_QUBITS, reps=ZZ_REPS,
        coupling_threshold=0.01, connectivity="all",
    )
    np.save(cache, K)
    return K


def build_zz_kernel(X_enc: np.ndarray) -> np.ndarray:
    from src.quantum_kernels import compute_pqk_kernel_matrix

    cache = os.path.join(SAVE_DIR, "K_zz.npy")
    if os.path.exists(cache):
        logger.info("[CACHE] Loading ZZ-PQK kernel")
        return np.load(cache)

    K = compute_pqk_kernel_matrix(X_enc, n_qubits=N_QUBITS,
                                  reps=ZZ_REPS, gamma=PQK_GAMMA)
    np.save(cache, K)
    return K


def build_agpqk(X_raw: np.ndarray, X_norm: np.ndarray,
                y: np.ndarray) -> np.ndarray:
    cache = os.path.join(SAVE_DIR, "K_agpqk.npy")
    if os.path.exists(cache):
        logger.info("[CACHE] Loading AGPQK kernel")
        return np.load(cache)

    sel_idx, fisher_scores = select_features_by_fisher(X_raw, y)
    fisher_sel = fisher_scores[sel_idx]

    gamma_base = 0.50
    gamma_pq = gamma_base * (fisher_sel / fisher_sel.mean())
    gamma_pq = np.clip(gamma_pq, 0.20, 1.50)

    from src.attention_kernel import FEATURE_MODALITY_16
    modalities = [FEATURE_MODALITY_16[i] for i in sel_idx]
    pairs = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)
             if modalities[i] != modalities[j]][:4]

    K, _ = build_agpqk_kernel(
        X_raw, y, sel_idx, gamma_pq, pairs,
        n_qubits=N_QUBITS, reps=ZZ_REPS, outer_gamma=PQK_GAMMA,
    )
    np.save(cache, K)
    return K


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logger.info("=" * 70)
    logger.info("  E26: Fair SRQFM Benchmark (C-tuned)")
    logger.info("=" * 70)
    logger.info(f"  Date:  {datetime.now().isoformat()}")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  C grid: {C_GRID}  (3-fold CV on training split)")

    # Load data
    X_norm, X_raw, y = load_physics_16()
    logger.info(f"  Data: N={len(y)}, features={X_norm.shape[1]}, "
                f"classes={len(np.unique(y))}")

    sel_idx, _ = select_features_by_fisher(X_raw, y)
    X_sel = X_norm[:, sel_idx]   # Encoded features for quantum kernels

    # Phase 1: Build full kernel matrices (computed once, shared across seeds)
    logger.info("\n--- Phase 1: Kernel matrices ---")

    t0 = time.time()
    logger.info("  Building SRQFM-PQK...")
    K_srqfm = build_srqfm_kernel(X_sel)
    logger.info(f"    {time.time()-t0:.1f}s")

    t0 = time.time()
    logger.info("  Building ZZ-PQK...")
    K_zz = build_zz_kernel(X_sel)
    logger.info(f"    {time.time()-t0:.1f}s")

    t0 = time.time()
    logger.info("  Building AGPQK...")
    K_agpqk = build_agpqk(X_raw, X_norm, y)
    logger.info(f"    {time.time()-t0:.1f}s")

    # Spectral diagnostics
    for name, K in [("SRQFM-PQK", K_srqfm), ("ZZ-PQK", K_zz),
                    ("AGPQK", K_agpqk)]:
        sm = spectral_metrics(K)
        logger.info(f"  {name}: off_diag_mean={sm['off_diag_mean']:.4f}  "
                    f"eff_rank={sm['eff_rank_shannon']:.1f}  "
                    f"off_diag_var={sm['off_diag_var']:.5f}")

    # Phase 2: Evaluate all methods across seeds
    logger.info("\n--- Phase 2: Seed-level evaluation (C-tuned) ---")

    rows = []
    for seed in SEEDS:
        logger.info(f"\n  Seed {seed}:")

        for name, K in [("SRQFM-PQK", K_srqfm), ("ZZ-PQK", K_zz),
                        ("AGPQK", K_agpqk)]:
            r = eval_kernel_tuned(K, y, seed)
            rows.append({"method": name, "seed": seed,
                         "macro_f1": r["macro_f1"], "best_C": r["best_C"]})
            logger.info(f"    {name:<12} F1={r['macro_f1']:.4f}  C={r['best_C']}")

        r_rbf = eval_rbf_tuned(X_raw, y, seed)
        rows.append({"method": "RBF-SVM", "seed": seed,
                     "macro_f1": r_rbf["macro_f1"],
                     "best_C": r_rbf["best_C"],
                     "best_gamma": r_rbf.get("best_gamma")})
        logger.info(f"    {'RBF-SVM':<12} F1={r_rbf['macro_f1']:.4f}  "
                    f"C={r_rbf['best_C']}  γ={r_rbf.get('best_gamma')}")

        r_rf = eval_rf(X_raw, y, seed)
        rows.append({"method": "RandomForest", "seed": seed,
                     "macro_f1": r_rf["macro_f1"]})
        logger.info(f"    {'RandomForest':<12} F1={r_rf['macro_f1']:.4f}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(SAVE_DIR, "metrics.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"\n  Saved: {out_csv}")

    # Phase 3: Statistical summary
    logger.info("\n--- Phase 3: Statistical tests (SRQFM vs others) ---")

    methods = ["SRQFM-PQK", "ZZ-PQK", "AGPQK", "RBF-SVM", "RandomForest"]
    f1_by_method = {}
    for m in methods:
        sub = df[df.method == m]["macro_f1"].values
        f1_by_method[m] = sub
        mean, std = sub.mean(), sub.std()
        logger.info(f"  {m:<14} mean={mean:.4f}  std={std:.4f}")

    srqfm_f1 = f1_by_method["SRQFM-PQK"]
    comparisons = ["ZZ-PQK", "AGPQK", "RBF-SVM", "RandomForest"]
    pvals, ds = [], []
    for m in comparisons:
        other = f1_by_method[m]
        n = min(len(srqfm_f1), len(other))
        try:
            _, p = wilcoxon(srqfm_f1[:n], other[:n], alternative="two-sided")
        except ValueError:
            p = 1.0
        d = cohens_d(srqfm_f1[:n], other[:n])
        pvals.append(p)
        ds.append(d)

    corrected = holm_bonferroni(pvals)

    logger.info("\n  Wilcoxon tests (SRQFM vs X), Holm-Bonferroni corrected:")
    logger.info(f"  {'Method':<14} {'p-value':>10} {'d':>8} {'sig*':>6}")
    for m, p, d, sig in zip(comparisons, pvals, ds, corrected):
        logger.info(f"  {m:<14} {p:>10.4f} {d:>8.3f} {'*' if sig else '':>6}")

    logger.info("\n  E26 complete.")
    return df


if __name__ == "__main__":
    run()
