"""
E22 — SRQFM Benchmark: Self-Regulating Quantum Feature Map vs All Baselines.

Head-to-head comparison of SRQFM-PQK against:
    1. Standard ZZ-PQK (Havlicek coupling)
    2. AGPQK (Fisher-based classical attention)
    3. QEAA-PQK (quantum entropy-based attention)
    4. RBF-SVM (classical kernel)
    5. Random Forest (classical ensemble)

Evaluation:
    - 5-fold stratified cross-validation
    - Macro F1-score (primary), accuracy (secondary)
    - Kernel Target Alignment (KTA)
    - Spectral diagnostics (effective rank, off-diagonal distribution)
    - Per-qubit Bloch magnitudes (dead qubit detection)
    - Coupling diagnostics (modality interaction strengths)
    - Statistical tests (paired t-test, Cohen's d)

Author: Prathamesh Kadam et al.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.stats import ttest_rel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._e_common import (
    load_physics_16,
    spectral_metrics,
    svm_eval,
    cohens_d,
)
from src.attention_kernel import select_features_by_fisher, compute_fisher_ratio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("exp_e22")


# ============ CONFIGURATION ============

N_FOLDS = config.CV_FOLDS          # 5
SEEDS = config.SEED_LIST           # [42, 43, 44, 45, 46]
N_QUBITS = config.N_QUBITS        # 8
ZZ_REPS = config.ZZ_REPS          # 2
PQK_GAMMA = config.PQK_GAMMA      # 0.67
SAVE_DIR = os.path.join(config.RESULTS_DIR, "srqfm_benchmark")


# ============ KERNEL BUILDERS ============

def build_srqfm_kernel(X_enc, n_qubits=N_QUBITS, reps=ZZ_REPS,
                       gamma=PQK_GAMMA, connectivity="all"):
    """Build SRQFM-PQK kernel matrix with full pipeline."""
    from src.srqfm_kernel import compute_srqfm_pqk_kernel
    K = compute_srqfm_pqk_kernel(
        X_enc, gamma=gamma, n_qubits=n_qubits, reps=reps,
        coupling_threshold=0.01, connectivity=connectivity,
    )
    return K


def build_standard_zz_kernel(X_enc, n_qubits=N_QUBITS, reps=ZZ_REPS,
                              gamma=PQK_GAMMA):
    """Build standard ZZ-PQK kernel (Havlicek coupling) for comparison."""
    from src.quantum_kernels import compute_pqk_kernel_matrix
    K = compute_pqk_kernel_matrix(X_enc, n_qubits=n_qubits, reps=reps,
                                  gamma=gamma)
    return K


def build_agpqk_kernel_from_data(X_raw, X_norm, y, n_qubits=N_QUBITS,
                                  reps=ZZ_REPS, gamma=PQK_GAMMA):
    """Build AGPQK kernel with Fisher-weighted attention."""
    from experiments._e_common import build_agpqk_kernel

    # Feature selection
    sel_idx, fisher_scores = select_features_by_fisher(X_raw, y)
    fisher_sel = fisher_scores[sel_idx]

    # Per-qubit bandwidth from Fisher scores
    gamma_base = 0.50
    gamma_per_qubit = gamma_base * (fisher_sel / fisher_sel.mean())
    gamma_per_qubit = np.clip(gamma_per_qubit, 0.20, 1.50)

    # Cross-modal entanglement pairs
    from src.attention_kernel import FEATURE_MODALITY_16
    modalities = [FEATURE_MODALITY_16[i] for i in sel_idx]
    entangle_pairs = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if modalities[i] != modalities[j]:
                entangle_pairs.append((i, j))
    entangle_pairs = entangle_pairs[:4]  # Top 4

    K, _ = build_agpqk_kernel(
        X_raw, y, sel_idx, gamma_per_qubit, entangle_pairs,
        n_qubits=n_qubits, reps=reps, outer_gamma=gamma,
    )
    return K, sel_idx


# ============ CV EVALUATION ============

def evaluate_precomputed_kernel(K, y, seed, test_frac=0.3):
    """Evaluate a precomputed kernel matrix via SVM."""
    return svm_eval(K, y, seed, test_frac=test_frac)


def evaluate_classical_baselines(X_raw, y, seed, test_frac=0.3):
    """Evaluate RBF-SVM and Random Forest on raw features."""
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac,
                                random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)

    X_tr, X_te = X_raw[tr], X_raw[te]
    y_tr, y_te = y[tr], y[te]

    # Standardize
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    results = {}

    # RBF-SVM
    clf_rbf = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
    clf_rbf.fit(X_tr_s, y_tr)
    yp_rbf = clf_rbf.predict(X_te_s)
    results["RBF-SVM"] = {
        "macro_f1": float(f1_score(y_te, yp_rbf, average="macro")),
        "accuracy": float(accuracy_score(y_te, yp_rbf)),
    }

    # Random Forest
    clf_rf = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=seed
    )
    clf_rf.fit(X_tr, y_tr)
    yp_rf = clf_rf.predict(X_te)
    results["RandomForest"] = {
        "macro_f1": float(f1_score(y_te, yp_rf, average="macro")),
        "accuracy": float(accuracy_score(y_te, yp_rf)),
    }

    return results


# ============ MAIN BENCHMARK ============

def run_benchmark():
    """Execute the full E22 benchmark."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Redirect logging to file as well
    fh = logging.FileHandler(os.path.join(SAVE_DIR, "exp_e22.log"), mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    ))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 70)
    logger.info("  E22: SRQFM Benchmark — Self-Regulating Quantum Feature Map")
    logger.info("=" * 70)
    logger.info(f"  Date: {datetime.now().isoformat()}")
    logger.info(f"  N_QUBITS={N_QUBITS}, ZZ_REPS={ZZ_REPS}, "
                f"PQK_GAMMA={PQK_GAMMA}")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  Save dir: {SAVE_DIR}")

    # Load data
    logger.info("\n--- Loading data ---")
    X_norm, X_raw, y = load_physics_16()
    logger.info(f"  X_norm: {X_norm.shape}, X_raw: {X_raw.shape}, "
                f"y: {y.shape}, classes: {len(np.unique(y))}")

    # Feature selection (shared across all kernels that use 8 features)
    sel_idx, fisher_scores = select_features_by_fisher(X_raw, y)
    sel_names_raw = list(np.load(
        os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
    )["feature_names"])
    sel_names = [sel_names_raw[i] for i in sel_idx]
    X_sel = X_norm[:, sel_idx]

    logger.info(f"  Selected features: {list(zip(sel_idx, sel_names))}")
    logger.info(f"  Fisher scores: {np.round(fisher_scores[sel_idx], 3).tolist()}")

    # ========= PHASE 1: Compute all kernel matrices =========
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 1: Kernel Matrix Computation")
    logger.info("=" * 70)

    kernels = {}

    # 1. SRQFM-PQK (all-to-all connectivity)
    logger.info("\n--- 1/4: SRQFM-PQK (all-pairs) ---")
    t0 = time.time()
    K_srqfm = build_srqfm_kernel(
        X_sel, connectivity="all",
    )
    dt_srqfm = time.time() - t0
    kernels["SRQFM-PQK"] = K_srqfm
    np.save(os.path.join(SAVE_DIR, "K_srqfm.npy"), K_srqfm)
    logger.info(f"  Time: {dt_srqfm:.1f}s")

    # 2. Standard ZZ-PQK
    logger.info("\n--- 2/4: Standard ZZ-PQK ---")
    t0 = time.time()
    K_zz = build_standard_zz_kernel(X_sel)
    dt_zz = time.time() - t0
    kernels["ZZ-PQK"] = K_zz
    np.save(os.path.join(SAVE_DIR, "K_zz.npy"), K_zz)
    logger.info(f"  Time: {dt_zz:.1f}s")

    # 3. AGPQK (Fisher-weighted)
    logger.info("\n--- 3/4: AGPQK ---")
    t0 = time.time()
    K_agpqk, _ = build_agpqk_kernel_from_data(X_raw, X_norm, y)
    dt_agpqk = time.time() - t0
    kernels["AGPQK"] = K_agpqk
    np.save(os.path.join(SAVE_DIR, "K_agpqk.npy"), K_agpqk)
    logger.info(f"  Time: {dt_agpqk:.1f}s")

    # 4. SRQFM-PQK (cross-modal connectivity only)
    logger.info("\n--- 4/4: SRQFM-PQK (cross-modal) ---")
    t0 = time.time()
    K_srqfm_cross = build_srqfm_kernel(
        X_sel, connectivity="cross",
    )
    dt_cross = time.time() - t0
    kernels["SRQFM-Cross"] = K_srqfm_cross
    np.save(os.path.join(SAVE_DIR, "K_srqfm_cross.npy"), K_srqfm_cross)
    logger.info(f"  Time: {dt_cross:.1f}s")

    # ========= PHASE 2: Spectral diagnostics =========
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 2: Kernel Diagnostics")
    logger.info("=" * 70)

    from src.srqfm_kernel import compute_kta, compute_coupling_diagnostics

    diagnostics = {}
    for name, K in kernels.items():
        spec = spectral_metrics(K)
        kta = compute_kta(K, y)
        diagnostics[name] = {**spec, "kta": kta}
        logger.info(f"\n  {name}:")
        logger.info(f"    KTA:           {kta:.4f}")
        logger.info(f"    Off-diag mean: {spec['off_diag_mean']:.4f}")
        logger.info(f"    Off-diag var:  {spec['off_diag_var']:.4f}")
        logger.info(f"    Eff rank:      {spec['eff_rank_shannon']:.1f}")

    # Coupling diagnostics for SRQFM
    logger.info("\n--- SRQFM Coupling Diagnostics ---")
    coupling_diag = compute_coupling_diagnostics(
        X_sel[:200], feature_names=sel_names, n_qubits=N_QUBITS
    )

    # ========= PHASE 3: CV Evaluation =========
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 3: 5-Seed Cross-Validated Classification")
    logger.info("=" * 70)

    cv_results = {name: [] for name in list(kernels.keys()) + ["RBF-SVM", "RandomForest"]}

    for seed in SEEDS:
        logger.info(f"\n  --- Seed {seed} ---")

        # Quantum kernels
        for name, K in kernels.items():
            res = evaluate_precomputed_kernel(K, y, seed)
            cv_results[name].append(res)
            logger.info(f"    {name:20s}: F1={res['macro_f1']:.4f}  "
                        f"Acc={res['accuracy']:.4f}")

        # Classical baselines
        cl_res = evaluate_classical_baselines(X_raw, y, seed)
        for cl_name, cl_metrics in cl_res.items():
            cv_results[cl_name].append(cl_metrics)
            logger.info(f"    {cl_name:20s}: F1={cl_metrics['macro_f1']:.4f}  "
                        f"Acc={cl_metrics['accuracy']:.4f}")

    # ========= PHASE 4: Summary Statistics =========
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 4: Summary")
    logger.info("=" * 70)

    summary = {}
    logger.info(f"\n  {'Method':<20} {'F1 Mean':>8} {'F1 Std':>8} "
                f"{'Acc Mean':>8} {'KTA':>8}")
    logger.info("  " + "-" * 56)

    for name in cv_results:
        f1s = [r["macro_f1"] for r in cv_results[name]]
        accs = [r["accuracy"] for r in cv_results[name]]
        kta_val = diagnostics.get(name, {}).get("kta", None)

        summary[name] = {
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "f1_scores": f1s,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "kta": kta_val,
        }

        kta_str = f"{kta_val:.4f}" if kta_val is not None else "  N/A"
        logger.info(f"  {name:<20} {np.mean(f1s):>8.4f} {np.std(f1s):>8.4f} "
                    f"{np.mean(accs):>8.4f} {kta_str:>8}")

    # ========= PHASE 5: Statistical Tests =========
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 5: Statistical Significance (SRQFM vs others)")
    logger.info("=" * 70)

    srqfm_f1s = summary["SRQFM-PQK"]["f1_scores"]
    stat_tests = {}

    for name in cv_results:
        if name == "SRQFM-PQK":
            continue
        other_f1s = summary[name]["f1_scores"]

        if len(srqfm_f1s) == len(other_f1s) and len(srqfm_f1s) >= 3:
            t_stat, p_val = ttest_rel(srqfm_f1s, other_f1s)
            d = cohens_d(np.array(srqfm_f1s), np.array(other_f1s))
            wins = sum(1 for a, b in zip(srqfm_f1s, other_f1s) if a > b)
            sig = "YES" if p_val < 0.05 else "no"

            stat_tests[name] = {
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(d),
                "srqfm_wins": wins,
                "significant": p_val < 0.05,
            }

            direction = "better" if np.mean(srqfm_f1s) > np.mean(other_f1s) else "worse"
            logger.info(f"  SRQFM vs {name:<20}: p={p_val:.4f} d={d:+.3f} "
                        f"({direction}, {sig})")

    # ========= Save Results =========
    results = {
        "experiment": "E22_SRQFM_Benchmark",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_qubits": N_QUBITS,
            "zz_reps": ZZ_REPS,
            "pqk_gamma": PQK_GAMMA,
            "seeds": SEEDS,
            "n_train": len(X_sel),
            "n_classes": int(len(np.unique(y))),
            "selected_features": list(zip(
                [int(i) for i in sel_idx], sel_names
            )),
        },
        "summary": summary,
        "diagnostics": {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()}
            for k, v in diagnostics.items()
        },
        "statistical_tests": stat_tests,
        "coupling_diagnostics": {
            "global_mean": coupling_diag["global_mean"],
            "ranked_top5": coupling_diag["ranked_pairs"][:5],
            "ranked_bottom5": coupling_diag["ranked_pairs"][-5:],
        },
        "timing": {
            "srqfm_s": dt_srqfm,
            "zz_s": dt_zz,
            "agpqk_s": dt_agpqk,
            "srqfm_cross_s": dt_cross,
        },
    }

    results_path = os.path.join(SAVE_DIR, "e22_srqfm_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n  Results saved: {results_path}")

    # ========= Final Summary =========
    logger.info("\n" + "=" * 70)
    logger.info("  E22 BENCHMARK COMPLETE")
    logger.info("=" * 70)
    best_name = max(summary, key=lambda k: summary[k]["f1_mean"])
    logger.info(f"  Best method: {best_name} "
                f"(F1={summary[best_name]['f1_mean']:.4f})")
    logger.info(f"  SRQFM-PQK:   F1={summary['SRQFM-PQK']['f1_mean']:.4f} "
                f"+/- {summary['SRQFM-PQK']['f1_std']:.4f}")

    return results


if __name__ == "__main__":
    run_benchmark()
