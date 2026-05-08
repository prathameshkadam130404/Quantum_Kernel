"""
Experiment E20: QEA-2PQK Main Benchmark.

Full benchmark comparing the novel QEA-2PQK kernel (quantum attention +
2-body RDM measurement) against all existing quantum and classical kernels
on the So2Sat LCZ42 physics-16 feature set.

Kernels tested:
    - QEA-2PQK  : Quantum probe attention → 2-body RDM features → RBF
    - QEAA-PQK  : Quantum probe attention → single-qubit Bloch → RBF
    - AGPQK     : Classical Fisher attention → single-qubit Bloch → RBF
    - PQK       : No attention, standard ZZ → single-qubit Bloch → RBF
    - FQK       : Standard fidelity kernel
    - RBF-Tuned : Classical RBF with GridSearchCV
    - RandomForest: Classical ensemble baseline

Protocol:
    - 5-fold StratifiedKFold CV on physics-16 features (N=2000)
    - Macro-F1 as primary metric
    - Paired t-tests with Holm–Bonferroni correction
    - Bootstrap 95% CIs (1000 resamples)
    - Cohen's d effect sizes

Author: Prathamesh Kadam et al.
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._e_common import (
    load_physics_16, spectral_metrics, holm_bonferroni, cohens_d,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("exp_e20")

# ============ CONFIGURATION ============

RESULTS_DIR = os.path.join(config.RESULTS_DIR, "qea_2pqk_benchmark")
N_FOLDS = config.CV_FOLDS
CV_SEED = config.CV_SEED
N_QUBITS = config.N_QUBITS
ZZ_REPS = config.ZZ_REPS
PQK_GAMMA = config.PQK_GAMMA
TOP_K_PAIRS = 4
GAMMA_BASE = 0.50


def _ensure_results_dir():
    """Create results directory if it doesn't exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ============ KERNEL BUILDERS ============

def build_qea_2pqk(X_train_raw, y_train, save_dir):
    """
    Build QEA-2PQK kernel: quantum probe attention → 2-body measurement.

    Pipeline:
        1. Fisher feature selection (8/16 features, modality-balanced)
        2. Quantum probe (Pass 1) → von Neumann entropy + QMI
        3. Topology selection from quantum attention
        4. 2-body feature extraction (Pass 2) → 15 features per pair
        5. RBF kernel on 2-body features
    """
    from src.attention_kernel import select_features_by_fisher
    from src.qeaa_kernel import compute_quantum_attention, select_qeaa_topology
    from src.two_body_pqk import compute_2body_features, compute_2pqk_kernel

    # Step 1: Feature selection (same as AGPQK for fair comparison)
    selected_indices, fisher_scores = select_features_by_fisher(X_train_raw, y_train)
    X_sel = X_train_raw[:, selected_indices]

    # Step 2: Quantum probe → attention
    attn_path = os.path.join(save_dir, "qeaa_attention.json")
    attention = compute_quantum_attention(X_sel, n_qubits=N_QUBITS, save_path=attn_path)

    # Step 3: Topology from quantum attention
    topology = select_qeaa_topology(
        w=attention["w"],
        I_Q=attention["I_Q"],
        gamma_base=GAMMA_BASE,
        top_k_pairs=TOP_K_PAIRS,
        require_cross_modal=True,
        selected_indices=selected_indices,
        max_appearances=2,
    )

    gamma_per_qubit = topology["gamma_per_qubit"]
    ent_pairs = topology["entanglement_pairs"]

    # Apply per-qubit bandwidth to selected features
    X_enc = X_sel * gamma_per_qubit[np.newaxis, :]

    # Step 4: 2-body feature extraction
    feats_path = os.path.join(save_dir, "qea_2pqk_features.npy")
    features = compute_2body_features(
        X_enc, gamma_per_qubit=np.ones(N_QUBITS),  # Already applied bandwidth
        entanglement_pairs=ent_pairs,
        pairs_to_measure=ent_pairs,
        n_qubits=N_QUBITS, reps=ZZ_REPS,
        save_path=feats_path,
    )

    # Step 5: Kernel
    K_path = os.path.join(save_dir, "K_qea_2pqk.npy")
    K = compute_2pqk_kernel(features, gamma=PQK_GAMMA,
                            n_pairs=len(ent_pairs), save_path=K_path)

    return K, {
        "selected_indices": selected_indices.tolist(),
        "gamma_per_qubit": gamma_per_qubit.tolist(),
        "entanglement_pairs": ent_pairs,
        "w": attention["w"].tolist(),
        "feature_dim": features.shape[1],
    }


def build_qeaa_pqk(X_train_raw, y_train, save_dir):
    """
    Build QEAA-PQK: quantum probe attention → single-qubit PQK.
    Uses existing qeaa_kernel pipeline.
    """
    from src.attention_kernel import select_features_by_fisher
    from src.qeaa_kernel import compute_qeaa_pqk_full_pipeline

    selected_indices, fisher_scores = select_features_by_fisher(X_train_raw, y_train)
    X_sel = X_train_raw[:, selected_indices]

    result = compute_qeaa_pqk_full_pipeline(
        X_train=X_sel, X_test=None, y_train=y_train,
        selected_indices=selected_indices,
        gamma_base=GAMMA_BASE, gamma_kernel=PQK_GAMMA,
        top_k_pairs=TOP_K_PAIRS, require_cross_modal=True,
        n_qubits=N_QUBITS, reps=ZZ_REPS,
        save_dir=os.path.join(save_dir, "qeaa_pqk"),
    )
    return result["K_train"], {
        "attention_w": result["attention"]["w"].tolist(),
        "topology_pairs": result["topology"]["entanglement_pairs"],
    }


def build_agpqk(X_train_raw, y_train, save_dir):
    """Build AGPQK: classical Fisher/attention → PQK."""
    from src.attention_kernel import (
        select_features_by_fisher, compute_attention_matrix,
        compute_per_qubit_gamma, get_attention_entanglement_pairs,
    )
    from experiments._e_common import build_agpqk_kernel

    selected_indices, fisher_scores = select_features_by_fisher(X_train_raw, y_train)
    A = compute_attention_matrix(X_train_raw)
    gamma_pq = compute_per_qubit_gamma(fisher_scores, selected_indices)
    ent_pairs = get_attention_entanglement_pairs(
        A, selected_indices, top_k=TOP_K_PAIRS,
    )

    K, _ = build_agpqk_kernel(
        X_train_raw, y_train, selected_indices, gamma_pq,
        ent_pairs, n_qubits=N_QUBITS, reps=ZZ_REPS, outer_gamma=PQK_GAMMA,
    )
    return K, {
        "selected_indices": selected_indices.tolist(),
        "gamma_per_qubit": gamma_pq.tolist(),
        "entanglement_pairs": ent_pairs,
    }


def build_pqk(X_sel, save_dir):
    """Build standard PQK with linear topology (no attention)."""
    from src.quantum_kernels import compute_pqk_kernel_matrix
    K_path = os.path.join(save_dir, "K_pqk.npy")
    K = compute_pqk_kernel_matrix(X_sel, save_path=K_path)
    return K


def build_fqk(X_sel, save_dir):
    """Build standard FQK."""
    from src.quantum_kernels import compute_fqk_kernel_matrix
    K_path = os.path.join(save_dir, "K_fqk.npy")
    K = compute_fqk_kernel_matrix(X_sel, save_path=K_path)
    return K


# ============ EVALUATION ============

def evaluate_kernel(K, y, n_folds=N_FOLDS, seed=CV_SEED):
    """
    Evaluate a precomputed kernel matrix via stratified K-fold CV.

    Returns per-fold and aggregate metrics.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        K_tr = K[np.ix_(tr_idx, tr_idx)]
        K_te = K[np.ix_(te_idx, tr_idx)]

        clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
        clf.fit(K_tr, y[tr_idx])
        y_pred = clf.predict(K_te)

        fold_results.append({
            "macro_f1": float(f1_score(y[te_idx], y_pred, average="macro")),
            "accuracy": float(accuracy_score(y[te_idx], y_pred)),
        })

    f1s = [r["macro_f1"] for r in fold_results]
    accs = [r["accuracy"] for r in fold_results]

    return {
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "fold_f1s": f1s,
        "fold_results": fold_results,
    }


def evaluate_classical_rbf(X, y, n_folds=N_FOLDS, seed=CV_SEED):
    """Evaluate RBF-SVM with GridSearchCV for γ and C."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        param_grid = {"C": [1.0, 10.0, 100.0], "gamma": ["scale", "auto", 0.1]}
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        gs = GridSearchCV(
            SVC(kernel="rbf", class_weight="balanced"),
            param_grid, cv=inner_cv, scoring="f1_macro", n_jobs=-1,
        )
        gs.fit(X_tr, y_tr)
        y_pred = gs.predict(X_te)

        fold_results.append({
            "macro_f1": float(f1_score(y_te, y_pred, average="macro")),
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "best_params": gs.best_params_,
        })

    f1s = [r["macro_f1"] for r in fold_results]
    return {
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "fold_f1s": f1s,
        "fold_results": fold_results,
    }


def evaluate_random_forest(X, y, n_folds=N_FOLDS, seed=CV_SEED):
    """Evaluate Random Forest baseline."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=None, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        clf.fit(X[tr_idx], y[tr_idx])
        y_pred = clf.predict(X[te_idx])

        fold_results.append({
            "macro_f1": float(f1_score(y[te_idx], y_pred, average="macro")),
            "accuracy": float(accuracy_score(y[te_idx], y_pred)),
        })

    f1s = [r["macro_f1"] for r in fold_results]
    return {
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "fold_f1s": f1s,
        "fold_results": fold_results,
    }


# ============ STATISTICAL TESTS ============

def compute_statistics(results: Dict[str, Dict]) -> Dict:
    """
    Compute pairwise paired t-tests between QEA-2PQK and all other methods,
    with Holm–Bonferroni correction and Cohen's d effect sizes.
    """
    ref_key = "QEA-2PQK"
    if ref_key not in results:
        logger.warning(f"{ref_key} not in results, skipping statistics.")
        return {}

    ref_f1s = results[ref_key]["fold_f1s"]
    comparisons = {}
    p_values = []
    comp_keys = []

    for key, val in results.items():
        if key == ref_key:
            continue
        other_f1s = val["fold_f1s"]
        t_stat, p_val = stats.ttest_rel(ref_f1s, other_f1s)
        d = cohens_d(ref_f1s, other_f1s)
        comparisons[key] = {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(d),
            "delta_f1": float(np.mean(ref_f1s) - np.mean(other_f1s)),
        }
        p_values.append(p_val)
        comp_keys.append(key)

    # Holm–Bonferroni correction
    if p_values:
        sig_flags = holm_bonferroni(p_values)
        for i, key in enumerate(comp_keys):
            comparisons[key]["significant_holm"] = bool(sig_flags[i])

    return comparisons


def bootstrap_ci(f1s, n_boot=1000, alpha=0.05, seed=42):
    """Compute bootstrap confidence interval for mean F1."""
    rng = np.random.RandomState(seed)
    boot_means = []
    f1s = np.asarray(f1s)
    for _ in range(n_boot):
        sample = rng.choice(f1s, size=len(f1s), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.sort(boot_means)
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return {"ci_lo": lo, "ci_hi": hi, "ci_mean": float(np.mean(boot_means))}


# ============ MAIN ============

def main():
    """Run the full E20 benchmark."""
    _ensure_results_dir()
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("  Experiment E20: QEA-2PQK Main Benchmark")
    logger.info("=" * 70)

    # Load data
    X_norm, X_raw, y = load_physics_16()
    N = len(X_norm)
    logger.info(f"Data: {N} samples, {X_norm.shape[1]} normalized features, "
                f"{X_raw.shape[1]} raw features, {len(np.unique(y))} classes")

    # Feature selection (performed once; shared across kernel builders)
    from src.attention_kernel import select_features_by_fisher
    selected_indices, fisher_scores = select_features_by_fisher(X_raw, y)
    X_sel = X_raw[:, selected_indices]
    logger.info(f"Selected features: indices={selected_indices.tolist()}")

    results = {}
    kernel_configs = {}

    # ---- 1. QEA-2PQK (the novel kernel) ----
    logger.info("\n" + "=" * 50)
    logger.info("Building QEA-2PQK kernel...")
    t0 = time.time()
    K_qea, cfg_qea = build_qea_2pqk(X_raw, y, RESULTS_DIR)
    t_qea = time.time() - t0
    logger.info(f"QEA-2PQK built in {t_qea:.1f}s")
    results["QEA-2PQK"] = evaluate_kernel(K_qea, y)
    results["QEA-2PQK"]["build_time_s"] = t_qea
    kernel_configs["QEA-2PQK"] = cfg_qea
    logger.info(f"QEA-2PQK: F1={results['QEA-2PQK']['macro_f1_mean']:.4f} "
                f"± {results['QEA-2PQK']['macro_f1_std']:.4f}")

    # ---- 2. QEAA-PQK (quantum attention + single-qubit PQK) ----
    logger.info("\n" + "=" * 50)
    logger.info("Building QEAA-PQK kernel...")
    t0 = time.time()
    K_qeaa, cfg_qeaa = build_qeaa_pqk(X_raw, y, RESULTS_DIR)
    t_qeaa = time.time() - t0
    logger.info(f"QEAA-PQK built in {t_qeaa:.1f}s")
    results["QEAA-PQK"] = evaluate_kernel(K_qeaa, y)
    results["QEAA-PQK"]["build_time_s"] = t_qeaa
    kernel_configs["QEAA-PQK"] = cfg_qeaa
    logger.info(f"QEAA-PQK: F1={results['QEAA-PQK']['macro_f1_mean']:.4f}")

    # ---- 3. AGPQK (classical attention + PQK) ----
    logger.info("\n" + "=" * 50)
    logger.info("Building AGPQK kernel...")
    t0 = time.time()
    K_agpqk, cfg_agpqk = build_agpqk(X_raw, y, RESULTS_DIR)
    t_agpqk = time.time() - t0
    results["AGPQK"] = evaluate_kernel(K_agpqk, y)
    results["AGPQK"]["build_time_s"] = t_agpqk
    kernel_configs["AGPQK"] = cfg_agpqk
    logger.info(f"AGPQK: F1={results['AGPQK']['macro_f1_mean']:.4f}")

    # ---- 4. Standard PQK (no attention) ----
    logger.info("\n" + "=" * 50)
    logger.info("Building PQK kernel...")
    t0 = time.time()
    K_pqk = build_pqk(X_sel, RESULTS_DIR)
    t_pqk = time.time() - t0
    results["PQK"] = evaluate_kernel(K_pqk, y)
    results["PQK"]["build_time_s"] = t_pqk
    logger.info(f"PQK: F1={results['PQK']['macro_f1_mean']:.4f}")

    # ---- 5. FQK ----
    logger.info("\n" + "=" * 50)
    logger.info("Building FQK kernel...")
    t0 = time.time()
    K_fqk = build_fqk(X_sel, RESULTS_DIR)
    t_fqk = time.time() - t0
    results["FQK"] = evaluate_kernel(K_fqk, y)
    results["FQK"]["build_time_s"] = t_fqk
    logger.info(f"FQK: F1={results['FQK']['macro_f1_mean']:.4f}")

    # ---- 6. Classical RBF (tuned) ----
    logger.info("\n" + "=" * 50)
    logger.info("Evaluating RBF-Tuned...")
    results["RBF-Tuned"] = evaluate_classical_rbf(X_sel, y)
    logger.info(f"RBF-Tuned: F1={results['RBF-Tuned']['macro_f1_mean']:.4f}")

    # ---- 7. Random Forest ----
    logger.info("\n" + "=" * 50)
    logger.info("Evaluating RandomForest...")
    results["RandomForest"] = evaluate_random_forest(X_sel, y)
    logger.info(f"RandomForest: F1={results['RandomForest']['macro_f1_mean']:.4f}")

    # ---- Statistical tests ----
    logger.info("\n" + "=" * 50)
    logger.info("Computing statistical tests...")
    stat_tests = compute_statistics(results)

    # ---- Bootstrap CIs ----
    for key, val in results.items():
        if "fold_f1s" in val:
            val["bootstrap_ci"] = bootstrap_ci(val["fold_f1s"])

    # ---- Spectral diagnostics ----
    logger.info("Computing spectral diagnostics...")
    spectral = {}
    for name, K in [("QEA-2PQK", K_qea), ("QEAA-PQK", K_qeaa),
                     ("AGPQK", K_agpqk), ("PQK", K_pqk), ("FQK", K_fqk)]:
        spectral[name] = spectral_metrics(K)

    # ---- Save results ----
    total_time = time.time() - start_time
    output = {
        "experiment": "E20_qea2pqk_benchmark",
        "total_time_s": total_time,
        "n_folds": N_FOLDS,
        "n_train": N,
        "n_qubits": N_QUBITS,
        "zz_reps": ZZ_REPS,
        "pqk_gamma": PQK_GAMMA,
        "top_k_pairs": TOP_K_PAIRS,
        "results": results,
        "statistical_tests": stat_tests,
        "spectral_diagnostics": spectral,
        "kernel_configs": kernel_configs,
    }

    out_path = os.path.join(RESULTS_DIR, "exp_e20_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved: {out_path}")

    # ---- Summary table ----
    logger.info("\n" + "=" * 70)
    logger.info("  E20 RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Kernel':<16} {'Macro-F1':>10} {'± std':>8} "
                f"{'CI-lo':>8} {'CI-hi':>8}")
    logger.info("-" * 52)
    for key in ["QEA-2PQK", "QEAA-PQK", "AGPQK", "PQK", "FQK",
                 "RBF-Tuned", "RandomForest"]:
        if key in results:
            r = results[key]
            ci = r.get("bootstrap_ci", {})
            logger.info(
                f"{key:<16} {r['macro_f1_mean']:>10.4f} "
                f"{r['macro_f1_std']:>8.4f} "
                f"{ci.get('ci_lo', 0):>8.4f} "
                f"{ci.get('ci_hi', 0):>8.4f}"
            )

    if stat_tests:
        logger.info("\n  Statistical comparisons vs QEA-2PQK:")
        logger.info(f"  {'Method':<16} {'ΔF1':>8} {'p-value':>10} "
                     f"{'d':>8} {'Sig?':>6}")
        logger.info("  " + "-" * 50)
        for key, st in stat_tests.items():
            logger.info(
                f"  {key:<16} {st['delta_f1']:>+8.4f} "
                f"{st['p_value']:>10.4f} "
                f"{st['cohens_d']:>8.2f} "
                f"{'YES' if st.get('significant_holm') else 'no':>6}"
            )

    logger.info(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
