"""
Experiment E21: 2-Body Feature Ablation Study.

Tests whether the 9 two-qubit correlation terms in 2-PQK add discriminative
power beyond single-qubit Bloch vectors (PQK), and whether these quantum
features can be replicated by classical polynomial features.

Conditions:
    A) PQK-only:     6 single-qubit features per pair (⟨XI⟩,...,⟨IZ⟩) → RBF
    B) Corr-only:    9 two-qubit correlation features per pair (⟨XX⟩,...,⟨ZZ⟩) → RBF
    C) Full 2-PQK:   All 15 features per pair → RBF (= QEA-2PQK Pass 2)
    D) Poly-classical: 9 polynomial/trig features per pair → RBF (dequantization)

Interpretation:
    - If B > A → entanglement correlations carry discriminative signal
    - If C > A and C > B → both components contribute, full 2-PQK is best
    - If C ≈ D → quantum correlations are classically simulable
    - If C > D → evidence of genuine quantum contribution

Protocol:
    - Uses quantum attention (QEAA) for pair selection and bandwidth
    - 5-fold CV, Macro-F1, paired t-tests, bootstrap CIs
    - All conditions use the same data splits for fair comparison

Author: Prathamesh Kadam et al.
"""

import os
import sys
import json
import logging
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._e_common import (
    load_physics_16, holm_bonferroni, cohens_d,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("exp_e21")

# ============ CONFIGURATION ============

RESULTS_DIR = os.path.join(config.RESULTS_DIR, "2body_ablation")
N_FOLDS = config.CV_FOLDS
CV_SEED = config.CV_SEED
N_QUBITS = config.N_QUBITS
ZZ_REPS = config.ZZ_REPS
PQK_GAMMA = config.PQK_GAMMA
TOP_K_PAIRS = 4
GAMMA_BASE = 0.50


def _ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ============ FEATURE EXTRACTION ============

def extract_all_features(X_raw, y):
    """
    Run quantum attention + 2-body feature extraction pipeline.

    Returns the full 2-body features, the encoded X, and topology config.
    """
    from src.attention_kernel import select_features_by_fisher
    from src.qeaa_kernel import compute_quantum_attention, select_qeaa_topology
    from src.two_body_pqk import compute_2body_features

    # Feature selection
    selected_indices, fisher_scores = select_features_by_fisher(X_raw, y)
    X_sel = X_raw[:, selected_indices]

    # Quantum attention
    attn_path = os.path.join(RESULTS_DIR, "qeaa_attention.json")
    attention = compute_quantum_attention(X_sel, n_qubits=N_QUBITS, save_path=attn_path)

    # Topology
    topology = select_qeaa_topology(
        w=attention["w"], I_Q=attention["I_Q"],
        gamma_base=GAMMA_BASE, top_k_pairs=TOP_K_PAIRS,
        require_cross_modal=True, selected_indices=selected_indices,
    )

    gamma_per_qubit = topology["gamma_per_qubit"]
    ent_pairs = topology["entanglement_pairs"]

    # Encode features with per-qubit bandwidth
    X_enc = X_sel * gamma_per_qubit[np.newaxis, :]

    # 2-body features
    feats_path = os.path.join(RESULTS_DIR, "2body_features_full.npy")
    features = compute_2body_features(
        X_enc, gamma_per_qubit=np.ones(N_QUBITS),
        entanglement_pairs=ent_pairs, pairs_to_measure=ent_pairs,
        n_qubits=N_QUBITS, reps=ZZ_REPS, save_path=feats_path,
    )

    return features, X_enc, ent_pairs, topology


# ============ EVALUATION ============

def evaluate_features_cv(features, y, gamma, n_folds=N_FOLDS, seed=CV_SEED,
                         condition_name=""):
    """
    Evaluate feature vector → RBF kernel → SVM via stratified CV.

    Computes a standard RBF kernel from the feature vectors, then evaluates
    with precomputed kernel SVM.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        f_tr = features[tr_idx]
        f_te = features[te_idx]

        # Standardize features within each fold (no leakage)
        scaler = StandardScaler()
        f_tr_s = scaler.fit_transform(f_tr)
        f_te_s = scaler.transform(f_te)

        # Compute RBF kernel
        from sklearn.metrics.pairwise import rbf_kernel
        K_tr = rbf_kernel(f_tr_s, f_tr_s, gamma=gamma)
        K_te = rbf_kernel(f_te_s, f_tr_s, gamma=gamma)

        clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
        clf.fit(K_tr, y[tr_idx])
        y_pred = clf.predict(K_te)

        fold_results.append({
            "macro_f1": float(f1_score(y[te_idx], y_pred, average="macro")),
            "accuracy": float(accuracy_score(y[te_idx], y_pred)),
        })

    f1s = [r["macro_f1"] for r in fold_results]
    result = {
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "fold_f1s": f1s,
        "fold_results": fold_results,
        "n_features": features.shape[1],
    }

    logger.info(f"  {condition_name}: F1={result['macro_f1_mean']:.4f} "
                f"± {result['macro_f1_std']:.4f} (dim={features.shape[1]})")
    return result


def evaluate_quantum_kernel_cv(K, y, n_folds=N_FOLDS, seed=CV_SEED,
                               condition_name=""):
    """Evaluate a precomputed quantum kernel matrix via stratified CV."""
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
    result = {
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "fold_f1s": f1s,
        "fold_results": fold_results,
    }

    logger.info(f"  {condition_name}: F1={result['macro_f1_mean']:.4f} "
                f"± {result['macro_f1_std']:.4f}")
    return result


# ============ STATISTICAL ANALYSIS ============

def pairwise_comparisons(results):
    """Compare all conditions pairwise via paired t-tests."""
    keys = list(results.keys())
    comparisons = {}
    p_values = []
    comp_names = []

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a_key, b_key = keys[i], keys[j]
            a_f1 = results[a_key]["fold_f1s"]
            b_f1 = results[b_key]["fold_f1s"]
            t_stat, p_val = stats.ttest_rel(a_f1, b_f1)
            d = cohens_d(a_f1, b_f1)

            name = f"{a_key} vs {b_key}"
            comparisons[name] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
                "cohens_d": float(d),
                "delta_f1": float(np.mean(a_f1) - np.mean(b_f1)),
            }
            p_values.append(p_val)
            comp_names.append(name)

    # Holm–Bonferroni
    if p_values:
        sig_flags = holm_bonferroni(p_values)
        for i, name in enumerate(comp_names):
            comparisons[name]["significant_holm"] = bool(sig_flags[i])

    return comparisons


# ============ MAIN ============

def main():
    _ensure_dir()
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("  Experiment E21: 2-Body Feature Ablation")
    logger.info("=" * 70)

    # Load data
    X_norm, X_raw, y = load_physics_16()
    logger.info(f"Data: {len(X_norm)} samples, {X_raw.shape[1]} raw features")

    # Extract all features via quantum attention + 2-body
    logger.info("\nExtracting 2-body quantum features...")
    features_full, X_enc, ent_pairs, topology = extract_all_features(X_raw, y)
    n_pairs = len(ent_pairs)
    logger.info(f"  Full features: {features_full.shape}")
    logger.info(f"  Pairs: {ent_pairs}")

    # Prepare feature subsets
    from src.two_body_pqk import (
        extract_1body_features, extract_2body_correlations,
        compute_polynomial_features, compute_2pqk_kernel,
    )

    # Condition A: 1-body only (single-qubit expectations)
    feats_1body = extract_1body_features(features_full, n_pairs)
    logger.info(f"  1-body features: {feats_1body.shape}")

    # Condition B: 2-body correlations only
    feats_corr = extract_2body_correlations(features_full, n_pairs)
    logger.info(f"  Correlation features: {feats_corr.shape}")

    # Condition C: Full 2-body (all 15 per pair)
    # Already have: features_full

    # Condition D: Classical polynomial features
    feats_poly = compute_polynomial_features(X_enc, ent_pairs)
    logger.info(f"  Polynomial features: {feats_poly.shape}")

    # Also compute the full 2-PQK kernel for quantum kernel evaluation
    K_2pqk_path = os.path.join(RESULTS_DIR, "K_2pqk_ablation.npy")
    K_2pqk = compute_2pqk_kernel(features_full, gamma=PQK_GAMMA,
                                  n_pairs=n_pairs, save_path=K_2pqk_path)

    # ---- Evaluate all conditions ----
    logger.info("\n" + "=" * 50)
    logger.info("Evaluating ablation conditions...")

    # Use a moderate gamma for RBF on feature vectors
    # (auto-scaled based on feature dimension)
    gamma_feat = 1.0 / features_full.shape[1]

    results = {}

    # Condition A: 1-body → RBF
    results["A_1body"] = evaluate_features_cv(
        feats_1body, y, gamma=1.0 / feats_1body.shape[1],
        condition_name="A) 1-body only (PQK analog)",
    )

    # Condition B: Correlations → RBF
    results["B_corr"] = evaluate_features_cv(
        feats_corr, y, gamma=1.0 / feats_corr.shape[1],
        condition_name="B) 2-body correlations only",
    )

    # Condition C: Full → RBF (features)
    results["C_full_feat"] = evaluate_features_cv(
        features_full, y, gamma=gamma_feat,
        condition_name="C) Full 2-body features → RBF",
    )

    # Condition C': Full → quantum 2-PQK kernel
    results["C_full_qkernel"] = evaluate_quantum_kernel_cv(
        K_2pqk, y, condition_name="C') Full 2-body → 2-PQK kernel",
    )

    # Condition D: Polynomial classical → RBF (dequantization test)
    results["D_poly"] = evaluate_features_cv(
        feats_poly, y, gamma=1.0 / feats_poly.shape[1],
        condition_name="D) Classical polynomial (dequant.)",
    )

    # ---- Validate 2-RDMs ----
    logger.info("\n" + "=" * 50)
    logger.info("Validating 2-RDM reconstructions...")
    from src.two_body_pqk import validate_2rdm

    validation_results = {}
    for p_idx in range(n_pairs):
        for s_idx in [0, 1, 2]:
            val = validate_2rdm(features_full, pair_idx=p_idx,
                                sample_idx=s_idx, n_pairs=n_pairs)
            key = f"pair{p_idx}_sample{s_idx}"
            validation_results[key] = val

    all_valid = all(
        v["hermitian"] and v["trace_one"] and v["psd"]
        for v in validation_results.values()
    )
    logger.info(f"  All 2-RDMs valid: {all_valid}")

    # ---- Statistics ----
    logger.info("\n" + "=" * 50)
    logger.info("Statistical comparisons...")
    comparisons = pairwise_comparisons(results)

    # ---- Interpretation ----
    logger.info("\n" + "=" * 50)
    logger.info("Interpretation:")

    a_f1 = results["A_1body"]["macro_f1_mean"]
    b_f1 = results["B_corr"]["macro_f1_mean"]
    c_f1 = results["C_full_feat"]["macro_f1_mean"]
    d_f1 = results["D_poly"]["macro_f1_mean"]

    if b_f1 > a_f1:
        logger.info("  [+] B > A: Entanglement correlations carry discriminative signal!")
    else:
        logger.info("  [-] B <= A: 2-body correlations add no value beyond 1-body.")

    if c_f1 > a_f1 and c_f1 > b_f1:
        logger.info("  [+] C > A and C > B: Both components contribute; full 2-PQK "
                     "is best.")
    elif c_f1 > a_f1:
        logger.info("  [~] C > A: Full 2-PQK improves over PQK-analog.")

    if c_f1 > d_f1 + 0.01:
        logger.info("  [+] C > D: Quantum features NOT classically simulable by "
                     "polynomial expansion!")
    elif abs(c_f1 - d_f1) < 0.01:
        logger.info("  [~] C ~ D: Quantum features approximately match polynomial -- "
                     "possible dequantization.")
    else:
        logger.info("  [-] C < D: Classical polynomial features outperform quantum "
                     "2-body features.")

    # ---- Save ----
    total_time = time.time() - start_time
    output = {
        "experiment": "E21_2body_ablation",
        "total_time_s": total_time,
        "n_folds": N_FOLDS,
        "n_train": len(y),
        "n_qubits": N_QUBITS,
        "pairs": ent_pairs,
        "results": results,
        "comparisons": comparisons,
        "validation": validation_results,
        "all_2rdm_valid": all_valid,
        "topology": {
            "gamma_per_qubit": topology["gamma_per_qubit"].tolist(),
            "entanglement_pairs": topology["entanglement_pairs"],
            "w": topology["w"].tolist(),
        },
    }

    out_path = os.path.join(RESULTS_DIR, "exp_e21_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved: {out_path}")

    # ---- Summary table ----
    logger.info("\n" + "=" * 70)
    logger.info("  E21 ABLATION RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Condition':<30} {'Macro-F1':>10} {'± std':>8} {'dim':>6}")
    logger.info("-" * 56)
    for key in ["A_1body", "B_corr", "C_full_feat", "C_full_qkernel", "D_poly"]:
        r = results[key]
        dim = r.get("n_features", "-")
        logger.info(f"{key:<30} {r['macro_f1_mean']:>10.4f} "
                    f"{r['macro_f1_std']:>8.4f} {str(dim):>6}")

    logger.info(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
