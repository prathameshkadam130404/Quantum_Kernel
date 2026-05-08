"""
Experiment 6: Dequantization Defense.

Tests whether quantum kernel advantage survives classical approximation via RFF.
Reports honestly: dequantizable or not.

Phase 7. Expected runtime: 3-5 hours.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.quantum_kernels import compute_fqk_kernel_matrix
from src.classical_kernels import compute_rbf_kernel
from src.dequantization import sweep_rff_approximation, compute_rff_kernel, dequantization_analysis
from src.kernel_target_alignment import compute_kta
from src.kernel_concentration import compute_concentration_metrics
from src.data_loader import load_modality, load_subsample, get_fewshot_subset
from src.classifiers import train_precomputed_svm, evaluate_classifier
from src.utils import save_results, setup_logging, Timer, check_and_skip, ensure_dir, create_publication_plot

logger = setup_logging("exp6", log_file=os.path.join(config.RESULTS_DIR, "exp6.log"))


def run_experiment(force: bool = False):
    """Dequantization defense experiment."""
    results_dir = ensure_dir(os.path.join(config.RESULTS_DIR, "dequantization"))
    results_file = os.path.join(results_dir, "exp6_results.json")

    if not force and check_and_skip(results_file, logger):
        return
    elif force and os.path.exists(results_file):
        logger.info("[FORCE] Overriding existing results file and rerunning.")

    np.random.seed(config.RANDOM_SEED)

    # Load data
    with Timer("Loading data", logger=logger):
        subsample = load_subsample()
        X_train = subsample["fused_X_train"]
        X_test = subsample["fused_X_test"]
        y_train = subsample["y_train"]
        y_test = subsample["y_test"]

    # Re-use FQK kernel from Exp1
    exp1_fused_dir = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
    K_fqk = compute_fqk_kernel_matrix(
        X_train, save_path=os.path.join(exp1_fused_dir, "K_fqk_train.npy")
    )
    K_fqk_test = compute_fqk_kernel_matrix(
        X_train, X2=X_test,
        save_path=os.path.join(exp1_fused_dir, "K_fqk_test.npy")
    )

    # Full dequantization analysis
    # gamma=1/n_features matches RBF baseline used across all other experiments
    gamma_rff = 1.0 / X_train.shape[1]
    results = dequantization_analysis(K_fqk, X_train, y_train, gamma=gamma_rff)
    results["gamma_rff_used"] = float(gamma_rff)
    logger.info(f"  RFF gamma={gamma_rff:.4f} (=1/n_features, matches RBF baseline)")

    # RFF classification at each D
    # gamma matches classical_kernels.py default: 1/n_features = 1/8 = 0.125
    gamma_rff = 1.0 / X_train.shape[1]
    logger.info(f"\n--- RFF Classification (gamma={gamma_rff:.4f}, matching RBF baseline) ---")
    rff_results = {}
    for D in config.RFF_N_FEATURES_LIST:
        K_rff      = compute_rff_kernel(X_train, n_features=D, gamma=gamma_rff)
        K_rff_test = compute_rff_kernel(X_train, X2=X_test, n_features=D, gamma=gamma_rff)

        clf = train_precomputed_svm(K_rff, y_train)
        m   = evaluate_classifier(clf, K_rff_test, y_test, f"RFF-SVM D={D}")
        rff_results[D] = m["macro_f1"]
        logger.info(f"  RFF D={D:5d}: macro_f1={m['macro_f1']:.4f}, "
                    f"accuracy={m.get('accuracy', float('nan')):.4f}")

    # FQK classification
    clf_fqk = train_precomputed_svm(K_fqk, y_train)
    m_fqk = evaluate_classifier(clf_fqk, K_fqk_test, y_test, "FQK-SVM")
    results["f1_fqk"] = m_fqk["macro_f1"]
    results["rff_classification"] = rff_results

    logger.info(f"  FQK baseline: macro_f1={results.get('f1_fqk', float('nan')):.4f}")
    logger.info(f"  Best RFF F1:  {max(rff_results.values()):.4f} "
                f"(D={max(rff_results, key=rff_results.get)})")

    # --- Centered weighted KTA comparison (PRIMARY metric, consistent with exp1) ---
    # Standard KTA in dequantization_analysis() inflated by RBF mean_offdiag≈0.80.
    # Centered KTA removes mean-offset bias. Reference: Cortes et al. JMLR 2012.
    logger.info("\n--- Centered Weighted KTA Comparison (PRIMARY, consistent with exp1) ---")
    from src.kernel_target_alignment import compute_centered_kta

    kta_fqk_centered = compute_centered_kta(K_fqk, y_train, class_weighted=True)
    results["kta_fqk_centered_weighted"] = float(kta_fqk_centered)
    logger.info(f"  FQK centered weighted KTA: {kta_fqk_centered:.4f} "
                f"(exp1 reference: 0.1840)")

    kta_rff_centered = {}
    for D in config.RFF_N_FEATURES_LIST:
        K_rff_d = compute_rff_kernel(X_train, n_features=D, gamma=gamma_rff)
        kta_d   = compute_centered_kta(K_rff_d, y_train, class_weighted=True)
        kta_rff_centered[D] = float(kta_d)
        logger.info(f"  RFF D={D:5d} centered weighted KTA: {kta_d:.4f}")

    results["kta_rff_centered_weighted"] = kta_rff_centered

    best_rff_kta = max(kta_rff_centered.values())
    delta_kta = kta_fqk_centered - best_rff_kta
    results["delta_kta_fqk_vs_best_rff"] = float(delta_kta)
    logger.info(f"  ΔKTA (FQK - best RFF): {delta_kta:+.4f} "
                f"({'quantum more aligned' if delta_kta > 0 else 'RFF matches or exceeds quantum'})")

    # --- Classification-based dequantization verdict ---
    # Correct test: can classical RFF match quantum F1?
    # Frobenius error measures structural similarity, NOT task performance.
    # A kernel can be structurally distinct yet produce equivalent classifiers.
    best_rff_f1   = max(rff_results.values())
    best_rff_D    = max(rff_results, key=rff_results.get)
    fqk_f1        = results["f1_fqk"]
    f1_margin     = best_rff_f1 - fqk_f1

    if f1_margin > 0.01:
        verdict = (
            f"DEQUANTIZABLE (by classification) — "
            f"RFF D={best_rff_D} achieves macro-F1={best_rff_f1:.4f}, "
            f"exceeding FQK-SVM ({fqk_f1:.4f}) by +{f1_margin:.4f}. "
            f"Quantum kernel provides no F1 advantage over classical RFF approximation."
        )
    elif f1_margin > -0.01:
        verdict = (
            f"MARGINAL (by classification) — "
            f"RFF D={best_rff_D} ({best_rff_f1:.4f}) and FQK ({fqk_f1:.4f}) "
            f"within 0.01 macro-F1. No statistically meaningful classification "
            f"advantage for quantum kernel."
        )
    else:
        verdict = (
            f"NOT DEQUANTIZABLE (by classification) — "
            f"FQK-SVM ({fqk_f1:.4f}) exceeds best RFF D={best_rff_D} "
            f"({best_rff_f1:.4f}) by {-f1_margin:.4f} macro-F1."
        )

    # Keep original kernel-approximation assessment under a distinct key
    results["assessment_kernel_approx"] = results.pop("assessment", "N/A")
    results["assessment_classification"] = verdict
    results["assessment_basis"] = "downstream_macro_f1"
    results["f1_margin_rff_minus_fqk"] = float(f1_margin)
    results["best_rff_D"] = int(best_rff_D)

    logger.info(f"\n--- Dequantization Verdict ---")
    logger.info(f"  Kernel approx assessment: {results['assessment_kernel_approx']}")
    logger.info(f"  Classification assessment: {verdict}")
    logger.info(f"  F1 margin (RFF - FQK): {f1_margin:+.4f}")

    # Few-shot dequantization learning curves
    logger.info("\n--- Few-shot dequantization ---")
    fewshot_fqk, fewshot_rff = [], []
    data_full = load_modality("fused")

    for N in config.FEWSHOT_SIZES:
        X_sub, y_sub = get_fewshot_subset(data_full["X_train"], data_full["y_train"], N, config.RANDOM_SEED)
        X_te = data_full["X_test"][:config.SUBSAMPLE_TEST]
        y_te = data_full["y_test"][:config.SUBSAMPLE_TEST]

        # FQK
        K_fqk_n    = compute_fqk_kernel_matrix(X_sub,
                         save_path=os.path.join(results_dir, f"K_fqk_N{N}.npy"))
        K_fqk_te_n = compute_fqk_kernel_matrix(X_sub, X2=X_te,
                         save_path=os.path.join(results_dir, f"K_fqk_test_N{N}.npy"))
        clf = train_precomputed_svm(K_fqk_n, y_sub)
        m_fqk_n = evaluate_classifier(clf, K_fqk_te_n, y_te, f"FQK N={N}")
        fewshot_fqk.append(m_fqk_n["macro_f1"])

        # RFF (D=256), gamma matches RBF baseline
        gamma_rff  = 1.0 / X_sub.shape[1]
        K_rff_n    = compute_rff_kernel(X_sub, n_features=256, gamma=gamma_rff)
        K_rff_te_n = compute_rff_kernel(X_sub, X2=X_te, n_features=256, gamma=gamma_rff)
        clf = train_precomputed_svm(K_rff_n, y_sub)
        m_rff_n = evaluate_classifier(clf, K_rff_te_n, y_te, f"RFF N={N}")
        fewshot_rff.append(m_rff_n["macro_f1"])

        logger.info(f"  N={N:5d}: FQK F1={m_fqk_n['macro_f1']:.4f}, "
                    f"RFF F1={m_rff_n['macro_f1']:.4f}, "
                    f"margin={m_rff_n['macro_f1'] - m_fqk_n['macro_f1']:+.4f}")

    # Plot: FQK vs RFF learning curves
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(config.FEWSHOT_SIZES, fewshot_fqk, "o-", linewidth=2, label="FQK-SVM", color="steelblue")
    ax.plot(config.FEWSHOT_SIZES, fewshot_rff, "s-", linewidth=2, label="RFF-SVM (D=256)", color="coral")
    ax.set_xlabel("N (training samples)", fontsize=12)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_xscale("log")
    ax.set_xticks(config.FEWSHOT_SIZES)
    ax.set_xticklabels([str(n) for n in config.FEWSHOT_SIZES])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title("Dequantization Defense: FQK-SVM vs RFF-SVM", fontsize=14, fontweight="bold")
    create_publication_plot(fig, os.path.join(results_dir, "fig6_dequantization_curves"))

    results["fewshot_fqk"] = fewshot_fqk
    results["fewshot_rff"] = fewshot_rff
    results["fewshot_rff_D"] = 256
    results["fewshot_rff_gamma"] = float(1.0 / X_train.shape[1])
    results["fewshot_rff_n_seeds"] = 1  # single RFF draw per N — limitation
    results["fewshot_sizes"] = list(config.FEWSHOT_SIZES)
    results["metadata"] = {
        "note_rff_variance": (
            "fewshot_rff uses a single random seed per N. "
            "RFF has sampling variance; multi-seed averaging recommended "
            "for publication-quality few-shot curves."
        ),
        "gamma_rff": float(1.0 / X_train.shape[1]),
        "gamma_matches_rbf_baseline": True,
        "kta_primary_metric": "centered_weighted (kta_fqk_centered_weighted)",
        "assessment_primary": "assessment_classification (not assessment_kernel_approx)",
    }

    save_results(results, results_file)
    logger.info(f"\nResults saved: {results_file}")
    logger.info(f"  Primary verdict: {results['assessment_classification']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Force rerun even if results file exists.")
    args = parser.parse_args()
    run_experiment(force=args.force)
