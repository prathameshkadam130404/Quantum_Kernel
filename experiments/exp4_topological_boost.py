"""
Experiment 4: Topological Enhancement.

Tests TDA synergy: does persistent homology complement quantum kernels more
than classical kernels?

Feature sets: PCA-only(8), TDA-only(8), PCA+TDA(16→8 for quantum, 16 for classical).
TDA synergy score = (Acc_q_with_TDA - Acc_q_without) - (Acc_c_with_TDA - Acc_c_without).

Phase 5. Expected runtime: 4-6 hours.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

import config
from src.quantum_kernels import compute_fqk_kernel_matrix
from src.classical_kernels import compute_rbf_kernel
from src.data_loader import load_modality, load_subsample
from src.classifiers import train_precomputed_svm, evaluate_classifier
from src.feature_extraction import fit_and_transform_full_pipeline
from src.utils import save_results, setup_logging, Timer, check_and_skip, ensure_dir

logger = setup_logging("exp4", log_file=os.path.join(config.RESULTS_DIR, "exp4.log"))


def run_experiment():
    """Topological boost experiment."""
    results_dir = ensure_dir(os.path.join(config.RESULTS_DIR, "topological"))
    results_file = os.path.join(results_dir, "exp4_results.json")

    if check_and_skip(results_file, logger):
        return

    np.random.seed(config.RANDOM_SEED)

    # Load PCA features (same subsample as Exp1)
    with Timer("Loading PCA data", logger=logger):
        subsample = load_subsample()
        X_train_pca = subsample["fused_X_train"]
        X_test_pca = subsample["fused_X_test"]
        y_train = subsample["y_train"]
        y_test = subsample["y_test"]

    # Load TDA features
    tda_file = os.path.join(config.TOPO_DIR, "topological_features.npz")
    if not os.path.exists(tda_file):
        raise FileNotFoundError(
            f"TDA features not found: {tda_file}\n"
            "Run: python experiments/prepare_tda_features.py\n"
            "Do NOT proceed with dummy features — results would be meaningless."
        )

    with Timer("Loading TDA features", logger=logger):
        tda_data = np.load(tda_file)
        if "sar_tda_train" not in tda_data or "sar_tda_test" not in tda_data:
            raise KeyError(
                f"TDA file missing required keys. "
                f"Available keys: {list(tda_data.keys())}\n"
                "Regenerate with: python experiments/prepare_tda_features.py --force"
            )
        X_train_tda = tda_data["sar_tda_train"]
        X_test_tda  = tda_data["sar_tda_test"]
        logger.info(
            f"  TDA features loaded: train={X_train_tda.shape}, "
            f"test={X_test_tda.shape}"
        )

    # Verify TDA labels match PCA labels (catch any sample mismatch early)
    tda_y_train = tda_data["y_train"]
    tda_y_test  = tda_data["y_test"]
    if not np.array_equal(tda_y_train, y_train):
        raise RuntimeError(
            "TDA train labels do not match PCA train labels. "
            "Regenerate TDA features with prepare_tda_features.py."
        )
    if not np.array_equal(tda_y_test, y_test):
        raise RuntimeError(
            "TDA test labels do not match PCA test labels. "
            "Regenerate TDA features with prepare_tda_features.py."
        )
    logger.info("  Label consistency: VERIFIED")

    # Ensure matching sample counts
    n_train = min(len(X_train_pca), len(X_train_tda))
    n_test = min(len(X_test_pca), len(X_test_tda))
    X_train_pca = X_train_pca[:n_train]
    X_test_pca = X_test_pca[:n_test]
    X_train_tda = X_train_tda[:n_train]
    X_test_tda = X_test_tda[:n_test]
    y_train = y_train[:n_train]
    y_test = y_test[:n_test]

    # Combined PCA+TDA (16d → PCA to 8d for quantum)
    X_train_combined = np.hstack([X_train_pca, X_train_tda])
    X_test_combined = np.hstack([X_test_pca, X_test_tda])

    # Reduce combined to 8d for quantum
    combined_result = fit_and_transform_full_pipeline(
        X_train_combined, None, X_test_combined, n_components=8,
    )
    X_train_combined_8 = combined_result["X_train"]
    X_test_combined_8 = combined_result["X_test"]

    # Feature sets
    feature_sets = {
        "PCA-only": (X_train_pca, X_test_pca),
        "TDA-only": (X_train_tda, X_test_tda),
        "PCA+TDA (8d quantum)": (X_train_combined_8, X_test_combined_8),
    }

    results = {}
    for feat_name, (X_tr, X_te) in feature_sets.items():
        logger.info(f"\n--- {feat_name} ---")
        feat_dir = ensure_dir(os.path.join(results_dir, feat_name.replace(" ", "_")))

        # Check for Exp1 (fused) kernel reuse if this is PCA-only
        exp1_fused_dir = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
        
        fqk_train_path = os.path.join(feat_dir, "K_fqk.npy")
        fqk_test_path  = os.path.join(feat_dir, "K_fqk_test.npy")
        rbf_train_path = os.path.join(feat_dir, "K_rbf.npy")
        rbf_test_path  = os.path.join(feat_dir, "K_rbf_test.npy")

        if feat_name == "PCA-only":
            exp1_fqk_tr = os.path.join(exp1_fused_dir, "K_fqk_train.npy")
            exp1_fqk_te = os.path.join(exp1_fused_dir, "K_fqk_test.npy")
            exp1_rbf_tr = os.path.join(exp1_fused_dir, "K_rbf_train.npy")
            exp1_rbf_te = os.path.join(exp1_fused_dir, "K_rbf_test.npy")

            import shutil
            if os.path.exists(exp1_fqk_tr) and not os.path.exists(fqk_train_path):
                logger.info(f"  Reusing FQK train kernel from Exp1: {exp1_fqk_tr}")
                shutil.copy(exp1_fqk_tr, fqk_train_path)
            if os.path.exists(exp1_fqk_te) and not os.path.exists(fqk_test_path):
                logger.info(f"  Reusing FQK test kernel from Exp1: {exp1_fqk_te}")
                shutil.copy(exp1_fqk_te, fqk_test_path)
            if os.path.exists(exp1_rbf_tr) and not os.path.exists(rbf_train_path):
                logger.info(f"  Reusing RBF train kernel from Exp1: {exp1_rbf_tr}")
                shutil.copy(exp1_rbf_tr, rbf_train_path)
            if os.path.exists(exp1_rbf_te) and not os.path.exists(rbf_test_path):
                logger.info(f"  Reusing RBF test kernel from Exp1: {exp1_rbf_te}")
                shutil.copy(exp1_rbf_te, rbf_test_path)

        # FQK
        K_fqk = compute_fqk_kernel_matrix(X_tr, save_path=fqk_train_path)
        K_fqk_te = compute_fqk_kernel_matrix(X_tr, X2=X_te, save_path=fqk_test_path)
        clf = train_precomputed_svm(K_fqk, y_train)
        m_fqk = evaluate_classifier(clf, K_fqk_te, y_test, f"FQK ({feat_name})")
        m_fqk.pop("predictions", None)

        # RBF
        K_rbf = compute_rbf_kernel(X_tr, save_path=rbf_train_path)
        K_rbf_te = compute_rbf_kernel(X_tr, X_te, save_path=rbf_test_path)
        clf = train_precomputed_svm(K_rbf, y_train)
        m_rbf = evaluate_classifier(clf, K_rbf_te, y_test, f"RBF ({feat_name})")
        m_rbf.pop("predictions", None)

        results[feat_name] = {"FQK": m_fqk, "RBF": m_rbf}

    # TDA synergy score and interpretation
    if "PCA-only" in results and "PCA+TDA (8d quantum)" in results:
        fqk_pca   = results["PCA-only"]["FQK"]["macro_f1"]
        fqk_combo = results["PCA+TDA (8d quantum)"]["FQK"]["macro_f1"]
        rbf_pca   = results["PCA-only"]["RBF"]["macro_f1"]
        rbf_combo = results["PCA+TDA (8d quantum)"]["RBF"]["macro_f1"]

        fqk_boost = fqk_combo - fqk_pca
        rbf_boost = rbf_combo - rbf_pca
        synergy   = fqk_boost - rbf_boost

        interpretation = (
            "TDA complements quantum MORE than classical"
            if synergy > 0.005
            else "TDA complements classical MORE than quantum"
            if synergy < -0.005
            else "No differential TDA synergy (|Δ| < 0.005)"
        )

        results["tda_synergy"] = {
            "fqk_pca_f1": fqk_pca,
            "fqk_combo_f1": fqk_combo,
            "rbf_pca_f1": rbf_pca,
            "rbf_combo_f1": rbf_combo,
            "fqk_boost": fqk_boost,
            "rbf_boost": rbf_boost,
            "synergy_score": synergy,
            "interpretation": interpretation,
            "note": (
                "synergy = (FQK_with_TDA - FQK_without) - "
                "(RBF_with_TDA - RBF_without). "
                "Positive = TDA complements quantum more."
            ),
        }

        logger.info("\n--- TDA Synergy Summary ---")
        logger.info(f"  FQK: {fqk_pca:.4f} (PCA) → {fqk_combo:.4f} (PCA+TDA), "
                    f"boost={fqk_boost:+.4f}")
        logger.info(f"  RBF: {rbf_pca:.4f} (PCA) → {rbf_combo:.4f} (PCA+TDA), "
                    f"boost={rbf_boost:+.4f}")
        logger.info(f"  Synergy score: {synergy:+.4f}")
        logger.info(f"  Interpretation: {interpretation}")

    save_results(results, results_file)
    logger.info(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    run_experiment()
