"""
Experiment 9: Full Ablation Study.

Ablation table with cumulative improvements:
    1. FQK-SVM baseline (Fused)
    2. + Class-weighted KTA → verify ΔKTA > 0
    3. + PQK instead of FQK
    4. + TDA features
    5. + Balanced sampling
    6. + TFK (kernel alignment training via parameter-shift KTA optimization)
    7. + CM-FQK (cross-modal encoding: SAR qubits 0-3, optical qubits 4-7)
    CNN baseline for comparison.

Phase 10. Expected runtime: 6-10 hours.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

import config
from src.quantum_kernels import compute_fqk_kernel_matrix, compute_pqk_kernel_matrix
from src.classical_kernels import compute_rbf_kernel
from src.data_loader import load_modality, load_subsample, get_fewshot_subset
from src.classifiers import train_precomputed_svm, evaluate_classifier, train_classical_svms
from src.kernel_target_alignment import compute_kta
from src.utils import save_results, setup_logging, Timer, check_and_skip, ensure_dir, compute_full_metrics, validate_test_kernel_shape

logger = setup_logging("exp9", log_file=os.path.join(config.RESULTS_DIR, "exp9.log"))


def load_precomputed_kernel(train_path, test_path, name):
    """Load a precomputed kernel from disk. Raises clear error if missing."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"{name} train kernel not found: {train_path}\n"
            f"Run kernel computation before exp9."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"{name} test kernel not found: {test_path}\n"
            f"Run kernel computation before exp9."
        )
    K_train = np.load(train_path)
    K_test  = np.load(test_path)
    logger.info(f"  Loaded {name}: train={K_train.shape}, test={K_test.shape}")
    return K_train, K_test


def run_experiment():
    results_dir = ensure_dir(os.path.join(config.RESULTS_DIR, "ablation"))
    results_file = os.path.join(results_dir, "exp9_results.json")
    if check_and_skip(results_file, logger):
        return

    np.random.seed(config.RANDOM_SEED)

    with Timer("Loading data", logger=logger):
        data = load_modality("fused")
        X_train = data["X_train"][:config.SUBSAMPLE_TRAIN]
        X_test = data["X_test"][:config.SUBSAMPLE_TEST]
        y_train = data["y_train"][:config.SUBSAMPLE_TRAIN]
        y_test = data["y_test"][:config.SUBSAMPLE_TEST]

    # ---- Copy Exp1 kernel cache to avoid recomputation ----
    # Exp1 already computed FQK, PQK, TFK, CM-FQK on the same 2000 samples.
    # Copying to the ablation directory allows compute_*_kernel_matrix()
    # to find and skip them via the save_path cache check.
    import shutil
    exp1_fused = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
    exp1_tfk   = os.path.join(exp1_fused, "tfk")

    kernel_cache_map = {
        # (source in exp1, destination in ablation)
        "K_fqk_train.npy":    "K_fqk.npy",
        "K_fqk_test.npy":     "K_fqk_te.npy",
        "K_pqk_train.npy":    "K_pqk.npy",
        "K_pqk_test.npy":     "K_pqk_te.npy",
    }
    for src_name, dst_name in kernel_cache_map.items():
        src = os.path.join(exp1_fused, src_name)
        dst = os.path.join(results_dir, dst_name)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            logger.info(f"  [CACHE] Copied {src_name} → ablation/{dst_name}")
        elif not os.path.exists(src):
            logger.warning(f"  [CACHE] Source not found: {src} — will recompute.")

    logger.info("Kernel cache check complete.")
    rows = []

    # ---- Step 1: FQK Baseline (cached from Exp1) ----
    logger.info("\n--- Step 1: FQK (baseline quantum kernel) ---")
    K_fqk    = compute_fqk_kernel_matrix(
        X_train, save_path=os.path.join(results_dir, "K_fqk.npy"))
    K_fqk_te = compute_fqk_kernel_matrix(
        X_train, X2=X_test, save_path=os.path.join(results_dir, "K_fqk_te.npy"))
    clf = train_precomputed_svm(K_fqk, y_train)
    m = evaluate_classifier(clf, K_fqk_te, y_test, "FQK-SVM")
    kta_fqk = compute_kta(K_fqk, y_train, class_weighted=True)
    rows.append({
        "Architecture": "FQK-SVM",
        "Change from baseline": "—",
        "Macro-F1": round(m["macro_f1"], 4),
        "Accuracy": round(m["accuracy"], 4),
        "KTA_c": round(kta_fqk, 4),
        "Note": "Standard ZZFeatureMap, linear connectivity",
    })

    # ---- Step 2: PQK — anti-concentration design ----
    logger.info("\n--- Step 2: PQK (anti-concentration projection) ---")
    K_pqk    = compute_pqk_kernel_matrix(
        X_train, save_path=os.path.join(results_dir, "K_pqk.npy"),
        bloch_save_path=os.path.join(results_dir, "bloch.npy"))
    K_pqk_te = compute_pqk_kernel_matrix(
        X_train, X2=X_test, save_path=os.path.join(results_dir, "K_pqk_te.npy"),
        bloch_save_path=os.path.join(results_dir, "bloch.npy"))
    validate_test_kernel_shape(K_pqk_te, X_train, X_test, "PQK ablation")
    clf = train_precomputed_svm(K_pqk, y_train)
    m_pqk = evaluate_classifier(clf, K_pqk_te, y_test, "PQK-SVM")
    kta_pqk = compute_kta(K_pqk, y_train, class_weighted=True)
    delta_fqk = round(m_pqk["macro_f1"] - rows[0]["Macro-F1"], 4)
    rows.append({
        "Architecture": "PQK-SVM",
        "Change from baseline": f"{delta_fqk:+.4f}",
        "Macro-F1": round(m_pqk["macro_f1"], 4),
        "Accuracy": round(m_pqk["accuracy"], 4),
        "KTA_c": round(kta_pqk, 4),
        "Note": "Bloch-vector projection; designed to reduce concentration",
    })

    # ---- Step 3: TFK — trained kernel ----
    logger.info("\n--- Step 3: TFK (KTA-trained kernel) ---")
    fused_gd_dir = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
    tfk_dir      = os.path.join(fused_gd_dir, "tfk")
    try:
        K_tfk, K_tfk_te = load_precomputed_kernel(
            os.path.join(tfk_dir, "K_tfk_train.npy"),
            os.path.join(tfk_dir, "K_tfk_test.npy"),
            "TFK",
        )
        clf = train_precomputed_svm(K_tfk, y_train)
        m_tfk = evaluate_classifier(clf, K_tfk_te, y_test, "TFK-SVM")
        kta_tfk = compute_kta(K_tfk, y_train, class_weighted=True)
        delta_tfk = round(m_tfk["macro_f1"] - rows[0]["Macro-F1"], 4)
        rows.append({
            "Architecture": "TFK-SVM",
            "Change from baseline": f"{delta_tfk:+.4f}",
            "Macro-F1": round(m_tfk["macro_f1"], 4),
            "Accuracy": round(m_tfk["accuracy"], 4),
            "KTA_c": round(kta_tfk, 4),
            "Note": "75 epochs KTA optimisation on 51-sample anchor set",
        })
    except FileNotFoundError:
        logger.warning("  TFK kernels not found. Run TFK training first.")
        rows.append({
            "Architecture": "TFK-SVM", "Change from baseline": "N/A",
            "Macro-F1": "N/A", "Accuracy": "N/A", "KTA_c": "N/A",
            "Note": "TFK kernels not found — skipping",
        })

    # ---- Step 4: CM-FQK — cross-modal entanglement ----
    logger.info("\n--- Step 4: CM-FQK (cross-modal ZZ entanglement) ---")
    try:
        K_cm_fqk, K_cm_fqk_te = load_precomputed_kernel(
            os.path.join(fused_gd_dir, "K_cm_fqk_train.npy"),
            os.path.join(fused_gd_dir, "K_cm_fqk_test.npy"),
            "CM-FQK",
        )
        clf = train_precomputed_svm(K_cm_fqk, y_train)
        m_cm = evaluate_classifier(clf, K_cm_fqk_te, y_test, "CM-FQK-SVM")
        kta_cm = compute_kta(K_cm_fqk, y_train, class_weighted=True)
        delta_cm = round(m_cm["macro_f1"] - rows[0]["Macro-F1"], 4)
        rows.append({
            "Architecture": "CM-FQK-SVM",
            "Change from baseline": f"{delta_cm:+.4f}",
            "Macro-F1": round(m_cm["macro_f1"], 4),
            "Accuracy": round(m_cm["accuracy"], 4),
            "KTA_c": round(kta_cm, 4),
            "Note": "Top-3 MI cross-modal ZZ gates (training-split MI only)",
        })
    except FileNotFoundError:
        logger.warning("  CM-FQK kernels not found. Run CM-FQK computation first.")
        rows.append({
            "Architecture": "CM-FQK-SVM", "Change from baseline": "N/A",
            "Macro-F1": "N/A", "Accuracy": "N/A", "KTA_c": "N/A",
            "Note": "CM-FQK kernels not found — skipping",
        })

    # ---- Step 5: TDA features (conditional on exp4 completion) ----
    logger.info("\n--- Step 5: FQK + TDA features ---")
    tda_file = os.path.join(config.TOPO_DIR, "topological_features.npz")
    if os.path.exists(tda_file):
        tda = np.load(tda_file)
        X_tr_tda = np.hstack([X_train, tda["sar_tda_train"][:len(X_train)]])
        X_te_tda = np.hstack([X_test,  tda["sar_tda_test"][:len(X_test)]])
        from src.feature_extraction import fit_and_transform_full_pipeline
        r = fit_and_transform_full_pipeline(X_tr_tda, None, X_te_tda, n_components=8)
        K_tda    = compute_fqk_kernel_matrix(
            r["X_train"],
            save_path=os.path.join(results_dir, "K_fqk_tda.npy"))
        K_tda_te = compute_fqk_kernel_matrix(
            r["X_train"], X2=r["X_test"],
            save_path=os.path.join(results_dir, "K_fqk_tda_te.npy"))
        clf = train_precomputed_svm(K_tda, y_train)
        m_tda = evaluate_classifier(clf, K_tda_te, y_test, "FQK+TDA-SVM")
        kta_tda = compute_kta(K_tda, y_train, class_weighted=True)
        delta_tda = round(m_tda["macro_f1"] - rows[0]["Macro-F1"], 4)
        rows.append({
            "Architecture": "FQK+TDA-SVM",
            "Change from baseline": f"{delta_tda:+.4f}",
            "Macro-F1": round(m_tda["macro_f1"], 4),
            "Accuracy": round(m_tda["accuracy"], 4),
            "KTA_c": round(kta_tda, 4),
            "Note": "Cubical TDA features (SAR), PCA→8d, then FQK",
        })
        logger.info(f"  FQK+TDA macro-F1: {m_tda['macro_f1']:.4f} "
                    f"(Δ={delta_tda:+.4f} vs FQK baseline)")
    else:
        logger.warning("  TDA features not found — skipping Step 5. "
                       "Run: python experiments/prepare_tda_features.py")
        rows.append({
            "Architecture": "FQK+TDA-SVM",
            "Change from baseline": "N/A",
            "Macro-F1": "N/A",
            "Accuracy": "N/A",
            "KTA_c": "N/A",
            "Note": "TDA features not computed — see prepare_tda_features.py",
        })

    # ---- Classical reference ----
    logger.info("\n--- Classical reference: RBF-SVM (untuned) ---")
    K_rbf    = compute_rbf_kernel(X_train)
    K_rbf_te = compute_rbf_kernel(X_train, X_test)
    clf = train_precomputed_svm(K_rbf, y_train)
    m_rbf = evaluate_classifier(clf, K_rbf_te, y_test, "RBF-SVM")
    kta_rbf = compute_kta(K_rbf, y_train, class_weighted=True)
    rows.append({
        "Architecture": "RBF-SVM",
        "Change from baseline": "classical ref.",
        "Macro-F1": round(m_rbf["macro_f1"], 4),
        "Accuracy": round(m_rbf["accuracy"], 4),
        "KTA_c": round(kta_rbf, 4),
        "Note": "γ=1/d, untuned",
    })

    # ---- Summary ----
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "ablation_table.csv"), index=False)
    logger.info(f"\nAblation table:\n{df.to_string()}")

    save_results({"ablation_table": rows, "n_train": len(X_train),
                  "n_test": len(X_test), "seed": config.RANDOM_SEED},
                 results_file)
    logger.info(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    run_experiment()
