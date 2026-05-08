"""
Experiment 8: Multi-Dataset Comparison.

So2Sat + EuroSAT. Per dataset: FQK, PQK, g, KTA, SVM accuracy.
Key test: g should be higher for So2Sat (multi-modal) than EuroSAT.

Phase 9. Expected runtime: 4-6 hours.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

import config
from src.quantum_kernels import compute_fqk_kernel_matrix, compute_pqk_kernel_matrix
from src.classical_kernels import compute_rbf_kernel
from src.geometric_difference import compute_geometric_difference
from src.kernel_target_alignment import compute_kta
from src.data_loader import load_modality, get_fewshot_subset
from src.classifiers import train_precomputed_svm, evaluate_classifier
from src.utils import save_results, setup_logging, Timer, check_and_skip, ensure_dir, create_publication_plot

logger = setup_logging("exp8", log_file=os.path.join(config.RESULTS_DIR, "exp8.log"))


def run_experiment():
    results_dir = ensure_dir(os.path.join(config.RESULTS_DIR, "multidataset"))
    results_file = os.path.join(results_dir, "exp8_results.json")
    if check_and_skip(results_file, logger):
        return

    np.random.seed(config.RANDOM_SEED)
    datasets = {"So2Sat": ("fused", "so2sat"), "EuroSAT": ("fused", "eurosat")}
    all_results = {}

    for ds_name, (mod, ds) in datasets.items():
        logger.info(f"\n--- {ds_name} ---")
        try:
            data = load_modality(mod, ds)
        except FileNotFoundError:
            logger.warning(f"{ds_name} not found, skipping.")
            all_results[ds_name] = {"status": "SKIPPED"}
            continue

        X_tr = data["X_train"][:config.SUBSAMPLE_TRAIN]
        X_te = data["X_test"][:config.SUBSAMPLE_TEST]
        y_tr = data["y_train"][:config.SUBSAMPLE_TRAIN]
        y_te = data["y_test"][:config.SUBSAMPLE_TEST]
        ds_dir = ensure_dir(os.path.join(results_dir, ds_name))

        K_fqk = compute_fqk_kernel_matrix(X_tr, save_path=os.path.join(ds_dir, "K_fqk.npy"))
        K_fqk_te = compute_fqk_kernel_matrix(X_tr, X2=X_te, save_path=os.path.join(ds_dir, "K_fqk_te.npy"))
        K_rbf = compute_rbf_kernel(X_tr)
        K_rbf_te = compute_rbf_kernel(X_tr, X_te)

        g, _ = compute_geometric_difference(K_fqk, K_rbf)
        kta_q = compute_kta(K_fqk, y_tr)
        kta_c = compute_kta(K_rbf, y_tr)

        clf = train_precomputed_svm(K_fqk, y_tr)
        m_q = evaluate_classifier(clf, K_fqk_te, y_te, f"FQK-{ds_name}")
        clf = train_precomputed_svm(K_rbf, y_tr)
        m_c = evaluate_classifier(clf, K_rbf_te, y_te, f"RBF-{ds_name}")

        all_results[ds_name] = {
            "g": g, "kta_q": kta_q, "kta_c": kta_c, "delta_kta": kta_q - kta_c,
            "f1_fqk": m_q["macro_f1"], "f1_rbf": m_c["macro_f1"],
        }

    save_results(all_results, results_file)
    logger.info(f"\nResults saved: {results_file}")

if __name__ == "__main__":
    run_experiment()
