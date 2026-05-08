"""
Experiment 3: Few-Shot Learning Curves.

N ∈ [50, 100, 200, 500, 1000, 2000]. 5 seeds. Stratified subsets.
FQK-SVM, PQK-SVM, RBF-SVM, Linear-SVM. Error bands, sample efficiency, AUC-LC.

Phase 4 of the pipeline. Expected runtime: 8-14 hours.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config
from src.quantum_kernels import compute_fqk_kernel_matrix, compute_pqk_kernel_matrix
from src.classical_kernels import compute_rbf_kernel
from src.data_loader import load_modality, get_fewshot_subset
from src.classifiers import train_precomputed_svm, evaluate_classifier
from src.utils import (
    save_results, setup_logging, Timer, check_and_skip, ensure_dir,
    plot_learning_curves, compute_full_metrics,
)

logger = setup_logging("exp3", log_file=os.path.join(config.RESULTS_DIR, "exp3.log"))


def run_experiment():
    """Few-shot learning curves experiment."""
    results_dir = ensure_dir(os.path.join(config.RESULTS_DIR, "fewshot"))
    results_file = os.path.join(results_dir, "exp3_results.json")

    if check_and_skip(results_file, logger):
        return

    # Load full training + test data
    with Timer("Loading data", logger=logger):
        data = load_modality("fused")
        X_train_full = data["X_train"]
        y_train_full = data["y_train"]
        X_test = data["X_test"][:config.SUBSAMPLE_TEST]
        y_test = data["y_test"][:config.SUBSAMPLE_TEST]

    all_results = {}
    models = ["FQK-SVM", "PQK-SVM", "RBF-SVM", "Linear-SVM"]

    # Pre-initialize results structure
    for model in models:
        all_results[model] = {N: [] for N in config.FEWSHOT_SIZES}

    for seed in config.SEED_LIST:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Seed: {seed}")
        logger.info(f"{'='*60}")

        for N in config.FEWSHOT_SIZES:
            logger.info(f"\n  N={N}, seed={seed}")

            # Get few-shot subset
            X_sub, y_sub = get_fewshot_subset(
                X_train_full, y_train_full, N, seed, strategy="stratified"
            )

            cache_dir = ensure_dir(os.path.join(results_dir, f"N{N}_seed{seed}"))

            # FQK
            K_fqk = compute_fqk_kernel_matrix(
                X_sub, save_path=os.path.join(cache_dir, "K_fqk.npy")
            )
            K_fqk_test = compute_fqk_kernel_matrix(
                X_sub, X2=X_test,
                save_path=os.path.join(cache_dir, "K_fqk_test.npy")
            )
            clf = train_precomputed_svm(K_fqk, y_sub, seed=seed)
            m = evaluate_classifier(clf, K_fqk_test, y_test, f"FQK N={N}")
            all_results["FQK-SVM"][N].append(m["macro_f1"])

            # PQK
            K_pqk = compute_pqk_kernel_matrix(
                X_sub, save_path=os.path.join(cache_dir, "K_pqk.npy"),
                bloch_save_path=os.path.join(cache_dir, "bloch.npy")
            )
            K_pqk_test = compute_pqk_kernel_matrix(
                X_sub, X2=X_test,
                save_path=os.path.join(cache_dir, "K_pqk_test.npy"),
                bloch_save_path=os.path.join(cache_dir, "bloch.npy")
            )
            clf = train_precomputed_svm(K_pqk, y_sub, seed=seed)
            m = evaluate_classifier(clf, K_pqk_test, y_test, f"PQK N={N}")
            all_results["PQK-SVM"][N].append(m["macro_f1"])

            # RBF
            K_rbf = compute_rbf_kernel(X_sub)
            K_rbf_test = compute_rbf_kernel(X_test, X_sub)
            clf = train_precomputed_svm(K_rbf, y_sub, seed=seed)
            m = evaluate_classifier(clf, K_rbf_test, y_test, f"RBF N={N}")
            all_results["RBF-SVM"][N].append(m["macro_f1"])

            # Linear SVM
            from sklearn.svm import SVC
            clf_lin = SVC(kernel="linear", class_weight="balanced", random_state=seed)
            clf_lin.fit(X_sub, y_sub)
            y_pred = clf_lin.predict(X_test)
            m_lin = compute_full_metrics(y_test, y_pred)
            all_results["Linear-SVM"][N].append(m_lin["macro_f1"])

    # ---- Aggregate ----
    plot_data = {}
    for model in models:
        means, stds = [], []
        for N in config.FEWSHOT_SIZES:
            vals = all_results[model][N]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        plot_data[model] = {"mean": np.array(means), "std": np.array(stds)}

    # ---- Learning curves plot ----
    plot_learning_curves(
        config.FEWSHOT_SIZES, plot_data,
        os.path.join(results_dir, "fig5_learning_curves"),
        title="Few-Shot Learning Curves (Macro-F1)",
    )

    # ---- Sample efficiency ratio ----
    # N at which quantum matches classical at N=2000
    rbf_at_2000 = plot_data["RBF-SVM"]["mean"][-1]
    for model in ["FQK-SVM", "PQK-SVM"]:
        means = plot_data[model]["mean"]
        for i, N in enumerate(config.FEWSHOT_SIZES):
            if means[i] >= rbf_at_2000:
                logger.info(f"  {model} matches RBF@N=2000 ({rbf_at_2000:.4f}) at N={N}")
                break

    # ---- AUC-LC ----
    for model in models:
        auc = np.trapz(plot_data[model]["mean"], x=np.log(config.FEWSHOT_SIZES))
        logger.info(f"  AUC-LC({model}) = {auc:.4f}")

    # ---- Wilcoxon signed-rank test ----
    logger.info("\n--- Wilcoxon Signed-Rank Tests ---")
    for N in config.FEWSHOT_SIZES:
        fqk_vals = all_results["FQK-SVM"][N]
        rbf_vals = all_results["RBF-SVM"][N]
        if len(fqk_vals) >= 5:
            try:
                stat, p = wilcoxon(fqk_vals, rbf_vals)
                logger.info(f"  N={N}: W={stat:.4f}, p={p:.4f} {'*' if p < 0.05 else ''}")
            except ValueError:
                logger.info(f"  N={N}: Wilcoxon not applicable (identical values?)")

    # ---- Save summary ----
    summary_rows = []
    for model in models:
        for i, N in enumerate(config.FEWSHOT_SIZES):
            summary_rows.append({
                "Model": model, "N": N,
                "Mean_F1": plot_data[model]["mean"][i],
                "Std_F1": plot_data[model]["std"][i],
            })
    pd.DataFrame(summary_rows).to_csv(os.path.join(results_dir, "fewshot_summary.csv"), index=False)

    save_results(
        {k: {str(n): v for n, v in vals.items()} for k, vals in all_results.items()},
        results_file
    )
    logger.info(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    run_experiment()
