"""
Experiment 5: Equivariant QNN Comparison.

4 models: Equivariant QNN, Non-equivariant QNN, Classical equivariant MLP, Classical MLP.
Few-shot across N. Report accuracy, F1, param count, convergence speed.

Phase 6. Expected runtime: 2-4 days (QNN training is expensive).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch

import config
from src.equivariant_qnn import EquivariantQNN, ClassicalMLP, EquivariantMLP, train_model, predict
from src.data_loader import load_modality, get_fewshot_subset
from src.utils import save_results, setup_logging, Timer, check_and_skip, ensure_dir, compute_full_metrics, create_publication_plot

logger = setup_logging("exp5", log_file=os.path.join(config.RESULTS_DIR, "exp5.log"))


def run_experiment():
    """Equivariant comparison experiment."""
    results_dir = ensure_dir(os.path.join(config.RESULTS_DIR, "equivariant"))
    results_file = os.path.join(results_dir, "exp5_results.json")

    if check_and_skip(results_file, logger):
        return

    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    # Load data
    with Timer("Loading data", logger=logger):
        data = load_modality("fused")
        X_train_full = data["X_train"]
        y_train_full = data["y_train"]
        X_test = data["X_test"][:config.SUBSAMPLE_TEST]
        y_test = data["y_test"][:config.SUBSAMPLE_TEST]

    fewshot_sizes = [100, 200, 500]  # Limited sizes for QNN training speed
    results = {}

    for N in fewshot_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"  N = {N}")
        logger.info(f"{'='*60}")

        X_sub, y_sub = get_fewshot_subset(X_train_full, y_train_full, N, config.RANDOM_SEED)

        n_results = {}

        # Classical MLP
        logger.info("\n--- Classical MLP ---")
        mlp = ClassicalMLP()
        history = train_model(mlp, X_sub, y_sub, epochs=config.EQUIVARIANT_EPOCHS)
        y_pred = predict(mlp, X_test)
        m = compute_full_metrics(y_test, y_pred)
        params = sum(p.numel() for p in mlp.parameters())
        n_results["Classical MLP"] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "params": params}

        # Equivariant MLP
        logger.info("\n--- Equivariant MLP ---")
        eq_mlp = EquivariantMLP()
        history = train_model(eq_mlp, X_sub, y_sub, epochs=config.EQUIVARIANT_EPOCHS)
        y_pred = predict(eq_mlp, X_test)
        m = compute_full_metrics(y_test, y_pred)
        params = sum(p.numel() for p in eq_mlp.parameters())
        n_results["Equivariant MLP"] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "params": params}

        # Non-equivariant QNN (skip if N > 200 to save time)
        if N <= 200:
            logger.info("\n--- Non-Equivariant QNN ---")
            try:
                qnn = EquivariantQNN(equivariant=False)
                qnn_params = qnn.count_parameters()
                history = train_model(qnn, X_sub, y_sub, epochs=min(20, config.EQUIVARIANT_EPOCHS), batch_size=8)
                y_pred = predict(qnn, X_test, batch_size=8)
                m = compute_full_metrics(y_test, y_pred)
                n_results["Non-Equivariant QNN"] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "params": qnn_params["total"]}
            except Exception as e:
                logger.error(f"QNN training failed: {e}")
                n_results["Non-Equivariant QNN"] = {"macro_f1": 0.0, "accuracy": 0.0, "params": 0, "error": str(e)}

            # Equivariant QNN
            logger.info("\n--- Equivariant QNN ---")
            try:
                eq_qnn = EquivariantQNN(equivariant=True)
                eq_params = eq_qnn.count_parameters()
                history = train_model(eq_qnn, X_sub, y_sub, epochs=min(20, config.EQUIVARIANT_EPOCHS), batch_size=8)
                y_pred = predict(eq_qnn, X_test, batch_size=8)
                m = compute_full_metrics(y_test, y_pred)
                n_results["Equivariant QNN"] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "params": eq_params["total"]}
            except Exception as e:
                logger.error(f"Equivariant QNN failed: {e}")
                n_results["Equivariant QNN"] = {"macro_f1": 0.0, "accuracy": 0.0, "params": 0, "error": str(e)}

        results[f"N={N}"] = n_results

    # Summary table
    table_rows = []
    for n_key, n_res in results.items():
        for model, m in n_res.items():
            table_rows.append({"N": n_key, "Model": model, **m})

    df = pd.DataFrame(table_rows)
    df.to_csv(os.path.join(results_dir, "equivariant_results.csv"), index=False)
    logger.info(f"\n{df.to_string()}")

    save_results(results, results_file)
    logger.info(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    run_experiment()
