"""
Experiment 11: Expressibility Analysis.

Compares expressibility of ZZFeatureMap across SAR, Optical, Fused modalities.
Tests mechanistic chain: More MI → More expressibility → Higher g → Higher KTA.

Phase 2.75 of the pipeline. Expected runtime: 1-2 hours.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import config
from src.quantum_kernels import compute_fqk_entry, _build_fqk_circuit
from src.expressibility import compute_expressibility, compare_modality_expressibility
from src.data_loader import load_modality, load_subsample
from src.utils import save_results, setup_logging, Timer, create_publication_plot, check_and_skip, ensure_dir

logger = setup_logging("exp11", log_file=os.path.join(config.RESULTS_DIR, "exp11.log"))


def run_experiment():
    """Run expressibility analysis."""
    results_dir = ensure_dir(os.path.join(config.RESULTS_DIR, "expressibility"))
    results_file = os.path.join(results_dir, "exp11_results.json")

    if check_and_skip(results_file, logger):
        return

    np.random.seed(config.RANDOM_SEED)

    # Build FQK circuit for fidelity computation
    circuit = _build_fqk_circuit(config.N_QUBITS, config.ZZ_REPS)

    def fidelity_fn(x1, x2):
        return compute_fqk_entry(x1, x2, circuit)

    # Load data
    # Load data — load each modality from its dedicated processed file
    # to guarantee correct SAR-only and optical-only features.
    # Use the same 2000-sample indices as the subsample for consistency.
    with Timer("Loading data", logger=logger):
        subsample = load_subsample()
        y_train = subsample["y_train"]

        # Try dedicated modality keys first (set by setup_data.py)
        # Fall back to loading from processed .npz files directly.
        if "sar_X_train" in subsample:
            X_sar = subsample["sar_X_train"]
            logger.info(f"  SAR features from subsample: {X_sar.shape}")
        else:
            sar_path = os.path.join(config.PROCESSED_DIR, "sar_pca8.npz")
            if os.path.exists(sar_path):
                sar_data = np.load(sar_path)
                # Use same indices as subsample — load train split
                X_sar = sar_data["X_train"][:config.SUBSAMPLE_TRAIN]
                logger.info(f"  SAR features from sar_pca8.npz: {X_sar.shape}")
            else:
                raise FileNotFoundError(
                    f"SAR features not found in subsample or {sar_path}.\n"
                    "Run setup_data.py to generate processed features."
                )

        if "optical_X_train" in subsample:
            X_opt = subsample["optical_X_train"]
            logger.info(f"  Optical features from subsample: {X_opt.shape}")
        else:
            opt_path = os.path.join(config.PROCESSED_DIR, "optical_pca8.npz")
            if os.path.exists(opt_path):
                opt_data = np.load(opt_path)
                X_opt = opt_data["X_train"][:config.SUBSAMPLE_TRAIN]
                logger.info(f"  Optical features from optical_pca8.npz: {X_opt.shape}")
            else:
                raise FileNotFoundError(
                    f"Optical features not found in subsample or {opt_path}.\n"
                    "Run setup_data.py to generate processed features."
                )

        X_fused = subsample["fused_X_train"]
        logger.info(f"  Fused features from subsample: {X_fused.shape}")

    # Verify that all three modalities are distinct
    # (if SAR == Optical, the fallback silently fired and results are wrong)
    if np.allclose(X_sar[:10], X_opt[:10]):
        raise RuntimeError(
            "SAR and Optical features are identical — the wrong fallback fired.\n"
            "Check that subsample_2000.npz contains 'sar_X_train' and "
            "'optical_X_train' as separate keys, or that sar_pca8.npz and "
            "optical_pca8.npz exist in the processed data directory."
        )
    if np.allclose(X_sar[:10], X_fused[:10]):
        raise RuntimeError(
            "SAR and Fused features are identical — all three modalities are "
            "the same data. Expressibility comparison would be meaningless."
        )
    logger.info("  Modality distinctness check: PASSED")

    # Ensure all have exactly N_QUBITS features for the circuit
    X_sar   = X_sar[:,   :config.N_QUBITS]
    X_opt   = X_opt[:,   :config.N_QUBITS]
    X_fused = X_fused[:, :config.N_QUBITS]

    # Use a subset for expressibility (expensive — 1000 fidelity evaluations each)
    n_expr = min(200, len(X_sar))
    X_sar_sub   = X_sar[:n_expr]
    X_opt_sub   = X_opt[:n_expr]
    X_fused_sub = X_fused[:n_expr]

    logger.info(f"  Expressibility subset: {n_expr} samples per modality, "
                f"{config.EXPRESSIBILITY_N_SAMPLES} pairs each")

    # ---- Compare modalities ----
    with Timer("Expressibility comparison", logger=logger):
        modality_results = compare_modality_expressibility(
            X_sar_sub, X_opt_sub, X_fused_sub,
            fidelity_fn, config.N_QUBITS,
            n_samples=config.EXPRESSIBILITY_N_SAMPLES,
        )

    # ---- Plots ----
    import matplotlib.pyplot as plt

    # Fidelity histograms overlaid with Haar
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, res) in zip(axes, modality_results.items()):
        ax.bar(res["bin_centers"], res["fidelity_histogram"], width=1.0/75,
               alpha=0.7, label=f"{name} (KL={res['kl_divergence']:.4f})", color="steelblue")
        ax.plot(res["bin_centers"], res["haar_reference"], "r--", linewidth=2, label="Haar reference")
        ax.set_xlabel("Fidelity F", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_title(f"{name}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Expressibility: Fidelity Distributions vs Haar Reference",
                fontsize=14, fontweight="bold")
    create_publication_plot(fig, os.path.join(results_dir, "fig11_expressibility_histograms"))

    # KL comparison bar chart
    fig2, ax = plt.subplots(figsize=(8, 5))
    names = list(modality_results.keys())
    kls = [modality_results[n]["kl_divergence"] for n in names]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    ax.bar(names, kls, color=colors, edgecolor="white", linewidth=1.5)
    ax.set_ylabel("KL Divergence from Haar", fontsize=12)
    ax.set_title("Expressibility Comparison (lower = more expressive)", fontsize=14, fontweight="bold")
    for i, (n, kl) in enumerate(zip(names, kls)):
        ax.text(i, kl + 0.001, f"{kl:.4f}", ha="center", fontsize=10)

    create_publication_plot(fig2, os.path.join(results_dir, "fig11b_kl_comparison"))

    # ---- Save ----
    results = {
        mod: {k: v for k, v in res.items() if k != "fidelities"}
        for mod, res in modality_results.items()
    }
    save_results(results, results_file)
    logger.info(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    run_experiment()
