"""
Experiment 10: Controlled Mutual Information Experiment.

Transforms the cross-modal finding from CORRELATIONAL to CAUSAL.

Varies alpha ∈ [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    X_opt_mixed = alpha * X_opt_real + (1-alpha) * X_noise
    alpha=0: pure noise (MI≈0), alpha=1: real data (MI=natural).

For each alpha: compute MI, FQK, g, ΔKTA, accuracy.
Repeats for 3 noise seeds. Reports error bars.

Phase 2.5 of the pipeline. Expected runtime: 4-6 hours.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from src.quantum_kernels import (
    compute_fqk_kernel_matrix,
    compute_crossmodal_fqk_kernel_matrix,
    get_top_mi_pairs,
)
from src.classical_kernels import compute_rbf_kernel, compute_tensor_product_kernel
from src.geometric_difference import compute_geometric_difference
from src.kernel_target_alignment import (
    compute_kta, compute_delta_kta, compute_centered_kta
)
from src.geometric_bound import compute_cross_modal_mutual_information
from src.data_loader import load_modality, load_subsample
from src.classifiers import train_precomputed_svm, evaluate_classifier
from src.feature_extraction import normalize_to_pi
from src.utils import (
    save_results, setup_logging, Timer, create_publication_plot,
    check_and_skip, ensure_dir,
)

logger = setup_logging("exp10", log_file=os.path.join(config.RESULTS_DIR, "exp10.log"))


def run_experiment(force: bool = False):
    """Run the controlled MI experiment."""
    results_dir = ensure_dir(os.path.join(config.RESULTS_DIR, "controlled_mi"))
    results_file = os.path.join(results_dir, "exp10_results.json")

    if not force and check_and_skip(results_file, logger):
        return
    elif force and os.path.exists(results_file):
        logger.info("[FORCE] Overriding existing results and rerunning.")

    np.random.seed(config.RANDOM_SEED)

    # ---- Load data ----
    with Timer("Loading data", logger=logger):
        subsample = load_subsample()
        # Load genuinely separate SAR and optical PCA features.
        # fused_pca8.npz uses joint PCA — [:, :4] is NOT pure SAR.
        # sar_pca8.npz and optical_pca8.npz are fitted independently.
        X_sar_full = subsample["sar_X_train"][:config.N_EXP10_TRAIN]
        X_opt_full = subsample["optical_X_train"][:config.N_EXP10_TRAIN]
        y_train    = subsample["y_train"][:config.N_EXP10_TRAIN]

        X_sar_test_full = subsample["sar_X_test"][:config.N_EXP10_TEST]
        X_opt_test_full = subsample["optical_X_test"][:config.N_EXP10_TEST]
        y_test          = subsample["y_test"][:config.N_EXP10_TEST]

        # Take first 4 components from each genuine modality.
        # Result: 8 features total with true 4+4 SAR/optical split.
        X_sar_base  = X_sar_full[:, :4]
        X_opt_real_base = X_opt_full[:, :4]
        X_train = np.hstack([X_sar_base, X_opt_real_base])

        X_sar_test_base  = X_sar_test_full[:, :4]
        X_opt_test_real_base = X_opt_test_full[:, :4]
        X_test = np.hstack([X_sar_test_base, X_opt_test_real_base])

    all_results = {"alpha_values": config.MIXING_RATIOS, "seeds": config.CONTROLLED_MI_NOISE_SEEDS}
    per_seed_results = []
    all_null_results = []

    for seed in config.CONTROLLED_MI_NOISE_SEEDS:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Noise seed: {seed}")
        logger.info(f"{'='*60}")

        rng = np.random.RandomState(seed)
        seed_results = []

        for alpha in config.MIXING_RATIOS:
            logger.info(f"\n--- Alpha = {alpha:.1f} (seed={seed}) ---")

            # X_sar and X_opt_real are now pre-split at load time (genuine modalities).
            X_sar_loop      = X_train[:, :4]
            X_opt_loop_real = X_train[:, 4:]
            X_noise = rng.uniform(0, np.pi, X_opt_loop_real.shape)

            X_opt_mixed = alpha * X_opt_loop_real + (1 - alpha) * X_noise
            # Normalize to [0, π]
            X_opt_mixed, _, _ = normalize_to_pi(X_opt_mixed)
            X_fused_mixed = np.hstack([X_sar_loop, X_opt_mixed])

            # Same for test
            X_sar_test = X_test[:, :4]
            X_opt_test_real = X_test[:, 4:]
            X_noise_test = rng.uniform(0, np.pi, X_opt_test_real.shape)
            X_opt_test_mixed = alpha * X_opt_test_real + (1 - alpha) * X_noise_test
            X_opt_test_mixed, _, _ = normalize_to_pi(X_opt_test_mixed)
            X_fused_test_mixed = np.hstack([X_sar_test, X_opt_test_mixed])

            # Compute MI
            mi = compute_cross_modal_mutual_information(X_sar_loop, X_opt_mixed)

            # Expressibility (KL from Haar) for this alpha
            from src.expressibility import compute_expressibility
            from src.quantum_kernels import _build_fqk_circuit, compute_fqk_entry

            circuit = _build_fqk_circuit(config.N_QUBITS, config.ZZ_REPS)
            def fid_fn(x1, x2):
                return compute_fqk_entry(x1, x2, circuit)

            n_expr = min(100, len(X_fused_mixed))
            kl_div = compute_expressibility(
                fid_fn, config.N_QUBITS,
                X_fused_mixed[:n_expr],
                n_samples=200, seed=seed,
            )["kl_divergence"]

            # Check if this (seed, alpha) is already complete
            cm_cache = os.path.join(results_dir, f"K_cm_fqk_alpha{alpha:.1f}_seed{seed}.npy")
            cm_cache_test = cm_cache.replace("K_cm_fqk_alpha", "K_cm_fqk_test_alpha")
            fqk_cache = os.path.join(results_dir, f"K_fqk_alpha{alpha:.1f}_seed{seed}.npy")
            
            if os.path.exists(cm_cache_test) and os.path.exists(fqk_cache):
                logger.info(f"  [SKIP] Alpha={alpha:.1f}, Seed={seed} already computed. Skipping kernels...")
            
            # --- Kernels ---
            # FQK (Standard entangling)
            with Timer(f"FQK Alpha={alpha:.1f} (seed={seed})", logger=logger):
                K_fqk = compute_fqk_kernel_matrix(X_fused_mixed, save_path=fqk_cache)
                K_fqk_test = compute_fqk_kernel_matrix(
                    X_fused_mixed, X2=X_fused_test_mixed,
                    save_path=fqk_cache.replace("K_fqk_alpha", "K_fqk_test_alpha")
                )

            # Classical kernels
            K_rbf = compute_rbf_kernel(X_fused_mixed)
            K_rbf_test = compute_rbf_kernel(X_fused_test_mixed, X_fused_mixed)
            K_tp = compute_tensor_product_kernel(X_fused_mixed)

            # Geometric difference
            g_tp, _ = compute_geometric_difference(K_fqk, K_tp)
            g_rbf, _ = compute_geometric_difference(K_fqk, K_rbf)

            # KTA
            kta_fqk = compute_centered_kta(K_fqk, y_train, class_weighted=True)
            kta_rbf = compute_centered_kta(K_rbf, y_train, class_weighted=True)
            delta_kta = kta_fqk - kta_rbf

            # Classification
            clf_fqk = train_precomputed_svm(K_fqk, y_train)
            metrics_fqk = evaluate_classifier(clf_fqk, K_fqk_test, y_test, f"FQK-α{alpha}")

            clf_rbf = train_precomputed_svm(K_rbf, y_train)
            metrics_rbf = evaluate_classifier(clf_rbf, K_rbf_test, y_test, f"RBF-α{alpha}")

            # ---- CM-FQK Addition ----
            # Compute MI pairs for this alpha's mixed data
            mi_pairs = get_top_mi_pairs(
                X_fused_mixed,
                n_sar=config.N_QUBITS // 2,
                n_opt=config.N_QUBITS // 2,
                top_k=config.CM_FQK_TOP_K_PAIRS,
            )

            cm_cache = os.path.join(
                results_dir,
                f"K_cm_fqk_alpha{alpha:.1f}_seed{seed}.npy"
            )
            K_cm_fqk = compute_crossmodal_fqk_kernel_matrix(
                X_fused_mixed,
                mi_pairs=mi_pairs,
                save_path=cm_cache,
            )

            cm_cache_test = cm_cache.replace("K_cm_fqk_alpha", "K_cm_fqk_test_alpha")
            K_cm_fqk_test = compute_crossmodal_fqk_kernel_matrix(
                X_fused_mixed,
                mi_pairs=mi_pairs,
                X2=X_fused_test_mixed,
                save_path=cm_cache_test,
            )

            g_cm_tp, _ = compute_geometric_difference(K_cm_fqk, K_tp)
            g_cm_rbf, _ = compute_geometric_difference(K_cm_fqk, K_rbf)

            kta_cm_fqk = compute_centered_kta(K_cm_fqk, y_train, class_weighted=True)
            delta_kta_cm = kta_cm_fqk - kta_rbf

            clf_cm = train_precomputed_svm(K_cm_fqk, y_train)
            metrics_cm = evaluate_classifier(clf_cm, K_cm_fqk_test, y_test, f"CM-FQK-α{alpha}")
            # -------------------------

            row = {
                "alpha": alpha,
                "seed": seed,
                "mi": mi,
                "kl_divergence": kl_div,
                "g_tensor": g_tp,
                "g_rbf": g_rbf,
                "kta_fqk": kta_fqk,
                "kta_rbf": kta_rbf,
                "delta_kta": delta_kta,
                "f1_fqk": metrics_fqk["macro_f1"],
                "f1_rbf": metrics_rbf["macro_f1"],
                "acc_fqk": metrics_fqk["accuracy"],
                "acc_rbf": metrics_rbf["accuracy"],
                "g_cm_tensor": g_cm_tp,
                "g_cm_rbf": g_cm_rbf,
                "kta_cm_fqk": kta_cm_fqk,
                "delta_kta_cm": delta_kta_cm,
                "f1_cm_fqk": metrics_cm["macro_f1"],
                "acc_cm_fqk": metrics_cm["accuracy"],
                "n_mi_pairs": len(mi_pairs),
            }
            seed_results.append(row)

            logger.info(
                f"  α={alpha:.1f}: MI={mi:.4f}, "
                f"g_FQK={g_tp:.4f}, g_CM={g_cm_tp:.4f}, "
                f"ΔKTA_FQK={delta_kta:.4f}, ΔKTA_CM={delta_kta_cm:.4f}, "
                f"F1_FQK={metrics_fqk['macro_f1']:.4f}, F1_CM={metrics_cm['macro_f1']:.4f}"
            )

        per_seed_results.append(seed_results)
        logger.info(
            f"\n  [COMPLETE] Seed {seed}: "
            f"{len(seed_results)}/{len(config.MIXING_RATIOS)} alpha values done."
        )
        if len(seed_results) < len(config.MIXING_RATIOS):
            logger.warning(
                f"  [INCOMPLETE] Seed {seed} only has {len(seed_results)} alpha values. "
                f"Expected {len(config.MIXING_RATIOS)}. Results will be partial."
            )

        # ---- Null control: corrupt SAR, keep optical real ----
        # If g/ΔKTA changes when we corrupt SAR but NOT when we corrupt
        # optical, this would suggest the effect is SAR-specific, not
        # cross-modal. We expect BOTH corruptions to reduce g, confirming
        # the cross-modal nature of the effect.
        logger.info(f"\n--- Null Control: SAR corruption (seed={seed}) ---")
        null_results = []
        for alpha in config.MIXING_RATIOS:
            X_sar_real = X_train[:, :4]
            X_opt = X_train[:, 4:]
            X_noise_sar = rng.uniform(0, np.pi, X_sar_real.shape)
            X_sar_mixed = alpha * X_sar_real + (1 - alpha) * X_noise_sar
            X_sar_mixed, _, _ = normalize_to_pi(X_sar_mixed)
            X_fused_sar_corrupted = np.hstack([X_sar_mixed, X_opt])

            # Check if this null control (seed, alpha) is already complete
            null_cache = os.path.join(results_dir, f"K_null_fqk_alpha{alpha:.1f}_seed{seed}.npy")
            
            if os.path.exists(null_cache):
                logger.info(f"  [SKIP] Null α={alpha:.1f}, Seed={seed} already computed.")

            mi_null = compute_cross_modal_mutual_information(X_sar_mixed, X_opt)
            K_null = compute_fqk_kernel_matrix(X_fused_sar_corrupted, save_path=null_cache)
            K_tp_null = compute_tensor_product_kernel(X_fused_sar_corrupted)
            g_null, _ = compute_geometric_difference(K_null, K_tp_null)
            kta_null = compute_centered_kta(K_null, y_train, class_weighted=True)
            kta_rbf_null = compute_centered_kta(compute_rbf_kernel(X_fused_sar_corrupted), y_train, class_weighted=True)
            delta_kta_null = kta_null - kta_rbf_null

            null_results.append({
                "alpha": alpha, "seed": seed,
                "mi_null": mi_null, "g_null": g_null, "delta_kta_null": delta_kta_null,
                "corruption": "SAR",
            })
            logger.info(f"  α={alpha:.1f} [SAR corrupt]: MI={mi_null:.4f}, g={g_null:.4f}, ΔKTA={delta_kta_null:.4f}")

        all_null_results.extend(null_results)

    # ---- Aggregate results ----
    flat_results = [r for seed_res in per_seed_results for r in seed_res]
    
    # Completeness check
    expected_seeds  = config.CONTROLLED_MI_NOISE_SEEDS
    expected_alphas = config.MIXING_RATIOS
    completed = {(r["seed"], round(r["alpha"], 1)) for r in flat_results}
    missing   = [
        (s, a) for s in expected_seeds for a in expected_alphas
        if (s, round(a, 1)) not in completed
    ]

    if missing:
        logger.warning(f"  INCOMPLETE DATA: {len(missing)} seed/alpha combos missing:")
        for s, a in missing:
            logger.warning(f"    seed={s}, alpha={a:.1f} — NOT in results")
        logger.warning(
            "  Summary std values will be NaN for missing alphas. "
            "Rerun with missing seeds before finalising paper results."
        )
    else:
        logger.info(
            f"  Data completeness: ALL "
            f"{len(expected_seeds) * len(expected_alphas)} combinations present."
        )

    df = pd.DataFrame(flat_results)
    df.to_csv(os.path.join(results_dir, "controlled_mi_results.csv"), index=False)

    # Drop any non-numeric columns before aggregation
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning(f"Dropping non-numeric columns before agg: {non_numeric}")
    df_numeric = df.select_dtypes(include=[np.number])

    # Compute mean +/- std over seeds
    summary = df_numeric.groupby(df["alpha"]).agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(c) for c in col).strip("_") for col in summary.columns
    ]
    summary.to_csv(os.path.join(results_dir, "controlled_mi_summary.csv"), index=False)

    logger.info(f"\n{'='*60}")
    logger.info("  Controlled MI Summary (mean over seeds)")
    logger.info(f"{'='*60}")
    logger.info(f"\n{summary.to_string()}")

    # ---- Plots ----
    import matplotlib.pyplot as plt

    alphas = config.MIXING_RATIOS
    mi_mean = summary["mi_mean"].values
    g_mean = summary["g_tensor_mean"].values
    delta_kta_mean = summary["delta_kta_mean"].values
    f1_fqk_mean = summary["f1_fqk_mean"].values
    f1_rbf_mean = summary["f1_rbf_mean"].values
    f1_fqk_std = summary["f1_fqk_std"].values
    f1_rbf_std = summary["f1_rbf_std"].values

    # Plot 1: g, ΔKTA, MI vs alpha (triple axis)
    fig, ax1 = plt.subplots(figsize=(10, 7))
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))

    l1 = ax1.plot(alphas, g_mean, "o-", color="steelblue", linewidth=2, label="g (geometric diff)")
    l2 = ax2.plot(alphas, delta_kta_mean, "s-", color="coral", linewidth=2, label="ΔKTA")
    l3 = ax3.plot(alphas, mi_mean, "^-", color="green", linewidth=2, label="MI")

    ax1.set_xlabel("Mixing ratio α", fontsize=12)
    ax1.set_ylabel("g (geometric difference)", fontsize=12, color="steelblue")
    ax2.set_ylabel("ΔKTA", fontsize=12, color="coral")
    ax3.set_ylabel("Mutual Information", fontsize=12, color="green")

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc="upper left")
    ax1.set_title("Controlled MI: g, ΔKTA, MI vs α", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    create_publication_plot(fig, os.path.join(results_dir, "fig4_controlled_mi_triple"))

    # Plot 2: FQK-SVM vs RBF-SVM accuracy
    fig2, ax = plt.subplots(figsize=(10, 7))
    ax.plot(alphas, f1_fqk_mean, "o-", linewidth=2, label="FQK-SVM", color="steelblue")
    ax.fill_between(alphas, f1_fqk_mean - f1_fqk_std, f1_fqk_mean + f1_fqk_std,
                    alpha=0.2, color="steelblue")
    ax.plot(alphas, f1_rbf_mean, "s-", linewidth=2, label="RBF-SVM", color="coral")
    ax.fill_between(alphas, f1_rbf_mean - f1_rbf_std, f1_rbf_mean + f1_rbf_std,
                    alpha=0.2, color="coral")

    ax.set_xlabel("Mixing ratio α", fontsize=12)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title("Classification Performance vs Cross-Modal MI", fontsize=14, fontweight="bold")

    create_publication_plot(fig2, os.path.join(results_dir, "fig4b_fqk_vs_rbf_alpha"))

    # Plot 3: Null Control (Optical vs SAR corruption)
    df_null = pd.DataFrame(all_null_results)
    df_null.to_csv(os.path.join(results_dir, "controlled_mi_null_control.csv"), index=False)

    # Drop non-numeric columns before aggregation to prevent
    # TypeError: dtype 'str' does not support operation 'mean'
    numeric_cols_null = ["alpha", "mi_null", "g_null", "delta_kta_null"]
    df_null_numeric   = df_null[numeric_cols_null].copy()
    summary_null = df_null_numeric.groupby("alpha").agg(["mean", "std"]).reset_index()
    summary_null.columns = [
        "_".join(str(c) for c in col).strip("_") for col in summary_null.columns
    ]
    
    fig3, ax = plt.subplots(figsize=(10, 7))
    g_opt_mean = summary["g_tensor_mean"].values
    g_opt_std = summary["g_tensor_std"].values
    g_sar_mean = summary_null["g_null_mean"].values
    g_sar_std = summary_null["g_null_std"].values
    
    ax.plot(alphas, g_opt_mean, "o-", linewidth=2, label="Corrupting Optical (Original)", color="blue")
    ax.fill_between(alphas, g_opt_mean - g_opt_std, g_opt_mean + g_opt_std, alpha=0.2, color="blue")
    
    ax.plot(alphas, g_sar_mean, "s-", linewidth=2, label="Corrupting SAR (Null Control)", color="red")
    ax.fill_between(alphas, g_sar_mean - g_sar_std, g_sar_mean + g_sar_std, alpha=0.2, color="red")
    
    ax.set_xlabel("Mixing ratio α", fontsize=12)
    ax.set_ylabel("g (geometric difference)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title("Null Control: Cross-Modal Specificity Check", fontsize=14, fontweight="bold")
    
    create_publication_plot(fig3, os.path.join(results_dir, "fig5_null_control_comparison"))

    # Plot 4: FQK vs CM-FQK geometric difference as function of alpha
    fig4, ax = plt.subplots(figsize=(10, 7))

    g_fqk_mean  = summary["g_tensor_mean"].values
    g_fqk_std   = summary["g_tensor_std"].values
    g_cm_mean   = summary["g_cm_tensor_mean"].values
    g_cm_std    = summary["g_cm_tensor_std"].values

    # Normalize g values to [0,1] for visual comparison of SLOPES
    g_fqk_norm = (g_fqk_mean - g_fqk_mean.min()) / (g_fqk_mean.max() - g_fqk_mean.min() + 1e-10)
    g_cm_norm  = (g_cm_mean  - g_cm_mean.min())  / (g_cm_mean.max()  - g_cm_mean.min() + 1e-10)

    ax.plot(alphas, g_fqk_norm, "o-", linewidth=2.5,
            label="FQK (linear ZZ)", color="steelblue")
    ax.fill_between(alphas,
                    g_fqk_norm - g_fqk_std / (g_fqk_mean.max() - g_fqk_mean.min() + 1e-10),
                    g_fqk_norm + g_fqk_std / (g_fqk_mean.max() - g_fqk_mean.min() + 1e-10),
                    alpha=0.2, color="steelblue")

    ax.plot(alphas, g_cm_norm, "s-", linewidth=2.5,
            label="CM-FQK (cross-modal ZZ)", color="darkorange")
    ax.fill_between(alphas,
                    g_cm_norm - g_cm_std / (g_cm_mean.max() - g_cm_mean.min() + 1e-10),
                    g_cm_norm + g_cm_std / (g_cm_mean.max() - g_cm_mean.min() + 1e-10),
                    alpha=0.2, color="darkorange")

    ax.set_xlabel("Mixing ratio α (0=pure noise, 1=real MI)", fontsize=12)
    ax.set_ylabel("Normalised geometric difference g (relative to α=1.0)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_title(
        "FQK vs CM-FQK: Geometric Difference Sensitivity to Cross-Modal MI",
        fontsize=13, fontweight="bold"
    )
    ax.annotate(
        "If CM-FQK slope > FQK slope:\nentangled encoding\nexploits MI structure",
        xy=(0.05, 0.5), xycoords="axes fraction",
        fontsize=9, color="gray",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7)
    )

    create_publication_plot(fig4, os.path.join(results_dir, "fig6_fqk_vs_cm_g_sensitivity"))

    # ---- Save ----
    all_results["summary"] = summary.to_dict()
    all_results["raw_data"] = df.to_dict()
    save_results(all_results, results_file)
    logger.info(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Force rerun even if results file exists.")
    args = parser.parse_args()
    run_experiment(force=args.force)
