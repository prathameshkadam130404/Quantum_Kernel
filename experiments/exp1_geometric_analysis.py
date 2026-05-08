"""
Experiment 1: Geometric Difference + KTA + Concentration Analysis.

Computes geometric difference g, kernel-target alignment (KTA), ΔKTA,
and concentration diagnostics for all modality × kernel combinations.

Key outputs:
    - g for every quantum-vs-classical pair
    - KTA and ΔKTA per modality
    - Concentration CV + eigenvalue spectra
    - MI scatter: g vs cross-modal MI
    - Tables and plots for the paper

Phase 2 of the pipeline. Expected runtime: 3-5 hours.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import logging
import numpy as np
import pandas as pd

import config
from src.quantum_kernels import compute_fqk_kernel_matrix, compute_pqk_kernel_matrix, compute_tfk_kernel_matrix, train_quantum_kernel_kta, compute_crossmodal_fqk_kernel_matrix, get_top_mi_pairs
from src.classical_kernels import compute_rbf_kernel, compute_tensor_product_kernel, compute_all_classical_kernels
from src.geometric_difference import compute_geometric_difference, compute_geometric_difference_batch, compute_geometric_difference_lambda_sweep
from src.kernel_target_alignment import compute_kta, compute_centered_kta, compute_delta_kta, compute_full_kta_analysis
from src.kernel_concentration import compute_concentration_metrics, compare_concentration
from src.geometric_bound import compute_cross_modal_mutual_information, analyze_geometric_amplification
from src.data_loader import load_modality, load_subsample
from src.classifiers import train_precomputed_svm, evaluate_classifier
from src.utils import (
    save_results, setup_logging, Timer, create_publication_plot,
    plot_eigenvalue_spectrum, plot_kernel_heatmap, check_and_skip, ensure_dir,
    validate_test_kernel_shape,
)
from src.bandwidth import log_bandwidth_info

logger = setup_logging("exp1", log_file=os.path.join(config.RESULTS_DIR, "exp1.log"))


def run_experiment():
    """Main experiment: geometric analysis across modalities."""
    results_dir = ensure_dir(os.path.join(config.RESULTS_DIR, "geometric_difference"))
    results_file = os.path.join(results_dir, "exp1_results.json")

    if check_and_skip(results_file, logger):
        return

    np.random.seed(config.RANDOM_SEED)

    all_results = {}

    # ---- Load data ----
    with Timer("Loading data", logger=logger):
        subsample = load_subsample()

    modalities = {}
    for mod in ["sar", "optical", "fused"]:
        key_x = f"{mod}_X_train"
        if key_x in subsample:
            modalities[mod] = {
                "X_train": subsample[key_x],
                "X_test": subsample[f"{mod}_X_test"],
            }
    y_train = subsample["y_train"]
    y_test = subsample["y_test"]



    # Quantum kernels only for SAR + Fused (from config)
    QUANTUM_MODALITIES = config.QUANTUM_MODALITIES

    logger.info(f"Loaded {len(modalities)} modalities, {len(y_train)} train, {len(y_test)} test")
    logger.info(f"Quantum kernels for: {QUANTUM_MODALITIES}")

    # ---- Compute kernels per modality ----
    kernel_matrices = {}

    for mod_name, mod_data in modalities.items():
        X_train = mod_data["X_train"]
        X_test = mod_data["X_test"]
        mod_dir = ensure_dir(os.path.join(results_dir, mod_name))

        logger.info(f"\n{'='*60}")
        logger.info(f"  Modality: {mod_name.upper()} ({X_train.shape})")
        logger.info(f"{'='*60}")

        if mod_name == "fused":
            # Log bandwidth info — γ=1.0 (CV-selected), encoding range [0, π]
            log_bandwidth_info(X_train, "pca", logger=logger)

        compute_quantum = mod_name in QUANTUM_MODALITIES

        K_fqk = K_fqk_test = K_pqk = K_pqk_test = None

        if compute_quantum:
            # Quantum kernels
            with Timer(f"FQK kernel ({mod_name})", logger=logger):
                K_fqk = compute_fqk_kernel_matrix(
                    X_train, save_path=os.path.join(mod_dir, "K_fqk_train.npy"),
                )
                K_fqk_test = compute_fqk_kernel_matrix(
                    X_train, X2=X_test,
                    save_path=os.path.join(mod_dir, "K_fqk_test.npy"),
                )

            with Timer(f"PQK kernel ({mod_name})", logger=logger):
                K_pqk = compute_pqk_kernel_matrix(
                    X_train, save_path=os.path.join(mod_dir, "K_pqk_train.npy"),
                    bloch_save_path=os.path.join(mod_dir, "bloch_train.npy"),
                )
                K_pqk_test = compute_pqk_kernel_matrix(
                    X_train, X2=X_test,
                    save_path=os.path.join(mod_dir, "K_pqk_test.npy"),
                    bloch_save_path=os.path.join(mod_dir, "bloch_train.npy"),
                )
                validate_test_kernel_shape(K_pqk_test, X_train, X_test, f"PQK-{mod_name}")

            # TFK and CrossModal (fused only)
            if mod_name == "fused":
                tfk_dir = ensure_dir(os.path.join(mod_dir, "tfk"))
                with Timer("TFK training", logger=logger):
                    theta_trained = train_quantum_kernel_kta(
                        X_train, y_train,
                        save_path=os.path.join(tfk_dir, "theta_trained.npy"),
                    )
                K_tfk = compute_tfk_kernel_matrix(
                    X_train, theta_trained,
                    save_path=os.path.join(tfk_dir, "K_tfk_train.npy"),
                )
                K_tfk_test = compute_tfk_kernel_matrix(
                    X_train, theta_trained, X2=X_test,
                    save_path=os.path.join(tfk_dir, "K_tfk_test.npy"),
                )

                # CrossModal FQK (fused only)
                mi_pairs = get_top_mi_pairs(X_train, n_sar=4, n_opt=4, top_k=3)
                K_cm_fqk = compute_crossmodal_fqk_kernel_matrix(
                    X_train, mi_pairs,
                    save_path=os.path.join(mod_dir, "K_cm_fqk_train.npy"),
                )
                K_cm_fqk_test = compute_crossmodal_fqk_kernel_matrix(
                    X_train, mi_pairs, X2=X_test,
                    save_path=os.path.join(mod_dir, "K_cm_fqk_test.npy"),
                )

        else:
            logger.info(f"  [SKIP] Quantum kernels for {mod_name} (classical PCA captures 84% variance)")

        # Classical kernels (always computed — instant)
        with Timer(f"Classical kernels ({mod_name})", logger=logger):
            K_rbf = compute_rbf_kernel(X_train, save_path=os.path.join(mod_dir, "K_rbf_train.npy"))
            K_rbf_test = compute_rbf_kernel(X_train, X2=X_test)

            classical_kernels = compute_all_classical_kernels(
                X_train, save_dir=mod_dir, prefix=f"",
            )

        # ---- Classical Kernel Conditioning Diagnostics ----
        logger.info(f"\n--- Classical Kernel Conditioning ({mod_name}) ---")
        conditioning_results = []
        for c_name, K_c in classical_kernels.items():
            from src.kernel_concentration import compute_concentration_metrics
            c_metrics = compute_concentration_metrics(K_c)
            eff_rank = c_metrics["effective_rank"]
            cond_num = c_metrics.get("condition_number", float('nan'))
            mean_od = c_metrics["mean_offdiag"]
            
            conditioning_results.append({
                "kernel": c_name,
                "effective_rank": eff_rank,
                "condition_number": cond_num,
                "mean_offdiag": mean_od,
            })
            
            logger.info(
                f"  {c_name}: eff_rank={eff_rank:.1f}, "
                f"cond_num={cond_num:.2e}, mean_offdiag={mean_od:.4f}"
            )

        cond_df = pd.DataFrame(conditioning_results)
        cond_df.to_csv(os.path.join(mod_dir, "classical_kernel_conditioning.csv"), index=False)

        kernel_matrices[mod_name] = {
            "RBF": K_rbf, "RBF_test": K_rbf_test,
            **{k: v for k, v in classical_kernels.items()},
        }
        if compute_quantum:
            kernel_matrices[mod_name].update({
                "FQK": K_fqk, "PQK": K_pqk,
                "FQK_test": K_fqk_test, "PQK_test": K_pqk_test,
            })
            if mod_name == "fused":
                kernel_matrices[mod_name].update({
                    "TFK": K_tfk, "TFK_test": K_tfk_test,
                    "CM_FQK": K_cm_fqk, "CM_FQK_test": K_cm_fqk_test,
                })

            # ---- PQK Gamma Sweep (fused only) ----
            if mod_name == "fused":
                logger.info("\n--- PQK Gamma Sweep (fused) ---")
                gamma_results = {}
                for gamma in config.PQK_GAMMA_LIST:
                    K_pqk_g = compute_pqk_kernel_matrix(
                        X_train, gamma=gamma,
                        save_path=os.path.join(mod_dir, f"K_pqk_gamma{gamma}.npy"),
                        bloch_save_path=os.path.join(mod_dir, "bloch_train.npy"),
                    )
                    kta_g = compute_centered_kta(K_pqk_g, y_train, class_weighted=True)
                    g_g, _ = compute_geometric_difference(K_pqk_g, K_rbf)
                    gamma_results[gamma] = {"kta": kta_g, "g": g_g}
                    logger.info(f"  gamma={gamma}: KTA={kta_g:.4f}, g={g_g:.4f}")

                best_gamma = max(gamma_results, key=lambda g: gamma_results[g]["kta"])
                logger.info(f"  Best gamma by KTA: {best_gamma} (KTA={gamma_results[best_gamma]['kta']:.4f})")
                mod_results["pqk_gamma_sweep"] = gamma_results
                mod_results["best_pqk_gamma"] = best_gamma

        # ---- Geometric difference (quantum modalities only) ----
        if compute_quantum:
            logger.info(f"\n--- Geometric Difference ({mod_name}) ---")
            quantum_kernels = {"FQK": K_fqk, "PQK": K_pqk}
            if mod_name == "fused" and "TFK" in kernel_matrices[mod_name]:
                quantum_kernels["TFK"] = kernel_matrices[mod_name]["TFK"]
                if "CM_FQK" in kernel_matrices[mod_name]:
                    quantum_kernels["CM_FQK"] = kernel_matrices[mod_name]["CM_FQK"]
            g_results = compute_geometric_difference_batch(quantum_kernels, classical_kernels)

            if mod_name == "fused" and "RBF" in kernel_matrices[mod_name]:
                logger.info(f"\n--- Geometric Difference Lambda Sweep ({mod_name}) ---")
                sweep_results = compute_geometric_difference_lambda_sweep(
                    K_fqk, kernel_matrices[mod_name]["RBF"],
                    save_path=os.path.join(mod_dir, "lambda_sweep.png")
                )
                logger.info(f"  Recommended lambda: {sweep_results['recommended_lambda']:.2e} (g={sweep_results['recommended_g']:.4f})")
                logger.info(f"  Sweep stable: {sweep_results['stable']}")

            mod_results = {"geometric_difference": {}}
            for key, val in g_results.items():
                g_val = val.get("g", float("nan"))
                mod_results["geometric_difference"][key] = g_val
                logger.info(f"  g({key}) = {g_val:.4f}")
        else:
            mod_results = {
                "geometric_difference": {},
                "bandwidth_gamma": 1.0,
                "bandwidth_source": "5-fold CV, Shaydulin & Wild 2022",
                "encoding_range": "0_to_pi",
            }

        # ---- KTA ----
        logger.info(f"\n--- KTA ({mod_name}) ---")
        mod_results["kta"] = {}
        kernels_to_eval = [("RBF", K_rbf)]
        if compute_quantum:
            kernels_to_eval = [("FQK", K_fqk), ("PQK", K_pqk)] + kernels_to_eval
            if mod_name == "fused" and "TFK" in kernel_matrices[mod_name]:
                kernels_to_eval.insert(0, ("TFK", kernel_matrices[mod_name]["TFK"]))
                if "CM_FQK" in kernel_matrices[mod_name]:
                    kernels_to_eval.insert(1, ("CM_FQK", kernel_matrices[mod_name]["CM_FQK"]))
        for k_name, K in kernels_to_eval:
            kta_full = compute_full_kta_analysis(K, y_train)
            mod_results["kta"][k_name] = kta_full
            kta_vals = {k: f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for k, v in kta_full.items()}
            logger.info(f"  KTA({k_name}): {kta_vals}")

        if compute_quantum and "FQK" in mod_results["kta"] and "PQK" in mod_results["kta"] and "RBF" in mod_results["kta"]:
            kta_fqk = mod_results["kta"]["FQK"]
            kta_pqk = mod_results["kta"]["PQK"]
            kta_rbf = mod_results["kta"]["RBF"]
            logger.info("")
            logger.info("  --- Centered Weighted KTA Summary (PRIMARY metric) ---")
            logger.info(f"  FQK: {float(kta_fqk['kta_centered_weighted']):.4f}")
            logger.info(f"  PQK: {float(kta_pqk['kta_centered_weighted']):.4f}")
            if "TFK" in mod_results["kta"]:
                kta_tfk = mod_results["kta"]["TFK"]
                logger.info(f"  TFK: {float(kta_tfk['kta_centered_weighted']):.4f}")
            if "CM_FQK" in mod_results["kta"]:
                kta_cm = mod_results["kta"]["CM_FQK"]
                logger.info(f"  CM_FQK: {float(kta_cm['kta_centered_weighted']):.4f}")
            logger.info(f"  RBF: {float(kta_rbf['kta_centered_weighted']):.4f}")
            logger.info(f"  ΔKTA_centered(FQK - RBF) = {float(kta_fqk['kta_centered_weighted']) - float(kta_rbf['kta_centered_weighted']):+.4f}")
            logger.info(f"  ΔKTA_centered(PQK - RBF) = {float(kta_pqk['kta_centered_weighted']) - float(kta_rbf['kta_centered_weighted']):+.4f}")
            if "TFK" in mod_results["kta"]:
                logger.info(f"  ΔKTA_centered(TFK - RBF) = {float(kta_tfk['kta_centered_weighted']) - float(kta_rbf['kta_centered_weighted']):+.4f}")
            if "CM_FQK" in mod_results["kta"]:
                logger.info(f"  ΔKTA_centered(CM_FQK - RBF) = {float(kta_cm['kta_centered_weighted']) - float(kta_rbf['kta_centered_weighted']):+.4f}")
            logger.info("")
            logger.info("  --- Standard Weighted KTA Summary (shown for comparison only) ---")
            logger.info(f"  FQK: {float(kta_fqk['kta_weighted']):.4f}")
            logger.info(f"  PQK: {float(kta_pqk['kta_weighted']):.4f}")
            logger.info(f"  RBF: {float(kta_rbf['kta_weighted']):.4f}")
            logger.info(f"  Note: RBF mean_offdiag≈0.80 inflates standard KTA — use centered values above.")

        # ΔKTA
        mod_results["delta_kta"] = {}
        if compute_quantum:
            logger.info("")
            logger.info("--- ΔKTA Summary ---")
            logger.info("  PRIMARY metric: centered + class-weighted KTA")
            logger.info("  Reference: Cortes, Mohri & Rostamizadeh (JMLR 2012)")
            logger.info("  Standard KTA shown alongside for transparency only.")
            logger.info("  RBF mean_offdiag=0.8048 — high value inflates standard KTA numerator artificially.")
            
            q_kernels_to_eval = [("FQK", K_fqk), ("PQK", K_pqk)]
            if mod_name == "fused" and "TFK" in kernel_matrices[mod_name]:
                q_kernels_to_eval.append(("TFK", kernel_matrices[mod_name]["TFK"]))
                if "CM_FQK" in kernel_matrices[mod_name]:
                    q_kernels_to_eval.append(("CM_FQK", kernel_matrices[mod_name]["CM_FQK"]))
            
            for q_name, K_q in q_kernels_to_eval:
                for c_name, K_c in classical_kernels.items():
                    key = f"{q_name}_vs_{c_name}"
                    delta_centered = compute_delta_kta(K_q, K_c, y_train, use_centered=True, class_weighted=True)
                    delta_standard = compute_delta_kta(K_q, K_c, y_train, use_centered=False, class_weighted=True)
                    mod_results["delta_kta"][key] = delta_centered
                    logger.info(
                        f"  ΔKTA({key}): "
                        f"centered={delta_centered:+.4f} [PRIMARY], "
                        f"standard={delta_standard:+.4f} [biased by mean_offdiag]"
                    )

        # ---- Concentration ----
        if compute_quantum:
            logger.info(f"\n--- Concentration ({mod_name}) ---")
            conc = compare_concentration(K_fqk, K_pqk, K_rbf)
            mod_results["concentration"] = {
                k: {kk: vv for kk, vv in v.items() if kk != "eigenvalue_spectrum"}
                for k, v in conc.items()
            }
            # Log concentration metrics explicitly
            for k_name, metrics in conc.items():
                logger.info(
                    f"  {k_name}: CV={metrics['cv']:.4f}, "
                    f"mean_offdiag={metrics['mean_offdiag']:.6f}, "
                    f"std_offdiag={metrics['std_offdiag']:.6f}, "
                    f"eff_rank={metrics['effective_rank']:.1f}, "
                    f"top_eig_ratio={metrics['top_eigenvalue_ratio']:.4f}, "
                    f"n_sig_eigs={metrics['n_significant_eigs']}"
                )
            cv_fqk, cv_pqk = conc["FQK"]["cv"], conc["PQK"]["cv"]
            if cv_pqk > cv_fqk:
                logger.info("  ✓ PQK less concentrated than FQK (as expected)")
            else:
                logger.info("  ✗ PQK MORE concentrated than FQK (unexpected)")

            # Eigenvalue spectrum plot
            eig_dict = {}
            for k_name, K in [("FQK", K_fqk), ("PQK", K_pqk), ("RBF", K_rbf)]:
                eigvals = np.linalg.eigvalsh(K)
                eig_dict[k_name] = np.sort(eigvals)[::-1]

            plot_eigenvalue_spectrum(
                eig_dict,
                os.path.join(mod_dir, f"eigenvalue_spectrum_{mod_name}"),
                title=f"Eigenvalue Spectrum — {mod_name.upper()}",
            )

        # ---- Classification ----
        logger.info(f"\n--- Classification ({mod_name}) ---")
        mod_results["classification"] = {}

        classifiers_to_eval = [("RBF", K_rbf, K_rbf_test)]
        if compute_quantum:
            classifiers_to_eval = [
                ("FQK", K_fqk, K_fqk_test),
                ("PQK", K_pqk, K_pqk_test),
            ] + classifiers_to_eval
            if mod_name == "fused" and "TFK" in kernel_matrices[mod_name]:
                classifiers_to_eval.insert(0, ("TFK", kernel_matrices[mod_name]["TFK"], kernel_matrices[mod_name]["TFK_test"]))
                if "CM_FQK" in kernel_matrices[mod_name]:
                    classifiers_to_eval.insert(1, ("CM_FQK", kernel_matrices[mod_name]["CM_FQK"], kernel_matrices[mod_name]["CM_FQK_test"]))

        for k_name, K_tr, K_te in classifiers_to_eval:
            clf = train_precomputed_svm(K_tr, y_train)
            metrics = evaluate_classifier(clf, K_te, y_test, f"{k_name}-SVM ({mod_name})")
            metrics.pop("predictions", None)
            mod_results["classification"][k_name] = metrics

        all_results[mod_name] = mod_results

    # ---- Cross-modal MI analysis (fused only) ----
    if "fused" in modalities:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Cross-Modal MI Analysis")
        logger.info(f"{'='*60}")

        X_fused = modalities["fused"]["X_train"]
        X_sar_pca = X_fused[:, :4]
        X_opt_pca = X_fused[:, 4:]

        with Timer("Cross-modal MI", logger=logger):
            mi = compute_cross_modal_mutual_information(X_sar_pca, X_opt_pca, y_train)

        all_results["cross_modal_mi"] = mi

        # Create g vs MI data points by varying feature subsets
        g_values, mi_values = [], []
        if "fused" in kernel_matrices:
            K_fqk_fused = kernel_matrices["fused"]["FQK"]
            for n_feat in [2, 3, 4]:
                X_sub = X_fused[:, :n_feat * 2]
                # For each subset, compute MI and use corresponding g
                X_s = X_sub[:, :n_feat]
                X_o = X_sub[:, n_feat:]
                mi_sub = compute_cross_modal_mutual_information(X_s, X_o)
                K_rbf_sub = compute_rbf_kernel(X_sub)
                g_sub, _ = compute_geometric_difference(K_fqk_fused, K_rbf_sub)
                g_values.append(g_sub)
                mi_values.append(mi_sub)

        if len(g_values) >= 3:
            amp_results = analyze_geometric_amplification(
                np.array(g_values), np.array(mi_values),
                save_dir=results_dir,
            )
            all_results["geometric_amplification"] = amp_results

    # ---- Build summary table ----
    logger.info(f"\n{'='*60}")
    logger.info("  Summary Table")
    logger.info(f"{'='*60}")

    table_rows = []
    for mod_name, mod_results in all_results.items():
        if mod_name in ["cross_modal_mi", "geometric_amplification"]:
            continue
        for q_name in ["FQK", "PQK", "TFK", "CM_FQK"]:
            g_vs_rbf = mod_results["geometric_difference"].get(f"{q_name}_vs_rbf", float("nan"))
            g_vs_tp = mod_results["geometric_difference"].get(
                f"{q_name}_vs_tensor_product", float("nan")
            )
            kta = mod_results["kta"].get(q_name, {}).get("kta_weighted", float("nan"))
            delta_kta = mod_results["delta_kta"].get(f"{q_name}_vs_rbf", float("nan"))
            acc = mod_results["classification"].get(q_name, {}).get("accuracy", float("nan"))
            f1 = mod_results["classification"].get(q_name, {}).get("macro_f1", float("nan"))

            table_rows.append({
                "Modality": mod_name,
                "Kernel": q_name,
                "g(vs RBF)": g_vs_rbf,
                "g(vs Tensor)": g_vs_tp,
                "KTA": kta,
                "ΔKTA": delta_kta,
                "Macro-F1": f1,
                "Accuracy": acc,
            })

    df = pd.DataFrame(table_rows)
    df.to_csv(os.path.join(results_dir, "summary_table.csv"), index=False)
    logger.info(f"\n{df.to_string()}")

    # ---- Save all results ----
    save_results(all_results, results_file)
    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    run_experiment()
