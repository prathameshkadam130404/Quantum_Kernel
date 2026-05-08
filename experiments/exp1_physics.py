"""
exp1_physics.py — Geometric Analysis on Physics-Informed Features (PIQFM v5)

Runs the same analysis as exp1_geometric_analysis.py but on 8 physics
features (NDVI, NDRE, MNDWI, BSI, CrossPol, NDVI_tex, SAR_total, PolCoherence)
plus the novel PIQFM kernel (2 cross-modal bridges, analytically-set bandwidth,
SPSA+AMSGrad trained bridge params).

v5 changes vs v3:
  - NDBI replaced by NDRE (NDBI suppressed θ→floor in 3 consecutive training runs)
  - BSI↔PolCoh bridge removed (converged to 0.044 in v4, uninformative)
  - PIQFM bandwidth now analytic: θ_bw[i] = clip((π/2)/std(X[:,i]), 0.5, 2π)
  - Encoding bandwidth γ=0.75 selected by 5-fold CV on training set (Shaydulin & Wild 2022)
  - Features scaled to [0, 0.75π] — CV-optimal, prevents exponential concentration
  - Bridge optimizer changed from central finite diff + Adam to SPSA + AMSGrad
  - Anchor size increased from 51 to 102 for stable KTA gradient estimation

The key scientific question this answers:
  "Does FQK on physics features achieve higher KTA and g than FQK on PCA?"
  "Does PIQFM trained bandwidth+bridges beat vanilla FQK on physics features?"

Runtime: ~3-5 hours (same as exp1, same n=2000, same circuit)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import pandas as pd
import logging

import config
from src.quantum_kernels import (
    compute_fqk_kernel_matrix,
    compute_pqk_kernel_matrix,
    compute_tfk_kernel_matrix,
    train_quantum_kernel_kta,
    compute_crossmodal_fqk_kernel_matrix,
    get_top_mi_pairs,
)
from src.classical_kernels import (
    compute_rbf_kernel, compute_all_classical_kernels,
    compute_rbf_kernel_cv
)
from src.geometric_difference import compute_geometric_difference_batch, compute_geometric_difference_lambda_sweep
from src.kernel_target_alignment import compute_centered_kta, compute_delta_kta
from src.kernel_concentration import compute_concentration_metrics
from src.geometric_bound import compute_cross_modal_mutual_information
from src.classifiers import train_precomputed_svm, evaluate_classifier
from src.utils import setup_logging, Timer, ensure_dir, save_results
from src.bandwidth import apply_bandwidth, log_bandwidth_info

# ---- Output goes to results/physics/ — does NOT overwrite exp1 results ----
PHYSICS_RESULTS_DIR = os.path.join(config.RESULTS_DIR, "physics")
os.makedirs(PHYSICS_RESULTS_DIR, exist_ok=True)

logger = setup_logging(
    "exp1_physics",
    log_file=os.path.join(PHYSICS_RESULTS_DIR, "exp1_physics.log")
)

# ---- Physics feature names for logging (PIQFM v5) ----
OPTICAL_NAMES = ["NDVI", "NDRE", "MNDWI", "BSI"]             # cols 0-3
SAR_NAMES     = ["CrossPol", "NDVI_tex", "SAR_total", "PolCoh"] # cols 4-7
FEATURE_NAMES = OPTICAL_NAMES + SAR_NAMES

# Feature indices in physics_features_16.npz (PIQFM v5):
# 0:NDVI 1:NDBI 2:NDWI 3:BSI 4:SAVI 5:NDRE 6:MNDWI 7:EVI
# 8:VV/VH 9:SAR_total 10:CrossPol 11:PolCoh 12:VH_dB 13:VV_dB 14:VV_tex 15:NDVI_tex
# v5 change: NDBI (idx 1) replaced by NDRE (idx 5)
# NDBI was suppressed (θ→floor) in 3 consecutive training runs on So2Sat LCZ42.
FEATURE_INDICES = [0, 5, 6, 3, 10, 15, 9, 11]
# = NDVI, NDRE, MNDWI, BSI, CrossPol, NDVI_tex, SAR_total, PolCoh


def load_physics_features():
    """
    Load 8 physics features from physics_features_16.npz.

    Returns dict with same structure as load_subsample() for compatibility:
        fused_X_train, fused_X_test, y_train, y_test
        (no sar/optical split — physics features are already fused)
    """
    phys16_path = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
    phys8_path  = os.path.join(config.PROCESSED_DIR, "physics_features_piqfm.npz")

    if os.path.exists(phys8_path):
        logger.info(f"Loading pre-extracted 8 physics features: {phys8_path}")
        data = np.load(phys8_path)
        # This file has X_all and y_all (combined train+test)
        # Split 50/50 stratified to match n=2000 train / 2000 test convention
        from sklearn.model_selection import train_test_split
        X_all = data["X_all"]
        y_all = data["y_all"]
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.5,
            stratify=y_all, random_state=config.RANDOM_SEED
        )
        # Apply CV-selected bandwidth scaling — same as phys16 branch
        X_train = apply_bandwidth(X_train, "physics")
        X_test  = apply_bandwidth(X_test,  "physics")

    elif os.path.exists(phys16_path):
        logger.info(f"Extracting 8 features from 16-feature set: {phys16_path}")
        data = np.load(phys16_path)

        # Use normalised [0,π] features for quantum encoding
        X_train_16 = data["X_train"]   # shape (2000, 16), normalised to [0,π]
        X_test_16  = data["X_test"]    # shape (2000, 16)
        y_train    = data["y_train"]
        y_test     = data["y_test"]

        X_train = X_train_16[:, FEATURE_INDICES]  # (2000, 8), currently in [0,π]
        X_test  = X_test_16[:, FEATURE_INDICES]   # (2000, 8), currently in [0,π]
        # Apply CV-selected bandwidth scaling (γ=0.75 → range [0, 0.75π]).
        # γ selected by 5-fold CV on training set (Shaydulin & Wild 2022).
        # [0,π] caused exponential concentration: off-diag mean=0.021, eff_rank=460.
        # γ=0.75 is the CV-optimal value, not chosen by inspecting test metrics.
        X_train = apply_bandwidth(X_train, "physics")
        X_test  = apply_bandwidth(X_test,  "physics")

    else:
        raise FileNotFoundError(
            f"Physics features not found. Run extract_physics_features.py first.\n"
            f"Expected: {phys16_path}"
        )

    logger.info(f"Physics features loaded:")
    logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    log_bandwidth_info(X_train, "physics", logger=logger)
    logger.info(f"  Features: {FEATURE_NAMES}")
    logger.info(f"  Classes: {len(np.unique(y_train))}, "
                f"Train samples: {len(y_train)}, Test samples: {len(y_test)}")

    return {
        "fused_X_train": X_train,
        "fused_X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
    }


def run_experiment():
    results_file = os.path.join(PHYSICS_RESULTS_DIR, "exp1_physics_results.json")

    # Do NOT use check_and_skip — always allow rerun on physics features
    # (comment out if you want caching)
    # if os.path.exists(results_file):
    #     logger.info(f"Results exist: {results_file}. Delete to rerun.")
    #     return

    np.random.seed(config.RANDOM_SEED)
    all_results = {}

    # ---- Load physics features ----
    with Timer("Loading physics features", logger=logger):
        subsample = load_physics_features()

    X_train = subsample["fused_X_train"]
    X_test  = subsample["fused_X_test"]
    y_train = subsample["y_train"]
    y_test  = subsample["y_test"]

    mod_dir = ensure_dir(os.path.join(PHYSICS_RESULTS_DIR, "fused"))

    logger.info(f"\n{'='*60}")
    logger.info(f"  Physics Features: {X_train.shape}")
    logger.info(f"  Optical (cols 0-3): {OPTICAL_NAMES}")
    logger.info(f"  SAR    (cols 4-7): {SAR_NAMES}")
    logger.info(f"{'='*60}")

    mod_results = {
        "feature_set": "physics_8_v5_bw050",  # γ=0.50, concentration criterion
        "bandwidth_gamma": 0.5,
        "bandwidth_source": "concentration criterion, Shaydulin & Wild 2022",
        "encoding_range": "0_to_0.5pi",
        "feature_names": FEATURE_NAMES,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "geometric_difference": {},
        "kta": {},
        "delta_kta": {},
        "concentration": {},
        "classification": {},
    }

    # ================================================================
    # QUANTUM KERNELS
    # ================================================================

    # ---- FQK ----
    with Timer("FQK kernel (physics)", logger=logger):
        K_fqk = compute_fqk_kernel_matrix(
            X_train,
            save_path=os.path.join(mod_dir, "K_fqk_physics_train.npy"),
        )
        K_fqk_test = compute_fqk_kernel_matrix(
            X_train, X2=X_test,
            save_path=os.path.join(mod_dir, "K_fqk_physics_test.npy"),
        )

    # ---- PQK ----
    with Timer("PQK kernel (physics)", logger=logger):
        K_pqk = compute_pqk_kernel_matrix(
            X_train,
            save_path=os.path.join(mod_dir, "K_pqk_physics_train.npy"),
            bloch_save_path=os.path.join(mod_dir, "bloch_physics_train.npy"),
        )
        K_pqk_test = compute_pqk_kernel_matrix(
            X_train, X2=X_test,
            save_path=os.path.join(mod_dir, "K_pqk_physics_test.npy"),
            bloch_save_path=os.path.join(mod_dir, "bloch_physics_train.npy"),
        )

    # ---- TFK ----
    tfk_dir = ensure_dir(os.path.join(mod_dir, "tfk_physics"))
    with Timer("TFK training (physics)", logger=logger):
        theta_trained = train_quantum_kernel_kta(
            X_train, y_train,
            save_path=os.path.join(tfk_dir, "theta_physics_trained.npy"),
        )
    K_tfk = compute_tfk_kernel_matrix(
        X_train, theta_trained,
        save_path=os.path.join(tfk_dir, "K_tfk_physics_train.npy"),
    )
    K_tfk_test = compute_tfk_kernel_matrix(
        X_train, theta_trained, X2=X_test,
        save_path=os.path.join(tfk_dir, "K_tfk_physics_test.npy"),
    )

    # ---- CM_FQK (physics) ----
    # CRITICAL FIX vs exp1: physics features have optical in cols 0-3, SAR in 4-7
    # get_top_mi_pairs default assumes SAR=cols 0-3, optical=cols 4-7
    # We pass n_sar=4, n_opt=4 — verify your get_top_mi_pairs signature
    # If it assumes first n_sar cols are SAR, we need to reorder X_train
    # Safe approach: compute MI directly on known column assignments
    logger.info("\nComputing MI pairs on physics features (optical=0:4, SAR=4:8)...")
    X_optical = X_train[:, :4]   # NDVI, NDBI, MNDWI, BSI
    X_sar     = X_train[:, 4:]   # CrossPol, NDVI_tex, SAR_total, PolCoh

    # get_top_mi_pairs expects full X_train with SAR first, optical second
    # Reorder to match expectation: SAR=0:4, optical=4:8
    X_train_reordered = np.concatenate([X_sar, X_optical], axis=1)
    mi_pairs = get_top_mi_pairs(X_train_reordered, n_sar=4, n_opt=4, top_k=3)

    # get_top_mi_pairs returns (sar_qubit, opt_qubit) in REORDERED space:
    #   sar_qubit  = 0..3  (SAR cols 0-3 in reordered = cols 4-7 in original)
    #   opt_qubit  = 4..7  (optical cols 4-7 in reordered = cols 0-3 in original)
    # Convert back to ORIGINAL column indices:
    #   original_sar_col = sar_qubit + 4
    #   original_opt_col = opt_qubit - 4
    mi_pairs_original = []
    for s_idx, o_idx in mi_pairs:
        orig_sar = s_idx + 4
        orig_opt = o_idx - 4
        
        # AUDIT: Ensure s is in SAR range and o is in Optical range
        assert 4 <= orig_sar <= 7, f"Audit failed: s={orig_sar} is not a SAR index [4-7]"
        assert 0 <= orig_opt <= 3, f"Audit failed: o={orig_opt} is not an Optical index [0-3]"
        
        mi_pairs_original.append((orig_sar, orig_opt))
        
    logger.info(f"  MI pairs (original indices): {mi_pairs_original}")
    for s, o in mi_pairs_original:
        logger.info(f"    PAIR: [SAR] {SAR_NAMES[s-4]} (idx {s}) <--> [Optical] {OPTICAL_NAMES[o]} (idx {o})")

    K_cm_fqk = compute_crossmodal_fqk_kernel_matrix(
        X_train, mi_pairs_original,
        save_path=os.path.join(mod_dir, "K_cm_fqk_physics_train.npy"),
    )
    K_cm_fqk_test = compute_crossmodal_fqk_kernel_matrix(
        X_train, mi_pairs_original, X2=X_test,
        save_path=os.path.join(mod_dir, "K_cm_fqk_physics_test.npy"),
    )

    # ---- PIQFM (novel kernel with trained θ) ----
    from scripts.compute_piqfm_kernel import (
        compute_piqfm_kernel_matrix, train_piqfm_kta,
        compute_analytic_bandwidth,
        CROSS_MODAL_BRIDGES as PIQFM_BRIDGES,
    )
    piqfm_dir = ensure_dir(os.path.join(mod_dir, "piqfm"))
    theta_path = os.path.join(config.RESULTS_DIR, "piqfm", "trained_theta.npz")

    if os.path.exists(theta_path):
        logger.info(f"\nLoading trained PIQFM θ from {theta_path}")
        theta_data = np.load(theta_path)
        theta_bw = theta_data["theta_bandwidth"]
        theta_br = theta_data["theta_bridge"]
    else:
        logger.info("\nTraining PIQFM θ via SPSA+AMSGrad (102 samples, 100 epochs)...")
        from sklearn.model_selection import train_test_split
        from scripts.compute_piqfm_kernel import compute_analytic_bandwidth
        _, X_kta, _, y_kta = train_test_split(
            X_train, y_train, test_size=102,
            stratify=y_train, random_state=config.RANDOM_SEED,
        )
        # v5: compute bandwidth analytically, train only bridge params
        theta_bw = compute_analytic_bandwidth(X_kta)
        logger.info(f"  Analytic θ_bw = [{', '.join(f'{v:.3f}' for v in theta_bw)}]")
        
        checkpoint_path = os.path.join(mod_dir, "piqfm_theta_checkpoint.npz")
        theta_bw, theta_br, _ = train_piqfm_kta(
            X_kta, y_kta,
            theta_bandwidth=theta_bw,
            n_epochs=100,
            lr=0.03,
            spsa_delta=0.05,
            checkpoint_path=checkpoint_path,
        )
        np.savez(theta_path, theta_bandwidth=theta_bw, theta_bridge=theta_br)
        logger.info(f"  Saved trained θ: {theta_path}")

    logger.info(f"  θ_bw = [{', '.join(f'{v:.3f}' for v in theta_bw)}]")
    logger.info(f"  θ_br = [{', '.join(f'{v:+.4f}' for v in theta_br)}]")

    piqfm_train_path = os.path.join(piqfm_dir, "K_piqfm_physics_train.npy")
    piqfm_test_path  = os.path.join(piqfm_dir, "K_piqfm_physics_test.npy")

    if os.path.exists(piqfm_train_path) and os.path.exists(piqfm_test_path):
        logger.info("[SKIP] PIQFM kernels exist, loading...")
        K_piqfm      = np.load(piqfm_train_path)
        K_piqfm_test = np.load(piqfm_test_path)
    else:
        with Timer("PIQFM kernel (physics)", logger=logger):
            K_piqfm = compute_piqfm_kernel_matrix(
                X_train, theta_bw, theta_br,
                desc="PIQFM train",
                silent=False,
            )
            np.save(piqfm_train_path, K_piqfm)

            K_piqfm_test = compute_piqfm_kernel_matrix(
                X_train, theta_bw, theta_br,
                X2=X_test,
                desc="PIQFM test",
                silent=False,
            )
            np.save(piqfm_test_path, K_piqfm_test)

    # ================================================================
    # CLASSICAL KERNELS
    # ================================================================
    with Timer("Classical kernels (physics)", logger=logger):
        # Fixed gamma=0.125 baseline
        K_rbf = compute_rbf_kernel(
            X_train,
            save_path=os.path.join(mod_dir, "K_rbf_physics_train.npy")
        )
        K_rbf_test = compute_rbf_kernel(X_train, X2=X_test)

        # CV-tuned gamma baseline
        K_rbf_cv, K_rbf_cv_test, best_gamma, _ = compute_rbf_kernel_cv(
            X_train, y_train, X_test=X_test,
            save_path_train=os.path.join(mod_dir, "K_rbf_cv_physics_train.npy"),
            save_path_test=os.path.join(mod_dir, "K_rbf_cv_physics_test.npy"),
        )
        logger.info(f"RBF_CV best gamma: {best_gamma}")

        classical_kernels = compute_all_classical_kernels(
            X_train, save_dir=mod_dir, prefix="physics_"
        )
        # Add RBF_CV to classical kernels for geometric difference
        classical_kernels["rbf_cv"] = K_rbf_cv

    # ================================================================
    # GEOMETRIC DIFFERENCE
    # ================================================================
    logger.info("\n--- Geometric Difference (physics features) ---")
    quantum_kernels = {
        "FQK": K_fqk, "PQK": K_pqk,
        "TFK": K_tfk, "CM_FQK": K_cm_fqk, "PIQFM": K_piqfm
    }
    g_results = compute_geometric_difference_batch(quantum_kernels, classical_kernels)

    for pair, g_val in g_results.items():
        mod_results["geometric_difference"][pair] = g_val
        g_float = g_val["g"] if isinstance(g_val, dict) else g_val
        logger.info(f"  g({pair}) = {g_float:.2f}")

    # Lambda sweep for stability disclosure
    logger.info("\n  Lambda sweep (FQK vs RBF, physics)...")
    sweep = compute_geometric_difference_lambda_sweep(
        K_fqk, K_rbf,
        save_path=os.path.join(mod_dir, "lambda_sweep_physics_fqk_rbf.pdf")
    )
    mod_results["lambda_sweep_fqk"] = {
        str(k): float(v) for k, v in sweep["sweep_results"].items()
    }
    logger.info(f"  g at lambda=1/n: {sweep['recommended_g']:.2f}")
    logger.info(f"  Sweep stable: {sweep.get('stable', 'unknown')}")

    # ================================================================
    # KTA
    # ================================================================
    logger.info("\n--- KTA (physics features) ---")
    kta_rbf = compute_centered_kta(K_rbf, y_train, class_weighted=True)
    kta_rbf_cv = compute_centered_kta(K_rbf_cv, y_train, class_weighted=True)
    
    mod_results["kta"]["RBF"] = float(kta_rbf)
    mod_results["kta"]["RBF_CV"] = float(kta_rbf_cv)
    
    logger.info(f"  RBF     KTA = {kta_rbf:.4f}")
    logger.info(f"  RBF_CV  KTA = {kta_rbf_cv:.4f}")

    for qname, K_q in quantum_kernels.items():
        kta_q = compute_centered_kta(K_q, y_train, class_weighted=True)
        delta_vs_rbf = kta_q - kta_rbf
        delta_vs_cv  = kta_q - kta_rbf_cv
        
        mod_results["kta"][qname] = float(kta_q)
        mod_results["delta_kta"][f"{qname}_vs_RBF"] = float(delta_vs_rbf)
        mod_results["delta_kta"][f"{qname}_vs_RBF_CV"] = float(delta_vs_cv)
        
        logger.info(f"  {qname:<7} KTA = {kta_q:.4f}  ΔKTA(vs RBF)={delta_vs_rbf:+.4f}  ΔKTA(vs CV)={delta_vs_cv:+.4f}")

    # ================================================================
    # CONCENTRATION
    # ================================================================
    logger.info("\n--- Concentration (physics features) ---")
    for kname, K in [("FQK", K_fqk), ("PQK", K_pqk),
                     ("TFK", K_tfk), ("CM_FQK", K_cm_fqk),
                     ("PIQFM", K_piqfm), ("RBF", K_rbf),
                     ("RBF_CV", K_rbf_cv)]:
        m = compute_concentration_metrics(K)
        mod_results["concentration"][kname] = {
            k: float(v) for k, v in m.items()
            if k != "eigenvalue_spectrum"
        }
        logger.info(
            f"  {kname}: CV={m['cv']:.4f}, "
            f"mean_offdiag={m['mean_offdiag']:.4f}, "
            f"eff_rank={m['effective_rank']:.2f}"
        )

    # ================================================================
    # CLASSIFICATION
    # ================================================================
    logger.info("\n--- Classification (physics features) ---")
    classifiers = [
        ("FQK",    K_fqk,    K_fqk_test),
        ("PQK",    K_pqk,    K_pqk_test),
        ("TFK",    K_tfk,    K_tfk_test),
        ("CM_FQK", K_cm_fqk, K_cm_fqk_test),
        ("PIQFM",  K_piqfm,  K_piqfm_test),
        ("RBF",    K_rbf,    K_rbf_test),
        ("RBF_CV", K_rbf_cv, K_rbf_cv_test),
    ]
    for kname, K_tr, K_te in classifiers:
        clf = train_precomputed_svm(K_tr, y_train)
        metrics = evaluate_classifier(clf, K_te, y_test, f"{kname}-SVM (physics)")
        metrics.pop("predictions", None)
        mod_results["classification"][kname] = metrics
        logger.info(
            f"  {kname}: macro_F1={metrics.get('macro_f1', float('nan')):.4f}, "
            f"acc={metrics.get('accuracy', float('nan')):.4f}"
        )

    # ================================================================
    # CROSS-MODAL MI (physics — optical cols 0-3, SAR cols 4-7)
    # ================================================================
    logger.info("\n--- Cross-Modal MI (physics features) ---")
    mi = compute_cross_modal_mutual_information(X_optical, X_sar, y_train)
    all_results["cross_modal_mi_physics"] = mi

    # ================================================================
    # COMPARISON VS PCA RESULTS
    # ================================================================
    logger.info(f"\n{'='*60}")
    logger.info("  PHYSICS vs PCA COMPARISON")
    logger.info(f"{'='*60}")

    # Load PCA results if available
    pca_results_path = os.path.join(
        config.RESULTS_DIR, "geometric_difference", "exp1_results.json"
    )
    if os.path.exists(pca_results_path):
        with open(pca_results_path) as f:
            pca_results = json.load(f)

        pca_fused = pca_results.get("fused", {})
        pca_kta   = pca_fused.get("kta", {})
        pca_g     = pca_fused.get("geometric_difference", {})
        pca_f1    = pca_fused.get("classification", {})

        logger.info(f"\n  {'Kernel':<10} {'PCA KTA':>10} {'Physics KTA':>12} "
                    f"{'Δ KTA':>8} {'PCA F1':>8} {'Physics F1':>11}")
        logger.info("  " + "-" * 65)

        for qname in ["FQK", "PQK", "TFK", "CM_FQK", "PIQFM", "RBF"]:
            pca_k = float(pca_kta.get(qname, {}).get("kta_centered_weighted",
                          pca_kta.get(qname, {}).get("kta_weighted", float("nan")))
                          if isinstance(pca_kta.get(qname, {}), dict)
                          else pca_kta.get(qname, float("nan")))
            phys_k = mod_results["kta"].get(qname, float("nan"))
            delta_k = phys_k - pca_k if not (
                np.isnan(pca_k) or np.isnan(phys_k)) else float("nan")
            pca_f = float(pca_f1.get(qname, {}).get("macro_f1", float("nan"))
                          if isinstance(pca_f1.get(qname, {}), dict)
                          else float("nan"))
            phys_f = mod_results["classification"].get(
                qname, {}).get("macro_f1", float("nan"))

            logger.info(
                f"  {qname:<10} {pca_k:>10.4f} {phys_k:>12.4f} "
                f"{delta_k:>+8.4f} {pca_f:>8.4f} {phys_f:>11.4f}"
            )
    else:
        logger.info("  PCA results not found for comparison.")

    # ================================================================
    # SAVE
    # ================================================================
    all_results["physics_fused"] = mod_results
    save_results(all_results, results_file)
    logger.info(f"\nAll results saved: {results_file}")

    # Summary CSV
    rows = []
    for qname in ["FQK", "PQK", "TFK", "CM_FQK", "PIQFM", "RBF", "RBF_CV"]:
        rows.append({
            "Kernel": qname,
            "Feature_Set": "physics_8",
            "KTA": mod_results["kta"].get(qname, float("nan")),
            "ΔKTA_vs_RBF": mod_results["delta_kta"].get(
                f"{qname}_vs_RBF", float("nan")),
            "g_vs_RBF": mod_results["geometric_difference"].get(
                f"{qname}_vs_rbf", float("nan")),
            "Macro_F1": mod_results["classification"].get(
                qname, {}).get("macro_f1", float("nan")),
            "g_vs_RBF_CV": mod_results["geometric_difference"].get(
                f"{qname}_vs_rbf_cv", {}).get("g", float("nan"))
                if isinstance(mod_results["geometric_difference"].get(f"{qname}_vs_rbf_cv"), dict)
                else float("nan"),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(PHYSICS_RESULTS_DIR, "physics_summary.csv"), index=False
    )
    logger.info(f"Summary CSV: {PHYSICS_RESULTS_DIR}/physics_summary.csv")


if __name__ == "__main__":
    run_experiment()
