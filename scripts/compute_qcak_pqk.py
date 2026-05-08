"""
compute_qcak_pqk.py — QCAK-PQK kernel computation and attention training script.

Stages:
  1. Load physics features (16d) and AGPQK config.
  2. Compute variance-normalized per-qubit bandwidth.
  3. Determine qubit layout (OPT/SAR) and bipartite attention pairs.
  4. Train QCAK attention parameters (s_params) via SPSA.
  5. Search for optimal outer RBF gamma.
  6. Compute full training and test kernel matrices.
  7. Eval concentration and KTA vs baselines.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import json
import numpy as np
import time
import logging
from typing import List, Tuple
from tqdm import tqdm

import config
from src.qcak_pqk import (
    compute_qcak_pqk_bloch_vectors,
    compute_qcak_pqk_kernel_from_bloch
)
from src.kernel_target_alignment import compute_centered_kta, compute_full_kta_analysis
from src.kernel_concentration import compute_concentration_metrics
from src.utils import setup_logging, ensure_dir, Timer

# ── Global configuration ───────────────────────────────────────────────────
N_QUBITS    = config.N_QUBITS    # 8
ZZ_REPS     = config.ZZ_REPS     # 2
RANDOM_SEED = config.RANDOM_SEED # 42
BASE_GAMMA  = 0.50               # Initial gamma_base for per-qubit scaling

QCAK_EXP_DIR = ensure_dir(os.path.join(config.RESULTS_DIR, "physics", "qcak_pqk"))
PHYS16_PATH  = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
AGPQK_CFG    = os.path.join(config.RESULTS_DIR, "physics", "agpqk_config.json")

# SPSA Hyperparameters
N_EPOCHS    = 60
N_PER_CLASS = 5      # ~85 samples per mini-batch (5 * 17)
SPSA_LR     = 0.1
SPSA_EPS    = 0.05
S_CLIP_MIN  = 0.3
S_CLIP_MAX  = 1.2

# Baselines from exp1_physics
BASELINES = {
    "RBF":   0.2956,
    "FQK":   0.2345,
    "PQK":   0.2329,
    "TFK":   0.1947,
    "AGPQK": 0.2518
}

logger = setup_logging(os.path.join(QCAK_EXP_DIR, "qcak_pqk.log"))

def get_fresh_batch(y, n_per_class, seed=None):
    """Draw a fresh stratified random subset."""
    if seed is not None:
        np.random.seed(seed)
    
    indices = []
    classes = np.unique(y)
    for c in classes:
        c_idx = np.where(y == c)[0]
        n_sel = min(n_per_class, len(c_idx))
        indices.extend(np.random.choice(c_idx, size=n_sel, replace=False))
    
    indices = np.array(indices)
    np.random.shuffle(indices)
    return indices

def run_qcak_experiment():
    print("=" * 65)
    print("  QCAK-PQK — Quantum Cross-Attention Kernel (Projected)")
    print("=" * 65)

    # ── STAGE 1 — Load data ───────────────────────────────────────────────
    logger.info("\n--- Stage 1: Loading Data ---")
    if not os.path.exists(PHYS16_PATH):
        logger.error(f"Data not found: {PHYS16_PATH}")
        sys.exit(1)
    
    data = np.load(PHYS16_PATH)
    X_train_norm = data["X_train"]          # Normalized [0, pi]
    X_test_norm  = data["X_test"]
    y_train      = data["y_train"]
    y_test       = data["y_test"]

    # Try to load raw features for variance computation
    if "X_train_raw" in data:
        X_train_var_src = data["X_train_raw"]
    else:
        logger.warning("X_train_raw not found in npz. Using normalized X_train for variance.")
        X_train_var_src = X_train_norm

    if not os.path.exists(AGPQK_CFG):
        logger.error(f"AGPQK configuration not found: {AGPQK_CFG}")
        sys.exit(1)
    
    with open(AGPQK_CFG, 'r') as f:
        agpqk_config = json.load(f)
    
    selected_indices = agpqk_config["selected_feature_indices"]
    feature_names    = agpqk_config["selected_feature_names"]
    modalities       = agpqk_config["selected_feature_modalities"]

    logger.info(f"  Train: {len(y_train)}, Test: {len(y_test)}")
    logger.info(f"  Selected Features: {feature_names}")
    logger.info(f"  Modalities: {modalities}")


    # ── STAGE 2 — Per-qubit bandwidth (variance-normalized) ──────────────
    # Normalize each feature to [0,1] range for comparable variance
    X8_raw = X_train_var_src[:, selected_indices]   # physical units
    feat_mins = X8_raw.min(axis=0)
    feat_maxs = X8_raw.max(axis=0)
    feat_ranges = feat_maxs - feat_mins + 1e-8
    X8_norm_01 = (X8_raw - feat_mins) / feat_ranges  # each feature in [0,1]

    # Variance on unit-normalized features: all features now comparable
    stds_01 = X8_norm_01.std(axis=0) + 1e-8
    mean_std_01 = stds_01.mean()

    # Variance-normalized gamma on unit-normalized features
    gamma_base = 0.5
    gamma_raw = gamma_base * (mean_std_01 / stds_01)
    # Tighter clip: min=0.35 prevents weak encoding, max=0.9 prevents
    # high-frequency phase scrambling that causes concentration
    gamma_per_qubit = np.clip(gamma_raw, 0.35, 0.90)

    # Apply gamma to the [0,pi]-normalized features for encoding
    X8_for_enc = X_train_norm[:, selected_indices]          # already in [0,pi]
    X8_test_for_enc = X_test_norm[:, selected_indices]
    X8_enc = X8_for_enc * gamma_per_qubit[np.newaxis, :]
    X8_test_enc = X8_test_for_enc * gamma_per_qubit[np.newaxis, :]

    # Print diagnostic
    logger.info(f"  Per-qubit gamma (unit-normalized variance):")
    for q in range(len(selected_indices)):
        logger.info(f"    qubit {q} ({feature_names[q]:<12}): "
                    f"std_01={stds_01[q]:.4f}  gamma={gamma_per_qubit[q]:.4f}  "
                    f"enc_range=[{X8_enc[:,q].min():.3f}, {X8_enc[:,q].max():.3f}]")

    # Verify no feature is clipping at boundary
    n_at_max = (gamma_per_qubit >= 0.89).sum()
    n_at_min = (gamma_per_qubit <= 0.36).sum()
    if n_at_max > 2 or n_at_min > 2:
        logger.warning(f"  WARNING: {n_at_max} features at gamma ceiling, "
                       f"{n_at_min} at floor. Check unit normalization.")


    # ── STAGE 3 — Determine qubit layout and attention pairs ──────────────
    logger.info("\n--- Stage 3: Qubit Layout and Attention Bipartite Pairs ---")
    opt_qubits = [q for q, m in enumerate(modalities) if m == "OPT"]
    sar_qubits = [q for q, m in enumerate(modalities) if m == "SAR"]

    attention_pairs = []
    used_qubits = set()

    # Try taking pairs from agpqk_config first
    ag_pairs = agpqk_config.get("entanglement_pairs_qubit_space", [])
    for q1, q2 in ag_pairs:
        # Check if cross-modal
        m1, m2 = modalities[q1], modalities[q2]
        if m1 != m2 and q1 not in used_qubits and q2 not in used_qubits:
            # Order as (SAR, OPT)
            if m1 == "SAR":
                attention_pairs.append((q1, q2))
            else:
                attention_pairs.append((q2, q1))
            used_qubits.add(q1)
            used_qubits.add(q2)

    # Fill remaining bipartite slots
    remaining_sar = [q for q in sar_qubits if q not in used_qubits]
    remaining_opt = [q for q in opt_qubits if q not in used_qubits]

    for s_q, o_q in zip(remaining_sar, remaining_opt):
        attention_pairs.append((s_q, o_q))
        used_qubits.add(s_q)
        used_qubits.add(o_q)

    logger.info(f"  OPT Qubits: {opt_qubits}")
    logger.info(f"  SAR Qubits: {sar_qubits}")
    logger.info(f"  Attention Pairs (SAR ↔ OPT):")
    for s_q, o_q in attention_pairs:
        logger.info(f"    ({feature_names[s_q]} [SAR] ↔ {feature_names[o_q]} [OPT])")
    
    n_pairs = len(attention_pairs)

    # ── STAGE 4 — Train QCAK-PQK attention parameters (SPSA) ──────────────
    logger.info("\n--- Stage 4: Training Attention Parameters (SPSA) ---")
    s_params_path = os.path.join(QCAK_EXP_DIR, "s_params.npy")

    if os.path.exists(s_params_path):
        logger.info(f"  [SKIP] Loading cached s_params: {s_params_path}")
        s_params = np.load(s_params_path)
    else:
        s_params = np.ones((ZZ_REPS, n_pairs), dtype=np.float64)
        kta_history = []

        def mini_kta(s_p, X_e, y):
            batch_idx = get_fresh_batch(y, N_PER_CLASS)
            X_b = X_e[batch_idx]
            y_b = y[batch_idx]
            
            bloch_b = compute_qcak_pqk_bloch_vectors(
                X_b, s_p, opt_qubits, sar_qubits, attention_pairs,
                n_qubits=N_QUBITS, reps=ZZ_REPS, save_path=None
            )
            # Standard outer gamma 0.67 for training
            K_b = compute_qcak_pqk_kernel_from_bloch(bloch_b, bloch_b, outer_gamma=0.67, n_qubits=N_QUBITS)
            return compute_centered_kta(K_b, y_b, class_weighted=True)

        for epoch in range(N_EPOCHS):
            # SPSA step
            delta = np.random.choice([-1.0, 1.0], size=s_params.shape)
            
            kta_plus  = mini_kta(s_params + SPSA_EPS * delta, X8_enc, y_train)
            kta_minus = mini_kta(s_params - SPSA_EPS * delta, X8_enc, y_train)
            
            grad = (kta_plus - kta_minus) / (2.0 * SPSA_EPS) * delta
            s_params = s_params + SPSA_LR * grad
            s_params = np.clip(s_params, S_CLIP_MIN, S_CLIP_MAX)

            # Record KTA history at every epoch
            kta_current = mini_kta(s_params, X8_enc, y_train)
            kta_history.append(float(kta_current))

            if (epoch + 1) % 5 == 0 or epoch == N_EPOCHS - 1:
                logger.info(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: KTA={kta_current:.4f}  "
                            f"s_mean={s_params.mean():.4f}  s_std={s_params.std():.4f}  "
                            f"grad_norm={np.linalg.norm(grad):.4f}")
                logger.info(f"    s_params: {np.round(s_params, 4).tolist()}")

            # Quick concentration check on mini-batch kernel
            if (epoch + 1) % 10 == 0:
                # Reuse X_b and y_b if possible, but they are localized to mini_kta. 
                # Let's just use a fresh small check here for monitoring.
                check_idx = get_fresh_batch(y_train, n_per_class=3) # 51 samples
                bloch_check = compute_qcak_pqk_bloch_vectors(
                    X8_enc[check_idx], s_params, opt_qubits, sar_qubits,
                    attention_pairs, save_path=None
                )
                K_check = compute_qcak_pqk_kernel_from_bloch(
                    bloch_check, bloch_check, outer_gamma=0.67, n_qubits=N_QUBITS
                )
                eigvals_c = np.linalg.eigvalsh(K_check)
                eigvals_c = np.maximum(eigvals_c, 0)
                p_c = eigvals_c / (eigvals_c.sum() + 1e-12)
                eff_rank_c = float(np.exp(-np.sum(p_c * np.log(p_c + 1e-12))))
                mean_od_c = float(K_check[np.triu_indices(len(K_check), k=1)].mean())
                
                logger.info(f"    [Concentration Monitoring] eff_rank={eff_rank_c:.1f}  mean_offdiag={mean_od_c:.4f}")
                
                if eff_rank_c < 5.0:
                    logger.warning(f"    WARNING: eff_rank={eff_rank_c:.1f} < 5.0. Severe concentration detected.")

        np.save(s_params_path, s_params)
        logger.info(f"  Trained s_params saved to {s_params_path}")
        
        # Save training log
        log_data = {
            "kta_history": kta_history,
            "s_params_final": s_params.tolist(),
            "n_epochs": N_EPOCHS,
            "n_per_class": N_PER_CLASS,
            "spsa_lr": SPSA_LR,
            "spsa_eps": SPSA_EPS,
            "feature_names": feature_names,
            "attention_pairs": attention_pairs
        }
        with open(os.path.join(QCAK_EXP_DIR, "training_log.json"), 'w') as f:
            json.dump(log_data, f, indent=2)

    # ── STAGE 5 — Outer gamma search (anchored, rank-constrained) ──────────
    logger.info("\n--- Stage 5: Outer Gamma Search (anchored, rank-constrained) ---")
    bloch_train_path = os.path.join(QCAK_EXP_DIR, "bloch_train.npy")
    bloch_train = compute_qcak_pqk_bloch_vectors(
        X8_enc, s_params, opt_qubits, sar_qubits, attention_pairs,
        n_qubits=N_QUBITS, reps=ZZ_REPS, save_path=bloch_train_path
    )

    # Step 1: Median heuristic distance on 200-sample subset
    rng_gs = np.random.default_rng(config.RANDOM_SEED)
    idx200 = rng_gs.choice(len(bloch_train), 200, replace=False)
    b200 = bloch_train[idx200].reshape(200, N_QUBITS, 3)

    distances = []
    for i in range(200):
        for j in range(i + 1, 200):
            diff = b200[i] - b200[j]         # (N_QUBITS, 3)
            dist = 0.5 * np.sum(diff ** 2)   # scalar Frobenius
            distances.append(dist)

    median_dist = float(np.median(distances))
    gamma_median = 1.0 / (2.0 * median_dist + 1e-8)

    logger.info(f"  Median pairwise Frobenius distance: {median_dist:.4f}")
    logger.info(f"  Median heuristic gamma:             {gamma_median:.4f}")
    logger.info(f"  Current outer_gamma reference:      0.5")

    # Anchor to standard PQK gamma=0.67 as reference point
    all_candidates = [0.1, 0.2, 0.3, 0.5, 0.67, 0.8, 1.0, 1.2, 1.5, 2.0]
    
    # Larger stratified subset for gamma search
    N_GAMMA_SEARCH = 1500
    N_FOLDS = 3
    unique_cls = np.unique(y_train)
    n_per_cls_search = max(3, N_GAMMA_SEARCH // len(unique_cls))
    search_indices = []
    for cls in unique_cls:
        cls_idx = np.where(y_train == cls)[0]
        n_take = min(n_per_cls_search, len(cls_idx))
        search_indices.extend(rng_gs.choice(cls_idx, n_take, replace=False).tolist())
    search_indices = np.array(search_indices[:N_GAMMA_SEARCH])
    bloch_search = bloch_train[search_indices]
    y_search     = y_train[search_indices]
    n_search     = len(search_indices)
    fold_size    = n_search // N_FOLDS

    logger.info(f"  Gamma Candidates: {all_candidates}")
    logger.info(f"  Search subset: {n_search} samples, {N_FOLDS} folds")

    MIN_EFF_RANK = 15.0
    gamma_results = {}

    for g_cand in all_candidates:
        # Quick rank check on full search subset before CV
        K_full_search = compute_qcak_pqk_kernel_from_bloch(bloch_search, bloch_search, g_cand, n_qubits=N_QUBITS)
        eigvals = np.linalg.eigvalsh(K_full_search)
        eigvals_pos = np.maximum(eigvals, 0)
        p = eigvals_pos / (eigvals_pos.sum() + 1e-12)
        eff_rank_check = float(np.exp(-np.sum(p * np.log(p + 1e-12))))

        if eff_rank_check < MIN_EFF_RANK:
            logger.info(f"  gamma={g_cand:.5f}  eff_rank={eff_rank_check:.1f}  SKIPPED (rank < {MIN_EFF_RANK})")
            continue

        # Cross-validated KTA
        fold_ktas = []
        for fold in range(N_FOLDS):
            val_start, val_end = fold * fold_size, (fold + 1) * fold_size
            val_idx = np.arange(val_start, val_end)
            if len(np.unique(y_search[val_idx])) < 8:
                continue
            K_val = compute_qcak_pqk_kernel_from_bloch(bloch_search[val_idx], bloch_search[val_idx], g_cand, n_qubits=N_QUBITS)
            kta_fold = compute_centered_kta(K_val, y_search[val_idx], class_weighted=True)
            fold_ktas.append(float(kta_fold))

        if fold_ktas:
            mean_kta = float(np.mean(fold_ktas))
            gamma_results[g_cand] = {"mean_kta": mean_kta, "eff_rank": eff_rank_check}
            logger.info(f"  gamma={g_cand:.5f}  eff_rank={eff_rank_check:.1f}  cv_KTA={mean_kta:.4f}")

    if not gamma_results:
        logger.warning(
            "  WARNING: all candidates failed rank check. "
            "Falling back to gamma=0.67 (AGPQK reference, known stable)."
        )
        outer_gamma = 0.67
    else:
        outer_gamma = max(gamma_results, key=lambda g: gamma_results[g]["mean_kta"])
        logger.info(f"\n  Selected outer_gamma = {outer_gamma}")
        logger.info(f"  eff_rank at selection = {gamma_results[outer_gamma]['eff_rank']:.1f}")
        logger.info(f"  cv_KTA at selection   = {gamma_results[outer_gamma]['mean_kta']:.4f}")

        # Change 2 — validate selected gamma on full training Bloch vectors
        logger.info(f"\n  Validating selected gamma={outer_gamma} on full n={len(bloch_train)} Bloch vectors...")
        K_val_full = compute_qcak_pqk_kernel_from_bloch(
            bloch_train, bloch_train, outer_gamma, n_qubits=N_QUBITS
        )
        eigvals_val = np.linalg.eigvalsh(K_val_full)
        eigvals_val = np.maximum(eigvals_val, 0)    
        p_val = eigvals_val / (eigvals_val.sum() + 1e-12)
        eff_rank_full = float(np.exp(-np.sum(p_val * np.log(p_val + 1e-12))))
        logger.info(f"  Full-dataset eff_rank at gamma={outer_gamma}: {eff_rank_full:.2f}")

        if eff_rank_full < 8.0:
            logger.warning(
                f"  Selected gamma={outer_gamma} collapses at full n "
                f"(eff_rank={eff_rank_full:.2f} < 8.0). "
                f"Falling back to gamma=0.67 (AGPQK reference)."
            )
            outer_gamma = 0.67
            logger.info(f"  Fallback outer_gamma = {outer_gamma}")


    # Save test Bloch
    bloch_test_path = os.path.join(QCAK_EXP_DIR, "bloch_test.npy")
    bloch_test = compute_qcak_pqk_bloch_vectors(
        X8_test_enc, s_params, opt_qubits, sar_qubits, attention_pairs,
        n_qubits=N_QUBITS, reps=ZZ_REPS, save_path=bloch_test_path
    )


    # ── STAGE 6 — Compute full kernel matrices ───────────────────────────
    logger.info("\n--- Stage 6: Full Kernel Computation ---")
    K_train_path = os.path.join(QCAK_EXP_DIR, "K_qcak_pqk_train.npy")
    K_test_path  = os.path.join(QCAK_EXP_DIR, "K_qcak_pqk_test.npy")

    if os.path.exists(K_train_path):
        logger.info(f"  [SKIP] Loading K_train: {K_train_path}")
        K_train = np.load(K_train_path)
    else:
        K_train = compute_qcak_pqk_kernel_from_bloch(bloch_train, bloch_train, outer_gamma, n_qubits=N_QUBITS)
        np.save(K_train_path, K_train)

    if os.path.exists(K_test_path):
        logger.info(f"  [SKIP] Loading K_test: {K_test_path}")
        K_test = np.load(K_test_path)
    else:
        K_test = compute_qcak_pqk_kernel_from_bloch(bloch_test, bloch_train, outer_gamma, n_qubits=N_QUBITS)
        np.save(K_test_path, K_test)

    # Untrained baseline (all s_params = 1)
    s_untrained = np.ones_like(s_params)
    bloch_untrained_path = os.path.join(QCAK_EXP_DIR, "bloch_untrained.npy")
    bloch_untrained = compute_qcak_pqk_bloch_vectors(
        X8_enc, s_untrained, opt_qubits, sar_qubits, attention_pairs,
        save_path=bloch_untrained_path
    )
    K_untrained_path = os.path.join(QCAK_EXP_DIR, "K_qcak_pqk_untrained.npy")
    if os.path.exists(K_untrained_path):
        K_untrained = np.load(K_untrained_path)
    else:
        K_untrained = compute_qcak_pqk_kernel_from_bloch(bloch_untrained, bloch_untrained, outer_gamma)
        np.save(K_untrained_path, K_untrained)


    # ── STAGE 7 — Diagnostics and comparison ──────────────────────────────
    logger.info("\n--- Stage 7: Final Diagnostics ---")
    kta_trained   = compute_centered_kta(K_train, y_train, class_weighted=True)
    kta_untrained = compute_centered_kta(K_untrained, y_train, class_weighted=True)
    conc = compute_concentration_metrics(K_train)
    full_analysis = compute_full_kta_analysis(K_train, y_train)

    logger.info(f"  QCAK-PQK Trained KTA:   {kta_trained:.4f}")
    logger.info(f"  QCAK-PQK Untrained KTA: {kta_untrained:.4f}")
    logger.info(f"  Training Gain:         {kta_trained - kta_untrained:+.4f}")
    logger.info(f"  Concentration: CV={conc['cv']:.4f}, Rank={conc['effective_rank']:.2f}")

    # Results table
    logger.info("\n  Kernel               KTA      vs RBF")
    logger.info("  -------              -----    ------")
    for name, kta in BASELINES.items():
        logger.info(f"  {name:<20} {kta:>7.4f}  {kta - BASELINES['RBF']:>+7.4f}")
    logger.info(f"  {'QCAK-PQK (untrained)':<20} {kta_untrained:>7.4f}  {kta_untrained - BASELINES['RBF']:>+7.4f}")
    
    marker = " ★ BEATS RBF" if kta_trained > BASELINES['RBF'] else ""
    logger.info(f"  {'QCAK-PQK (trained)':<20} {kta_trained:>7.4f}  {kta_trained - BASELINES['RBF']:>+7.4f}{marker}")

    # Save summary
    summary = {
        "kta_trained": float(kta_trained),
        "kta_untrained": float(kta_untrained),
        "training_gain": float(kta_trained - kta_untrained),
        "delta_kta_vs_rbf": float(kta_trained - BASELINES['RBF']),
        "full_kta": {k: float(v) for k, v in full_analysis.items()},
        "concentration": {k: float(v) for k, v in conc.items() if k != "eigenvalue_spectrum"},
        "s_params_trained": s_params.tolist(),
        "outer_gamma": float(outer_gamma),
        "attention_pairs": attention_pairs,
        "opt_qubits": opt_qubits,
        "sar_qubits": sar_qubits,
        "feature_names": feature_names,
        "gamma_per_qubit": gamma_per_qubit.tolist(),
        "baselines": BASELINES
    }
    with open(os.path.join(QCAK_EXP_DIR, "qcak_pqk_results.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to results/physics/qcak_pqk/")
    print("\nQCAK-PQK Experiment Complete.")

if __name__ == "__main__":
    run_qcak_experiment()
