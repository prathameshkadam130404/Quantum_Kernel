"""
Geometric difference computation for quantum vs classical kernels.

Implements Huang's geometric difference metric:
    g(K_q, K_c) = √(||K_q^{1/2} (K_c + λI)^{-1} K_q^{1/2}||_∞)

where ||·||_∞ is the spectral norm (max eigenvalue).

Interpretation:
    - g ≈ 1: quantum and classical kernels have identical geometry.
    - g >> 1: quantum kernel captures structure that classical kernel misses.

IMPORTANT: High g alone is insufficient. Must verify ΔKTA > 0 to confirm
the geometric difference is TASK-RELEVANT.

Reference: Huang et al., Nature Communications 12, 2631 (2021).

Numerical stability:
    - Uses np.linalg.eigh (not eig) for Hermitian matrices.
    - Clips eigenvalues < 1e-10 to avoid division by zero.
    - Regularization λ = 1e-6 default.
"""

import os
import sys
import logging
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


def compute_geometric_difference(
    K_q: np.ndarray,
    K_c: np.ndarray,
    lambda_reg: Optional[float] = None,
    normalize_trace: bool = True,
) -> Tuple[float, Dict]:
    """
    Compute Huang's geometric difference g(K_q, K_c).

    g = sqrt(spectral_norm(K_q^{1/2} @ (K_c + λI)^{-1} @ K_q^{1/2}))

    IMPORTANT: Uses λ = 1/n (data-adaptive) by default, NOT 1e-6.
    Classical kernels with near-degenerate spectra (e.g., RBF with
    mean_offdiag≈0.80, effective_rank≈1.2) cause g to explode numerically
    when λ is too small. λ = 1/n is the statistically principled choice.

    If normalize_trace=True (default): both K_q and K_c are divided by
    their respective traces before computing g, so diagonal means ≈ 1.
    This removes scale differences and makes g values comparable across
    kernel types.

    Args:
        K_q: Quantum kernel matrix, shape (n, n). Must be symmetric, PSD.
        K_c: Classical kernel matrix, shape (n, n). Must be symmetric.
        lambda_reg: Regularization. If None, uses 1/n (RECOMMENDED).
        normalize_trace: Divide K by trace(K)/n before computing. Default True.

    Returns:
        Tuple[float, Dict]:
            - g: Geometric difference value (≥ 1.0 - 1e-6).
            - diagnostics: Dict with intermediate values:
                lambda_used, effective_rank_Kc, condition_number_Kc,
                mean_offdiag_Kc, trace_norm_applied, etc.
    """
    # ---- Validate inputs ----
    if K_q.shape != K_c.shape:
        raise ValueError(
            f"Shape mismatch: K_q={K_q.shape}, K_c={K_c.shape}"
        )
    if K_q.shape[0] != K_q.shape[1]:
        raise ValueError(f"K_q is not square: {K_q.shape}")

    n = K_q.shape[0]

    if lambda_reg is None:
        lambda_reg = 1.0 / n

    if normalize_trace:
        tr_q = np.trace(K_q)
        tr_c = np.trace(K_c)
        if tr_q > 1e-15:
            K_q = K_q / (tr_q / n)
        if tr_c > 1e-15:
            K_c = K_c / (tr_c / n)

    # ---- Symmetrize (handle numerical asymmetry) ----
    K_q = (K_q + K_q.T) / 2
    K_c = (K_c + K_c.T) / 2

    # Pre-computation diagnostic metrics for K_c
    eig_c, V_c = np.linalg.eigh(K_c)
    max_eig_c = max(eig_c[-1], 1e-10)
    eff_rank_c = np.trace(K_c) / max_eig_c
    cond_num_c = eig_c[-1] / (max(eig_c[0], 0.0) + lambda_reg)
    offdiag_c = K_c[~np.eye(n, dtype=bool)]
    mean_od_c = float(np.mean(offdiag_c))

    # ---- Compute K_q^{1/2} via eigendecomposition ----
    eig_q, V_q = np.linalg.eigh(K_q)
    # Clip small/negative eigenvalues for stability
    eig_q_clipped = np.clip(eig_q, 1e-10, None)
    K_q_sqrt = V_q @ np.diag(np.sqrt(eig_q_clipped)) @ V_q.T

    # ---- Compute (K_c + λI)^{-1} ----
    # Regularized inverse using already computed eigendecomposition
    eig_c_reg_inv = 1.0 / (np.clip(eig_c, 0.0, None) + lambda_reg)
    K_c_reg_inv = V_c @ np.diag(eig_c_reg_inv) @ V_c.T

    # ---- Compute M = K_q^{1/2} @ (K_c + λI)^{-1} @ K_q^{1/2} ----
    M = K_q_sqrt @ K_c_reg_inv @ K_q_sqrt

    # Symmetrize M (numerical)
    M = (M + M.T) / 2

    # ---- Spectral norm = max eigenvalue of M ----
    eig_M = np.linalg.eigvalsh(M)
    spectral_norm = float(np.max(eig_M))

    # ---- g = sqrt(spectral_norm) ----
    g = float(np.sqrt(max(spectral_norm, 0.0)))

    # ---- Validity check ----
    if g < 1.0 - 1e-6:
        logger.warning(
            f"Geometric difference g={g:.6f} < 1.0. "
            f"This may indicate numerical issues. "
            f"K_q eigenvalues: min={eig_q.min():.2e}, max={eig_q.max():.2e}. "
            f"K_c eigenvalues: min={eig_c.min():.2e}, max={eig_c.max():.2e}."
        )

    # ---- Diagnostics ----
    K_q_rank = int(np.sum(eig_q > 1e-10))
    K_c_rank = int(np.sum(eig_c > 1e-10))

    diagnostics = {
        "lambda_used": lambda_reg,
        "effective_rank_Kc": float(eff_rank_c),
        "condition_number_Kc": float(cond_num_c),
        "mean_offdiag_Kc": mean_od_c,
        "trace_norm_applied": normalize_trace,
        "K_q_eigenvalues": eig_q,
        "K_c_eigenvalues": eig_c,
        "M_eigenvalues": eig_M,
        "K_q_rank": K_q_rank,
        "K_c_rank": K_c_rank,
        "spectral_norm": spectral_norm,
        "g": g,
    }

    logger.info(
        f"g(K_q, K_c) = {g:.4f} | "
        f"n={n}, lambda={lambda_reg:.2e} (1/n={1.0/n:.2e}) | "
        f"K_q rank={K_q_rank}/{n}, K_c rank={K_c_rank}/{n} | "
        f"spectral_norm={spectral_norm:.4f} | "
        f"trace_normalized={normalize_trace}"
    )
    if abs(lambda_reg - 1.0 / n) > 1e-10:
        logger.warning(
            f"  WARNING: lambda_reg={lambda_reg:.2e} != 1/n={1.0/n:.2e}. "
            f"g values computed with different lambda are NOT directly comparable. "
            f"Use lambda=1/n for all experiments."
        )

    return g, diagnostics


def compute_geometric_difference_lambda_sweep(
    K_q: np.ndarray,
    K_c: np.ndarray,
    lambda_values: Optional[list] = None,
    normalize_trace: bool = True,
    save_path: Optional[str] = None,
) -> dict:
    """
    Compute g across a range of lambda values to assess sensitivity.

    Runs compute_geometric_difference() for each lambda in lambda_values.
    Plots g vs lambda (log scale x-axis) if save_path provided.
    The stability of g across lambda values indicates whether geometric
    separation is genuine or a numerical conditioning artifact.

    Default lambda_values: [1e-6, 1e-5, 1e-4, 1e-3, 1/n, 0.01, 0.1]
    where 1/n is computed from K_q.shape[0].

    Returns:
        dict: {lambda: g_value} plus 'recommended_lambda' = 1/n,
              'stable' = True if std(g_values) / mean(g_values) < 0.5
    """
    n = K_q.shape[0]
    rec_lambda = 1.0 / n
    if lambda_values is None:
        lambda_values = [1e-6, 1e-5, 1e-4, 1e-3, rec_lambda, 0.01, 0.1]
    
    # Sort just in case, but keep recommended in the list if it isn't specifically
    if rec_lambda not in lambda_values:
        lambda_values.append(rec_lambda)
    lambda_values = sorted(lambda_values)

    results = {}
    g_vals = []
    
    # Suppress verbose logging during sweep
    old_level = logger.level
    logger.setLevel(logging.WARNING)
    
    for l_reg in lambda_values:
        g, _ = compute_geometric_difference(
            K_q, K_c, lambda_reg=l_reg, normalize_trace=normalize_trace
        )
        results[l_reg] = g
        g_vals.append(g)
        
    logger.setLevel(old_level)
    
    std_g = np.std(g_vals)
    mean_g = np.mean(g_vals)
    is_stable = (std_g / mean_g) < 0.5 if mean_g > 0 else False
    
    results_dict = {
        "sweep_results": results,
        "recommended_lambda": rec_lambda,
        "recommended_g": results[rec_lambda],
        "stable": bool(is_stable),
    }

    if save_path:
        import matplotlib.pyplot as plt
        import config
        from src.utils import create_publication_plot
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(lambda_values, g_vals, 'o-', linewidth=2, color='steelblue')
        ax.axvline(x=rec_lambda, color='coral', linestyle='--', linewidth=2, label=f'Recommended λ = 1/n ({rec_lambda:.1e})')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Regularization Parameter (λ)', fontsize=12)
        ax.set_ylabel('Geometric Difference (g)', fontsize=12)
        ax.set_title('Sensitivity of Geometric Difference to Regularization', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        create_publication_plot(fig, save_path.replace('.png', ''))
        
    return results_dict


def compute_geometric_difference_batch(
    K_quantum_dict: Dict[str, np.ndarray],
    K_classical_dict: Dict[str, np.ndarray],
    lambda_reg: Optional[float] = None,
    normalize_trace: bool = True,
) -> Dict[str, Dict]:
    """
    Compute geometric difference for all quantum-classical pairs.

    Args:
        K_quantum_dict: {name: kernel_matrix} for quantum kernels.
        K_classical_dict: {name: kernel_matrix} for classical kernels.
        lambda_reg: Regularization parameter. Defaults to 1/n.
        normalize_trace: Apply trace normalization.

    Returns:
        Dict[str, Dict]: {f"{q_name}_vs_{c_name}": {"g": float, "diagnostics": dict}}.
    """
    results = {}

    for q_name, K_q in K_quantum_dict.items():
        for c_name, K_c in K_classical_dict.items():
            key = f"{q_name}_vs_{c_name}"
            try:
                g, diag = compute_geometric_difference(
                    K_q, K_c, lambda_reg=lambda_reg, normalize_trace=normalize_trace
                )
                results[key] = {"g": g, "diagnostics": diag}
            except Exception as e:
                logger.error(f"Failed computing g({q_name}, {c_name}): {e}")
                results[key] = {"g": float("nan"), "error": str(e)}

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  src/geometric_difference.py — Self-test")
    print("=" * 60)

    np.random.seed(config.RANDOM_SEED)

    n = 20

    # Identity test: g(K, K) should be 1.0
    K = np.eye(n) * 0.5 + 0.5
    K = K @ K.T  # Ensure PSD
    K = K / K.max()  # Normalize
    np.fill_diagonal(K, 1.0)

    g_self, diag_self = compute_geometric_difference(K, K)
    print(f"\n  g(K, K) = {g_self:.6f} (expected ≈ 1.0)")
    assert abs(g_self - 1.0) < 0.01, f"g(K,K) = {g_self}, expected 1.0"

    # Different kernels: g should be > 1
    K1 = np.eye(n) + np.random.rand(n, n) * 0.1
    K1 = (K1 + K1.T) / 2
    np.fill_diagonal(K1, 1.0)

    K2 = np.eye(n) + np.random.rand(n, n) * 0.3
    K2 = (K2 + K2.T) / 2
    np.fill_diagonal(K2, 1.0)

    g_diff, diag_diff = compute_geometric_difference(K1, K2)
    print(f"  g(K1, K2) = {g_diff:.6f} (expected > 1.0)")
    assert g_diff >= 1.0 - 1e-6, f"g = {g_diff} < 1.0"

    # Near-singular test
    K_singular = np.eye(n) * 0.01
    K_singular[0, 0] = 1.0
    K_singular = (K_singular + K_singular.T) / 2

    g_sing, _ = compute_geometric_difference(K1, K_singular)
    print(f"  g(K1, K_singular) = {g_sing:.6f} (near-singular test)")

    print("\n  All self-tests passed.")
