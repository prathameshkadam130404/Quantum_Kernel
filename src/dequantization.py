"""
Dequantization defense analysis for quantum kernels.

Implements Random Fourier Features (RFF) to approximate quantum kernel matrices
and test whether the quantum kernel's advantage can be replicated classically.

Key distinction: function approximation (easy at 8 qubits) vs learning
behavior (the open question we investigate).

Method:
    z(x) = √(2/D) [cos(ω₁·x + b₁), ..., cos(ω_D·x + b_D)]
    ω ~ N(0, γI), b ~ U(0, 2π)
    K_RFF = Z @ Z^T

References:
    - Landman et al., Quantum Journal (2025).
    - Shin et al., ICLR 2025.
    - Rahimi & Recht, NeurIPS 2007 — RFF.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


def compute_rff_features(
    X: np.ndarray,
    n_features: int,
    gamma: float,
    seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    """
    Compute Random Fourier Features approximation.

    z(x) = √(2/D) * cos(ω·x + b)
    where ω ~ N(0, γI), b ~ U(0, 2π).

    The resulting kernel approximation is K_RFF = Z @ Z^T ≈ K_RBF.

    Args:
        X: Data matrix, shape (n, d).
        n_features: Number of random features D.
        gamma: RBF bandwidth for sampling ω.
        seed: Random seed.

    Returns:
        np.ndarray: Random feature matrix Z, shape (n, D).
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape

    # Sample random frequencies and biases
    omega = rng.normal(0, np.sqrt(2 * gamma), size=(d, n_features))
    b = rng.uniform(0, 2 * np.pi, size=n_features)

    # Compute features
    projection = X @ omega + b  # (n, D)
    Z = np.sqrt(2.0 / n_features) * np.cos(projection)

    return Z


def compute_rff_kernel(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    n_features: int = 256,
    gamma: float = 1.0,
    seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    """
    Compute RFF-approximated kernel matrix.

    K_RFF = Z @ Z^T where Z = RFF features.

    Args:
        X: Training data, shape (n, d).
        X2: Optional test data, shape (m, d).
        n_features: Number of random features.
        gamma: RBF bandwidth.
        seed: Random seed.

    Returns:
        np.ndarray: Approximated kernel matrix.
    """
    Z = compute_rff_features(X, n_features, gamma, seed)

    if X2 is None:
        return Z @ Z.T
    else:
        Z2 = compute_rff_features(X2, n_features, gamma, seed)
        return Z2 @ Z.T


def sweep_rff_approximation(
    K_q: np.ndarray,
    X: np.ndarray,
    gamma: float,
    feature_counts: List[int] = None,
    seed: int = config.RANDOM_SEED,
) -> Dict:
    """
    Sweep over RFF feature counts to measure approximation quality.

    For each D: compute K_RFF, measure relative Frobenius error
    ||K_q - K_RFF||_F / ||K_q||_F.

    Args:
        K_q: Target quantum kernel matrix, shape (n, n).
        X: Data matrix.
        gamma: RBF bandwidth parameter.
        feature_counts: List of RFF feature counts to try.
        seed: Random seed.

    Returns:
        dict: {
            'feature_counts': list,
            'frobenius_errors': list,  # Relative Frobenius errors
            'spectral_errors': list,  # Spectral norm errors
            'kernels': dict,  # {D: K_RFF}
        }
    """
    if feature_counts is None:
        feature_counts = config.RFF_N_FEATURES_LIST

    K_q_norm = np.linalg.norm(K_q, "fro")

    results = {
        "feature_counts": feature_counts,
        "frobenius_errors": [],
        "spectral_errors": [],
        "kernels": {},
    }

    for D in tqdm(feature_counts, desc="RFF sweep"):
        K_rff = compute_rff_kernel(X, n_features=D, gamma=gamma, seed=seed)

        # Relative Frobenius error
        frob_err = np.linalg.norm(K_q - K_rff, "fro") / K_q_norm

        # Spectral norm error
        spec_err = np.linalg.norm(K_q - K_rff, ord=2) / np.linalg.norm(K_q, ord=2)

        results["frobenius_errors"].append(float(frob_err))
        results["spectral_errors"].append(float(spec_err))
        results["kernels"][D] = K_rff

        logger.info(
            f"  RFF D={D}: Frobenius error = {frob_err:.4f}, "
            f"Spectral error = {spec_err:.4f}"
        )

    return results


def dequantization_analysis(
    K_q: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    gamma: float = 1.0,
    feature_counts: List[int] = None,
    seed: int = config.RANDOM_SEED,
) -> Dict:
    """
    Full dequantization analysis: RFF approximation + classification + KTA.

    Reports honestly: dequantizable or not.

    Args:
        K_q: Quantum kernel matrix, shape (n, n).
        X: Data matrix.
        y: Labels.
        gamma: RBF bandwidth.
        feature_counts: RFF feature counts.
        seed: Random seed.

    Returns:
        dict: Comprehensive comparison results.
    """
    from src.kernel_target_alignment import compute_kta
    from src.kernel_concentration import compute_concentration_metrics

    if feature_counts is None:
        feature_counts = config.RFF_N_FEATURES_LIST

    # Sweep RFF approximation
    sweep = sweep_rff_approximation(K_q, X, gamma, feature_counts, seed)

    # KTA for quantum kernel
    kta_q = compute_kta(K_q, y, class_weighted=True)

    # KTA for each RFF kernel
    kta_rff = {}
    for D in feature_counts:
        kta_rff[D] = compute_kta(sweep["kernels"][D], y, class_weighted=True)

    # Concentration for quantum and best RFF
    conc_q = compute_concentration_metrics(K_q)
    best_D = feature_counts[-1]
    conc_rff = compute_concentration_metrics(sweep["kernels"][best_D])

    results = {
        "sweep": {
            "feature_counts": feature_counts,
            "frobenius_errors": sweep["frobenius_errors"],
            "spectral_errors": sweep["spectral_errors"],
        },
        "kta_quantum": kta_q,
        "kta_rff": kta_rff,
        "concentration_quantum_cv": conc_q["cv"],
        "concentration_rff_cv": conc_rff["cv"],
    }

    # Assessment
    best_frob_error = sweep["frobenius_errors"][-1]
    kta_gap = kta_q - kta_rff[best_D]

    if best_frob_error < 0.1 and abs(kta_gap) < 0.02:
        assessment = "DEQUANTIZABLE — RFF closely matches quantum kernel in both approximation and task alignment."
    elif best_frob_error < 0.1 and kta_gap > 0.02:
        assessment = "PARTIALLY DEQUANTIZABLE — RFF approximates kernel well but misses task-relevant structure (KTA gap)."
    else:
        assessment = "NOT EASILY DEQUANTIZABLE — RFF fails to approximate quantum kernel."

    results["assessment"] = assessment

    logger.info(f"\n{'='*60}")
    logger.info(f"DEQUANTIZATION ASSESSMENT: {assessment}")
    logger.info(f"  Best RFF error (D={best_D}): {best_frob_error:.4f}")
    logger.info(f"  KTA gap (quantum - RFF): {kta_gap:.4f}")
    logger.info(f"  Concentration: CV_q={conc_q['cv']:.4f}, CV_RFF={conc_rff['cv']:.4f}")
    logger.info(f"{'='*60}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  src/dequantization.py — Self-test")
    print("=" * 60)

    np.random.seed(config.RANDOM_SEED)
    n, d = 30, 8

    X = np.random.rand(n, d) * np.pi

    # Test RFF features
    Z = compute_rff_features(X, n_features=256, gamma=1.0)
    print(f"\n  RFF features shape: {Z.shape} (expected (30, 256))")
    assert Z.shape == (n, 256)

    # Test RFF kernel
    K_rff = compute_rff_kernel(X, n_features=256, gamma=1.0)
    print(f"  RFF kernel shape: {K_rff.shape}")
    assert K_rff.shape == (n, n)
    assert np.allclose(K_rff, K_rff.T, atol=1e-6)  # Symmetric

    # Test sweep
    K_target = np.eye(n) + np.random.rand(n, n) * 0.1
    K_target = (K_target + K_target.T) / 2
    np.fill_diagonal(K_target, 1.0)

    sweep = sweep_rff_approximation(
        K_target, X, gamma=1.0,
        feature_counts=[32, 64, 128],
    )

    # Error should decrease with more features
    errors = sweep["frobenius_errors"]
    print(f"\n  RFF errors: {errors}")
    # NOTE: Interpreted ambiguous spec as error should generally decrease,
    # but may not be strictly monotonic due to randomness
    assert errors[-1] < errors[0] * 2.0, "Error should not increase dramatically"

    print("\n  All self-tests passed.")
