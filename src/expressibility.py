"""
Expressibility analysis for quantum feature maps.

Measures how much of Hilbert space the circuit can access by comparing
the pairwise fidelity distribution to the Haar-random reference.

Metric: KL divergence KL(P_data || P_Haar).
    - Lower KL = more expressive circuit.
    - Haar reference: P_Haar(F) = (2^n - 1)(1 - F)^{2^n - 2}.

Hypothesis: Fused features → lower KL → higher g from geometric difference.

Reference: Sim et al., Advanced Quantum Technologies 2(12), 1900070 (2019).
"""

import os
import sys
import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.special import rel_entr
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


def compute_haar_reference(
    n_qubits: int,
    n_bins: int = 75,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Haar-random fidelity distribution.

    P_Haar(F) = (2^n - 1)(1 - F)^{2^n - 2}

    Args:
        n_qubits: Number of qubits.
        n_bins: Number of histogram bins.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (bin_centers, probability_density).
    """
    dim = 2 ** n_qubits
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Haar density: (2^n - 1)(1 - F)^(2^n - 2)
    pdf = (dim - 1) * np.power(1 - bin_centers, dim - 2)

    # Normalize to sum to 1 (probability mass function)
    pmf = pdf * bin_width
    pmf = pmf / pmf.sum()

    return bin_centers, pmf


def compute_expressibility(
    feature_map_fn: Callable,
    n_qubits: int,
    X_data: np.ndarray,
    n_samples: int = config.EXPRESSIBILITY_N_SAMPLES,
    n_bins: int = 75,
    seed: int = config.RANDOM_SEED,
) -> Dict:
    """
    Compute expressibility of a quantum feature map.

    Samples n_samples pairs from X_data, computes F = |⟨ψ(x_i)|ψ(x_j)⟩|²
    for each pair, builds histogram, and computes KL divergence from Haar reference.

    Lower KL divergence = more expressive circuit.

    Args:
        feature_map_fn: Function that builds the FQK circuit and returns
                        the fidelity |⟨ψ(x_i)|ψ(x_j)⟩|². Signature: fn(x_i, x_j) → float.
        n_qubits: Number of qubits.
        X_data: Data matrix, shape (n, n_qubits). Pairs are sampled from here.
        n_samples: Number of pairs to sample.
        n_bins: Number of histogram bins.
        seed: Random seed for pair selection.

    Returns:
        dict: {
            'kl_divergence': float,  # KL(P_data || P_Haar)
            'fidelities': np.ndarray,  # Raw fidelity values
            'fidelity_histogram': np.ndarray,  # Histogram counts (normalized)
            'haar_reference': np.ndarray,  # Haar PMF
            'bin_centers': np.ndarray,
        }
    """
    rng = np.random.RandomState(seed)
    n = len(X_data)

    # Sample random pairs
    idx1 = rng.randint(0, n, n_samples)
    idx2 = rng.randint(0, n, n_samples)

    # Ensure different samples
    mask = idx1 == idx2
    while mask.any():
        idx2[mask] = rng.randint(0, n, mask.sum())
        mask = idx1 == idx2

    # Compute fidelities
    fidelities = np.zeros(n_samples)
    logger.info(f"Computing {n_samples} fidelities for expressibility...")

    for k in tqdm(range(n_samples), desc="Expressibility"):
        fidelities[k] = feature_map_fn(X_data[idx1[k]], X_data[idx2[k]])

    # Build histogram
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    hist, _ = np.histogram(fidelities, bins=bin_edges, density=False)
    # Normalize to probability distribution
    hist_pmf = hist.astype(float) / hist.sum()

    # Haar reference
    _, haar_pmf = compute_haar_reference(n_qubits, n_bins)

    # KL divergence: KL(P_data || P_Haar)
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    hist_pmf_safe = hist_pmf + eps
    haar_pmf_safe = haar_pmf + eps

    # Re-normalize after adding epsilon
    hist_pmf_safe /= hist_pmf_safe.sum()
    haar_pmf_safe /= haar_pmf_safe.sum()

    kl = float(np.sum(rel_entr(hist_pmf_safe, haar_pmf_safe)))

    logger.info(
        f"Expressibility: KL = {kl:.6f}, "
        f"mean fidelity = {fidelities.mean():.4f}, "
        f"std fidelity = {fidelities.std():.4f}"
    )

    return {
        "kl_divergence": kl,
        "fidelities": fidelities,
        "fidelity_histogram": hist_pmf,
        "haar_reference": haar_pmf,
        "bin_centers": bin_centers,
        "mean_fidelity": float(fidelities.mean()),
        "std_fidelity": float(fidelities.std()),
    }


def compare_modality_expressibility(
    X_sar: np.ndarray,
    X_opt: np.ndarray,
    X_fused: np.ndarray,
    feature_map_fn: Callable,
    n_qubits: int,
    n_samples: int = config.EXPRESSIBILITY_N_SAMPLES,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, Dict]:
    """
    Compare expressibility across SAR, Optical, and Fused modalities.

    Hypothesis: KL_fused < KL_sar, KL_optical (fused is more expressive).

    Args:
        X_sar: SAR features, shape (n, n_qubits).
        X_opt: Optical features, shape (n, n_qubits).
        X_fused: Fused features, shape (n, n_qubits).
        feature_map_fn: Fidelity computation function.
        n_qubits: Number of qubits.
        n_samples: Pairs per modality.
        seed: Random seed.

    Returns:
        dict: {modality_name: expressibility_results}.
    """
    results = {}

    for name, X in [("SAR", X_sar), ("Optical", X_opt), ("Fused", X_fused)]:
        logger.info(f"\n--- {name} Expressibility ---")
        results[name] = compute_expressibility(
            feature_map_fn, n_qubits, X, n_samples, seed=seed,
        )

    # Compare
    kl_sar = results["SAR"]["kl_divergence"]
    kl_opt = results["Optical"]["kl_divergence"]
    kl_fused = results["Fused"]["kl_divergence"]

    logger.info(f"\n--- Expressibility Comparison ---")
    logger.info(f"  KL(SAR)    = {kl_sar:.6f}")
    logger.info(f"  KL(Opt)    = {kl_opt:.6f}")
    logger.info(f"  KL(Fused)  = {kl_fused:.6f}")

    if kl_fused < kl_sar and kl_fused < kl_opt:
        logger.info("  ✓ Fused is most expressive (KL_fused < KL_sar, KL_opt)")
    else:
        logger.info("  ✗ Fused is NOT the most expressive (hypothesis not supported)")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  src/expressibility.py — Self-test")
    print("=" * 60)

    np.random.seed(config.RANDOM_SEED)

    # Test Haar reference
    n_qubits = 4
    centers, pmf = compute_haar_reference(n_qubits)
    print(f"\n  Haar reference: {len(centers)} bins, sum={pmf.sum():.4f}")
    assert abs(pmf.sum() - 1.0) < 1e-6

    # Small test: use a simple mock fidelity function
    def mock_fidelity(x1, x2):
        # Simple dot-product-based fidelity
        return float(np.exp(-np.sum((x1 - x2) ** 2)))

    X_test = np.random.rand(50, n_qubits) * np.pi

    result = compute_expressibility(
        mock_fidelity, n_qubits, X_test, n_samples=100, seed=42,
    )
    print(f"  KL divergence = {result['kl_divergence']:.6f}")
    print(f"  Mean fidelity = {result['mean_fidelity']:.4f}")
    assert result["kl_divergence"] >= 0.0

    print("\n  All self-tests passed.")
