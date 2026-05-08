"""
Kernel-Target Alignment (KTA) for the Quantum-Sat Classification pipeline.

Implements:
    - Standard KTA: KTA(K, y) = ⟨K, y_ideal⟩_F / (||K||_F · ||y_ideal||_F)
    - Centered KTA (HSIC-normalized): center K and y_ideal with H = I - (1/n)11ᵀ
    - ΔKTA: KTA(K_quantum, y) - KTA(K_classical, y)
    - Class-weighted KTA (Rule I6): normalizes ideal kernel to prevent
      dominant classes from controlling alignment.

CRITICAL INSIGHT:
    g measures if quantum is DIFFERENT; KTA measures if that difference is USEFUL.
    Need BOTH g >> 1 AND ΔKTA > 0 for a meaningful finding.

References:
    - Cristianini et al., NeurIPS 2001 — KTA definition.
    - arXiv:2502.08225 — quantum KTA.
"""

import os
import sys
import logging
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


def _build_ideal_kernel(
    y: np.ndarray,
    class_weighted: bool = True,
) -> np.ndarray:
    """
    Build the ideal kernel matrix from labels.

    Standard: y_ideal[i,j] = 1 if y[i] == y[j], else 0.
    Weighted (Rule I6): y_ideal[i,j] = 1/n_c if y[i] == y[j] == c, else 0.
    Weighting prevents Water-Water pairs (18.5%) from dominating alignment.

    Args:
        y: Label array, shape (n,).
        class_weighted: If True, use class-weighted version (DEFAULT).

    Returns:
        np.ndarray: Ideal kernel matrix, shape (n, n).
    """
    n = len(y)
    y_ideal = np.zeros((n, n), dtype=np.float64)

    if class_weighted:
        classes, counts = np.unique(y, return_counts=True)
        count_dict = dict(zip(classes, counts))

        for i in range(n):
            for j in range(n):
                if y[i] == y[j]:
                    y_ideal[i, j] = 1.0 / count_dict[y[i]]
    else:
        for i in range(n):
            y_ideal[i, :] = (y == y[i]).astype(float)

    return y_ideal


def compute_kta(
    K: np.ndarray,
    y: np.ndarray,
    class_weighted: bool = True,
) -> float:
    """
    Compute Kernel-Target Alignment.

    KTA(K, y) = ⟨K, y_ideal⟩_F / (||K||_F · ||y_ideal||_F)

    If class_weighted=True (DEFAULT per Rule I6): normalizes ideal kernel
    so each class contributes equally. Prevents Water-Water pairs from
    dominating the alignment score.

    Args:
        K: Kernel matrix, shape (n, n).
        y: Labels, shape (n,).
        class_weighted: Use class-weighted ideal kernel (default: True).

    Returns:
        float: KTA value in [-1, 1].
    """
    y_ideal = _build_ideal_kernel(y, class_weighted)

    # Frobenius inner product
    kta_num = np.sum(K * y_ideal)
    kta_den = np.sqrt(np.sum(K * K) * np.sum(y_ideal * y_ideal))

    if kta_den < 1e-15:
        logger.warning("KTA denominator near zero. Returning 0.0.")
        return 0.0

    kta = float(kta_num / kta_den)

    return kta


def compute_centered_kta(
    K: np.ndarray,
    y: np.ndarray,
    class_weighted: bool = True,
) -> float:
    """
    Compute Centered (HSIC-normalized) Kernel-Target Alignment.

    Centers both K and y_ideal with H = I - (1/n)11ᵀ before computing alignment.
    K_c = HKH, y_c = HyH.

    Args:
        K: Kernel matrix, shape (n, n).
        y: Labels, shape (n,).
        class_weighted: Use class-weighted ideal kernel.

    Returns:
        float: Centered KTA value.
    """
    n = K.shape[0]
    y_ideal = _build_ideal_kernel(y, class_weighted)

    # Centering matrix H = I - (1/n)11ᵀ
    H = np.eye(n) - np.ones((n, n)) / n

    K_c = H @ K @ H
    y_c = H @ y_ideal @ H

    kta_num = np.sum(K_c * y_c)
    kta_den = np.sqrt(np.sum(K_c * K_c) * np.sum(y_c * y_c))

    if kta_den < 1e-15:
        logger.warning("Centered KTA denominator near zero. Returning 0.0.")
        return 0.0

    return float(kta_num / kta_den)


def compute_delta_kta(
    K_quantum: np.ndarray,
    K_classical: np.ndarray,
    y: np.ndarray,
    class_weighted: bool = True,
    use_centered: bool = True,
) -> float:
    """
    Compute ΔKTA = KTA(K_quantum, y) - KTA(K_classical, y).

    Positive ΔKTA means quantum kernel is more task-aligned.

    Args:
        K_quantum: Quantum kernel matrix.
        K_classical: Classical kernel matrix.
        y: Labels.
        class_weighted: Use class-weighted KTA.
        use_centered: Use centered KTA variant (default: True).
            Centered KTA is the primary metric per Cortes, Mohri & Rostamizadeh
            (JMLR, 2012) as it removes mean-offset bias from high-similarity kernels
            such as RBF (mean_offdiag ≈ 0.80) that inflate standard KTA numerators.

    Returns:
        float: ΔKTA value. Positive = quantum more aligned.
    """
    kta_fn = compute_centered_kta if use_centered else compute_kta

    kta_q = kta_fn(K_quantum, y, class_weighted)
    kta_c = kta_fn(K_classical, y, class_weighted)
    delta = kta_q - kta_c

    logger.info(
        f"ΔKTA = {delta:.6f} (KTA_q={kta_q:.6f}, KTA_c={kta_c:.6f}) "
        f"[{'weighted' if class_weighted else 'unweighted'}, "
        f"{'centered (PRIMARY)' if use_centered else 'standard (BIASED — do not use as primary)'}]"
    )

    return delta


def compute_full_kta_analysis(
    K: np.ndarray,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all KTA variants for comprehensive reporting.

    Returns both weighted and unweighted, standard and centered KTA values.
    Weighted is PRIMARY per Rule I6.

    Args:
        K: Kernel matrix.
        y: Labels.

    Returns:
        dict: {
            'kta_weighted': float,  # PRIMARY
            'kta_unweighted': float,
            'kta_centered_weighted': float,
            'kta_centered_unweighted': float,
        }
    """
    return {
        "kta_weighted": compute_kta(K, y, class_weighted=True),
        "kta_unweighted": compute_kta(K, y, class_weighted=False),
        "kta_centered_weighted": compute_centered_kta(K, y, class_weighted=True),
        "kta_centered_unweighted": compute_centered_kta(K, y, class_weighted=False),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  src/kernel_target_alignment.py — Self-test")
    print("=" * 60)

    np.random.seed(config.RANDOM_SEED)
    n = 30

    # Test: KTA of ideal kernel should be 1.0 (unweighted)
    y = np.random.randint(0, 5, n)
    K_ideal = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K_ideal[i, j] = 1.0 if y[i] == y[j] else 0.0

    kta_ideal = compute_kta(K_ideal, y, class_weighted=False)
    print(f"\n  KTA(ideal kernel, y) = {kta_ideal:.6f} (expected 1.0)")
    assert abs(kta_ideal - 1.0) < 1e-6

    # Test: KTA in valid range
    K_rand = np.random.rand(n, n)
    K_rand = (K_rand + K_rand.T) / 2
    kta_rand = compute_kta(K_rand, y, class_weighted=False)
    print(f"  KTA(random kernel) = {kta_rand:.6f} (expected in [-1, 1])")
    assert -1.0 <= kta_rand <= 1.0

    # Test: ΔKTA = 0 for identical kernels
    delta = compute_delta_kta(K_rand, K_rand, y)
    print(f"  ΔKTA(K, K) = {delta:.6f} (expected 0.0)")
    assert abs(delta) < 1e-10

    # Test: class-weighted KTA
    kta_w = compute_kta(K_rand, y, class_weighted=True)
    kta_uw = compute_kta(K_rand, y, class_weighted=False)
    print(f"  Weighted KTA = {kta_w:.6f}, Unweighted = {kta_uw:.6f}")

    # Test: all-same-class edge case
    y_same = np.zeros(n, dtype=int)
    kta_same = compute_kta(K_rand, y_same, class_weighted=False)
    print(f"  KTA(all same class) = {kta_same:.6f}")

    # Full analysis
    full = compute_full_kta_analysis(K_rand, y)
    print(f"\n  Full KTA analysis: {full}")

    print("\n  All self-tests passed.")
