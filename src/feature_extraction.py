"""
Feature extraction pipeline for Quantum-Sat Classification.

Provides IncrementalPCA dimensionality reduction and min-max normalization
to [0, π] for quantum encoding. All transformations fit on training data
ONLY to prevent data leakage.

Usage:
    from src.feature_extraction import (
        fit_pca_pipeline,
        transform_features,
        fit_and_transform_full_pipeline,
    )
"""

import os
import sys
import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import IncrementalPCA
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


def fit_pca_pipeline(
    X_train: np.ndarray,
    n_components: int = config.N_PCA_COMPONENTS,
    batch_size: int = 1000,
) -> IncrementalPCA:
    """
    Fit IncrementalPCA on training data only.

    Uses IncrementalPCA for memory efficiency with large datasets.

    Args:
        X_train: Training features, shape (n_train, d).
        n_components: Number of PCA components (default: 8 from config).
        batch_size: Batch size for incremental fitting.

    Returns:
        IncrementalPCA: Fitted PCA transformer.

    Raises:
        ValueError: If n_components > X_train.shape[1].
    """
    if n_components > X_train.shape[1]:
        raise ValueError(
            f"n_components ({n_components}) > n_features ({X_train.shape[1]}). "
            f"Reduce n_components or use higher-dimensional input."
        )

    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    n_samples = X_train.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size

    logger.info(
        f"Fitting IncrementalPCA: {X_train.shape} → {n_components}d "
        f"({n_batches} batches of {batch_size})"
    )

    for i in range(0, n_samples, batch_size):
        batch = X_train[i: i + batch_size]
        pca.partial_fit(batch)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(f"PCA explained variance: {explained:.4f} ({explained * 100:.1f}%)")

    return pca


def normalize_to_pi(
    X: np.ndarray,
    fit_min: Optional[np.ndarray] = None,
    fit_max: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Percentile-robust normalize features to [0, π] for quantum encoding.

    Uses 1st and 99th percentiles instead of absolute min/max to prevent
    extreme PCA outliers from compressing the bulk of data into a tiny
    sub-range. Values outside [P1, P99] are clipped to [0, π] boundaries.

    The ZZFeatureMap uses angles in [0, π] for Pauli rotations. Without
    robust normalization, SAR PCA features (range ~16,000) compress 99%
    of data into <0.3% of [0, π], causing severe kernel concentration
    (all off-diagonal entries ≈ 1.0).

    Args:
        X: Features, shape (n, d).
        fit_min: Pre-computed per-feature 1st percentiles (from training set).
                 If None, compute from X (only valid for training data).
        fit_max: Pre-computed per-feature 99th percentiles (from training set).
                 If None, compute from X (only valid for training data).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - Normalized features in [0, π], shape (n, d).
            - Per-feature percentile minimums used.
            - Per-feature percentile maximums used.
    """
    if fit_min is None:
        fit_min = np.percentile(X, 1, axis=0)
    if fit_max is None:
        fit_max = np.percentile(X, 99, axis=0)

    # Avoid division by zero for constant features
    feature_range = fit_max - fit_min
    feature_range = np.where(feature_range == 0, 1.0, feature_range)

    X_norm = (X - fit_min) / feature_range  # ~[0, 1] for 98% of data
    X_norm = np.clip(X_norm, 0.0, 1.0)  # Clip outliers to [0, 1]
    X_scaled = X_norm * np.pi  # [0, π]

    return X_scaled, fit_min, fit_max


def fit_and_transform_full_pipeline(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray],
    X_test: np.ndarray,
    n_components: int = config.N_PCA_COMPONENTS,
    batch_size: int = 1000,
    save_dir: Optional[str] = None,
    prefix: str = "",
) -> Dict[str, np.ndarray]:
    """
    Complete feature extraction pipeline: PCA → [0, π] normalization.

    Fits PCA and normalization on training data ONLY, then transforms
    all splits. Optionally saves PCA model and normalization params.

    Args:
        X_train: Training features, shape (n_train, d_original).
        X_val: Validation features, shape (n_val, d_original), or None.
        X_test: Test features, shape (n_test, d_original).
        n_components: Number of PCA components.
        batch_size: Batch size for IncrementalPCA.
        save_dir: Directory to save PCA model + normalization params.
        prefix: Filename prefix for saved models (e.g., "sar_", "optical_").

    Returns:
        dict: Keys 'X_train', 'X_val' (if provided), 'X_test',
              'pca', 'norm_min', 'norm_max', 'explained_variance'.
    """
    # Step 1: PCA
    pca = fit_pca_pipeline(X_train, n_components, batch_size)

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    X_val_pca = pca.transform(X_val) if X_val is not None else None

    # Step 2: Normalize to [0, π] — fit on training ONLY
    X_train_scaled, fit_min, fit_max = normalize_to_pi(X_train_pca)
    X_test_scaled, _, _ = normalize_to_pi(X_test_pca, fit_min, fit_max)
    X_val_scaled = None
    if X_val_pca is not None:
        X_val_scaled, _, _ = normalize_to_pi(X_val_pca, fit_min, fit_max)

    # Verify feature ranges
    train_min, train_max = X_train_scaled.min(), X_train_scaled.max()
    logger.info(
        f"Feature range after normalization: "
        f"train=[{train_min:.4f}, {train_max:.4f}], "
        f"expected=[0, π={np.pi:.4f}]"
    )

    # Save models if requested
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        pca_path = os.path.join(save_dir, f"{prefix}pca_model.joblib")
        norm_path = os.path.join(save_dir, f"{prefix}norm_params.npz")

        joblib.dump(pca, pca_path)
        np.savez(norm_path, fit_min=fit_min, fit_max=fit_max)
        logger.info(f"Saved PCA model: {pca_path}")
        logger.info(f"Saved normalization params: {norm_path}")

    result = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "pca": pca,
        "norm_min": fit_min,
        "norm_max": fit_max,
        "explained_variance": pca.explained_variance_ratio_,
    }
    if X_val_scaled is not None:
        result["X_val"] = X_val_scaled

    return result


def transform_with_saved_pipeline(
    X: np.ndarray,
    pca_path: str,
    norm_path: str,
) -> np.ndarray:
    """
    Transform features using a previously saved PCA + normalization pipeline.

    Args:
        X: Raw features, shape (n, d_original).
        pca_path: Path to saved PCA model (.joblib).
        norm_path: Path to saved normalization params (.npz).

    Returns:
        np.ndarray: Transformed features in [0, π], shape (n, n_components).

    Raises:
        FileNotFoundError: If model files are not found.
    """
    if not os.path.exists(pca_path):
        raise FileNotFoundError(f"PCA model not found: {pca_path}")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalization params not found: {norm_path}")

    pca = joblib.load(pca_path)
    norm = np.load(norm_path)

    X_pca = pca.transform(X)
    X_scaled, _, _ = normalize_to_pi(X_pca, norm["fit_min"], norm["fit_max"])

    return X_scaled


if __name__ == "__main__":
    print("=" * 60)
    print("  src/feature_extraction.py — Self-test")
    print("=" * 60)

    logging.basicConfig(level=logging.INFO)

    np.random.seed(config.RANDOM_SEED)

    # Simulate data
    X_train = np.random.randn(200, 100)
    X_test = np.random.randn(50, 100)

    result = fit_and_transform_full_pipeline(
        X_train, None, X_test,
        n_components=8,
        save_dir=None,
    )

    print(f"  Train shape: {result['X_train'].shape}")
    print(f"  Test shape:  {result['X_test'].shape}")
    print(f"  Train range: [{result['X_train'].min():.4f}, {result['X_train'].max():.4f}]")
    print(f"  Test range:  [{result['X_test'].min():.4f}, {result['X_test'].max():.4f}]")
    print(f"  Explained var: {result['explained_variance'].sum():.4f}")

    assert result["X_train"].shape == (200, 8)
    assert result["X_test"].shape == (50, 8)
    assert result["X_train"].min() >= -0.01  # Allow tiny float error
    assert result["X_train"].max() <= np.pi + 0.01

    print()
    print("  All self-tests passed.")
