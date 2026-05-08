"""
Topological feature extraction using cubical persistent homology (Giotto-TDA).

Pipeline per SAR image:
  - For each of 8 channels independently:
      CubicalPersistence(H0, H1) on the 32×32 pixel sublevel-set filtration
      PersistenceImage(sigma=0.1, n_bins=10) → 200-d per channel
  - Concatenate across channels → 1600-d per image
  - PCA to N_PCA_COMPONENTS dimensions (fit on training set only)
  - Normalise to [0, π]

WHY CUBICAL PERSISTENCE, NOT VIETORIS-RIPS:
  VR on spatial+spectral point clouds mixes coordinate scales with
  intensity scales — the filtration distance is not physically meaningful.
  Cubical persistence uses the pixel grid directly as a cubical complex,
  filtering by pixel intensity (sublevel-set filtration). This is the
  standard approach for image TDA in the literature and correctly captures
  topological structure of building patterns, vegetation density, and
  enclosed spaces in SAR imagery.

WHY PER-CHANNEL:
  SAR channels (VV, VH, derived quantities) have different intensity ranges
  and physical meanings. Joint persistence across channels conflates these.
  Per-channel processing is standard in multi-channel image TDA.

References:
  - Tauzin et al., giotto-tda (JMLR 2021) — CubicalPersistence for images
  - Sgier et al., "A TDA approach for Local Climate Zones" (2021)
  - Arroyo & Grela, "Improving Remote Sensing Classification using TDA and
    CNNs" (arXiv:2507.10381, 2025)
"""

import os
import sys
import logging
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


def compute_cubical_tda_features_single_channel(
    images_2d: np.ndarray,
    sigma: float = 0.1,
    n_bins: int = 10,
    homology_dims: tuple = (0, 1),
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute cubical persistence image features for a single-channel image batch.

    Args:
        images_2d: shape (N, H, W) — single-channel greyscale images.
        sigma: Gaussian kernel bandwidth for persistence images.
        n_bins: Resolution of persistence image (n_bins × n_bins per dimension).
        homology_dims: Homology dimensions to compute (0=components, 1=loops).
        n_jobs: Parallel jobs.

    Returns:
        np.ndarray: shape (N, len(homology_dims) * n_bins * n_bins)
    """
    from gtda.homology import CubicalPersistence
    from gtda.diagrams import PersistenceImage, Scaler

    # CubicalPersistence expects (N, H, W) for 2D images
    cubical = CubicalPersistence(
        homology_dimensions=homology_dims,
        n_jobs=n_jobs,
        reduced_homology=True,
    )
    diagrams = cubical.fit_transform(images_2d)

    # Scale diagrams to [0,1] before computing persistence images
    # This is the standard step in giotto-tda image pipelines
    scaler = Scaler()
    diagrams_scaled = scaler.fit_transform(diagrams)

    # Vectorize via persistence images
    pi = PersistenceImage(
        sigma=sigma,
        n_bins=n_bins,
        n_jobs=n_jobs,
    )
    features = pi.fit_transform(diagrams_scaled)

    # features shape: (N, n_dims, n_bins, n_bins) → flatten to (N, n_dims*n_bins*n_bins)
    return features.reshape(len(images_2d), -1)


def compute_sar_tda_features_batch(
    sar_images: np.ndarray,
    sigma: float = 0.1,
    n_bins: int = 10,
    homology_dims: tuple = (0, 1),
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute cubical TDA features for multi-channel SAR images.

    Processes each channel independently and concatenates features.
    This avoids scale mixing between SAR polarimetric quantities.

    Args:
        sar_images: shape (N, H, W, C) — raw SAR patches, channel-last.
        sigma: Persistence image bandwidth.
        n_bins: Persistence image resolution.
        homology_dims: H0 and H1 by default.
        n_jobs: Parallelism for each channel.

    Returns:
        np.ndarray: shape (N, C * n_dims * n_bins * n_bins)
    """
    N, H, W, C = sar_images.shape
    n_features_per_channel = len(homology_dims) * n_bins * n_bins
    all_features = np.zeros((N, C * n_features_per_channel), dtype=np.float32)

    logger.info(
        f"Computing cubical TDA: {N} images, {H}×{W}, {C} channels, "
        f"H{list(homology_dims)}, sigma={sigma}, n_bins={n_bins}"
    )
    logger.info(
        f"  Expected feature dim: {C} × {len(homology_dims)} × {n_bins}² = "
        f"{C * n_features_per_channel}"
    )

    for ch in range(C):
        logger.info(f"  Processing channel {ch+1}/{C}...")
        images_ch = sar_images[:, :, :, ch]  # (N, H, W)

        # Normalise channel to [0, 1] for consistent filtration scale
        ch_min = images_ch.min()
        ch_max = images_ch.max()
        if ch_max - ch_min > 1e-10:
            images_ch = (images_ch - ch_min) / (ch_max - ch_min)
        else:
            images_ch = np.zeros_like(images_ch)

        ch_features = compute_cubical_tda_features_single_channel(
            images_ch, sigma=sigma, n_bins=n_bins,
            homology_dims=homology_dims, n_jobs=n_jobs,
        )  # (N, n_features_per_channel)

        start = ch * n_features_per_channel
        end = start + n_features_per_channel
        all_features[:, start:end] = ch_features

        logger.info(f"    Channel {ch+1} done: {ch_features.shape}")

    logger.info(f"  All channels done. Total shape: {all_features.shape}")
    return all_features


def compute_and_save_topological_features(
    sar_train: np.ndarray,
    sar_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    save_dir: str = None,
    n_pca: int = None,
    sigma: float = 0.1,
    n_bins: int = 10,
    pca_fit_samples: int = None,
) -> Dict[str, np.ndarray]:
    """
    Full topological feature pipeline for SAR images:
        SAR patches → cubical persistence per channel → persistence images
        → concatenate channels → PCA (fit on training only) → [0, π]

    Args:
        sar_train: shape (N_train, 32, 32, 8) — raw SAR training patches.
        sar_test: shape (N_test, 32, 32, 8) — raw SAR test patches.
        y_train: Training labels, shape (N_train,).
        y_test: Test labels, shape (N_test,).
        save_dir: Directory to save output .npz file.
        n_pca: Number of PCA components (default: config.N_PCA_COMPONENTS).
        sigma: Persistence image Gaussian bandwidth.
        n_bins: Persistence image grid resolution per dimension.
        pca_fit_samples: Number of training samples used to fit PCA.
                         Default: all N_train. For large datasets, subsample.

    Returns:
        dict with keys:
            'sar_tda_train': (N_train, n_pca) normalised TDA features
            'sar_tda_test':  (N_test, n_pca) normalised TDA features
            'y_train': labels
            'y_test': labels
    """
    if save_dir is None:
        save_dir = config.TOPO_DIR
    if n_pca is None:
        n_pca = config.N_PCA_COMPONENTS

    os.makedirs(save_dir, exist_ok=True)
    output_file = os.path.join(save_dir, "topological_features.npz")

    if os.path.exists(output_file):
        logger.info(f"[SKIP] Topological features already exist: {output_file}")
        data = np.load(output_file)
        return {k: data[k] for k in data.files}

    logger.info("=== Computing SAR cubical TDA features ===")

    # Step 1: Compute raw TDA features for train and test
    logger.info("Computing training TDA features...")
    feat_train_raw = compute_sar_tda_features_batch(
        sar_train, sigma=sigma, n_bins=n_bins,
    )

    logger.info("Computing test TDA features...")
    feat_test_raw = compute_sar_tda_features_batch(
        sar_test, sigma=sigma, n_bins=n_bins,
    )

    # Step 2: Fit PCA on training features ONLY (prevent test leakage)
    n_fit = pca_fit_samples if pca_fit_samples else len(feat_train_raw)
    n_fit = min(n_fit, len(feat_train_raw))
    logger.info(
        f"Fitting PCA on {n_fit} training samples "
        f"({feat_train_raw.shape[1]}d → {n_pca}d)..."
    )
    pca = PCA(n_components=n_pca, random_state=config.RANDOM_SEED)
    pca.fit(feat_train_raw[:n_fit])
    logger.info(
        f"  Explained variance ratio (cumulative): "
        f"{pca.explained_variance_ratio_.cumsum()[-1]:.4f}"
    )

    feat_train_pca = pca.transform(feat_train_raw)  # (N_train, n_pca)
    feat_test_pca = pca.transform(feat_test_raw)    # (N_test, n_pca)

    # Step 3: Normalise to [0, π] using training statistics only
    x_min = feat_train_pca.min(axis=0)
    x_max = feat_train_pca.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range < 1e-10] = 1.0  # avoid division by zero

    feat_train_norm = (feat_train_pca - x_min) / x_range * np.pi
    feat_test_norm = (feat_test_pca - x_min) / x_range * np.pi

    # Clip test features to [0, π] (test values may exceed training range)
    feat_train_norm = np.clip(feat_train_norm, 0.0, np.pi)
    feat_test_norm = np.clip(feat_test_norm, 0.0, np.pi)

    logger.info(
        f"  Train TDA features: {feat_train_norm.shape}, "
        f"range [{feat_train_norm.min():.3f}, {feat_train_norm.max():.3f}]"
    )
    logger.info(
        f"  Test TDA features:  {feat_test_norm.shape}, "
        f"range [{feat_test_norm.min():.3f}, {feat_test_norm.max():.3f}]"
    )

    result = {
        "sar_tda_train": feat_train_norm.astype(np.float32),
        "sar_tda_test": feat_test_norm.astype(np.float32),
        "y_train": y_train,
        "y_test": y_test,
        # Save PCA parameters for reproducibility
        "pca_components": pca.components_,
        "pca_explained_variance": pca.explained_variance_ratio_,
        "norm_min": x_min,
        "norm_range": x_range,
        "tda_params": np.array([sigma, n_bins, n_pca]),
    }

    np.savez_compressed(output_file, **result)
    logger.info(f"Saved topological features: {output_file}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("  src/topological.py — Self-test (cubical persistence)")
    print("=" * 60)
    import numpy as np
    np.random.seed(config.RANDOM_SEED)

    # Test: 5 synthetic SAR patches
    sar_batch = np.random.rand(5, 32, 32, 8).astype(np.float32)
    try:
        features = compute_sar_tda_features_batch(
            sar_batch, sigma=0.1, n_bins=5,
        )
        print(f"  Feature shape: {features.shape}")
        # Expected: (5, 8 channels × 2 dims × 5² bins) = (5, 400)
        assert features.shape == (5, 8 * 2 * 25), \
            f"Unexpected shape: {features.shape}"
        print("  Cubical TDA self-test: PASS")
    except ImportError:
        print("  giotto-tda not installed. Install: pip install giotto-tda")
    print("\n  Self-test complete.")
