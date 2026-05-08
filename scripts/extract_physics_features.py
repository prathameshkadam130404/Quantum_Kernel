"""
Physics-Informed Feature Extraction for Quantum Kernels.

Replaces PCA with domain-specific spectral indices designed for LCZ classification.
Each feature is physically meaningful, bounded, and interpretable — matching the
ZZFeatureMap's need for bounded inputs with meaningful cross-feature interactions.

So2Sat LCZ42 Band Structure:
  Sentinel-1 (8 channels, 32×32):
    0: VH_real, 1: VH_imag, 2: VV_real, 3: VV_imag
    4: VH_lee,  5: VV_lee,  6: PolSAR_cov_real, 7: PolSAR_cov_imag

  Sentinel-2 (10 channels, 32×32):
    0: B2 (Blue),  1: B3 (Green),  2: B4 (Red),  3: B5 (VegRedEdge1)
    4: B6 (VRE2),  5: B7 (VRE3),   6: B8 (NIR),  7: B8A (NarrowNIR)
    8: B11 (SWIR1), 9: B12 (SWIR2)

8 Physics-Informed Features:
  From Optical (Sentinel-2):
    1. NDVI = (NIR - Red) / (NIR + Red)        → Vegetation density
    2. NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)    → Built-up/impervious surfaces
    3. NDWI = (Green - NIR) / (Green + NIR)     → Water bodies
    4. BSI  = ((SWIR1+Red)-(NIR+Blue)) / ((SWIR1+Red)+(NIR+Blue))  → Bare soil
  From SAR (Sentinel-1):
    5. VV/VH ratio = VV_lee / (VH_lee + ε)     → Surface roughness/type
    6. SAR_total = VH_lee + VV_lee              → Total backscatter intensity
    7. Cross-pol ratio = VH_lee / (VV_lee + ε)  → Volume scattering (vegetation)
    8. PolSAR coherence = sqrt(cov_re² + cov_im²) / sqrt(VV_lee * VH_lee + ε)
       → Coherence between polarizations (urban structure indicator)

All features are computed as spatial MEANS over the 32×32 patch, then normalized to [0, π].

Output: data/processed/physics_features.npz
"""

import os, sys, gc, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config
from setup_data import find_local_h5_files, load_labels_only, load_h5_by_indices, get_stratified_indices

EPS = 1e-8  # Numerical stability


def safe_ratio(a, b):
    """Normalized difference: (a - b) / (a + b + ε)."""
    return (a - b) / (a + b + EPS)


def extract_optical_indices(optical_patches):
    """
    Compute physics-informed spectral indices from Sentinel-2 patches.

    Args:
        optical_patches: np.ndarray, shape (N, 32, 32, 10)

    Returns:
        np.ndarray, shape (N, 4) — [NDVI, NDBI, NDWI, BSI]
    """
    N = optical_patches.shape[0]

    # Spatial mean of each band (32×32 → scalar per band)
    # Shape: (N, 10)
    band_means = optical_patches.mean(axis=(1, 2))

    blue  = band_means[:, 0]   # B2
    green = band_means[:, 1]   # B3
    red   = band_means[:, 2]   # B4
    nir   = band_means[:, 6]   # B8
    swir1 = band_means[:, 8]   # B11

    # 1. NDVI: vegetation index [-1, 1]
    ndvi = safe_ratio(nir, red)

    # 2. NDBI: built-up index [-1, 1]
    ndbi = safe_ratio(swir1, nir)

    # 3. NDWI: water index [-1, 1]
    ndwi = safe_ratio(green, nir)

    # 4. BSI: bare soil index [-1, 1]
    bsi = safe_ratio(swir1 + red, nir + blue)

    return np.stack([ndvi, ndbi, ndwi, bsi], axis=1)


def extract_sar_indices(sar_patches):
    """
    Compute physics-informed SAR features from Sentinel-1 patches.

    Args:
        sar_patches: np.ndarray, shape (N, 32, 32, 8)

    Returns:
        np.ndarray, shape (N, 4) — [VV/VH ratio, total_backscatter,
                                     cross_pol_ratio, PolSAR_coherence]
    """
    N = sar_patches.shape[0]

    # Spatial mean of each channel
    band_means = sar_patches.mean(axis=(1, 2))

    vh_lee   = np.abs(band_means[:, 4]) + EPS   # Lee filtered VH intensity
    vv_lee   = np.abs(band_means[:, 5]) + EPS   # Lee filtered VV intensity
    cov_re   = band_means[:, 6]                  # PolSAR covariance real part
    cov_im   = band_means[:, 7]                  # PolSAR covariance imaginary part

    # 5. VV/VH ratio — surface roughness indicator
    #    High for smooth surfaces (water), low for rough (vegetation)
    vv_vh_ratio = vv_lee / (vh_lee + EPS)

    # 6. Total backscatter — overall radar return intensity
    total_backscatter = vh_lee + vv_lee

    # 7. Cross-polarization ratio — volume scattering indicator
    #    High for vegetation canopy, low for smooth surfaces
    cross_pol = vh_lee / (vv_lee + EPS)

    # 8. PolSAR coherence magnitude — urban structure indicator
    #    |cov| / sqrt(VV * VH) — high for ordered structures (buildings)
    cov_magnitude = np.sqrt(cov_re**2 + cov_im**2)
    coherence = cov_magnitude / (np.sqrt(vv_lee * vh_lee) + EPS)

    return np.stack([vv_vh_ratio, total_backscatter, cross_pol, coherence], axis=1)


def normalize_to_pi(features):
    """
    Normalize each feature to [0, π] using min-max scaling.
    This is the encoding range expected by PennyLane's ZZFeatureMap.

    Uses per-feature robust scaling (clip at 1st and 99th percentiles)
    to reduce sensitivity to outliers.
    """
    result = np.zeros_like(features)
    for i in range(features.shape[1]):
        col = features[:, i]
        lo = np.percentile(col, 1)
        hi = np.percentile(col, 99)
        if abs(hi - lo) < EPS:
            result[:, i] = np.pi / 2  # constant feature → middle of range
        else:
            scaled = np.clip((col - lo) / (hi - lo), 0.0, 1.0)
            result[:, i] = scaled * np.pi
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract physics-informed features from So2Sat LCZ42")
    parser.add_argument("--data-path", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("PHYSICS-INFORMED FEATURE EXTRACTION")
    print("Replacing PCA with domain-specific spectral indices")
    print("=" * 70)

    # ---- Find data ----
    h5_files = find_local_h5_files(override_path=args.data_path)
    if not h5_files["train"] or not h5_files["test"]:
        print("ERROR: HDF5 files not found. Use --data-path.")
        sys.exit(1)

    # ---- Load labels and get stratified indices ----
    print("\nLoading labels...")
    y_train_full = load_labels_only(h5_files["train"])
    y_test_full  = load_labels_only(h5_files["test"])

    n_train_load = min(config.PCA_FIT_SAMPLES, len(y_train_full))
    n_test_load  = min(config.SUBSAMPLE_TEST * 4, len(y_test_full))

    train_indices = get_stratified_indices(y_train_full, n_train_load)
    test_indices  = get_stratified_indices(y_test_full, n_test_load)

    y_train_loaded = y_train_full[train_indices]
    y_test_loaded  = y_test_full[test_indices]

    print(f"  Loading {len(train_indices)} train, {len(test_indices)} test samples")

    # ---- Load raw SAR + Optical patches ----
    print("\nLoading raw SAR patches...")
    sar_train, _, _ = load_h5_by_indices(h5_files["train"], train_indices, load_only="sar")
    sar_test, _, _  = load_h5_by_indices(h5_files["test"], test_indices, load_only="sar")
    print(f"  SAR shapes: train={sar_train.shape}, test={sar_test.shape}")

    print("\nLoading raw Optical patches...")
    _, opt_train, _ = load_h5_by_indices(h5_files["train"], train_indices, load_only="optical")
    _, opt_test, _  = load_h5_by_indices(h5_files["test"], test_indices, load_only="optical")
    print(f"  Optical shapes: train={opt_train.shape}, test={opt_test.shape}")

    # ---- Extract features ----
    print("\nExtracting physics-informed features...")

    print("  Optical indices (NDVI, NDBI, NDWI, BSI)...")
    opt_feat_train = extract_optical_indices(opt_train)
    opt_feat_test  = extract_optical_indices(opt_test)
    del opt_train, opt_test
    gc.collect()

    print("  SAR indices (VV/VH, backscatter, cross-pol, coherence)...")
    sar_feat_train = extract_sar_indices(sar_train)
    sar_feat_test  = extract_sar_indices(sar_test)
    del sar_train, sar_test
    gc.collect()

    # Combine: 4 optical + 4 SAR = 8 features
    features_train = np.concatenate([opt_feat_train, sar_feat_train], axis=1)
    features_test  = np.concatenate([opt_feat_test, sar_feat_test], axis=1)

    feature_names = [
        "NDVI", "NDBI", "NDWI", "BSI",
        "VV/VH_ratio", "SAR_total", "CrossPol", "PolCoherence"
    ]

    print(f"\n  Combined features: {features_train.shape[1]} dimensions")
    print(f"  Feature statistics (train):")
    for i, name in enumerate(feature_names):
        col = features_train[:, i]
        print(f"    {name:15s}: min={col.min():+8.4f}  max={col.max():+8.4f}  "
              f"mean={col.mean():+8.4f}  std={col.std():.4f}")

    # ---- Normalize to [0, π] ----
    print("\nNormalizing to [0, π] (robust min-max)...")

    # Fit normalization on train, apply same to test
    train_lo = np.percentile(features_train, 1, axis=0)
    train_hi = np.percentile(features_train, 99, axis=0)

    def apply_norm(features, lo, hi):
        result = np.zeros_like(features)
        for i in range(features.shape[1]):
            if abs(hi[i] - lo[i]) < EPS:
                result[:, i] = np.pi / 2
            else:
                scaled = np.clip((features[:, i] - lo[i]) / (hi[i] - lo[i]), 0.0, 1.0)
                result[:, i] = scaled * np.pi
        return result

    features_train_norm = apply_norm(features_train, train_lo, train_hi)
    features_test_norm  = apply_norm(features_test, train_lo, train_hi)

    print(f"  Normalized range: [{features_train_norm.min():.4f}, {features_train_norm.max():.4f}]")

    # ---- Subsample to 2000/2000 ----
    print(f"\nCreating stratified subsample ({config.SUBSAMPLE_TRAIN} train, {config.SUBSAMPLE_TEST} test)...")

    if config.SUBSAMPLE_TRAIN < len(features_train_norm):
        _, sub_train, _, y_sub_train = train_test_split(
            features_train_norm, y_train_loaded,
            test_size=config.SUBSAMPLE_TRAIN, stratify=y_train_loaded,
            random_state=config.RANDOM_SEED,
        )
    else:
        sub_train, y_sub_train = features_train_norm, y_train_loaded

    if config.SUBSAMPLE_TEST < len(features_test_norm):
        _, sub_test, _, y_sub_test = train_test_split(
            features_test_norm, y_test_loaded,
            test_size=config.SUBSAMPLE_TEST, stratify=y_test_loaded,
            random_state=config.RANDOM_SEED,
        )
    else:
        sub_test, y_sub_test = features_test_norm, y_test_loaded

    # Also save the raw (un-normalized) features for the subsample
    # for classical methods that don't need [0, π] encoding
    if config.SUBSAMPLE_TRAIN < len(features_train):
        _, sub_raw_train, _, _ = train_test_split(
            features_train, y_train_loaded,
            test_size=config.SUBSAMPLE_TRAIN, stratify=y_train_loaded,
            random_state=config.RANDOM_SEED,
        )
    else:
        sub_raw_train = features_train

    if config.SUBSAMPLE_TEST < len(features_test):
        _, sub_raw_test, _, _ = train_test_split(
            features_test, y_test_loaded,
            test_size=config.SUBSAMPLE_TEST, stratify=y_test_loaded,
            random_state=config.RANDOM_SEED,
        )
    else:
        sub_raw_test = features_test

    classes, counts = np.unique(y_sub_train, return_counts=True)
    print(f"  Train: {len(y_sub_train)}, Test: {len(y_sub_test)}, Classes: {len(classes)}")
    print(f"  Min class: {counts.min()} samples, Max class: {counts.max()} samples")

    # ---- Save ----
    output_file = os.path.join(config.PROCESSED_DIR, "physics_features.npz")
    np.savez_compressed(
        output_file,
        # Normalized [0, π] for quantum encoding
        X_train=sub_train,
        X_test=sub_test,
        # Raw (unnormalized) for classical methods
        X_train_raw=sub_raw_train,
        X_test_raw=sub_raw_test,
        # Labels
        y_train=y_sub_train,
        y_test=y_sub_test,
        # Metadata
        feature_names=np.array(feature_names),
        normalization_lo=train_lo,
        normalization_hi=train_hi,
    )
    print(f"\nSaved: {output_file}")
    print(f"  Size: {os.path.getsize(output_file) / 1e6:.2f} MB")

    # ---- Quick sanity: per-class feature means ----
    print(f"\n{'='*70}")
    print("PER-CLASS FEATURE MEANS (showing class discriminability)")
    print(f"{'='*70}")

    print(f"\n{'Class':25s}", end="")
    for name in feature_names:
        print(f"  {name[:7]:>7s}", end="")
    print()
    print("-" * (25 + 9 * len(feature_names)))

    for c in sorted(np.unique(y_sub_train)):
        mask = y_sub_train == c
        class_name = config.LCZ_CLASS_NAMES[c] if c < len(config.LCZ_CLASS_NAMES) else f"Class {c}"
        print(f"{class_name:25s}", end="")
        for i in range(len(feature_names)):
            val = sub_raw_train[mask, i].mean()
            print(f"  {val:>7.3f}", end="")
        print()


if __name__ == "__main__":
    main()
