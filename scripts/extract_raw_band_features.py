"""
Extract Spatial Means directly from raw Sentinel bands for FQK encoding.

This script replaces classical physics formulas (e.g., NDVI) with the
barest possible spatial averaging of 8 raw satellite bands. This is necessary
for the "Quantum-Heavy Pipeline", which aims for zero classical feature engineering.

Bands Selected (8):
  Optical (5): B2(Blue), B3(Green), B4(Red), B8(NIR), B11(SWIR1)
  SAR (3): VH_lee, VV_lee, PolSAR_Cov_Magnitude

Output: data/processed/raw_band_features_8.npz
"""

import os
import sys
import gc
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from setup_data import find_local_h5_files, load_labels_only, load_h5_by_indices, get_stratified_indices

EPS = 1e-8


def extract_raw_optical_means(optical_patches):
    """
    Extract spatial means for 5 raw optical bands.
    optical_patches: (N, 32, 32, 10)
    """
    # Mean across 32x32 spatial dimensions -> (N, 10)
    band_means = optical_patches.mean(axis=(1, 2))
    
    # 0: B2 (Blue), 1: B3 (Green), 2: B4 (Red), 6: B8 (NIR), 8: B11 (SWIR1)
    selected_means = band_means[:, [0, 1, 2, 6, 8]]
    return selected_means


def extract_raw_sar_means(sar_patches):
    """
    Extract spatial means for 3 fundamental SAR properties.
    sar_patches: (N, 32, 32, 8)
    """
    band_means = sar_patches.mean(axis=(1, 2))
    
    vh_lee = np.abs(band_means[:, 4])
    vv_lee = np.abs(band_means[:, 5])
    cov_re = band_means[:, 6]
    cov_im = band_means[:, 7]
    
    cov_mag = np.sqrt(cov_re**2 + cov_im**2)
    
    return np.stack([vh_lee, vv_lee, cov_mag], axis=1)


def normalize_to_pi(features):
    """Min-max scale to [0, pi] per feature robustly."""
    result = np.zeros_like(features)
    for i in range(features.shape[1]):
        col = features[:, i]
        lo = np.percentile(col, 1)
        hi = np.percentile(col, 99)
        if abs(hi - lo) < EPS:
            result[:, i] = np.pi / 2
        else:
            scaled = np.clip((col - lo) / (hi - lo), 0.0, 1.0)
            result[:, i] = scaled * np.pi
    return result


def main():
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    
    # 1. Locate H5 files
    h5_paths = find_local_h5_files(config.DATA_DIR)
    if not h5_paths['train']:
        print("ERROR: Training H5 file not found.")
        return
        
    print(f"Loading data from: {h5_paths['train']}")
    
    # 2. Get balanced subset indices (exact same 2000 as other experiments)
    print("Loading labels for stratification...")
    y_all = load_labels_only(h5_paths['train'])
    
    n_samples = getattr(config, 'N_TRAIN_SAMPLES', 2000)
    print(f"Selecting {n_samples} stratified samples...")
    indices = get_stratified_indices(y_all, n_samples=n_samples, seed=42)
    
    # 3. Load patches
    print("Loading raw patches into memory...")
    X_s1_raw, X_s2_raw, _ = load_h5_by_indices(h5_paths['train'], indices)
    y = y_all[indices]
    
    # 4. Compute bare spatial means
    print("Computing raw band spatial means...")
    opt_feats = extract_raw_optical_means(X_s2_raw)
    sar_feats = extract_raw_sar_means(X_s1_raw)
    
    # 5. Combine and normalize
    X_raw = np.hstack([opt_feats, sar_feats])
    print("Normalizing to [0, pi]...")
    X_norm = normalize_to_pi(X_raw)
    
    # Names for debugging/logging
    names = [
        "Opt_B2_Blue", "Opt_B3_Green", "Opt_B4_Red", "Opt_B8_NIR", "Opt_B11_SWIR1",
        "SAR_VH_Lee", "SAR_VV_Lee", "SAR_Cov_Magnitude"
    ]
    
    # Validation checks
    assert X_norm.shape[1] == 8, f"Expected 8 features, got {X_norm.shape[1]}"
    assert np.all((X_norm >= 0) & (X_norm <= np.pi)), "Normalization bounds failed"
    
    out_path = os.path.join(config.PROCESSED_DIR, "raw_band_features_8.npz")
    np.savez_compressed(
        out_path,
        X_norm=X_norm,
        X_raw=X_raw,
        y=y,
        indices=indices,
        feature_names=np.array(names)
    )
    
    print(f"\n=============================================")
    print(f" SUCCESS: Extracted Raw Band Features")
    print(f"=============================================")
    print(f" Output: {out_path}")
    print(f" Shape: {X_norm.shape}")
    print(f" Features: {names}")
    
    # Quick variance check to ensure features are non-trivial
    for i, name in enumerate(names):
        print(f"   {name:20s}: var={np.var(X_norm[:, i]):.3f}, mean={np.mean(X_norm[:, i]):.3f}")


if __name__ == "__main__":
    main()
