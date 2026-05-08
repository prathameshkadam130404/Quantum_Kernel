#!/usr/bin/env python3
"""
EuroSAT 13-Band Data Loader & Physics Feature Extractor
========================================================

Loads EuroSATallBands GeoTIFFs (13 Sentinel-2 bands), computes per-patch
spatial-mean reflectance, and derives 8 optical physics indices matching
the So2Sat LCZ42 pipeline.

Band ordering (EuroSATallBands, 0-indexed):
  0: B1  (Coastal aerosol, 443 nm, 60 m)
  1: B2  (Blue,            490 nm, 10 m)
  2: B3  (Green,           560 nm, 10 m)
  3: B4  (Red,             665 nm, 10 m)
  4: B5  (Red Edge 1,      705 nm, 20 m)
  5: B6  (Red Edge 2,      740 nm, 20 m)
  6: B7  (Red Edge 3,      783 nm, 20 m)
  7: B8  (NIR,             842 nm, 10 m)
  8: B8A (Narrow NIR,      865 nm, 20 m)
  9: B9  (Water vapour,    945 nm, 60 m)
 10: B10 (SWIR Cirrus,    1375 nm, 60 m)
 11: B11 (SWIR 1,         1610 nm, 20 m)
 12: B12 (SWIR 2,         2190 nm, 20 m)

References:
  - Helber et al. (2019), "EuroSAT: A Novel Dataset and Deep Learning
    Benchmark for Land Use and Land Cover Classification", IEEE JSTARS.
  - Sentinel-2 User Handbook, ESA (2015).
"""

import os
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

EUROSAT_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

BAND_NAMES = [
    "B1_Coastal", "B2_Blue", "B3_Green", "B4_Red",
    "B5_RE1", "B6_RE2", "B7_RE3", "B8_NIR",
    "B8A_NarrowNIR", "B9_WaterVapour", "B10_Cirrus",
    "B11_SWIR1", "B12_SWIR2",
]

PHYSICS_FEATURE_NAMES = [
    "NDVI", "NDBI", "NDWI", "BSI", "SAVI", "NDRE", "MNDWI", "EVI",
]

# Sentinel-2 Level-2A surface reflectance quantification value
S2_SCALE_FACTOR = 10000.0


# ── Physics Index Computation ────────────────────────────────────────────────

def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Compute element-wise ratio with zero-division protection.

    Replaces denominators with absolute value < 1e-10 by 1e-10 to prevent
    division by zero while preserving sign information.
    """
    safe_denom = np.where(np.abs(denominator) < 1e-10, 1e-10, denominator)
    return numerator / safe_denom


def compute_physics_indices(band_means: np.ndarray) -> np.ndarray:
    """
    Compute 8 optical physics indices from 13-band Sentinel-2 spatial means.

    All indices are standard remote sensing vegetation/urban/water indices
    computed from Sentinel-2 surface reflectance bands. Raw uint16 digital
    numbers are divided by 10000 to obtain approximate [0, 1] reflectance.

    Indices:
      NDVI  = (B8 - B4) / (B8 + B4)              [vegetation vigor]
      NDBI  = (B11 - B8) / (B11 + B8)             [built-up area]
      NDWI  = (B3 - B8) / (B3 + B8)               [water content]
      BSI   = ((B11+B4)-(B8+B2)) / ((B11+B4)+(B8+B2))  [bare soil]
      SAVI  = 1.5·(B8-B4) / (B8+B4+0.5)           [soil-adjusted veg, L=0.5]
      NDRE  = (B8 - B5) / (B8 + B5)               [red-edge chlorophyll]
      MNDWI = (B3 - B11) / (B3 + B11)             [modified water]
      EVI   = 2.5·(B8-B4) / (B8+6·B4-7.5·B2+1)   [enhanced vegetation]

    Args:
        band_means: (N, 13) per-patch spatial-mean reflectance (raw uint16).

    Returns:
        (N, 8) float32 array of physics indices, clipped to [-2, 2].
    """
    # Convert digital numbers to approximate surface reflectance [0, 1]
    r = band_means.astype(np.float64) / S2_SCALE_FACTOR

    # Extract named bands for readability
    B2  = r[:, 1]   # Blue   (490 nm)
    B3  = r[:, 2]   # Green  (560 nm)
    B4  = r[:, 3]   # Red    (665 nm)
    B5  = r[:, 4]   # RE1    (705 nm)
    B8  = r[:, 7]   # NIR    (842 nm)
    B11 = r[:, 11]  # SWIR1  (1610 nm)

    ndvi  = _safe_ratio(B8 - B4, B8 + B4)
    ndbi  = _safe_ratio(B11 - B8, B11 + B8)
    ndwi  = _safe_ratio(B3 - B8, B3 + B8)
    bsi   = _safe_ratio((B11 + B4) - (B8 + B2), (B11 + B4) + (B8 + B2))
    savi  = _safe_ratio(1.5 * (B8 - B4), B8 + B4 + 0.5)
    ndre  = _safe_ratio(B8 - B5, B8 + B5)
    mndwi = _safe_ratio(B3 - B11, B3 + B11)
    evi   = _safe_ratio(2.5 * (B8 - B4), B8 + 6.0 * B4 - 7.5 * B2 + 1.0)

    physics = np.column_stack([ndvi, ndbi, ndwi, bsi, savi, ndre, mndwi, evi])

    # Clip physically implausible outliers and sanitize NaN/Inf
    physics = np.clip(physics, -2.0, 2.0)
    physics = np.nan_to_num(physics, nan=0.0, posinf=2.0, neginf=-2.0)

    return physics.astype(np.float32)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_eurosat_allbands(
    data_dir: str = "data/raw/eurosat/EuroSATallBands",
    max_per_class: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load EuroSATallBands 13-band GeoTIFFs and extract per-patch spatial means.

    Optionally caches the extracted features to an .npz file for fast reloading
    on subsequent runs.

    Args:
        data_dir:      Path to EuroSATallBands directory.
        max_per_class: If set, limit samples per class (for testing).
        cache_path:    If set, save/load extracted features to/from this path.

    Returns:
        band_means:  (N, 13) float32 spatial-mean reflectance per patch
        labels:      (N,)    int64 class labels
        class_names: list of class name strings
    """
    # Check cache first
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached features from {cache_path}")
        d = np.load(cache_path)
        return d["band_means"], d["labels"], list(EUROSAT_CLASSES)

    # Import image reader
    try:
        import rasterio
        reader = "rasterio"
    except ImportError:
        try:
            import tifffile
            reader = "tifffile"
        except ImportError:
            raise ImportError(
                "Either rasterio or tifffile is required to read GeoTIFFs. "
                "Install with: pip install rasterio  OR  pip install tifffile"
            )

    # Import tqdm for progress bars
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback: no-op wrapper
        def tqdm(iterable, **kwargs):
            return iterable

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"EuroSATallBands directory not found at: {data_path}\n"
            f"Download from: https://zenodo.org/records/7711810"
        )

    all_means = []
    all_labels = []

    # Count total files for progress bar
    total_files = 0
    class_files = {}
    for cls_idx, cls_name in enumerate(EUROSAT_CLASSES):
        cls_dir = data_path / cls_name
        if not cls_dir.exists():
            logger.warning(f"Class directory missing: {cls_dir}")
            class_files[cls_idx] = []
            continue
        files = sorted(cls_dir.glob("*.tif"))
        if max_per_class is not None:
            files = files[:max_per_class]
        class_files[cls_idx] = files
        total_files += len(files)

    logger.info(f"Loading {total_files} GeoTIFFs across {len(EUROSAT_CLASSES)} classes...")

    pbar = tqdm(total=total_files, desc="Loading EuroSAT", unit="img")

    for cls_idx, cls_name in enumerate(EUROSAT_CLASSES):
        files = class_files.get(cls_idx, [])
        if not files:
            continue

        for fpath in files:
            if reader == "rasterio":
                import rasterio as rio
                with rio.open(fpath) as ds:
                    img = ds.read()  # (13, 64, 64)
            else:
                import tifffile
                img = tifffile.imread(str(fpath))
                if img.ndim == 3 and img.shape[-1] == 13:
                    img = np.transpose(img, (2, 0, 1))

            # Spatial mean per band → (13,) feature vector
            band_mean = img.reshape(img.shape[0], -1).mean(axis=1)
            all_means.append(band_mean)
            all_labels.append(cls_idx)
            pbar.update(1)

    pbar.close()

    band_means = np.array(all_means, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    logger.info(f"Loaded {len(labels)} patches, {len(EUROSAT_CLASSES)} classes")
    logger.info(f"Class distribution: {dict(zip(EUROSAT_CLASSES, np.bincount(labels).tolist()))}")

    # Cache for fast reloading
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, band_means=band_means, labels=labels)
        logger.info(f"Cached features to {cache_path}")

    return band_means, labels, list(EUROSAT_CLASSES)


# ── Feature Preparation (train-only fitting) ────────────────────────────────

def prepare_split_features(
    band_means_train: np.ndarray,
    band_means_test: np.ndarray,
    pca_dim: int = 8,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Prepare PCA-8 and Physics-8 features with TRAIN-ONLY fitting.

    Critical for publication: all scalers and PCA are fit exclusively on the
    training split, then applied to the test split, to prevent data leakage.

    Args:
        band_means_train: (N_train, 13) raw band means for training set.
        band_means_test:  (N_test, 13) raw band means for test set.
        pca_dim:          Number of PCA components (default 8).
        seed:             Random seed for PCA.

    Returns:
        dict with 'pca8_train', 'pca8_test', 'physics8_train', 'physics8_test',
        and 'pca_explained_var'.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # ── PCA path: StandardScaler → PCA → MinMaxScaler[0, π] ──
    scaler_pca = StandardScaler()
    X_tr_scaled = scaler_pca.fit_transform(band_means_train)
    X_te_scaled = scaler_pca.transform(band_means_test)

    pca = PCA(n_components=pca_dim, random_state=seed)
    X_tr_pca = pca.fit_transform(X_tr_scaled)
    X_te_pca = pca.transform(X_te_scaled)
    explained = pca.explained_variance_ratio_.sum()

    mm_pca = MinMaxScaler(feature_range=(0, np.pi))
    X_tr_pca = mm_pca.fit_transform(X_tr_pca).astype(np.float32)
    X_te_pca = mm_pca.transform(X_te_pca).astype(np.float32)

    # ── Physics path: compute indices → StandardScaler → MinMaxScaler[0, π] ──
    phys_train = compute_physics_indices(band_means_train)
    phys_test = compute_physics_indices(band_means_test)

    scaler_phys = StandardScaler()
    phys_train = scaler_phys.fit_transform(phys_train)
    phys_test = scaler_phys.transform(phys_test)

    mm_phys = MinMaxScaler(feature_range=(0, np.pi))
    phys_train = mm_phys.fit_transform(phys_train).astype(np.float32)
    phys_test = mm_phys.transform(phys_test).astype(np.float32)

    return {
        "pca8_train": X_tr_pca,
        "pca8_test": X_te_pca,
        "physics8_train": phys_train,
        "physics8_test": phys_test,
        "pca_explained_var": float(explained),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    bm, y, cn = load_eurosat_allbands(
        cache_path="data/processed/eurosat_band_means.npz"
    )
    # Quick validation with a dummy split
    feats = prepare_split_features(bm[:100], bm[100:200])
    print(f"PCA-8 train: {feats['pca8_train'].shape}")
    print(f"Physics-8 train: {feats['physics8_train'].shape}")
    print(f"PCA explained var: {feats['pca_explained_var']:.3f}")
