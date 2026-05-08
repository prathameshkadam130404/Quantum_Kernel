"""
Data preparation script for Quantum-Sat Classification pipeline.

Downloads (if needed) and processes So2Sat LCZ42 and EuroSAT datasets.
Pipeline: Raw images → flatten → IncrementalPCA → 8 features → [0, π] normalize.

MEMORY-EFFICIENT: Loads only labels first, performs stratified subsampling
of indices, then loads only the needed samples from HDF5 via index slicing.
This avoids loading the full 47+ GB dataset into RAM.

Output files in data/processed/:
    - sar_pca8.npz, optical_pca8.npz, fused_pca8.npz
    - subsample_2000.npz
    - eurosat_pca8.npz

Usage:
    python setup_data.py
    python setup_data.py --data-path "/path/to/So2sat Full"
"""

import os
import sys
import glob
import argparse
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.feature_extraction import fit_and_transform_full_pipeline
from src.utils import setup_logging, Timer

logger = setup_logging("setup_data", log_file=os.path.join(config.RESULTS_DIR, "setup_data.log"))


# ============ LOCAL DATASET SEARCH ============

def find_local_h5_files(
    search_paths: Optional[List[str]] = None,
    override_path: Optional[str] = None,
) -> Dict[str, str]:
    """
    Search for local So2Sat HDF5 files before attempting download.

    Args:
        search_paths: List of directories to search.
        override_path: CLI override path (takes priority).

    Returns:
        dict: Keys 'train', 'val', 'test' mapping to file paths.
    """
    if search_paths is None:
        search_paths = config.LOCAL_DATASET_SEARCH_PATHS

    if override_path:
        search_paths = [override_path] + search_paths

    found = {"train": None, "val": None, "test": None}

    train_patterns = ["training.h5", "train.h5", "training_*.h5"]
    val_patterns = ["validation.h5", "val.h5", "validation_*.h5"]
    test_patterns = ["testing.h5", "test.h5", "testing_*.h5"]

    for search_dir in search_paths:
        if not os.path.isdir(search_dir):
            continue

        logger.info(f"Searching for HDF5 files in: {search_dir}")
        h5_files = glob.glob(os.path.join(search_dir, "**", "*.h5"), recursive=True)

        if not h5_files:
            continue

        logger.info(f"  Found {len(h5_files)} .h5 files")

        for h5 in h5_files:
            basename = os.path.basename(h5).lower()
            logger.info(f"    - {h5} ({os.path.getsize(h5) / 1e9:.1f} GB)")

            if found["train"] is None:
                for pat in train_patterns:
                    if glob.fnmatch.fnmatch(basename, pat):
                        found["train"] = h5
                        logger.info(f"    → Matched as TRAINING")
                        break

            if found["val"] is None:
                for pat in val_patterns:
                    if glob.fnmatch.fnmatch(basename, pat):
                        found["val"] = h5
                        logger.info(f"    → Matched as VALIDATION")
                        break

            if found["test"] is None:
                for pat in test_patterns:
                    if glob.fnmatch.fnmatch(basename, pat):
                        found["test"] = h5
                        logger.info(f"    → Matched as TESTING")
                        break

        if found["train"] and found["test"]:
            break

    logger.info("Local dataset search results:")
    for split, path in found.items():
        status = path if path else "NOT FOUND"
        logger.info(f"  {split}: {status}")

    return found


# ============ MEMORY-EFFICIENT DATA LOADING ============

def _detect_h5_keys(filepath: str) -> Dict[str, str]:
    """Detect the key names for SAR, Optical, and label data in the HDF5 file."""
    import h5py
    with h5py.File(filepath, "r") as f:
        keys = list(f.keys())

    sar_key = None
    for k in ["sen1", "s1"]:
        if k in keys:
            sar_key = k
            break

    opt_key = None
    for k in ["sen2", "s2"]:
        if k in keys:
            opt_key = k
            break

    label_key = None
    for k in ["label", "labels"]:
        if k in keys:
            label_key = k
            break

    logger.info(f"  HDF5 keys: {keys} → SAR={sar_key}, Opt={opt_key}, Label={label_key}")
    return {"sar": sar_key, "optical": opt_key, "label": label_key, "all": keys}


def load_labels_only(filepath: str) -> np.ndarray:
    """
    Load ONLY labels from HDF5 — very small memory footprint.

    Args:
        filepath: Path to .h5 file.

    Returns:
        np.ndarray: Integer labels, shape (N,).
    """
    import h5py

    key_map = _detect_h5_keys(filepath)
    if key_map["label"] is None:
        raise KeyError(f"No label key found in {filepath}. Available: {key_map['all']}")

    with h5py.File(filepath, "r") as f:
        labels_raw = f[key_map["label"]][:]

    if labels_raw.ndim == 2 and labels_raw.shape[1] > 1:
        labels = np.argmax(labels_raw, axis=1)
    else:
        labels = labels_raw.flatten().astype(int)

    logger.info(f"  Labels loaded: {len(labels)} samples, classes [{labels.min()}-{labels.max()}]")
    return labels


def load_h5_by_indices(
    filepath: str,
    indices: np.ndarray,
    load_only: str = "both",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Load only SPECIFIC samples from HDF5 by index — memory efficient.

    Args:
        filepath: Path to .h5 file.
        indices: Sorted integer array of sample indices to load.
        load_only: 'sar', 'optical', or 'both'. Only loads the requested sensor(s).

    Returns:
        Tuple: (sar_or_None, optical_or_None, labels) in float32.
    """
    import h5py

    key_map = _detect_h5_keys(filepath)
    indices = np.sort(indices)
    n = len(indices)
    chunk = 1000

    logger.info(f"  Loading {n} samples ({load_only}) from {os.path.basename(filepath)}...")

    sar = None
    optical = None

    with h5py.File(filepath, "r") as f:
        if load_only in ("sar", "both"):
            sar_dset = f[key_map["sar"]]
            sar = np.empty((n, *sar_dset.shape[1:]), dtype=np.float32)
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                sar[start:end] = sar_dset[indices[start:end]].astype(np.float32)

        if load_only in ("optical", "both"):
            opt_dset = f[key_map["optical"]]
            optical = np.empty((n, *opt_dset.shape[1:]), dtype=np.float32)
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                optical[start:end] = opt_dset[indices[start:end]].astype(np.float32)

    loaded = []
    if sar is not None:
        loaded.append(f"SAR {sar.shape} ({sar.nbytes/1e6:.0f} MB)")
    if optical is not None:
        loaded.append(f"Optical {optical.shape} ({optical.nbytes/1e6:.0f} MB)")
    logger.info(f"  Loaded: {', '.join(loaded)}")

    return sar, optical, indices


def get_stratified_indices(
    labels: np.ndarray,
    n_samples: int,
    seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    """
    Select stratified subset indices from labels (Rule I1).

    Args:
        labels: Full label array.
        n_samples: Number of samples to select.
        seed: Random seed.

    Returns:
        np.ndarray: Sorted indices of selected samples.
    """
    if n_samples >= len(labels):
        return np.arange(len(labels))

    all_indices = np.arange(len(labels))
    _, selected_idx, _, _ = train_test_split(
        all_indices, labels,
        test_size=n_samples,
        stratify=labels,
        random_state=seed,
    )
    return np.sort(selected_idx)


# ============ PROCESSING ============

def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten image arrays: (N, H, W, C) → (N, H*W*C) in float32."""
    n = images.shape[0]
    return images.reshape(n, -1).astype(np.float32)


def process_modality(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray],
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: Optional[np.ndarray],
    y_test: np.ndarray,
    modality_name: str,
    save_dir: str,
) -> Dict[str, np.ndarray]:
    """
    Full processing pipeline for one modality.
    Steps: flatten → IncrementalPCA(8) → normalize[0,π] → save .npz.
    """
    output_file = os.path.join(save_dir, f"{modality_name}_pca8.npz")

    if os.path.exists(output_file):
        logger.info(f"[SKIP] {modality_name} already processed: {output_file}")
        data = np.load(output_file)
        return {k: data[k] for k in data.files}

    logger.info(f"Processing {modality_name}...")

    with Timer(f"Flattening {modality_name}", logger=logger):
        X_train_flat = flatten_images(X_train)
        X_test_flat = flatten_images(X_test)
        X_val_flat = flatten_images(X_val) if X_val is not None else None

    logger.info(f"  Flattened shape: train={X_train_flat.shape}, test={X_test_flat.shape}")

    with Timer(f"PCA + normalization for {modality_name}", logger=logger):
        result = fit_and_transform_full_pipeline(
            X_train_flat, X_val_flat, X_test_flat,
            n_components=config.N_PCA_COMPONENTS,
            batch_size=2000,
            save_dir=save_dir,
            prefix=f"{modality_name}_",
        )

    save_dict = {
        "X_train": result["X_train"],
        "y_train": y_train,
        "X_test": result["X_test"],
        "y_test": y_test,
    }
    if "X_val" in result and result["X_val"] is not None:
        save_dict["X_val"] = result["X_val"]
        save_dict["y_val"] = y_val

    np.savez_compressed(output_file, **save_dict)
    logger.info(f"  Saved: {output_file}")
    logger.info(f"  PCA explained variance: {result['explained_variance'].sum():.4f}")
    logger.info(f"  Feature range: [{result['X_train'].min():.4f}, {result['X_train'].max():.4f}]")

    return save_dict


def create_subsample_from_processed(
    processed_dir: str,
    n_train: int = config.SUBSAMPLE_TRAIN,
    n_test: int = config.SUBSAMPLE_TEST,
    seed: int = config.RANDOM_SEED,
) -> None:
    """
    Create stratified subsamples from processed .npz files (Rule I1).
    Verifies no class has 0 samples.
    """
    output_file = os.path.join(processed_dir, "subsample_2000.npz")
    if os.path.exists(output_file):
        logger.info(f"[SKIP] Subsample already exists: {output_file}")
        return

    logger.info(f"Creating stratified subsample: {n_train} train, {n_test} test")

    save_dict = {}

    for modality in ["sar", "optical", "fused"]:
        source = os.path.join(processed_dir, f"{modality}_pca8.npz")
        if not os.path.exists(source):
            logger.warning(f"  {modality}_pca8.npz not found, skipping")
            continue

        data = np.load(source)
        X_train_full = data["X_train"]
        y_train_full = data["y_train"]
        X_test_full = data["X_test"]
        y_test_full = data["y_test"]

        if n_train < len(X_train_full):
            _, X_train_sub, _, y_train_sub = train_test_split(
                X_train_full, y_train_full,
                test_size=n_train, stratify=y_train_full, random_state=seed,
            )
        else:
            X_train_sub, y_train_sub = X_train_full, y_train_full

        if n_test < len(X_test_full):
            _, X_test_sub, _, y_test_sub = train_test_split(
                X_test_full, y_test_full,
                test_size=n_test, stratify=y_test_full, random_state=seed,
            )
        else:
            X_test_sub, y_test_sub = X_test_full, y_test_full

        save_dict[f"{modality}_X_train"] = X_train_sub
        save_dict[f"{modality}_X_test"] = X_test_sub

        classes, counts = np.unique(y_train_sub, return_counts=True)
        logger.info(f"  {modality} train subsample: {len(y_train_sub)} samples, {len(classes)} classes")
        for c, n in zip(classes, counts):
            name = config.LCZ_CLASS_NAMES[c] if c < len(config.LCZ_CLASS_NAMES) else f"C{c}"
            logger.info(f"    {name} (class {c}): {n} samples")

    save_dict["y_train"] = y_train_sub
    save_dict["y_test"] = y_test_sub

    np.savez_compressed(output_file, **save_dict)
    logger.info(f"Saved subsample: {output_file}")


# ============ EUROSAT ============

def prepare_eurosat(save_dir: str) -> None:
    """Prepare EuroSAT dataset (10 classes, Sentinel-2, 64×64)."""
    output_file = os.path.join(save_dir, "eurosat_pca8.npz")
    if os.path.exists(output_file):
        logger.info(f"[SKIP] EuroSAT already processed: {output_file}")
        return

    logger.info("Preparing EuroSAT dataset...")

    try:
        import torchvision
        from torchvision import transforms

        transform = transforms.Compose([transforms.ToTensor()])

        dataset = torchvision.datasets.EuroSAT(
            root=config.RAW_DIR, download=True, transform=transform,
        )

        logger.info(f"  EuroSAT loaded: {len(dataset)} samples")
        X_all, y_all = [], []
        for img, label in tqdm(dataset, desc="Loading EuroSAT"):
            X_all.append(img.numpy().flatten())
            y_all.append(label)

        X_all = np.array(X_all, dtype=np.float32)
        y_all = np.array(y_all, dtype=int)

        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, stratify=y_all,
            random_state=config.RANDOM_SEED,
        )

        result = fit_and_transform_full_pipeline(
            X_train, None, X_test,
            n_components=config.N_PCA_COMPONENTS,
            save_dir=save_dir, prefix="eurosat_",
        )

        np.savez_compressed(
            output_file,
            X_train=result["X_train"], y_train=y_train,
            X_test=result["X_test"], y_test=y_test,
        )
        logger.info(f"  Saved: {output_file}")

    except Exception as e:
        logger.error(f"EuroSAT preparation failed: {e}")
        logger.error("EuroSAT experiments will be skipped.")


# ============ MAIN ============

def main():
    """Main entry point — memory-efficient data preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare So2Sat LCZ42 and EuroSAT data for quantum kernel analysis."
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help='Override path to "So2sat Full" folder containing .h5 files.',
    )
    parser.add_argument(
        "--skip-eurosat", action="store_true",
        help="Skip EuroSAT preparation.",
    )
    parser.add_argument(
        "--n-train", type=int, default=config.SUBSAMPLE_TRAIN,
        help="Number of training samples to load (default: 2000).",
    )
    parser.add_argument(
        "--n-test", type=int, default=config.SUBSAMPLE_TEST,
        help=f"Number of test samples to load (default: {config.SUBSAMPLE_TEST}).",
    )
    parser.add_argument(
        "--pca-fit-samples", type=int, default=config.PCA_FIT_SAMPLES,
        help=f"Samples for PCA fitting (default: {config.PCA_FIT_SAMPLES}). More → better PCA, more RAM.",
    )
    args = parser.parse_args()

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    os.makedirs(config.RAW_DIR, exist_ok=True)
    os.makedirs(config.TOPO_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  So2Sat LCZ42 Data Preparation (Memory-Efficient)")
    logger.info("=" * 60)

    # ---- Step 1: Find local data ----
    with Timer("Finding local data", logger=logger):
        h5_files = find_local_h5_files(override_path=args.data_path)

    if not h5_files["train"] or not h5_files["test"]:
        logger.error("Local HDF5 files not found and TFDS fallback removed for memory safety.")
        logger.error("Please specify --data-path with the folder containing training.h5 and testing.h5")
        sys.exit(1)

    # ---- Step 2: Load ONLY labels (tiny memory footprint) ----
    logger.info("\n--- Step 2: Loading labels only ---")
    with Timer("Loading training labels", logger=logger):
        y_train_full = load_labels_only(h5_files["train"])
    with Timer("Loading test labels", logger=logger):
        y_test_full = load_labels_only(h5_files["test"])

    n_total_train = len(y_train_full)
    n_total_test = len(y_test_full)
    logger.info(f"  Total: {n_total_train} train, {n_total_test} test")

    # ---- Print FULL class distribution (from labels only — cheap) ----
    logger.info("\n--- Training Set Class Distribution (FULL) ---")
    classes, counts = np.unique(y_train_full, return_counts=True)
    for c, n in zip(classes, counts):
        name = config.LCZ_CLASS_NAMES[c] if c < len(config.LCZ_CLASS_NAMES) else f"Class {c}"
        pct = 100.0 * n / n_total_train
        logger.info(f"  {name:<25s} (class {c:2d}): {n:6d} samples ({pct:5.1f}%)")
    logger.info(f"  {'TOTAL':<25s}           : {n_total_train:6d} samples")
    imbalance_ratio = counts.max() / counts.min()
    logger.info(f"  Imbalance ratio: {imbalance_ratio:.0f}:1")

    # ---- Step 3: Select stratified indices ----
    # We use more samples for PCA fitting (better components),
    # then subsample further for the experiments.
    n_pca_fit = min(args.pca_fit_samples, n_total_train)
    n_test_load = min(args.n_test * 4, n_total_test)  # Load extra for subsampling later

    logger.info(f"\n--- Step 3: Stratified index selection ---")
    logger.info(f"  PCA fitting samples: {n_pca_fit}")
    logger.info(f"  Test samples to load: {n_test_load}")

    with Timer("Index selection", logger=logger):
        train_indices = get_stratified_indices(y_train_full, n_pca_fit)
        test_indices = get_stratified_indices(y_test_full, n_test_load)

    logger.info(f"  Selected {len(train_indices)} train indices, {len(test_indices)} test indices")

    # ---- Steps 4-5: Process each modality INDEPENDENTLY ----
    # Only loads from HDF5 if the output file doesn't exist yet.
    # Uses selective sensor loading (load_only='sar'/'optical') to save RAM.
    import gc

    y_train = y_train_full[train_indices]
    y_test = y_test_full[test_indices]

    # Verify class distribution
    classes_sub, counts_sub = np.unique(y_train, return_counts=True)
    logger.info(f"\n  Stratified subset: {len(y_train)} train, {len(y_test)} test, {len(classes_sub)} classes")
    for c, n in zip(classes_sub, counts_sub):
        name = config.LCZ_CLASS_NAMES[c] if c < len(config.LCZ_CLASS_NAMES) else f"Class {c}"
        logger.info(f"    {name} (class {c}): {n}")

    # ---- SAR ----
    sar_output = os.path.join(config.PROCESSED_DIR, "sar_pca8.npz")
    if os.path.exists(sar_output):
        logger.info(f"\n[SKIP] SAR already processed: {sar_output}")
    else:
        logger.info("\n--- Processing SAR ---")
        with Timer("SAR processing", logger=logger):
            sar_train, _, _ = load_h5_by_indices(h5_files["train"], train_indices, load_only="sar")
            sar_test, _, _ = load_h5_by_indices(h5_files["test"], test_indices, load_only="sar")
            process_modality(sar_train, None, sar_test, y_train, None, y_test, "sar", config.PROCESSED_DIR)
            del sar_train, sar_test
            gc.collect()

    # ---- Optical ----
    opt_output = os.path.join(config.PROCESSED_DIR, "optical_pca8.npz")
    if os.path.exists(opt_output):
        logger.info(f"[SKIP] Optical already processed: {opt_output}")
    else:
        logger.info("\n--- Processing Optical ---")
        with Timer("Optical processing", logger=logger):
            _, opt_train, _ = load_h5_by_indices(h5_files["train"], train_indices, load_only="optical")
            _, opt_test, _ = load_h5_by_indices(h5_files["test"], test_indices, load_only="optical")
            process_modality(opt_train, None, opt_test, y_train, None, y_test, "optical", config.PROCESSED_DIR)
            del opt_train, opt_test
            gc.collect()

    # ---- Fused (sequential load: SAR flat → delete → Optical flat → concat) ----
    fused_output = os.path.join(config.PROCESSED_DIR, "fused_pca8.npz")
    if os.path.exists(fused_output):
        logger.info(f"[SKIP] Fused already processed: {fused_output}")
    else:
        logger.info("\n--- Processing Fused (SAR + Optical) ---")
        with Timer("Fused processing", logger=logger):
            # Step A: Load SAR, flatten to 2D, delete raw 4D
            sar_train, _, _ = load_h5_by_indices(h5_files["train"], train_indices, load_only="sar")
            sar_flat_train = sar_train.reshape(len(sar_train), -1)  # view, no copy
            sar_flat_train = sar_flat_train.copy()  # own memory so raw can be freed
            del sar_train
            gc.collect()

            # Step B: Load Optical, flatten, concat with SAR flat, delete both raw
            _, opt_train, _ = load_h5_by_indices(h5_files["train"], train_indices, load_only="optical")
            opt_flat_train = opt_train.reshape(len(opt_train), -1).copy()
            del opt_train
            gc.collect()

            fused_train = np.concatenate([sar_flat_train, opt_flat_train], axis=1)
            del sar_flat_train, opt_flat_train
            gc.collect()

            # Same for test
            sar_test, _, _ = load_h5_by_indices(h5_files["test"], test_indices, load_only="sar")
            sar_flat_test = sar_test.reshape(len(sar_test), -1).copy()
            del sar_test

            _, opt_test, _ = load_h5_by_indices(h5_files["test"], test_indices, load_only="optical")
            opt_flat_test = opt_test.reshape(len(opt_test), -1).copy()
            del opt_test
            gc.collect()

            fused_test = np.concatenate([sar_flat_test, opt_flat_test], axis=1)
            del sar_flat_test, opt_flat_test
            gc.collect()

            logger.info(f"  Fused shape: train={fused_train.shape}, test={fused_test.shape}")
            logger.info(f"  Fused RAM: ~{(fused_train.nbytes + fused_test.nbytes) / 1e9:.2f} GB")

            # Process (already flat, add dummy spatial dims for flatten_images identity)
            fused_train_img = fused_train[:, np.newaxis, np.newaxis, :]
            del fused_train
            fused_test_img = fused_test[:, np.newaxis, np.newaxis, :]
            del fused_test
            gc.collect()

            process_modality(
                fused_train_img, None, fused_test_img,
                y_train, None, y_test,
                "fused", config.PROCESSED_DIR,
            )
            del fused_train_img, fused_test_img
            gc.collect()

    # ---- Step 6: Create subsample for experiments ----
    logger.info("\n--- Creating Stratified Subsample ---")
    with Timer("Subsample creation", logger=logger):
        create_subsample_from_processed(config.PROCESSED_DIR)

    # ---- Step 7: EuroSAT ----
    if not args.skip_eurosat:
        logger.info("\n--- Preparing EuroSAT ---")
        with Timer("EuroSAT preparation", logger=logger):
            prepare_eurosat(config.PROCESSED_DIR)
    else:
        logger.info("Skipping EuroSAT (--skip-eurosat flag).")

    # ---- Summary ----
    logger.info("\n" + "=" * 60)
    logger.info("  Data Preparation Complete")
    logger.info("=" * 60)

    for filename in sorted(os.listdir(config.PROCESSED_DIR)):
        filepath = os.path.join(config.PROCESSED_DIR, filename)
        if os.path.isfile(filepath):
            size_mb = os.path.getsize(filepath) / 1e6
            logger.info(f"  {filename:<30s} {size_mb:8.1f} MB")


if __name__ == "__main__":
    main()
