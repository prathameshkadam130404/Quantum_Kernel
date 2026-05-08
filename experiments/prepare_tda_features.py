"""
Standalone TDA preprocessing script for Exp4.

Loads the SAME 2000 training / 2000 test SAR patches used in all other
experiments (stratified subsample, seed=42), computes cubical TDA features,
and saves to data/topological/topological_features.npz.

Must be run BEFORE experiments/exp4_topological_boost.py.

Runtime estimate: 1–4 hours depending on hardware.

Usage:
    python experiments/prepare_tda_features.py
    python experiments/prepare_tda_features.py --n-bins 8  # faster, less detail
    python experiments/prepare_tda_features.py --force      # recompute if exists
"""

import os
import sys
import argparse
import logging

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import setup_logging, Timer

logger = setup_logging(
    "prepare_tda",
    log_file=os.path.join(config.RESULTS_DIR, "prepare_tda.log"),
)


def load_raw_sar_subsample(seed: int = config.RANDOM_SEED) -> dict:
    """
    Load the raw SAR patches for the same 2000-sample stratified subsample
    used in all other experiments.

    This replicates the stratified sampling from setup_data.py with the
    SAME seed to guarantee identical sample indices.

    Returns:
        dict with 'sar_train', 'sar_test', 'y_train', 'y_test'
    """
    import h5py
    import glob
    from sklearn.model_selection import train_test_split

    # Find HDF5 files
    train_h5, test_h5 = None, None
    for sp in config.LOCAL_DATASET_SEARCH_PATHS:
        if not os.path.isdir(sp):
            continue
        for f in glob.glob(os.path.join(sp, "**", "*.h5"), recursive=True):
            bn = os.path.basename(f).lower()
            if "train" in bn and train_h5 is None:
                train_h5 = f
            elif "test" in bn and test_h5 is None:
                test_h5 = f

    if train_h5 is None or test_h5 is None:
        raise FileNotFoundError(
            "Cannot find So2Sat HDF5 files. "
            "Check config.LOCAL_DATASET_SEARCH_PATHS."
        )

    logger.info(f"Train HDF5: {train_h5}")
    logger.info(f"Test  HDF5: {test_h5}")

    def load_stratified_sar(h5path, n_final, seed, n_intermediate=None, is_train=True):
        with h5py.File(h5path, 'r') as f:
            sar_key = 'sen1' if 'sen1' in f.keys() else 's1'
            label_key = 'label' if 'label' in f.keys() else 'labels'

            labels_raw = f[label_key][:]
            if labels_raw.ndim == 2 and labels_raw.shape[1] > 1:
                labels_full = np.argmax(labels_raw, axis=1)
            else:
                labels_full = labels_raw.flatten().astype(int)

            # Step 1: Replicate Step 3 from setup_data.py (Full -> Intermediate)
            # setup_data.py uses n_pca_fit=15000 for train, n_test_load=8000 for test
            all_idx_full = np.arange(len(labels_full))
            _, intermediate_idx, _, labels_intermediate = train_test_split(
                all_idx_full, labels_full,
                test_size=n_intermediate,
                stratify=labels_full,
                random_state=seed,
            )
            # setup_data.py does NOT sort them here, it passes them to process_modality
            # wait, it DOES sort them in get_stratified_indices!
            intermediate_idx = np.sort(intermediate_idx)
            labels_intermediate = labels_full[intermediate_idx]

            # Step 2: Replicate Step 6 from setup_data.py (Intermediate -> Final 2000)
            all_idx_intermediate = np.arange(len(labels_intermediate))
            _, final_sub_idx, _, _ = train_test_split(
                all_idx_intermediate, labels_intermediate,
                test_size=n_final,
                stratify=labels_intermediate,
                random_state=seed,
            )
            # The indices in final_sub_idx are indices INTO intermediate_idx
            final_global_idx = intermediate_idx[final_sub_idx]

            # h5py requires elements to be in increasing order.
            # We sort indices, load, and then permute back to the original unsorted order
            # to keep consistency with y labels.
            sort_idx = np.argsort(final_global_idx)
            sorted_global_idx = final_global_idx[sort_idx]
            inverse_idx = np.argsort(sort_idx)

            sar_data_sorted = f[sar_key][sorted_global_idx]  # shape (2000, 32, 32, 8)
            sar_data = sar_data_sorted[inverse_idx]
            
            y = labels_full[final_global_idx]

        logger.info(
            f"  Loaded {len(final_global_idx)} SAR samples from {os.path.basename(h5path)} "
            f"(via {n_intermediate} intermediate)"
        )
        return sar_data, y

    with Timer("Loading training SAR patches", logger=logger):
        sar_train, y_train = load_stratified_sar(
            train_h5, config.SUBSAMPLE_TRAIN, seed=seed,
            n_intermediate=config.PCA_FIT_SAMPLES, is_train=True
        )

    with Timer("Loading test SAR patches", logger=logger):
        # setup_data.py uses n_test_load = min(args.n_test * 4, n_total_test)
        # where n_test defaults to config.SUBSAMPLE_TEST (2000)
        sar_test, y_test = load_stratified_sar(
            test_h5, config.SUBSAMPLE_TEST, seed=seed,
            n_intermediate=8000, is_train=False
        )

    logger.info(f"SAR train: {sar_train.shape}, test: {sar_test.shape}")
    return {
        "sar_train": sar_train,
        "sar_test": sar_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def verify_label_consistency(y_tda: np.ndarray, split: str = "train") -> None:
    """
    Verify that TDA labels match the subsample labels from exp1/exp2.
    This ensures the same 2000 samples are being used.
    """
    subsample_file = os.path.join(
        config.PROCESSED_DIR, "subsample_2000.npz"
    )
    if not os.path.exists(subsample_file):
        logger.warning(
            f"Cannot verify label consistency: {subsample_file} not found."
        )
        return

    subsample = np.load(subsample_file)
    y_key = f"y_{split}"
    if y_key not in subsample:
        logger.warning(f"Key '{y_key}' not in subsample file.")
        return

    y_ref = subsample[y_key]
    if not np.array_equal(y_tda, y_ref):
        raise RuntimeError(
            f"LABEL MISMATCH: TDA {split} labels do not match subsample labels. "
            f"This means different samples are being used. "
            f"Check that both use seed={config.RANDOM_SEED} and "
            f"n_samples={config.SUBSAMPLE_TRAIN}."
        )
    logger.info(f"  Label consistency check: PASSED ({split}, {len(y_ref)} samples)")


def main():
    parser = argparse.ArgumentParser(
        description="Compute cubical TDA features for Exp4."
    )
    parser.add_argument(
        "--n-bins", type=int, default=10,
        help="Persistence image grid resolution (default: 10). "
             "Lower = faster, less detail. Min recommended: 5.",
    )
    parser.add_argument(
        "--sigma", type=float, default=0.1,
        help="Persistence image Gaussian bandwidth (default: 0.1).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force recompute even if output file exists.",
    )
    args = parser.parse_args()

    output_file = os.path.join(config.TOPO_DIR, "topological_features.npz")
    if os.path.exists(output_file) and not args.force:
        logger.info(f"[SKIP] Already exists: {output_file}")
        logger.info("Use --force to recompute.")
        return

    if os.path.exists(output_file) and args.force:
        logger.info(f"[FORCE] Removing existing file: {output_file}")
        os.remove(output_file)

    logger.info("=" * 65)
    logger.info("  Exp4 TDA Feature Preprocessing")
    logger.info("=" * 65)
    logger.info(f"  n_bins={args.n_bins}, sigma={args.sigma}")
    logger.info(
        f"  Feature dim before PCA: "
        f"8 channels × 2 dims × {args.n_bins}² = "
        f"{8 * 2 * args.n_bins**2}"
    )
    logger.info(f"  Feature dim after PCA: {config.N_PCA_COMPONENTS}")

    # Step 1: Load raw SAR patches
    with Timer("Loading raw SAR patches", logger=logger):
        data = load_raw_sar_subsample(seed=config.RANDOM_SEED)

    # Step 2: Verify label consistency with existing subsample
    logger.info("Verifying label consistency with subsample_2000.npz...")
    verify_label_consistency(data["y_train"], "train")
    verify_label_consistency(data["y_test"], "test")

    # Step 3: Compute and save TDA features
    from src.topological import compute_and_save_topological_features

    with Timer("Computing cubical TDA features", logger=logger):
        result = compute_and_save_topological_features(
            sar_train=data["sar_train"],
            sar_test=data["sar_test"],
            y_train=data["y_train"],
            y_test=data["y_test"],
            save_dir=config.TOPO_DIR,
            n_pca=config.N_PCA_COMPONENTS,
            sigma=args.sigma,
            n_bins=args.n_bins,
        )

    logger.info("")
    logger.info("=" * 65)
    logger.info("  TDA Feature Preprocessing Complete")
    logger.info("=" * 65)
    logger.info(f"  sar_tda_train: {result['sar_tda_train'].shape}")
    logger.info(f"  sar_tda_test:  {result['sar_tda_test'].shape}")
    logger.info(f"  Output: {output_file}")
    logger.info("")
    logger.info("Next step: python experiments/exp4_topological_boost.py")


if __name__ == "__main__":
    main()
