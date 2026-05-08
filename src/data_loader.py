"""
Data loader for the Quantum-Sat Classification pipeline.

Provides functions to load processed .npz data files, create stratified
and balanced few-shot subsets, and handle class imbalance gracefully.

Key functions:
    - load_modality(): Load SAR/Optical/Fused/EuroSAT processed features.
    - get_fewshot_subset(): Stratified or balanced subsampling (Rule I4).
    - get_balanced_subset(): Equal samples per class.

All data files are expected in config.PROCESSED_DIR as .npz archives.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)

# Valid modality names
VALID_MODALITIES = ["sar", "optical", "fused"]
VALID_DATASETS = ["so2sat", "eurosat"]


def load_modality(
    modality: str = "fused",
    dataset: str = "so2sat",
    data_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Load processed PCA features for a given modality and dataset.

    Args:
        modality: One of 'sar', 'optical', 'fused'. Default: 'fused'.
        dataset: One of 'so2sat', 'eurosat'. Default: 'so2sat'.
        data_dir: Override data directory. Default: config.PROCESSED_DIR.

    Returns:
        dict: Keys 'X_train', 'y_train', 'X_val' (if available),
              'y_val' (if available), 'X_test', 'y_test'.

    Raises:
        ValueError: If modality or dataset name is invalid.
        FileNotFoundError: If processed data file does not exist.
    """
    modality = modality.lower()
    dataset = dataset.lower()

    if modality not in VALID_MODALITIES:
        raise ValueError(
            f"Invalid modality '{modality}'. Choose from: {VALID_MODALITIES}"
        )
    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"Invalid dataset '{dataset}'. Choose from: {VALID_DATASETS}"
        )

    if data_dir is None:
        data_dir = config.PROCESSED_DIR

    if dataset == "eurosat":
        filename = "eurosat_pca8.npz"
    else:
        filename = f"{modality}_pca8.npz"

    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Processed data not found: {filepath}\n"
            f"Run 'python setup_data.py' first to generate processed features."
        )

    data = np.load(filepath, allow_pickle=True)
    result = {
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
    }

    # Include validation set if available
    if "X_val" in data:
        result["X_val"] = data["X_val"]
        result["y_val"] = data["y_val"]

    logger.info(
        f"Loaded {dataset}/{modality}: "
        f"train={result['X_train'].shape}, test={result['X_test'].shape}, "
        f"n_classes={len(np.unique(result['y_train']))}"
    )

    return result


def load_subsample(
    data_dir: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Load the stratified 2000/500 subsample.

    Args:
        data_dir: Override data directory.

    Returns:
        dict: Keys with modality prefix, e.g. 'sar_X_train', 'fused_X_test',
              'y_train', 'y_test'.
    """
    if data_dir is None:
        data_dir = config.PROCESSED_DIR

    filepath = os.path.join(data_dir, "subsample_2000.npz")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Subsample not found: {filepath}\n"
            f"Run 'python setup_data.py' first."
        )

    data = np.load(filepath, allow_pickle=True)
    result = {}
    for key in data.files:
        result[key] = data[key]

    logger.info(
        f"Loaded subsample: {list(result.keys())}"
    )

    return result


def get_fewshot_subset(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    seed: int,
    strategy: str = "stratified",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a few-shot subset with stratified or balanced sampling (Rule I4).

    Strategies:
        - 'stratified': Proportional to class distribution (preserves imbalance).
          Uses sklearn's train_test_split with stratify=y.
        - 'balanced': Equal samples per class (n_per_class = n_samples // n_classes).
          Useful for comparing against stratified in ablation studies.

    At N=50 with stratified, some rare classes (e.g., Heavy Industry at 0.3%)
    may get 0 samples. This is handled gracefully.

    Args:
        X: Feature matrix, shape (n, d).
        y: Labels, shape (n,).
        n_samples: Total number of samples in subset.
        seed: Random seed for reproducibility.
        strategy: 'stratified' (default) or 'balanced'.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (X_subset, y_subset)

    Raises:
        ValueError: If strategy is invalid or n_samples > len(X).
    """
    if n_samples >= len(X):
        logger.warning(
            f"Requested {n_samples} samples but only {len(X)} available. "
            f"Returning all data."
        )
        return X.copy(), y.copy()

    if strategy == "stratified":
        return _stratified_subset(X, y, n_samples, seed)
    elif strategy == "balanced":
        return _balanced_subset(X, y, n_samples, seed)
    else:
        raise ValueError(
            f"Invalid strategy '{strategy}'. Choose 'stratified' or 'balanced'."
        )


def _stratified_subset(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified subsampling proportional to class distribution (Rule I1).

    Uses sklearn's train_test_split with stratify=y. Handles edge cases
    where rare classes have too few samples by ensuring at least 1 sample
    per class when possible.

    Args:
        X: Feature matrix.
        y: Labels.
        n_samples: Target subset size.
        seed: Random seed.

    Returns:
        Tuple of (X_subset, y_subset).
    """
    n_total = len(X)
    frac = n_samples / n_total

    # Check if any class would get 0 samples
    classes, counts = np.unique(y, return_counts=True)
    expected_per_class = (counts * frac).astype(int)
    zero_classes = classes[expected_per_class == 0]

    if len(zero_classes) > 0:
        logger.warning(
            f"Stratified sampling at N={n_samples}: classes {zero_classes.tolist()} "
            f"would get 0 samples. Using modified stratified approach."
        )
        # Ensure at least 1 sample per class by pre-selecting one sample
        # from each class, then stratified-sample the remainder
        rng = np.random.RandomState(seed)
        pre_indices = []
        remaining_indices = list(range(n_total))

        for c in classes:
            c_indices = np.where(y == c)[0]
            chosen = rng.choice(c_indices, size=1)[0]
            pre_indices.append(chosen)
            remaining_indices.remove(chosen)

        remaining_indices = np.array(remaining_indices)
        n_remaining = n_samples - len(pre_indices)

        if n_remaining > 0 and n_remaining < len(remaining_indices):
            y_remaining = y[remaining_indices]
            try:
                _, idx_extra, _, _ = train_test_split(
                    remaining_indices,
                    y_remaining,
                    test_size=n_remaining,
                    stratify=y_remaining,
                    random_state=seed,
                )
            except ValueError:
                # If stratify fails, fall back to random sampling
                idx_extra = rng.choice(remaining_indices, n_remaining, replace=False)
            all_indices = np.concatenate([pre_indices, idx_extra])
        else:
            all_indices = np.array(pre_indices[:n_samples])

        return X[all_indices], y[all_indices]

    # Standard stratified split
    try:
        _, X_sub, _, y_sub = train_test_split(
            X, y,
            test_size=n_samples,
            stratify=y,
            random_state=seed,
        )
    except ValueError:
        # Fallback: random sampling if stratify fails
        logger.warning("Stratified split failed, falling back to random sampling")
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(X), n_samples, replace=False)
        X_sub, y_sub = X[indices], y[indices]

    # Verify
    _log_class_distribution(y_sub, f"Few-shot N={n_samples}")

    return X_sub, y_sub


def _balanced_subset(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balanced subsampling: equal samples per class (Rule I4).

    n_per_class = n_samples // n_classes. Classes with fewer samples than
    n_per_class are sampled with replacement.

    Args:
        X: Feature matrix.
        y: Labels.
        n_samples: Target total subset size.
        seed: Random seed.

    Returns:
        Tuple of (X_subset, y_subset).
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    n_per_class = n_samples // len(classes)

    if n_per_class < 1:
        logger.warning(
            f"Balanced sampling: n_per_class={n_per_class} < 1. "
            f"Setting to 1 (total will be {len(classes)})."
        )
        n_per_class = 1

    indices = []
    for c in classes:
        c_indices = np.where(y == c)[0]
        if len(c_indices) >= n_per_class:
            chosen = rng.choice(c_indices, n_per_class, replace=False)
        else:
            # Use replacement for classes with fewer samples
            chosen = rng.choice(c_indices, n_per_class, replace=True)
        indices.extend(chosen)

    indices = np.array(indices)
    rng.shuffle(indices)

    _log_class_distribution(y[indices], f"Balanced N={len(indices)}")

    return X[indices], y[indices]


def get_balanced_subset(
    X: np.ndarray,
    y: np.ndarray,
    n_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a balanced subset with exactly n_per_class samples per class.

    Convenience wrapper for _balanced_subset.

    Args:
        X: Feature matrix.
        y: Labels.
        n_per_class: Samples per class.
        seed: Random seed.

    Returns:
        Tuple of (X_subset, y_subset).
    """
    n_classes = len(np.unique(y))
    return _balanced_subset(X, y, n_per_class * n_classes, seed)


def _log_class_distribution(y: np.ndarray, label: str = ""):
    """
    Log per-class sample counts.

    Args:
        y: Label array.
        label: Description string for log message.
    """
    classes, counts = np.unique(y, return_counts=True)
    msg = f"[{label}] Class distribution ({len(classes)} classes): "
    parts = []
    for c, n in zip(classes, counts):
        name = config.LCZ_CLASS_NAMES[c] if c < len(config.LCZ_CLASS_NAMES) else f"C{c}"
        parts.append(f"{name}={n}")
    msg += ", ".join(parts)
    logger.info(msg)

    # Warn if any class has 0 samples
    if 0 in counts:
        zero_classes = classes[counts == 0]
        logger.warning(f"[{label}] WARNING: Classes with 0 samples: {zero_classes.tolist()}")


def verify_data_integrity(
    X: np.ndarray,
    y: np.ndarray,
    expected_features: int = config.N_PCA_COMPONENTS,
    expected_label_range: Tuple[int, int] = (0, config.LCZ_N_CLASSES - 1),
    expected_feature_range: Tuple[float, float] = (0.0, np.pi),
    check_name: str = "Data",
) -> bool:
    """
    Verify data integrity: shapes, label range, feature range.

    Args:
        X: Feature matrix.
        y: Labels.
        expected_features: Expected number of features (columns).
        expected_label_range: (min_label, max_label) inclusive.
        expected_feature_range: (min_value, max_value) approximate.
        check_name: Name for log messages.

    Returns:
        bool: True if all checks pass.
    """
    ok = True
    tolerance = 0.1  # Allow small float errors

    # Shape check
    if X.ndim != 2 or X.shape[1] != expected_features:
        logger.error(
            f"[{check_name}] Shape error: expected (n, {expected_features}), got {X.shape}"
        )
        ok = False

    if len(X) != len(y):
        logger.error(
            f"[{check_name}] Length mismatch: X has {len(X)} rows, y has {len(y)} entries"
        )
        ok = False

    # Label range
    y_min, y_max = y.min(), y.max()
    if y_min < expected_label_range[0] or y_max > expected_label_range[1]:
        logger.error(
            f"[{check_name}] Label range error: [{y_min}, {y_max}], "
            f"expected [{expected_label_range[0]}, {expected_label_range[1]}]"
        )
        ok = False

    # Feature range
    x_min, x_max = X.min(), X.max()
    if x_min < expected_feature_range[0] - tolerance:
        logger.warning(
            f"[{check_name}] Feature min {x_min:.4f} < {expected_feature_range[0]}"
        )
    if x_max > expected_feature_range[1] + tolerance:
        logger.warning(
            f"[{check_name}] Feature max {x_max:.4f} > {expected_feature_range[1]:.4f}"
        )

    if ok:
        logger.info(f"[{check_name}] Integrity check PASSED")
    else:
        logger.error(f"[{check_name}] Integrity check FAILED")

    return ok


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  src/data_loader.py — Self-test")
    print("=" * 60)

    np.random.seed(config.RANDOM_SEED)

    # Create synthetic test data
    n_train = 400
    X = np.random.rand(n_train, 8) * np.pi
    y = np.random.randint(0, 17, n_train)

    # Test stratified
    X_s, y_s = get_fewshot_subset(X, y, 100, seed=42, strategy="stratified")
    print(f"  Stratified N=100: X={X_s.shape}, y classes={len(np.unique(y_s))}")
    assert len(X_s) == 100

    # Test balanced
    X_b, y_b = get_fewshot_subset(X, y, 170, seed=42, strategy="balanced")
    print(f"  Balanced N=170: X={X_b.shape}, y classes={len(np.unique(y_b))}")

    # Test data integrity
    verify_data_integrity(X, y, check_name="Synthetic")

    # Test edge case: very small N
    X_tiny, y_tiny = get_fewshot_subset(X, y, 17, seed=42, strategy="balanced")
    print(f"  Balanced N=17:  X={X_tiny.shape}")

    print()
    print("  All self-tests passed.")
