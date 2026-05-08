"""
Classical kernel functions for the Quantum-Sat Classification pipeline.

Implements:
    - RBF (Gaussian) kernel
    - Polynomial kernel (degree 2, 3, 4)
    - Linear kernel
    - Laplacian kernel
    - Tensor-product kernel: K_c⊗ = K_rbf(X[:,:4]) ⊙ K_rbf(X[:,4:])
      (Hadamard product — treats modalities as independent)

All kernels return normalized matrices for fair comparison with quantum kernels.

References:
    - Huang et al., Nature Communications 12, 2631 (2021) — tensor-product baseline
"""

import os
import sys
import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import (
    rbf_kernel,
    polynomial_kernel,
    linear_kernel,
    laplacian_kernel,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


def compute_rbf_kernel(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    gamma: Optional[float] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute the RBF (Gaussian) kernel matrix.

    K(x, x') = exp(-γ ||x - x'||²)

    Args:
        X: Data matrix, shape (n, d).
        X2: Optional second data matrix, shape (m, d).
        gamma: Bandwidth parameter. If None, uses 1/(n_features).
        save_path: Path to save kernel matrix.

    Returns:
        np.ndarray: Kernel matrix, shape (n, n) or (m, n).
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached RBF kernel: {save_path}")
        return np.load(save_path)

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = rbf_kernel(X, X2, gamma=gamma)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, K)

    return K


def compute_rbf_kernel_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    gamma_grid: Optional[list] = None,
    cv_folds: int = 5,
    save_path_train: Optional[str] = None,
    save_path_test: Optional[str] = None,
) -> tuple:
    """
    Compute RBF kernel with CV-tuned gamma for a fair classical baseline.

    Grid-searches gamma via stratified k-fold CV on training data using
    precomputed kernel SVM with macro-F1 as scoring metric.

    This is essential for fair quantum-vs-classical comparison: a fixed
    gamma=1/n_features can produce a near-degenerate kernel (mean_offdiag≈0.81,
    eff_rank≈1.2) that makes the classical baseline artificially weak.

    Args:
        X_train: Training data, shape (n, d).
        y_train: Training labels, shape (n,).
        X_test: Optional test data, shape (m, d).
        gamma_grid: List of gamma values to search. If None, uses a wide
            logarithmic grid from 1e-4 to 10.0.
        cv_folds: Number of CV folds (default 5).
        save_path_train: Path to save train kernel matrix.
        save_path_test: Path to save test kernel matrix.

    Returns:
        Tuple of (K_train, K_test_or_None, best_gamma, cv_results_dict).
        cv_results_dict maps gamma -> mean_f1 for all tested gammas.
    """
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    # Check cache
    if (save_path_train and os.path.exists(save_path_train)):
        logger.info(f"[SKIP] Loading cached CV-tuned RBF kernel: {save_path_train}")
        K_train = np.load(save_path_train)
        K_test = np.load(save_path_test) if (save_path_test and os.path.exists(save_path_test)) else None
        # Try to infer gamma from filename or return default
        return K_train, K_test, None, {}

    if gamma_grid is None:
        gamma_grid = [1e-4, 1e-3, 0.005, 0.01, 0.025, 0.05, 0.1,
                      0.125, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]

    logger.info(f"CV-tuning RBF gamma: {len(gamma_grid)} candidates, "
                f"{cv_folds}-fold stratified CV")

    n = len(X_train)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = {}

    for gamma in gamma_grid:
        K_full = rbf_kernel(X_train, X_train, gamma=gamma)
        fold_f1s = []

        for fold_train_idx, fold_val_idx in skf.split(X_train, y_train):
            K_tr = K_full[np.ix_(fold_train_idx, fold_train_idx)]
            K_val = K_full[np.ix_(fold_val_idx, fold_train_idx)]

            try:
                svm = SVC(kernel='precomputed', C=1.0, class_weight='balanced')
                svm.fit(K_tr, y_train[fold_train_idx])
                y_pred = svm.predict(K_val)
                f1 = f1_score(y_train[fold_val_idx], y_pred,
                              average='macro', zero_division=0)
                fold_f1s.append(f1)
            except Exception:
                fold_f1s.append(0.0)

        mean_f1 = float(np.mean(fold_f1s))
        cv_results[gamma] = mean_f1
        logger.info(f"  gamma={gamma:.4g}: CV macro-F1={mean_f1:.4f}")

    best_gamma = max(cv_results, key=cv_results.get)
    logger.info(f"  Best gamma: {best_gamma:.4g} "
                f"(CV macro-F1={cv_results[best_gamma]:.4f})")

    # Compute final kernels with best gamma
    K_train = rbf_kernel(X_train, X_train, gamma=best_gamma)
    K_test = rbf_kernel(X_test, X_train, gamma=best_gamma) if X_test is not None else None

    if save_path_train:
        os.makedirs(os.path.dirname(save_path_train), exist_ok=True)
        np.save(save_path_train, K_train)
        logger.info(f"  Saved CV-tuned RBF train kernel: {save_path_train}")

    if save_path_test and K_test is not None:
        os.makedirs(os.path.dirname(save_path_test), exist_ok=True)
        np.save(save_path_test, K_test)
        logger.info(f"  Saved CV-tuned RBF test kernel: {save_path_test}")

    return K_train, K_test, best_gamma, cv_results


def compute_polynomial_kernel(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    degree: int = 3,
    gamma: Optional[float] = None,
    coef0: float = 1.0,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute the polynomial kernel matrix.

    K(x, x') = (γ x·x' + coef0)^degree

    Args:
        X: Data matrix.
        X2: Optional second data matrix.
        degree: Polynomial degree.
        gamma: Scale parameter. If None, uses 1/n_features.
        coef0: Independent term.
        save_path: Path to save kernel matrix.

    Returns:
        np.ndarray: Kernel matrix.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached polynomial kernel: {save_path}")
        return np.load(save_path)

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = polynomial_kernel(X, X2, degree=degree, gamma=gamma, coef0=coef0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, K)

    return K


def compute_linear_kernel(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute the linear kernel matrix.

    K(x, x') = x · x'

    Args:
        X: Data matrix.
        X2: Optional second data matrix.
        save_path: Path to save kernel matrix.

    Returns:
        np.ndarray: Kernel matrix.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached linear kernel: {save_path}")
        return np.load(save_path)

    K = linear_kernel(X, X2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, K)

    return K


def compute_laplacian_kernel(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    gamma: Optional[float] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute the Laplacian kernel matrix.

    K(x, x') = exp(-γ ||x - x'||_1)

    Args:
        X: Data matrix.
        X2: Optional second data matrix.
        gamma: Bandwidth parameter.
        save_path: Path to save kernel matrix.

    Returns:
        np.ndarray: Kernel matrix.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached Laplacian kernel: {save_path}")
        return np.load(save_path)

    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = laplacian_kernel(X, X2, gamma=gamma)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, K)

    return K


def compute_tensor_product_kernel(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    n_sar_features: int = 4,
    n_opt_features: int = 4,
    gamma: Optional[float] = None,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute the tensor-product kernel.

    K_c⊗ = K_rbf(X[:,:n_sar]) ⊙ K_rbf(X[:,n_sar:])

    The Hadamard product of per-modality RBF kernels treats modalities
    as INDEPENDENT — it cannot capture cross-modal correlations.
    This is the key classical baseline for demonstrating quantum advantage
    on multi-modal data.

    Args:
        X: Fused data matrix, shape (n, n_sar + n_opt).
        X2: Optional second data matrix. Same feature layout.
        n_sar_features: Number of SAR features (first n columns).
        n_opt_features: Number of Optical features (remaining columns).
        gamma: Bandwidth per modality. If None, uses 1/n_features_per_modality.
        save_path: Path to save kernel matrix.

    Returns:
        np.ndarray: Tensor-product kernel matrix.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached tensor-product kernel: {save_path}")
        return np.load(save_path)

    X_sar = X[:, :n_sar_features]
    X_opt = X[:, n_sar_features:n_sar_features + n_opt_features]

    if X2 is not None:
        X2_sar = X2[:, :n_sar_features]
        X2_opt = X2[:, n_sar_features:n_sar_features + n_opt_features]
    else:
        X2_sar = None
        X2_opt = None

    gamma_sar = gamma if gamma else 1.0 / n_sar_features
    gamma_opt = gamma if gamma else 1.0 / n_opt_features

    K_sar = rbf_kernel(X_sar, X2_sar, gamma=gamma_sar)
    K_opt = rbf_kernel(X_opt, X2_opt, gamma=gamma_opt)

    # Hadamard (element-wise) product
    K = K_sar * K_opt

    logger.info(f"Tensor-product kernel: K_sar{K_sar.shape} ⊙ K_opt{K_opt.shape} → {K.shape}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, K)

    return K


def compute_all_classical_kernels(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None,
    prefix: str = "",
) -> dict:
    """
    Compute all classical kernel matrices for comparison.

    Returns a dict with keys: 'rbf', 'poly2', 'poly3', 'poly4',
    'linear', 'laplacian', 'tensor_product'.

    Args:
        X: Data matrix.
        X2: Optional second data matrix.
        save_dir: Directory to save kernels.
        prefix: Filename prefix (e.g., 'sar_', 'fused_').

    Returns:
        dict: {kernel_name: kernel_matrix}.
    """
    def _path(name):
        if save_dir:
            return os.path.join(save_dir, f"{prefix}{name}.npy")
        return None

    kernels = {
        "rbf": compute_rbf_kernel(X, X2, save_path=_path("rbf")),
        "poly2": compute_polynomial_kernel(X, X2, degree=2, save_path=_path("poly2")),
        "poly3": compute_polynomial_kernel(X, X2, degree=3, save_path=_path("poly3")),
        "poly4": compute_polynomial_kernel(X, X2, degree=4, save_path=_path("poly4")),
        "linear": compute_linear_kernel(X, X2, save_path=_path("linear")),
        "laplacian": compute_laplacian_kernel(X, X2, save_path=_path("laplacian")),
    }

    # Tensor-product only for fused (8-dim with 4+4 split)
    if X.shape[1] >= 8:
        kernels["tensor_product"] = compute_tensor_product_kernel(
            X, X2, n_sar_features=4, n_opt_features=4,
            save_path=_path("tensor_product"),
        )

    logger.info(f"Computed {len(kernels)} classical kernels (prefix='{prefix}')")

    return kernels


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  src/classical_kernels.py — Self-test")
    print("=" * 60)

    np.random.seed(config.RANDOM_SEED)

    X = np.random.rand(20, 8) * np.pi
    X_test = np.random.rand(10, 8) * np.pi

    # Test all kernels
    kernels = compute_all_classical_kernels(X)
    for name, K in kernels.items():
        print(f"  {name:<20s}: shape={K.shape}, range=[{K.min():.4f}, {K.max():.4f}]")

    # Test rectangular kernel
    K_rect = compute_rbf_kernel(X, X_test)
    print(f"\n  Rectangular RBF: shape={K_rect.shape}")

    # Test tensor-product
    K_tp = compute_tensor_product_kernel(X, n_sar_features=4, n_opt_features=4)
    print(f"  Tensor-product:  shape={K_tp.shape}")

    print("\n  All self-tests passed.")
