"""
D4-Augmented PCA for SAR Modality.

Fits a PCA model on the orbit-augmented SAR dataset to guarantee
D4 group-averaging invariance: theta(M_g @ x) = theta(x) for all g in D4.

Standard PCA fitted on upright images only does not produce a D4-invariant
subspace. The M_g matrices derived from such a PCA do not form a closed
group, causing group-averaging invariance to fail (empirically: ~0.045 error).

Fix: augment the training data with all 8 D4 transforms of each patch before
PCA fitting. The resulting PCA subspace is D4-invariant by construction,
and group-averaging invariance holds to machine precision.

Usage:
    from src.d4_augmented_pca import fit_d4_augmented_pca, verify_invariance
"""

import os
import sys
import logging
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import IncrementalPCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

D4_ELEMENTS = ['e', 'r', 'r2', 'r3', 's', 'sr', 'sr2', 'sr3']


class D4OrbitPCA:
    """
    PCA-like object using D4-invariant orbit-indicator components.
    Invariance guarantee: theta(g*x) = theta(x) to machine precision.
    Compatible with sklearn's precomputed kernel SVM interface.
    """
    def __init__(self, components, mean):
        self.components_ = components.astype(np.float64)
        self.mean_ = mean.astype(np.float64)
        self.n_components_ = components.shape[0]

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.transform(X)


def apply_d4_to_flat_patch(
    x_flat: np.ndarray,
    element: str,
    height: int = 32,
    width: int = 32,
    n_channels: int = 8,
) -> np.ndarray:
    """
    Apply a D4 transformation to a flattened image patch.

    The flat patch has layout: all pixels of channel 0, then channel 1, etc.
    i.e. x_flat[ch * H * W : (ch+1) * H * W] = channel ch pixels in row-major order.

    Args:
        x_flat: Flattened patch, shape (H*W*C,).
        element: D4 element name.
        height: Patch height.
        width: Patch width.
        n_channels: Number of channels.

    Returns:
        np.ndarray: Transformed flat patch, same shape.
    """
    from src.d4_representation import get_d4_pixel_permutation

    assert height == width, "D4 requires square patches"
    spatial_perm = get_d4_pixel_permutation(element, height, width)
    n_spatial = height * width

    result = np.empty_like(x_flat)
    for ch in range(n_channels):
        start = ch * n_spatial
        end = start + n_spatial
        channel_data = x_flat[start:end]
        result[start:end] = channel_data[spatial_perm]

    return result


def augment_with_d4_orbits(
    X_flat: np.ndarray,
    height: int = 32,
    width: int = 32,
    n_channels: int = 8,
) -> np.ndarray:
    """
    Augment flat image dataset with all 8 D4 transforms.

    For n input samples, returns 8n samples — one per D4 element per sample.
    The identity transform is included (first element 'e') so the original
    samples are preserved in the output.

    Args:
        X_flat: Flattened patches, shape (n, H*W*C).
        height: Patch height (32 for So2Sat).
        width: Patch width (32 for So2Sat).
        n_channels: Number of channels (8 for SAR).

    Returns:
        np.ndarray: Augmented dataset, shape (8*n, H*W*C).
    """
    n = len(X_flat)
    n_spatial = height * width
    assert X_flat.shape[1] == n_spatial * n_channels, (
        f"Expected X_flat.shape[1] = {n_spatial * n_channels}, got {X_flat.shape[1]}"
    )

    logger.info(f"D4 augmentation: {n} samples × 8 elements = {8*n} total")

    from src.d4_representation import get_d4_pixel_permutation

    # Precompute full permutations for all 8 elements
    perms = {}
    for elem in D4_ELEMENTS:
        spatial_perm = get_d4_pixel_permutation(elem, height, width)
        full_perm = np.concatenate([
            spatial_perm + ch * n_spatial for ch in range(n_channels)
        ])
        perms[elem] = full_perm

    # Apply all 8 transforms
    augmented = np.empty((8 * n, X_flat.shape[1]), dtype=X_flat.dtype)
    for k, elem in enumerate(D4_ELEMENTS):
        augmented[k*n:(k+1)*n] = X_flat[:, perms[elem]]

    return augmented


def _find_d4_orbits(height: int, width: int) -> list:
    """
    Find all D4 orbits on a height×width pixel grid.

    Each orbit is a list of pixel flat-indices that are connected
    by D4 transformations. The D4 group on a 32×32 grid produces
    136 orbits (120 of size 8, 16 of size 4).

    Returns:
        List of sorted orbit lists, e.g. [[0,3,12,15], [1,2,...], ...]
    """
    from src.d4_representation import get_d4_pixel_permutation

    assert height == width, "D4 requires square patches"
    n_pixels = height * width
    perms = {
        elem: get_d4_pixel_permutation(elem, height, width)
        for elem in D4_ELEMENTS
    }

    visited = [False] * n_pixels
    orbits = []
    for start in range(n_pixels):
        if visited[start]:
            continue
        orbit = set()
        for elem in D4_ELEMENTS:
            # perm[new] = old (pull convention) → orbit of pixel p is all
            # pixels reachable: {perm_g[p] for all g}
            orbit.add(int(perms[elem][start]))
        for p in orbit:
            visited[p] = True
        orbits.append(sorted(orbit))

    return orbits


def fit_d4_augmented_pca(
    X_flat_train: np.ndarray,
    n_components: int = 8,
    height: int = 32,
    width: int = 32,
    n_channels: int = 8,
    batch_size: int = 2000,
    save_path: Optional[str] = None,
    normalize_output: bool = True,
) -> Tuple[object, np.ndarray, np.ndarray]:
    """
    Fit a D4-INVARIANT PCA basis using exact orbit-indicator vectors.

    The orbit-based approach is provably correct:
    - D4 orbits on a height×width pixel grid are computed exactly.
    - Normalized orbit-indicator vectors are D4-invariant by construction:
      v[perm_g] = v for all g in D4, to machine precision (< 1e-15).
    - The top-k invariant directions are selected by explained variance
      on training data.

    For 32×32 images under D4: 136 orbits exist (120 of size 8,
    16 of size 4), giving 136 × n_channels = 1088 total invariant
    directions. Selecting 8 is trivially achievable.

    This replaces Reynolds-averaging + QR which fails because:
    - Top PCA pool vectors have near-zero projection onto invariant
      subspace (Reynolds maps them nearly to zero).
    - QR on near-zero vectors returns numerical noise, not invariant dirs.

    Args:
        X_flat_train: Flattened SAR patches, shape (n, H*W*C).
                      Channel-first layout: ch_k occupies
                      x_flat[k*H*W : (k+1)*H*W].
        n_components: Number of components (== N_QUBITS, default 8).
        height: Patch height (32 for So2Sat).
        width: Patch width (32 for So2Sat).
        n_channels: Number of SAR channels (8 for So2Sat).
        batch_size: Unused (kept for API compatibility).
        save_path: Path to save joblib model.
        normalize_output: Normalise PCA features to [0, pi].

    Returns:
        Tuple:
            pca: Object with .transform() and .components_ attributes.
            X_train_pca: Projected features, shape (n, n_components).
            X_train_pca_normalized: Features normalised to [0, pi].
    """
    import joblib

    logger.info(f"Fitting D4-invariant orbit-based PCA: "
                f"{len(X_flat_train)} samples, "
                f"patch {height}×{width}, channels={n_channels}, "
                f"n_components={n_components}")

    n_spatial = height * width
    n_raw = n_spatial * n_channels

    assert X_flat_train.shape[1] == n_raw, (
        f"X_flat_train.shape[1]={X_flat_train.shape[1]} != "
        f"height*width*n_channels={n_raw}"
    )

    # ---- Step 1: Find all D4 orbits on the pixel grid ----
    orbits = _find_d4_orbits(height, width)
    n_orbits = len(orbits)
    logger.info(f"  D4 orbits on {height}×{width} grid: {n_orbits} "
                f"(total basis candidates: {n_orbits * n_channels})")

    # ---- Step 2: Build normalized orbit-indicator basis vectors ----
    # Each basis vector has shape (n_raw,).
    # For channel ch and orbit O:
    #   v[ch * n_spatial + i] = 1/sqrt(|O|)  if i in O
    #   v[...] = 0                            otherwise
    # These are EXACTLY D4-invariant: v[perm_g] = v for all g.
    n_basis = n_orbits * n_channels
    basis = np.zeros((n_basis, n_raw), dtype=np.float64)
    for k, orbit in enumerate(orbits):
        inv_sqrt_size = 1.0 / np.sqrt(len(orbit))
        for ch in range(n_channels):
            row_idx = ch * n_orbits + k
            for pixel_idx in orbit:
                basis[row_idx, ch * n_spatial + pixel_idx] = inv_sqrt_size

    logger.info(f"  Orbit basis shape: {basis.shape}")

    # Sanity check: basis rows are orthonormal
    gram_diag_err = float(np.max(np.abs(
        np.einsum('ij,ij->i', basis, basis) - 1.0
    )))
    logger.info(f"  Orthonormality check (diag): max_err={gram_diag_err:.2e} "
                f"(expected < 1e-14)")
    if gram_diag_err > 1e-10:
        logger.warning(
            f"Orbit basis failed orthonormality check: {gram_diag_err:.2e}. "
            f"Check channel-first layout assumption."
        )

    # ---- Step 3: Select top-n_components by explained variance ----
    # Project training data onto all basis vectors.
    # Variance of each direction = explained variance.
    X_train_f64 = X_flat_train.astype(np.float64)
    mean_vec = X_train_f64.mean(axis=0)

    # Batch projection to limit memory
    n_train = len(X_train_f64)
    projections = np.zeros((n_train, n_basis), dtype=np.float64)
    chunk = max(1, 10000 // n_raw + 1) * 100
    for start in range(0, n_train, chunk):
        end = min(start + chunk, n_train)
        projections[start:end] = (
            X_train_f64[start:end] - mean_vec
        ) @ basis.T

    variances = projections.var(axis=0)
    top_k = np.argsort(variances)[::-1][:n_components]
    selected_variances = variances[top_k]

    logger.info(f"  Top-{n_components} variance (out of {n_basis} candidates):")
    for i, (idx, var) in enumerate(zip(top_k, selected_variances)):
        orbit_idx = idx % n_orbits
        ch_idx = idx // n_orbits
        logger.info(f"    [{i}] basis#{idx} (ch={ch_idx}, "
                    f"orbit_size={len(orbits[orbit_idx])}): var={var:.6f}")

    components = basis[top_k]  # shape (n_components, n_raw)
    logger.info(f"  Final components shape: {components.shape}")

    # ---- Step 4: Build PCA-like object ----
    pca = D4OrbitPCA(components, mean_vec)

    # ---- Step 5: Transform training data ----
    X_pca = projections[:, top_k]  # already computed, shape (n, n_components)

    # ---- Step 6: Normalise to [0, pi] ----
    if normalize_output:
        x_min = X_pca.min(axis=0)
        x_max = X_pca.max(axis=0)
        x_range = x_max - x_min
        x_range[x_range < 1e-10] = 1.0
        X_pca_norm = (X_pca - x_min) / x_range * np.pi
    else:
        X_pca_norm = X_pca

    # ---- Step 7: Save ----
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(pca, save_path)
        logger.info(f"  Saved D4OrbitPCA: {save_path}")

    return pca, X_pca, X_pca_norm



def verify_invariance(
    pca_components: np.ndarray,
    X_sample_raw: np.ndarray,
    height: int = 32,
    width: int = 32,
    n_channels: int = 8,
    tol: float = 1e-4,
    x_min: Optional[np.ndarray] = None,
    x_range: Optional[np.ndarray] = None,
) -> Tuple[bool, float]:
    """
    Verify invariance: theta(x_raw) == theta(g * x_raw).

    The invariance property is guaranteed by fitting PCA on D4-augmented orbits.
    The test applies D4 permutations to raw patches, projects via PCA,
    normalizes (if x_min/range provided), and compares to original features.

    Args:
        pca_components: PCA .components_, shape (n_components, raw_dim).
        X_sample_raw: A few raw sample patches, shape (n_test, raw_dim).
        height: Patch height.
        width: Patch width.
        n_channels: Number of channels.
        tol: Acceptable invariance error (default 1e-4).
        x_min: Optional normalization min.
        x_range: Optional normalization range.

    Returns:
        Tuple[bool, float]: (passed, max_error)
    """
    def theta(x_raw_batch):
        """Standard projection pipeline."""
        x_pca = x_raw_batch @ pca_components.T
        if x_min is not None and x_range is not None:
            x_norm = (x_pca - x_min) / x_range * np.pi
            return np.clip(x_norm, 0, np.pi)
        return x_pca

    # Reference features
    theta_orig = theta(X_sample_raw)

    max_error = 0.0
    passed = True

    for elem in D4_ELEMENTS:
        if elem == 'e': continue # Skip identity

        # Apply transformation to raw patches
        X_transformed = np.array([
            apply_d4_to_flat_patch(x, elem, height, width, n_channels)
            for x in X_sample_raw
        ])

        # Project transformed patches
        theta_transformed = theta(X_transformed)

        # Compute max difference across all samples for this group element
        err = float(np.max(np.abs(theta_orig - theta_transformed)))
        max_error = max(max_error, err)

        if err > tol:
            passed = False
            logger.warning(
                f"  Invariance FAIL: g={elem}, "
                f"||theta(x) - theta(gx)||_inf = {err:.6f} > tol={tol}"
            )

    passed = max_error <= tol
    logger.info(
        f"D4 Invariance check (raw space): max_error={max_error:.8f}, "
        f"tol={tol}, result={'PASS' if passed else 'FAIL'}"
    )
    return passed, max_error
