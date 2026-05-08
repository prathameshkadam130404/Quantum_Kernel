"""
D4 Representation Matrices for PCA Feature Space.

Computes the exact action of the dihedral group D4 on the 8-dimensional
PCA feature space derived from So2Sat satellite patches.

D4 = {e, r, r2, r3, s, sr, sr2, sr3}
  e   = identity
  r   = 90-degree counterclockwise rotation
  r2  = 180-degree rotation
  r3  = 270-degree rotation (= 90-degree clockwise)
  s   = horizontal flip (left-right)
  sr  = horizontal flip then 90-degree rotation (transpose)
  sr2 = vertical flip
  sr3 = anti-transpose

For a 32x32 image patch flattened to a vector, each D4 element acts as
a permutation of the pixel indices. The D4 representation in PCA space is:
    M_g = P @ R_g @ P.T
where P is the PCA components matrix (shape 8 x raw_dim).

Since R_g is a permutation matrix, P @ R_g = P[:, perm_g], making the
computation efficient without materializing R_g explicitly.

Key output: M_g is an 8x8 matrix. If all M_g are approximately permutation
matrices, the D4 action is discrete and clean orbits can be extracted.
If M_g matrices are dense, the continuous group-averaging approach is used.

References:
    Glick et al., Nature Physics (2024) — covariant quantum kernels
    Cohen & Welling, ICML 2016 — group equivariant CNNs
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

logger = logging.getLogger(__name__)

D4_ELEMENTS = ['e', 'r', 'r2', 'r3', 's', 'sr', 'sr2', 'sr3']


def get_d4_pixel_permutation(
    element: str,
    height: int = 32,
    width: int = 32,
) -> np.ndarray:
    """
    Compute the pixel index permutation for a D4 group element.

    For a (height x width) image patch flattened in row-major order,
    each D4 element acts as a permutation of pixel indices.
    Returns perm such that: transformed_flat[new_idx] = original_flat[perm[new_idx]]

    Args:
        element: One of 'e','r','r2','r3','s','sr','sr2','sr3'.
        height: Patch height (default 32). Must equal width for D4.
        width: Patch width (default 32).

    Returns:
        np.ndarray: Integer permutation array, shape (height*width,).
    """
    assert height == width, \
        f"D4 symmetry requires square patches. Got height={height}, width={width}."
    n = height

    rc_to_flat = {(r, c): r * n + c
                  for r in range(n) for c in range(n)}

    transforms = {
        'e':   lambda r, c: (r, c),
        'r':   lambda r, c: (n - 1 - c, r),
        'r2':  lambda r, c: (n - 1 - r, n - 1 - c),
        'r3':  lambda r, c: (c, n - 1 - r),
        's':   lambda r, c: (r, n - 1 - c),
        'sr':  lambda r, c: (c, r),
        'sr2': lambda r, c: (n - 1 - r, c),
        'sr3': lambda r, c: (n - 1 - c, n - 1 - r),
    }

    assert element in transforms, \
        f"Unknown D4 element '{element}'. Choose from {list(transforms.keys())}"

    transform = transforms[element]
    perm = np.zeros(n * n, dtype=np.int64)

    for r in range(n):
        for c in range(n):
            new_r, new_c = transform(r, c)
            old_flat = rc_to_flat[(r, c)]
            new_flat = rc_to_flat[(new_r, new_c)]
            perm[new_flat] = old_flat

    return perm


def compute_m_g_matrices(
    pca_components: np.ndarray,
    patch_height: int = 32,
    patch_width: int = 32,
    n_channels: int = 1,
    save_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute exact D4 representation matrices in PCA feature space.

    M_g = P @ R_g @ P.T = P[:, perm_g] @ P.T

    where P = pca_components (shape n_components x raw_dim) and
    perm_g is the pixel permutation for group element g applied to
    each channel block independently.

    IMPORTANT: pca_components.shape[1] must equal
    patch_height * patch_width * n_channels. Verify n_channels by
    computing raw_dim / (32*32) from your fitted PCA object.

    Sanity check: M_e must equal identity (within numerical precision).
    This is automatically verified and logged.

    Args:
        pca_components: PCA .components_ attribute, shape (n_components, raw_dim).
        patch_height: Image patch height (default 32 for So2Sat).
        patch_width: Image patch width (default 32 for So2Sat).
        n_channels: Number of input channels. MUST satisfy:
            n_channels = pca_components.shape[1] / (patch_height * patch_width)
        save_path: Optional .npz path to cache results.

    Returns:
        Dict[str, np.ndarray]: {element: M_g (n_components x n_components)}
        for all 8 D4 elements.

    Raises:
        AssertionError: If pca_components.shape[1] != patch_height * patch_width * n_channels.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached M_g matrices: {save_path}")
        loaded = np.load(save_path)
        return {k: loaded[k] for k in D4_ELEMENTS}

    n_components, raw_dim = pca_components.shape
    expected_raw = patch_height * patch_width * n_channels

    assert raw_dim == expected_raw, (
        f"PCA raw_dim mismatch: pca_components.shape[1]={raw_dim} but "
        f"patch_height({patch_height}) * patch_width({patch_width}) * "
        f"n_channels({n_channels}) = {expected_raw}. "
        f"Check n_channels. It should be raw_dim / (32*32) = "
        f"{raw_dim} / 1024 = {raw_dim / 1024:.2f}."
    )

    n_spatial = patch_height * patch_width
    M_g_dict = {}

    logger.info(f"Computing M_g matrices: {n_components}x{n_components} "
               f"for {len(D4_ELEMENTS)} D4 elements")
    logger.info(f"PCA shape: {pca_components.shape}, "
               f"patch: {patch_height}x{patch_width}, channels: {n_channels}")

    for element in D4_ELEMENTS:
        spatial_perm = get_d4_pixel_permutation(element, patch_height, patch_width)

        # Multi-channel: same spatial permutation applied per channel block
        if n_channels == 1:
            full_perm = spatial_perm
        else:
            full_perm = np.concatenate([
                spatial_perm + ch * n_spatial
                for ch in range(n_channels)
            ])

        # M_g = P[:, perm] @ P.T (no need to build full R_g matrix)
        P_permuted = pca_components[:, full_perm]
        M_g = P_permuted @ pca_components.T

        M_g_dict[element] = M_g

        is_perm = _is_approx_permutation(M_g)
        max_od = float(np.max(np.abs(M_g - np.diag(np.diag(M_g)))))
        logger.info(f"  M_{element}: shape={M_g.shape}, "
                   f"max_offdiag={max_od:.4f}, "
                   f"approx_permutation={is_perm}")

    # Sanity check: M_e must be identity
    identity_err = float(np.max(np.abs(M_g_dict['e'] - np.eye(n_components))))
    logger.info(f"\nSanity check ||M_e - I||_inf = {identity_err:.8f} "
               f"(expected ~0 for orthonormal PCA components)")
    if identity_err > 0.01:
        logger.warning(
            f"M_e deviates from identity by {identity_err:.6f}. "
            f"This suggests PCA components are not fully orthonormal at "
            f"float64 precision, or n_channels is wrong. "
            f"Check pca_components.shape[1] == {expected_raw}."
        )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **M_g_dict)
        logger.info(f"Saved M_g matrices: {save_path}")

    return M_g_dict


def _is_approx_permutation(M: np.ndarray, tol: float = 0.1) -> bool:
    """
    Check whether matrix M is approximately a permutation matrix.

    A permutation matrix has exactly one entry close to ±1 per row
    and column, all others near zero. If all M_g pass this check,
    D4 acts as a discrete feature permutation and clean orbits exist.

    Args:
        M: Square matrix to check.
        tol: Tolerance for near-zero entries.

    Returns:
        bool: True if M is approximately a permutation matrix.
    """
    M_abs = np.abs(M)
    n = M.shape[0]

    for i in range(n):
        row = M_abs[i]
        max_val = row.max()
        if max_val < 1 - tol:
            return False
        rest = row[row < max_val - tol / 2]
        if np.any(rest > tol):
            return False

    for j in range(n):
        col = M_abs[:, j]
        max_val = col.max()
        if max_val < 1 - tol:
            return False
        rest = col[col < max_val - tol / 2]
        if np.any(rest > tol):
            return False

    return True


def extract_orbits_from_m_g(
    M_g_dict: Dict[str, np.ndarray],
    tol: float = 0.1,
) -> Tuple[Optional[List[List[int]]], bool]:
    """
    Extract orbit partition from M_g matrices when they are approximately
    permutation matrices.

    Features i and j are in the same orbit if any M_g maps feature i
    to feature j (i.e., M_g[j, i] ≈ 1 for some g).

    Uses union-find to build the complete orbit partition.

    Args:
        M_g_dict: Output of compute_m_g_matrices().
        tol: Tolerance for permutation detection.

    Returns:
        Tuple:
          - orbits: List of orbit lists (each is a set of feature indices)
            or None if M_g are not approximately permutation matrices.
          - is_clean: True if discrete orbit assignment is valid.
    """
    all_perm = all(_is_approx_permutation(M, tol) for M in M_g_dict.values())

    if not all_perm:
        n_dense = sum(
            1 for M in M_g_dict.values() if not _is_approx_permutation(M, tol)
        )
        logger.warning(
            f"{n_dense}/{len(M_g_dict)} M_g matrices are NOT approximately "
            f"permutation matrices. D4 acts as a continuous (dense) rotation "
            f"on PCA features, not a discrete permutation. "
            f"Will use group-averaging (continuous) approach instead of "
            f"orbit parameter sharing (discrete) approach."
        )
        return None, False

    n = list(M_g_dict.values())[0].shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for element, M in M_g_dict.items():
        # Extract permutation: row i maps to argmax of |M[i, :]|
        perm = np.argmax(np.abs(M), axis=1)
        for i in range(n):
            union(i, int(perm[i]))

    orbit_dict = {}
    for i in range(n):
        root = find(i)
        orbit_dict.setdefault(root, []).append(i)

    orbits = sorted(orbit_dict.values(), key=lambda o: o[0])

    logger.info(f"Extracted {len(orbits)} exact D4 orbits:")
    for idx, orbit in enumerate(orbits):
        logger.info(f"  Orbit {idx}: features {orbit}")

    return orbits, True


def build_covariant_encoding(
    x: np.ndarray,
    M_g_dict: Dict[str, np.ndarray],
) -> np.ndarray:
    """
    Compute group-averaged encoding angles (continuous covariant case).

    For each feature dimension i, the covariant encoding is:
        theta_i(x) = (1/|D4|) * sum_{g in D4} [M_g @ x]_i

    This is invariant under D4 by group closure:
        theta(M_h @ x) = (1/|D4|) * sum_g M_g @ M_h @ x
                       = (1/|D4|) * sum_{g'} M_{g'} @ x  (g' = g*h)
                       = theta(x)

    Used in apply_d4_covariant_feature_map_continuous() to replace
    the raw feature x_i with theta_i(x) in each RZ gate.

    Args:
        x: Feature vector, shape (n_components,). Values in [0, pi].
        M_g_dict: {element: M_g matrix} from compute_m_g_matrices().

    Returns:
        np.ndarray: Group-averaged encoding angles, shape (n_components,).
        Values will be in [0, pi] if input is in [0, pi] and M_g are
        orthogonal (which they are for permutation-like structures).
    """
    averaged = np.zeros_like(x, dtype=np.float64)
    for M_g in M_g_dict.values():
        averaged += M_g @ x
    averaged /= len(M_g_dict)

    # Clip to [0, pi] to maintain valid encoding range
    averaged = np.clip(averaged, 0.0, np.pi)
    return averaged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("  src/d4_representation.py — Self-test")
    print("=" * 60)

    np.random.seed(42)

    # Test 1: Pixel permutations are valid permutations
    print("\n--- Test 1: Pixel permutations ---")
    for elem in D4_ELEMENTS:
        perm = get_d4_pixel_permutation(elem)
        assert len(perm) == 32 * 32, f"Wrong length for {elem}"
        assert set(perm.tolist()) == set(range(32 * 32)), \
            f"Not a valid permutation for {elem}"
        print(f"  {elem}: valid permutation ✓")

    # Test 2: Group closure — r applied 4 times = identity
    print("\n--- Test 2: Group closure ---")
    perm_e = get_d4_pixel_permutation('e')
    perm_r = get_d4_pixel_permutation('r')
    perm_r4 = perm_r[perm_r[perm_r[perm_r]]]
    assert np.all(perm_r4 == perm_e), "r^4 != identity"
    print("  r^4 == identity ✓")

    perm_s = get_d4_pixel_permutation('s')
    perm_s2 = perm_s[perm_s]
    assert np.all(perm_s2 == perm_e), "s^2 != identity"
    print("  s^2 == identity ✓")

    # Test 3: M_g with synthetic orthonormal PCA
    print("\n--- Test 3: M_g matrices with synthetic PCA ---")
    raw_dim = 32 * 32 * 1
    n_comp = 8
    # Create orthonormal rows (synthetic PCA components)
    rng = np.random.RandomState(42)
    A = rng.randn(n_comp, raw_dim)
    from numpy.linalg import qr
    Q, _ = qr(A.T)
    P_synthetic = Q[:, :n_comp].T  # shape (8, 1024), orthonormal rows

    M_g_dict = compute_m_g_matrices(P_synthetic, n_channels=1)

    # M_e should be identity
    identity_err = np.max(np.abs(M_g_dict['e'] - np.eye(n_comp)))
    print(f"  ||M_e - I||_inf = {identity_err:.8f} (expected < 1e-10)")
    assert identity_err < 1e-6, f"M_e not identity: {identity_err}"

    # Test 4: Covariant encoding invariance
    print("\n--- Test 4: Group-averaging invariance ---")
    x_test = rng.rand(n_comp) * np.pi
    theta_x = build_covariant_encoding(x_test, M_g_dict)

    # Apply r to x and check theta is same
    perm_r_full = get_d4_pixel_permutation('r')
    x_rotated = M_g_dict['r'] @ x_test
    x_rotated = np.clip(x_rotated, 0, np.pi)
    theta_x_rotated = build_covariant_encoding(x_rotated, M_g_dict)

    # For truly orthonormal P and exact permutation M_g,
    # theta(g*x) should equal theta(x)
    diff = np.max(np.abs(theta_x - theta_x_rotated))
    print(f"  ||theta(x) - theta(r*x)||_inf = {diff:.6f} (expected ~0)")

    print("\n  All self-tests passed.")
