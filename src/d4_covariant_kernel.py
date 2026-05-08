"""
D4 Covariant Quantum Kernel for LCZ Classification (Group-Averaged FQK).

Achieves D4 symmetry by explicitly group-averaging the standard FQK kernel
over the dihedral group D4 applied to raw pixel patches before PCA projection.
This ensures invariance/covariance even when PCA features mix linearly.

Mathematical formula:
    K_D4(x, y) = (1/8) * sum_{g in D4} K_FQK( PCA(g * x_raw), PCA(y_raw) )

Reference:
    Glick et al., Nature Physics (2024) — covariant quantum kernels
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

logger = logging.getLogger(__name__)

def apply_standard_zz_feature_map(
    x: np.ndarray,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
) -> None:
    """
    Standard ZZFeatureMap for 8 PCA components.
    Used as the inner core for D4 kernel-level group averaging.
    """
    import pennylane as qml
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(float(x[i]), wires=i)
        for i in range(n_qubits - 1):
            j = i + 1
            zz_angle = (np.pi - float(x[i])) * (np.pi - float(x[j]))
            qml.CNOT(wires=[i, j])
            qml.RZ(zz_angle, wires=j)
            qml.CNOT(wires=[i, j])

def compute_group_averaged_fqk(
    X1_raw: np.ndarray,
    pca_model: object,
    norm_min: np.ndarray,
    norm_range: np.ndarray,
    X2_raw: Optional[np.ndarray] = None,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    patch_height: int = 32,
    patch_width: int = 32,
    n_channels: int = 8,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute D4-covariant kernel by averaging over raw rotations.
    
    Formula: K(x, y) = 1/8 * sum_g K_std(PCA(g*x_raw), PCA(y_raw))
    
    Args:
        X1_raw: Raw flattened patches, shape (n1, raw_dim).
        pca_model: Fitted PCA object (e.g. from joblib).
        norm_min: Scaling min for [0, pi] normalization, shape (n_qubits,).
        norm_range: Scaling range for [0, pi] normalization, shape (n_qubits,).
        X2_raw: Optional second raw data matrix, shape (n2, raw_dim).
        n_qubits: Number of qubits.
        reps: Circuit repetitions.
        patch_height: Patch height (32).
        patch_width: Patch width (32).
        n_channels: Number of channels.
        save_path: Path to cache the kernel matrix.
        
    Returns:
        np.ndarray: D4-covariant/invariant kernel matrix.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached D4 GA-FQK: {save_path}")
        return np.load(save_path)

    from src.d4_augmented_pca import D4_ELEMENTS
    from src.d4_representation import get_d4_pixel_permutation
    import pennylane as qml

    dev = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def fqk_circuit(x1, x2):
        apply_standard_zz_feature_map(x1, n_qubits, reps)
        qml.adjoint(apply_standard_zz_feature_map)(x2, n_qubits, reps)
        return qml.probs(wires=range(n_qubits))

    def project_and_normalize(raw):
        # raw shape: (n, raw_dim)
        x_pca = pca_model.transform(raw)
        x_norm = (x_pca - norm_min) / norm_range * np.pi
        return np.clip(x_norm, 0, np.pi)

    # 1. Prepare X2 reference (standard PCA transform)
    if X2_raw is None:
        X2_norm = project_and_normalize(X1_raw)
        m, n = len(X1_raw), len(X1_raw)
        symmetric = True
    else:
        X2_norm = project_and_normalize(X2_raw)
        m, n = len(X2_raw), len(X1_raw)
        symmetric = False

    K = np.zeros((m, n), dtype=np.float64)
    logger.info(f"Computing D4 GA-FQK: {m}x{n} matrix over 8 D4 elements")

    # Precompute permutations to save time
    n_spatial = patch_height * patch_width
    perms = []
    for elem in D4_ELEMENTS:
        spatial_perm = get_d4_pixel_permutation(elem, patch_height, patch_width)
        full_perm = np.concatenate([
            spatial_perm + ch * n_spatial for ch in range(n_channels)
        ])
        perms.append(full_perm)

    K = np.zeros((m, n), dtype=np.float64)
    for k, perm in enumerate(perms):
        X1_g_norm = project_and_normalize(X1_raw[:, perm])
        logger.info(f"  Group element {k+1}/8: {D4_ELEMENTS[k]}")
        for i in range(m):
            for j in range(n):
                prob = float(fqk_circuit(X1_g_norm[j], X2_norm[i])[0])
                K[i, j] += prob

    K /= len(D4_ELEMENTS)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, K)
        logger.info(f"Saved D4 GA-FQK: {save_path}")
        
    return K

def verify_d4_covariance(
    pca_model: object,
    norm_min: np.ndarray,
    norm_range: np.ndarray,
    n_checks: int = 5,
    tol: float = 1e-4,
) -> bool:
    """
    Verify invariance: K(rotated_patch, ref) == K(patch, ref).
    """
    from src.d4_augmented_pca import D4_ELEMENTS
    
    # Create dummy raw patch (random but structured)
    rng = np.random.RandomState(42)
    raw_dim = 32 * 32 * 8
    patch = rng.rand(1, raw_dim).astype(np.float32)
    ref = rng.rand(1, raw_dim).astype(np.float32)
    
    # Compute K(patch, ref)
    K_orig = compute_group_averaged_fqk(patch, pca_model, norm_min, norm_range, X2_raw=ref)
    val_orig = K_orig[0, 0]
    
    passed = True
    max_err = 0.0
    
    from src.d4_representation import get_d4_pixel_permutation
    
    for elem in D4_ELEMENTS[1:]: # Skip identity
        # Rotate patch
        spatial_perm = get_d4_pixel_permutation(elem, 32, 32)
        full_perm = np.concatenate([spatial_perm + ch*(32*32) for ch in range(8)])
        patch_g = patch[:, full_perm]
        
        # Compute K(patch_g, ref)
        K_rot = compute_group_averaged_fqk(patch_g, pca_model, norm_min, norm_range, X2_raw=ref)
        val_rot = K_rot[0, 0]
        
        err = abs(val_orig - val_rot)
        max_err = max(max_err, err)
        if err > tol:
            passed = False
            logger.warning(f"  D4 Invariance check FAILED for {elem}: err={err:.8f} > tol={tol}")
            
    logger.info(f"D4 Invariance verification: max_error={max_err:.8f}, PASSED={passed}")
    return passed

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("  src/d4_covariant_kernel.py — GA-FQK Self-test")
    print("=" * 60)

    # 1. Create synthetic PCA and dummy normalization params
    from sklearn.decomposition import PCA
    raw_dim = 32 * 32 * 8
    n_comp = 8
    rng = np.random.RandomState(42)
    dummy_data = rng.rand(100, raw_dim)
    pca = PCA(n_components=n_comp).fit(dummy_data)
    
    # Dummy normalization
    norm_min = np.zeros(n_comp)
    norm_range = np.ones(n_comp)
    
    # 2. Verify invariance
    print("\nVerifying spatial rotation invariance...")
    ok = verify_d4_covariance(pca, norm_min, norm_range, n_checks=3)
    
    if ok:
        print("\n  Self-test PASSED.")
    else:
        print("\n  Self-test FAILED.")
        sys.exit(1)
