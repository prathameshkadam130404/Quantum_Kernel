"""
Self-Regulating Quantum Feature Map (SRQFM) kernel.

The per-pair entangling angle is the single-qubit fidelity distance between
the two encoded states, rather than the bilinear Havlicek angle:

    Standard ZZ:  J(x_i, x_j) = (pi - x_i)(pi - x_j)        (bilinear)
    SRQFM:        J(x_i, x_j) = sin^2((x_i - x_j) / 2)      (fidelity distance)

The SRQFM coupling equals the quantum fidelity distance between the
single-qubit states |psi(x)> = RZ(x) H |0>:
    d(x_i, x_j) = 1 - |<psi(x_i)|psi(x_j)>|^2
                = sin^2((x_i - x_j) / 2),
so the entangling angle is bounded in [0, 1] and vanishes when x_i = x_j
(coupling-threshold sparsification when feature values collide).
Properties (proofs in Appendix A of the paper):
    d = 0 at x_i = x_j           -- no entanglement (redundant features),
    d = 1 at |x_i - x_j| = pi    -- maximal entangling angle,
    differentiable interpolation between the two extremes.

Architecture:
    Per repetition:
        1. H on all qubits (superposition)
        2. RZ(x_i) on qubit i (feature encoding)
        3. For each pair (i, j):
           IsingZZ(sin^2((x_i - x_j)/2), wires=[i, j])
           (self-regulating entanglement)

Kernel Variants:
    - SRQFM-PQK: Bloch vector extraction + RBF kernel (1-body)
    - SRQFM-2PQK: 2-body RDM features + Hilbert-Schmidt kernel

References:
    - Havlicek et al., Nature 567, 209 (2019) — ZZFeatureMap
    - Huang et al., Nature Communications 12, 2631 (2021) — PQK
    - Heisenberg exchange: J * (X_iX_j + Y_iY_j + Z_iZ_j)

Author: Prathamesh Kadam et al.
"""

import os
import sys
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


# ============ COUPLING FUNCTIONS ============

def fidelity_coupling(x_i: float, x_j: float) -> float:
    """
    Compute the fidelity distance between two encoded qubit states.

    This is the core self-regulating coupling function:
        J(x_i, x_j) = sin^2((x_i - x_j) / 2)

    Derived from:
        |psi(x)> = (e^{-ix/2} |0> + e^{ix/2} |1>) / sqrt(2)  (after H -> RZ(x))
        |<psi(x_i)|psi(x_j)>|^2 = |(1/2)(e^{i(x_i-x_j)/2} + e^{-i(x_i-x_j)/2})|^2
                                = cos^2((x_i - x_j)/2)
        d = 1 - |<psi(x_i)|psi(x_j)>|^2 = sin^2((x_i - x_j)/2)

    Args:
        x_i: Encoded feature value for qubit i (in [0, pi]).
        x_j: Encoded feature value for qubit j (in [0, pi]).

    Returns:
        float: Coupling strength in [0, 1].
               0 = redundant features, 1 = maximally complementary.
    """
    return np.sin((x_i - x_j) / 2.0) ** 2


def standard_zz_coupling(x_i: float, x_j: float) -> float:
    """
    Standard Havlicek ZZ coupling: (pi - x_i)(pi - x_j).

    Provided for ablation comparison against fidelity coupling.

    Args:
        x_i: Encoded feature value for qubit i.
        x_j: Encoded feature value for qubit j.

    Returns:
        float: Coupling angle (unbounded, typically in [0, pi^2]).
    """
    return (np.pi - x_i) * (np.pi - x_j)


# ============ SRQFM CIRCUIT ============

def apply_srqfm_feature_map(
    x: np.ndarray,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    coupling_threshold: float = 0.01,
    connectivity: str = "all",
):
    """
    Apply the Self-Regulating Quantum Feature Map to the current quantum state.

    Must be called inside a PennyLane QNode context. Modifies the quantum
    state in-place via gate operations.

    The coupling strength between every qubit pair is automatically
    determined by the fidelity distance between their encoded states:
        J(x_i, x_j) = sin^2((x_i - x_j) / 2)

    Pairs with J < coupling_threshold are skipped (no entanglement),
    providing a natural sparsity that reduces circuit depth.

    Args:
        x: Input features, shape (n_qubits,). Values in [0, pi].
        n_qubits: Number of qubits.
        reps: Number of repetitions (layers).
        coupling_threshold: Minimum coupling to apply entanglement.
            Pairs with J < threshold are treated as non-interacting.
            Default 0.01 (skip only truly negligible couplings).
        connectivity: Pair topology for entanglement candidates.
            "all" — all (n choose 2) pairs tested (default, fully connected).
            "linear" — only adjacent pairs (i, i+1).
            "cross" — only pairs from different halves (modality-aware).
    """
    import pennylane as qml

    # Determine candidate pairs based on connectivity
    if connectivity == "all":
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    elif connectivity == "linear":
        pairs = [(i, i + 1) for i in range(n_qubits - 1)]
    elif connectivity == "cross":
        # First half and second half interact (modality fusion)
        mid = n_qubits // 2
        pairs = [(i, j) for i in range(mid) for j in range(mid, n_qubits)]
    else:
        raise ValueError(f"Unknown connectivity: {connectivity}")

    for _ in range(reps):
        # Layer 1: Hadamard (superposition)
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        # Layer 2: Single-qubit encoding
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)

        # Layer 3: Self-regulating entanglement
        for qi, qj in pairs:
            coupling = fidelity_coupling(x[qi], x[qj])
            if coupling > coupling_threshold:
                qml.IsingZZ(coupling, wires=[qi, qj])


# ============ BLOCH VECTOR EXTRACTION ============

def _build_srqfm_bloch_circuits(
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    coupling_threshold: float = 0.01,
    connectivity: str = "all",
):
    """
    Build PQK measurement circuits using the SRQFM feature map.

    Creates three QNodes measuring <X_i>, <Y_i>, <Z_i> for all qubits.

    Args:
        n_qubits: Number of qubits.
        reps: Feature map repetitions.
        coupling_threshold: Minimum coupling for entanglement.
        connectivity: Pair topology.

    Returns:
        Tuple of (measure_x, measure_y, measure_z) QNodes.
    """
    import pennylane as qml

    dev = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def measure_x(x):
        apply_srqfm_feature_map(x, n_qubits, reps, coupling_threshold,
                                connectivity)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def measure_y(x):
        apply_srqfm_feature_map(x, n_qubits, reps, coupling_threshold,
                                connectivity)
        return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def measure_z(x):
        apply_srqfm_feature_map(x, n_qubits, reps, coupling_threshold,
                                connectivity)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return measure_x, measure_y, measure_z


def compute_srqfm_bloch_vectors(
    X: np.ndarray,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    coupling_threshold: float = 0.01,
    connectivity: str = "all",
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Extract Bloch vectors from the SRQFM-encoded quantum states.

    For each sample x, measures <X_i>, <Y_i>, <Z_i> on all qubits
    to form the 3n-dimensional Bloch vector.

    Args:
        X: Input data, shape (N, n_qubits). Values in [0, pi].
        n_qubits: Number of qubits.
        reps: Feature map repetitions.
        coupling_threshold: Minimum coupling for entanglement.
        connectivity: Pair topology.
        save_path: Optional path to cache Bloch vectors (.npy).

    Returns:
        np.ndarray: Bloch vectors, shape (N, 3 * n_qubits).
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached SRQFM Bloch vectors: {save_path}")
        return np.load(save_path)

    N = len(X)
    measure_x, measure_y, measure_z = _build_srqfm_bloch_circuits(
        n_qubits, reps, coupling_threshold, connectivity
    )

    bloch = np.zeros((N, 3 * n_qubits), dtype=np.float64)

    logger.info(
        f"Computing SRQFM Bloch vectors: N={N}, qubits={n_qubits}, "
        f"reps={reps}, connectivity={connectivity}"
    )

    for idx in tqdm(range(N), desc="SRQFM Bloch", leave=True):
        ex = np.array(measure_x(X[idx]))
        ey = np.array(measure_y(X[idx]))
        ez = np.array(measure_z(X[idx]))
        for q in range(n_qubits):
            bloch[idx, 3 * q] = ex[q]
            bloch[idx, 3 * q + 1] = ey[q]
            bloch[idx, 3 * q + 2] = ez[q]

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, bloch)
        logger.info(f"Saved SRQFM Bloch vectors: {save_path}")

    # Diagnostic: per-qubit Bloch magnitudes
    magnitudes = np.zeros(n_qubits)
    for q in range(n_qubits):
        bx = bloch[:, 3 * q]
        by = bloch[:, 3 * q + 1]
        bz = bloch[:, 3 * q + 2]
        magnitudes[q] = np.sqrt(bx ** 2 + by ** 2 + bz ** 2).mean()
    logger.info(
        f"  Bloch magnitudes: {np.round(magnitudes, 3).tolist()} "
        f"(mean={magnitudes.mean():.3f})"
    )

    return bloch


# ============ KERNEL COMPUTATION ============

def compute_srqfm_pqk_kernel(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    gamma: float = config.PQK_GAMMA,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    coupling_threshold: float = 0.01,
    connectivity: str = "all",
    bloch_save_path: Optional[str] = None,
    kernel_save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute the SRQFM-PQK kernel matrix.

    Pipeline: SRQFM encoding -> Bloch vector extraction -> RBF kernel.

    K(x, x') = exp(-gamma * sum_q ||b_q(x) - b_q(x')||^2 / 2)

    where b_q = (<X_q>, <Y_q>, <Z_q>) is the Bloch vector for qubit q.

    Args:
        X: Training data, shape (N, n_qubits). Values in [0, pi].
        X2: Optional test data, shape (M, n_qubits). If None, compute K(X, X).
        gamma: RBF bandwidth parameter.
        n_qubits: Number of qubits.
        reps: Feature map repetitions.
        coupling_threshold: Minimum coupling for entanglement.
        connectivity: Pair topology.
        bloch_save_path: Path to cache Bloch vectors.
        kernel_save_path: Path to cache kernel matrix.

    Returns:
        np.ndarray: Kernel matrix, shape (N, N) or (M, N).
    """
    if kernel_save_path and os.path.exists(kernel_save_path):
        logger.info(f"[SKIP] Loading cached SRQFM-PQK kernel: {kernel_save_path}")
        return np.load(kernel_save_path)

    # Compute Bloch vectors
    bloch1 = compute_srqfm_bloch_vectors(
        X, n_qubits, reps, coupling_threshold, connectivity,
        save_path=bloch_save_path,
    )

    if X2 is not None:
        bloch2_path = (
            bloch_save_path.replace(".npy", "_test.npy")
            if bloch_save_path else None
        )
        bloch2 = compute_srqfm_bloch_vectors(
            X2, n_qubits, reps, coupling_threshold, connectivity,
            save_path=bloch2_path,
        )
    else:
        bloch2 = bloch1

    # Compute PQK RBF kernel from Bloch vectors
    n_train = len(bloch1)
    n_test = len(bloch2)
    b1 = bloch1.reshape(n_train, n_qubits, 3)
    b2 = bloch2.reshape(n_test, n_qubits, 3)

    if X2 is None:
        logger.info(f"Computing SRQFM-PQK kernel: {n_train}x{n_train} (symmetric)")
        K = np.zeros((n_train, n_train), dtype=np.float64)
        for i in tqdm(range(n_train), desc="SRQFM-PQK Kernel", leave=True):
            diff = b1[i] - b2  # (n_train, n_qubits, 3)
            sq_diff = np.sum(diff ** 2, axis=2)  # (n_train, n_qubits)
            frob_dist = 0.5 * np.sum(sq_diff, axis=1)  # (n_train,)
            K[i, :] = np.exp(-gamma * frob_dist)
    else:
        logger.info(f"Computing SRQFM-PQK kernel: {n_test}x{n_train} (rectangular)")
        K = np.zeros((n_test, n_train), dtype=np.float64)
        for i in tqdm(range(n_test), desc="SRQFM-PQK Kernel (test)", leave=True):
            diff = b2[i] - b1  # (n_train, n_qubits, 3)
            sq_diff = np.sum(diff ** 2, axis=2)
            frob_dist = 0.5 * np.sum(sq_diff, axis=1)
            K[i, :] = np.exp(-gamma * frob_dist)

    if kernel_save_path:
        os.makedirs(os.path.dirname(kernel_save_path), exist_ok=True)
        np.save(kernel_save_path, K)
        logger.info(f"Saved SRQFM-PQK kernel: {kernel_save_path}")

    return K


# ============ COUPLING DIAGNOSTICS ============

def compute_coupling_diagnostics(
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_qubits: int = config.N_QUBITS,
) -> Dict:
    """
    Compute coupling strength statistics for diagnostic purposes.

    Reports the mean, std, and modality breakdown of fidelity coupling
    across all pairs and samples. Useful for verifying that the
    self-regulating mechanism is working as expected.

    Args:
        X: Input data, shape (N, n_qubits). Values in [0, pi].
        feature_names: Optional list of feature names (length n_qubits).
        n_qubits: Number of qubits.

    Returns:
        dict: Coupling statistics including per-pair means and modality breakdown.
    """
    N = len(X)
    n_pairs = n_qubits * (n_qubits - 1) // 2
    pair_data = {}

    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            couplings = np.array([
                fidelity_coupling(X[s, i], X[s, j]) for s in range(N)
            ])
            name_i = feature_names[i] if feature_names else f"q{i}"
            name_j = feature_names[j] if feature_names else f"q{j}"
            pair_key = f"{name_i}-{name_j}"
            pair_data[pair_key] = {
                "mean": float(couplings.mean()),
                "std": float(couplings.std()),
                "max": float(couplings.max()),
                "min": float(couplings.min()),
                "frac_active": float((couplings > 0.01).mean()),
                "qubits": (i, j),
            }

    # Sort by mean coupling (descending)
    sorted_pairs = sorted(pair_data.items(), key=lambda x: -x[1]["mean"])

    result = {
        "n_pairs": n_pairs,
        "n_samples": N,
        "pairs": pair_data,
        "ranked_pairs": [(k, v["mean"]) for k, v in sorted_pairs],
        "global_mean": float(np.mean([v["mean"] for v in pair_data.values()])),
    }

    logger.info(f"  Coupling diagnostics: {n_pairs} pairs, "
                f"global mean J={result['global_mean']:.4f}")
    for k, mean_j in result["ranked_pairs"][:5]:
        logger.info(f"    Top: {k}: J={mean_j:.4f}")
    for k, mean_j in result["ranked_pairs"][-3:]:
        logger.info(f"    Low: {k}: J={mean_j:.4f}")

    return result


# ============ KERNEL TARGET ALIGNMENT ============

def compute_kta(K: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Kernel-Target Alignment (Cristianini et al., 2001).

    KTA = <K, K_y>_F / (||K||_F * ||K_y||_F)

    where K_y[i,j] = 1 if y[i] == y[j] else 0 (ideal kernel).

    Args:
        K: Kernel matrix, shape (N, N).
        y: Labels, shape (N,).

    Returns:
        float: KTA value in [-1, 1]. Higher = better class alignment.
    """
    K_ideal = (y[:, None] == y[None, :]).astype(np.float64)
    numerator = np.sum(K * K_ideal)
    denominator = np.sqrt(np.sum(K ** 2) * np.sum(K_ideal ** 2))
    if denominator < 1e-12:
        return 0.0
    return float(numerator / denominator)


# ============ HIGH-LEVEL API ============

def compute_srqfm_full_pipeline(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    gamma: float = config.PQK_GAMMA,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    coupling_threshold: float = 0.01,
    connectivity: str = "all",
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Run the complete SRQFM-PQK pipeline.

    Steps:
        1. Compute coupling diagnostics (for logging/analysis)
        2. Extract Bloch vectors via SRQFM circuit
        3. Compute PQK kernel matrix
        4. Compute KTA if labels provided

    Args:
        X_train: Training data, shape (N, n_qubits). Values in [0, pi].
        X_test: Optional test data, shape (M, n_qubits).
        y_train: Optional labels for KTA computation.
        gamma: RBF bandwidth for PQK kernel.
        n_qubits: Number of qubits.
        reps: Feature map repetitions.
        coupling_threshold: Minimum coupling for entanglement.
        connectivity: Pair topology ("all", "linear", "cross").
        save_dir: Directory to cache intermediate results.

    Returns:
        dict with keys:
            'K_train': np.ndarray, shape (N, N)
            'K_test': np.ndarray or None, shape (M, N)
            'bloch_train': np.ndarray, shape (N, 3*n_qubits)
            'coupling_diagnostics': dict
            'kta': float or None
            'config': dict of hyperparameters
    """
    start_time = time.time()

    # Paths
    bloch_path = os.path.join(save_dir, "srqfm_bloch.npy") if save_dir else None
    k_train_path = os.path.join(save_dir, "K_srqfm.npy") if save_dir else None
    k_test_path = os.path.join(save_dir, "K_srqfm_test.npy") if save_dir else None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  SRQFM-PQK Pipeline")
    logger.info(f"  N_train={len(X_train)}, qubits={n_qubits}, reps={reps}")
    logger.info(f"  connectivity={connectivity}, threshold={coupling_threshold}")
    logger.info("=" * 60)

    # Step 1: Coupling diagnostics
    logger.info("\nStep 1: Coupling diagnostics...")
    coupling_diag = compute_coupling_diagnostics(
        X_train[:min(200, len(X_train))], n_qubits=n_qubits
    )

    # Step 2: Train kernel
    logger.info("\nStep 2: Computing SRQFM-PQK train kernel...")
    K_train = compute_srqfm_pqk_kernel(
        X_train, gamma=gamma, n_qubits=n_qubits, reps=reps,
        coupling_threshold=coupling_threshold, connectivity=connectivity,
        bloch_save_path=bloch_path, kernel_save_path=k_train_path,
    )

    # Step 3: Test kernel (if test data provided)
    K_test = None
    if X_test is not None:
        logger.info("\nStep 3: Computing SRQFM-PQK test kernel...")
        K_test = compute_srqfm_pqk_kernel(
            X_train, X2=X_test, gamma=gamma, n_qubits=n_qubits, reps=reps,
            coupling_threshold=coupling_threshold, connectivity=connectivity,
            bloch_save_path=bloch_path, kernel_save_path=k_test_path,
        )

    # Step 4: KTA
    kta_val = None
    if y_train is not None:
        kta_val = compute_kta(K_train, y_train)
        logger.info(f"\n  KTA = {kta_val:.4f}")

    elapsed = time.time() - start_time
    logger.info(f"\n  Total pipeline time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return {
        "K_train": K_train,
        "K_test": K_test,
        "coupling_diagnostics": coupling_diag,
        "kta": kta_val,
        "elapsed_s": elapsed,
        "config": {
            "gamma": gamma,
            "n_qubits": n_qubits,
            "reps": reps,
            "coupling_threshold": coupling_threshold,
            "connectivity": connectivity,
        },
    }


# ============ SELF-TEST ============

def _self_test():
    """
    Validate SRQFM kernel properties:
        1. Coupling function bounds [0, 1]
        2. Kernel symmetry: K = K^T
        3. Kernel diagonal: K[i,i] = 1
        4. Kernel PSD: all eigenvalues >= 0
        5. Bloch vector magnitudes: all > 0 (no dead qubits)
        6. Self-regulating: redundant features have low coupling
    """
    import pennylane as qml

    print("=" * 60)
    print("  SRQFM Kernel Self-Test")
    print("=" * 60)

    np.random.seed(config.RANDOM_SEED)
    n_q = 4  # Use fewer qubits for speed
    N = 20

    # Generate test data
    X = np.random.rand(N, n_q) * np.pi

    # Test 1: Coupling function bounds
    print("\n[1] Coupling function bounds...")
    for _ in range(1000):
        a, b = np.random.rand(2) * np.pi
        c = fidelity_coupling(a, b)
        assert 0.0 <= c <= 1.0, f"Coupling out of bounds: {c}"
    assert abs(fidelity_coupling(1.5, 1.5)) < 1e-10, "Same input should give 0"
    assert abs(fidelity_coupling(0.0, np.pi) - 1.0) < 1e-10, "Max diff should give 1"
    print("  [OK] Coupling in [0, 1], boundary conditions correct.")

    # Test 2: Kernel computation
    print("\n[2] Computing SRQFM-PQK kernel (4 qubits, 20 samples)...")
    K = compute_srqfm_pqk_kernel(
        X, gamma=0.67, n_qubits=n_q, reps=2,
        coupling_threshold=0.01, connectivity="all",
    )

    # Test 3: Symmetry
    print("\n[3] Checking kernel symmetry...")
    asym = np.max(np.abs(K - K.T))
    assert asym < 1e-10, f"Asymmetry: {asym}"
    print(f"  [OK] Max asymmetry: {asym:.2e}")

    # Test 4: Diagonal
    print("\n[4] Checking diagonal = 1...")
    diag_err = np.max(np.abs(K.diagonal() - 1.0))
    assert diag_err < 1e-10, f"Diagonal error: {diag_err}"
    print(f"  [OK] Max diagonal deviation: {diag_err:.2e}")

    # Test 5: PSD
    print("\n[5] Checking PSD...")
    eig = np.linalg.eigvalsh((K + K.T) / 2)
    min_eig = eig.min()
    assert min_eig > -1e-8, f"Negative eigenvalue: {min_eig}"
    print(f"  [OK] Min eigenvalue: {min_eig:.6f}")

    # Test 6: Bloch magnitudes
    print("\n[6] Checking Bloch vector magnitudes (no dead qubits)...")
    bloch = compute_srqfm_bloch_vectors(X, n_qubits=n_q, reps=2)
    for q in range(n_q):
        bx = bloch[:, 3 * q]
        by = bloch[:, 3 * q + 1]
        bz = bloch[:, 3 * q + 2]
        mag = np.sqrt(bx ** 2 + by ** 2 + bz ** 2).mean()
        assert mag > 0.1, f"Qubit {q} has low Bloch magnitude: {mag}"
        print(f"  q{q}: |b| = {mag:.4f}")
    print("  [OK] All qubits have non-trivial Bloch vectors.")

    # Test 7: Self-regulation check
    print("\n[7] Self-regulation: redundant features should have low coupling...")
    # Create data where features 0 and 1 are identical (redundant)
    X_redundant = np.random.rand(50, n_q) * np.pi
    X_redundant[:, 1] = X_redundant[:, 0]  # Feature 1 = copy of feature 0
    couplings_01 = np.array([
        fidelity_coupling(X_redundant[s, 0], X_redundant[s, 1])
        for s in range(50)
    ])
    couplings_02 = np.array([
        fidelity_coupling(X_redundant[s, 0], X_redundant[s, 2])
        for s in range(50)
    ])
    assert couplings_01.mean() < 0.001, (
        f"Redundant pair coupling too high: {couplings_01.mean()}"
    )
    assert couplings_02.mean() > 0.05, (
        f"Non-redundant pair coupling too low: {couplings_02.mean()}"
    )
    print(f"  Redundant pair (0,1):     mean J = {couplings_01.mean():.6f}")
    print(f"  Non-redundant pair (0,2): mean J = {couplings_02.mean():.4f}")
    print("  [OK] Self-regulation working correctly.")

    # Test 8: Coupling diagnostics
    print("\n[8] Coupling diagnostics...")
    diag = compute_coupling_diagnostics(X, n_qubits=n_q)
    assert diag["n_pairs"] == n_q * (n_q - 1) // 2
    print(f"  [OK] {diag['n_pairs']} pairs analyzed, "
          f"global mean J = {diag['global_mean']:.4f}")

    print("\n" + "=" * 60)
    print("  ALL SELF-TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    _self_test()
