"""
Two-Body Projected Quantum Kernel (2-PQK) measurement module.

Extracts 2-qubit Reduced Density Matrix (2-RDM) features from a quantum
feature map circuit, capturing pairwise entanglement signatures that
standard single-qubit PQK (Bloch vectors) provably cannot access.

The 2-body feature vector for each qubit pair (i, j) contains 15 real
numbers: 6 single-qubit expectation values + 9 two-qubit correlation
terms ⟨σ_a ⊗ σ_b⟩ for a,b ∈ {X, Y, Z}.

The 9 correlation terms encode the entanglement structure created by
ZZ gates in the feature map — information that single-qubit Bloch
vectors discard.

Mathematical Foundation:
    ρ_{ij}(x) = Tr_{≠i,j}(|ψ(x)⟩⟨ψ(x)|) ∈ ℂ^{4×4}
    ρ_{ij} = (1/4) Σ_{a,b} c_{ab}(x) · (σ_a ⊗ σ_b)
    c_{ab}(x) = ⟨σ_a^(i) ⊗ σ_b^(j)⟩_x

    Kernel:
    K_{2-PQK}(x,x') = exp(-γ Σ_{(i,j)∈S} ‖ρ_{ij}(x) − ρ_{ij}(x')‖²_F)

References:
    - Thanasilp et al. (2024): PQK anti-concentration (1-RDM level)
    - Havlíček et al. (2019): ZZFeatureMap
    - This work: extends PQK to 2-RDM level (novel)

Author: Prathamesh Kadam et al.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)

# ============ CONSTANTS ============

# The 15 non-trivial Pauli product labels for a 2-qubit subsystem.
# Ordering: 3 single-qubit (i), 3 single-qubit (j), 9 two-qubit correlations.
# This matches the Pauli decomposition ρ_{ij} = (1/4) Σ c_{ab} σ_a ⊗ σ_b.
PAULI_LABELS_2BODY = [
    # Single-qubit (i side): a ∈ {X,Y,Z}, b = I
    "XI", "YI", "ZI",
    # Single-qubit (j side): a = I, b ∈ {X,Y,Z}
    "IX", "IY", "IZ",
    # Two-qubit correlations: a,b ∈ {X,Y,Z}
    "XX", "XY", "XZ",
    "YX", "YY", "YZ",
    "ZX", "ZY", "ZZ",
]

N_FEATURES_PER_PAIR = len(PAULI_LABELS_2BODY)  # 15


# ============ CIRCUIT BUILDER ============

def _build_2body_circuits(
    gamma_per_qubit: np.ndarray,
    entanglement_pairs: List[Tuple[int, int]],
    pairs_to_measure: List[Tuple[int, int]],
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
):
    """
    Build PennyLane QNodes that measure all 15 Pauli products for each
    pair in pairs_to_measure, after applying the ZZFeatureMap with
    custom per-qubit bandwidth and entanglement topology.

    Circuit structure per repetition:
        1. H on all qubits
        2. RZ(γ_i × x_i) on qubit i
        3. For each (i,j) in entanglement_pairs:
           CNOT(i,j) → RZ((π − γ_i·x_i)(π − γ_j·x_j)) → CNOT(i,j)

    Args:
        gamma_per_qubit: Per-qubit bandwidth array, shape (n_qubits,).
        entanglement_pairs: List of (int, int) for ZZ interactions in the
            feature map circuit topology.
        pairs_to_measure: List of (int, int) tuples — qubit pairs to
            extract 2-body features from. These can differ from the
            entanglement_pairs (but are typically the same).
        n_qubits: Total number of qubits.
        reps: Number of ZZFeatureMap repetitions.

    Returns:
        Dict mapping pair (qi, qj) to a callable qnode(x) that returns
        a 15-element array of Pauli product expectations.
    """
    import pennylane as qml

    dev = config.get_device(n_qubits)

    def apply_feature_map(x):
        """Apply the ZZFeatureMap with data-dependent topology."""
        for _ in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for i in range(n_qubits):
                qml.RZ(gamma_per_qubit[i] * x[i], wires=i)
            for qi, qj in entanglement_pairs:
                if qi < n_qubits and qj < n_qubits:
                    zz_angle = (
                        (np.pi - gamma_per_qubit[qi] * x[qi])
                        * (np.pi - gamma_per_qubit[qj] * x[qj])
                    )
                    qml.CNOT(wires=[qi, qj])
                    qml.RZ(zz_angle, wires=qj)
                    qml.CNOT(wires=[qi, qj])

    # Build one QNode per pair that returns all 15 Pauli products.
    # PennyLane supports multiple return values in a single QNode,
    # so we batch all 15 measurements into one circuit execution.
    pair_circuits = {}

    for qi, qj in pairs_to_measure:

        @qml.qnode(dev, diff_method=None)
        def measure_pair(x, _qi=qi, _qj=qj):
            """Measure all 15 non-trivial Pauli products for pair (_qi, _qj)."""
            apply_feature_map(x)
            return [
                # 3 single-qubit expectations for qubit i
                qml.expval(qml.PauliX(_qi)),
                qml.expval(qml.PauliY(_qi)),
                qml.expval(qml.PauliZ(_qi)),
                # 3 single-qubit expectations for qubit j
                qml.expval(qml.PauliX(_qj)),
                qml.expval(qml.PauliY(_qj)),
                qml.expval(qml.PauliZ(_qj)),
                # 9 two-qubit correlation terms
                qml.expval(qml.PauliX(_qi) @ qml.PauliX(_qj)),
                qml.expval(qml.PauliX(_qi) @ qml.PauliY(_qj)),
                qml.expval(qml.PauliX(_qi) @ qml.PauliZ(_qj)),
                qml.expval(qml.PauliY(_qi) @ qml.PauliX(_qj)),
                qml.expval(qml.PauliY(_qi) @ qml.PauliY(_qj)),
                qml.expval(qml.PauliY(_qi) @ qml.PauliZ(_qj)),
                qml.expval(qml.PauliZ(_qi) @ qml.PauliX(_qj)),
                qml.expval(qml.PauliZ(_qi) @ qml.PauliY(_qj)),
                qml.expval(qml.PauliZ(_qi) @ qml.PauliZ(_qj)),
            ]

        pair_circuits[(qi, qj)] = measure_pair

    return pair_circuits


# ============ FEATURE EXTRACTION ============

def compute_2body_features(
    X: np.ndarray,
    gamma_per_qubit: np.ndarray,
    entanglement_pairs: List[Tuple[int, int]],
    pairs_to_measure: List[Tuple[int, int]],
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute 2-body RDM feature vectors for all data points.

    For each data point x and each pair (i,j) in pairs_to_measure,
    measures the 15 non-trivial Pauli products:
        [⟨XI⟩, ⟨YI⟩, ⟨ZI⟩, ⟨IX⟩, ⟨IY⟩, ⟨IZ⟩,
         ⟨XX⟩, ⟨XY⟩, ⟨XZ⟩, ⟨YX⟩, ⟨YY⟩, ⟨YZ⟩, ⟨ZX⟩, ⟨ZY⟩, ⟨ZZ⟩]

    Args:
        X: Data matrix, shape (N, n_qubits). Values in [0, π].
        gamma_per_qubit: Per-qubit bandwidth array, shape (n_qubits,).
        entanglement_pairs: List of (int, int) for ZZ gate topology.
        pairs_to_measure: List of (int, int) tuples to extract features from.
        n_qubits: Number of qubits.
        reps: Feature map repetitions.
        save_path: Optional path to cache features (.npy).

    Returns:
        np.ndarray: Feature matrix, shape (N, 15 × |pairs_to_measure|).
                    Columns are ordered by pair, then by Pauli label.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached 2-body features: {save_path}")
        return np.load(save_path)

    N = len(X)
    n_pairs = len(pairs_to_measure)
    n_features = N_FEATURES_PER_PAIR * n_pairs

    logger.info(
        f"Computing 2-body features: {N} samples × {n_pairs} pairs "
        f"= {n_features} features ({N * n_pairs * N_FEATURES_PER_PAIR} "
        f"circuit evaluations)"
    )

    pair_circuits = _build_2body_circuits(
        gamma_per_qubit, entanglement_pairs, pairs_to_measure,
        n_qubits, reps,
    )

    features = np.zeros((N, n_features), dtype=np.float64)

    for idx in tqdm(range(N), desc="2-Body Features"):
        x = X[idx]
        for p_idx, (qi, qj) in enumerate(pairs_to_measure):
            circuit = pair_circuits[(qi, qj)]
            result = np.array(circuit(x), dtype=np.float64)
            col_start = p_idx * N_FEATURES_PER_PAIR
            col_end = col_start + N_FEATURES_PER_PAIR
            features[idx, col_start:col_end] = result

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, features)
        logger.info(f"Saved 2-body features: {save_path}")

    return features


def extract_1body_features(features_2body: np.ndarray, n_pairs: int) -> np.ndarray:
    """
    Extract ONLY the single-qubit (1-body) features from 2-body feature matrix.

    Returns the 6 single-qubit expectations per pair:
    [⟨XI⟩, ⟨YI⟩, ⟨ZI⟩, ⟨IX⟩, ⟨IY⟩, ⟨IZ⟩]

    Args:
        features_2body: Full 2-body feature matrix, shape (N, 15 × n_pairs).
        n_pairs: Number of measured pairs.

    Returns:
        np.ndarray: 1-body features, shape (N, 6 × n_pairs).
    """
    N = features_2body.shape[0]
    feats_1body = np.zeros((N, 6 * n_pairs), dtype=np.float64)
    for p in range(n_pairs):
        src_start = p * N_FEATURES_PER_PAIR
        dst_start = p * 6
        # First 6 columns of each pair block are 1-body
        feats_1body[:, dst_start:dst_start + 6] = (
            features_2body[:, src_start:src_start + 6]
        )
    return feats_1body


def extract_2body_correlations(features_2body: np.ndarray, n_pairs: int) -> np.ndarray:
    """
    Extract ONLY the 9 two-qubit correlation terms from 2-body feature matrix.

    Returns the 9 correlation expectations per pair:
    [⟨XX⟩, ⟨XY⟩, ⟨XZ⟩, ⟨YX⟩, ⟨YY⟩, ⟨YZ⟩, ⟨ZX⟩, ⟨ZY⟩, ⟨ZZ⟩]

    Args:
        features_2body: Full 2-body feature matrix, shape (N, 15 × n_pairs).
        n_pairs: Number of measured pairs.

    Returns:
        np.ndarray: 2-body correlation features, shape (N, 9 × n_pairs).
    """
    N = features_2body.shape[0]
    feats_corr = np.zeros((N, 9 * n_pairs), dtype=np.float64)
    for p in range(n_pairs):
        src_start = p * N_FEATURES_PER_PAIR + 6  # Skip first 6 (1-body)
        dst_start = p * 9
        feats_corr[:, dst_start:dst_start + 9] = (
            features_2body[:, src_start:src_start + 9]
        )
    return feats_corr


# ============ KERNEL COMPUTATION ============

def compute_2pqk_kernel(
    features: np.ndarray,
    features2: Optional[np.ndarray] = None,
    gamma: float = config.PQK_GAMMA,
    n_pairs: int = 4,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute the 2-PQK kernel matrix from 2-body feature vectors.

    K(x,x') = exp(-γ Σ_{(i,j)∈S} ‖ρ_{ij}(x) − ρ_{ij}(x')‖²_F)

    The Hilbert-Schmidt distance decomposes as:
    ‖ρ_{ij}(x) − ρ_{ij}(x')‖²_F = (1/4) Σ_{(a,b)≠(I,I)} (c_ab(x) − c_ab(x'))²

    Args:
        features: Train feature matrix, shape (N, 15 × n_pairs).
        features2: Optional test features, shape (M, 15 × n_pairs).
            If None, computes symmetric kernel K(X, X).
        gamma: RBF bandwidth parameter.
        n_pairs: Number of measured pairs (for block-wise distance).
        save_path: Optional path to save kernel matrix.

    Returns:
        np.ndarray: Kernel matrix, shape (N, N) or (M, N).
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached 2-PQK kernel: {save_path}")
        return np.load(save_path)

    N = len(features)
    symmetric = features2 is None
    f2 = features if symmetric else features2
    M = len(f2)

    logger.info(
        f"Computing 2-PQK kernel: {M}×{N} "
        f"({'symmetric' if symmetric else 'rectangular'}), γ={gamma:.4f}"
    )

    # Reshape into (N, n_pairs, 15) for block-wise Frobenius distance.
    f1_blocks = features.reshape(N, n_pairs, N_FEATURES_PER_PAIR)
    f2_blocks = f2.reshape(M, n_pairs, N_FEATURES_PER_PAIR)

    K = np.zeros((M, N), dtype=np.float64)

    for i in tqdm(range(M), desc="2-PQK Kernel"):
        # diff shape: (N, n_pairs, 15)
        diff = f1_blocks - f2_blocks[i]
        # Per-pair Hilbert-Schmidt distance: (1/4) Σ (c_ab - c'_ab)²
        # Factor 1/4 from Pauli decomposition normalization
        hs_dist_per_pair = 0.25 * np.sum(diff ** 2, axis=2)  # (N, n_pairs)
        total_dist = np.sum(hs_dist_per_pair, axis=1)  # (N,)
        K[i, :] = np.exp(-gamma * total_dist)

    # Enforce exact symmetry for symmetric case
    if symmetric:
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, K)
        logger.info(f"Saved 2-PQK kernel: {save_path}")

    return K


# ============ VALIDATION ============

def validate_2rdm(
    features_2body: np.ndarray,
    pair_idx: int = 0,
    sample_idx: int = 0,
    n_pairs: int = 4,
    tol: float = 1e-6,
) -> Dict[str, bool]:
    """
    Reconstruct and validate the 4×4 density matrix ρ_{ij} from 15 Pauli
    coefficients. Checks: Hermitian, Tr=1, PSD.

    Args:
        features_2body: Full feature matrix, shape (N, 15 × n_pairs).
        pair_idx: Which pair to validate (0-indexed).
        sample_idx: Which data point to validate.
        n_pairs: Number of pairs.
        tol: Numerical tolerance for validation checks.

    Returns:
        Dict with boolean checks: 'hermitian', 'trace_one', 'psd'.
    """
    # Extract 15 coefficients for the specified pair and sample
    start = pair_idx * N_FEATURES_PER_PAIR
    coeffs = features_2body[sample_idx, start:start + N_FEATURES_PER_PAIR]

    # Pauli matrices
    I2 = np.eye(2, dtype=complex)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = {"I": I2, "X": sx, "Y": sy, "Z": sz}

    # Map label to Pauli pair
    label_map = {
        "XI": ("X", "I"), "YI": ("Y", "I"), "ZI": ("Z", "I"),
        "IX": ("I", "X"), "IY": ("I", "Y"), "IZ": ("I", "Z"),
        "XX": ("X", "X"), "XY": ("X", "Y"), "XZ": ("X", "Z"),
        "YX": ("Y", "X"), "YY": ("Y", "Y"), "YZ": ("Y", "Z"),
        "ZX": ("Z", "X"), "ZY": ("Z", "Y"), "ZZ": ("Z", "Z"),
    }

    # Reconstruct ρ_{ij} = (1/4) [I⊗I + Σ c_{ab} σ_a⊗σ_b]
    rho = 0.25 * np.kron(I2, I2)  # II term (c_II = 1 by Tr normalization)
    for k, label in enumerate(PAULI_LABELS_2BODY):
        a_name, b_name = label_map[label]
        rho += 0.25 * coeffs[k] * np.kron(paulis[a_name], paulis[b_name])

    # Validation checks
    is_hermitian = np.allclose(rho, rho.conj().T, atol=tol)
    trace_val = np.real(np.trace(rho))
    is_trace_one = abs(trace_val - 1.0) < tol
    eigvals = np.real(np.linalg.eigvalsh(rho))
    is_psd = np.all(eigvals >= -tol)

    result = {
        "hermitian": bool(is_hermitian),
        "trace_one": bool(is_trace_one),
        "psd": bool(is_psd),
        "trace": float(trace_val),
        "min_eigenvalue": float(eigvals.min()),
        "max_eigenvalue": float(eigvals.max()),
    }

    logger.info(
        f"2-RDM validation (pair={pair_idx}, sample={sample_idx}): "
        f"Hermitian={is_hermitian}, Tr={trace_val:.6f}, "
        f"λ_min={eigvals.min():.6f}, PSD={is_psd}"
    )

    return result


# ============ POLYNOMIAL FEATURE BASELINE ============

def compute_polynomial_features(
    X: np.ndarray,
    pairs: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Compute classical polynomial interaction features for dequantization test.

    For each pair (i, j), computes:
        [x_i, x_j, x_i², x_j², x_i·x_j, (π−x_i)·(π−x_j),
         sin(x_i)·sin(x_j), cos(x_i)·cos(x_j), sin(x_i)·cos(x_j)]

    This provides a 9-dimensional classical analog of the 9 quantum
    correlation terms — specifically designed to test whether the
    quantum circuit's nonlinear pairwise features can be classically
    simulated by polynomial/trigonometric expansions.

    Args:
        X: Data matrix, shape (N, d). Columns indexed by qubit.
        pairs: List of (int, int) column pairs to compute features for.

    Returns:
        np.ndarray: Polynomial features, shape (N, 9 × |pairs|).
    """
    N = len(X)
    n_pairs = len(pairs)
    poly_feats = np.zeros((N, 9 * n_pairs), dtype=np.float64)

    for p, (qi, qj) in enumerate(pairs):
        xi = X[:, qi]
        xj = X[:, qj]
        base = p * 9
        poly_feats[:, base + 0] = xi
        poly_feats[:, base + 1] = xj
        poly_feats[:, base + 2] = xi ** 2
        poly_feats[:, base + 3] = xj ** 2
        poly_feats[:, base + 4] = xi * xj
        poly_feats[:, base + 5] = (np.pi - xi) * (np.pi - xj)
        poly_feats[:, base + 6] = np.sin(xi) * np.sin(xj)
        poly_feats[:, base + 7] = np.cos(xi) * np.cos(xj)
        poly_feats[:, base + 8] = np.sin(xi) * np.cos(xj)

    return poly_feats


# ============ SELF-TEST ============

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("  src/two_body_pqk.py — Self-test")
    print("=" * 70)

    np.random.seed(config.RANDOM_SEED)

    # Synthetic data: 10 samples, 8 qubits
    N_TEST = 10
    n_q = 8
    X_test = np.random.rand(N_TEST, n_q) * np.pi

    # Dummy topology (same as AGPQK defaults)
    gamma = np.full(n_q, 0.5)
    ent_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
    measure_pairs = ent_pairs

    # Compute features
    feats = compute_2body_features(
        X_test, gamma, ent_pairs, measure_pairs, n_qubits=n_q, reps=2,
    )
    print(f"\n  Feature shape: {feats.shape}")
    assert feats.shape == (N_TEST, 15 * len(measure_pairs)), (
        f"Expected ({N_TEST}, {15 * len(measure_pairs)}), got {feats.shape}"
    )

    # Validate 2-RDM
    for p_idx in range(len(measure_pairs)):
        val = validate_2rdm(feats, pair_idx=p_idx, sample_idx=0,
                            n_pairs=len(measure_pairs))
        assert val["hermitian"], f"Pair {p_idx}: NOT Hermitian!"
        assert val["trace_one"], f"Pair {p_idx}: Trace ≠ 1 (got {val['trace']})!"
        assert val["psd"], f"Pair {p_idx}: NOT PSD (λ_min={val['min_eigenvalue']})!"

    # Compute kernel
    K = compute_2pqk_kernel(feats, gamma=0.67, n_pairs=len(measure_pairs))
    print(f"  Kernel shape: {K.shape}")
    assert K.shape == (N_TEST, N_TEST)
    assert np.allclose(K, K.T, atol=1e-10), "Kernel not symmetric!"
    assert np.allclose(np.diag(K), 1.0, atol=1e-10), "Diagonal not 1!"
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-8), f"Kernel not PSD! min eigval={eigvals.min()}"

    # Test feature extraction helpers
    f1 = extract_1body_features(feats, len(measure_pairs))
    f2 = extract_2body_correlations(feats, len(measure_pairs))
    assert f1.shape == (N_TEST, 6 * len(measure_pairs))
    assert f2.shape == (N_TEST, 9 * len(measure_pairs))

    # Polynomial baseline
    poly = compute_polynomial_features(X_test, measure_pairs)
    assert poly.shape == (N_TEST, 9 * len(measure_pairs))

    print("\n  [OK] All self-tests passed.")
    print("=" * 70)
