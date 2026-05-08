"""
Quantum Entanglement-Aware Attention Projected Quantum Kernel (QEAA-PQK).

A novel two-pass quantum kernel that derives attention scores from the quantum
state's own entanglement structure — not from classical preprocessing.

Architecture:
    Pass 1 (Probe):
        - Encode training data via a shallow entangling circuit (depth-2).
        - Compute per-qubit reduced density matrices ρ_i via statevector
          partial trace.
        - Extract quantum-native attention:
            w_i   = S(ρ̄_i)         — von Neumann entropy → per-qubit importance
            I^Q_ij = S(ρ̄_i) + S(ρ̄_j) - S(ρ̄_ij)  — quantum MI → pair interaction

    Pass 2 (Full Kernel):
        - Set per-qubit bandwidth: γ_i ∝ w_i
        - Select entanglement pairs: top-k by I^Q_ij (cross-modal preferred)
        - Run ZZ feature map with data-dependent topology
        - Extract PQK Bloch vectors and compute RBF kernel

Key Novelty:
    The attention mechanism is QUANTUM-NATIVE — it arises from the reduced
    density matrix of the encoded quantum state, not from classical covariance.
    No existing published method (QKSAN, QSANN, TD-QAS) uses von Neumann
    entropy of the feature-map state to self-determine the circuit topology.

References:
    - von Neumann entropy: S(ρ) = -Tr(ρ log₂ ρ)
    - Quantum MI: I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)
    - PQK: Huang et al., Nature Communications 12, 2631 (2021)
    - ZZFeatureMap: Havlíček et al., Nature 567, 209 (2019)

Author: Prathamesh Kadam et al.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)

# ============ CONSTANTS ============

# Modality tags for physics features (same as attention_kernel.py)
FEATURE_MODALITY_16 = [
    "OPT", "OPT", "OPT", "OPT", "OPT", "OPT", "OPT", "OPT",
    "SAR", "SAR", "SAR", "SAR", "SAR", "SAR", "SAR", "SAR",
]

FEATURE_NAMES_16 = [
    "NDVI", "NDBI", "NDWI", "BSI", "SAVI", "NDRE", "MNDWI", "EVI",
    "VV/VH", "SAR_total", "CrossPol", "PolCoh",
    "VH_dB", "VV_dB", "VV_tex", "NDVI_tex",
]


# ============ PASS 1: QUANTUM PROBE ============

def _build_probe_statevector(n_qubits: int = config.N_QUBITS):
    """
    Build the probe circuit that returns the full statevector.

    The probe uses a minimal entangling circuit (depth-2):
        Layer 1: H → RY(x_i) on each qubit (product state encoding)
        Layer 2: Nearest-neighbor CNOT layer (introduces entanglement)

    This is NOT a simple product state — the CNOT layer creates data-dependent
    entanglement, ensuring non-trivial quantum mutual information I^Q_ij.
    Without the CNOT layer, all I^Q_ij = 0 (product states have zero QMI).

    Args:
        n_qubits: Number of qubits.

    Returns:
        QNode returning the full statevector (2^n complex amplitudes).
    """
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, diff_method=None)
    def probe_circuit(x):
        # Layer 1: Product-state encoding
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
            qml.RY(x[i], wires=i)

        # Layer 2: Nearest-neighbor CNOT entanglement (even pairs)
        for i in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])

        # Layer 3: Nearest-neighbor CNOT entanglement (odd pairs)
        # This creates a full chain of entanglement across all qubits.
        for i in range(1, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])

        return qml.state()

    return probe_circuit


def _partial_trace_single_qubit(
    statevector: np.ndarray, qubit_idx: int, n_qubits: int
) -> np.ndarray:
    """
    Compute the single-qubit reduced density matrix ρ_i by partial trace.

    Given the full statevector |ψ⟩ of n qubits, computes:
        ρ_i = Tr_{all qubits except i}(|ψ⟩⟨ψ|)

    This is done efficiently by reshaping the statevector into a tensor
    and contracting over all indices except qubit i.

    Args:
        statevector: Full statevector, shape (2^n,), complex.
        qubit_idx: Index of the qubit to keep (0-indexed).
        n_qubits: Total number of qubits.

    Returns:
        np.ndarray: 2×2 reduced density matrix, complex.
    """
    # Reshape statevector into rank-n tensor: (2, 2, ..., 2)
    psi = statevector.reshape([2] * n_qubits)

    # Full density matrix as outer product: ρ = |ψ⟩⟨ψ|
    # In tensor form: ρ[i0,i1,...,j0,j1,...] = psi[i0,i1,...] * conj(psi[j0,j1,...])
    rho_full = np.tensordot(psi, np.conj(psi), axes=0)
    # Shape: (2,2,...,2,  2,2,...,2) — 2n indices

    # Partial trace: contract all qubit pairs except qubit_idx
    # rho_full has indices (i_0,...,i_{n-1}, j_0,...,j_{n-1})
    # We need to trace over all k != qubit_idx: set i_k = j_k and sum
    axes_to_trace = []
    for k in range(n_qubits):
        if k != qubit_idx:
            axes_to_trace.append(k)

    # Sort in reverse order to avoid index shifting during contraction
    rho_reduced = rho_full
    offset = 0
    for k in sorted(axes_to_trace, reverse=True):
        # Trace over axis k (bra side) and k+n_qubits-offset (ket side)
        bra_axis = k - (n_qubits - rho_reduced.ndim // 2 - offset)
        # Simpler: use np.trace along the two matching axes
        ket_axis = bra_axis + rho_reduced.ndim // 2
        rho_reduced = np.trace(rho_reduced, axis1=bra_axis, axis2=ket_axis)
        offset += 1

    return rho_reduced


def _partial_trace_two_qubits(
    statevector: np.ndarray, q_i: int, q_j: int, n_qubits: int
) -> np.ndarray:
    """
    Compute the two-qubit reduced density matrix ρ_{ij} by partial trace.

    Given the full statevector |ψ⟩, computes:
        ρ_{ij} = Tr_{all qubits except i,j}(|ψ⟩⟨ψ|)

    Args:
        statevector: Full statevector, shape (2^n,), complex.
        q_i: First qubit index.
        q_j: Second qubit index (must differ from q_i).
        n_qubits: Total number of qubits.

    Returns:
        np.ndarray: 4×4 reduced density matrix, complex.
    """
    assert q_i != q_j, f"q_i and q_j must differ, got {q_i} and {q_j}"

    psi = statevector.reshape([2] * n_qubits)
    rho_full = np.tensordot(psi, np.conj(psi), axes=0)

    keep = {q_i, q_j}
    axes_to_trace = [k for k in range(n_qubits) if k not in keep]

    rho_reduced = rho_full
    offset = 0
    for k in sorted(axes_to_trace, reverse=True):
        bra_axis = k - (n_qubits - rho_reduced.ndim // 2 - offset)
        ket_axis = bra_axis + rho_reduced.ndim // 2
        rho_reduced = np.trace(rho_reduced, axis1=bra_axis, axis2=ket_axis)
        offset += 1

    # Result shape: (2, 2, 2, 2) — reshape to (4, 4)
    return rho_reduced.reshape(4, 4)


def _von_neumann_entropy(rho: np.ndarray, eps: float = 1e-15) -> float:
    """
    Compute the von Neumann entropy S(ρ) = -Tr(ρ log₂ ρ).

    Uses eigenvalue decomposition for numerical stability.
    Eigenvalues < eps are clamped to avoid log(0).

    Args:
        rho: Density matrix (Hermitian, PSD, Tr=1).
        eps: Minimum eigenvalue floor.

    Returns:
        float: Von Neumann entropy in bits (log base 2). Range [0, log₂(d)].
    """
    eigenvalues = np.real(np.linalg.eigvalsh(rho))
    eigenvalues = np.clip(eigenvalues, eps, None)
    # Renormalize to handle numerical drift
    eigenvalues = eigenvalues / eigenvalues.sum()
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def compute_quantum_attention(
    X_train: np.ndarray,
    n_qubits: int = config.N_QUBITS,
    save_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    PASS 1: Compute quantum-native attention from probe circuit.

    Encodes all training samples through a shallow entangling probe circuit,
    computes average reduced density matrices, and extracts:
        - w_i:     per-qubit von Neumann entropy (feature importance)
        - I^Q_ij:  pairwise quantum mutual information (interaction strength)

    All computations use TRAINING DATA ONLY — no labels, no test leakage.

    Args:
        X_train: Training feature matrix, shape (N, n_qubits). Values in [0, π].
        n_qubits: Number of qubits (default 8).
        save_path: Optional path to cache the attention results (.json).

    Returns:
        dict with keys:
            'w':       np.ndarray, shape (n_qubits,) — per-qubit importance
            'I_Q':     np.ndarray, shape (n_qubits, n_qubits) — pairwise QMI
            'rho_avg': list of average single-qubit RDMs (for diagnostics)
    """
    # Check cache
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached quantum attention: {save_path}")
        with open(save_path) as f:
            cached = json.load(f)
        return {
            "w": np.array(cached["w"]),
            "I_Q": np.array(cached["I_Q"]),
        }

    probe = _build_probe_statevector(n_qubits)
    N = len(X_train)

    logger.info(f"QEAA Pass 1: Computing quantum attention from {N} training samples")
    logger.info(f"  Probe circuit: H + RY + 2-layer CNOT, {n_qubits} qubits")

    # Accumulate average RDMs across all training samples
    rho_avg_single = [np.zeros((2, 2), dtype=np.complex128) for _ in range(n_qubits)]
    rho_avg_pair = {}
    for qi in range(n_qubits):
        for qj in range(qi + 1, n_qubits):
            rho_avg_pair[(qi, qj)] = np.zeros((4, 4), dtype=np.complex128)

    for idx in tqdm(range(N), desc="QEAA Probe"):
        sv = np.array(probe(X_train[idx]), dtype=np.complex128)

        # Single-qubit RDMs
        for qi in range(n_qubits):
            rho_i = _partial_trace_single_qubit(sv, qi, n_qubits)
            rho_avg_single[qi] += rho_i / N

        # Two-qubit RDMs
        for qi in range(n_qubits):
            for qj in range(qi + 1, n_qubits):
                rho_ij = _partial_trace_two_qubits(sv, qi, qj, n_qubits)
                rho_avg_pair[(qi, qj)] += rho_ij / N

    # Compute von Neumann entropy for each qubit
    w = np.zeros(n_qubits)
    for qi in range(n_qubits):
        w[qi] = _von_neumann_entropy(rho_avg_single[qi])

    logger.info(f"  Per-qubit von Neumann entropy (w): {np.array2string(w, precision=4)}")

    # Compute quantum mutual information for each pair
    I_Q = np.zeros((n_qubits, n_qubits))
    for qi in range(n_qubits):
        for qj in range(qi + 1, n_qubits):
            S_ij = _von_neumann_entropy(rho_avg_pair[(qi, qj)])
            I_Q[qi, qj] = w[qi] + w[qj] - S_ij
            I_Q[qj, qi] = I_Q[qi, qj]  # Symmetric

    logger.info(f"  QMI matrix I^Q range: [{I_Q.min():.6f}, {I_Q.max():.6f}]")
    logger.info(f"  QMI mean off-diagonal: {I_Q[I_Q > 0].mean():.6f}")

    # Cache results
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cache_data = {
            "w": w.tolist(),
            "I_Q": I_Q.tolist(),
            "n_qubits": n_qubits,
            "n_train": N,
        }
        with open(save_path, "w") as f:
            json.dump(cache_data, f, indent=2)
        logger.info(f"  Saved quantum attention: {save_path}")

    return {"w": w, "I_Q": I_Q}


# ============ DATA-DEPENDENT TOPOLOGY ============

def select_qeaa_topology(
    w: np.ndarray,
    I_Q: np.ndarray,
    gamma_base: float = 0.50,
    gamma_min: float = 0.20,
    gamma_max: float = 0.80,
    top_k_pairs: int = 4,
    require_cross_modal: bool = True,
    selected_indices: Optional[np.ndarray] = None,
    max_appearances: int = 2,
) -> Dict:
    """
    Derive data-dependent circuit topology from quantum attention scores.

    Uses the von Neumann entropy (per-qubit importance) and quantum mutual
    information (pairwise interaction) to set:
        1. Per-qubit bandwidth γ_i ∝ w_i (higher entropy → wider encoding range)
        2. Entanglement pairs: top-k pairs by I^Q_ij

    Args:
        w: Per-qubit importance vector from von Neumann entropy, shape (n_qubits,).
        I_Q: Quantum mutual information matrix, shape (n_qubits, n_qubits).
        gamma_base: Base bandwidth for per-qubit modulation.
        gamma_min: Minimum per-qubit bandwidth (safety floor).
        gamma_max: Maximum per-qubit bandwidth (concentration ceiling).
        top_k_pairs: Number of entanglement pairs to select.
        require_cross_modal: If True, prefer cross-modal (SAR↔OPT) pairs.
        selected_indices: If provided, maps qubit indices to feature indices
                          for cross-modal checking. Shape (n_qubits,).
        max_appearances: Max times any qubit can appear across selected pairs.

    Returns:
        dict with keys:
            'gamma_per_qubit': np.ndarray, shape (n_qubits,)
            'entanglement_pairs': list of (int, int) tuples in qubit space
            'w': per-qubit importance
            'I_Q': quantum mutual information matrix
    """
    n_qubits = len(w)

    # Per-qubit bandwidth: γ_i = γ_base × (w_i / mean(w))
    w_safe = np.clip(w, 1e-10, None)
    w_normalized = w_safe / (w_safe.mean() + 1e-12)
    gamma_per_qubit = gamma_base * w_normalized
    gamma_per_qubit = np.clip(gamma_per_qubit, gamma_min, gamma_max)

    logger.info(f"  QEAA per-qubit bandwidths: {np.array2string(gamma_per_qubit, precision=4)}")

    # Entanglement pair selection by descending I^Q_ij
    candidates = []
    for qi in range(n_qubits):
        for qj in range(qi + 1, n_qubits):
            is_cross_modal = False
            if selected_indices is not None and require_cross_modal:
                fi, fj = selected_indices[qi], selected_indices[qj]
                if fi < len(FEATURE_MODALITY_16) and fj < len(FEATURE_MODALITY_16):
                    mod_i = FEATURE_MODALITY_16[fi]
                    mod_j = FEATURE_MODALITY_16[fj]
                    is_cross_modal = (mod_i != mod_j)

            candidates.append((qi, qj, float(I_Q[qi, qj]), is_cross_modal))

    # Sort: cross-modal first (if required), then by I^Q descending
    if require_cross_modal:
        cross_modal = [(qi, qj, iq, cm) for qi, qj, iq, cm in candidates if cm]
        same_modal = [(qi, qj, iq, cm) for qi, qj, iq, cm in candidates if not cm]
        cross_modal.sort(key=lambda x: x[2], reverse=True)
        same_modal.sort(key=lambda x: x[2], reverse=True)
        sorted_candidates = cross_modal + same_modal
    else:
        sorted_candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

    # Greedy selection with diversity constraint
    pairs = []
    appearances = {}
    for qi, qj, iq_val, _ in sorted_candidates:
        if len(pairs) >= top_k_pairs:
            break
        count_i = appearances.get(qi, 0)
        count_j = appearances.get(qj, 0)
        if count_i < max_appearances and count_j < max_appearances:
            pairs.append((qi, qj))
            appearances[qi] = count_i + 1
            appearances[qj] = count_j + 1

    logger.info(f"  QEAA entanglement pairs (top-{top_k_pairs}): {pairs}")
    for qi, qj in pairs:
        logger.info(f"    qubit {qi} ↔ qubit {qj}: I^Q = {I_Q[qi, qj]:.6f}")

    return {
        "gamma_per_qubit": gamma_per_qubit,
        "entanglement_pairs": pairs,
        "w": w,
        "I_Q": I_Q,
    }


# ============ PASS 2: QEAA FEATURE MAP + PQK KERNEL ============

def _build_qeaa_pqk_circuits(
    gamma_per_qubit: np.ndarray,
    entanglement_pairs: List[Tuple[int, int]],
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
):
    """
    Build the QEAA-PQK measurement circuits with data-dependent topology.

    The feature map uses:
        - Per-qubit bandwidth γ_i (from quantum attention)
        - Data-dependent entanglement pairs (from quantum mutual information)

    Circuit structure per repetition:
        1. H on all qubits
        2. RZ(γ_i × x_i) on qubit i
        3. For each selected pair (i, j):
           CNOT(i,j) → RZ((π - γ_i·x_i)(π - γ_j·x_j)) → CNOT(i,j)

    Args:
        gamma_per_qubit: Per-qubit bandwidth array, shape (n_qubits,).
        entanglement_pairs: List of (int, int) tuples for ZZ interactions.
        n_qubits: Number of qubits.
        reps: Number of feature map repetitions.

    Returns:
        Tuple of QNodes (measure_x, measure_y, measure_z) for Bloch vectors.
    """
    import pennylane as qml

    dev = config.get_device(n_qubits)

    def apply_qeaa_feature_map(x):
        """Apply the QEAA feature map with data-dependent topology."""
        for _ in range(reps):
            # Hadamard layer
            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            # Per-qubit bandwidth-modulated encoding
            for i in range(n_qubits):
                qml.RZ(gamma_per_qubit[i] * x[i], wires=i)

            # Attention-selected ZZ interactions
            for qi, qj in entanglement_pairs:
                if qi < n_qubits and qj < n_qubits:
                    zz_angle = (
                        (np.pi - gamma_per_qubit[qi] * x[qi])
                        * (np.pi - gamma_per_qubit[qj] * x[qj])
                    )
                    qml.CNOT(wires=[qi, qj])
                    qml.RZ(zz_angle, wires=qj)
                    qml.CNOT(wires=[qi, qj])

    @qml.qnode(dev, diff_method=None)
    def measure_x(x):
        apply_qeaa_feature_map(x)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def measure_y(x):
        apply_qeaa_feature_map(x)
        return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def measure_z(x):
        apply_qeaa_feature_map(x)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return measure_x, measure_y, measure_z


def compute_qeaa_bloch_vectors(
    X: np.ndarray,
    gamma_per_qubit: np.ndarray,
    entanglement_pairs: List[Tuple[int, int]],
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute PQK Bloch vectors using the QEAA data-dependent feature map.

    For each data point x, returns 3 × n_qubits values:
    [⟨X₀⟩, ⟨Y₀⟩, ⟨Z₀⟩, ⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩, ...].

    Args:
        X: Data matrix, shape (N, n_qubits).
        gamma_per_qubit: Per-qubit bandwidths from quantum attention.
        entanglement_pairs: Entanglement pairs from quantum attention.
        n_qubits: Number of qubits.
        reps: Feature map repetitions.
        save_path: Path to cache Bloch vectors.

    Returns:
        np.ndarray: Bloch vectors, shape (N, 3 * n_qubits).
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached QEAA Bloch vectors: {save_path}")
        return np.load(save_path)

    mx, my, mz = _build_qeaa_pqk_circuits(
        gamma_per_qubit, entanglement_pairs, n_qubits, reps
    )

    N = len(X)
    bloch = np.zeros((N, 3 * n_qubits), dtype=np.float64)

    logger.info(f"QEAA Pass 2: Computing Bloch vectors for {N} samples")

    for i in tqdm(range(N), desc="QEAA Bloch"):
        ex = np.array(mx(X[i]))
        ey = np.array(my(X[i]))
        ez = np.array(mz(X[i]))
        for q in range(n_qubits):
            bloch[i, 3 * q] = ex[q]
            bloch[i, 3 * q + 1] = ey[q]
            bloch[i, 3 * q + 2] = ez[q]

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, bloch)
        logger.info(f"  Saved QEAA Bloch vectors: {save_path}")

    return bloch


def compute_qeaa_pqk_kernel(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    gamma_kernel: float = config.PQK_GAMMA,
    gamma_per_qubit: Optional[np.ndarray] = None,
    entanglement_pairs: Optional[List[Tuple[int, int]]] = None,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_path: Optional[str] = None,
    bloch_save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute the full QEAA-PQK kernel matrix.

    K_QEAA(x, x') = exp(-γ_kernel Σ_i ||ρ_i(x) - ρ_i(x')||²_F)

    where ρ_i is the single-qubit PQK projection, and the feature map uses
    the QEAA data-dependent topology (per-qubit bandwidth + selected pairs).

    Args:
        X: Training data, shape (N, n_qubits).
        X2: Optional test data, shape (M, n_qubits). If None, compute K(X,X).
        gamma_kernel: Global RBF bandwidth for the PQK kernel.
        gamma_per_qubit: Per-qubit bandwidths from quantum attention.
        entanglement_pairs: Entanglement pairs from quantum attention.
        n_qubits: Number of qubits.
        reps: Feature map repetitions.
        save_path: Path to save kernel matrix.
        bloch_save_path: Path to cache Bloch vectors.

    Returns:
        np.ndarray: Kernel matrix, shape (N, N) or (M, N).
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached QEAA-PQK kernel: {save_path}")
        return np.load(save_path)

    if gamma_per_qubit is None or entanglement_pairs is None:
        raise ValueError(
            "gamma_per_qubit and entanglement_pairs are required. "
            "Run compute_quantum_attention() and select_qeaa_topology() first."
        )

    # Compute Bloch vectors with QEAA feature map
    bloch1 = compute_qeaa_bloch_vectors(
        X, gamma_per_qubit, entanglement_pairs, n_qubits, reps,
        save_path=bloch_save_path,
    )

    if X2 is not None:
        bloch2_path = (
            bloch_save_path.replace(".npy", "_test.npy") if bloch_save_path else None
        )
        bloch2 = compute_qeaa_bloch_vectors(
            X2, gamma_per_qubit, entanglement_pairs, n_qubits, reps,
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
        logger.info(f"Computing QEAA-PQK kernel: {n_train}×{n_train} (symmetric)")
        K = np.zeros((n_train, n_train), dtype=np.float64)
        for i in tqdm(range(n_train), desc="QEAA-PQK Kernel"):
            diff = b1[i] - b2  # (n_train, n_qubits, 3)
            sq_diff = np.sum(diff ** 2, axis=2)  # (n_train, n_qubits)
            frob_dist = 0.5 * np.sum(sq_diff, axis=1)  # (n_train,)
            K[i, :] = np.exp(-gamma_kernel * frob_dist)
    else:
        logger.info(f"Computing QEAA-PQK kernel: {n_test}×{n_train} (test vs train)")
        K = np.zeros((n_test, n_train), dtype=np.float64)
        for i in tqdm(range(n_test), desc="QEAA-PQK Kernel (rect)"):
            diff = b2[i] - b1  # (n_train, n_qubits, 3)
            sq_diff = np.sum(diff ** 2, axis=2)  # (n_train, n_qubits)
            frob_dist = 0.5 * np.sum(sq_diff, axis=1)  # (n_train,)
            K[i, :] = np.exp(-gamma_kernel * frob_dist)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, K)
        logger.info(f"Saved QEAA-PQK kernel: {save_path}")

    return K


# ============ HIGH-LEVEL API ============

def compute_qeaa_pqk_full_pipeline(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    selected_indices: Optional[np.ndarray] = None,
    gamma_base: float = 0.50,
    gamma_kernel: float = config.PQK_GAMMA,
    top_k_pairs: int = 4,
    require_cross_modal: bool = True,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Run the complete QEAA-PQK pipeline: probe → attention → kernel.

    This is the primary entry point for using QEAA-PQK.

    Args:
        X_train: Training data, shape (N, n_qubits). Values in [0, π].
        X_test: Optional test data, shape (M, n_qubits).
        y_train: Training labels (unused by kernel — for downstream SVM only).
        selected_indices: Feature indices mapping qubits to original features.
                          Required for cross-modal pair selection.
        gamma_base: Base bandwidth for per-qubit modulation.
        gamma_kernel: Global RBF bandwidth for PQK kernel.
        top_k_pairs: Number of entanglement pairs.
        require_cross_modal: Prefer cross-modal pairs if selected_indices given.
        n_qubits: Number of qubits.
        reps: Feature map repetitions.
        save_dir: Directory to cache intermediate results.

    Returns:
        dict with keys:
            'K_train': np.ndarray, shape (N, N)
            'K_test': np.ndarray or None, shape (M, N)
            'attention': dict with 'w', 'I_Q'
            'topology': dict with 'gamma_per_qubit', 'entanglement_pairs'
    """
    # Paths for caching
    attn_path = os.path.join(save_dir, "qeaa_attention.json") if save_dir else None
    k_train_path = os.path.join(save_dir, "K_qeaa_train.npy") if save_dir else None
    k_test_path = os.path.join(save_dir, "K_qeaa_test.npy") if save_dir else None
    bloch_path = os.path.join(save_dir, "qeaa_bloch_train.npy") if save_dir else None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Pass 1: Quantum probe → attention scores
    attention = compute_quantum_attention(X_train, n_qubits, save_path=attn_path)

    # Derive topology from attention scores
    topology = select_qeaa_topology(
        w=attention["w"],
        I_Q=attention["I_Q"],
        gamma_base=gamma_base,
        top_k_pairs=top_k_pairs,
        require_cross_modal=require_cross_modal and (selected_indices is not None),
        selected_indices=selected_indices,
    )

    # Pass 2: Compute QEAA-PQK kernel
    K_train = compute_qeaa_pqk_kernel(
        X_train, X2=None,
        gamma_kernel=gamma_kernel,
        gamma_per_qubit=topology["gamma_per_qubit"],
        entanglement_pairs=topology["entanglement_pairs"],
        n_qubits=n_qubits, reps=reps,
        save_path=k_train_path,
        bloch_save_path=bloch_path,
    )

    K_test = None
    if X_test is not None:
        K_test = compute_qeaa_pqk_kernel(
            X_train, X2=X_test,
            gamma_kernel=gamma_kernel,
            gamma_per_qubit=topology["gamma_per_qubit"],
            entanglement_pairs=topology["entanglement_pairs"],
            n_qubits=n_qubits, reps=reps,
            save_path=k_test_path,
            bloch_save_path=bloch_path,
        )

    # Save topology config
    if save_dir:
        topo_path = os.path.join(save_dir, "qeaa_topology.json")
        topo_save = {
            "gamma_per_qubit": topology["gamma_per_qubit"].tolist(),
            "entanglement_pairs": topology["entanglement_pairs"],
            "w": topology["w"].tolist(),
            "gamma_base": gamma_base,
            "gamma_kernel": gamma_kernel,
            "top_k_pairs": top_k_pairs,
        }
        with open(topo_path, "w") as f:
            json.dump(topo_save, f, indent=2)
        logger.info(f"  Saved QEAA topology: {topo_path}")

    return {
        "K_train": K_train,
        "K_test": K_test,
        "attention": attention,
        "topology": topology,
    }


def print_qeaa_summary(attention: Dict, topology: Dict,
                        selected_indices: Optional[np.ndarray] = None) -> None:
    """Print a human-readable summary of QEAA attention and topology."""
    w = attention["w"]
    I_Q = attention["I_Q"]
    gamma = topology["gamma_per_qubit"]
    pairs = topology["entanglement_pairs"]

    print("\n" + "=" * 70)
    print("  QEAA-PQK: Quantum Entanglement-Aware Attention Summary")
    print("=" * 70)

    print(f"\n  Per-qubit von Neumann entropy (quantum importance):")
    print(f"  {'Qubit':>6s}  {'Feature':>12s}  {'Modality':>8s}  {'S(ρ_i)':>10s}  {'γ_i':>8s}")
    print("  " + "-" * 52)
    for qi in range(len(w)):
        if selected_indices is not None and qi < len(selected_indices):
            fi = selected_indices[qi]
            fname = FEATURE_NAMES_16[fi] if fi < len(FEATURE_NAMES_16) else f"feat_{fi}"
            fmod = FEATURE_MODALITY_16[fi] if fi < len(FEATURE_MODALITY_16) else "?"
        else:
            fname = f"qubit_{qi}"
            fmod = "?"
        print(f"  {qi:>6d}  {fname:>12s}  {fmod:>8s}  {w[qi]:>10.6f}  {gamma[qi]:>8.4f}")

    print(f"\n  Selected entanglement pairs (by quantum mutual information):")
    for qi, qj in pairs:
        print(f"    qubit {qi} ↔ qubit {qj}:  I^Q = {I_Q[qi, qj]:.6f}")

    print()


# ============ SELF-TEST ============

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("  src/qeaa_kernel.py — Self-test")
    print("=" * 70)

    np.random.seed(config.RANDOM_SEED)

    # Synthetic 8-feature data
    N = 30
    n_q = 8
    X = np.random.rand(N, n_q) * np.pi

    # Pass 1: Quantum attention
    attn = compute_quantum_attention(X, n_qubits=n_q)
    print(f"\n  Von Neumann entropy per qubit: {attn['w']}")
    print(f"  QMI matrix diagonal (should be 0): {np.diag(attn['I_Q'])}")
    print(f"  QMI max off-diagonal: {attn['I_Q'].max():.6f}")

    assert attn["w"].shape == (n_q,), f"Expected ({n_q},), got {attn['w'].shape}"
    assert attn["I_Q"].shape == (n_q, n_q), f"Expected ({n_q},{n_q}), got {attn['I_Q'].shape}"
    assert np.all(attn["w"] >= 0), "Entropy must be non-negative"

    # Topology selection
    topo = select_qeaa_topology(
        attn["w"], attn["I_Q"], top_k_pairs=4, require_cross_modal=False
    )
    print(f"\n  Per-qubit bandwidth: {topo['gamma_per_qubit']}")
    print(f"  Entanglement pairs: {topo['entanglement_pairs']}")

    assert len(topo["gamma_per_qubit"]) == n_q
    assert len(topo["entanglement_pairs"]) == 4

    # Pass 2: QEAA-PQK kernel
    K = compute_qeaa_pqk_kernel(
        X, gamma_kernel=0.67,
        gamma_per_qubit=topo["gamma_per_qubit"],
        entanglement_pairs=topo["entanglement_pairs"],
        n_qubits=n_q,
    )

    print(f"\n  Kernel shape: {K.shape}")
    print(f"  Kernel diagonal: [{K.diagonal().min():.4f}, {K.diagonal().max():.4f}]")
    print(f"  Kernel off-diagonal mean: {K[np.triu_indices(N, k=1)].mean():.4f}")
    print(f"  Kernel range: [{K.min():.4f}, {K.max():.4f}]")

    assert K.shape == (N, N)
    assert np.allclose(K, K.T, atol=1e-8), "Kernel must be symmetric"
    assert K.min() >= -1e-6, "Kernel values must be non-negative"

    print("\n  All self-tests passed.")
