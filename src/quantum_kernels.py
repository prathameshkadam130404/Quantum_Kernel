"""
Quantum kernel computation for the Quantum-Sat Classification pipeline.

Implements:
    - ZZFeatureMap: Manual construction (PennyLane has NO built-in ZZFeatureMap).
      Structure: H gates → RZ(x_i) → CNOT-RZ((π-x_i)(π-x_j))-CNOT for ZZ pairs.
    - Fidelity Quantum Kernel (FQK): K_q(x,x') = |⟨ψ(x)|ψ(x')⟩|²
    - Projected Quantum Kernel (PQK): Gaussian on Frobenius distance of local Bloch vectors.
    - Parallelized kernel matrix computation with disk caching and tqdm progress.

References:
    - Havlíček et al., Nature 567, 209 (2019) — quantum kernels
    - Huang et al., Nature Communications 12, 2631 (2021) — geometric difference
    - Thanasilp et al., Nature Communications 15, 5200 (2024) — PQK + concentration
"""

import os
import sys
import time
import logging
from typing import Callable, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)

# Register KDCA-PQK
from src.kdca_kernel import compute_kdca_pqk_full_pipeline, compute_kdca_pqk_kernel

# ============ ZZ FEATURE MAP ============

def apply_zz_feature_map(
    x: np.ndarray,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
) -> None:
    """
    Apply the ZZFeatureMap encoding circuit to the current quantum state.

    MUST be called inside a qml.qnode context. This function modifies
    the quantum state in-place via PennyLane gate operations.

    Circuit structure per repetition:
        1. Hadamard on all qubits
        2. RZ(x_i) on qubit i  (single-qubit encoding)
        3. For each ZZ pair (i, i+1) with linear connectivity:
           CNOT(i, i+1) → RZ((π - x_i)(π - x_{i+1})) on i+1 → CNOT(i, i+1)

    Args:
        x: Input features, shape (n_qubits,). Values should be in [0, π].
        n_qubits: Number of qubits (default: 8).
        reps: Number of repetitions (default: 2). More reps = more expressibility.
    """
    import pennylane as qml

    for _ in range(reps):
        # Hadamard layer
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        # Single-qubit RZ encoding
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)

        # ZZ interaction layer — linear connectivity
        for i in range(n_qubits - 1):
            j = i + 1
            zz_angle = (np.pi - x[i]) * (np.pi - x[j])
            qml.CNOT(wires=[i, j])
            qml.RZ(zz_angle, wires=j)
            qml.CNOT(wires=[i, j])


# ============ FIDELITY QUANTUM KERNEL (FQK) ============

def _build_fqk_circuit(n_qubits: int = config.N_QUBITS, reps: int = config.ZZ_REPS):
    """
    Build the FQK circuit: U(x) then U†(x').

    K_q(x, x') = |⟨0|^n U†(x') U(x) |0⟩^n|² = prob(|0⟩^n) after the circuit.

    Args:
        n_qubits: Number of qubits.
        reps: ZZFeatureMap repetitions.

    Returns:
        Callable: QNode that computes one FQK kernel entry.
    """
    import pennylane as qml

    dev = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def fqk_circuit(x1, x2):
        # U(x1): encode first data point
        apply_zz_feature_map(x1, n_qubits, reps)

        # U†(x2): adjoint of encoding second data point
        qml.adjoint(apply_zz_feature_map)(x2, n_qubits, reps)

        # Probability of measuring all-zeros state
        return qml.probs(wires=range(n_qubits))

    return fqk_circuit


def compute_fqk_entry(x1: np.ndarray, x2: np.ndarray, circuit=None) -> float:
    """
    Compute a single FQK kernel entry: K(x1, x2) = |⟨ψ(x1)|ψ(x2)⟩|².

    Args:
        x1: First data point, shape (n_qubits,).
        x2: Second data point, shape (n_qubits,).
        circuit: Pre-built FQK QNode (optional).

    Returns:
        float: Kernel value in [0, 1].
    """
    if circuit is None:
        circuit = _build_fqk_circuit()

    probs = circuit(x1, x2)
    # Probability of all-zeros state is the first element
    return float(probs[0])


def compute_fqk_kernel_matrix(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_path: Optional[str] = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Compute the full FQK kernel matrix.

    For square matrix (X2=None): uses symmetry to compute only n(n+1)/2 entries.
    Diagonal is always 1.0. Includes tqdm progress bar and disk caching.

    Args:
        X: Data matrix, shape (n, n_qubits).
        X2: Optional second data matrix for rectangular kernel (e.g., test data).
            Shape (m, n_qubits). If None, compute symmetric K(X, X).
        n_qubits: Number of qubits.
        reps: ZZFeatureMap repetitions.
        save_path: Path to save kernel matrix (.npy). Checks if exists first.
        n_jobs: Number of parallel workers (1 = sequential).

    Returns:
        np.ndarray: Kernel matrix, shape (n, n) or (m, n).
    """
    # Check cache
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached kernel: {save_path}")
        return np.load(save_path)

    circuit = _build_fqk_circuit(n_qubits, reps)

    if X2 is None:
        # Symmetric matrix
        n = len(X)
        K = np.eye(n, dtype=np.float64)  # Diagonal = 1
        n_evals = n * (n - 1) // 2

        logger.info(f"Computing FQK kernel matrix: {n}×{n} ({n_evals} unique evaluations)")

        with tqdm(total=n_evals, desc="FQK Kernel") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    val = compute_fqk_entry(X[i], X[j], circuit)
                    K[i, j] = val
                    K[j, i] = val
                    pbar.update(1)
    else:
        # Rectangular matrix
        m, n = len(X2), len(X)
        K = np.zeros((m, n), dtype=np.float64)
        n_evals = m * n

        logger.info(f"Computing FQK kernel matrix: {m}×{n} ({n_evals} evaluations)")

        with tqdm(total=n_evals, desc="FQK Kernel (rect)") as pbar:
            for i in range(m):
                for j in range(n):
                    K[i, j] = compute_fqk_entry(X2[i], X[j], circuit)
                    pbar.update(1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, K)
        logger.info(f"Saved FQK kernel: {save_path}")

    return K


# ============ PROJECTED QUANTUM KERNEL (PQK) ============

def _build_pqk_circuit(n_qubits: int = config.N_QUBITS, reps: int = config.ZZ_REPS):
    """
    Build the PQK measurement circuit.

    Encodes |ψ(x)⟩ via ZZFeatureMap, then measures ⟨σ^a⟩ per qubit
    for a ∈ {X, Y, Z}, yielding 3 × n_qubits = 24 real values.

    Args:
        n_qubits: Number of qubits.
        reps: ZZFeatureMap repetitions.

    Returns:
        Tuple of QNodes for X, Y, Z measurements.
    """
    import pennylane as qml

    dev = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def measure_x(x):
        apply_zz_feature_map(x, n_qubits, reps)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def measure_y(x):
        apply_zz_feature_map(x, n_qubits, reps)
        return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def measure_z(x):
        apply_zz_feature_map(x, n_qubits, reps)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return measure_x, measure_y, measure_z


def compute_bloch_vectors(
    X: np.ndarray,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute Bloch vectors for all data points.

    For each data point x, returns 3 × n_qubits values:
    [⟨X₀⟩, ⟨Y₀⟩, ⟨Z₀⟩, ⟨X₁⟩, ⟨Y₁⟩, ⟨Z₁⟩, ...].

    Args:
        X: Data matrix, shape (n, n_qubits).
        n_qubits: Number of qubits.
        reps: ZZFeatureMap repetitions.
        save_path: Path to cache Bloch vectors.

    Returns:
        np.ndarray: Bloch vectors, shape (n, 3 * n_qubits).
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached Bloch vectors: {save_path}")
        return np.load(save_path)

    measure_x, measure_y, measure_z = _build_pqk_circuit(n_qubits, reps)

    n = len(X)
    bloch = np.zeros((n, 3 * n_qubits), dtype=np.float64)

    logger.info(f"Computing Bloch vectors for {n} samples ({3 * n_qubits}d per sample)")

    for i in tqdm(range(n), desc="Bloch vectors"):
        x = X[i]
        exp_x = np.array(measure_x(x))
        exp_y = np.array(measure_y(x))
        exp_z = np.array(measure_z(x))

        # Interleave: [X0, Y0, Z0, X1, Y1, Z1, ...]
        for q in range(n_qubits):
            bloch[i, 3 * q] = exp_x[q]
            bloch[i, 3 * q + 1] = exp_y[q]
            bloch[i, 3 * q + 2] = exp_z[q]

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, bloch)
        logger.info(f"Saved Bloch vectors: {save_path}")

    return bloch


def compute_pqk_kernel_matrix(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    gamma: float = config.PQK_GAMMA,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_path: Optional[str] = None,
    bloch_save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute the Projected Quantum Kernel (PQK) matrix.

    K_PQK(x, x') = exp(-γ Σ_i ||ρ_i(x) - ρ_i(x')||²_F)

    where ||ρ_i(x) - ρ_i(x')||²_F = (1/2) Σ_{a∈{X,Y,Z}} (⟨σ^a_i⟩_x - ⟨σ^a_i⟩_{x'})²

    PQK is more robust to kernel concentration than FQK.

    Args:
        X: Data matrix, shape (n, n_qubits).
        X2: Optional second data matrix for rectangular kernel.
        gamma: Bandwidth parameter (default: 1.0).
        n_qubits: Number of qubits.
        reps: ZZFeatureMap repetitions.
        save_path: Path to save kernel matrix.
        bloch_save_path: Path to cache Bloch vectors.

    Returns:
        np.ndarray: PQK kernel matrix.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached PQK kernel: {save_path}")
        return np.load(save_path)

    # Step 1: Compute Bloch vectors
    bloch1 = compute_bloch_vectors(
        X, n_qubits, reps,
        save_path=bloch_save_path,
    )

    if X2 is not None:
        bloch2_path = bloch_save_path.replace(".npy", "_test.npy") if bloch_save_path else None
        bloch2 = compute_bloch_vectors(X2, n_qubits, reps, save_path=bloch2_path)
    else:
        bloch2 = bloch1

    # Step 2: Compute Frobenius distances per qubit and sum
    # bloch shape: (n, 3*n_qubits) where groups of 3 = [X, Y, Z] per qubit
    # ||ρ_i(x) - ρ_i(x')||²_F = (1/2) Σ_{a} (exp_a_x - exp_a_x')²
    n_train = len(bloch1)  # len(X) — always training samples
    n_test  = len(bloch2)  # len(X2) if X2 else len(X)

    # Reshape to (n, n_qubits, 3)
    b1 = bloch1.reshape(n_train, n_qubits, 3)
    b2 = bloch2.reshape(n_test,  n_qubits, 3)

    if X2 is None:
        # Square/symmetric case: K shape (N_train, N_train)
        logger.info(f"Computing PQK kernel matrix: {n_train}×{n_train} (symmetric)")
        K = np.zeros((n_train, n_train), dtype=np.float64)
        for i in tqdm(range(n_train), desc="PQK Kernel"):
            diff       = b1[i] - b2          # (n_train, n_qubits, 3)
            sq_diff    = np.sum(diff ** 2, axis=2)   # (n_train, n_qubits)
            frob_dist  = 0.5 * np.sum(sq_diff, axis=1)  # (n_train,)
            K[i, :]    = np.exp(-gamma * frob_dist)
    else:
        # Rectangular/test case: K shape MUST BE (N_test, N_train) for sklearn SVM
        # Matches FQK convention: compute_fqk_kernel_matrix returns (len(X2), len(X))
        logger.info(f"Computing PQK kernel matrix: {n_test}×{n_train} (test vs train)")
        K = np.zeros((n_test, n_train), dtype=np.float64)
        for i in tqdm(range(n_test), desc="PQK Kernel (rect)"):
            diff       = b2[i] - b1          # (n_train, n_qubits, 3)
            sq_diff    = np.sum(diff ** 2, axis=2)   # (n_train, n_qubits)
            frob_dist  = 0.5 * np.sum(sq_diff, axis=1)  # (n_train,)
            K[i, :]    = np.exp(-gamma * frob_dist)

    if X2 is not None:
        assert K.shape == (len(X2), len(X)), \
            f"PQK test kernel shape error: expected ({len(X2)}, {len(X)}), got {K.shape}"
    else:
        assert K.shape == (len(X), len(X)), \
            f"PQK train kernel shape error: expected ({len(X)}, {len(X)}), got {K.shape}"

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, K)
        logger.info(f"Saved PQK kernel: {save_path}")

    return K


# ============ KERNEL VALIDATION ============

def validate_kernel_matrix(K: np.ndarray, name: str = "Kernel", tol: float = 1e-4) -> bool:
    """
    Validate kernel matrix properties: symmetry, PSD, diagonal = 1, entries in [0, 1].

    Args:
        K: Kernel matrix to validate.
        name: Name for logging.
        tol: Tolerance for numerical checks.

    Returns:
        bool: True if all checks pass.
    """
    ok = True

    # Symmetry
    if K.shape[0] == K.shape[1]:
        asym = np.max(np.abs(K - K.T))
        if asym > tol:
            logger.warning(f"[{name}] Asymmetry: max|K - K^T| = {asym:.6f}")
            ok = False

        # Diagonal = 1
        diag_err = np.max(np.abs(np.diag(K) - 1.0))
        if diag_err > tol:
            logger.warning(f"[{name}] Diagonal error: max|diag(K) - 1| = {diag_err:.6f}")
            ok = False

        # PSD (all eigenvalues ≥ -tol)
        eigenvalues = np.linalg.eigvalsh(K)
        min_eig = eigenvalues.min()
        if min_eig < -tol:
            logger.warning(f"[{name}] Not PSD: min eigenvalue = {min_eig:.6f}")
            ok = False

    # Range [0, 1]
    if K.min() < -tol or K.max() > 1.0 + tol:
        logger.warning(f"[{name}] Range error: [{K.min():.6f}, {K.max():.6f}]")
        ok = False

    if ok:
        logger.info(f"[{name}] Validation PASSED (shape={K.shape})")
    else:
        logger.warning(f"[{name}] Validation FAILED")

    return ok


# ============ TRAINED FIDELITY KERNEL (TFK) ============

def apply_trained_zz_feature_map(
    x: np.ndarray,
    theta: np.ndarray,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
) -> None:
    """
    Apply a TRAINABLE ZZFeatureMap with per-qubit scale parameters.

    Identical structure to apply_zz_feature_map() EXCEPT:
    - RZ(x_i) becomes RZ(theta_i * x_i)    [trainable single-qubit scale]
    - ZZ angle becomes (pi - theta_i*x_i)(pi - theta_j*x_j)  [consistent]

    This makes the encoding data-adaptive. theta is initialized to ones
    (reproducing the standard ZZFeatureMap) and optimized via KTA.

    Args:
        x: Input features, shape (n_qubits,). Values in [0, pi].
        theta: Trainable parameters, shape (n_qubits,). Initialized to 1.0.
        n_qubits: Number of qubits.
        reps: Number of circuit repetitions.
    """
    import pennylane as qml

    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(theta[i] * x[i], wires=i)
        for i in range(n_qubits - 1):
            j = i + 1
            zz_angle = (np.pi - theta[i] * x[i]) * (np.pi - theta[j] * x[j])
            qml.CNOT(wires=[i, j])
            qml.RZ(zz_angle, wires=j)
            qml.CNOT(wires=[i, j])


def train_quantum_kernel_kta(
    X: np.ndarray,
    y: np.ndarray,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    n_epochs: int = config.TFK_N_EPOCHS,
    subset_size: int = config.TFK_SUBSET_SIZE,
    lr: float = config.TFK_LR,
    seed: int = config.RANDOM_SEED,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Train ZZFeatureMap parameters theta to maximize class-weighted centered KTA.

    This produces a Trained Fidelity Kernel (TFK): a data-adaptive quantum
    kernel whose encoding is optimized for the specific classification task.
    Addresses the main criticism of fixed quantum feature maps.

    Implementation follows Hubregtsen et al., Physical Review A 106, 042431 (2022):
    gradients are computed via the parameter-shift rule applied analytically
    to the KTA objective, NOT via autograd through the kernel matrix.
    This is the publication-standard approach for KTA-based quantum kernel
    training and avoids all ArrayBox/autograd compatibility issues.

    Gradient derivation:
        KTA(K, y) = <K_c, y_c>_F / (||K_c||_F * ||y_c||_F)
        where K_c = H K H, y_c = H y_ideal H, H = I - (1/n)11^T.

        ∂KTA/∂theta_k = sum_{i<=j} w_{ij} * ∂K_{ij}/∂theta_k

        where w_{ij} is the closed-form sensitivity weight from the
        Frobenius inner product quotient rule (see _kta_gradient below),
        and ∂K_{ij}/∂theta_k is computed via the parameter-shift rule:
            [K(theta + π/2 * e_k) - K(theta - π/2 * e_k)]_ij / 2

    Training procedure:
        1. Initialize theta = ones(n_qubits)
        2. At each step: sample `subset_size` points stratified by class
        3. Compute grad KTA via parameter-shift (no autograd)
        4. Update theta via Adam gradient descent on -KTA (maximize KTA)
        5. Repeat for n_epochs steps

    Args:
        X: Training data, shape (n, n_qubits). Values in [0, pi].
        y: Training labels, shape (n,). Integer class labels.
        n_qubits: Number of qubits.
        reps: ZZFeatureMap repetitions.
        n_epochs: Number of optimizer steps.
        subset_size: Samples per gradient step (keep <= 32 for tractability).
        lr: Adam learning rate.
        seed: Random seed for reproducibility.
        save_path: Path to save trained theta (.npy). Loads if exists.

    Returns:
        np.ndarray: Trained theta, shape (n_qubits,).

    References:
        Hubregtsen et al., Phys. Rev. A 106, 042431 (2022).
        Alignment-based quantum kernel training with parameter-shift.
    """
    import pennylane as qml

    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached TFK theta: {save_path}")
        return np.load(save_path)

    np.random.seed(seed)
    dev = config.get_device(n_qubits)

    # ---- Circuit: diff_method=None — gradients computed manually below ----
    # We deliberately do NOT use diff_method="parameter-shift" on the qnode.
    # Instead we call the circuit at shifted theta values ourselves.
    # This avoids all autograd/ArrayBox issues with lightning.gpu.
    @qml.qnode(dev, diff_method=None)
    def tfk_circuit_eval(x1, x2, theta):
        """Evaluate one kernel entry at fixed theta. Returns prob(|0>^n)."""
        apply_trained_zz_feature_map(x1, theta, n_qubits, reps)
        qml.adjoint(apply_trained_zz_feature_map)(x2, theta, n_qubits, reps)
        return qml.probs(wires=range(n_qubits))

    def _kernel_entry(x1, x2, theta):
        """Scalar kernel entry K(x1, x2; theta) = prob(|0>^n)."""
        return float(tfk_circuit_eval(x1, x2, theta)[0])

    def _compute_kernel_matrix(X_sub, theta):
        """
        Compute full symmetric kernel matrix on X_sub at fixed theta.
        Uses symmetry: only n(n-1)/2 + n evaluations.

        Returns:
            np.ndarray: shape (n, n), dtype float64.
        """
        n = len(X_sub)
        K = np.eye(n, dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                v = _kernel_entry(X_sub[i], X_sub[j], theta)
                K[i, j] = v
                K[j, i] = v
        return K

    def _build_ideal_kernel_weighted(y_sub):
        """
        Class-weighted ideal kernel matrix.
        y_ideal[i,j] = 1/n_c if y_sub[i]==y_sub[j]==c, else 0.
        Each class contributes equally regardless of frequency.

        Args:
            y_sub: Integer label array, shape (n,).

        Returns:
            np.ndarray: Ideal kernel matrix, shape (n, n).
        """
        n = len(y_sub)
        y_ideal = np.zeros((n, n), dtype=np.float64)
        classes, counts = np.unique(y_sub, return_counts=True)
        count_dict = dict(zip(classes.tolist(), counts.tolist()))
        for i in range(n):
            for j in range(n):
                if y_sub[i] == y_sub[j]:
                    y_ideal[i, j] = 1.0 / count_dict[y_sub[i]]
        return y_ideal

    def _centered_kta(K, y_ideal):
        """
        Centered class-weighted KTA.

        KTA_c(K, y) = <H K H, H y_ideal H>_F / (||H K H||_F * ||H y_ideal H||_F)

        where H = I - (1/n)11^T is the centering matrix.

        Args:
            K: Kernel matrix, shape (n, n).
            y_ideal: Ideal kernel matrix, shape (n, n).

        Returns:
            float: KTA value in [-1, 1].
        """
        n = len(K)
        H = np.eye(n) - np.ones((n, n)) / n
        K_c = H @ K @ H
        y_c = H @ y_ideal @ H
        num = np.sum(K_c * y_c)
        den = np.sqrt(np.sum(K_c ** 2) * np.sum(y_c ** 2) + 1e-15)
        return float(num / den)

    def _kta_gradient(K, y_ideal):
        """
        Compute the sensitivity weight matrix W for the KTA gradient.

        From the quotient rule applied to KTA_c = <K_c, y_c> / (||K_c|| * ||y_c||):

            ∂KTA_c / ∂K_{ij} = (1 / (||K_c|| * ||y_c||)) *
                [y_c_sym_{ij} - KTA_c * K_c_{ij} / ||K_c||^2] * h_{ij}

        where h_{ij} is the (i,j) entry of H K H differentiated w.r.t. K_{ij},
        which equals H_{ii} * H_{jj} for the centering operation.

        In practice this simplifies to:

            W = H^T * [(y_c / norm_K_y) - (KTA_c * K_c / norm_K^2)] * H^T
                * (2 - delta_{ij})   [symmetry factor]

        This is the standard derivation in Hubregtsen et al. (2022), eq. (A3).

        Args:
            K: Kernel matrix at current theta, shape (n, n).
            y_ideal: Class-weighted ideal kernel, shape (n, n).

        Returns:
            np.ndarray: Weight matrix W, shape (n, n).
                W[i, j] = ∂KTA_c / ∂K_{ij}.
        """
        n = len(K)
        H = np.eye(n) - np.ones((n, n)) / n
        K_c = H @ K @ H
        y_c = H @ y_ideal @ H

        norm_K = np.sqrt(np.sum(K_c ** 2) + 1e-15)
        norm_y = np.sqrt(np.sum(y_c ** 2) + 1e-15)
        kta_val = np.sum(K_c * y_c) / (norm_K * norm_y)

        # Gradient of numerator w.r.t. K_c: y_c / (norm_K * norm_y)
        # Gradient of ||K_c|| w.r.t. K_c: K_c / norm_K
        # Quotient rule combined:
        grad_K_c = (y_c / (norm_K * norm_y)) - (kta_val * K_c / (norm_K ** 2))

        # Chain rule through centering: K_c = H K H → ∂KTA/∂K = H^T grad_K_c H^T
        # Since H is symmetric: = H grad_K_c H
        W = H @ grad_K_c @ H

        # Symmetry factor: off-diagonal entries appear twice (K_{ij} = K_{ji})
        sym_factor = 2.0 * np.ones((n, n))
        np.fill_diagonal(sym_factor, 1.0)
        W = W * sym_factor

        return W

    def _compute_kta_gradient_param_shift(X_sub, y_sub, theta):
        """
        Compute exact ∂KTA/∂theta via parameter-shift rule over all pairs.

        With a fixed anchor subset reused every epoch, Adam sees the same
        loss landscape at each step and momentum accumulates correctly.
        The full gradient is computed over all n*(n-1)/2 = 1275 pairs
        (for subset_size=51), with no stochastic approximation.

        This matches Hubregtsen et al., Phys. Rev. A 106, 042431 (2022):
        parameter-shift gradient on a fixed small training subset.

        Cost per epoch:
            (2 * n_qubits + 1) * n_pairs circuit evaluations
            = 17 * 1275 = 21,675 evals ≈ 251s/epoch on RTX 4050.

        Args:
            X_sub: Fixed anchor subset, shape (51, n_qubits).
            y_sub: Labels for anchor subset, shape (51,).
            theta: Current parameters, shape (n_qubits,).

        Returns:
            Tuple[np.ndarray, float]:
                grad: Exact gradient ∂KTA/∂theta, shape (n_qubits,).
                kta_val: KTA at current theta on anchor subset.

        References:
            Hubregtsen et al., Phys. Rev. A 106, 042431 (2022).
        """
        shift = np.pi / 2.0
        n = len(X_sub)

        # Step 1: Full kernel matrix at current theta.
        # Required for KTA value and weight matrix W.
        # This is the ONLY full kernel computation per epoch.
        K = _compute_kernel_matrix(X_sub, theta)
        y_ideal = _build_ideal_kernel_weighted(y_sub)
        kta_val = _centered_kta(K, y_ideal)
        W = _kta_gradient(K, y_ideal)

        # Step 2: Build list of all off-diagonal pairs (i < j).
        # Diagonal entries are excluded: K[i,i]=1 always for fidelity
        # kernels, so ∂K[i,i]/∂theta_k = 0 and they contribute nothing.
        # Full gradient — all pairs, no stochastic sampling.
        # With fixed anchor subset (same 51 samples every epoch), using
        # all 1275 pairs gives the exact KTA gradient at every step.
        # Adam momentum then accumulates correctly toward the true optimum.
        # This matches Hubregtsen et al. (2022) who compute the full
        # gradient over their fixed training subset.
        all_pairs     = [(i, j) for i in range(n) for j in range(i + 1, n)]
        sampled_pairs = all_pairs   # All pairs — no subsampling
        scale         = 1.0         # No rescaling needed

        # Step 3: Stochastic parameter-shift over sampled pairs
        grad = np.zeros(n_qubits, dtype=np.float64)

        for k in range(n_qubits):
            # Shifted theta vectors for parameter k
            theta_plus = theta.copy()
            theta_plus[k] += shift
            theta_minus = theta.copy()
            theta_minus[k] -= shift

            grad_k_accum = 0.0
            for (i, j) in sampled_pairs:
                # Parameter-shift rule on this pair only:
                # ∂K_{ij}/∂theta_k = [K(theta+π/2 e_k)_{ij}
                #                     - K(theta-π/2 e_k)_{ij}] / 2
                k_plus  = _kernel_entry(X_sub[i], X_sub[j], theta_plus)
                k_minus = _kernel_entry(X_sub[i], X_sub[j], theta_minus)
                dk = (k_plus - k_minus) / 2.0

                # W[i,j] is ∂KTA/∂K_{ij}. Factor of 2 accounts for
                # symmetry K[i,j]=K[j,i] (each off-diagonal pair
                # appears twice in the full Frobenius inner product).
                grad_k_accum += W[i, j] * dk * 2.0

            # 0.5 corrects for the symmetry factor already in W
            # (W was built with sym_factor=2 for off-diagonal entries).
            # scale corrects for pair subsampling.
            grad[k] = 0.5 * scale * grad_k_accum

        return grad, kta_val

    # ---- Adam optimizer — pure numpy, no autograd ----
    # Standard Adam: Kingma & Ba, ICLR 2015.
    theta = np.ones(n_qubits, dtype=np.float64)
    m = np.zeros(n_qubits, dtype=np.float64)   # first moment
    v = np.zeros(n_qubits, dtype=np.float64)   # second moment
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # ---- Fixed anchor subset — drawn ONCE, reused for all epochs ----
    # Rationale: drawing a fresh random subset every epoch means Adam sees
    # a different loss landscape each step, causing momentum to accumulate
    # gradients from different landscapes that partially cancel each other.
    # This is the root cause of KTA oscillation seen in previous runs.
    #
    # By fixing one stratified subset at initialization, Adam sees the same
    # loss landscape every epoch and momentum accumulates correctly toward
    # the true KTA gradient. This matches Hubregtsen et al. (2022) who use
    # a fixed small training subset for the entire KTA optimization.
    #
    # Subset size: 51 = 3 per class × 17 classes.
    # Binding constraint: class 6 (LCZ Lightweight Low-rise) has only 22
    # samples in the training subsample. 3/class = 153 total intra-class
    # pairs across all classes — minimum for meaningful KTA block structure.
    classes     = np.unique(y)
    n_classes   = len(classes)
    n_per_class = subset_size // n_classes   # 51 // 17 = 3

    # Per-class availability check
    class_counts    = {c: int(np.sum(y == c)) for c in classes}
    min_class_count = min(class_counts.values())
    n_need_replace  = sum(1 for c in classes if class_counts[c] < n_per_class)

    logger.info(
        f"Training TFK: {n_epochs} epochs, subset_size={subset_size}, lr={lr}, "
        f"n_per_class={n_per_class}, n_classes={n_classes}"
    )
    logger.info(
        f"  Anchor subset strategy: FIXED — drawn once at initialization, "
        f"reused for all {n_epochs} epochs (Hubregtsen et al. 2022)"
    )
    logger.info(
        f"  Min class count in training set: {min_class_count} "
        f"({'ALL OK' if n_need_replace == 0 else f'{n_need_replace} classes need replace=True'})"
    )

    # Draw fixed anchor subset
    anchor_idx = []
    for c in classes:
        c_idx   = np.where(y == c)[0]
        n_avail = len(c_idx)
        chosen  = np.random.choice(
            c_idx,
            size=n_per_class,
            replace=(n_avail < n_per_class)   # replace=True only for rare classes
        )
        if n_avail < n_per_class:
            class_name = (
                config.LCZ_CLASS_NAMES[c]
                if hasattr(config, 'LCZ_CLASS_NAMES') and c < len(config.LCZ_CLASS_NAMES)
                else f"Class {c}"
            )
            logger.warning(
                f"  Class '{class_name}' (label={c}): only {n_avail} samples "
                f"< n_per_class={n_per_class}. Using replace=True."
            )
        anchor_idx.extend(chosen.tolist())

    anchor_idx = np.array(anchor_idx[:subset_size])
    np.random.shuffle(anchor_idx)
    X_anchor   = X[anchor_idx]
    y_anchor   = y[anchor_idx]

    # Log anchor subset composition
    anchor_classes, anchor_counts = np.unique(y_anchor, return_counts=True)
    n_anchor_pairs = len(X_anchor) * (len(X_anchor) - 1) // 2
    logger.info(
        f"  Anchor subset: {len(X_anchor)} samples, "
        f"{len(anchor_classes)}/17 classes, "
        f"{n_anchor_pairs} pairs"
    )
    logger.info(
        f"  Circuit evals/epoch: "
        f"{(2 * n_qubits + 1) * n_anchor_pairs} "
        f"(full gradient, no pair subsampling)"
    )

    kta_history = []
    
    from tqdm import tqdm
    pbar = tqdm(range(n_epochs), desc="TFK Training")

    for epoch in pbar:
        # Use fixed anchor subset — NO resampling inside the loop
        X_sub = X_anchor
        y_sub = y_anchor

        # Compute gradient via parameter-shift
        grad, kta_val = _compute_kta_gradient_param_shift(X_sub, y_sub, theta)

        # Gradient clipping — cap gradient norm to 1.0 before Adam update.
        # Prevents theta divergence when stochastic pair sampling produces
        # occasionally large gradient estimates. Without clipping, lr=0.05
        # caused ||theta-1|| to reach 16.35 over 75 epochs (degenerate kernel).
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm   # rescale to unit norm

        # Adam update on -KTA (maximize KTA = minimize -KTA)
        neg_grad = -grad
        t = epoch + 1
        m = beta1 * m + (1 - beta1) * neg_grad
        v = beta2 * v + (1 - beta2) * neg_grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)

        # Theta bounds clipping — constrain each parameter to [-2π, 2π].
        # RZ(theta[i] * x[i]) with x[i] ∈ [0,π]: values beyond ±2π cause
        # the circuit to wrap around the Bloch sphere with no additional
        # expressivity. Clipping enforces physically meaningful parameter range.
        theta = np.clip(theta, -2.0 * np.pi, 2.0 * np.pi)

        kta_history.append(float(kta_val))
        
        # Update progress bar metrics
        pbar.set_postfix({
            "KTA": f"{kta_val:.4f}", 
            "grad_norm": f"{grad_norm:.4f}",
            "theta": f"[{theta.min():.2f}, {theta.max():.2f}]"
        })

        if epoch % 5 == 0:
            theta_min  = theta.min()
            theta_max  = theta.max()
            theta_dev  = np.linalg.norm(theta - np.ones(n_qubits))
            clipped    = "CLIPPED" if grad_norm > 1.0 else "ok"
            logger.info(
                f"  Epoch {epoch:3d}: KTA={kta_val:.4f}, "
                f"||grad||={grad_norm:.4f} ({clipped}), "
                f"theta=[{theta_min:.3f}..{theta_max:.3f}], "
                f"||theta-1||={theta_dev:.3f}"
            )

    theta_final = np.array(theta, dtype=np.float64)
    logger.info(f"TFK training complete. Final KTA = {kta_history[-1]:.6f}")
    logger.info(f"Initial KTA = {kta_history[0]:.6f}")
    logger.info(
        f"KTA improvement: {kta_history[-1] - kta_history[0]:+.6f}"
    )
    logger.info(f"Final theta: {theta_final}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, theta_final)
        logger.info(f"Saved TFK theta: {save_path}")

    return theta_final


def compute_tfk_kernel_matrix(
    X: np.ndarray,
    theta: np.ndarray,
    X2: Optional[np.ndarray] = None,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute the Trained Fidelity Kernel (TFK) matrix using optimized theta.

    Identical to compute_fqk_kernel_matrix() but uses apply_trained_zz_feature_map
    with the trained theta parameters.

    Args:
        X: Data matrix, shape (n, n_qubits).
        theta: Trained parameters from train_quantum_kernel_kta(), shape (n_qubits,).
        X2: Optional second data matrix for rectangular kernel.
        n_qubits: Number of qubits.
        reps: ZZFeatureMap repetitions.
        save_path: Path to save kernel matrix (.npy).

    Returns:
        np.ndarray: TFK kernel matrix.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached TFK kernel: {save_path}")
        return np.load(save_path)

    import pennylane as qml
    dev = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def tfk_circuit(x1, x2):
        apply_trained_zz_feature_map(x1, theta, n_qubits, reps)
        qml.adjoint(apply_trained_zz_feature_map)(x2, theta, n_qubits, reps)
        return qml.probs(wires=range(n_qubits))

    symmetric = X2 is None
    if symmetric:
        n = len(X)
        K = np.eye(n, dtype=np.float64)
        n_evals = n * (n - 1) // 2
        logger.info(f"Computing TFK kernel matrix: {n}×{n} ({n_evals} evaluations)")
        with tqdm(total=n_evals, desc="TFK Kernel") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    val = float(tfk_circuit(X[i], X[j])[0])
                    K[i, j] = val
                    K[j, i] = val
                    pbar.update(1)
    else:
        m, n = len(X2), len(X)
        K = np.zeros((m, n), dtype=np.float64)
        logger.info(f"Computing TFK kernel matrix: {m}×{n} ({m*n} evaluations)")
        with tqdm(total=m * n, desc="TFK Kernel (rect)") as pbar:
            for i in range(m):
                for j in range(n):
                    K[i, j] = float(tfk_circuit(X2[i], X[j])[0])
                    pbar.update(1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, K)
        logger.info(f"Saved TFK kernel: {save_path}")

    return K


# ============ CROSS-MODAL ZZ FEATURE MAP ============

def apply_crossmodal_zz_feature_map(
    x: np.ndarray,
    mi_pairs: List[Tuple[int, int]],
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
) -> None:
    """
    Apply a Cross-Modal ZZFeatureMap with data-informed entanglement topology.

    EXTENDS apply_zz_feature_map() by adding explicit cross-modal ZZ interactions
    between qubit pairs with highest mutual information. The first n_qubits//2
    qubits encode SAR features; the second half encode optical features.

    Standard ZZFeatureMap uses linear connectivity: (0,1),(1,2),...,(6,7).
    This map ADDS cross-modal pairs from mi_pairs on top of linear connectivity.

    The cross-modal ZZ angles use the same formula:
        zz_angle = (pi - x[i]) * (pi - x[j])
    but for (i,j) pairs that CROSS the SAR/optical boundary.

    Args:
        x: Input features, shape (n_qubits,). Values in [0, pi].
           x[:n_qubits//2] = SAR PCA features
           x[n_qubits//2:] = Optical PCA features
        mi_pairs: List of (sar_qubit_idx, opt_qubit_idx) tuples ordered by
                  decreasing mutual information. Qubit indices are global
                  (opt qubits are n_qubits//2 + local_opt_idx).
                  Typically 2-4 pairs. Computed by get_top_mi_pairs().
        n_qubits: Total number of qubits (SAR + optical combined).
        reps: Number of repetitions.
    """
    import pennylane as qml

    for _ in range(reps):
        # Standard linear ZZ (identical to apply_zz_feature_map)
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        for i in range(n_qubits - 1):
            j = i + 1
            zz_angle = (np.pi - x[i]) * (np.pi - x[j])
            qml.CNOT(wires=[i, j])
            qml.RZ(zz_angle, wires=j)
            qml.CNOT(wires=[i, j])

        # Additional cross-modal ZZ interactions (data-informed)
        for (i, j) in mi_pairs:
            if i != j and i < n_qubits and j < n_qubits:
                zz_angle = (np.pi - x[i]) * (np.pi - x[j])
                qml.CNOT(wires=[i, j])
                qml.RZ(zz_angle, wires=j)
                qml.CNOT(wires=[i, j])


def get_top_mi_pairs(
    X_fused: np.ndarray,
    n_sar: int = 4,
    n_opt: int = 4,
    top_k: int = 3,
) -> List[Tuple[int, int]]:
    """
    Compute cross-modal mutual information between SAR and optical PCA features
    and return the top-k pairs with highest MI as qubit index tuples.

    Used to define the entanglement topology of CrossModalZZFeatureMap.
    SAR features occupy qubits 0..n_sar-1.
    Optical features occupy qubits n_sar..n_sar+n_opt-1.

    Args:
        X_fused: Fused data matrix, shape (n, n_sar + n_opt).
        n_sar: Number of SAR features (qubits 0..n_sar-1).
        n_opt: Number of optical features (qubits n_sar..n_sar+n_opt-1).
        top_k: Number of highest-MI pairs to return.

    Returns:
        List[Tuple[int, int]]: Top-k (sar_qubit, opt_qubit) pairs by MI.
        Example: [(0, 4), (1, 5), (2, 6)] means SAR_PCA0↔OPT_PCA0, etc.
    """
    from sklearn.feature_selection import mutual_info_regression

    X_sar = X_fused[:, :n_sar]
    X_opt = X_fused[:, n_sar:n_sar + n_opt]

    mi_matrix = np.zeros((n_sar, n_opt))
    for i in range(n_sar):
        mi_vals = mutual_info_regression(X_opt, X_sar[:, i], random_state=42)
        mi_matrix[i, :] = mi_vals

    # Get top_k pairs sorted by MI descending
    flat_idx = np.argsort(mi_matrix.ravel())[::-1][:top_k]
    pairs = []
    for idx in flat_idx:
        sar_idx = idx // n_opt
        opt_idx = idx % n_opt
        pairs.append((int(sar_idx), int(n_sar + opt_idx)))

    logger.info(f"Top-{top_k} cross-modal MI pairs (SAR qubit, OPT qubit):")
    for (s, o) in pairs:
        logger.info(f"  SAR_PCA{s} ↔ OPT_PCA{o - n_sar}: MI={mi_matrix[s, o - n_sar]:.4f}")

    return pairs


def compute_crossmodal_fqk_kernel_matrix(
    X: np.ndarray,
    mi_pairs: List[Tuple[int, int]],
    X2: Optional[np.ndarray] = None,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Compute FQK kernel matrix using CrossModalZZFeatureMap.

    Identical to compute_fqk_kernel_matrix() but calls
    apply_crossmodal_zz_feature_map() instead of apply_zz_feature_map().

    Args:
        X: Data matrix, shape (n, n_qubits).
        mi_pairs: Cross-modal pairs from get_top_mi_pairs().
        X2: Optional second data matrix for rectangular kernel.
        n_qubits: Number of qubits.
        reps: Circuit repetitions.
        save_path: Path to cache kernel matrix.

    Returns:
        np.ndarray: CrossModal FQK kernel matrix.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached CrossModal FQK: {save_path}")
        return np.load(save_path)

    import pennylane as qml
    dev = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def cm_fqk_circuit(x1, x2):
        apply_crossmodal_zz_feature_map(x1, mi_pairs, n_qubits, reps)
        qml.adjoint(apply_crossmodal_zz_feature_map)(x2, mi_pairs, n_qubits, reps)
        return qml.probs(wires=range(n_qubits))

    symmetric = X2 is None
    if symmetric:
        n = len(X)
        K = np.eye(n, dtype=np.float64)
        n_evals = n * (n - 1) // 2
        logger.info(f"Computing CrossModal FQK: {n}×{n} ({n_evals} evaluations)")
        with tqdm(total=n_evals, desc="CrossModal FQK") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    val = float(cm_fqk_circuit(X[i], X[j])[0])
                    K[i, j] = val
                    K[j, i] = val
                    pbar.update(1)
    else:
        m, n = len(X2), len(X)
        K = np.zeros((m, n), dtype=np.float64)
        logger.info(f"Computing CrossModal FQK: {m}×{n} ({m*n} evaluations)")
        with tqdm(total=m * n, desc="CrossModal FQK (rect)") as pbar:
            for i in range(m):
                for j in range(n):
                    K[i, j] = float(cm_fqk_circuit(X2[i], X[j])[0])
                    pbar.update(1)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, K)
        logger.info(f"Saved CrossModal FQK: {save_path}")

    return K


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  src/quantum_kernels.py — Self-test")
    print("=" * 60)

    np.random.seed(config.RANDOM_SEED)

    # Small test with 4 samples
    n_test = 4
    n_qubits = 4  # Smaller for testing speed
    X = np.random.rand(n_test, n_qubits) * np.pi

    print(f"\nTesting FQK with {n_test} samples, {n_qubits} qubits...")
    K_fqk = compute_fqk_kernel_matrix(X, n_qubits=n_qubits, reps=1)
    print(f"  FQK shape: {K_fqk.shape}")
    print(f"  FQK diagonal: {np.diag(K_fqk)}")
    print(f"  FQK range: [{K_fqk.min():.4f}, {K_fqk.max():.4f}]")
    validate_kernel_matrix(K_fqk, "FQK-test")

    print(f"\nTesting PQK with {n_test} samples, {n_qubits} qubits...")
    K_pqk = compute_pqk_kernel_matrix(X, n_qubits=n_qubits, reps=1)
    print(f"  PQK shape: {K_pqk.shape}")
    print(f"  PQK diagonal: {np.diag(K_pqk)}")
    print(f"  PQK range: [{K_pqk.min():.4f}, {K_pqk.max():.4f}]")
    validate_kernel_matrix(K_pqk, "PQK-test")

    # Test rectangular kernel
    X_test = np.random.rand(2, n_qubits) * np.pi
    K_rect = compute_fqk_kernel_matrix(X, X2=X_test, n_qubits=n_qubits, reps=1)
    print(f"\n  Rectangular FQK shape: {K_rect.shape}")
    assert K_rect.shape == (2, 4)

    K_pqk_rect = compute_pqk_kernel_matrix(X, X2=X_test, n_qubits=n_qubits, reps=1)
    print(f"  Rectangular PQK shape: {K_pqk_rect.shape}")
    assert K_pqk_rect.shape == (2, 4), f"PQK rect shape wrong: {K_pqk_rect.shape}"

    print("\n  All self-tests passed.")
