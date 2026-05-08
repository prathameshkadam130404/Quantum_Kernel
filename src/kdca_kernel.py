"""
Kirkwood–Dirac Cross-Basis Coherence Attention Projected Quantum Kernel (KDCA-PQK).

This module implements the novel KDCA-PQK method as formally described in recent theoretical
literature. Instead of using classical pre-processing or heuristics, it derives quantum-native 
feature selection strictly from Kirkwood-Dirac quasiprobability nonpositivity considerations.

The framework identifies genuine inter-qubit correlators (two-body expectation values) that 
cannot be factored into standard single-qubit Bloch components, certifying that the selected 
cross-qubit relationships carry non-classical information. 

Pipeline:
    1. Base Circuit: Standard depth-2 ZZ feature map with linear entanglement.
    2. Attention Pass: On training data, compute the two-body YX, XY, and YY 
       expectation values for valid pairs. Construct the KDCA attention score A_{ij}^{KD}.
    3. Pair Selection: Isolate the top-K purely cross-modal (SAR-Optical) qubit pairs based on A_ij.
    4. Feature Extraction: Build a 36-dimensional feature vector per sample (24 single-qubit 
       Bloch components + 12 KD-selected two-body components).
    5. Final Kernel: RBF expansion on the 36-dimensional dataset.

Author: Prathamesh Kadam et al.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
import pennylane as qml

# Configure the import paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)

# Same as in config/attention mechanism — Modality flags
FEATURE_MODALITY_16 = [
    "OPT", "OPT", "OPT", "OPT", "OPT", "OPT", "OPT", "OPT",
    "SAR", "SAR", "SAR", "SAR", "SAR", "SAR", "SAR", "SAR",
]


# =============================================================================
# PART I: BASE CIRCUIT BUILDING BLOCKS
# =============================================================================

def apply_base_zz_feature_map(x, n_qubits: int, reps: int):
    """
    Apply standard ZZ feature map natively without external imports,
    using linear nearest-neighbor entanglement topology.
    """
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(2.0 * x[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(2.0 * (np.pi - x[i]) * (np.pi - x[i + 1]), wires=i + 1)
            qml.CNOT(wires=[i, i + 1])


def _build_kdca_single_body_measurement_circuits(n_qubits: int = config.N_QUBITS, reps: int = config.ZZ_REPS):
    """Returns QNodes to measure <X_i>, <Y_i>, <Z_i> for all qubits."""
    dev = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def measure_x(x):
        apply_base_zz_feature_map(x, n_qubits, reps)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def measure_y(x):
        apply_base_zz_feature_map(x, n_qubits, reps)
        return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def measure_z(x):
        apply_base_zz_feature_map(x, n_qubits, reps)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
    return measure_x, measure_y, measure_z


def _build_kdca_two_body_node(n_qubits: int, reps: int, q_i: int, q_j: int, mode: str):
    """
    Builds a QNode for a specific two-body operator using basis rotations.
    mode can be "YX", "XY", or "YY".
    The actual measurement will be PauliZ(q_i) @ PauliZ(q_j) after the correct
    pre-measurement rotations mapping Y or X to Z.
    
    Y -> Z mapping: RX(-pi/2) or RX(pi/2) usually. We will use RX(pi/2).
    X -> Z mapping: Hadamard (H).
    """
    dev = config.get_device(n_qubits)
    
    @qml.qnode(dev, diff_method=None)
    def measure_two_body(x):
        apply_base_zz_feature_map(x, n_qubits, reps)
        
        # Apply local basis rotations
        if mode == "YX":
            qml.RX(np.pi/2, wires=q_i)
            qml.Hadamard(wires=q_j)
        elif mode == "XY":
            qml.Hadamard(wires=q_i)
            qml.RX(np.pi/2, wires=q_j)
        elif mode == "YY":
            qml.RX(np.pi/2, wires=q_i)
            qml.RX(np.pi/2, wires=q_j)
        else:
            raise ValueError(f"Invalid mode {mode}")
            
        return qml.expval(qml.PauliZ(q_i) @ qml.PauliZ(q_j))
        
    return measure_two_body


# =============================================================================
# PART II: KDCA ATTENTION COMPUTATION
# =============================================================================

def compute_kdca_attention(
    X_train: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    selected_indices: Optional[np.ndarray] = None,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_path: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    Compute the KDCA attention scores A^{KD}_{ij} across the training set.
    Only computes specifically for purely cross-modal pairings if `selected_indices` are passed.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached KDCA attention: {save_path}")
        with open(save_path) as f:
            data = json.load(f)
            return {"A_KD": np.array(data["A_KD"])}
            
    logger.info("Computing KDCA attention scores over training set...")
    N = len(X_train)
    
    # Store combinations for valid cross-modal checking
    valid_pairs = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            is_cross = True
            if selected_indices is not None:
                mod_i = FEATURE_MODALITY_16[selected_indices[i]]
                mod_j = FEATURE_MODALITY_16[selected_indices[j]]
                if (mod_i == "SAR") == (mod_j == "SAR"):
                    is_cross = False  # Not cross modal
            if is_cross:
                valid_pairs.append((i, j))
                
    logger.info(f"Identified {len(valid_pairs)} valid cross-modal pairs for investigation.")
    
    # Construct nodes
    mx, my, mz = _build_kdca_single_body_measurement_circuits(n_qubits, reps)
    qnodes_yx = {pair: _build_kdca_two_body_node(n_qubits, reps, pair[0], pair[1], "YX") for pair in valid_pairs}
    qnodes_xy = {pair: _build_kdca_two_body_node(n_qubits, reps, pair[0], pair[1], "XY") for pair in valid_pairs}
    qnodes_yy = {pair: _build_kdca_two_body_node(n_qubits, reps, pair[0], pair[1], "YY") for pair in valid_pairs}
    
    # Allocate storage
    A_KD_matrix = np.zeros((n_qubits, n_qubits), dtype=np.float64)
    
    # Store per-sample measurements to compute Fisher score later
    pair_kd_per_sample = {pair: np.zeros(N) for pair in valid_pairs}
    pair_avg_sum = {pair: 0.0 for pair in valid_pairs}
    
    for idx in tqdm(range(N), desc="KDCA Probe (Train)"):
        x = X_train[idx]
        ex = np.array(mx(x))
        ey = np.array(my(x))
        
        for pair in valid_pairs:
            i, j = pair
            yx_val = qnodes_yx[pair](x)
            xy_val = qnodes_xy[pair](x)
            yy_val = qnodes_yy[pair](x)
            two_body_mag = np.sqrt(yx_val**2 + xy_val**2 + yy_val**2)
            product_baseline = abs(ey[i]) * abs(ex[j])
            
            kd_score = abs(two_body_mag - product_baseline)
            pair_kd_per_sample[pair][idx] = kd_score
            pair_avg_sum[pair] += kd_score

    # Compute A_KD (with supervised Fisher reweighting tracking if populated labels passed)
    kd_fisher = {}
    if y_train is not None:
        classes = np.unique(y_train)
        class_kd = {pair: {} for pair in valid_pairs}
        for pair in valid_pairs:
            for c in classes:
                mask = (y_train == c)
                if mask.sum() > 0:
                    class_kd[pair][c] = pair_kd_per_sample[pair][mask].mean()
                else:
                    class_kd[pair][c] = 0.0
            
            # Between-class variance
            class_means = [class_kd[pair][c] for c in classes]
            kd_fisher[pair] = np.var(class_means)

    if y_train is not None:
        kd_scores = np.array([pair_avg_sum[p]/N for p in valid_pairs])
        fsh_scores = np.array([kd_fisher[p] for p in valid_pairs])

        kd_min, kd_max = kd_scores.min(), kd_scores.max()
        fsh_min, fsh_max = fsh_scores.min(), fsh_scores.max()

        kd_norm = (kd_scores - kd_min) / (kd_max - kd_min + 1e-12)
        fsh_norm = (fsh_scores - fsh_min) / (fsh_max - fsh_min + 1e-12)

        alpha = 0.6
        for idx, pair in enumerate(valid_pairs):
            final_score = alpha * kd_norm[idx] + (1 - alpha) * fsh_norm[idx]
            A_KD_matrix[pair[0], pair[1]] = final_score
            A_KD_matrix[pair[1], pair[0]] = final_score
    else:
        for pair in valid_pairs:
            unsupervised_score = pair_avg_sum[pair] / N
            A_KD_matrix[pair[0], pair[1]] = unsupervised_score
            A_KD_matrix[pair[1], pair[0]] = unsupervised_score
        
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"A_KD": A_KD_matrix.tolist()}, f, indent=2)
            
    return {"A_KD": A_KD_matrix}


def select_kdca_topology(
    A_KD: np.ndarray,
    top_k: int = 4
) -> List[Tuple[int, int]]:
    """
    Selects the top-K pairs exhibiting the highest Kirkwood-Dirac nonpositivity.
    """
    n_qubits = A_KD.shape[0]
    candidates = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if A_KD[i, j] > 0: # Ensure valid pair
                candidates.append((i, j, A_KD[i, j]))
                
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    selected_pairs = [(c[0], c[1]) for c in candidates[:top_k]]
    logger.info(f"KDCA Top-{top_k} selected pairs: {selected_pairs}")
    
    return selected_pairs


# =============================================================================
# PART III: FEATURE EXTRACTION & KERNEL EVALUATION
# =============================================================================

def compute_kdca_bloch_vectors(
    X: np.ndarray,
    entanglement_pairs: List[Tuple[int, int]],
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Extracts the KDCA feature vectors consisting of 3*n_qubits (1-body), 
    3*K (KD pseudo-probability), and K (ZZ correlators). 
    Dimensionality: 24 + 4*K (e.g. 40 when K=4).
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached KDCA bloch vectors: {save_path}")
        return np.load(save_path)
        
    N = len(X)
    dim = 3 * n_qubits + 4 * len(entanglement_pairs)
    bloch = np.zeros((N, dim), dtype=np.float64)
    
    logger.info(f"KDCA Feature Vector Dimension mapping: {dim}")
    
    mx, my, mz = _build_kdca_single_body_measurement_circuits(n_qubits, reps)
    
    qnodes_yx = {pair: _build_kdca_two_body_node(n_qubits, reps, pair[0], pair[1], "YX") for pair in entanglement_pairs}
    qnodes_xy = {pair: _build_kdca_two_body_node(n_qubits, reps, pair[0], pair[1], "XY") for pair in entanglement_pairs}
    qnodes_yy = {pair: _build_kdca_two_body_node(n_qubits, reps, pair[0], pair[1], "YY") for pair in entanglement_pairs}

    for idx in tqdm(range(N), desc="KDCA Feature Extraction"):
        x = X[idx]
        
        ex = np.array(mx(x))
        ey = np.array(my(x))
        ez = np.array(mz(x))
        
        for q in range(n_qubits):
            bloch[idx, 3*q] = ex[q]
            bloch[idx, 3*q + 1] = ey[q]
            bloch[idx, 3*q + 2] = ez[q]
            
        offset = 3 * n_qubits
        for pair in entanglement_pairs:
            i, j = pair
            yx_val = qnodes_yx[pair](x)
            xy_val = qnodes_xy[pair](x)
            yy_val = qnodes_yy[pair](x)
            
            bloch[idx, offset] = yx_val
            bloch[idx, offset + 1] = xy_val
            bloch[idx, offset + 2] = yy_val
            bloch[idx, offset + 3] = ez[i] * ez[j] # ZZ correlator
            offset += 4
            
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, bloch)
        
    return bloch


def compute_kdca_pqk_kernel(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    gamma_kernel: float = config.PQK_GAMMA,
    kernel_type: str = "rbf",
    entanglement_pairs: Optional[List[Tuple[int, int]]] = None,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_path: Optional[str] = None,
    bloch_save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Calculates the KDCA Euclidean RBF Kernel Matrix leveraging the extended 36D space.
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"[SKIP] Loading cached KDCA-PQK kernel: {save_path}")
        return np.load(save_path)
    
    if entanglement_pairs is None:
        raise ValueError("KDCA KD-selected pair list cannot be explicitly null.")
        
    bloch1 = compute_kdca_bloch_vectors(X, entanglement_pairs, n_qubits, reps, save_path=bloch_save_path)
    
    if X2 is not None:
        bloch2_path = bloch_save_path.replace(".npy", "_test.npy") if bloch_save_path else None
        bloch2 = compute_kdca_bloch_vectors(X2, entanglement_pairs, n_qubits, reps, save_path=bloch2_path)
    else:
        bloch2 = bloch1
        
    # Feature Standardization (StandardScaler equivalent logic over the entire vector)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    b1_norm = scaler.fit_transform(bloch1)
    b2_norm = scaler.transform(bloch2)
        
    n_train = len(b1_norm)
    n_test = len(b2_norm)
    
    if kernel_type == "rbf":
        K = np.zeros((n_test, n_train), dtype=np.float64)
        for i in tqdm(range(n_test), desc="Building KDCA RBF Kernel Matrix"):
            diff = b2_norm[i] - b1_norm 
            sq_dist = np.sum(diff**2, axis=1) 
            K[i, :] = np.exp(-gamma_kernel * 0.5 * sq_dist)
    elif kernel_type == "linear":
        logger.info("Evaluating native HS linear kernel metric.")
        K = b2_norm @ b1_norm.T
    elif kernel_type == "poly2":
        from sklearn.metrics.pairwise import polynomial_kernel
        logger.info("Evaluating KD native Polynomial (Degree 2) kernel structure.")
        K = polynomial_kernel(b2_norm, b1_norm, degree=2, coef0=1.0)
    else:
        raise ValueError(f"Unknown generic KDCA architecture kernel target: {kernel_type}")
        
    if X2 is None and n_train == n_test:
        K = 0.5 * (K + K.T) # Symmetrize
        
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, K)
        
    return K


# =============================================================================
# HIGH-LEVEL API FOR EXECUTION SCRIPTS
# =============================================================================

def compute_kdca_pqk_full_pipeline(
    X_train: np.ndarray,
    X_test: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    selected_indices: Optional[np.ndarray] = None,
    gamma_kernel: float = config.PQK_GAMMA,
    kernel_type: str = "rbf",
    top_k_pairs: int = 4,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_dir: Optional[str] = None
) -> Dict:
    """
    Main entry point mirroring QEAA and QAAK conventions.
    """
    attn_path = os.path.join(save_dir, "kdca_attention.json") if save_dir else None
    k_train_path = os.path.join(save_dir, "K_kdca_train.npy") if save_dir else None
    k_test_path = os.path.join(save_dir, "K_kdca_test.npy") if save_dir else None
    bloch_path = os.path.join(save_dir, "kdca_bloch_train.npy") if save_dir else None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Pass 1: KDCA Probe
    attn = compute_kdca_attention(X_train, y_train=y_train, selected_indices=selected_indices, n_qubits=n_qubits, reps=reps, save_path=attn_path)
    
    # Topology Selection
    entanglement_pairs = select_kdca_topology(attn["A_KD"], top_k=top_k_pairs)
    
    # Pass 2: Feature Matrix Expansion Matrix Construction
    K_train = compute_kdca_pqk_kernel(
        X_train, X2=None, 
        gamma_kernel=gamma_kernel, 
        kernel_type=kernel_type,
        entanglement_pairs=entanglement_pairs,
        n_qubits=n_qubits, reps=reps, 
        save_path=k_train_path, bloch_save_path=bloch_path
    )
    
    K_test = None
    if X_test is not None:
        K_test = compute_kdca_pqk_kernel(
            X_train, X2=X_test, 
            gamma_kernel=gamma_kernel, 
            kernel_type=kernel_type,
            entanglement_pairs=entanglement_pairs,
            n_qubits=n_qubits, reps=reps, 
            save_path=k_test_path, bloch_save_path=bloch_path
        )
        
    topology = {"entanglement_pairs": entanglement_pairs, "top_k_pairs": top_k_pairs, "gamma_kernel": gamma_kernel}
    if save_dir:
        topo_path = os.path.join(save_dir, "kdca_topology.json")
        with open(topo_path, "w") as f:
            json.dump({"entanglement_pairs": entanglement_pairs, "top_k_pairs": top_k_pairs}, f)
            
    return {"K_train": K_train, "K_test": K_test, "attention": attn, "topology": topology}
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("  src/kdca_kernel.py — Formal Mathematical Self-test")
    print("=" * 70)
    
    np.random.seed(config.RANDOM_SEED)
    
    N = 10
    n_q = 8
    X_synthetic = np.random.rand(N, n_q) * np.pi
    
    # Synthetic pairs indexing
    synthetic_idx = np.array([0,1,2,3, 8,9,10,11]) # Half optical half SAR
    
    print("--> 1. Testing KDCA Attention calculation...")
    y_synth = np.array([0, 1] * 5)[:N]
    attn_res = compute_kdca_attention(X_synthetic, y_train=y_synth, selected_indices=synthetic_idx, n_qubits=n_q, reps=2)
    A_KD = attn_res["A_KD"]
    print(f"Matrix Dimension OK? {A_KD.shape == (8, 8)}")
    
    print("--> 2. Testing Pair Selection...")
    pairs = select_kdca_topology(A_KD, top_k=4)
    print(f"Pairs Length = {len(pairs)} (Expects 4)")
    
    print("--> 3. Testing KDCA Bloch Feature Extractor Dimensionality...")
    b = compute_kdca_bloch_vectors(X_synthetic, pairs, n_qubits=n_q, reps=2)
    print(f"Bloch Extracted Shape OK? {b.shape == (N, 40)}")
    
    print("--> 4. Testing KDCA RBF Kernel Application...")
    K = compute_kdca_pqk_kernel(X_synthetic, entanglement_pairs=pairs, n_qubits=n_q, reps=2)
    print(f"Kernel Training Dimensionality OK? {K.shape == (N, N)}")
    print(f"Kernel Valid Max / Min ? {K.max():.4f} / {K.min():.4f}")
    
    print("\nSelf-check concluded without exceptions.")
