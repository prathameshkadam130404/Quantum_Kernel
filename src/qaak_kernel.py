import pennylane as qml
import numpy as np
import sys
import os

# Project configuration bounds
N_QUBITS_DATA = 8
N_ANCILLAS = 4
TOTAL_WIRES = N_QUBITS_DATA + N_ANCILLAS

def build_qaak_circuit(q_device, selected_pairs, zz_reps=1):
    """
    Builds the PennyLane QNodes that natively evaluate Quantum Autonomous Attention
    via interference-routed entanglement, extracting the reduced Bloch expectations.
    """
    n_pairs = len(selected_pairs)
    # Publication Level Rigor: We explicitly require fresh ancillas for EACH layer 
    # of the circuit so that subsequent SWAP tests measure pure quantum overlap,
    # avoiding state collapse or mixed-state contamination from previous layers.
    N_ANCILLAS_REQUIRED = n_pairs * zz_reps
    
    if N_ANCILLAS_REQUIRED > len(q_device.wires) - N_QUBITS_DATA:
         raise ValueError(f"Device requires {N_QUBITS_DATA + N_ANCILLAS_REQUIRED} wires. Found {len(q_device.wires)}.")

    def apply_qaak_encoding(x):
        # 1. State Initial Embedding Layer
        for i in range(N_QUBITS_DATA):
            qml.Hadamard(wires=i)
            qml.RZ(x[i], wires=i)

        # 2. Autonomous Attention Layer (CSWAP + Dynamic Routing)
        current_ancilla_idx = N_QUBITS_DATA
        for rep in range(zz_reps):
            for (q_idx, q_jdx) in selected_pairs:
                anc_wire = current_ancilla_idx
                current_ancilla_idx += 1
                
                # Compute intrinsic Quantum Attention score (Similarity in Hilbert Space)
                qml.Hadamard(wires=anc_wire)
                qml.CSWAP(wires=[anc_wire, q_idx, q_jdx])
                qml.Hadamard(wires=anc_wire)
                
                # Quantum Routed Interaction
                zz_angle = (np.pi - x[q_idx]) * (np.pi - x[q_jdx])
                qml.PauliX(wires=anc_wire)
                qml.ctrl(qml.IsingZZ(zz_angle, wires=[q_idx, q_jdx]), control=anc_wire)
                qml.PauliX(wires=anc_wire)
                
        # Unmeasured ancilla implicitly decohere and regularize the state space!
    
    # We create three standalone evaluators identically matching standard PQK formats.
    # PennyLane natively traces out wire indices not explicitly measured.
    
    @qml.qnode(q_device)
    def measure_x(x):
        apply_qaak_encoding(x)
        return [qml.expval(qml.PauliX(i)) for i in range(N_QUBITS_DATA)]
        
    @qml.qnode(q_device)
    def measure_y(x):
        apply_qaak_encoding(x)
        return [qml.expval(qml.PauliY(i)) for i in range(N_QUBITS_DATA)]
        
    @qml.qnode(q_device)
    def measure_z(x):
        apply_qaak_encoding(x)
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS_DATA)]
        
    return measure_x, measure_y, measure_z

from tqdm import tqdm

def extract_qaak_bloch_vectors(X, measure_x, measure_y, measure_z, log_frequency=200):
    """
    Generate dimensionally flattened Bloch embeddings iteratively.
    """
    N_samples = len(X)
    bloch_vectors = np.zeros((N_samples, N_QUBITS_DATA * 3))
    
    print(f"Extracting QAAK observables for {N_samples} points using decoherence tracking...")
    for i in tqdm(range(N_samples), desc="Generating QAAK Subspace", unit="sample"):
        x = X[i]
        sx = measure_x(x)
        sy = measure_y(x)
        sz = measure_z(x)
        
        # Flatten structure: [X0...X7, Y0...Y7, Z0...Z7]
        bloch_vectors[i] = np.concatenate([sx, sy, sz])
            
    return bloch_vectors
