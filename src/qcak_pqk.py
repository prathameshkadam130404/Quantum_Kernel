"""
qcak_pqk.py — Quantum Cross-Attention Kernel (Projected variant) core functions.

This module implements:
  1. apply_qcak_encoding: encoding circuit with bidirectional cross-modal attention.
  2. compute_qcak_pqk_bloch_vectors: local measurement (⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩ per qubit).
  3. compute_qcak_pqk_kernel_from_bloch: outer RBF kernel on Bloch vector distances.
"""

import os
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
import pennylane as qml

import config

def apply_qcak_encoding(x, s_params, opt_qubits, sar_qubits, 
                        attention_pairs, n_qubits=8, reps=2):
    """
    QCAK encoding circuit. Called inside a qml.qnode context.
    Modifies quantum state in-place via PennyLane gate operations.
    
    Args:
        x:               Bandwidth-scaled feature vector, shape (n_qubits,).
                         Values in [0, ~pi/2]. Already pre-scaled externally.
        s_params:        Attention scale parameters, shape (reps, n_pairs).
                         s_params[rep, pair_idx] scales the bilinear angle
                         for attention_pairs[pair_idx] at repetition rep.
        opt_qubits:      List of qubit indices carrying OPT features, e.g. [0,1,2,3]
        sar_qubits:      List of qubit indices carrying SAR features, e.g. [4,5,6,7]
        attention_pairs: List of (sar_qubit_idx, opt_qubit_idx) pairs.
                         These are QUBIT INDICES (0-7).
        n_qubits:        Total qubits.
        reps:            Number of circuit repetitions.
    """
    for rep in range(reps):
        # Layer 1: H on all, then RZ(x[i]) on all
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(float(x[i]), wires=i)

        # Layer 2a: Linear ZZ within opt_qubits
        if len(opt_qubits) > 1:
            for k in range(len(opt_qubits) - 1):
                i, j = opt_qubits[k], opt_qubits[k+1]
                angle = (np.pi - float(x[i])) * (np.pi - float(x[j]))
                qml.CNOT(wires=[i, j])
                qml.RZ(angle, wires=j)
                qml.CNOT(wires=[i, j])

        # Layer 2b: Linear ZZ within sar_qubits
        if len(sar_qubits) > 1:
            for k in range(len(sar_qubits) - 1):
                i, j = sar_qubits[k], sar_qubits[k+1]
                angle = (np.pi - float(x[i])) * (np.pi - float(x[j]))
                qml.CNOT(wires=[i, j])
                qml.RZ(angle, wires=j)
                qml.CNOT(wires=[i, j])

        # Layer 3: Unidirectional cross-attention
        # Only Direction A: OPT qubit attends to SAR qubit.
        # This reduction in depth prevents exponential concentration while
        # preserving cross-modal interaction encoding.
        for pair_idx, (sar_q, opt_q) in enumerate(attention_pairs):
            if sar_q == opt_q or sar_q >= n_qubits or opt_q >= n_qubits:
                continue
            
            # Angle is s_ij * x_i * x_j
            theta = float(s_params[rep, pair_idx]) * float(x[sar_q]) * float(x[opt_q])
            
            # Direction A only: SAR controls phase of OPT qubit
            qml.CNOT(wires=[sar_q, opt_q])
            qml.RZ(theta, wires=opt_q)
            qml.CNOT(wires=[sar_q, opt_q])


def compute_qcak_pqk_bloch_vectors(X_enc, s_params, opt_qubits, sar_qubits,
                                    attention_pairs, n_qubits=8, reps=2,
                                    save_path=None):
    """
    Compute Bloch vectors for all samples using the QCAK encoding circuit.
    
    Measures ⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩ for each qubit i after apply_qcak_encoding.
    Returns bloch vectors in the format: (n_samples, 3*n_qubits)
    """
    if save_path and os.path.exists(save_path):
        print(f"  [SKIP] Loading cached QCAK-PQK Bloch vectors: {save_path}")
        return np.load(save_path)

    dev = config.get_device(n_qubits)

    # Building QNodes with diff_method=None as they are only used for measurement.
    @qml.qnode(dev, diff_method=None)
    def measure_x(x):
        apply_qcak_encoding(x, s_params, opt_qubits, sar_qubits,
                            attention_pairs, n_qubits, reps)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def measure_y(x):
        apply_qcak_encoding(x, s_params, opt_qubits, sar_qubits,
                            attention_pairs, n_qubits, reps)
        return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def measure_z(x):
        apply_qcak_encoding(x, s_params, opt_qubits, sar_qubits,
                            attention_pairs, n_qubits, reps)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    n_samples = X_enc.shape[0]
    bloch = np.zeros((n_samples, 3 * n_qubits), dtype=np.float64)

    for i in tqdm(range(n_samples), desc="QCAK-PQK Bloch vectors"):
        x_i = X_enc[i]
        
        # Measure ⟨X_q⟩, ⟨Y_q⟩, ⟨Z_q⟩ per qubit
        mx = measure_x(x_i)
        my = measure_y(x_i)
        mz = measure_z(x_i)
        
        for q in range(n_qubits):
            bloch[i, 3*q]     = mx[q]
            bloch[i, 3*q + 1] = my[q]
            bloch[i, 3*q + 2] = mz[q]

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, bloch)
        print(f"  Saved QCAK-PQK Bloch vectors: {save_path}")

    return bloch


def compute_qcak_pqk_kernel_from_bloch(bloch1, bloch2, outer_gamma, n_qubits=8):
    """
    Compute outer RBF kernel from QCAK-PQK Bloch vectors.
    K(x,x') = exp(-outer_gamma * 0.5 * sum_i ||b_i(x) - b_i(x')||^2)
    """
    n1 = bloch1.shape[0]
    n2 = bloch2.shape[0]

    b1 = bloch1.reshape(n1, n_qubits, 3)
    b2 = bloch2.reshape(n2, n_qubits, 3)

    K = np.zeros((n1, n2), dtype=np.float64)
    for i in tqdm(range(n1), desc="QCAK-PQK kernel", leave=False):
        # Vectorized over all j in bloch2
        diff = b1[i] - b2          # (n2, n_qubits, 3)
        sq   = np.sum(diff**2, axis=2)     # (n2, n_qubits)
        dist = 0.5 * np.sum(sq, axis=1)    # (n2,)
        K[i, :] = np.exp(-outer_gamma * dist)
    
    return K
