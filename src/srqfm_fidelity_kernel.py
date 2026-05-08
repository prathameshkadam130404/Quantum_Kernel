"""
SRQFM Fidelity Kernel (companion to BSCM, for apples-to-apples comparison).

The canonical SRQFM module (src/srqfm_kernel.py) implements the *projected*
variant K = exp(-gamma ||b(x) - b(x')||^2 / 2) where b(x) is the per-qubit
Bloch vector.  That is what the fair-tuning experiment E26 evaluated.

This module implements the *fidelity* variant
    K(x, x') = | <psi(x') | psi(x)> |^2
            = | <0...0 | U_SRQFM^dagger(x') U_SRQFM(x) | 0...0> |^2
with the same all-to-all H + RZ(x) + IsingZZ(sin^2((x_i - x_j)/2))
feature map.  This matches the construction used in the EuroSAT experiment
script and is the appropriate baseline against which to compare the BSCM
fidelity kernel.

Author: Prathamesh Kadam et al.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


def fidelity_coupling(x_i: float, x_j: float) -> float:
    return float(np.sin((x_i - x_j) / 2.0) ** 2)


def _resolve_pairs(n_qubits: int, connectivity: str) -> List[Tuple[int, int]]:
    if connectivity == "all":
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    if connectivity == "linear":
        return [(i, i + 1) for i in range(n_qubits - 1)]
    raise ValueError(f"Unknown connectivity: {connectivity}")


def _apply_srqfm(x, n_qubits, reps, threshold, connectivity):
    import pennylane as qml

    pairs = _resolve_pairs(n_qubits, connectivity)
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        for qi, qj in pairs:
            c = fidelity_coupling(x[qi], x[qj])
            if c > threshold:
                qml.IsingZZ(c, wires=[qi, qj])


def _apply_srqfm_adjoint(x, n_qubits, reps, threshold, connectivity):
    import pennylane as qml

    pairs = _resolve_pairs(n_qubits, connectivity)
    for _ in range(reps):
        for qi, qj in reversed(pairs):
            c = fidelity_coupling(x[qi], x[qj])
            if c > threshold:
                qml.IsingZZ(-c, wires=[qi, qj])
        for i in range(n_qubits - 1, -1, -1):
            qml.RZ(-x[i], wires=i)
        for i in range(n_qubits - 1, -1, -1):
            qml.Hadamard(wires=i)


def compute_srqfm_fidelity_kernel(
    X: np.ndarray,
    X2: Optional[np.ndarray] = None,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    coupling_threshold: float = 0.01,
    connectivity: str = "all",
    kernel_save_path: Optional[str] = None,
    desc: str = "SRQFM-fid",
) -> np.ndarray:
    import pennylane as qml

    if kernel_save_path is not None and os.path.exists(kernel_save_path):
        logger.info(f"[CACHE] Loading SRQFM-fidelity kernel from {kernel_save_path}")
        return np.load(kernel_save_path)

    dev = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def circuit(x1, x2):
        _apply_srqfm(x1, n_qubits, reps, coupling_threshold, connectivity)
        _apply_srqfm_adjoint(x2, n_qubits, reps, coupling_threshold, connectivity)
        return qml.probs(wires=range(n_qubits))

    if kernel_save_path is not None:
        os.makedirs(os.path.dirname(kernel_save_path), exist_ok=True)

    n = len(X)
    if X2 is None:
        ckpt_path = (kernel_save_path + ".ckpt.npz") if kernel_save_path else None
        start_i = 0
        if ckpt_path is not None and os.path.exists(ckpt_path):
            ck = np.load(ckpt_path)
            if ck["n"].item() == n:
                K = ck["K"].copy()
                start_i = int(ck["next_i"].item())
                logger.info("[CKPT] Resuming %s from row %d/%d", desc, start_i, n)
            else:
                K = np.ones((n, n), dtype=np.float64)
        else:
            K = np.ones((n, n), dtype=np.float64)

        n_evals_total = n * (n - 1) // 2
        n_evals_done = start_i * n - start_i * (start_i + 1) // 2
        pbar = tqdm(total=n_evals_total, desc=f"{desc} K({n}x{n})", unit="eval",
                    initial=n_evals_done)
        for i in range(start_i, n):
            for j in range(i + 1, n):
                p0 = float(circuit(X[i], X[j])[0])
                K[i, j] = p0
                K[j, i] = p0
                pbar.update(1)
            if ckpt_path is not None and ((i + 1) % 25 == 0 or i == n - 1):
                np.savez(ckpt_path, K=K, next_i=np.array(i + 1), n=np.array(n))
        pbar.close()
    else:
        m = len(X2)
        K = np.zeros((m, n), dtype=np.float64)
        pbar = tqdm(total=m * n, desc=f"{desc} K({m}x{n})", unit="eval")
        for i in range(m):
            for j in range(n):
                K[i, j] = float(circuit(X[j], X2[i])[0])
                pbar.update(1)
        pbar.close()

    if kernel_save_path is not None:
        os.makedirs(os.path.dirname(kernel_save_path), exist_ok=True)
        np.save(kernel_save_path, K)
        logger.info(f"Saved SRQFM-fidelity kernel to {kernel_save_path}")
        ckpt_path = kernel_save_path + ".ckpt.npz"
        try:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
        except OSError:
            pass  # Windows occasionally holds the file briefly; non-critical.

    return K
