"""
Experiment E7 — Entanglement topology comparison for FQK.

Three FQK variants on physics-16 features (top-8 Fisher-selected), N=2000:
    linear       : (i, i+1) for i=0..n-2
    full         : all pairs (i, j), i<j
    cross-modal  : linear + top-3 SAR-optical MI pairs (CM-FQK)

Report Macro-F1, ΔKTA, off-diagonal variance, effective rank per seed.

Output: results/topology/{K_*.npy, metrics.csv}
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pennylane as qml
from tqdm import tqdm

import config
from src.utils import ensure_dir, setup_logging
from src.attention_kernel import select_features_by_fisher
from src.quantum_kernels import get_top_mi_pairs
from src.kernel_target_alignment import compute_centered_kta
from experiments._e_common import load_physics_16, spectral_metrics, svm_eval

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "topology"))
logger = setup_logging("exp_e7", log_file=os.path.join(RES, "e7.log"))

N_QUBITS, REPS = config.N_QUBITS, config.ZZ_REPS


def apply_zz(x, pairs, n_qubits, reps):
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        for (i, j) in pairs:
            zz = (np.pi - x[i]) * (np.pi - x[j])
            qml.CNOT(wires=[i, j]); qml.RZ(zz, wires=j); qml.CNOT(wires=[i, j])


def build_fqk_circuit(pairs, n_qubits=N_QUBITS, reps=REPS):
    dev = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def circ(x1, x2):
        apply_zz(x1, pairs, n_qubits, reps)
        qml.adjoint(apply_zz)(x2, pairs, n_qubits, reps)
        return qml.probs(wires=range(n_qubits))
    return circ


def compute_K(X, pairs):
    circ = build_fqk_circuit(pairs)
    n = len(X); K = np.eye(n)
    total = n * (n - 1) // 2
    with tqdm(total=total, desc="FQK", leave=False) as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                K[i, j] = K[j, i] = float(circ(X[i], X[j])[0])
                pbar.update(1)
    return K


def run():
    X_norm, X_raw, y = load_physics_16()
    selected, _ = select_features_by_fisher(X_raw, y, n_select=N_QUBITS,
                                            min_sar=4, min_opt=4)
    X_sel = X_norm[:, selected]

    linear = [(i, i + 1) for i in range(N_QUBITS - 1)]
    full   = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)]
    mi     = get_top_mi_pairs(X_raw, n_sar=4, n_opt=4,
                              top_k=config.CM_FQK_TOP_K_PAIRS)
    # Map raw 16-index MI pairs onto 8-qubit space via selected_indices mapping
    idx_map = {int(f): q for q, f in enumerate(selected)}
    mi_qubit = [(idx_map[a], idx_map[b]) for a, b in mi
                if int(a) in idx_map and int(b) in idx_map]
    cross_modal = list({*linear, *mi_qubit})

    topologies = {"linear": linear, "full": full, "cross-modal": cross_modal}

    rows = []
    for tname, pairs in topologies.items():
        path = os.path.join(RES, f"K_{tname}.npy")
        if os.path.exists(path):
            K = np.load(path)
        else:
            logger.info(f"Computing FQK-{tname}, |pairs|={len(pairs)}")
            K = compute_K(X_sel, pairs); np.save(path, K)
        spm  = spectral_metrics(K)
        kta  = float(compute_centered_kta(K, y, class_weighted=True))
        for seed in config.SEED_LIST:
            ev = svm_eval(K, y, seed)
            rows.append(dict(topology=tname, seed=seed,
                             macro_f1=ev["macro_f1"],
                             accuracy=ev["accuracy"], kta=kta, **spm))

    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "metrics.csv"),
                                        index=False)
    with open(os.path.join(RES, "topologies.json"), "w") as f:
        json.dump({k: [list(p) for p in v] for k, v in topologies.items()},
                  f, indent=2)


if __name__ == "__main__":
    run()
