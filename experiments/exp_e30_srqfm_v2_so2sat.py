#!/usr/bin/env python3
"""
Experiment E30: SRQFM-v2 (Active Regulation) on So2Sat LCZ42
============================================================

This experiment evaluates the upgraded SRQFM-v2 architecture:
1. Increased Depth: 3 repetitions (vs 2 in v1).
2. All-to-All Connectivity: Captures long-range spectral correlations.
3. Contrast Amplification: A=5.0 scaling on the fidelity coupling.

Goal: Demonstrate that SRQFM-v2 beats the standard ZZ-FQK baseline on
the complex, fine-grained 17-class So2Sat dataset.

Usage:
    python experiments/exp_e30_srqfm_v2_so2sat.py
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from tqdm import tqdm

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from experiments._e_common import load_physics_16, load_pca_8
from src.srqfm_kernel import compute_kta

# --- Config ---
RESULTS_DIR = os.path.join(config.RESULTS_DIR, "srqfm_v2_so2sat")
KERNEL_CACHE_DIR = os.path.join(RESULTS_DIR, "kernels")
E29_CACHE_DIR = os.path.join(config.RESULTS_DIR, "pure_srqfm_so2sat", "kernels")

V2_REPS = 3
V2_MULTIPLIER = 5.0
V2_CONNECTIVITY = "all"
N_QUBITS = 8

# --- Kernel Implementation ---

def apply_srqfm_v2_logic(x, n_qubits, reps, multiplier):
    """Local implementation of SRQFM-v2 logic for E30."""
    import pennylane as qml
    pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        for qi, qj in pairs:
            c = np.sin((x[qi] - x[qj]) / 2.0) ** 2 * multiplier
            if c > 0.01:
                qml.IsingZZ(c, wires=[qi, qj])

def apply_srqfm_v2_adjoint(x, n_qubits, reps, multiplier):
    """Local implementation of the adjoint SRQFM-v2 logic."""
    import pennylane as qml
    pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    
    for _ in range(reps):
        for qi, qj in reversed(pairs):
            c = np.sin((x[qi] - x[qj]) / 2.0) ** 2 * multiplier
            if c > 0.01:
                qml.IsingZZ(-c, wires=[qi, qj])
        for i in range(n_qubits - 1, -1, -1):
            qml.RZ(-x[i], wires=i)
        for i in range(n_qubits - 1, -1, -1):
            qml.Hadamard(wires=i)

def apply_srqfm_v2_logic(x, n_qubits, reps, multiplier):
    """Local implementation of SRQFM-v2 logic for E30."""
    import pennylane as qml
    pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        for qi, qj in pairs:
            c = np.sin((x[qi] - x[qj]) / 2.0) ** 2 * multiplier
            if c > 0.01:
                qml.IsingZZ(c, wires=[qi, qj])

def apply_srqfm_v2_adjoint(x, n_qubits, reps, multiplier):
    """Local implementation of the adjoint SRQFM-v2 logic."""
    import pennylane as qml
    pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    
    for _ in range(reps):
        for qi, qj in reversed(pairs):
            c = np.sin((x[qi] - x[qj]) / 2.0) ** 2 * multiplier
            if c > 0.01:
                qml.IsingZZ(-c, wires=[qi, qj])
        for i in range(n_qubits - 1, -1, -1):
            qml.RZ(-x[i], wires=i)
        for i in range(n_qubits - 1, -1, -1):
            qml.Hadamard(wires=i)

def build_srqfm_v2_matrix(X):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=N_QUBITS)
    @qml.qnode(dev)
    def circuit(x1, x2):
        apply_srqfm_v2_logic(x1, N_QUBITS, V2_REPS, V2_MULTIPLIER)
        apply_srqfm_v2_adjoint(x2, N_QUBITS, V2_REPS, V2_MULTIPLIER)
        return qml.probs(wires=range(N_QUBITS))
    N = len(X)
    K = np.zeros((N, N))
    pbar = tqdm(total=N * N, desc=f"    SRQFM-v2 ({N}x{N})", unit="eval")
    for i in range(N):
        for j in range(N):
            K[i, j] = circuit(X[i], X[j])[0]
            pbar.update(1)
    pbar.close()
    return K

def apply_srqfm_v3_logic(x, n_qubits, reps):
    """Local implementation of SRQFM-v3 (Adaptive) logic."""
    import pennylane as qml
    pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        for qi, qj in pairs:
            # Adaptive Multiplier: pi / sum_intensity (capped at 10.0)
            multiplier = min(10.0, np.pi / (x[qi] + x[qj] + 0.01))
            c = np.sin((x[qi] - x[qj]) / 2.0) ** 2 * multiplier
            if c > 0.01:
                qml.IsingZZ(c, wires=[qi, qj])

def apply_srqfm_v3_adjoint(x, n_qubits, reps):
    """Local implementation of the adjoint SRQFM-v3 logic."""
    import pennylane as qml
    pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    for _ in range(reps):
        for qi, qj in reversed(pairs):
            multiplier = min(10.0, np.pi / (x[qi] + x[qj] + 0.01))
            c = np.sin((x[qi] - x[qj]) / 2.0) ** 2 * multiplier
            if c > 0.01:
                qml.IsingZZ(-c, wires=[qi, qj])
        for i in range(n_qubits - 1, -1, -1):
            qml.RZ(-x[i], wires=i)
        for i in range(n_qubits - 1, -1, -1):
            qml.Hadamard(wires=i)

def build_srqfm_v3_matrix(X):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=N_QUBITS)
    @qml.qnode(dev)
    def circuit(x1, x2):
        apply_srqfm_v3_logic(x1, N_QUBITS, V2_REPS)
        apply_srqfm_v3_adjoint(x2, N_QUBITS, V2_REPS)
        return qml.probs(wires=range(N_QUBITS))
    N = len(X)
    K = np.zeros((N, N))
    pbar = tqdm(total=N * N, desc=f"    SRQFM-v3 ({N}x{N})", unit="eval")
    for i in range(N):
        for j in range(N):
            K[i, j] = circuit(X[i], X[j])[0]
            pbar.update(1)
    pbar.close()
    return K

# --- Helpers ---

def tuned_svm_eval(K_full, y, seed, test_frac=0.3):
    """SVM evaluation with GridSearch for C tuning."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)
    
    K_tr = K_full[np.ix_(tr, tr)]
    K_te = K_full[np.ix_(te, tr)]
    
    # Tune C parameter to be fair to the kernel's alignment
    param_grid = {'C': [0.1, 1.0, 10.0, 100.0]}
    grid = GridSearchCV(
        SVC(kernel="precomputed", class_weight="balanced"),
        param_grid, cv=3, scoring='f1_macro', n_jobs=-1
    )
    grid.fit(K_tr, y[tr])
    
    best_clf = grid.best_estimator_
    yp = best_clf.predict(K_te)
    
    return {
        "macro_f1": float(f1_score(y[te], yp, average="macro")),
        "best_C": float(grid.best_params_['C']),
        "train_idx": tr, "test_idx": te
    }

def load_cached_fqk(feat_name):
    """Load FQK matrix from primary E-series global cache."""
    if feat_name == "pca8":
        p = os.path.join(config.RESULTS_DIR, "pca_cv", "full_kernels", "K_fqk_full.npy")
    else:
        p = os.path.join(config.RESULTS_DIR, "physics_cv", "full_kernels", "K_fqk_full.npy")
        
    if os.path.exists(p):
        return np.load(p)
    
    # Secondary fallback for local caches
    p_alt = os.path.join(E29_CACHE_DIR, f"FQK_{feat_name}_full.npy")
    if os.path.exists(p_alt):
        return np.load(p_alt)
        
    return None

# --- Main ---

def main():
    os.makedirs(KERNEL_CACHE_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger()
    
    logger.info("=" * 60)
    logger.info("E30: SRQFM-v2 vs SRQFM-v3 (Adaptive) vs FQK on So2Sat")
    logger.info(f"Settings: Reps={V2_REPS}, Connectivity=all")
    logger.info("=" * 60)

    # 1. Load Data
    X_phys_norm, _, y_phys = load_physics_16()
    X_phys_norm = X_phys_norm[:, :N_QUBITS]
    
    X_pca_norm, y_pca = load_pca_8()
    # Normalize PCA for quantum phase encoding [0, pi]
    X_pca_norm = (X_pca_norm - X_pca_norm.min(axis=0)) / (X_pca_norm.max(axis=0) - X_pca_norm.min(axis=0)) * np.pi
    
    datasets = {
        "physics8": (X_phys_norm, y_phys),
        "pca8": (X_pca_norm, y_pca)
    }
    
    results = {"timestamp": datetime.now().isoformat()}
    seeds = config.SEED_LIST # Standard 5 seeds [42, 43, 44, 45, 46]

    for feat_name, (X, y) in datasets.items():
        logger.info(f"\nEvaluating Feature Set: {feat_name.upper()}")
        
        # Load FQK (Baseline)
        K_fqk = load_cached_fqk(feat_name)
        
        # Compute/Load SRQFM-v2
        p_v2 = os.path.join(KERNEL_CACHE_DIR, f"SRQFM_v2_{feat_name}.npy")
        if os.path.exists(p_v2):
            logger.info(f"  Loading cached SRQFM-v2: {p_v2}")
            K_v2 = np.load(p_v2)
        else:
            logger.info(f"  Computing SRQFM-v2...")
            K_v2 = build_srqfm_v2_matrix(X)
            np.save(p_v2, K_v2)
            
        # Compute/Load SRQFM-v3 (Adaptive)
        p_v3 = os.path.join(KERNEL_CACHE_DIR, f"SRQFM_v3_{feat_name}.npy")
        if os.path.exists(p_v3):
            logger.info(f"  Loading cached SRQFM-v3: {p_v3}")
            K_v3 = np.load(p_v3)
        else:
            logger.info(f"  Computing SRQFM-v3 (Adaptive)...")
            K_v3 = build_srqfm_v3_matrix(X)
            np.save(p_v3, K_v3)
            
        logger.info(f"  FQK KTA = {compute_kta(K_fqk, y):.4f}" if K_fqk is not None else "  FQK: No Cache")
        logger.info(f"  SRQFM-v2 KTA = {compute_kta(K_v2, y):.4f}")
        logger.info(f"  SRQFM-v3 KTA = {compute_kta(K_v3, y):.4f}")
        
        # Cross-Validation
        cv_res = {"FQK": [], "SRQFM-v2": [], "SRQFM-v3": []}
        for seed in seeds:
            res_v2 = tuned_svm_eval(K_v2, y, seed)
            res_v3 = tuned_svm_eval(K_v3, y, seed)
            cv_res["SRQFM-v2"].append(res_v2["macro_f1"])
            cv_res["SRQFM-v3"].append(res_v3["macro_f1"])
            
            line = f"    Seed {seed} | v2 F1: {res_v2['macro_f1']:.4f} | v3 F1: {res_v3['macro_f1']:.4f}"
            
            if K_fqk is not None:
                res_fqk = tuned_svm_eval(K_fqk, y, seed)
                cv_res["FQK"].append(res_fqk["macro_f1"])
                line = f"    Seed {seed} | FQK F1: {res_fqk['macro_f1']:.4f} | " + line
            
            logger.info(line)
            
        results[feat_name] = {
            "FQK": {"mean": np.mean(cv_res["FQK"]), "std": np.std(cv_res["FQK"])} if cv_res["FQK"] else None,
            "SRQFM-v2": {"mean": np.mean(cv_res["SRQFM-v2"]), "std": np.std(cv_res["SRQFM-v2"])},
            "SRQFM-v3": {"mean": np.mean(cv_res["SRQFM-v3"]), "std": np.std(cv_res["SRQFM-v3"])}
        }

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {RESULTS_DIR}/results.json")

if __name__ == "__main__":
    main()
