#!/usr/bin/env python3
"""
Experiment E29: Pure SRQFM (Global Fidelity) vs FQK on So2Sat LCZ42
===================================================================

This experiment rigorously evaluates the pure global fidelity formulation of the
Self-Regulating Quantum Feature Map (SRQFM) against the standard IBM ZZFeatureMap (FQK)
and classical baselines on the primary So2Sat LCZ42 dataset.

Unlike SRQFM-PQK (which extracts Bloch vectors and applies a classical RBF kernel),
this script computes the exact quantum state overlap: |<psi(x') | psi(x)>|^2.

Data Parity:
    This script strictly uses `load_physics_16` and `load_pca_8` from `_e_common.py`
    to ensure the exact same 2,000 samples and feature representations used in E1-E28
    are evaluated here.

Usage:
    python experiments/exp_e29_pure_srqfm_so2sat.py
"""

import os
import sys
import json
import time
import logging
import warnings
from datetime import datetime

import numpy as np
from scipy.stats import ttest_rel

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

warnings.filterwarnings("ignore")

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from experiments._e_common import load_physics_16, load_pca_8, svm_eval, cohens_d
from experiments.exp_e22_srqfm_benchmark import evaluate_classical_baselines
from src.srqfm_kernel import compute_kta

RESULTS_DIR = os.path.join(config.RESULTS_DIR, "pure_srqfm_so2sat")
KERNEL_CACHE_DIR = os.path.join(RESULTS_DIR, "kernels")
FQK_REPS = 2
N_QUBITS = 8

# ── Pure Quantum Kernels (No RBF) ──────────────────────────────────────────

def build_pure_fqk_matrix(X):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit(x1, x2):
        for _ in range(FQK_REPS):
            for i in range(N_QUBITS):
                qml.Hadamard(wires=i)
                qml.RZ(x1[i], wires=i)
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ((np.pi - x1[i]) * (np.pi - x1[i + 1]), wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
        for _ in range(FQK_REPS):
            for i in range(N_QUBITS - 2, -1, -1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(-(np.pi - x2[i]) * (np.pi - x2[i + 1]), wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            for i in range(N_QUBITS - 1, -1, -1):
                qml.RZ(-x2[i], wires=i)
                qml.Hadamard(wires=i)
        return qml.probs(wires=range(N_QUBITS))

    N = len(X)
    K = np.zeros((N, N))
    pbar = tqdm(total=N * N, desc=f"    FQK ({N}x{N})", unit="eval")
    for i in range(N):
        for j in range(N):
            K[i, j] = circuit(X[i], X[j])[0]
            pbar.update(1)
    pbar.close()
    return K

def build_pure_srqfm_matrix(X):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit(x1, x2):
        for _ in range(FQK_REPS):
            for i in range(N_QUBITS):
                qml.Hadamard(wires=i)
                qml.RZ(x1[i], wires=i)
            for i in range(N_QUBITS - 1):
                c = np.sin((x1[i] - x1[i + 1]) / 2.0) ** 2
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(c, wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
        for _ in range(FQK_REPS):
            for i in range(N_QUBITS - 2, -1, -1):
                c = np.sin((x2[i] - x2[i + 1]) / 2.0) ** 2
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(-c, wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            for i in range(N_QUBITS - 1, -1, -1):
                qml.RZ(-x2[i], wires=i)
                qml.Hadamard(wires=i)
        return qml.probs(wires=range(N_QUBITS))

    N = len(X)
    K = np.zeros((N, N))
    pbar = tqdm(total=N * N, desc=f"    SRQFM ({N}x{N})", unit="eval")
    for i in range(N):
        for j in range(N):
            K[i, j] = circuit(X[i], X[j])[0]
            pbar.update(1)
    pbar.close()
    return K

def cache_io(name, feat_name, func, X):
    """Checkpointing wrapper for full N x N matrix. Checks global results paths for FQK."""
    # Check if FQK is already computed globally by previous E-series exps
    if name == "FQK":
        if feat_name == "pca8":
            p_global = os.path.join(config.RESULTS_DIR, "pca_cv", "full_kernels", "K_fqk_full.npy")
        else:
            p_global = os.path.join(config.RESULTS_DIR, "physics_cv", "full_kernels", "K_fqk_full.npy")
            
        if os.path.exists(p_global):
            logging.info(f"    Loaded FQK from global E-series cache: {p_global}")
            return np.load(p_global)

    # Local fallback/cache for SRQFM
    os.makedirs(KERNEL_CACHE_DIR, exist_ok=True)
    p = os.path.join(KERNEL_CACHE_DIR, f"{name}_{feat_name}_full.npy")
    if os.path.exists(p):
        logging.info(f"    Loaded {name} from local cache.")
        return np.load(p)
        
    logging.info(f"    Computing {name} full matrix...")
    K = func(X)
    np.save(p, K)
    return K

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger()
    
    logger.info("=" * 60)
    logger.info(f"E29: Pure SRQFM vs FQK on So2Sat (N=2000)")
    logger.info("=" * 60)

    # 1. Load Data with exact E-series Parity
    t0 = time.time()
    logger.info("Loading cached E-series data (2000 samples)...")
    
    # Physics 16 -> Slice to top 8
    X_phys_norm, X_phys_raw, y_phys = load_physics_16()
    X_phys_norm = X_phys_norm[:, :N_QUBITS]
    X_phys_raw = X_phys_raw[:, :N_QUBITS]
    
    # PCA 8 (Fused)
    X_pca_norm, y_pca = load_pca_8()
    
    # Standardize scale for PCA to [0, pi] since load_pca_8 returns pre-scaled from another exp
    # We ensure strict [0, pi] mapping for pure quantum phase encoding
    X_pca_norm = (X_pca_norm - X_pca_norm.min(axis=0)) / (X_pca_norm.max(axis=0) - X_pca_norm.min(axis=0)) * np.pi
    
    datasets = {
        "physics8": (X_phys_norm, X_phys_raw, y_phys),
        "pca8": (X_pca_norm, X_pca_norm, y_pca)  # Raw is same for PCA baseline comparison
    }
    
    results = {"timestamp": datetime.now().isoformat()}
    seeds = config.SEED_LIST

    # 2. Loop over feature sets
    for feat_name, (X_norm, X_raw, y) in datasets.items():
        logger.info(f"\n" + "-" * 50)
        logger.info(f" Evaluating Feature Set: {feat_name.upper()}")
        logger.info("-" * 50)
        
        cv_results = {"FQK": [], "SRQFM": []}
        
        # Compute/Load Full Matrices
        K_fqk = cache_io("FQK", feat_name, build_pure_fqk_matrix, X_norm)
        K_srqfm = cache_io("SRQFM", feat_name, build_pure_srqfm_matrix, X_norm)
        
        logger.info(f"  FQK   KTA = {compute_kta(K_fqk, y):.4f}")
        logger.info(f"  SRQFM KTA = {compute_kta(K_srqfm, y):.4f}")
        
        logger.info("\n  Running Cross-Validation...")
        for seed in seeds:
            # Evaluate Quantum (precomputed full matrix)
            res_fqk = svm_eval(K_fqk, y, seed)
            res_srqfm = svm_eval(K_srqfm, y, seed)
            
            cv_results["FQK"].append(res_fqk)
            cv_results["SRQFM"].append(res_srqfm)
            
            logger.info(f"    Seed {seed:2d} | FQK F1: {res_fqk['macro_f1']:.4f} | SRQFM F1: {res_srqfm['macro_f1']:.4f}")
            
        # Add Classical Baselines (evaluated dynamically)
        cl_res = evaluate_classical_baselines(X_raw, y, 42) # Evaluate once per feature set using standard logic
        for cl_name, cl_metrics in cl_res.items():
            cv_results[cl_name] = [cl_metrics] * len(seeds) # Duplicate for summary structure
            logger.info(f"    Baseline | {cl_name}: F1={cl_metrics['macro_f1']:.4f}")
            
        # Aggregate
        feat_summary = {}
        for name in cv_results:
            f1s = [r["macro_f1"] for r in cv_results[name]]
            feat_summary[name] = {"f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))}
            
        results[feat_name] = feat_summary
        
    results["total_time_s"] = time.time() - t0
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"\nDone in {results['total_time_s']:.0f}s. Results saved to {RESULTS_DIR}/results.json")

if __name__ == "__main__":
    main()
