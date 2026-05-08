"""
E24: Pure Quantum-Heavy Pipeline — Fidelity Quantum Kernels on Raw Data.

This is the definitive test of the SRQFM feature map's capability to learn
feature interactions natively, without relying on classical physics formulas
or classical kernel projections (Gamma/RBF).

Pipeline:
1. Data Encoding: Spatial means of 8 raw bands (no physics domain knowledge).
   - Optical: B2, B3, B4, B8, B11
   - SAR: VH, VV, PolSAR_Cov_Magnitude
2. Quantum Feature Maps (2 reps):
   - Standard FQK: Havlicek (pi-x_i)(pi-x_j) blind entanglement.
   - SRQFM-FQK: sin²((x_i-x_j)/2) self-regulating entanglement.
3. Measurement: Global State Fidelity K(x,x') = |⟨ψ(x)|ψ(x')⟩|²
   - Zero classical hyperparameters (no Gamma).
4. Evaluation: 5-seed SVM on the precomputed kernel matrices.

Author: Prathamesh Kadam et al.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_rel
import pennylane as qml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._e_common import spectral_metrics, svm_eval, cohens_d
from experiments.exp_e22_srqfm_benchmark import evaluate_classical_baselines
from src.srqfm_kernel import compute_kta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("exp_e24")

# Parameters
N_QUBITS = 8
REPS = config.ZZ_REPS
SEEDS = config.SEED_LIST
SAVE_DIR = os.path.join(config.RESULTS_DIR, "pure_quantum_pipeline")

# Setup PennyLane device (lightning is fastest for exact statevector)
dev = qml.device("lightning.qubit", wires=N_QUBITS)


# ================== STATEVECTOR QNODES ==================

@qml.qnode(dev, diff_method=None)
def statevector_standard_zz(x, reps=REPS):
    """Standard Havlicek ZZ Feature Map: U(x)|0>."""
    for _ in range(reps):
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)
        for i in range(N_QUBITS):
            qml.RZ(x[i], wires=i)
        
        # All-to-all blind entanglement
        for i in range(N_QUBITS):
            for j in range(i + 1, N_QUBITS):
                coupling = (np.pi - x[i]) * (np.pi - x[j])
                qml.IsingZZ(coupling, wires=[i, j])
                
    return qml.state()


@qml.qnode(dev, diff_method=None)
def statevector_srqfm(x, reps=REPS):
    """SRQFM Feature Map: U(x)|0> with fidelity coupling."""
    for _ in range(reps):
        for i in range(N_QUBITS):
            qml.Hadamard(wires=i)
        for i in range(N_QUBITS):
            qml.RZ(x[i], wires=i)
        
        # All-to-all self-regulating entanglement
        for i in range(N_QUBITS):
            for j in range(i + 1, N_QUBITS):
                coupling = np.sin((x[i] - x[j]) / 2.0)**2
                # Only entangle if structurally meaningful
                if coupling > 0.01:
                    qml.IsingZZ(coupling, wires=[i, j])
                    
    return qml.state()


def compute_fqk_kernel_matrix(X, state_func, batch_size=200):
    """
    Compute exactly K_ij = |⟨ψ(x_i)|ψ(x_j)⟩|² using full statevectors.
    Dramatically faster than running U(x) U^dagger(x') for every pair.
    """
    N = len(X)
    dim = 2**N_QUBITS
    
    logger.info(f"Computing statevectors for {N} samples...")
    states = np.zeros((N, dim), dtype=np.complex128)
    
    t0 = time.time()
    for i in tqdm(range(N), desc="Statevector extraction", leave=False):
        states[i] = state_func(X[i])
    dt_state = time.time() - t0
    
    logger.info(f"Computing exact fidelity kernel matrix ({N}x{N})...")
    # Gram matrix K = |V V^dagger|^2
    t0 = time.time()
    K = np.abs(states @ states.conj().T)**2
    dt_kernel = time.time() - t0
    
    # Numerical stability
    np.fill_diagonal(K, 1.0)
    K = np.clip(K, 0, 1.0)
    
    return K, dt_state + dt_kernel


# ================== MAIN EXPERIMENT ==================

def run_experiment():
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    fh = logging.FileHandler(os.path.join(SAVE_DIR, "exp_e24.log"), mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    ))
    logging.getLogger().addHandler(fh)
    
    logger.info("=" * 70)
    logger.info("  E24: Pure Quantum-Heavy Pipeline — FQK on Raw Data")
    logger.info("=" * 70)
    logger.info(f"  Date: {datetime.now().isoformat()}")
    
    # 1. Load Raw Band Means data
    data_path = os.path.join(config.PROCESSED_DIR, "raw_band_features_8.npz")
    if not os.path.exists(data_path):
        logger.error(f"Data not found: {data_path}")
        logger.error("Run 'python scripts/extract_raw_band_features.py' first.")
        return
        
    data = np.load(data_path)
    X_norm = data["X_norm"]    # [0, pi]
    X_raw = data["X_raw"]      # Original scale (for RF/RBF)
    y = data["y"]
    names = data["feature_names"]
    
    logger.info(f"\n--- Loading Raw Data ---")
    logger.info(f"  N = {len(X_norm)}, Features = {len(names)}")
    logger.info(f"  Bands: {list(names)}")
    
    # 2. Compute FQK Kernels
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 1: FQK Computations (No classical gamma)")
    logger.info("=" * 70)
    
    kernels = {}
    timings = {}
    
    logger.info("\n--- 1/2: Standard FQK (Havlicek ZZ) ---")
    K_zz, dt_zz = compute_fqk_kernel_matrix(X_norm, statevector_standard_zz)
    kernels["FQK_Standard"] = K_zz
    timings["FQK_Standard"] = dt_zz
    np.save(os.path.join(SAVE_DIR, "K_fqk_standard.npy"), K_zz)
    logger.info(f"  Time: {dt_zz:.1f}s")
    
    logger.info("\n--- 2/2: SRQFM-FQK (Self-Regulating) ---")
    K_sr, dt_sr = compute_fqk_kernel_matrix(X_norm, statevector_srqfm)
    kernels["FQK_SRQFM"] = K_sr
    timings["FQK_SRQFM"] = dt_sr
    np.save(os.path.join(SAVE_DIR, "K_fqk_srqfm.npy"), K_sr)
    logger.info(f"  Time: {dt_sr:.1f}s")
    
    # 3. Spectral Diagnostics
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 2: Kernel Diagnostics")
    logger.info("=" * 70)
    
    diagnostics = {}
    for name, K in kernels.items():
        spec = spectral_metrics(K)
        kta = compute_kta(K, y)
        diagnostics[name] = {**spec, "kta": kta}
        logger.info(f"\n  {name}:")
        logger.info(f"    KTA:           {kta:.4f}")
        logger.info(f"    Eff rank:      {spec['eff_rank_shannon']:.1f}")
        logger.info(f"    Off-diag mean: {spec['off_diag_mean']:.4f}")
        logger.info(f"    Off-diag std:  {spec['off_diag_var']**0.5:.4f}")
    
    # 4. CV Evaluation
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 3: Cross-Validated Classification")
    logger.info("=" * 70)
    
    cv_results = {name: [] for name in list(kernels.keys()) + ["RBF-SVM", "RandomForest"]}
    
    for seed in SEEDS:
        logger.info(f"\n  --- Seed {seed} ---")
        
        # Quantum FQKs (no hyperparameter tuning)
        for name, K in kernels.items():
            res = svm_eval(K, y, seed)
            cv_results[name].append(res)
            logger.info(f"    {name:20s}: F1={res['macro_f1']:.4f}")
            
        # Classical baselines (evaluated on bare 8 bands)
        cl_res = evaluate_classical_baselines(X_raw, y, seed)
        for cl_name, cl_metrics in cl_res.items():
            cv_results[cl_name].append(cl_metrics)
            logger.info(f"    {cl_name:20s}: F1={cl_metrics['macro_f1']:.4f}")
            
    # 5. Summary and Statistics
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 4: Final Summary (Raw Data)")
    logger.info("=" * 70)
    
    summary = {}
    logger.info(f"\n  {'Method':<20} {'F1 Mean':>8} {'F1 Std':>8} {'Acc':>8} {'KTA':>8}")
    logger.info("  " + "-" * 56)
    
    for name in cv_results:
        f1s = [r["macro_f1"] for r in cv_results[name]]
        accs = [r["accuracy"] for r in cv_results[name]]
        kta_val = diagnostics.get(name, {}).get("kta", None)
        
        summary[name] = {
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "f1_scores": f1s,
            "kta": kta_val,
        }
        
        kta_str = f"{kta_val:.4f}" if kta_val is not None else "  N/A"
        logger.info(f"  {name:<20} {np.mean(f1s):>8.4f} {np.std(f1s):>8.4f} "
                    f"{np.mean(accs):>8.4f} {kta_str:>8}")
                    
    # Stat tests vs SRQFM
    logger.info("\n--- Statistical Significance (SRQFM-FQK vs others) ---")
    sr_f1s = summary["FQK_SRQFM"]["f1_scores"]
    stat_tests = {}
    
    for name in cv_results:
        if name == "FQK_SRQFM":
            continue
        other_f1s = summary[name]["f1_scores"]
        t, p = ttest_rel(sr_f1s, other_f1s)
        d = cohens_d(np.array(sr_f1s), np.array(other_f1s))
        
        direction = "wins" if np.mean(sr_f1s) > np.mean(other_f1s) else "loses"
        sig = "YES" if p < 0.05 else "no"
        
        stat_tests[name] = {
            "p_value": float(p),
            "cohens_d": float(d),
            "significant": p < 0.05
        }
        
        logger.info(f"  vs {name:<20}: p={p:.4f} d={d:+.3f} (SRQFM {direction}, {sig})")
        
    # Save
    results = {
        "experiment": "E24_Pure_Quantum_Pipeline",
        "timestamp": datetime.now().isoformat(),
        "bands": list(names),
        "summary": summary,
        "diagnostics": {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv 
                for kk, vv in v.items()}
            for k, v in diagnostics.items()
        },
        "statistical_tests": stat_tests,
        "timings_s": timings,
    }
    
    res_path = os.path.join(SAVE_DIR, "e24_pure_quantum_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info("\nRun complete. Results saved.")
    return results


if __name__ == "__main__":
    run_experiment()
