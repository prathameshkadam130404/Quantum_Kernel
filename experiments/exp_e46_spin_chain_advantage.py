"""
E46: Unbiased procedural audit on XY-chain and Ising-chain ground states.

Negative control for the kernel evaluation pipeline.  Two synthetic
binary classification tasks are constructed from the ground-state
expectation values of L=6 chains:
    XY    : H = j1 * sum X_i X_{i+1}  +  j2 * sum Y_i Y_{i+1};  label = (j1 > j2)
    Ising : H = j1 * sum Z_i Z_{i+1}  +  j2 * sum X_i;          label = (j1 > j2)
Both tasks are near-linearly-separable on the chosen features and should
saturate every reasonable classifier near 100% accuracy.  If BSCM-uniform,
Standard-ZZ, or RBF-SVM under-perform here, the evaluation pipeline has
a bug; passing this audit rules out a systematic implementation defect
in the kernel build / SVM evaluation path.

Run on n=6, N=100, depth=2; results in results/e46_unbiased_audit/.
"""
import os
import sys
import json
import time
import logging
import numpy as np
from typing import Tuple, Dict
from tqdm import tqdm

import pennylane as qml
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.bscm_kernel import bscm_pauli_coefficients

# ============================================================================
# CONFIGURATION
# ============================================================================
N_QUBITS = 6
N_SAMPLES = 100
DEPTH = 2

OUT_DIR = os.path.join(config.RESULTS_DIR, "e46_unbiased_audit")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(name)s] %(levelname)s:   %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUT_DIR, "exp_e46_audit.log"), mode="w"),
    ]
)
log = logging.getLogger("exp_e46")

# ============================================================================
# DUAL-PHYSICS DATA GENERATION
# ============================================================================

def generate_physics_dataset(n_samples: int, n_qubits: int, model_type: str) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42 if model_type == 'XY' else 123)
    X, y = [], []
    wire_order = range(n_qubits)
    
    log.info(f"Generating {model_type} Physics Dataset (Full Hilbert Space)...")
    for _ in tqdm(range(n_samples), desc=f"{model_type} Data"):
        j1 = np.random.uniform(0.1, 2.0)
        j2 = np.random.uniform(0.1, 2.0)
        
        coeffs, obs = [], []
        if model_type == 'XY':
            for i in range(n_qubits - 1):
                coeffs.append(j1); obs.append(qml.PauliX(i) @ qml.PauliX(i+1))
                coeffs.append(j2); obs.append(qml.PauliY(i) @ qml.PauliY(i+1))
            label = 0 if j1 > j2 else 1
        else: # Ising model
            for i in range(n_qubits - 1):
                coeffs.append(j1); obs.append(qml.PauliZ(i) @ qml.PauliZ(i+1))
            for i in range(n_qubits):
                coeffs.append(j2); obs.append(qml.PauliX(i))
            label = 0 if j1 > j2 else 1
            
        H = qml.Hamiltonian(coeffs, obs)
        # Use explicit wire_order to ensure matrix matches full Hilbert space (2^n x 2^n)
        H_matrix = qml.matrix(H, wire_order=wire_order)
        evals, evecs = np.linalg.eigh(H_matrix)
        psi_ground = evecs[:, 0]
        
        features = []
        for i in range(n_qubits - 1):
            if model_type == 'XY':
                op_x = qml.matrix(qml.PauliX(i) @ qml.PauliX(i+1), wire_order=wire_order)
                op_y = qml.matrix(qml.PauliY(i) @ qml.PauliY(i+1), wire_order=wire_order)
                features.append(np.real(np.conj(psi_ground).T @ op_x @ psi_ground))
                features.append(np.real(np.conj(psi_ground).T @ op_y @ psi_ground))
            else:
                op_zz = qml.matrix(qml.PauliZ(i) @ qml.PauliZ(i+1), wire_order=wire_order)
                op_x = qml.matrix(qml.PauliX(i), wire_order=wire_order)
                features.append(np.real(np.conj(psi_ground).T @ op_zz @ psi_ground))
                features.append(np.real(np.conj(psi_ground).T @ op_x @ psi_ground))
                
        X.append(features); y.append(label)
        
    return np.array(X), np.array(y)

# ============================================================================
# KERNELS
# ============================================================================

def bscm_ansatz(x, n_qubits):
    tau = 0.25
    for r in range(DEPTH):
        for i in range(n_qubits):
            qml.Hadamard(wires=i); qml.RZ(x[i % len(x)], wires=i)
        for i in range(n_qubits - 1):
            a_xx, a_yy, a_zz = bscm_pauli_coefficients(x[i % len(x)], x[(i+1) % len(x)], (1.0, 1.0, 1.0, 1.0))
            qml.IsingXX(2.0 * tau * a_xx, wires=[i, i+1])
            qml.IsingYY(2.0 * tau * a_yy, wires=[i, i+1])

def zz_ansatz(x, n_qubits):
    for r in range(DEPTH):
        for i in range(n_qubits):
            qml.Hadamard(wires=i); qml.RZ(x[i % len(x)], wires=i)
        for i in range(n_qubits - 1):
            theta = 2.0 * (np.pi - x[i % len(x)]) * (np.pi - x[(i+1) % len(x)])
            qml.IsingZZ(theta, wires=[i, i+1])

dev = qml.device("default.qubit", wires=N_QUBITS)
@qml.qnode(dev)
def fqk_bscm(x1, x2):
    bscm_ansatz(x1, N_QUBITS); qml.adjoint(bscm_ansatz)(x2, N_QUBITS)
    return qml.probs(wires=range(N_QUBITS))
@qml.qnode(dev)
def fqk_zz(x1, x2):
    zz_ansatz(x1, N_QUBITS); qml.adjoint(zz_ansatz)(x2, N_QUBITS)
    return qml.probs(wires=range(N_QUBITS))

# ============================================================================
# AUDIT PIPELINE
# ============================================================================

def run_audit():
    final_report = {}
    
    for model_type in ["XY", "Ising"]:
        X, y = generate_physics_dataset(N_SAMPLES, N_QUBITS, model_type)
        
        K_bscm = np.eye(N_SAMPLES)
        K_zz = np.eye(N_SAMPLES)
        for i in tqdm(range(N_SAMPLES), desc=f"Gram {model_type}"):
            for j in range(i+1, N_SAMPLES):
                K_bscm[i,j] = K_bscm[j,i] = float(fqk_bscm(X[i], X[j])[0])
                K_zz[i,j] = K_zz[j,i] = float(fqk_zz(X[i], X[j])[0])
                
        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        results = {"RBF-SVM": [], "Standard-ZZ": [], "BSCM-uniform": []}
        
        for tr, te in sss.split(X, y):
            scaler = StandardScaler()
            X_tr, X_te = scaler.fit_transform(X[tr]), scaler.transform(X[te])
            clf_rbf = GridSearchCV(SVC(kernel="rbf"), {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]}, cv=3)
            clf_rbf.fit(X_tr, y[tr])
            results["RBF-SVM"].append(accuracy_score(y[te], clf_rbf.predict(X_te)))
            
            for name, K in [("Standard-ZZ", K_zz), ("BSCM-uniform", K_bscm)]:
                K_tr, K_te = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
                clf_q = GridSearchCV(SVC(kernel="precomputed"), {"C": [0.1, 1, 10, 100]}, cv=3)
                clf_q.fit(K_tr, y[tr])
                results[name].append(accuracy_score(y[te], clf_q.predict(K_te)))
                
        final_report[model_type] = {k: f"{np.mean(v):.4f} +/- {np.std(v):.4f}" for k, v in results.items()}

    log.info("\n" + "="*80)
    log.info("FINAL UNBIASED AUDIT SUMMARY: INDUCTIVE BIAS ALIGNMENT")
    log.info("="*80)
    for model_type, scores in final_report.items():
        log.info(f"\nTARGET PHYSICS: {model_type} Chain")
        log.info("-" * 40)
        for k, v in scores.items():
            log.info(f"{k:<20s} : {v}")

    with open(os.path.join(OUT_DIR, "audit_results.json"), "w") as f:
        json.dump(final_report, f, indent=2)

if __name__ == "__main__":
    run_audit()
