"""
E38: Standard-ZZ-PQK arm of the maximum-capacity benchmark.

Evaluates the canonical Havlicek ZZ feature-map as a Bloch-vector PQK at
n=16, depth=6 on the So2Sat and EuroSAT N=10,000 pools, using the same
block-bipartite ladder connectivity and the same 10-stratified-shuffle-split
SVM evaluation as exp_e38_max_data_pqk.py.  This is the "vanilla" baseline
the BSCM family is compared against (paper Table 9).

Output: results/e38_max_data/cache_<dataset>_standard_pqk.npz   (K, y, gamma)
        results/e38_max_data/standard_pqk_summary.json
"""
import os
import sys
import json
import time
import numpy as np
from typing import Tuple, List

import pennylane as qml
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments.exp_e38_max_data_pqk import _load_so2sat_max, _load_eurosat_10k, compute_dynamic_pqk_gram, OUT_DIR

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s:   %(message)s")
log = logging.getLogger("e38_standard_pqk")

# ============================================================================
# CONFIGURATION
# ============================================================================
N_QUBITS = 16
DEPTH = 6
CV_FOLDS = 3
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]

# ============================================================================
# STANDARD PQK (HAVLICEK ZZFeatureMap PROJECTED)
# ============================================================================

def _build_standard_pqk_bloch_extractor(n_qubits: int, reps: int):
    """Build the three Bloch-readout QNodes for the canonical Havlicek
    ZZ feature map on the same block-bipartite ladder topology as the
    BSCM extractor in exp_e38_max_data_pqk.py.
    """
    dev = config.get_device(n_qubits)
    
    mid = n_qubits // 2
    pairs = []
    # Ladder topology to ensure fair comparison to BSCM and SRQFM
    pairs.extend([(i, i + 1) for i in range(mid - 1)])
    pairs.extend([(i, i + 1) for i in range(mid, n_qubits - 1)])
    pairs.extend([(i, i + mid) for i in range(mid)])

    def _apply_circuit(x):
        for _ in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            for qi, qj in pairs:
                # Standard ZZFeatureMap interaction: CNOT -> RZ((pi-xi)(pi-xj)) -> CNOT
                qml.CNOT(wires=[qi, qj])
                qml.RZ((np.pi - x[qi]) * (np.pi - x[qj]), wires=qj)
                qml.CNOT(wires=[qi, qj])

    @qml.qnode(dev, diff_method=None)
    def mx(x): _apply_circuit(x); return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def my(x): _apply_circuit(x); return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def mz(x): _apply_circuit(x); return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return mx, my, mz


def extract_standard_pqk_bloch_vectors(X: np.ndarray, n_qubits: int, reps: int, name: str) -> np.ndarray:
    N = len(X)
    out = np.zeros((N, n_qubits * 3), dtype=np.float32)
    mx, my, mz = _build_standard_pqk_bloch_extractor(n_qubits, reps)
    
    t0 = time.time()
    for i in tqdm(range(N), desc=f"Simulating {name} Standard PQK"):
        x_i = X[i]
        out[i, 0*n_qubits : 1*n_qubits] = mx(x_i)
        out[i, 1*n_qubits : 2*n_qubits] = my(x_i)
        out[i, 2*n_qubits : 3*n_qubits] = mz(x_i)
        
    log.info(f"Standard PQK Simulation completed in {time.time()-t0:.1f}s")
    return out


# ============================================================================
# EVALUATION (STRICTLY ALIGNED WITH E38)
# ============================================================================

def run_pqk_evaluations(K_pqk: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Runs exactly the same 10 splits as the BSCM-PQK E38 evaluation for fair comparison."""
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    
    scores = []
    for tr, te in sss.split(K_pqk, y):
        K_tr = K_pqk[np.ix_(tr, tr)]
        K_te = K_pqk[np.ix_(te, tr)]
        clf = GridSearchCV(SVC(kernel="precomputed", class_weight="balanced"), {"C": C_GRID}, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1)
        clf.fit(K_tr, y[tr])
        scores.append(f1_score(y[te], clf.predict(K_te), average="macro"))
        
    return np.mean(scores), np.std(scores)

# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("Starting EXP-38 Baselines: Standard PQK (Havlicek Feature Map)")
    res = {"depth": DEPTH}
    
    # ---- So2Sat ----
    cache_so2sat = os.path.join(OUT_DIR, "cache_so2sat_standard_pqk.npz")
    if os.path.exists(cache_so2sat):
        log.info("Loading cached So2Sat Standard PQK Gram Matrix...")
        d = np.load(cache_so2sat)
        K_s, y_s = d["K"], d["y"]
    else:
        log.info("Loading So2Sat (N=10,000)...")
        X_enc_s, _, y_s = _load_so2sat_max()
        b_so2sat = extract_standard_pqk_bloch_vectors(X_enc_s, N_QUBITS, DEPTH, "So2Sat")
        K_s, gamma_s = compute_dynamic_pqk_gram(b_so2sat, N_QUBITS)
        np.savez_compressed(cache_so2sat, K=K_s, y=y_s, gamma=gamma_s)

    log.info("Running Baseline Evaluations on So2Sat...")
    f1_mean_s, f1_std_s = run_pqk_evaluations(K_s, y_s)
    log.info(f"==> Standard PQK So2Sat F1: {f1_mean_s:.4f} +/- {f1_std_s:.4f}")
    
    res["so2sat"] = {
        "n_samples": len(y_s),
        "standard_pqk_f1_mean": float(f1_mean_s),
        "standard_pqk_f1_std": float(f1_std_s),
    }


    # ---- EuroSAT ----
    cache_eurosat = os.path.join(OUT_DIR, "cache_eurosat_standard_pqk.npz")
    if os.path.exists(cache_eurosat):
        log.info("Loading cached EuroSAT Standard PQK Gram Matrix...")
        d = np.load(cache_eurosat)
        K_e, y_e = d["K"], d["y"]
    else:
        log.info("Loading EuroSAT (N=10,000)...")
        X_enc_e, _, y_e = _load_eurosat_10k()
        b_eurosat = extract_standard_pqk_bloch_vectors(X_enc_e, N_QUBITS, DEPTH, "EuroSAT 10k")
        K_e, gamma_e = compute_dynamic_pqk_gram(b_eurosat, N_QUBITS)
        np.savez_compressed(cache_eurosat, K=K_e, y=y_e, gamma=gamma_e)

    log.info("Running Baseline Evaluations on EuroSAT...")
    f1_mean_e, f1_std_e = run_pqk_evaluations(K_e, y_e)
    log.info(f"==> Standard PQK EuroSAT F1: {f1_mean_e:.4f} +/- {f1_std_e:.4f}")
    
    res["eurosat"] = {
        "n_samples": len(y_e),
        "standard_pqk_f1_mean": float(f1_mean_e),
        "standard_pqk_f1_std": float(f1_std_e),
    }

    with open(os.path.join(OUT_DIR, "standard_pqk_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    log.info("Standard PQK Baseline Completed Successfully.")

if __name__ == "__main__":
    main()
