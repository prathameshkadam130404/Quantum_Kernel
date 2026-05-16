"""
E38 baselines arm: SRQFM-PQK + tuned classical (RBF-SVM, Random Forest).

Three methods on each of the E38 pools (So2Sat and EuroSAT, N=10,000):
* SRQFM-PQK: Bloch-vector PQK from the singlet-only IsingZZ circuit
  on the same block-bipartite ladder connectivity as BSCM (E38).
* RBF-SVM:    sklearn SVC with rbf kernel, C and gamma grid-tuned.
* RandomForest: 300 trees, balanced class weights.
All three are evaluated on the 10 stratified shuffle splits used by the
BSCM and Standard-ZZ arms.

Output: results/e38_max_data/cache_<dataset>_srqfm.npz   (K, y, gamma)
        results/e38_max_data/baselines_summary.json
"""
import os
import sys
import json
import time
import numpy as np
from typing import Tuple, List

import pennylane as qml
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments.exp_e38_max_data_pqk import _load_so2sat_max, _load_eurosat_10k, compute_dynamic_pqk_gram, OUT_DIR

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s:   %(message)s")
log = logging.getLogger("e38_baselines")

# ============================================================================
# CONFIGURATION
# ============================================================================
N_QUBITS = 16
DEPTH = 6
CV_FOLDS = 3
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]

# ============================================================================
# SRQFM-PQK (singlet IsingZZ, commuting Pauli generator)
# ============================================================================

def _build_srqfm_bloch_extractor(n_qubits: int, reps: int):
    """Bloch-readout QNodes for SRQFM (singlet-Bell IsingZZ angle) on the
    block-bipartite ladder connectivity (intra-OPT, intra-SAR, OPT-SAR rungs).
    """
    dev = config.get_device(n_qubits)
    
    mid = n_qubits // 2
    pairs = []
    # 1. Intra-modal: OPT-OPT linear chain
    pairs.extend([(i, i + 1) for i in range(mid - 1)])
    # 2. Intra-modal: SAR-SAR linear chain
    pairs.extend([(i, i + 1) for i in range(mid, n_qubits - 1)])
    # 3. Cross-modal: OPT-SAR rungs
    pairs.extend([(i, i + mid) for i in range(mid)])

    def _apply_circuit(x):
        for _ in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            for qi, qj in pairs:
                c = float(np.sin((x[qi] - x[qj]) / 2.0) ** 2)
                if c > 1e-5: qml.IsingZZ(c, wires=[qi, qj])

    @qml.qnode(dev, diff_method=None)
    def mx(x): _apply_circuit(x); return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def my(x): _apply_circuit(x); return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def mz(x): _apply_circuit(x); return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return mx, my, mz


def extract_srqfm_bloch_vectors(X: np.ndarray, n_qubits: int, reps: int, name: str) -> np.ndarray:
    N = len(X)
    out = np.zeros((N, n_qubits * 3), dtype=np.float32)
    mx, my, mz = _build_srqfm_bloch_extractor(n_qubits, reps)
    
    t0 = time.time()
    for i in tqdm(range(N), desc=f"Simulating {name} SRQFM"):
        x_i = X[i]
        out[i, 0*n_qubits : 1*n_qubits] = mx(x_i)
        out[i, 1*n_qubits : 2*n_qubits] = my(x_i)
        out[i, 2*n_qubits : 3*n_qubits] = mz(x_i)
        
    log.info(f"SRQFM Simulation completed in {time.time()-t0:.1f}s")
    return out


# ============================================================================
# EVALUATION (STRICTLY ALIGNED WITH E38)
# ============================================================================

def run_baseline_evaluations(K_srqfm: np.ndarray, X_raw: np.ndarray, y: np.ndarray) -> dict:
    """Runs exactly the same 10 splits as the BSCM-PQK E38 evaluation for fair comparison."""
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    
    scores_srqfm = []
    scores_rbf = []
    scores_rf = []
    
    for tr, te in sss.split(K_srqfm, y):
        # 1. SRQFM (Precomputed)
        K_tr = K_srqfm[np.ix_(tr, tr)]
        K_te = K_srqfm[np.ix_(te, tr)]
        clf_q = GridSearchCV(SVC(kernel="precomputed", class_weight="balanced"), {"C": C_GRID}, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1)
        clf_q.fit(K_tr, y[tr])
        scores_srqfm.append(f1_score(y[te], clf_q.predict(K_te), average="macro"))

        # 2. Classical RBF SVM
        # Scale strictly on training split
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_raw[tr])
        X_te_sc = sc.transform(X_raw[te])

        clf_c = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), {"C": C_GRID, "gamma": ["scale", "auto", 0.01, 0.1, 1.0]}, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1)
        clf_c.fit(X_tr_sc, y[tr])
        scores_rbf.append(f1_score(y[te], clf_c.predict(X_te_sc), average="macro"))
        
        # 3. Random Forest
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_raw[tr], y[tr])
        scores_rf.append(f1_score(y[te], rf.predict(X_raw[te]), average="macro"))
        
    return {
        "srqfm": (np.mean(scores_srqfm), np.std(scores_srqfm)),
        "rbf": (np.mean(scores_rbf), np.std(scores_rbf)),
        "rf": (np.mean(scores_rf), np.std(scores_rf))
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("Starting EXP-38 Baselines: SRQFM, RBF, RF")
    res = {"depth": DEPTH}
    
    # ---- So2Sat ----
    cache_so2sat = os.path.join(OUT_DIR, "cache_so2sat_srqfm.npz")
    if os.path.exists(cache_so2sat):
        log.info("Loading cached So2Sat SRQFM Gram Matrix...")
        d = np.load(cache_so2sat)
        K_s, y_s = d["K"], d["y"]
        _, X_raw_s, _ = _load_so2sat_max()
    else:
        log.info("Loading So2Sat (N=10,000)...")
        X_enc_s, X_raw_s, y_s = _load_so2sat_max()
        b_so2sat = extract_srqfm_bloch_vectors(X_enc_s, N_QUBITS, DEPTH, "So2Sat")
        K_s, gamma_s = compute_dynamic_pqk_gram(b_so2sat, N_QUBITS)
        np.savez_compressed(cache_so2sat, K=K_s, y=y_s, gamma=gamma_s)

    log.info("Running Baseline Evaluations on So2Sat...")
    eval_s = run_baseline_evaluations(K_s, X_raw_s, y_s)
    log.info(f"==> SRQFM-PQK So2Sat F1: {eval_s['srqfm'][0]:.4f} +/- {eval_s['srqfm'][1]:.4f}")
    log.info(f"==> RBF-SVM So2Sat F1:   {eval_s['rbf'][0]:.4f} +/- {eval_s['rbf'][1]:.4f}")
    log.info(f"==> RF So2Sat F1:        {eval_s['rf'][0]:.4f} +/- {eval_s['rf'][1]:.4f}")
    
    res["so2sat"] = {
        "n_samples": len(y_s),
        "srqfm_f1_mean": float(eval_s['srqfm'][0]), "srqfm_f1_std": float(eval_s['srqfm'][1]),
        "rbf_f1_mean": float(eval_s['rbf'][0]), "rbf_f1_std": float(eval_s['rbf'][1]),
        "rf_f1_mean": float(eval_s['rf'][0]), "rf_f1_std": float(eval_s['rf'][1]),
    }


    # ---- EuroSAT ----
    cache_eurosat = os.path.join(OUT_DIR, "cache_eurosat_srqfm.npz")
    if os.path.exists(cache_eurosat):
        log.info("Loading cached EuroSAT SRQFM Gram Matrix...")
        d = np.load(cache_eurosat)
        K_e, y_e = d["K"], d["y"]
        _, X_raw_e, _ = _load_eurosat_10k()
    else:
        log.info("Loading EuroSAT (N=10,000)...")
        X_enc_e, X_raw_e, y_e = _load_eurosat_10k()
        b_eurosat = extract_srqfm_bloch_vectors(X_enc_e, N_QUBITS, DEPTH, "EuroSAT 10k")
        K_e, gamma_e = compute_dynamic_pqk_gram(b_eurosat, N_QUBITS)
        np.savez_compressed(cache_eurosat, K=K_e, y=y_e, gamma=gamma_e)

    log.info("Running Baseline Evaluations on EuroSAT...")
    eval_e = run_baseline_evaluations(K_e, X_raw_e, y_e)
    log.info(f"==> SRQFM-PQK EuroSAT F1: {eval_e['srqfm'][0]:.4f} +/- {eval_e['srqfm'][1]:.4f}")
    log.info(f"==> RBF-SVM EuroSAT F1:   {eval_e['rbf'][0]:.4f} +/- {eval_e['rbf'][1]:.4f}")
    log.info(f"==> RF EuroSAT F1:        {eval_e['rf'][0]:.4f} +/- {eval_e['rf'][1]:.4f}")
    
    res["eurosat"] = {
        "n_samples": len(y_e),
        "srqfm_f1_mean": float(eval_e['srqfm'][0]), "srqfm_f1_std": float(eval_e['srqfm'][1]),
        "rbf_f1_mean": float(eval_e['rbf'][0]), "rbf_f1_std": float(eval_e['rbf'][1]),
        "rf_f1_mean": float(eval_e['rf'][0]), "rf_f1_std": float(eval_e['rf'][1]),
    }

    with open(os.path.join(OUT_DIR, "baselines_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    log.info("Baselines Completed Successfully.")

if __name__ == "__main__":
    main()
