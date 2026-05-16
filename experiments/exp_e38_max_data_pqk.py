"""
E38: Maximum-capacity PQK benchmark for the BSCM family at n=16, N=10,000.

For each of the three BSCM priors (uniform / phi_only / psi_only), this
script extracts Bloch vectors on So2Sat (Fisher-16 physics features) and
EuroSAT (Fisher-16 from physics+raw-bands), builds the PQK Gram matrix
using the dynamic (sklearn-`scale`) gamma heuristic, caches both, and
evaluates with C-tuned SVC over 10 stratified shuffle splits.

Pre-requisites:
    scripts/extract_so2sat_10k.py        --> physics_features_10k.npz
    scripts/eurosat_data.py (cached)     --> eurosat_band_means.npz

Output: results/e38_max_data/cache_<dataset>_bscm_<prior>.npz   (K, y, gamma)
        results/e38_max_data/max_summary_bscm_sectors.json
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.bscm_kernel import bscm_pauli_coefficients, BELL_WEIGHT_PRESETS

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s:   %(message)s")
log = logging.getLogger("exp_e38_max")

# ============================================================================
# CONFIGURATION
# ============================================================================
EUROSAT_N_TRAIN = 6700
EUROSAT_N_TEST = 3300   # 6700 + 3300 = 10,000 evaluation pool
N_QUBITS = 16
DEPTH = 6
TAU_LOCKED = 0.25
CV_FOLDS = 3
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]

OUT_DIR = os.path.join(config.RESULTS_DIR, "e38_max_data")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# Data loaders (background-fit normalisation, no test-set leakage)
# ============================================================================

def _load_so2sat_max() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the 10,000-sample So2Sat physics-Fisher-16 evaluation pool."""
    p = os.path.join(config.PROCESSED_DIR, "physics_features_10k.npz")
    if not os.path.exists(p):
        log.error(f"Cannot find {p}. Please run: python scripts/extract_so2sat_10k.py")
        sys.exit(1)
        
    d = np.load(p)
    X_raw_all = d["X_train_raw"]
    X_enc_all = d["X_train"]
    y_all = d["y_train"]
    # The downstream run_evaluations() draws 10 stratified 70/30 train/test
    # shuffle splits from this 10,000-sample pool.
    return X_enc_all, X_raw_all, y_all


def _load_eurosat_10k() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 10,000-sample EuroSAT Fisher-16 pool with background-only fit.

    The 27,000-sample EuroSAT corpus is split once: 10,000 samples form the
    evaluation pool, 17,000 form the background.  Fisher feature ranking,
    StandardScaler, and MinMaxScaler[0, pi] are all fit on background only;
    the evaluation pool is transformed but never seen by any fit.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from eurosat_data import load_eurosat_allbands, compute_physics_indices
    from src.attention_kernel import compute_fisher_ratio

    bm, labels, _ = load_eurosat_allbands(
        cache_path=os.path.join(config.PROCESSED_DIR, "eurosat_band_means.npz"),
    )

    # Candidate features: 8 physics indices + 13 raw band means = 21.
    phys_raw = compute_physics_indices(bm)
    bands_raw = bm.astype(np.float64)
    X_raw_all = np.hstack([phys_raw, bands_raw])

    # One-shot stratified pool selection of 10,000 samples; the remaining
    # ~17,000 are reserved as background for normalisation fit.
    pool_sss = StratifiedShuffleSplit(
        n_splits=1, train_size=EUROSAT_N_TRAIN, test_size=EUROSAT_N_TEST,
        random_state=0,
    )
    tr, te = next(pool_sss.split(X_raw_all, labels))
    pool_idx = np.concatenate([tr, te])

    background_mask = np.ones(len(X_raw_all), dtype=bool)
    background_mask[pool_idx] = False

    X_raw_background = X_raw_all[background_mask]
    y_background = labels[background_mask]

    # Fisher rank on the background only (zero leakage).
    fisher_scores = compute_fisher_ratio(X_raw_background, y_background)
    top_16_idx = np.argsort(fisher_scores)[::-1][:16]

    X_raw_background = X_raw_background[:, top_16_idx]
    X_raw_pool = X_raw_all[pool_idx][:, top_16_idx]
    y_pool = labels[pool_idx]

    scaler = StandardScaler()
    scaler.fit(X_raw_background)
    X_scaled = scaler.transform(X_raw_pool)
    X_scaled_bg = scaler.transform(X_raw_background)

    mm = MinMaxScaler(feature_range=(0, np.pi))
    mm.fit(X_scaled_bg)
    X_enc_pool = mm.transform(X_scaled).clip(0, np.pi).astype(np.float32)

    return X_enc_pool, X_raw_pool, y_pool


# ============================================================================
# QUANTUM CIRCUIT (LADDER TOPOLOGY)
# ============================================================================

def _build_bscm_bloch_extractor(n_qubits: int, reps: int, bell_weights: tuple):
    """XX+YY+ZZ non-commuting channel with sparse Ladder connectivity."""
    dev = config.get_device(n_qubits)
    
    mid = n_qubits // 2
    pairs = []
    # 1. Intra-modal: Rail 1 (OPT)
    pairs.extend([(i, i + 1) for i in range(mid - 1)])
    # 2. Intra-modal: Rail 2 (SAR)
    pairs.extend([(i, i + 1) for i in range(mid, n_qubits - 1)])
    # 3. Cross-modal: Rungs connecting OPT directly to SAR
    pairs.extend([(i, i + mid) for i in range(mid)])
    
    def _apply_circuit(x):
        for _ in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            for qi, qj in pairs:
                a_xx, a_yy, a_zz = bscm_pauli_coefficients(x[qi], x[qj], bell_weights)
                
                c_xx = 2.0 * TAU_LOCKED * a_xx
                c_yy = 2.0 * TAU_LOCKED * a_yy
                c_zz = 2.0 * TAU_LOCKED * a_zz
                
                if abs(c_xx) > 1e-5: qml.IsingXX(c_xx, wires=[qi, qj])
                if abs(c_yy) > 1e-5: qml.IsingYY(c_yy, wires=[qi, qj])
                if abs(c_zz) > 1e-5: qml.IsingZZ(c_zz, wires=[qi, qj])

    # diff_method=None prevents memory explosion when simulating 10,000 states
    @qml.qnode(dev, diff_method=None)
    def mx(x): _apply_circuit(x); return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def my(x): _apply_circuit(x); return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def mz(x): _apply_circuit(x); return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return mx, my, mz


def extract_bloch_vectors(X: np.ndarray, n_qubits: int, reps: int, name: str, bell_weights: tuple) -> np.ndarray:
    N = len(X)
    out = np.zeros((N, n_qubits * 3), dtype=np.float32)
    mx, my, mz = _build_bscm_bloch_extractor(n_qubits, reps, bell_weights)
    
    t0 = time.time()
    from tqdm import tqdm
    for i in tqdm(range(N), desc=f"Simulating {name}"):
        x_i = X[i]
        out[i, 0*n_qubits : 1*n_qubits] = mx(x_i)
        out[i, 1*n_qubits : 2*n_qubits] = my(x_i)
        out[i, 2*n_qubits : 3*n_qubits] = mz(x_i)
        
    log.info(f"Simulation completed in {time.time()-t0:.1f}s")
    return out


# ============================================================================
# DYNAMIC GAMMA KERNEL & EVALUATION
# ============================================================================

def compute_dynamic_pqk_gram(bloch: np.ndarray, n_qubits: int) -> Tuple[np.ndarray, float]:
    """PQK Gram K(x, x') = exp(-gamma/2 * ||phi(x) - phi(x')||^2) using the
    sklearn-`scale` heuristic gamma = 1 / (d * Var[phi]).

    The squared-distance matrix is computed via the dot-product identity
    ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y> to avoid the O(N^2 * d) broadcast
    tensor that a naive subtraction would build at N=10,000.
    """
    N = len(bloch)
    b_flat = bloch.reshape(N, -1)
    sq_norms = np.sum(b_flat**2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * np.dot(b_flat, b_flat.T)
    sq_dists = np.clip(sq_dists, 0, None)  # negative-distance fp noise
    d = 0.5 * sq_dists

    b_var = np.var(b_flat)
    dynamic_gamma = 1.0 / (b_flat.shape[1] * b_var) if b_var > 0 else 1.0

    K = np.exp(-dynamic_gamma * d)
    np.fill_diagonal(K, 1.0)
    K = (K + K.T) / 2.0
    return K, dynamic_gamma


def _eval_svm(K_full: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> float:
    """Precomputed SVM evaluation with nested GridSearch for C."""
    K_tr = K_full[np.ix_(tr, tr)]
    K_te = K_full[np.ix_(te, tr)]
    
    clf = GridSearchCV(
        SVC(kernel="precomputed", class_weight="balanced"),
        {"C": C_GRID}, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1,
    )
    clf.fit(K_tr, y[tr])
    yp = clf.predict(K_te)
    return f1_score(y[te], yp, average="macro")


def run_evaluations(K_full: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Runs 10 stratified shuffle splits (random_state=42 generates the 10 splits deterministically).
    Raised from 5 -> 10 so that Wilcoxon signed-rank has minimum achievable raw p = 1/2^9 = 0.002,
    well below alpha = 0.05, allowing within-family separations to be tested for significance."""
    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    scores = []
    for tr, te in sss.split(K_full, y):
        scores.append(_eval_svm(K_full, y, tr, te))
    return np.mean(scores), np.std(scores)


# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("Starting EXP-38: Maximum Capacity (L=6) Scaling with Bell Sectors")
    
    res = {
        "depth": DEPTH,
    }
    
    # We will load raw data ONCE to save memory
    log.info("Loading So2Sat (N=10,000)...")
    X_enc_s, X_raw_s, y_s = _load_so2sat_max()
    res["n_so2sat"] = len(y_s)
    
    log.info("Loading EuroSAT (N=10,000)...")
    X_enc_e, X_raw_e, y_e = _load_eurosat_10k()
    res["n_eurosat"] = len(y_e)
    
    # Evaluate each Bell Sector
    for sector_name, weights in BELL_WEIGHT_PRESETS.items():
        log.info(f"\n{'='*50}\nEvaluating BSCM-{sector_name}\n{'='*50}")
        
        # ---- So2Sat ----
        cache_so2sat = os.path.join(OUT_DIR, f"cache_so2sat_bscm_{sector_name}.npz")
        if os.path.exists(cache_so2sat):
            log.info(f"Loading cached So2Sat BSCM-{sector_name}...")
            d = np.load(cache_so2sat)
            K_s, gamma_s = d["K"], d["gamma"]
        else:
            log.info(f"Simulating BSCM-{sector_name} So2Sat...")
            b_s = extract_bloch_vectors(X_enc_s, N_QUBITS, DEPTH, f"So2Sat {sector_name}", weights)
            K_s, gamma_s = compute_dynamic_pqk_gram(b_s, N_QUBITS)
            np.savez_compressed(cache_so2sat, K=K_s, y=y_s, gamma=gamma_s)
            
        f1_mean_s, f1_std_s = run_evaluations(K_s, y_s)
        log.info(f"==> BSCM-{sector_name} (L=6) So2Sat F1: {f1_mean_s:.4f} +/- {f1_std_s:.4f}")
        res[f"so2sat_bscm_{sector_name}_f1_mean"] = float(f1_mean_s)
        res[f"so2sat_bscm_{sector_name}_f1_std"] = float(f1_std_s)
        
        # ---- EuroSAT ----
        cache_eurosat = os.path.join(OUT_DIR, f"cache_eurosat_bscm_{sector_name}.npz")
        if os.path.exists(cache_eurosat):
            log.info(f"Loading cached EuroSAT BSCM-{sector_name}...")
            d = np.load(cache_eurosat)
            K_e, gamma_e = d["K"], d["gamma"]
        else:
            log.info(f"Simulating BSCM-{sector_name} EuroSAT...")
            b_e = extract_bloch_vectors(X_enc_e, N_QUBITS, DEPTH, f"EuroSAT {sector_name}", weights)
            K_e, gamma_e = compute_dynamic_pqk_gram(b_e, N_QUBITS)
            np.savez_compressed(cache_eurosat, K=K_e, y=y_e, gamma=gamma_e)
            
        f1_mean_e, f1_std_e = run_evaluations(K_e, y_e)
        log.info(f"==> BSCM-{sector_name} (L=6) EuroSAT F1: {f1_mean_e:.4f} +/- {f1_std_e:.4f}")
        res[f"eurosat_bscm_{sector_name}_f1_mean"] = float(f1_mean_e)
        res[f"eurosat_bscm_{sector_name}_f1_std"] = float(f1_std_e)

    # Summary
    with open(os.path.join(OUT_DIR, "max_summary_bscm_sectors.json"), "w") as f:
        json.dump(res, f, indent=2)
    log.info("Experiment 38 Completed Successfully.")

if __name__ == "__main__":
    main()
