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

# We will dynamically split the new 10k dataset inside the script.

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s:   %(message)s")
log = logging.getLogger("exp_e38_max")

# ============================================================================
# CONFIGURATION
# ============================================================================
EUROSAT_N_TRAIN = 6700
EUROSAT_N_TEST = 3300   # Total Pool = 10,000
N_QUBITS = 16
DEPTH = 6
TAU_LOCKED = 0.25
CV_FOLDS = 3
C_GRID = [0.1, 1.0, 10.0]

# Output setup
OUT_DIR = os.path.join(config.RESULTS_DIR, "e38_max_data")
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# DATA LOADERS (Absolute Zero Leakage & Maximum Scale)
# ============================================================================

def _load_so2sat_max() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads the newly generated 10,000-sample So2Sat physical features."""
    p = os.path.join(config.PROCESSED_DIR, "physics_features_10k.npz")
    if not os.path.exists(p):
        log.error(f"Cannot find {p}. Please run: python scripts/extract_so2sat_10k.py")
        sys.exit(1)
        
    d = np.load(p)
    X_raw_all = d["X_train_raw"]
    X_enc_all = d["X_train"]
    y_all = d["y_train"]
    
    # We will use all 10,000 samples for the evaluation pool.
    # The SVM evaluation function `run_evaluations` already splits this pool
    # into 70% train (7000) and 30% test (3000) during the cross-validation loop.
    return X_enc_all, X_raw_all, y_all


def _load_eurosat_10k() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scales EuroSAT to N=10,000. Uses Fisher feature selection strictly on the background."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from eurosat_data import load_eurosat_allbands, compute_physics_indices
    from src.attention_kernel import compute_fisher_ratio

    bm, labels, _ = load_eurosat_allbands(
        cache_path=os.path.join(config.PROCESSED_DIR, "eurosat_band_means.npz"),
    )
    
    # Generate all 21 possible features
    phys_raw = compute_physics_indices(bm)
    bands_raw = bm.astype(np.float64)
    X_raw_all = np.hstack([phys_raw, bands_raw])
    
    # Select N=10,000 pool
    pool_sss = StratifiedShuffleSplit(
        n_splits=1, 
        train_size=EUROSAT_N_TRAIN, 
        test_size=EUROSAT_N_TEST, 
        random_state=0
    )
    tr, te = next(pool_sss.split(X_raw_all, labels))
    pool_idx = np.concatenate([tr, te])
    
    # Background selection (17,000 unseen samples)
    background_mask = np.ones(len(X_raw_all), dtype=bool)
    background_mask[pool_idx] = False
    
    X_raw_background = X_raw_all[background_mask]
    y_background = labels[background_mask]
    
    # FEATURE SELECTION: Compute Fisher Ratio STRICTLY on background
    fisher_scores = compute_fisher_ratio(X_raw_background, y_background)
    top_16_idx = np.argsort(fisher_scores)[::-1][:16]
    
    # Apply selection
    X_raw_background = X_raw_background[:, top_16_idx]
    X_raw_pool = X_raw_all[pool_idx][:, top_16_idx]
    y_pool = labels[pool_idx]
    
    # Scale strictly using the external background set
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

def _build_bscm_bloch_extractor(n_qubits: int, reps: int):
    """XX+YY non-commuting channel with sparse Ladder connectivity."""
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
                cp = np.cos((x[qi] + x[qj]) / 2.0)**2
                sp = np.sin((x[qi] + x[qj]) / 2.0)**2
                cm = np.cos((x[qi] - x[qj]) / 2.0)**2
                sm = np.sin((x[qi] - x[qj]) / 2.0)**2
                
                a_xx = (cp - sp + cm - sm) / 4.0
                a_yy = (-cp + sp + cm - sm) / 4.0
                
                c_xx = 2.0 * TAU_LOCKED * a_xx
                c_yy = 2.0 * TAU_LOCKED * a_yy
                
                if abs(c_xx) > 1e-5: qml.IsingXX(c_xx, wires=[qi, qj])
                if abs(c_yy) > 1e-5: qml.IsingYY(c_yy, wires=[qi, qj])

    # diff_method=None prevents memory explosion when simulating 10,000 states
    @qml.qnode(dev, diff_method=None)
    def mx(x): _apply_circuit(x); return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def my(x): _apply_circuit(x); return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def mz(x): _apply_circuit(x); return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return mx, my, mz


def extract_bloch_vectors(X: np.ndarray, n_qubits: int, reps: int, name: str) -> np.ndarray:
    N = len(X)
    out = np.zeros((N, n_qubits * 3), dtype=np.float32)
    mx, my, mz = _build_bscm_bloch_extractor(n_qubits, reps)
    
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
    """
    Computes Euclidean distance in Bloch space.
    Uses dynamic gamma = 1 / (n_features * variance) to prevent Identity Matrix collapse.
    """
    N = len(bloch)
    # Memory-efficient Euclidean distance computation using dot products
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
    b_flat = bloch.reshape(N, -1)
    sq_norms = np.sum(b_flat**2, axis=1)
    
    # This matrix multiplication only requires a (10000, 10000) matrix (~400MB)
    # instead of a (10000, 10000, 16, 3) broadcast tensor (~19GB)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * np.dot(b_flat, b_flat.T)
    # Clip to avoid negative distances due to floating point inaccuracies
    sq_dists = np.clip(sq_dists, 0, None)
    
    d = 0.5 * sq_dists  # (N, N)
    
    # Dynamic Gamma calculation (matching sklearn's 'scale' heuristic)
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
    
    grid = {"C": C_GRID}
    clf = GridSearchCV(
        SVC(kernel="precomputed", class_weight="balanced"), 
        grid, 
        cv=CV_FOLDS, 
        scoring="f1_macro", 
        n_jobs=-1  # Parallelized across CPU cores
    )
    clf.fit(K_tr, y[tr])
    yp = clf.predict(K_te)
    return f1_score(y[te], yp, average="macro")


def run_evaluations(K_full: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Runs 5 stratified shuffle splits (seeds 42-46 via base random_state=42) to ensure stability."""
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    scores = []
    for tr, te in sss.split(K_full, y):
        scores.append(_eval_svm(K_full, y, tr, te))
    return np.mean(scores), np.std(scores)


# ============================================================================
# MAIN
# ============================================================================

def main():
    log.info("Starting EXP-38: Maximum Capacity (L=6) Scaling")
    
    # 1. So2Sat ($N=10,000$)
    cache_so2sat = os.path.join(OUT_DIR, "cache_so2sat.npz")
    if os.path.exists(cache_so2sat):
        log.info("Loading cached So2Sat Gram Matrix...")
        d = np.load(cache_so2sat)
        K_s, y_s, gamma_s = d["K"], d["y"], d["gamma"]
        log.info(f"So2Sat Dynamic Gamma: {float(gamma_s):.4f} (Loaded from cache)")
    else:
        log.info("Loading So2Sat (N=10,000)...")
        X_enc_s, X_raw_s, y_s = _load_so2sat_max()
        log.info(f"So2Sat Pool: {len(X_enc_s)} samples")
        
        b_so2sat = extract_bloch_vectors(X_enc_s, N_QUBITS, DEPTH, "So2Sat")
        K_s, gamma_s = compute_dynamic_pqk_gram(b_so2sat, N_QUBITS)
        off_diag_s = K_s[~np.eye(len(K_s), dtype=bool)].mean()
        log.info(f"So2Sat Dynamic Gamma: {gamma_s:.4f} (Off-Diag: {off_diag_s:.4f})")
        
        np.savez_compressed(cache_so2sat, K=K_s, y=y_s, gamma=gamma_s)
        log.info(f"Saved So2Sat Kernel Cache: {cache_so2sat}")
        
    f1_mean_s, f1_std_s = run_evaluations(K_s, y_s)
    log.info(f"==> BSCM-PQK (L=6) So2Sat F1: {f1_mean_s:.4f} +/- {f1_std_s:.4f}")
    
    
    # 2. EuroSAT ($N=10,000$)
    cache_eurosat = os.path.join(OUT_DIR, "cache_eurosat.npz")
    if os.path.exists(cache_eurosat):
        log.info("Loading cached EuroSAT Gram Matrix...")
        d = np.load(cache_eurosat)
        K_e, y_e, gamma_e = d["K"], d["y"], d["gamma"]
        log.info(f"EuroSAT Dynamic Gamma: {float(gamma_e):.4f} (Loaded from cache)")
    else:
        log.info("Loading EuroSAT (N=10,000)...")
        X_enc_e, X_raw_e, y_e = _load_eurosat_10k()
        log.info(f"EuroSAT Pool: {len(X_enc_e)} samples")
        
        b_eurosat = extract_bloch_vectors(X_enc_e, N_QUBITS, DEPTH, "EuroSAT 10k")
        K_e, gamma_e = compute_dynamic_pqk_gram(b_eurosat, N_QUBITS)
        off_diag_e = K_e[~np.eye(len(K_e), dtype=bool)].mean()
        log.info(f"EuroSAT Dynamic Gamma: {gamma_e:.4f} (Off-Diag: {off_diag_e:.4f})")
        
        np.savez_compressed(cache_eurosat, K=K_e, y=y_e, gamma=gamma_e)
        log.info(f"Saved EuroSAT Kernel Cache: {cache_eurosat}")
        
    f1_mean_e, f1_std_e = run_evaluations(K_e, y_e)
    log.info(f"==> BSCM-PQK (L=6) EuroSAT F1: {f1_mean_e:.4f} +/- {f1_std_e:.4f}")

    # Summary
    res = {
        "depth": DEPTH,
        "n_so2sat": len(X_enc_s),
        "n_eurosat": len(X_enc_e),
        "so2sat_bscm_f1_mean": float(f1_mean_s),
        "so2sat_bscm_f1_std": float(f1_std_s),
        "eurosat_bscm_f1_mean": float(f1_mean_e),
        "eurosat_bscm_f1_std": float(f1_std_e),
    }
    with open(os.path.join(OUT_DIR, "max_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    log.info("Experiment 38 Completed Successfully.")


if __name__ == "__main__":
    main()
