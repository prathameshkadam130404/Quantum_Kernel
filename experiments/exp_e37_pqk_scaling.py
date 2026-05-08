"""
E37 -- PQK Scaling Analysis (XX+YY vs ZZ at 16 Qubits)

Motivation
----------
At 8 qubits, the Hilbert space (256 dimensions) is too small to show 
distinct quantum advantage over classical RBF kernels. To properly test 
if the non-commuting XX+YY generators (BSCM-PQK) overpower the commuting 
ZZ generators (Normal PQK), we must scale to 16 Qubits.

Data Structure:
Rather than using artificial data re-uploading (duplicating features), 
this experiment uses *true* 16-dimensional physical features:
1. So2Sat: Uses the native `physics_features_16.npz` (8 Optical + 8 SAR).
2. EuroSAT: Synthesizes 16 features by combining the 8 physics indices 
   with the first 8 raw Sentinel-2 bands.

We use a block-bipartite (8+8) connectivity: linear chains within each
block plus rungs between blocks. On So2Sat the two blocks correspond to
optical (Sentinel-2) and SAR (Sentinel-1) features; on EuroSAT (single-
modal Sentinel-2) the split is by Fisher rank for protocol consistency.
The sparser topology prevents instantaneous scrambling (concentration /
barren plateaus) while letting the XX+YY geometry build across depths
L=2, 4, 6, 8, 10.

Protocol
--------
Datasets:    So2Sat physics16 (N=1800), EuroSAT physics16 (N=1500 pool)
Qubits:      16
Depths (L):  [2, 4, 6, 8, 10]
Topology:    Block-bipartite (8+8) ladder
Methods:     1. SG-BSCM-PQK (XX+YY, gated)
             2. SRQFM-PQK (ZZ)
             3. RBF-SVM (Classical baseline)

Author: Prathamesh Kadam et al.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm

import pennylane as qml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.bscm_kernel import compute_bscm_pqk_gram
from experiments._bscm_split import make_or_load_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS           = config.SEED_LIST
N_QUBITS        = 16
DEPTHS          = [2, 4, 6, 8, 10]
TAU_LOCKED      = 0.25
PQK_GAMMA       = config.PQK_GAMMA
C_GRID          = [1, 10, 100]
CV_FOLDS        = 3

SO2SAT_TEST_FRAC = 0.30
EUROSAT_N_TRAIN = 1000
EUROSAT_N_TEST  = 500

SAVE_DIR = os.path.join(config.RESULTS_DIR, "e37_pqk_scaling")
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "exp_e37.log"), mode="w"),
    ],
)
log = logging.getLogger("exp_e37")

# ============================================================================
# True 16-Feature Loaders
# ============================================================================

def _load_so2sat_16() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load native So2Sat 16 features (8 OPT + 8 SAR)."""
    p = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
    d = np.load(p)
    # Use full dataset for train splits, then filter down to eval pool
    X_raw_all = d["X_train_raw"][: config.SUBSAMPLE_TRAIN]
    X_enc_all = d["X_train"][: config.SUBSAMPLE_TRAIN]
    y_all = d["y_train"][: config.SUBSAMPLE_TRAIN]
    
    split = make_or_load_split(y_all)
    return X_enc_all[split.eval_pool], X_raw_all[split.eval_pool], y_all[split.eval_pool]

def _load_eurosat_16() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize EuroSAT 16 features: Top 16 of (8 indices + 13 raw bands) by Fisher Ratio."""
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
    
    # Fixed Pool selection (N=1500)
    pool_sss = StratifiedShuffleSplit(n_splits=1, train_size=EUROSAT_N_TRAIN, 
                                      test_size=EUROSAT_N_TEST, random_state=0)
    tr, te = next(pool_sss.split(X_raw_all, labels))
    pool_idx = np.concatenate([tr, te])
    
    # Background selection (Everything NOT in the pool)
    background_mask = np.ones(len(X_raw_all), dtype=bool)
    background_mask[pool_idx] = False
    
    X_raw_background = X_raw_all[background_mask]
    y_background = labels[background_mask]
    
    # FEATURE SELECTION: Compute Fisher Ratio STRICTLY on the background set
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
    X_enc_pool = mm.transform(X_scaled).astype(np.float32)
    
    return X_enc_pool, X_raw_pool, y_pool

# ============================================================================
# SVM evaluation helpers
# ============================================================================

def _tune_c_precomputed(K_train: np.ndarray, y_train: np.ndarray, seed: int) -> float:
    best_c, best_f1 = C_GRID[0], -1.0
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    for C in C_GRID:
        fold_f1 = []
        for cv_tr, cv_val in skf.split(np.zeros(len(y_train)), y_train):
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_train[np.ix_(cv_tr, cv_tr)], y_train[cv_tr])
            yp = clf.predict(K_train[np.ix_(cv_val, cv_tr)])
            fold_f1.append(f1_score(y_train[cv_val], yp, average="macro", zero_division=0))
        m = float(np.mean(fold_f1))
        if m > best_f1: best_f1, best_c = m, C
    return best_c

def _eval_kernel(K_full: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, seed: int) -> dict:
    K_tr = K_full[np.ix_(tr, tr)]
    K_te = K_full[np.ix_(te, tr)]
    best_c = _tune_c_precomputed(K_tr, y[tr], seed=seed)
    clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    yp = clf.predict(K_te)
    return {"macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0)), "best_C": best_c}

# ============================================================================
# Quantum Circuit Constructors (16-qubit optimized)
# ============================================================================

def _build_sg_bscm_bloch_extractor(n_qubits: int, reps: int, singlet_gated: bool):
    """XX+YY channel, Ladder connectivity (OPT-OPT, SAR-SAR, and OPT-SAR)."""
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
                cp = np.cos((x[qi] + x[qj]) / 2.0)**2
                sp = np.sin((x[qi] + x[qj]) / 2.0)**2
                cm = np.cos((x[qi] - x[qj]) / 2.0)**2
                sm = np.sin((x[qi] - x[qj]) / 2.0)**2
                
                a_xx = (cp - sp + cm - sm) / 4.0
                a_yy = (-cp + sp + cm - sm) / 4.0
                
                if singlet_gated:
                    sg = float(np.sin((x[qi] - x[qj]) / 2.0)**2)
                else:
                    sg = 1.0
                    
                c_xx = 2.0 * TAU_LOCKED * a_xx * sg
                c_yy = 2.0 * TAU_LOCKED * a_yy * sg
                
                if abs(c_xx) > 1e-5: qml.IsingXX(c_xx, wires=[qi, qj])
                if abs(c_yy) > 1e-5: qml.IsingYY(c_yy, wires=[qi, qj])

    @qml.qnode(dev, diff_method=None)
    def mx(x): _apply_circuit(x); return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def my(x): _apply_circuit(x); return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def mz(x): _apply_circuit(x); return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return mx, my, mz


def _build_srqfm_bloch_extractor(n_qubits: int, reps: int):
    """ZZ channel, Ladder connectivity (OPT-OPT, SAR-SAR, and OPT-SAR)."""
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


def extract_bloch(X: np.ndarray, method: str, reps: int, dataset: str) -> np.ndarray:
    cache = os.path.join(SAVE_DIR, f"bloch_{method}_{dataset}_L{reps}.npy")
    if os.path.exists(cache):
        log.info("  [CACHE] Loaded %s (L=%d) %s", method, reps, dataset)
        return np.load(cache)

    log.info("  Computing %s (L=%d) %s...", method, reps, dataset)
    if method == "SG-BSCM-PQK":
        mx, my, mz = _build_sg_bscm_bloch_extractor(N_QUBITS, reps, singlet_gated=True)
    elif method == "BSCM-PQK":
        mx, my, mz = _build_sg_bscm_bloch_extractor(N_QUBITS, reps, singlet_gated=False)
    else:
        mx, my, mz = _build_srqfm_bloch_extractor(N_QUBITS, reps)
        
    N = len(X)
    bloch = np.zeros((N, 3 * N_QUBITS), dtype=np.float64)
    t0 = time.time()
    for i in tqdm(range(N), desc=f"{method} L={reps}"):
        bx, by, bz = mx(X[i]), my(X[i]), mz(X[i])
        for q in range(N_QUBITS):
            bloch[i, 3*q]   = bx[q]
            bloch[i, 3*q+1] = by[q]
            bloch[i, 3*q+2] = bz[q]
            
    mags = np.sqrt(bloch[:, 0::3]**2 + bloch[:, 1::3]**2 + bloch[:, 2::3]**2).mean()
    log.info("  Mean Bloch magnitude: %.4f (Time: %.1fs)", mags, time.time() - t0)
    
    np.save(cache, bloch)
    return bloch


# ============================================================================
# Main Execution
# ============================================================================

def run_condition(dataset: str, X_enc: np.ndarray, X_raw: np.ndarray, y: np.ndarray, rows: List[dict], summary: dict):
    log.info("\n" + "=" * 60)
    log.info("  Condition: %s (N=%d, features=%d)", dataset, len(y), X_enc.shape[1])
    log.info("=" * 60)
    
    # 1. Classical Baseline
    rbf_scores = []
    for seed in SEEDS:
        if dataset == "so2sat":
            sss = StratifiedShuffleSplit(n_splits=1, test_size=SO2SAT_TEST_FRAC, random_state=seed)
        else:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=EUROSAT_N_TRAIN, test_size=EUROSAT_N_TEST, random_state=seed)
        
        (tr, te), = sss.split(np.zeros(len(y)), y)
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_raw[tr])
        X_te = scaler.transform(X_raw[te])
        grid = {"C": C_GRID, "gamma": ["scale", 0.01, 0.1, 1.0]}
        clf = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), grid, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1)
        clf.fit(X_tr, y[tr])
        yp = clf.predict(X_te)
        rbf_scores.append(f1_score(y[te], yp, average="macro"))
        
    rbf_mean, rbf_std = np.mean(rbf_scores), np.std(rbf_scores)
    log.info("  RBF-SVM F1: %.4f +/- %.4f\n", rbf_mean, rbf_std)
    summary[f"{dataset}_RBF-SVM"] = {"mean": rbf_mean, "std": rbf_std}
    
    # 2. Quantum Methods Across Depths
    for L in DEPTHS:
        for method in ["SRQFM-PQK", "BSCM-PQK", "SG-BSCM-PQK"]:
            bloch = extract_bloch(X_enc, method, L, dataset)
            K = compute_bscm_pqk_gram(bloch, n_qubits=N_QUBITS, gamma=PQK_GAMMA)
            off_diag = K[~np.eye(len(K), dtype=bool)].mean()
            
            scores = []
            for seed in SEEDS:
                if dataset == "so2sat":
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=SO2SAT_TEST_FRAC, random_state=seed)
                else:
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=EUROSAT_N_TRAIN, test_size=EUROSAT_N_TEST, random_state=seed)
                (tr, te), = sss.split(np.zeros(len(y)), y)
                res = _eval_kernel(K, y, tr, te, seed)
                scores.append(res["macro_f1"])
                rows.append({"dataset": dataset, "method": method, "L": L, "seed": seed, "f1": res["macro_f1"]})
                
            m_mean, m_std = np.mean(scores), np.std(scores)
            log.info("  --> %s (L=%d) F1: %.4f +/- %.4f  (K_off_diag=%.4f)", method, L, m_mean, m_std, off_diag)
            summary[f"{dataset}_{method}_L{L}"] = {"mean": m_mean, "std": m_std, "off_diag": off_diag}

def run():
    log.info("E37: PQK Scaling Analysis (True 16 Features, 16 Qubits)")
    rows, summary = [], {}
    
    # Evaluate So2Sat (True 8 OPT + 8 SAR)
    X_enc_so, X_raw_so, y_so = _load_so2sat_16()
    run_condition("so2sat", X_enc_so, X_raw_so, y_so, rows, summary)
    
    # Evaluate EuroSAT (True 8 Indices + 8 Bands)
    X_enc_eu, X_raw_eu, y_eu = _load_eurosat_16()
    run_condition("eurosat", X_enc_eu, X_raw_eu, y_eu, rows, summary)
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)
    with open(os.path.join(SAVE_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    log.info("\nExperiment E37 Complete.")

if __name__ == "__main__":
    run()
