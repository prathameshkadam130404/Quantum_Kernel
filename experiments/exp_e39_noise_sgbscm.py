"""
E39 -- SG-BSCM noise robustness on the EuroSAT physics-8 collapse setting
       (Global-Depolarising Model).

Question
--------
Does the SG-BSCM rescue persist when accumulated noise depolarises the
final encoded state?  Because per-gate density-matrix simulation at
n=8 with N=600 takes ~200+ hours per kernel on PennyLane's
default.mixed (measured at 4.56 s/pair), we adopt the standard
global-depolarising noise model used widely in the QML noise-analysis
literature [Heyraud et al. 2022; Schnabel & Roth QMI 2025]:
    rho_noisy(x) = (1 - p_eff) |psi(x)><psi(x)| + p_eff * I/d
The fidelity readout reduces analytically to
    K_noisy(x, x') = (1 - p_eff) K_pure(x, x') + p_eff / d
where d = 2^n is the Hilbert-space dimension and p_eff is the
accumulated depolarising strength interpreted as a single composite
channel applied to the final encoded state.  At p_eff = 0 we recover
the noiseless kernel; at p_eff -> 1 the kernel converges to a
data-independent constant 1/d (maximally mixed state).

Protocol
--------
- Dataset:        EuroSAT physics-8 (10 classes)
- N_pool:         600 stratified sub-sample of E33's N=1500 pool
- Splits:         5 stratified shuffle splits, random_state in {42..46}
- Qubits:         8  (Hilbert-space dim d = 256)
- Repetitions:    REPS (= config.ZZ_REPS, default 2)
- tau:            0.25 (matches main paper E33)
- Coupling thr.:  1e-4 (matches main paper E33)
- Connectivity:   all-to-all (matches main paper E33)
- p_eff sweep:    {0.0, 0.05, 0.10, 0.20, 0.40}
                  spans noiseless to ~half-depolarised, capturing the
                  hardware-realistic regime via the accumulated-noise
                  interpretation.
- Pure-kernel sim: PennyLane lightning.qubit (statevector, fast).
- Noise transform: analytic K_noisy = (1 - p_eff) K_pure + p_eff / d.

Output
------
- results/e39_noise_sgbscm/metrics.csv
- results/e39_noise_sgbscm/summary.json
- results/e39_noise_sgbscm/exp_e39.log
- results/e39_noise_sgbscm/K_pure_<method>.npy   (one per method)

Wall-clock estimate: ~30-60 minutes total (3 pure kernels on
lightning.qubit; the noise transform itself is microseconds).

Validation
----------
The global-depolarising model has been validated by prior work on
quantum kernels (Heyraud et al. 2022, App. C; Schnabel & Roth 2025,
Sec. 3.4) against per-gate Monte-Carlo and density-matrix simulation:
the qualitative trend of F1 vs accumulated noise is faithfully
reproduced, with the global model serving as a conservative
upper-bound estimate of the per-gate kernel matrix in the relevant
regime.

Author: Prathamesh Kadam et al.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, StratifiedShuffleSplit,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.bscm_kernel import (
    BELL_WEIGHT_PRESETS, compute_bscm_fidelity_kernel,
)
from src.srqfm_fidelity_kernel import compute_srqfm_fidelity_kernel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS         = [42, 43, 44, 45, 46]
N_QUBITS      = 8
HILBERT_DIM   = 2 ** N_QUBITS  # d = 256
REPS          = config.ZZ_REPS
TAU_LOCKED    = 0.25
COUPLING_THR  = 1e-4
N_POOL        = 600
TEST_FRAC     = 1.0 / 3.0
P_EFF_LIST    = [0.0, 0.05, 0.10, 0.20, 0.40]
QUANTUM_METHODS = ["BSCM-uniform", "SG-BSCM", "SRQFM-fid"]
C_GRID        = [1, 10, 100]
CV_FOLDS      = 3

SAVE_DIR = os.path.join(config.RESULTS_DIR, "e39_noise_sgbscm")
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "exp_e39.log"), mode="w"),
    ],
)
log = logging.getLogger("exp_e39")


# ============================================================================
# Data loader (mirrors exp_e33 EuroSAT physics8 path)
# ============================================================================

def _load_eurosat_physics8() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads EuroSAT physics-8 stratified sub-pool of size N_POOL.

    Pool master seed is fixed (random_state=0) so the pool is
    deterministic and reproducible across reruns.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from eurosat_data import load_eurosat_allbands, compute_physics_indices

    bm, labels, _ = load_eurosat_allbands(
        cache_path=os.path.join(config.PROCESSED_DIR, "eurosat_band_means.npz"),
    )
    phys_raw = compute_physics_indices(bm)  # (N, 8)

    pool_sss = StratifiedShuffleSplit(
        n_splits=1, train_size=N_POOL, random_state=0,
    )
    pool_idx, _ = next(pool_sss.split(phys_raw, labels))
    X_raw = phys_raw[pool_idx]
    y = labels[pool_idx]

    bg_mask = np.ones(len(phys_raw), dtype=bool); bg_mask[pool_idx] = False
    sc = StandardScaler().fit(phys_raw[bg_mask])
    mm = MinMaxScaler(feature_range=(0, np.pi)).fit(sc.transform(phys_raw[bg_mask]))
    X_enc = mm.transform(sc.transform(X_raw)).clip(0, np.pi).astype(np.float64)
    return X_enc, X_raw, y


# ============================================================================
# Pure-kernel builders (lightning.qubit, fast)
# ============================================================================

def build_pure_kernel(method: str, X_enc: np.ndarray) -> np.ndarray:
    """Build the noiseless gram matrix on lightning.qubit (statevector)."""
    cache_path = os.path.join(SAVE_DIR, f"K_pure_{method}.npy")
    if method == "BSCM-uniform":
        return compute_bscm_fidelity_kernel(
            X_enc, n_qubits=N_QUBITS, reps=REPS, tau=TAU_LOCKED,
            coupling_threshold=COUPLING_THR, connectivity="all",
            bell_weights=BELL_WEIGHT_PRESETS["uniform"],
            singlet_gated=False,
            kernel_save_path=cache_path, desc=f"pure {method}",
        )
    if method == "SG-BSCM":
        return compute_bscm_fidelity_kernel(
            X_enc, n_qubits=N_QUBITS, reps=REPS, tau=TAU_LOCKED,
            coupling_threshold=COUPLING_THR, connectivity="all",
            bell_weights=BELL_WEIGHT_PRESETS["uniform"],
            singlet_gated=True,
            kernel_save_path=cache_path, desc=f"pure {method}",
        )
    if method == "SRQFM-fid":
        return compute_srqfm_fidelity_kernel(
            X_enc, n_qubits=N_QUBITS, reps=REPS,
            coupling_threshold=COUPLING_THR, connectivity="all",
            kernel_save_path=cache_path, desc=f"pure {method}",
        )
    raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Analytic global-depolarising noise transform
# ============================================================================

def apply_global_depolarising(K_pure: np.ndarray, p_eff: float,
                              dim: int = HILBERT_DIM) -> np.ndarray:
    """K_noisy = (1 - p_eff) * K_pure + p_eff / d   (Hermitian, PSD, valid kernel).

    For p_eff = 0 returns K_pure unchanged.  For p_eff = 1 returns the
    constant kernel 1/d (maximally mixed state, fully data-independent).
    Diagonal preserved (each entry = (1-p_eff)*1 + p_eff/d = (1 - p_eff(d-1)/d),
    so the kernel is no longer normalised on the diagonal under noise; this
    is the correct behaviour and matches the standard noisy-kernel
    definition).
    """
    K_noisy = (1.0 - p_eff) * K_pure + p_eff / dim
    # Enforce numerical symmetry (purity is preserved analytically; this is
    # to remove tiny floating-point asymmetry from earlier computation).
    K_noisy = (K_noisy + K_noisy.T) / 2.0
    return K_noisy


# ============================================================================
# SVM helpers
# ============================================================================

def _tune_c(K_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> int:
    best_c, best_f1 = C_GRID[0], -1.0
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    for C in C_GRID:
        fold = []
        for tr2, va2 in skf.split(np.zeros(len(y_tr)), y_tr):
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_tr[np.ix_(tr2, tr2)], y_tr[tr2])
            yp = clf.predict(K_tr[np.ix_(va2, tr2)])
            fold.append(float(
                f1_score(y_tr[va2], yp, average="macro", zero_division=0)
            ))
        m = float(np.mean(fold))
        if m > best_f1:
            best_f1, best_c = m, C
    return best_c


def _eval_kernel(K: np.ndarray, y: np.ndarray, tr: np.ndarray,
                 te: np.ndarray, seed: int) -> dict:
    K_tr = K[np.ix_(tr, tr)]
    K_te = K[np.ix_(te, tr)]
    best_c = _tune_c(K_tr, y[tr], seed=seed)
    clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    yp = clf.predict(K_te)
    return {
        "macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0)),
        "best_C": int(best_c),
    }


def _eval_rbf(X_raw: np.ndarray, y: np.ndarray, tr: np.ndarray,
              te: np.ndarray) -> dict:
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_raw[tr])
    X_te = sc.transform(X_raw[te])
    grid = {"C": C_GRID, "gamma": ["scale", 0.01, 0.1, 1.0]}
    clf = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced"),
        grid, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1,
    )
    clf.fit(X_tr, y[tr])
    yp = clf.predict(X_te)
    return {
        "macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0)),
        "best_C": int(clf.best_params_["C"]),
        "best_gamma": str(clf.best_params_["gamma"]),
    }


def _kernel_health(K: np.ndarray, y: np.ndarray) -> dict:
    n = K.shape[0]
    iu = np.triu_indices(n, k=1)
    od = K[iu]
    od_mean = float(od.mean()); od_var = float(od.var())
    od_std = float(od.std())
    within = float(((od >= od_mean - od_std) & (od <= od_mean + od_std)).mean())
    Y = (y[:, None] == y[None, :]).astype(np.float64) * 2 - 1
    Yc = Y - Y.mean(); Kc = K - K.mean()
    kta = float((Kc * Yc).sum() /
                (np.sqrt((Kc * Kc).sum()) * np.sqrt((Yc * Yc).sum()) + 1e-12))
    return {
        "off_diag_mean": od_mean,
        "off_diag_var": od_var,
        "within_1sigma": within,
        "kta_product": kta,
    }


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    log.info("=" * 70)
    log.info("E39 -- SG-BSCM noise robustness (global-depolarising model)")
    log.info("=" * 70)
    log.info("Splits via random_state: %s", SEEDS)
    log.info("p_eff sweep: %s", P_EFF_LIST)
    log.info("Methods: %s", QUANTUM_METHODS)
    log.info("N_pool=%d  n_qubits=%d  d=%d  reps=%d  tau=%.2f",
             N_POOL, N_QUBITS, HILBERT_DIM, REPS, TAU_LOCKED)

    rows: List[dict] = []

    X_enc, X_raw, y = _load_eurosat_physics8()
    log.info("Loaded pool: X=%s  y unique=%s", X_enc.shape, np.unique(y))

    # Pre-compute the 5 train/test splits once
    splits = []
    for seed in SEEDS:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=TEST_FRAC, random_state=seed,
        )
        tr, te = next(sss.split(np.zeros(len(y)), y))
        splits.append((seed, tr, te))

    # ---- Classical RBF baseline (noise-free reference) --------------------
    log.info("\n=== RBF-SVM baseline (classical, noise-free) ===")
    for seed, tr, te in splits:
        rbf = _eval_rbf(X_raw, y, tr, te)
        rows.append({
            "seed": seed, "method": "RBF-SVM", "p_eff": -1.0,
            "macro_f1": rbf["macro_f1"], "best_C": rbf["best_C"],
            "best_gamma": rbf["best_gamma"],
            "off_diag_mean": np.nan, "off_diag_var": np.nan,
            "within_1sigma": np.nan, "kta_product": np.nan,
        })
        log.info("  seed=%d  F1=%.4f", seed, rbf["macro_f1"])

    # ---- Build the three pure quantum kernels (lightning.qubit) -----------
    log.info("\n=== Building pure (noiseless) quantum kernels ===")
    K_pure: Dict[str, np.ndarray] = {}
    for method in QUANTUM_METHODS:
        log.info("\n--- pure %s ---", method)
        t0 = time.time()
        K_pure[method] = build_pure_kernel(method, X_enc)
        log.info("  built in %.0fs", time.time() - t0)
        h = _kernel_health(K_pure[method], y)
        log.info(
            "  noiseless health: off_diag_mean=%.4f  KTA=%.3f  within1sig=%.3f",
            h["off_diag_mean"], h["kta_product"], h["within_1sigma"],
        )

    # ---- Sweep p_eff via the analytic global-depolarising transform -------
    log.info("\n=== Sweeping p_eff via analytic noise transform ===")
    for method in QUANTUM_METHODS:
        for p_eff in P_EFF_LIST:
            K = apply_global_depolarising(K_pure[method], p_eff, dim=HILBERT_DIM)
            health = _kernel_health(K, y)
            for seed, tr, te in splits:
                ek = _eval_kernel(K, y, tr, te, seed=seed)
                rows.append({
                    "seed": seed, "method": method, "p_eff": float(p_eff),
                    "macro_f1": ek["macro_f1"], "best_C": ek["best_C"],
                    "best_gamma": "",
                    **health,
                })
            f1s = [r["macro_f1"] for r in rows
                   if r["method"] == method and r["p_eff"] == p_eff]
            log.info(
                "  %s p_eff=%.2f: F1=%.4f +- %.4f  (KTA=%.3f, off=%.4f)",
                method, p_eff, np.mean(f1s), np.std(f1s),
                health["kta_product"], health["off_diag_mean"],
            )

    # ---- Save -------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)
    log.info("\nWrote metrics.csv (%d rows)", len(df))

    summary: Dict = {
        "experiment": "E39_noise_sgbscm_global_depolarising",
        "noise_model": "global depolarising: K_noisy = (1-p_eff) K_pure + p_eff/d",
        "config": {
            "n_qubits": N_QUBITS, "hilbert_dim": HILBERT_DIM,
            "reps": REPS, "tau": TAU_LOCKED,
            "coupling_threshold": COUPLING_THR, "n_pool": N_POOL,
            "test_frac": TEST_FRAC, "seeds": SEEDS, "p_eff_list": P_EFF_LIST,
        },
        "results": {},
    }
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        if method == "RBF-SVM":
            summary["results"][method] = {
                "f1_mean": float(sub["macro_f1"].mean()),
                "f1_std": float(sub["macro_f1"].std()),
            }
            continue
        per_p = {}
        for p in sorted(sub["p_eff"].unique()):
            ss = sub[sub["p_eff"] == p]
            per_p[f"{p:.4f}"] = {
                "f1_mean": float(ss["macro_f1"].mean()),
                "f1_std": float(ss["macro_f1"].std()),
                "off_diag_mean": float(ss["off_diag_mean"].iloc[0]),
                "kta_product": float(ss["kta_product"].iloc[0]),
            }
        summary["results"][method] = per_p

    # Rescue gap = SG-BSCM - BSCM-uniform per p_eff (paired by seed)
    gap = {}
    for p in P_EFF_LIST:
        a = df[(df["method"] == "SG-BSCM") & (df["p_eff"] == p)]["macro_f1"].values
        b = df[(df["method"] == "BSCM-uniform") & (df["p_eff"] == p)]["macro_f1"].values
        gap[f"{p:.4f}"] = {
            "rescue_gap_mean": float((a - b).mean()),
            "rescue_gap_std": float((a - b).std()),
        }
    summary["rescue_gap"] = gap

    with open(os.path.join(SAVE_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote summary.json")
    log.info("DONE.")


if __name__ == "__main__":
    main()
