"""
CV-KTA bandwidth spot-check on E38 (16 qubits, So2Sat physics-Fisher-16)
Bloch features.

Cross-validates the sklearn-``scale`` PQK-bandwidth heuristic used by E38

    gamma_dyn  =  1 / (d * Var[phi])

(d = 3 * n_qubits = 48; Var[phi] = training-set Bloch variance) against
5-fold CV-KTA over a multiplier grid.  E26 / E32 / E37 use CV-KTA on a
small pool; this script tests whether the heuristic disagrees with what
CV-KTA would select at the larger E38 scale.

Procedure
---------
Re-extract Bloch vectors for N=1000 stratified sub-samples of the E38
So2Sat pool for both BSCM-uniform and Standard-ZZ.  For each kernel:
1.  Compute the dynamic heuristic gamma exactly as the headline run.
2.  Run 5-fold StratifiedKFold CV-KTA over the multiplier grid
        gamma_grid = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0] * gamma_dyn.
3.  Report the multiplier that maximises mean CV-KTA.  A value of 1.0
    (or neighbouring grid points) indicates the heuristic agrees with the
    CV-KTA optimum at the E38 scale; a value at the grid boundary
    indicates the heuristic is under- or over-tuned at this scale.

Output ``results/e38_max_data/cvkta_spotcheck.json`` records both the
heuristic and CV-KTA optima per kernel, the KTA at each grid point, and
the N=1000 stratified-shuffle indices used (so the audit is reproducible).

Wall-clock cost: ~10-20 min per kernel at n=16 on default.qubit
(~25-40 min total).  Both Bloch extractors are imported from the
production E38 scripts so the topology, depth, and tau match exactly.

Usage
-----
    python experiments/audit_e38_cvkta_bandwidth.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# These imports re-use the exact extractor topology used by the published
# E38 numbers (Ladder connectivity, depth=6, tau=0.25).
from experiments.exp_e38_max_data_pqk import (
    extract_bloch_vectors as extract_bscm_bloch,
    _load_so2sat_max,
    N_QUBITS, DEPTH,
)
from experiments.exp_e38_max_data_standard_pqk import (
    extract_standard_pqk_bloch_vectors as extract_standard_bloch,
)
from src.bscm_kernel import BELL_WEIGHT_PRESETS

N_SUB = 1000
GAMMA_MULTIPLIERS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
KFOLDS = 5
SUB_SEED = 42

E38_DIR = os.path.join(config.RESULTS_DIR, "e38_max_data")
OUT_PATH = os.path.join(E38_DIR, "cvkta_spotcheck.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(E38_DIR, "audit_cvkta_bandwidth.log"),
                            mode="w"),
    ],
)
log = logging.getLogger("audit_cvkta")


def _kta_centered(K: np.ndarray, y: np.ndarray) -> float:
    """Product-label KTA, the formulation used in the E37/E38 codebase."""
    K_ideal = (y[:, None] == y[None, :]).astype(np.float64)
    num = float(np.sum(K * K_ideal))
    den = float(np.sqrt(np.sum(K ** 2) * np.sum(K_ideal ** 2)))
    return num / den if den > 1e-12 else 0.0


def _build_distance_matrix(bloch: np.ndarray) -> np.ndarray:
    """Squared pairwise Euclidean distance in flattened Bloch space."""
    b = bloch.reshape(len(bloch), -1).astype(np.float64)
    sq = np.sum(b ** 2, axis=1)
    d = sq[:, None] + sq[None, :] - 2.0 * (b @ b.T)
    return np.clip(d, 0.0, None)


def _cv_kta(
    bloch: np.ndarray, y: np.ndarray, gamma: float,
    kf: StratifiedKFold,
) -> List[float]:
    """KTA on each KFold training partition at fixed gamma."""
    d_full = _build_distance_matrix(bloch)
    scores: List[float] = []
    for tr_idx, _ in kf.split(np.zeros(len(y)), y):
        d_tr = d_full[np.ix_(tr_idx, tr_idx)]
        K_tr = np.exp(-gamma * 0.5 * d_tr)
        np.fill_diagonal(K_tr, 1.0)
        K_tr = (K_tr + K_tr.T) / 2.0
        scores.append(_kta_centered(K_tr, y[tr_idx]))
    return scores


def _audit_one_kernel(
    name: str, bloch: np.ndarray, y: np.ndarray,
) -> Dict[str, object]:
    """Run the CV-KTA sweep for a single kernel and return audit fields."""
    b_flat = bloch.reshape(len(bloch), -1).astype(np.float64)
    b_var = float(np.var(b_flat))
    d = b_flat.shape[1]
    gamma_dyn = 1.0 / (d * b_var) if b_var > 0 else 1.0
    log.info("[%s] N=%d, d=%d, Var[phi]=%.6f, gamma_dyn=%.6f",
             name, len(bloch), d, b_var, gamma_dyn)

    kf = StratifiedKFold(n_splits=KFOLDS, shuffle=True, random_state=SUB_SEED)

    sweep = []
    best = {"gamma_multiplier": None, "gamma": None, "kta_mean": -np.inf,
            "kta_std": None}
    for gm in GAMMA_MULTIPLIERS:
        gamma = gamma_dyn * gm
        scores = _cv_kta(bloch, y, gamma, kf)
        m, s = float(np.mean(scores)), float(np.std(scores))
        sweep.append({
            "gamma_multiplier": gm, "gamma": gamma,
            "kta_per_fold": [float(x) for x in scores],
            "kta_mean": m, "kta_std": s,
        })
        log.info("  gm=%.2f  gamma=%.4g  KTA=%.4f +/- %.4f",
                 gm, gamma, m, s)
        if m > best["kta_mean"]:
            best = {"gamma_multiplier": float(gm), "gamma": float(gamma),
                    "kta_mean": m, "kta_std": s}

    # KTA at the heuristic gamma (multiplier == 1.0).
    heur_row = next(r for r in sweep if r["gamma_multiplier"] == 1.0)
    return {
        "n_subsample": int(len(bloch)),
        "bloch_dim": int(d),
        "bloch_variance_train": b_var,
        "gamma_dyn_heuristic": float(gamma_dyn),
        "gamma_grid_multipliers": GAMMA_MULTIPLIERS,
        "cv_kfolds": KFOLDS,
        "sweep": sweep,
        "cv_optimum": best,
        "heuristic_kta": heur_row["kta_mean"],
        "delta_optimum_vs_heuristic": float(
            best["kta_mean"] - heur_row["kta_mean"]
        ),
        "interpretation": (
            "If gamma_multiplier of the CV optimum is 1.0 the heuristic "
            "matches CV-KTA exactly.  Multipliers in {0.5, 2.0} indicate "
            "the heuristic is within a small constant factor of CV optimum; "
            "anything farther suggests the heuristic disadvantages the "
            "kernel at this scale."
        ),
    }


def main() -> None:
    log.info("=" * 72)
    log.info("  E38 CV-KTA bandwidth spot-check (N=%d sub-sample)", N_SUB)
    log.info("=" * 72)

    log.info("Loading So2Sat (full E38 pool, then stratified subsample)...")
    X_enc, X_raw, y = _load_so2sat_max()
    log.info("  Full pool: N=%d, %d classes", len(y), len(np.unique(y)))

    # Stratified subsample of size N_SUB.
    sss = StratifiedShuffleSplit(n_splits=1, train_size=N_SUB,
                                 test_size=None, random_state=SUB_SEED)
    (sub_idx, _), = sss.split(X_enc, y)
    sub_idx = np.sort(sub_idx)
    X_sub, y_sub = X_enc[sub_idx], y[sub_idx]
    log.info("  Sub-sample: N=%d, class counts %s",
             len(y_sub),
             dict(zip(*np.unique(y_sub, return_counts=True))))

    # Extract Bloch vectors for both kernels.
    log.info("")
    log.info("[1/2] Extracting BSCM-uniform Bloch (n=%d, depth=%d, N=%d)...",
             N_QUBITS, DEPTH, N_SUB)
    t0 = time.time()
    bscm_bloch = extract_bscm_bloch(
        X_sub.astype(np.float32), N_QUBITS, DEPTH, "BSCM-uniform spot",
        BELL_WEIGHT_PRESETS["uniform"],
    )
    log.info("  BSCM Bloch shape=%s, wall=%.0fs", bscm_bloch.shape,
             time.time() - t0)

    log.info("")
    log.info("[2/2] Extracting Standard-ZZ Bloch (n=%d, depth=%d, N=%d)...",
             N_QUBITS, DEPTH, N_SUB)
    t0 = time.time()
    std_bloch = extract_standard_bloch(
        X_sub.astype(np.float32), N_QUBITS, DEPTH, "Standard-ZZ spot",
    )
    log.info("  Standard-ZZ Bloch shape=%s, wall=%.0fs", std_bloch.shape,
             time.time() - t0)

    out = {
        "dataset": "so2sat",
        "feature_set": "physics_Fisher_16",
        "n_qubits": int(N_QUBITS),
        "depth": int(DEPTH),
        "tau_locked": 0.25,
        "subsample_seed": SUB_SEED,
        "subsample_indices": sub_idx.tolist(),
        "kernels": {
            "BSCM-uniform-PQK": _audit_one_kernel("BSCM-uniform", bscm_bloch, y_sub),
            "Standard-ZZ-PQK": _audit_one_kernel("Standard-ZZ", std_bloch, y_sub),
        },
        "rationale": (
            "Cross-validates the bandwidth heuristic (sklearn-scale style "
            "gamma = 1/(d*Var[phi])) actually used by E38 against 5-fold "
            "CV-KTA on a N=1,000 stratified sub-sample.  Multiplier = 1.0 "
            "is the heuristic; the CV-KTA optimum reports the relative "
            "scale at which the heuristic operates."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    log.info("")
    log.info("Wrote %s", OUT_PATH)

    # Console summary.
    for k in ("BSCM-uniform-PQK", "Standard-ZZ-PQK"):
        r = out["kernels"][k]
        opt = r["cv_optimum"]
        log.info("[%s] heuristic KTA=%.4f, CV-opt KTA=%.4f at gm=%.2f, "
                 "delta=%+.4f",
                 k, r["heuristic_kta"], opt["kta_mean"],
                 opt["gamma_multiplier"], r["delta_optimum_vs_heuristic"])


if __name__ == "__main__":
    main()
