"""
BSCM Phase 2 -- Hyperparameter LOCK-DOWN on the held-out pool.

Sweeps tau over a pre-registered grid using ONLY the 200-sample HOLDOUT
pool (defined in experiments/_bscm_split.py), then locks the chosen tau
to disk.  Phase 3 reads the locked tau and never sees these samples.

Pre-registered selection rule
-----------------------------
Pick the tau in TAU_GRID that maximises a *single* scalar criterion:

        score(tau)  =  KTA(K_BSCM(tau))      on the HOLDOUT pool

Ties broken by smaller tau (favour gentler perturbation).

This is *not* the full F1 because computing F1 here requires running the
SVM on the holdout pool, which would expose the holdout to the
classifier and contaminate Phase 3 baseline comparisons.  KTA is a
purely kernel-side, label-aware metric that does not see classifier
hyperparameters.

Output:
    results/bscm/locked_tau.json
    results/bscm/phase2_lockdown.log
    results/bscm/K_bscm_holdout_tau{tau}.npy   (cached holdout kernels)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._e_common import load_physics_16, spectral_metrics
from experiments._bscm_split import make_or_load_split
from src.attention_kernel import select_features_by_fisher, FEATURE_NAMES_16
from src.bscm_kernel import compute_bscm_fidelity_kernel
from src.srqfm_fidelity_kernel import compute_srqfm_fidelity_kernel

TAU_GRID = [0.25, 0.5, 0.75, 1.0, 1.5]   # pre-registered.
SAVE_DIR = os.path.join(config.RESULTS_DIR, "bscm")
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "phase2_lockdown.log"),
                             mode="w"),
    ],
)
logger = logging.getLogger("bscm_phase2_lockdown")


def kta(K: np.ndarray, y: np.ndarray) -> float:
    K_ideal = (y[:, None] == y[None, :]).astype(np.float64)
    num = float(np.sum(K * K_ideal))
    den = float(np.sqrt(np.sum(K ** 2) * np.sum(K_ideal ** 2)))
    return num / den if den > 1e-12 else 0.0


def main() -> int:
    logger.info("=" * 72)
    logger.info("  BSCM Phase 2 LOCK-DOWN  (holdout-only tau selection)")
    logger.info("=" * 72)

    X_norm, X_raw, y = load_physics_16()
    sel_idx, _ = select_features_by_fisher(X_raw, y)
    X_sel = X_norm[:, sel_idx]
    logger.info("  Selected features: %s",
                 [FEATURE_NAMES_16[i] for i in sel_idx])

    split = make_or_load_split(y)
    X_hold = X_sel[split.holdout]
    y_hold = y[split.holdout]
    logger.info("  Holdout pool: %d samples, %d unique classes",
                 len(y_hold), len(np.unique(y_hold)))

    # SRQFM-fid baseline at the same N for context.
    srqfm_path = os.path.join(SAVE_DIR, "K_srqfm_fid_holdout.npy")
    t0 = time.time()
    K_srqfm = compute_srqfm_fidelity_kernel(
        X_hold, n_qubits=config.N_QUBITS, reps=config.ZZ_REPS,
        coupling_threshold=0.01, connectivity="all",
        kernel_save_path=srqfm_path, desc="SRQFM-fid (holdout)",
    )
    sm_srqfm = spectral_metrics(K_srqfm)
    kta_srqfm = kta(K_srqfm, y_hold)
    logger.info("  SRQFM-fid: KTA=%.4f  off_var=%.4f  eff_rank=%.1f  (%.1fs)",
                 kta_srqfm, sm_srqfm["off_diag_var"],
                 sm_srqfm["eff_rank_shannon"], time.time() - t0)

    # Sweep BSCM over TAU_GRID.
    sweep = []
    for tau in TAU_GRID:
        cache = os.path.join(SAVE_DIR, f"K_bscm_holdout_tau{tau}.npy")
        t0 = time.time()
        K = compute_bscm_fidelity_kernel(
            X_hold, n_qubits=config.N_QUBITS, reps=config.ZZ_REPS, tau=tau,
            coupling_threshold=1e-4, connectivity="all",
            kernel_save_path=cache, desc=f"BSCM tau={tau}",
        )
        elapsed = time.time() - t0
        sm = spectral_metrics(K)
        k = kta(K, y_hold)
        sweep.append({
            "tau": tau, "kta": k,
            "off_diag_mean": sm["off_diag_mean"],
            "off_diag_var": sm["off_diag_var"],
            "eff_rank_shannon": sm["eff_rank_shannon"],
            "wall_clock_s": elapsed,
        })
        logger.info(
            "  tau=%.2f: KTA=%.4f  off_mean=%.4f  off_var=%.4f  "
            "eff_rank=%.1f  (%.1fs)",
            tau, k, sm["off_diag_mean"], sm["off_diag_var"],
            sm["eff_rank_shannon"], elapsed,
        )

    # Pre-registered selection rule.
    sweep_sorted = sorted(sweep, key=lambda r: (-r["kta"], r["tau"]))
    locked = sweep_sorted[0]
    logger.info("")
    logger.info("  Locked tau = %.2f  (KTA = %.4f)", locked["tau"], locked["kta"])

    out = {
        "selection_rule": "argmax KTA on holdout pool, ties broken by smaller tau",
        "tau_grid": TAU_GRID,
        "holdout_size": int(len(y_hold)),
        "split_seed": 1234,
        "sweep": sweep,
        "srqfm_fid_baseline": {
            "kta": kta_srqfm,
            "off_diag_mean": sm_srqfm["off_diag_mean"],
            "off_diag_var": sm_srqfm["off_diag_var"],
            "eff_rank_shannon": sm_srqfm["eff_rank_shannon"],
        },
        "locked": {"tau": locked["tau"], "kta": locked["kta"]},
    }
    out_path = os.path.join(SAVE_DIR, "locked_tau.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("  Saved: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
