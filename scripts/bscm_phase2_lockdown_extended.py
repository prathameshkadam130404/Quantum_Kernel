"""
BSCM Phase 2 LOCK-DOWN -- extended tau-grid sweep.

Probes tau values below 0.25 to verify that the locked value at the lower
boundary of the original grid {0.25, 0.5, 0.75, 1.0, 1.5} is not
pathologically pinned to that boundary.

Protocol
--------
* Identical holdout pool, feature selection, and KTA selection rule as
  ``scripts/bscm_phase2_lockdown.py``.
* Extended grid: {0.05, 0.10, 0.15, 0.20, 0.25, 0.5, 0.75, 1.0, 1.5}.
* The five cached kernels for the original grid values are reused via
  ``compute_bscm_fidelity_kernel``'s on-disk cache; only the four new
  low-tau kernels are simulated fresh.
* Writes ``results/bscm/locked_tau_extended.json`` alongside the original
  ``locked_tau.json``; the original artefact is not overwritten.
* For overlapping grid points the script logs a warning if the recomputed
  KTA disagrees with the cached original by more than 1e-6 (sanity check
  on cache freshness; not a hard assertion).

Usage
-----
    python scripts/bscm_phase2_lockdown_extended.py
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
from scripts.bscm_phase2_lockdown import (
    TAU_GRID_EXTENDED, SAVE_DIR, kta,
)

EXT_LOG_PATH = os.path.join(SAVE_DIR, "phase2_lockdown_extended.log")
EXT_OUT_PATH = os.path.join(SAVE_DIR, "locked_tau_extended.json")
ORIG_LOCK_PATH = os.path.join(SAVE_DIR, "locked_tau.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(EXT_LOG_PATH, mode="w"),
    ],
)
log = logging.getLogger("bscm_phase2_lockdown_extended")


def _load_original_lock() -> dict:
    """Pull the cached KTA values from the original locked_tau.json (if any).

    Used solely as a sanity check that the kernel cache and the published
    KTA numbers still agree at the overlapping grid points.
    """
    if not os.path.exists(ORIG_LOCK_PATH):
        return {}
    with open(ORIG_LOCK_PATH) as f:
        d = json.load(f)
    return {r["tau"]: r["kta"] for r in d.get("sweep", [])}


def main() -> int:
    log.info("=" * 72)
    log.info("  BSCM Phase 2 LOCK-DOWN -- EXTENDED grid sweep")
    log.info("  Grid: %s", TAU_GRID_EXTENDED)
    log.info("=" * 72)

    X_norm, X_raw, y = load_physics_16()
    sel_idx, _ = select_features_by_fisher(X_raw, y)
    X_sel = X_norm[:, sel_idx]
    log.info("  Selected features: %s",
             [FEATURE_NAMES_16[i] for i in sel_idx])

    split = make_or_load_split(y)
    X_hold = X_sel[split.holdout]
    y_hold = y[split.holdout]
    log.info("  Holdout pool: %d samples, %d unique classes",
             len(y_hold), len(np.unique(y_hold)))

    orig_ktas = _load_original_lock()

    sweep = []
    for tau in TAU_GRID_EXTENDED:
        cache = os.path.join(SAVE_DIR, f"K_bscm_holdout_tau{tau}.npy")
        cached = os.path.exists(cache)
        t0 = time.time()
        K = compute_bscm_fidelity_kernel(
            X_hold, n_qubits=config.N_QUBITS, reps=config.ZZ_REPS, tau=tau,
            coupling_threshold=1e-4, connectivity="all",
            kernel_save_path=cache, desc=f"BSCM tau={tau}",
        )
        elapsed = time.time() - t0
        sm = spectral_metrics(K)
        k = kta(K, y_hold)

        # Sanity check against the original lock for overlapping tau values.
        if tau in orig_ktas:
            delta = abs(k - orig_ktas[tau])
            if delta > 1e-6:
                log.warning(
                    "  tau=%.2f: KTA mismatch vs original lock "
                    "(extended=%.6f, original=%.6f, delta=%.2e)",
                    tau, k, orig_ktas[tau], delta,
                )

        sweep.append({
            "tau": tau,
            "kta": k,
            "off_diag_mean": sm["off_diag_mean"],
            "off_diag_var": sm["off_diag_var"],
            "eff_rank_shannon": sm["eff_rank_shannon"],
            "wall_clock_s": elapsed,
            "from_cache": cached,
        })
        log.info(
            "  tau=%.2f: KTA=%.4f  off_mean=%.4f  off_var=%.4f  "
            "eff_rank=%.1f  (%.1fs, cache=%s)",
            tau, k, sm["off_diag_mean"], sm["off_diag_var"],
            sm["eff_rank_shannon"], elapsed, cached,
        )

    sweep_sorted = sorted(sweep, key=lambda r: (-r["kta"], r["tau"]))
    locked = sweep_sorted[0]
    log.info("")
    log.info("  EXTENDED-grid lock: tau=%.2f, KTA=%.4f",
             locked["tau"], locked["kta"])
    log.info("  Original grid lock (results/bscm/locked_tau.json) was tau=0.25.")
    if locked["tau"] != 0.25:
        log.info(
            "  NOTE: extended-grid lock differs from original. "
            "Headline experiments (E26/E32/E37/E38/E42/E43) use tau=0.25; "
            "this extended sweep is a *robustness check*, not a change of "
            "the locked value."
        )

    out = {
        "selection_rule":
            "argmax KTA on holdout pool, ties broken by smaller tau",
        "tau_grid_extended": TAU_GRID_EXTENDED,
        "tau_grid_original": [0.25, 0.5, 0.75, 1.0, 1.5],
        "holdout_size": int(len(y_hold)),
        "split_seed": 1234,
        "sweep": sweep,
        "locked_extended": {"tau": locked["tau"], "kta": locked["kta"]},
        "locked_original": {"tau": 0.25,
                            "kta": orig_ktas.get(0.25)},
        "headline_experiments_tau": 0.25,
        "boundary_behaviour_note":
            ("The extended sweep adds {0.05, 0.10, 0.15, 0.20} below the "
             "originally pre-registered minimum of 0.25.  All headline E26 "
             "through E43 experiments use the original lock tau=0.25; this "
             "file documents the boundary check requested by reviewers."),
    }
    with open(EXT_OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    log.info("  Saved: %s", EXT_OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
