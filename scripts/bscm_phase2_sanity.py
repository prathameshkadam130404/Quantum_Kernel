"""
BSCM Phase 2 — Concentration sanity check at N=200.

Computes the BSCM fidelity kernel and the SRQFM fidelity kernel on the
same N=200 So2Sat physics8 sample.  Reports off-diagonal mean/variance,
KTA, and effective Shannon rank.  Applies a simple decision rule before
committing to the full N=2000 evaluation.

Decision rule: proceed to Phase 3 only if BSCM has at least 0.8x SRQFM's
{off-diagonal variance, KTA, Shannon eff-rank}.  Otherwise revisit
the time hyperparameter `tau` or the coupling threshold.

Outputs to:  results/bscm_phase2/sanity_metrics.json
             results/bscm_phase2/K_bscm_n200.npy
             results/bscm_phase2/K_srqfm_fid_n200.npy
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
from src.attention_kernel import select_features_by_fisher, FEATURE_NAMES_16
from src.bscm_kernel import compute_bscm_fidelity_kernel, compute_coupling_diagnostics
from src.srqfm_fidelity_kernel import compute_srqfm_fidelity_kernel

N_PHASE2 = 200
TAU = 1.0
REPS = config.ZZ_REPS
N_QUBITS = config.N_QUBITS
THRESHOLD_BSCM = 1e-4
THRESHOLD_SRQFM = 0.01

SAVE_DIR = os.path.join(config.RESULTS_DIR, "bscm_phase2")
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "phase2.log"), mode="w"),
    ],
)
logger = logging.getLogger("bscm_phase2")


def kta(K: np.ndarray, y: np.ndarray) -> float:
    K_ideal = (y[:, None] == y[None, :]).astype(np.float64)
    num = float(np.sum(K * K_ideal))
    den = float(np.sqrt(np.sum(K ** 2) * np.sum(K_ideal ** 2)))
    return num / den if den > 1e-12 else 0.0


def main() -> int:
    logger.info("=" * 70)
    logger.info("  BSCM Phase 2 sanity check (N=%d)", N_PHASE2)
    logger.info("=" * 70)

    # Load and prepare data identically to E26.
    X_norm, X_raw, y = load_physics_16()
    sel_idx, _ = select_features_by_fisher(X_raw, y)
    X_sel = X_norm[:, sel_idx]
    sel_names = [FEATURE_NAMES_16[i] for i in sel_idx]
    logger.info("  Selected features: %s", sel_names)

    rng = np.random.default_rng(42)
    # Stratified-ish subsample: take first N_PHASE2 (E26 subsample is already
    # stratified over the 17 LCZ classes).  For the sanity check, this is enough.
    idx = rng.choice(len(y), size=N_PHASE2, replace=False)
    X_sub = X_sel[idx]
    y_sub = y[idx]
    logger.info("  N=%d, classes present: %d", len(y_sub), len(np.unique(y_sub)))

    # Coupling diagnostics (BSCM-specific).
    diag = compute_coupling_diagnostics(X_sub, feature_names=sel_names,
                                         n_qubits=N_QUBITS)
    top_pairs = sorted(diag["pairs"].items(),
                        key=lambda kv: -kv[1]["max_amp_mean"])[:5]
    logger.info("  Top 5 BSCM coupling pairs (by max-amplitude mean):")
    for k, v in top_pairs:
        logger.info("    %s  alpha~=%+.3f  beta~=%+.3f  max-amp~=%.3f",
                    k, v["alpha_mean"], v["beta_mean"], v["max_amp_mean"])

    # ---- BSCM fidelity kernel ----
    bscm_path = os.path.join(SAVE_DIR, "K_bscm_n200.npy")
    t0 = time.time()
    K_bscm = compute_bscm_fidelity_kernel(
        X_sub,
        n_qubits=N_QUBITS, reps=REPS, tau=TAU,
        coupling_threshold=THRESHOLD_BSCM, connectivity="all",
        kernel_save_path=bscm_path, desc="BSCM",
    )
    t_bscm = time.time() - t0

    # ---- SRQFM fidelity kernel ----
    srqfm_path = os.path.join(SAVE_DIR, "K_srqfm_fid_n200.npy")
    t0 = time.time()
    K_srqfm = compute_srqfm_fidelity_kernel(
        X_sub,
        n_qubits=N_QUBITS, reps=REPS,
        coupling_threshold=THRESHOLD_SRQFM, connectivity="all",
        kernel_save_path=srqfm_path, desc="SRQFM-fid",
    )
    t_srqfm = time.time() - t0

    # ---- Metrics ----
    sm_bscm = spectral_metrics(K_bscm)
    sm_srqfm = spectral_metrics(K_srqfm)
    kta_bscm = kta(K_bscm, y_sub)
    kta_srqfm = kta(K_srqfm, y_sub)

    logger.info("")
    logger.info("  Wall-clock:  BSCM=%.1fs   SRQFM-fid=%.1fs", t_bscm, t_srqfm)
    logger.info("")
    logger.info("  %-12s | %-10s | %-10s", "metric", "BSCM", "SRQFM-fid")
    logger.info("  " + "-" * 38)
    for key in ["off_diag_mean", "off_diag_var", "polarization",
                "eff_rank_shannon", "eff_rank_huang"]:
        logger.info("  %-12s | %10.4f | %10.4f",
                    key, sm_bscm[key], sm_srqfm[key])
    logger.info("  %-12s | %10.4f | %10.4f", "KTA", kta_bscm, kta_srqfm)

    # Decision rule.
    pass_var = sm_bscm["off_diag_var"] >= 0.8 * sm_srqfm["off_diag_var"]
    pass_kta = kta_bscm >= 0.8 * kta_srqfm
    pass_rank = sm_bscm["eff_rank_shannon"] >= 0.8 * sm_srqfm["eff_rank_shannon"]
    decision = pass_var and pass_kta and pass_rank

    logger.info("")
    logger.info("  Decision rule (BSCM >= 0.8 x SRQFM):")
    logger.info("    off_diag_var      : %s", "PASS" if pass_var else "FAIL")
    logger.info("    KTA               : %s", "PASS" if pass_kta else "FAIL")
    logger.info("    eff_rank_shannon  : %s", "PASS" if pass_rank else "FAIL")
    logger.info("  Overall decision   : %s", "PROCEED to Phase 3" if decision
                else "STOP and revisit hyperparameters")

    out = {
        "config": {
            "N": N_PHASE2, "n_qubits": N_QUBITS, "reps": REPS, "tau": TAU,
            "threshold_bscm": THRESHOLD_BSCM, "threshold_srqfm": THRESHOLD_SRQFM,
            "selected_features": sel_names,
        },
        "wall_clock_s": {"bscm": t_bscm, "srqfm_fid": t_srqfm},
        "spectral": {"bscm": sm_bscm, "srqfm_fid": sm_srqfm},
        "kta": {"bscm": kta_bscm, "srqfm_fid": kta_srqfm},
        "decision": {
            "off_diag_var_pass": bool(pass_var),
            "kta_pass": bool(pass_kta),
            "eff_rank_pass": bool(pass_rank),
            "overall_pass": bool(decision),
        },
        "top_bscm_pairs": [
            {"pair": k, **v} for k, v in top_pairs
        ],
    }
    out_path = os.path.join(SAVE_DIR, "sanity_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("  Saved metrics to %s", out_path)
    return 0 if decision else 2


if __name__ == "__main__":
    sys.exit(main())
