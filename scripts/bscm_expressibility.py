"""
Expressibility metric for BSCM and SRQFM circuits.

Sim, Johnson and Aspuru-Guzik (2019, https://arxiv.org/abs/1905.10876)
defined the expressibility of a parameterised circuit as the KL
divergence between the distribution of pairwise fidelities

    F(theta, theta')  =  | <psi(theta) | psi(theta')> |^2,
        theta, theta'  i.i.d.  uniform over the parameter range,

and the same fidelity distribution induced by Haar-random unitaries on
n qubits, which is

    p_Haar(F)  =  (2^n - 1) (1 - F)^(2^n - 2),    F in [0, 1].

Lower KL = closer to Haar = higher expressibility.  This is a standard
QML circuit-design diagnostic; it complements the off-diagonal-variance
and effective-rank metrics already in the project by separating
"the circuit can produce diverse states" from "the kernel is
well-conditioned for SVM."

Computed for:
    - BSCM (uniform, phi_only, psi_only) at the LOCKED tau
    - BSCM (uniform) at a sweep of tau values
    - SRQFM-fidelity (apples-to-apples)

Output: results/bscm/expressibility.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Callable, List, Tuple

import numpy as np
import pennylane as qml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.bscm_kernel import BELL_WEIGHT_PRESETS, apply_bscm_feature_map
from src.srqfm_fidelity_kernel import _apply_srqfm

N_QUBITS = config.N_QUBITS
N_SAMPLES = 2000   # 2*N_SAMPLES random parameter pairs.
N_BINS = 75
SAVE_DIR = os.path.join(config.RESULTS_DIR, "bscm")
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("expressibility")


def _haar_pdf_bins(bin_edges: np.ndarray, n_qubits: int) -> np.ndarray:
    """Return Haar-fidelity probabilities per histogram bin (continuous PDF
    integrated over each bin)."""
    N = 2 ** n_qubits
    # CDF of Haar fidelity:  F(z) = 1 - (1 - z)^(N - 1).
    cdf = lambda z: 1.0 - (1.0 - z) ** (N - 1)
    p = cdf(bin_edges[1:]) - cdf(bin_edges[:-1])
    return np.clip(p, 1e-12, None)


def _kl_divergence(p_emp: np.ndarray, p_haar: np.ndarray) -> float:
    p_emp = np.clip(p_emp, 1e-12, None)
    return float(np.sum(p_emp * np.log(p_emp / p_haar)))


def _build_fidelity_circuit(apply_forward: Callable, apply_adjoint: Callable):
    dev = config.get_device(N_QUBITS)

    @qml.qnode(dev, diff_method=None)
    def circuit(x1, x2):
        apply_forward(x1)
        apply_adjoint(x2)
        return qml.probs(wires=range(N_QUBITS))

    return circuit


def _bscm_forward(bell_weights, tau):
    return lambda x: apply_bscm_feature_map(
        x, n_qubits=N_QUBITS, reps=config.ZZ_REPS, tau=tau,
        coupling_threshold=1e-4, connectivity="all",
        bell_weights=bell_weights,
    )


def _bscm_adjoint(bell_weights, tau):
    from src.bscm_kernel import apply_bscm_feature_map_adjoint
    return lambda x: apply_bscm_feature_map_adjoint(
        x, n_qubits=N_QUBITS, reps=config.ZZ_REPS, tau=tau,
        coupling_threshold=1e-4, connectivity="all",
        bell_weights=bell_weights,
    )


def _srqfm_forward():
    return lambda x: _apply_srqfm(
        x, N_QUBITS, config.ZZ_REPS, 0.01, "all",
    )


def _srqfm_adjoint():
    from src.srqfm_fidelity_kernel import _apply_srqfm_adjoint
    return lambda x: _apply_srqfm_adjoint(
        x, N_QUBITS, config.ZZ_REPS, 0.01, "all",
    )


def expressibility(circuit_factory: Tuple[Callable, Callable],
                    name: str, n_samples: int = N_SAMPLES) -> dict:
    forward, adjoint = circuit_factory
    circuit = _build_fidelity_circuit(forward, adjoint)
    rng = np.random.default_rng(2026)
    fidelities: List[float] = []
    logger.info("  computing %d fidelities for %s", n_samples, name)
    for _ in range(n_samples):
        x1 = rng.uniform(0.0, np.pi, size=N_QUBITS)
        x2 = rng.uniform(0.0, np.pi, size=N_QUBITS)
        f = float(circuit(x1, x2)[0])
        fidelities.append(f)
    fidelities = np.array(fidelities)

    edges = np.linspace(0.0, 1.0, N_BINS + 1)
    counts, _ = np.histogram(fidelities, bins=edges)
    p_emp = counts / counts.sum()
    p_haar = _haar_pdf_bins(edges, N_QUBITS)
    kl = _kl_divergence(p_emp, p_haar)

    return {
        "name": name,
        "n_samples": n_samples,
        "kl_to_haar": kl,
        "fidelity_mean": float(fidelities.mean()),
        "fidelity_std": float(fidelities.std()),
        "fidelity_p05": float(np.percentile(fidelities, 5)),
        "fidelity_p95": float(np.percentile(fidelities, 95)),
    }


def main() -> int:
    logger.info("=" * 70)
    logger.info("  Expressibility (Sim-Johnson-Aspuru-Guzik) -- BSCM vs SRQFM")
    logger.info("=" * 70)

    tau_path = os.path.join(SAVE_DIR, "locked_tau.json")
    if os.path.exists(tau_path):
        with open(tau_path) as f:
            tau_locked = float(json.load(f)["locked"]["tau"])
    else:
        tau_locked = 0.5
        logger.warning(
            "locked_tau.json not found; defaulting to tau=0.5 for the prior ablation"
        )

    records = []

    # Bell-prior ablation at locked tau.
    for prior in ["uniform", "phi_only", "psi_only"]:
        w = BELL_WEIGHT_PRESETS[prior]
        rec = expressibility(
            (_bscm_forward(w, tau_locked), _bscm_adjoint(w, tau_locked)),
            f"BSCM-{prior}(tau={tau_locked})",
        )
        logger.info("  %s: KL=%.4f  fid_mean=%.4f", rec["name"],
                     rec["kl_to_haar"], rec["fidelity_mean"])
        records.append(rec)

    # tau sweep on uniform prior.
    w_unif = BELL_WEIGHT_PRESETS["uniform"]
    for tau in [0.5, 1.0, 2.0]:
        rec = expressibility(
            (_bscm_forward(w_unif, tau), _bscm_adjoint(w_unif, tau)),
            f"BSCM-uniform(tau={tau})",
        )
        logger.info("  %s: KL=%.4f  fid_mean=%.4f", rec["name"],
                     rec["kl_to_haar"], rec["fidelity_mean"])
        records.append(rec)

    # SRQFM-fidelity baseline.
    rec = expressibility(
        (_srqfm_forward(), _srqfm_adjoint()),
        "SRQFM-fid",
    )
    logger.info("  %s: KL=%.4f  fid_mean=%.4f", rec["name"],
                 rec["kl_to_haar"], rec["fidelity_mean"])
    records.append(rec)

    out = {
        "n_qubits": N_QUBITS, "reps": config.ZZ_REPS,
        "n_samples_per_circuit": N_SAMPLES,
        "n_bins": N_BINS,
        "records": records,
        "haar_reference": "Beta(1, 2^n - 1) on F in [0,1]; KL=0 means Haar-random.",
    }
    out_path = os.path.join(SAVE_DIR, "expressibility.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Saved: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
