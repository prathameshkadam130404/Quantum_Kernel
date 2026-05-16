"""
E48: Density-matrix noise sweep for BSCM-uniform-PQK vs Standard-ZZ-PQK.

Tests whether the architectural gap observed at headline E38 scale
(noiseless statevector simulation) survives under realistic depolarising
gate noise.  Run at n=8, N=500 on So2Sat physics-Fisher-8 because
density-matrix simulation is computationally infeasible at n=16 (~230 h
per kernel on default.mixed).

Protocol
--------
Dataset      So2Sat physics-Fisher-8 (top-8 Fisher rank from physics-16)
Pool         N=500 stratified sub-sample
Splits       5 stratified shuffle splits per seed (test_size=0.3)
Seeds        {42, 43, 44}
Circuit      n=8, reps=2, BSCM tau=0.25 (locked from holdout)
Topology     all-to-all
Device       default.mixed (density-matrix)
Noise        DepolarizingChannel(p) on every qubit after every
             (Hadamard + RZ + entangling) layer
p sweep      {0.0, 0.001, 0.005, 0.01}
C grid       {0.1, 1, 10, 100, 1000} via inner 3-fold CV (f1_macro)
PQK gamma    sklearn 'scale' heuristic on training-fold Bloch variance
             (identical for both kernels; no gamma tuning so the noise
             effect is isolated)

Output
------
results/e48_noise_dm/summary.json     -- per (kernel, p) means, gaps,
                                         and Wilcoxon signed-rank p
results/e48_noise_dm/per_split.json   -- per (kernel, p, seed) splits
results/e48_noise_dm/exp_e48.log

Wall-clock cost on a single CPU: ~1.5-2 hours total
(2 kernels x 4 p values x 3 seeds x N=500 density-matrix extraction).

Usage
-----
    python experiments/exp_e48_noise_density_matrix.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pennylane as qml
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._e_common import load_physics_16
from src.attention_kernel import select_features_by_fisher
from src.bscm_kernel import bscm_pauli_coefficients, BELL_WEIGHT_PRESETS

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
N_QUBITS = 8
REPS = 2
TAU = 0.25
N_POOL = 500
TEST_FRACTION = 0.3
N_INNER_SPLITS = 5
SEEDS = [42, 43, 44]
NOISE_LEVELS = [0.0, 0.001, 0.005, 0.01]
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
CV_FOLDS = 3
COUPLING_THR = 1e-5

OUT_DIR = os.path.join(config.RESULTS_DIR, "e48_noise_dm")
os.makedirs(OUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUT_DIR, "exp_e48.log")
SUMMARY_PATH = os.path.join(OUT_DIR, "summary.json")
PER_SPLIT_PATH = os.path.join(OUT_DIR, "per_split.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
log = logging.getLogger("exp_e48")


# --------------------------------------------------------------------------- #
# Noisy Bloch extractors (default.mixed, depolarising noise after each layer)
# --------------------------------------------------------------------------- #
def _all_to_all_pairs(n_qubits: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]


def _apply_depolarise(p: float, n_qubits: int) -> None:
    if p <= 0:
        return
    for q in range(n_qubits):
        qml.DepolarizingChannel(p, wires=q)


def _bscm_layer(
    x: np.ndarray, pairs: List[Tuple[int, int]], n_qubits: int,
    bell_weights: tuple, tau: float, coupling_threshold: float,
) -> None:
    for q in range(n_qubits):
        qml.Hadamard(wires=q)
    for q in range(n_qubits):
        qml.RZ(x[q], wires=q)
    for qi, qj in pairs:
        a_xx, a_yy, a_zz = bscm_pauli_coefficients(x[qi], x[qj], bell_weights)
        if abs(a_xx) >= coupling_threshold:
            qml.IsingXX(2.0 * tau * a_xx, wires=[qi, qj])
        if abs(a_yy) >= coupling_threshold:
            qml.IsingYY(2.0 * tau * a_yy, wires=[qi, qj])
        if abs(a_zz) >= coupling_threshold:
            qml.IsingZZ(2.0 * tau * a_zz, wires=[qi, qj])


def _standard_zz_layer(
    x: np.ndarray, pairs: List[Tuple[int, int]], n_qubits: int,
) -> None:
    for q in range(n_qubits):
        qml.Hadamard(wires=q)
    for q in range(n_qubits):
        qml.RZ(x[q], wires=q)
    for qi, qj in pairs:
        # Canonical Havlicek ZZFeatureMap entangler: CNOT - RZ((pi-xi)(pi-xj)) - CNOT.
        qml.CNOT(wires=[qi, qj])
        qml.RZ((np.pi - x[qi]) * (np.pi - x[qj]), wires=qj)
        qml.CNOT(wires=[qi, qj])


def _build_noisy_bscm_bloch(
    n_qubits: int, reps: int, p: float, bell_weights: tuple,
):
    pairs = _all_to_all_pairs(n_qubits)
    dev = qml.device("default.mixed", wires=n_qubits)

    def _apply(x):
        for _ in range(reps):
            _bscm_layer(x, pairs, n_qubits, bell_weights, TAU, COUPLING_THR)
            _apply_depolarise(p, n_qubits)

    @qml.qnode(dev, diff_method=None)
    def mx(x):
        _apply(x); return [qml.expval(qml.PauliX(q)) for q in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def my(x):
        _apply(x); return [qml.expval(qml.PauliY(q)) for q in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def mz(x):
        _apply(x); return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
    return mx, my, mz


def _build_noisy_standard_bloch(n_qubits: int, reps: int, p: float):
    pairs = _all_to_all_pairs(n_qubits)
    dev = qml.device("default.mixed", wires=n_qubits)

    def _apply(x):
        for _ in range(reps):
            _standard_zz_layer(x, pairs, n_qubits)
            _apply_depolarise(p, n_qubits)

    @qml.qnode(dev, diff_method=None)
    def mx(x):
        _apply(x); return [qml.expval(qml.PauliX(q)) for q in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def my(x):
        _apply(x); return [qml.expval(qml.PauliY(q)) for q in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def mz(x):
        _apply(x); return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]
    return mx, my, mz


def _extract_bloch(X_enc: np.ndarray, mx, my, mz) -> np.ndarray:
    N = len(X_enc)
    out = np.zeros((N, 3 * N_QUBITS), dtype=np.float64)
    for i in tqdm(range(N), desc="bloch", leave=False):
        out[i, 0 * N_QUBITS:1 * N_QUBITS] = mx(X_enc[i])
        out[i, 1 * N_QUBITS:2 * N_QUBITS] = my(X_enc[i])
        out[i, 2 * N_QUBITS:3 * N_QUBITS] = mz(X_enc[i])
    return out


# --------------------------------------------------------------------------- #
# PQK Gram + SVM evaluation
# --------------------------------------------------------------------------- #
def _build_pqk_gram(
    bloch_tr: np.ndarray, bloch_te: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Train-train and test-train Gram with training-fold dynamic gamma."""
    b_tr = bloch_tr.astype(np.float64)
    b_te = bloch_te.astype(np.float64)
    var = float(np.var(b_tr))
    d = b_tr.shape[1]
    gamma = 1.0 / (d * var) if var > 0 else 1.0

    sq_norms_tr = np.sum(b_tr ** 2, axis=1)
    sq_norms_te = np.sum(b_te ** 2, axis=1)
    sq_dists_tr = np.clip(
        sq_norms_tr[:, None] + sq_norms_tr[None, :] - 2.0 * (b_tr @ b_tr.T),
        0, None,
    )
    sq_dists_te = np.clip(
        sq_norms_te[:, None] + sq_norms_tr[None, :] - 2.0 * (b_te @ b_tr.T),
        0, None,
    )
    K_tr = np.exp(-gamma * 0.5 * sq_dists_tr)
    np.fill_diagonal(K_tr, 1.0)
    K_tr = (K_tr + K_tr.T) / 2.0
    K_te = np.exp(-gamma * 0.5 * sq_dists_te)
    return K_tr, K_te, gamma


def _svm_eval_precomputed(K_tr: np.ndarray, K_te: np.ndarray,
                           y_tr: np.ndarray, y_te: np.ndarray) -> float:
    clf = GridSearchCV(
        SVC(kernel="precomputed", class_weight="balanced"),
        {"C": C_GRID}, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1,
    )
    clf.fit(K_tr, y_tr)
    return float(f1_score(y_te, clf.predict(K_te), average="macro"))


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _prepare_pool(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (X_enc, y) stratified subsample of size N_POOL, scaled to [0, pi].

    The Fisher feature ranking is computed on the full physics-16 ``X_raw``
    so it is independent of the per-seed subsample.  The standardisation and
    MinMax-to-[0, pi] scaling are fit on the post-Fisher 8-feature space.
    """
    X_norm, X_raw, y = load_physics_16()
    sel_idx, _ = select_features_by_fisher(X_raw, y)
    sel_idx = sel_idx[:N_QUBITS]
    X_sel_raw = X_raw[:, sel_idx]

    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=N_POOL, test_size=None, random_state=seed,
    )
    (idx, _), = sss.split(X_sel_raw, y)
    idx = np.sort(idx)
    X_sub_raw = X_sel_raw[idx]
    y_sub = y[idx]

    # Standardise and MinMax to [0, pi].  The pool itself acts as the fit set
    # here (consistent with E26 / E32 convention -- those experiments do not
    # split the angle scaler).  Per-fold scaling is applied later only for
    # classical baselines if any.
    ssx = StandardScaler().fit(X_sub_raw)
    X_std = ssx.transform(X_sub_raw)
    mmx = MinMaxScaler(feature_range=(0.0, np.pi)).fit(X_std)
    X_enc = mmx.transform(X_std).clip(0.0, np.pi).astype(np.float32)
    return X_enc, y_sub


def _eval_kernel_at_noise(
    kernel_name: str, X_enc: np.ndarray, y: np.ndarray, p: float, seed: int,
) -> Dict[str, object]:
    """Extract noisy Bloch and evaluate via 5 stratified shuffle splits."""
    if kernel_name == "BSCM-uniform":
        mx, my, mz = _build_noisy_bscm_bloch(
            N_QUBITS, REPS, p, BELL_WEIGHT_PRESETS["uniform"],
        )
    elif kernel_name == "Standard-ZZ":
        mx, my, mz = _build_noisy_standard_bloch(N_QUBITS, REPS, p)
    else:
        raise ValueError(kernel_name)

    t0 = time.time()
    bloch = _extract_bloch(X_enc, mx, my, mz)
    dt = time.time() - t0
    log.info("    %s p=%.3f seed=%d  Bloch(%d,%d) in %.1fs",
             kernel_name, p, seed, bloch.shape[0], bloch.shape[1], dt)

    sss = StratifiedShuffleSplit(
        n_splits=N_INNER_SPLITS, test_size=TEST_FRACTION, random_state=seed,
    )
    f1s, gammas = [], []
    for tr, te in sss.split(np.zeros(len(y)), y):
        K_tr, K_te, gamma = _build_pqk_gram(bloch[tr], bloch[te])
        f1 = _svm_eval_precomputed(K_tr, K_te, y[tr], y[te])
        f1s.append(f1)
        gammas.append(gamma)
    log.info("    %s p=%.3f seed=%d  F1=%.4f +/- %.4f  (gamma_mean=%.4f)",
             kernel_name, p, seed, float(np.mean(f1s)), float(np.std(f1s)),
             float(np.mean(gammas)))
    return {
        "f1_per_split": [float(x) for x in f1s],
        "gamma_per_split": [float(g) for g in gammas],
        "bloch_extraction_seconds": dt,
    }


def main() -> None:
    log.info("=" * 72)
    log.info("  E48 density-matrix noise: BSCM-uniform vs Standard-ZZ-PQK")
    log.info("  n_qubits=%d, reps=%d, N_pool=%d, tau=%.2f",
             N_QUBITS, REPS, N_POOL, TAU)
    log.info("  noise levels: %s, seeds: %s", NOISE_LEVELS, SEEDS)
    log.info("=" * 72)

    per_split: Dict[str, dict] = {}  # nested seed/kernel/p
    summary: Dict[str, dict] = {}

    for seed in SEEDS:
        log.info("[seed=%d] preparing pool", seed)
        X_enc, y = _prepare_pool(seed)
        log.info("  pool: N=%d, class counts=%s",
                 len(y), dict(zip(*np.unique(y, return_counts=True))))
        per_split[f"seed_{seed}"] = {}

        for kernel in ("BSCM-uniform", "Standard-ZZ"):
            per_split[f"seed_{seed}"][kernel] = {}
            for p in NOISE_LEVELS:
                rec = _eval_kernel_at_noise(kernel, X_enc, y, p, seed)
                per_split[f"seed_{seed}"][kernel][f"p_{p}"] = rec

    # Aggregate per (kernel, p) across seeds.
    for kernel in ("BSCM-uniform", "Standard-ZZ"):
        summary[kernel] = {}
        for p in NOISE_LEVELS:
            f1_all = []
            for seed in SEEDS:
                f1_all.extend(
                    per_split[f"seed_{seed}"][kernel][f"p_{p}"]["f1_per_split"]
                )
            summary[kernel][f"p_{p}"] = {
                "f1_mean": float(np.mean(f1_all)),
                "f1_std": float(np.std(f1_all)),
                "n_observations": len(f1_all),
            }

    # Architectural gap per noise level.
    gaps = {}
    for p in NOISE_LEVELS:
        bscm_f1 = []
        zz_f1 = []
        for seed in SEEDS:
            bscm_f1.extend(per_split[f"seed_{seed}"]["BSCM-uniform"][f"p_{p}"]["f1_per_split"])
            zz_f1.extend(per_split[f"seed_{seed}"]["Standard-ZZ"][f"p_{p}"]["f1_per_split"])
        bscm_f1 = np.array(bscm_f1)
        zz_f1 = np.array(zz_f1)
        # Paired Wilcoxon over (seed x split) -> we have 3 * N_INNER_SPLITS pairs.
        try:
            _, p_val = wilcoxon(bscm_f1 - zz_f1, zero_method="wilcox",
                                alternative="two-sided")
            p_val = float(p_val)
        except ValueError:
            p_val = 1.0
        gaps[f"p_{p}"] = {
            "bscm_f1_mean": float(bscm_f1.mean()),
            "zz_f1_mean": float(zz_f1.mean()),
            "gap": float(bscm_f1.mean() - zz_f1.mean()),
            "wilcoxon_p": p_val,
            "n_pairs": int(len(bscm_f1)),
        }

    out = {
        "protocol": {
            "n_qubits": N_QUBITS,
            "reps": REPS,
            "tau": TAU,
            "n_pool": N_POOL,
            "test_fraction": TEST_FRACTION,
            "n_inner_splits": N_INNER_SPLITS,
            "seeds": SEEDS,
            "noise_levels": NOISE_LEVELS,
            "device": "default.mixed",
            "C_grid": C_GRID,
            "inner_cv_folds": CV_FOLDS,
            "topology": "all-to-all",
            "feature_set": "physics_Fisher_8 from physics-16",
        },
        "summary": summary,
        "architectural_gaps": gaps,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(out, f, indent=2)
    with open(PER_SPLIT_PATH, "w") as f:
        json.dump(per_split, f, indent=2)
    log.info("Wrote %s and %s", SUMMARY_PATH, PER_SPLIT_PATH)

    log.info("")
    log.info("ARCHITECTURAL GAP (BSCM-uniform minus Standard-ZZ) BY NOISE:")
    log.info("%-10s %12s %12s %12s %12s",
             "p", "BSCM_f1", "ZZ_f1", "gap", "wilcoxon_p")
    for p in NOISE_LEVELS:
        g = gaps[f"p_{p}"]
        log.info("%-10s %12.4f %12.4f %12.4f %12.4g",
                 f"{p:.3f}", g["bscm_f1_mean"], g["zz_f1_mean"],
                 g["gap"], g["wilcoxon_p"])


if __name__ == "__main__":
    main()
