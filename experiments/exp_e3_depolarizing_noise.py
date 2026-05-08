"""
Experiment E3 — Depolarising noise sweep.

PennyLane `default.mixed` simulator.  Sweep p ∈ {0, 0.002, 0.005, 0.01, 0.02, 0.05}.
FQK and AGPQK on physics-16 (top-8 Fisher).  N=300, 3 seeds (42,43,44).

For each p report:
    Macro-F1, off-diag variance, effective rank, ΔKTA.

Noise model: after each gate layer, apply DepolarizingChannel(p) per qubit.

Output: results/noise/{metrics.csv, fig.pdf}
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pennylane as qml
from tqdm import tqdm

import config
from src.utils import ensure_dir, setup_logging
from src.attention_kernel import (
    compute_attention_matrix, select_features_by_fisher,
    compute_per_qubit_gamma, get_attention_entanglement_pairs,
)
from src.kernel_target_alignment import compute_centered_kta
from experiments._e_common import (
    load_physics_16, spectral_metrics, svm_eval,
)

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "noise"))
logger = setup_logging("exp_e3", log_file=os.path.join(RES, "e3.log"))

N        = 300
N_QUBITS = config.N_QUBITS
REPS     = config.ZZ_REPS
SEEDS    = [42, 43, 44]
P_LIST   = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05]


def zz_layer(x, pairs, n_qubits):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for i in range(n_qubits):
        qml.RZ(x[i], wires=i)
    for (i, j) in pairs:
        zz = (np.pi - x[i]) * (np.pi - x[j])
        qml.CNOT(wires=[i, j]); qml.RZ(zz, wires=j); qml.CNOT(wires=[i, j])


def depolarize(p, n_qubits):
    for i in range(n_qubits):
        qml.DepolarizingChannel(p, wires=i)


# ---- FQK fidelity under noise ---------------------------------------------

def build_fqk_noisy(pairs, p, n_qubits=N_QUBITS, reps=REPS):
    dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(dev, diff_method=None)
    def circ(x1, x2):
        for _ in range(reps):
            zz_layer(x1, pairs, n_qubits); depolarize(p, n_qubits)
        for _ in range(reps):
            qml.adjoint(zz_layer)(x2, pairs, n_qubits); depolarize(p, n_qubits)
        return qml.probs(wires=range(n_qubits))
    return circ


def compute_fqk_noisy(X, pairs, p):
    circ = build_fqk_noisy(pairs, p)
    n = len(X); K = np.eye(n)
    for i in tqdm(range(n), desc=f"FQK p={p}", leave=False):
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = float(circ(X[i], X[j])[0])
    return K


# ---- AGPQK (projected) under noise ----------------------------------------

def build_agpqk_bloch_noisy(pairs, p, n_qubits=N_QUBITS, reps=REPS):
    dev = qml.device("default.mixed", wires=n_qubits)

    def apply(x):
        for _ in range(reps):
            zz_layer(x, pairs, n_qubits); depolarize(p, n_qubits)

    @qml.qnode(dev, diff_method=None)
    def mx(x): apply(x); return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def my(x): apply(x); return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def mz(x): apply(x); return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return mx, my, mz


def compute_agpqk_noisy(X_enc, pairs, p, outer_gamma, n_qubits=N_QUBITS):
    mx, my, mz = build_agpqk_bloch_noisy(pairs, p, n_qubits)
    n = len(X_enc); bloch = np.zeros((n, 3 * n_qubits))
    for i in tqdm(range(n), desc=f"AGPQK bloch p={p}", leave=False):
        ex = np.array(mx(X_enc[i])); ey = np.array(my(X_enc[i]))
        ez = np.array(mz(X_enc[i]))
        for q in range(n_qubits):
            bloch[i, 3 * q]     = ex[q]
            bloch[i, 3 * q + 1] = ey[q]
            bloch[i, 3 * q + 2] = ez[q]
    K = np.zeros((n, n))
    for i in range(n):
        diff = bloch[i] - bloch
        sq = np.sum(diff.reshape(n, n_qubits, 3) ** 2, axis=2)
        frob = 0.5 * np.sum(sq, axis=1)
        K[i, :] = np.exp(-outer_gamma * frob)
    return K


def run():
    X_norm, X_raw, y_full = load_physics_16()

    # Subsample N=300 stratified, fixed across runs (seed=42)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=N, random_state=42)
    (idx, _), = sss.split(np.zeros(len(y_full)), y_full)
    X_n, X_r, y = X_norm[idx], X_raw[idx], y_full[idx]

    # Attention / Fisher selection on the subsample
    A = compute_attention_matrix(X_r)
    selected, fisher = select_features_by_fisher(X_r, y, n_select=N_QUBITS,
                                                 min_sar=4, min_opt=4)
    gamma_perq = compute_per_qubit_gamma(fisher, selected, gamma_base=0.5)
    pairs_agpqk = get_attention_entanglement_pairs(
        A, selected, top_k=4, require_cross_modal=True, max_appearances=1)
    pairs_fqk = [(i, i + 1) for i in range(N_QUBITS - 1)]

    X_sel_fqk   = X_n[:, selected]
    X_sel_agpqk = X_r[:, selected] * gamma_perq[np.newaxis, :]

    rows = []
    for p in P_LIST:
        K_fqk   = compute_fqk_noisy(X_sel_fqk,   pairs_fqk,   p)
        K_agpqk = compute_agpqk_noisy(X_sel_agpqk, pairs_agpqk, p,
                                      outer_gamma=config.PQK_GAMMA)
        np.save(os.path.join(RES, f"K_FQK_p{p}.npy"),   K_fqk)
        np.save(os.path.join(RES, f"K_AGPQK_p{p}.npy"), K_agpqk)

        for name, K in [("FQK", K_fqk), ("AGPQK", K_agpqk)]:
            spm = spectral_metrics(K)
            kta = float(compute_centered_kta(K, y, class_weighted=True))
            for seed in SEEDS:
                ev = svm_eval(K, y, seed)
                rows.append(dict(kernel=name, p=p, seed=seed,
                                 macro_f1=ev["macro_f1"],
                                 kta=kta, **spm))
                logger.info(f"{name} p={p} seed={seed}  "
                            f"F1={ev['macro_f1']:.3f}  eff={spm['eff_rank_shannon']:.1f}")

    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "metrics.csv"),
                                        index=False)
    _plot(df)


def _plot(df):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    metrics = [("macro_f1",        "Macro-F1"),
               ("off_diag_var",    "Off-diag variance"),
               ("eff_rank_shannon","Eff rank"),
               ("polarization",    "Polarization"),
               ("kta",             "ΔKTA"),
               ("off_diag_mean",   "Off-diag mean")]
    for ax, (k, lab) in zip(axes.flat, metrics):
        for kernel, c in [("FQK", "#d62728"), ("AGPQK", "#9467bd")]:
            sub = df[df.kernel == kernel]
            g = sub.groupby("p")[k].agg(["mean", "std"]).reset_index()
            ax.errorbar(g["p"], g["mean"], yerr=g["std"], marker="o",
                        capsize=4, label=kernel, color=c)
        ax.set_xlabel("p (depolarizing)"); ax.set_ylabel(lab)
        ax.grid(alpha=0.3); ax.legend()
    fig.suptitle("E3 — depolarising-noise sensitivity (N=300)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
