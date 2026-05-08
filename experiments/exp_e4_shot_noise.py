"""
Experiment E4 — Shot-noise convergence for FQK.

Physics-16 features (top-8 Fisher), N=300, seeds {42,43,44}.
Shots S ∈ {32, 64, 128, 256, 512, 1024, 4096, ∞ (statevector)}.

For each S report Macro-F1, ΔKTA, off-diag variance; build a
Macro-F1 vs log2(S) convergence curve and a QPU-time estimate.

Output: results/shots/{metrics.csv, fig.pdf}
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pennylane as qml
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit

import config
from src.utils import ensure_dir, setup_logging
from src.attention_kernel import select_features_by_fisher
from src.kernel_target_alignment import compute_centered_kta
from experiments._e_common import (
    load_physics_16, spectral_metrics, svm_eval,
)

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "shots"))
logger = setup_logging("exp_e4", log_file=os.path.join(RES, "e4.log"))

N        = 300
N_QUBITS = config.N_QUBITS
REPS     = config.ZZ_REPS
SEEDS    = [42, 43, 44]
SHOTS    = [32, 64, 128, 256, 512, 1024, 4096, None]  # None = statevector


def zz_feature(x, n_qubits, reps):
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        for i in range(n_qubits - 1):
            zz = (np.pi - x[i]) * (np.pi - x[i + 1])
            qml.CNOT(wires=[i, i + 1]); qml.RZ(zz, wires=i + 1); qml.CNOT(wires=[i, i + 1])


def build_fqk(shots, n_qubits=N_QUBITS, reps=REPS):
    dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

    @qml.qnode(dev, diff_method=None)
    def circ(x1, x2):
        zz_feature(x1, n_qubits, reps)
        qml.adjoint(zz_feature)(x2, n_qubits, reps)
        return qml.probs(wires=range(n_qubits))
    return circ


def compute_K(X, shots):
    circ = build_fqk(shots)
    n = len(X); K = np.eye(n)
    for i in tqdm(range(n), desc=f"FQK S={shots}", leave=False):
        for j in range(i + 1, n):
            K[i, j] = K[j, i] = float(circ(X[i], X[j])[0])
    return K


def run():
    X_norm, X_raw, y_full = load_physics_16()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=N, random_state=42)
    (idx, _), = sss.split(np.zeros(len(y_full)), y_full)
    X_r, y = X_raw[idx], y_full[idx]
    selected, _ = select_features_by_fisher(X_r, y, n_select=N_QUBITS,
                                            min_sar=4, min_opt=4)
    X_sel = X_norm[idx][:, selected]

    rows = []
    for S in SHOTS:
        K = compute_K(X_sel, S)
        lbl = "inf" if S is None else str(S)
        np.save(os.path.join(RES, f"K_S{lbl}.npy"), K)
        spm = spectral_metrics(K)
        kta = float(compute_centered_kta(K, y, class_weighted=True))
        for seed in SEEDS:
            ev = svm_eval(K, y, seed)
            rows.append(dict(shots=(0 if S is None else S),
                             statevector=(S is None),
                             seed=seed, macro_f1=ev["macro_f1"],
                             kta=kta, **spm))
        logger.info(f"S={lbl}  eff={spm['eff_rank_shannon']:.1f}  "
                    f"off_var={spm['off_diag_var']:.3e}")

    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "metrics.csv"),
                                        index=False)
    _plot(df)


def _plot(df):
    import matplotlib.pyplot as plt
    finite = df[~df.statevector]
    sv     = df[df.statevector]
    g = finite.groupby("shots")["macro_f1"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(g["shots"], g["mean"], yerr=g["std"], marker="o",
                capsize=4, color="#1f77b4", label="finite S")
    if len(sv):
        ax.axhline(sv["macro_f1"].mean(), ls="--", color="black",
                   label="statevector (S→∞)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Shots per circuit S"); ax.set_ylabel("Macro-F1")
    ax.set_title("E4 — shot-noise convergence (FQK, N=300)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
