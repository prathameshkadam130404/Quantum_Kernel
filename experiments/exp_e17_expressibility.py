"""
Experiment E17 — Expressibility + entangling capability.

Sim et al., Advanced Quantum Technologies 2(12), 1900070 (2019).

For FQK / PQK / AGPQK ansätze on physics-16 features (top-8 Fisher), n=8:
    - Expressibility KL: KL(P_fidelity || P_Haar), 5000 sampled pairs.
    - Entangling capability E: 1 − ⟨Tr(ρ_k^2)⟩ averaged over qubits and inputs
      (Meyer-Wallach Q = 2(1 − ⟨Tr(ρ_k^2)⟩)); we report the mean MW entanglement.

Output: results/expressibility/{metrics.csv, fig.pdf}
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
from src.expressibility import compute_expressibility
from experiments._e_common import load_physics_16

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "expressibility"))
logger = setup_logging("exp_e17", log_file=os.path.join(RES, "e17.log"))

N_QUBITS = config.N_QUBITS
REPS     = config.ZZ_REPS
N_PAIRS  = 1000
N_ENT    = 400  # samples for entangling capability


def zz_apply(x, pairs, n_qubits, reps):
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)
        for (i, j) in pairs:
            if i < n_qubits and j < n_qubits:
                zz = (np.pi - x[i]) * (np.pi - x[j])
                qml.CNOT(wires=[i, j]); qml.RZ(zz, wires=j)
                qml.CNOT(wires=[i, j])


def make_fidelity_fn(pairs, n_qubits=N_QUBITS, reps=REPS):
    dev = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def circ(x1, x2):
        zz_apply(x1, pairs, n_qubits, reps)
        qml.adjoint(zz_apply)(x2, pairs, n_qubits, reps)
        return qml.probs(wires=range(n_qubits))

    def fidelity(x1, x2):
        return float(circ(x1, x2)[0])
    return fidelity


def meyer_wallach(X_enc, pairs, n_qubits=N_QUBITS, reps=REPS):
    """
    Q(ψ) = 2(1 − (1/n) Σ_k Tr(ρ_k²)).
    For pure states under single-qubit reductions:
        ⟨σ_x⟩² + ⟨σ_y⟩² + ⟨σ_z⟩² = 2 Tr(ρ_k²) − 1
        → Tr(ρ_k²) = 0.5 (1 + r_k²) where r_k is the Bloch-vector length.
    """
    dev = config.get_device(n_qubits)

    def apply(x):
        zz_apply(x, pairs, n_qubits, reps)

    @qml.qnode(dev, diff_method=None)
    def mx(x):
        apply(x); return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def my(x):
        apply(x); return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def mz(x):
        apply(x); return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    Qs = np.zeros(len(X_enc))
    for i in tqdm(range(len(X_enc)), desc="MW", leave=False):
        ex = np.array(mx(X_enc[i])); ey = np.array(my(X_enc[i]))
        ez = np.array(mz(X_enc[i]))
        r2 = ex ** 2 + ey ** 2 + ez ** 2
        tr_rho_sq = 0.5 * (1.0 + r2)
        Qs[i] = 2.0 * (1.0 - tr_rho_sq.mean())
    return float(Qs.mean()), float(Qs.std())


def run():
    X_norm, X_raw, y = load_physics_16()
    selected, fisher = select_features_by_fisher(X_raw, y,
                                                 n_select=N_QUBITS,
                                                 min_sar=4, min_opt=4)
    A = compute_attention_matrix(X_raw)
    attn_pairs = get_attention_entanglement_pairs(
        A, selected, top_k=4, require_cross_modal=True, max_appearances=1)
    gamma = compute_per_qubit_gamma(fisher, selected, gamma_base=0.5)

    linear = [(i, i + 1) for i in range(N_QUBITS - 1)]

    configs = {
        "FQK":   (X_norm[:, selected],                         linear),
        "PQK":   (X_norm[:, selected],                         linear),
        "AGPQK": (X_raw[:, selected] * gamma[np.newaxis, :],   attn_pairs),
    }

    rows = []
    for name, (X_enc, pairs) in configs.items():
        logger.info(f"--- {name} ---")
        fid_fn = make_fidelity_fn(pairs)
        Xe = X_enc[:max(N_PAIRS, N_ENT)]
        expr = compute_expressibility(fid_fn, N_QUBITS, Xe,
                                      n_samples=N_PAIRS, n_bins=75, seed=42)
        mw_mean, mw_std = meyer_wallach(Xe[:N_ENT], pairs)
        rows.append(dict(ansatz=name,
                         kl_divergence=expr["kl_divergence"],
                         mean_fidelity=expr["mean_fidelity"],
                         std_fidelity=expr["std_fidelity"],
                         meyer_wallach_mean=mw_mean,
                         meyer_wallach_std=mw_std))
        logger.info(f"{name}  KL={expr['kl_divergence']:.5f}  "
                    f"MW={mw_mean:.4f}±{mw_std:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "metrics.csv"), index=False)
    _plot(df)


def _plot(df):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"FQK": "#d62728", "PQK": "#1f77b4", "AGPQK": "#9467bd"}

    ax = axes[0]
    ax.bar(df["ansatz"], df["kl_divergence"],
           color=[colors[a] for a in df["ansatz"]])
    ax.set_ylabel("KL(P_data || P_Haar)  — lower = more expressive")
    ax.grid(alpha=0.3, axis="y"); ax.set_title("Expressibility (Sim 2019)")

    ax = axes[1]
    ax.bar(df["ansatz"], df["meyer_wallach_mean"],
           yerr=df["meyer_wallach_std"], capsize=5,
           color=[colors[a] for a in df["ansatz"]])
    ax.set_ylabel("Meyer-Wallach entanglement Q")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Entangling capability")

    fig.suptitle("E17 — Expressibility + entangling capability")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
