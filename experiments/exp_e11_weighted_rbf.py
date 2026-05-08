"""
Experiment E11 — Fisher-weighted RBF classical surrogate.

Answers Schnabel & Roth (arXiv:2503.05602): "bandwidth-tuned QKs collapse to
RBF". Our defence is that AGPQK uses *per-qubit* γ_i modulated by Fisher
importance, not a single scalar. The falsification test: can a classical RBF
with **per-feature** γ_i = Fisher_i · γ_base match AGPQK?

If RBF-weighted matches AGPQK → quantum component adds nothing.
If RBF-weighted loses → AGPQK's structure is genuinely quantum.

Also report g(AGPQK ‖ weighted-RBF) via Huang's geometric difference.

Output: results/weighted_rbf/{metrics.csv, g_table.csv, fig.pdf}
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import f1_score

import config
from src.utils import ensure_dir, setup_logging
from src.attention_kernel import select_features_by_fisher, compute_per_qubit_gamma
from src.geometric_difference import compute_geometric_difference
from src.kernel_target_alignment import compute_centered_kta
from experiments._e_common import load_physics_16, spectral_metrics, svm_eval

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "weighted_rbf"))
logger = setup_logging("exp_e11", log_file=os.path.join(RES, "e11.log"))

AGPQK_FULL = os.path.join(config.RESULTS_DIR, "physics_cv",
                          "full_kernels", "K_agpqk_full.npy")
N_QUBITS = config.N_QUBITS
SEEDS    = config.SEED_LIST


def weighted_rbf_kernel(X_sel, gamma_per_feat):
    """
    K(x, x') = exp(-Σ_i γ_i (x_i - x'_i)^2)  — per-feature bandwidths.
    Equivalent to standard RBF on X * sqrt(γ) with gamma=1.
    """
    Xw = X_sel * np.sqrt(gamma_per_feat)[np.newaxis, :]
    sq = np.sum(Xw ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (Xw @ Xw.T)
    D2 = np.clip(D2, 0.0, None)
    return np.exp(-D2)


def uniform_rbf_kernel(X_sel, gamma_scalar):
    sq = np.sum(X_sel ** 2, axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X_sel @ X_sel.T)
    D2 = np.clip(D2, 0.0, None)
    return np.exp(-gamma_scalar * D2)


def run():
    X_norm, X_raw, y = load_physics_16()
    K_agpqk = np.load(AGPQK_FULL)

    selected, fisher = select_features_by_fisher(X_raw, y, n_select=N_QUBITS,
                                                 min_sar=4, min_opt=4)
    # Use same per-qubit γ mapping that AGPQK uses — but applied classically.
    gamma_perq = compute_per_qubit_gamma(fisher, selected, gamma_base=0.5)
    X_sel = X_norm[:, selected]

    # Scalar-RBF baseline (Schnabel & Roth assumption: single global γ).
    # Use median heuristic: γ = 1 / (2 * median pairwise sq-distance).
    D2 = np.sum((X_sel[:, None, :] - X_sel[None, :, :]) ** 2, axis=2)
    gamma_scalar = 1.0 / (2.0 * np.median(D2[D2 > 0]))

    K_wrbf = weighted_rbf_kernel(X_sel, gamma_perq)
    K_urbf = uniform_rbf_kernel(X_sel, gamma_scalar)

    np.save(os.path.join(RES, "K_weighted_rbf.npy"), K_wrbf)
    np.save(os.path.join(RES, "K_uniform_rbf.npy"), K_urbf)

    rows = []
    for name, K in [("AGPQK", K_agpqk),
                    ("Weighted-RBF", K_wrbf),
                    ("Uniform-RBF", K_urbf)]:
        spm = spectral_metrics(K)
        kta = float(compute_centered_kta(K, y, class_weighted=True))
        for seed in SEEDS:
            ev = svm_eval(K, y, seed)
            rows.append(dict(kernel=name, seed=seed,
                             macro_f1=ev["macro_f1"],
                             accuracy=ev["accuracy"],
                             kta=kta, **spm))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "metrics.csv"), index=False)

    # Geometric difference g(K_q ‖ K_c)
    g_rows = []
    for q_name, K_q in [("AGPQK", K_agpqk)]:
        for c_name, K_c in [("Weighted-RBF", K_wrbf),
                            ("Uniform-RBF", K_urbf)]:
            g, diag = compute_geometric_difference(K_q, K_c)
            g_rows.append(dict(quantum=q_name, classical=c_name, g=g,
                               lambda_used=diag["lambda_used"],
                               K_q_rank=diag["K_q_rank"],
                               K_c_rank=diag["K_c_rank"]))
            logger.info(f"g({q_name} ‖ {c_name}) = {g:.4f}")
    pd.DataFrame(g_rows).to_csv(os.path.join(RES, "g_table.csv"), index=False)

    logger.info("Summary Macro-F1 (mean ± std across seeds):")
    for k in ["AGPQK", "Weighted-RBF", "Uniform-RBF"]:
        sub = df[df.kernel == k]["macro_f1"]
        logger.info(f"  {k:15s}  {sub.mean():.4f} ± {sub.std():.4f}")

    _plot(df, g_rows)


def _plot(df, g_rows):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"AGPQK": "#9467bd", "Weighted-RBF": "#2ca02c",
              "Uniform-RBF": "#ff7f0e"}

    ax = axes[0]
    names, means, stds = [], [], []
    for k in ["AGPQK", "Weighted-RBF", "Uniform-RBF"]:
        sub = df[df.kernel == k]["macro_f1"]
        names.append(k); means.append(sub.mean()); stds.append(sub.std())
    ax.bar(names, means, yerr=stds, capsize=6,
           color=[colors[n] for n in names])
    ax.set_ylabel("Macro-F1"); ax.grid(alpha=0.3, axis="y")
    ax.set_title("AGPQK vs Fisher-weighted RBF")

    ax = axes[1]
    labels = [f"{r['quantum']}\n‖ {r['classical']}" for r in g_rows]
    gs = [r["g"] for r in g_rows]
    ax.bar(labels, gs, color="#1f77b4")
    ax.axhline(1.0, ls="--", color="black", label="g=1 (geometric identity)")
    ax.set_ylabel("Geometric difference g")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    ax.set_title("Huang geometric difference")

    fig.suptitle("E11 — Bandwidth-collapse falsification (Schnabel & Roth 2503.05602)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
