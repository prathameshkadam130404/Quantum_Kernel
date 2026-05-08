"""
Experiment E5 — Qubit scaling (concentration vs n).

Verify Thanasilp et al. (2022) FQK concentration scaling on physics features.

Settings:
    n ∈ {4, 6, 8}
    FQK & PQK   : n-dim PCA of physics-16 raw features
    AGPQK       : n=6, 8  (top-n Fisher selection)
    N = 500,  seeds {42,43,44}

Report: off-diag variance, effective rank, Macro-F1 vs n.

Output: results/qubit_scaling/{metrics.csv, fig.pdf}
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

import config
from src.utils import ensure_dir, setup_logging
from src.quantum_kernels import (
    compute_fqk_kernel_matrix, compute_pqk_kernel_matrix,
)
from src.attention_kernel import (
    compute_attention_matrix, select_features_by_fisher,
    compute_per_qubit_gamma, get_attention_entanglement_pairs,
)
from src.kernel_target_alignment import compute_centered_kta
from experiments._e_common import (
    load_physics_16, spectral_metrics, svm_eval, build_agpqk_kernel,
)

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "qubit_scaling"))
logger = setup_logging("exp_e5", log_file=os.path.join(RES, "e5.log"))

NS    = [4, 6, 8]
N     = 500
SEEDS = [42, 43, 44]


def make_pca_features(X_raw, n):
    mu = X_raw.mean(0); sd = X_raw.std(0) + 1e-8
    Xs = (X_raw - mu) / sd
    Z  = PCA(n_components=n, random_state=42).fit_transform(Xs)
    # scale to [0, π]
    lo, hi = Z.min(0), Z.max(0)
    return np.pi * (Z - lo) / np.maximum(hi - lo, 1e-12)


def run():
    _, X_raw_full, y_full = load_physics_16()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=N, random_state=42)
    (idx, _), = sss.split(np.zeros(len(y_full)), y_full)
    X_raw, y = X_raw_full[idx], y_full[idx]

    rows = []
    for n in NS:
        Xp = make_pca_features(X_raw, n)

        # -- FQK --
        K_fqk = compute_fqk_kernel_matrix(Xp, n_qubits=n, reps=config.ZZ_REPS)
        np.save(os.path.join(RES, f"K_fqk_n{n}.npy"), K_fqk)

        # -- PQK --
        K_pqk = compute_pqk_kernel_matrix(Xp, n_qubits=n, reps=config.ZZ_REPS)
        np.save(os.path.join(RES, f"K_pqk_n{n}.npy"), K_pqk)

        for name, K in [("FQK", K_fqk), ("PQK", K_pqk)]:
            spm = spectral_metrics(K)
            kta = float(compute_centered_kta(K, y, class_weighted=True))
            for seed in SEEDS:
                ev = svm_eval(K, y, seed)
                rows.append(dict(kernel=name, n=n, seed=seed,
                                 macro_f1=ev["macro_f1"], kta=kta, **spm))

        # -- AGPQK for n ∈ {6, 8} --
        if n >= 6:
            A = compute_attention_matrix(X_raw)
            sel, fisher = select_features_by_fisher(X_raw, y, n_select=n,
                                                    min_sar=max(2, n // 2),
                                                    min_opt=max(2, n // 2))
            gpq = compute_per_qubit_gamma(fisher, sel, gamma_base=0.5)
            pairs = get_attention_entanglement_pairs(
                A, sel, top_k=max(2, n // 2),
                require_cross_modal=True, max_appearances=1)
            K_ag, _ = build_agpqk_kernel(X_raw, y, sel, gpq, pairs,
                                         n_qubits=n, reps=config.ZZ_REPS,
                                         outer_gamma=config.PQK_GAMMA)
            np.save(os.path.join(RES, f"K_agpqk_n{n}.npy"), K_ag)
            spm = spectral_metrics(K_ag)
            kta = float(compute_centered_kta(K_ag, y, class_weighted=True))
            for seed in SEEDS:
                ev = svm_eval(K_ag, y, seed)
                rows.append(dict(kernel="AGPQK", n=n, seed=seed,
                                 macro_f1=ev["macro_f1"], kta=kta, **spm))

        logger.info(f"n={n} done")

    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "metrics.csv"),
                                        index=False)
    _plot(df)


def _plot(df):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [("off_diag_var",     "Off-diag variance"),
               ("eff_rank_shannon", "Effective rank"),
               ("macro_f1",         "Macro-F1")]
    colors = {"FQK": "#d62728", "PQK": "#1f77b4", "AGPQK": "#9467bd"}
    for ax, (k, lab) in zip(axes, metrics):
        for kname, c in colors.items():
            sub = df[df.kernel == kname]
            if sub.empty: continue
            g = sub.groupby("n")[k].agg(["mean", "std"]).reset_index()
            ax.errorbar(g["n"], g["mean"], yerr=g["std"], marker="o",
                        label=kname, capsize=4, color=c)
        ax.set_xlabel("n (qubits)"); ax.set_ylabel(lab)
        ax.grid(alpha=0.3); ax.legend()
        if k == "off_diag_var":
            ax.set_yscale("log")
    fig.suptitle("E5 — qubit-scaling concentration")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
