"""
Experiment E16 — Kernel-alignment variance vs (n, depth).

Barren-plateau analog for kernels (Thanasilp et al. 2208.11060 concentration).
Var(KTA) across 5 seeds for FQK / PQK / AGPQK at
    n ∈ {4, 6, 8}  and  reps ∈ {1, 2, 3}
on physics-16 features (n-dim PCA), N=400 subsample.

Interpretation:
    Var(KTA) decreasing with n or depth → signal vanishing (bad for trainability).
    AGPQK with per-qubit γ + attention should concentrate less than plain FQK.

Output: results/kta_variance/{metrics.csv, fig.pdf}
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
    load_physics_16, spectral_metrics, build_agpqk_kernel,
)

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "kta_variance"))
logger = setup_logging("exp_e16", log_file=os.path.join(RES, "e16.log"))

NS    = [4, 6, 8]
REPS  = [1, 2, 3]
N     = 400
SEEDS = [42, 43, 44, 45, 46]


def pca_pi(X_raw, n, seed):
    mu = X_raw.mean(0); sd = X_raw.std(0) + 1e-8
    Z = PCA(n_components=n, random_state=seed).fit_transform((X_raw - mu) / sd)
    lo, hi = Z.min(0), Z.max(0)
    return np.pi * (Z - lo) / np.maximum(hi - lo, 1e-12)


def subsample(X_raw, y_full, seed):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=N, random_state=seed)
    (idx, _), = sss.split(np.zeros(len(y_full)), y_full)
    return X_raw[idx], y_full[idx]


def run():
    _, X_raw_full, y_full = load_physics_16()

    rows = []
    for n in NS:
        for reps in REPS:
            for seed in SEEDS:
                X_raw, y = subsample(X_raw_full, y_full, seed)
                Xp = pca_pi(X_raw, n, seed)

                # FQK
                K_fqk = compute_fqk_kernel_matrix(Xp, n_qubits=n, reps=reps)
                kta_f = float(compute_centered_kta(K_fqk, y,
                                                   class_weighted=True))
                spm_f = spectral_metrics(K_fqk)

                # PQK
                K_pqk = compute_pqk_kernel_matrix(Xp, n_qubits=n, reps=reps)
                kta_p = float(compute_centered_kta(K_pqk, y,
                                                   class_weighted=True))
                spm_p = spectral_metrics(K_pqk)

                rows.append(dict(kernel="FQK", n=n, reps=reps, seed=seed,
                                 kta=kta_f, **spm_f))
                rows.append(dict(kernel="PQK", n=n, reps=reps, seed=seed,
                                 kta=kta_p, **spm_p))

                # AGPQK for n ∈ {6, 8}
                if n >= 6:
                    sel, fisher = select_features_by_fisher(
                        X_raw, y, n_select=n,
                        min_sar=max(2, n // 2), min_opt=max(2, n // 2))
                    A = compute_attention_matrix(X_raw)
                    pairs = get_attention_entanglement_pairs(
                        A, sel, top_k=max(2, n // 2),
                        require_cross_modal=True, max_appearances=1)
                    gamma = compute_per_qubit_gamma(fisher, sel,
                                                    gamma_base=0.5)
                    K_ag, _ = build_agpqk_kernel(X_raw, y, sel, gamma, pairs,
                                                 n_qubits=n, reps=reps,
                                                 outer_gamma=config.PQK_GAMMA)
                    kta_a = float(compute_centered_kta(K_ag, y,
                                                       class_weighted=True))
                    spm_a = spectral_metrics(K_ag)
                    rows.append(dict(kernel="AGPQK", n=n, reps=reps, seed=seed,
                                     kta=kta_a, **spm_a))
                logger.info(f"n={n} reps={reps} seed={seed} done")

    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "metrics.csv"),
                                        index=False)
    _plot(df)


def _plot(df):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"FQK": "#d62728", "PQK": "#1f77b4", "AGPQK": "#9467bd"}

    # Panel 1: Var(KTA) vs n (marginalized over depth)
    ax = axes[0]
    for k, c in colors.items():
        sub = df[df.kernel == k]
        if sub.empty: continue
        g = sub.groupby("n")["kta"].var().reset_index()
        ax.plot(g["n"], g["kta"], marker="o", color=c, label=k)
    ax.set_xlabel("n (qubits)"); ax.set_ylabel("Var(KTA) across seeds")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Concentration vs n")

    # Panel 2: Var(KTA) vs depth
    ax = axes[1]
    for k, c in colors.items():
        sub = df[df.kernel == k]
        if sub.empty: continue
        g = sub.groupby("reps")["kta"].var().reset_index()
        ax.plot(g["reps"], g["kta"], marker="o", color=c, label=k)
    ax.set_xlabel("ZZ-reps (depth)"); ax.set_ylabel("Var(KTA) across seeds")
    ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Concentration vs depth")

    fig.suptitle("E16 — KTA variance (kernel barren-plateau analog)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
