"""
Experiment E1 — Gram Matrix Spectral Diagnostics (Concentration Analysis).

For each kernel K in {FQK, PQK, CM-FQK, AGPQK} × regime in {PCA-8, physics-16}:
    1. Off-diagonal kernel variance
    2. Off-diagonal mean
    3. Shannon effective rank (exp of normalized spectral entropy)
    4. Kernel polarization E[K^2] - E[K]^2  (off-diagonal)
    5. Macro-F1 on held-out fold (SVM on precomputed kernel)

5 seeds of stratified subsampling (n=1000 rows/cols) for error bars.
Output: results/spectral_diagnostics/metrics.csv + fig_2x4.pdf

Reference:
    - Thanasilp et al., Nat Commun 15, 5200 (2024) — FQK concentration theory
    - HaQGNN (arXiv:2506.21161)  — does NOT include spectral analysis on LCZ42
"""
import os, sys, json, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import f1_score

import config
from src.utils import ensure_dir, setup_logging

RESULTS_DIR = ensure_dir(os.path.join(config.RESULTS_DIR, "spectral_diagnostics"))
logger = setup_logging("exp_e1", log_file=os.path.join(RESULTS_DIR, "e1.log"))

PCA_DIR = os.path.join(config.RESULTS_DIR, "pca_cv", "full_kernels")
PHYS_DIR = os.path.join(config.RESULTS_DIR, "physics_cv", "full_kernels")

KERNEL_FILES = {
    "PCA-8": {
        "FQK":    os.path.join(PCA_DIR,  "K_fqk_full.npy"),
        "PQK":    os.path.join(PCA_DIR,  "K_pqk_full.npy"),
        "CM-FQK": os.path.join(PCA_DIR,  "K_cm_fqk_full.npy"),
    },
    "physics-16": {
        "FQK":    os.path.join(PHYS_DIR, "K_fqk_full.npy"),
        "PQK":    os.path.join(PHYS_DIR, "K_pqk_full.npy"),
        "CM-FQK": os.path.join(PHYS_DIR, "K_cm_fqk_full.npy"),
        "AGPQK":  os.path.join(PHYS_DIR, "K_agpqk_full.npy"),
    },
}

SEEDS     = config.SEED_LIST            # [42,43,44,45,46]
N_SUB     = 1000                         # rows/cols per bootstrap
TEST_FRAC = 0.3


def load_labels():
    phys = np.load(os.path.join(config.PROCESSED_DIR, "physics_features_16.npz"))
    return phys["y_train"][: config.SUBSAMPLE_TRAIN]


def off_diag_mask(n):
    return ~np.eye(n, dtype=bool)


def spectral_metrics(K):
    n = K.shape[0]
    mask = off_diag_mask(n)
    off = K[mask]
    off_mean = float(off.mean())
    off_var  = float(off.var())
    # Shannon effective rank
    Ksym = (K + K.T) / 2.0
    eig  = np.linalg.eigvalsh(Ksym)
    eig  = np.clip(eig, 1e-12, None)
    p    = eig / eig.sum()
    eff_rank_shannon = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    # Polarization  (off-diagonal)
    pol  = float(np.mean(off ** 2) - off_mean ** 2)
    # Normalised eff rank (max_eig-based, Huang-style) for cross-check
    eff_rank_huang = float(eig.sum() / eig.max())
    return {
        "off_diag_mean":       off_mean,
        "off_diag_var":        off_var,
        "eff_rank_shannon":    eff_rank_shannon,
        "eff_rank_huang":      eff_rank_huang,
        "polarization":        pol,
    }


def svm_f1_on_precomputed_kernel(K, y, seed):
    """Single stratified 70/30 split, precomputed-kernel SVM, Macro-F1."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)
    K_tr = K[np.ix_(tr, tr)]
    K_te = K[np.ix_(te, tr)]
    clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    pred = clf.predict(K_te)
    return f1_score(y[te], pred, average="macro")


def run():
    y_full = load_labels()
    records = []

    for regime, kernels in KERNEL_FILES.items():
        for kname, path in kernels.items():
            if not os.path.exists(path):
                logger.warning(f"Missing: {path}")
                continue
            K_full = np.load(path)
            logger.info(f"[{regime}] {kname}: K shape {K_full.shape}")

            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                # Stratified subsample
                sss = StratifiedShuffleSplit(n_splits=1, train_size=N_SUB,
                                             random_state=seed)
                (idx, _), = sss.split(np.zeros(len(y_full)), y_full)
                Ks = K_full[np.ix_(idx, idx)]
                y_sub = y_full[idx]
                m = spectral_metrics(Ks)
                try:
                    f1 = svm_f1_on_precomputed_kernel(Ks, y_sub, seed)
                except Exception as e:
                    logger.error(f"SVM fail {kname}/{regime}/seed{seed}: {e}")
                    f1 = np.nan
                m.update(dict(regime=regime, kernel=kname, seed=seed,
                              macro_f1=f1))
                records.append(m)
                logger.info(f"  seed={seed}  F1={f1:.4f}  "
                            f"off_var={m['off_diag_var']:.3e}  "
                            f"eff_rank={m['eff_rank_shannon']:.1f}  "
                            f"pol={m['polarization']:.3e}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {csv_path}")

    # Aggregate mean±std for paper
    agg = (df.groupby(["regime", "kernel"])
             .agg(["mean", "std"])[["off_diag_var", "off_diag_mean",
                                    "eff_rank_shannon", "polarization",
                                    "macro_f1"]])
    agg_path = os.path.join(RESULTS_DIR, "metrics_aggregated.csv")
    agg.to_csv(agg_path)
    logger.info(f"Saved {agg_path}")

    plot_2x4(df)
    return df


def plot_2x4(df):
    import matplotlib.pyplot as plt
    metrics = [
        ("off_diag_var",     "Off-diag variance"),
        ("eff_rank_shannon", "Shannon effective rank"),
        ("polarization",     "Kernel polarization"),
        ("macro_f1",         "Macro-F1"),
    ]
    regimes = ["PCA-8", "physics-16"]
    kernel_order = ["FQK", "PQK", "CM-FQK", "AGPQK"]
    colors = {"FQK":"#d62728","PQK":"#1f77b4","CM-FQK":"#2ca02c","AGPQK":"#9467bd"}

    fig, axes = plt.subplots(4, 2, figsize=(9, 11), sharex=False)
    for col, regime in enumerate(regimes):
        sub = df[df.regime == regime]
        for row, (mkey, mlabel) in enumerate(metrics):
            ax = axes[row, col]
            kernels_present = [k for k in kernel_order if k in sub.kernel.unique()]
            means = [sub[sub.kernel == k][mkey].mean() for k in kernels_present]
            stds  = [sub[sub.kernel == k][mkey].std()  for k in kernels_present]
            bars = ax.bar(kernels_present, means, yerr=stds, capsize=4,
                          color=[colors[k] for k in kernels_present],
                          alpha=0.85, edgecolor="black")
            ax.set_title(f"{regime} — {mlabel}", fontsize=10)
            ax.set_ylabel(mlabel, fontsize=9)
            if mkey in ("off_diag_var", "polarization"):
                ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.3)
    fig.suptitle("E1 — Gram matrix spectral diagnostics "
                 "(N_sub=1000, 5 seeds)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ("pdf", "png"):
        p = os.path.join(RESULTS_DIR, f"fig_e1_2x4.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
        logger.info(f"Saved {p}")
    plt.close(fig)


if __name__ == "__main__":
    run()
