"""
Experiment E12 — Per-class F1 + 1000× stratified bootstrap 95 % CIs.

Across all cached kernels (FQK, PQK, CM-FQK, AGPQK on physics-16 and PCA-8),
plus a tuned RBF baseline, report:
    - Per-class F1 (17 LCZ classes)
    - Macro-F1 with bootstrap 95 % CI (1000 resamples of the test split)

The 0.487 AGPQK vs 0.502 RBF gap may be inside the CI; this test may
collapse the gap into a statistical tie, reframing the narrative.

Output: results/per_class_ci/{per_class.csv, bootstrap.csv, fig.pdf}
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score

import config
from src.utils import ensure_dir, setup_logging
from experiments._e_common import load_physics_16

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "per_class_ci"))
logger = setup_logging("exp_e12", log_file=os.path.join(RES, "e12.log"))

PHYS_DIR = os.path.join(config.RESULTS_DIR, "physics_cv", "full_kernels")
PCA_DIR  = os.path.join(config.RESULTS_DIR, "pca_cv",     "full_kernels")

KERNELS = [
    ("FQK-physics",    os.path.join(PHYS_DIR, "K_fqk_full.npy")),
    ("PQK-physics",    os.path.join(PHYS_DIR, "K_pqk_full.npy")),
    ("CM-FQK-physics", os.path.join(PHYS_DIR, "K_cm_fqk_full.npy")),
    ("AGPQK-physics",  os.path.join(PHYS_DIR, "K_agpqk_full.npy")),
    ("FQK-pca",        os.path.join(PCA_DIR,  "K_fqk_full.npy")),
    ("PQK-pca",        os.path.join(PCA_DIR,  "K_pqk_full.npy")),
    ("CM-FQK-pca",     os.path.join(PCA_DIR,  "K_cm_fqk_full.npy")),
]

SEEDS = config.SEED_LIST
N_BOOT = 1000


def split(y, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)
    return tr, te


def eval_precomputed(K, y, tr, te):
    clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
    clf.fit(K[np.ix_(tr, tr)], y[tr])
    return clf.predict(K[np.ix_(te, tr)])


def eval_rbf(X, y, tr, te):
    grid = {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1]}
    clf = GridSearchCV(SVC(class_weight="balanced"), grid, cv=3,
                       scoring="f1_macro", n_jobs=-1)
    clf.fit(X[tr], y[tr])
    return clf.predict(X[te])


def bootstrap_macro_f1(y_true, y_pred, classes, n_boot, rng):
    """Stratified bootstrap over test indices, preserving class support."""
    out = np.empty(n_boot)
    idx_by_c = {c: np.where(y_true == c)[0] for c in classes}
    for b in range(n_boot):
        picks = []
        for c in classes:
            idx_c = idx_by_c[c]
            if len(idx_c) == 0:
                continue
            picks.append(rng.choice(idx_c, size=len(idx_c), replace=True))
        idx = np.concatenate(picks)
        out[b] = f1_score(y_true[idx], y_pred[idx],
                          average="macro", zero_division=0)
    return out


def run():
    X, _, y = load_physics_16()
    classes = np.unique(y)

    # Predictions (seed=42 for per-class CSV; all seeds for bootstrap aggregation)
    per_class_rows = []
    boot_rows = []

    # Preload kernels
    kernels = {}
    for name, path in KERNELS:
        if os.path.exists(path):
            kernels[name] = np.load(path)
        else:
            logger.warning(f"missing kernel: {path}")

    rng = np.random.default_rng(42)

    for seed in SEEDS:
        tr, te = split(y, seed)
        y_te = y[te]

        # Each kernel
        preds = {}
        for name, K in kernels.items():
            preds[name] = eval_precomputed(K, y, tr, te)
        # RBF baseline
        preds["RBF-tuned"] = eval_rbf(X, y, tr, te)

        for name, yp in preds.items():
            f1_per = f1_score(y_te, yp, labels=classes,
                              average=None, zero_division=0)
            for c, v in zip(classes, f1_per):
                per_class_rows.append(dict(kernel=name, seed=seed,
                                           class_id=int(c), f1=float(v)))
            macro = float(np.mean(f1_per))
            boots = bootstrap_macro_f1(y_te, yp, classes, N_BOOT, rng)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            boot_rows.append(dict(kernel=name, seed=seed,
                                  macro_f1=macro,
                                  ci_lo=float(lo), ci_hi=float(hi),
                                  boot_std=float(boots.std())))
            logger.info(f"{name:16s} seed={seed} F1={macro:.4f} "
                        f"95%CI=[{lo:.4f},{hi:.4f}]")

    df_pc = pd.DataFrame(per_class_rows)
    df_pc.to_csv(os.path.join(RES, "per_class.csv"), index=False)
    df_b = pd.DataFrame(boot_rows)
    df_b.to_csv(os.path.join(RES, "bootstrap.csv"), index=False)
    _plot(df_pc, df_b)


def _plot(df_pc, df_b):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-class F1 heatmap-like bar groups (avg across seeds)
    ax = axes[0]
    piv = df_pc.groupby(["kernel", "class_id"])["f1"].mean().unstack()
    piv.plot(kind="bar", ax=ax, width=0.9, legend=False, colormap="tab20")
    ax.set_ylabel("Per-class F1"); ax.set_xlabel("Kernel")
    ax.set_title("Per-class F1 across 17 LCZ classes")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")

    # Macro-F1 with 95 % CI
    ax = axes[1]
    g = df_b.groupby("kernel").agg(mean=("macro_f1", "mean"),
                                   lo=("ci_lo", "mean"),
                                   hi=("ci_hi", "mean")).reset_index()
    g = g.sort_values("mean")
    y_pos = np.arange(len(g))
    ax.errorbar(g["mean"], y_pos,
                xerr=[g["mean"] - g["lo"], g["hi"] - g["mean"]],
                fmt="o", capsize=5, color="#1f77b4")
    ax.set_yticks(y_pos); ax.set_yticklabels(g["kernel"])
    ax.set_xlabel("Macro-F1 (mean ± 95 % bootstrap CI)")
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("Macro-F1 with 1000× stratified bootstrap CI")

    fig.suptitle("E12 — Per-class F1 + bootstrap 95 % CIs")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
