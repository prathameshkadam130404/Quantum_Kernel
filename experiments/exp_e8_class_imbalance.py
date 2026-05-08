"""
Experiment E8 — Class-imbalance sensitivity.

Vary the majority:minority ratio at fixed N_total = 2000 (approximate; we
rebalance by per-class subsampling to hit a target majority:minority ratio
while keeping ≥5 samples per minority class).  Ratios: 1:1, 5:1, 20:1, 67:1.

AGPQK (physics cached Gram) vs tuned RBF (raw physics-16).  Report:
    Macro-F1, ΔKTA, off-diagonal variance vs imbalance ratio, 5 seeds.

Output:  results/imbalance/{metrics.csv, fig.pdf}
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score

import config
from src.utils import ensure_dir, setup_logging
from src.kernel_target_alignment import compute_centered_kta
from experiments._e_common import (
    load_physics_16, spectral_metrics, svm_eval,
)

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "imbalance"))
logger = setup_logging("exp_e8", log_file=os.path.join(RES, "e8.log"))

AGPQK_PATH = os.path.join(config.RESULTS_DIR, "physics_cv",
                          "full_kernels", "K_agpqk_full.npy")
RATIOS = [1, 5, 20, 67]          # majority:minority
N_MINORITY_FLOOR = 5


def subsample_to_ratio(y, ratio, rng):
    """Return indices yielding a dataset whose majority:minority ratio == `ratio`
    approximately. Classes kept proportionally; per-class sizes
    geometrically interpolated between min and max."""
    counts = Counter(y.tolist())
    classes = sorted(counts.keys())
    c_max = max(counts.values())
    target_min = max(N_MINORITY_FLOOR, int(c_max / ratio))
    target_max = c_max
    # Rank classes by natural frequency (already ordered by count desc later)
    freq_sorted = sorted(classes, key=lambda c: counts[c], reverse=True)
    idx_all = []
    K = len(classes)
    for rank, c in enumerate(freq_sorted):
        if K == 1:
            target = counts[c]
        else:
            t = rank / (K - 1)
            target = int(round(target_max * (target_min / target_max) ** t))
        target = min(target, counts[c])
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)
        idx_all.extend(class_idx[:target].tolist())
    return np.array(sorted(idx_all))


def agpqk_f1(K_full, idx, y, seed):
    Ksub = K_full[np.ix_(idx, idx)]; ysub = y[idx]
    return svm_eval(Ksub, ysub, seed), Ksub, ysub


def rbf_f1(X, idx, y, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    (tr, te), = sss.split(np.zeros(len(idx)), y[idx])
    tr, te = idx[tr], idx[te]
    grid = {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1]}
    clf = GridSearchCV(SVC(class_weight="balanced"), grid, cv=3,
                       scoring="f1_macro", n_jobs=-1)
    clf.fit(X[tr], y[tr])
    return float(f1_score(y[te], clf.predict(X[te]), average="macro"))


def run():
    X, _, y = load_physics_16()
    K_full = np.load(AGPQK_PATH)

    rows = []
    for ratio in RATIOS:
        for seed in config.SEED_LIST:
            rng = np.random.default_rng(seed)
            idx = subsample_to_ratio(y, ratio, rng)
            ev, Ksub, ysub = agpqk_f1(K_full, idx, y, seed)
            kta = float(compute_centered_kta(Ksub, ysub, class_weighted=True))
            spm = spectral_metrics(Ksub)
            f1r = rbf_f1(X, idx, y, seed)
            rows.append(dict(ratio=ratio, seed=seed, n_kept=len(idx),
                             agpqk_f1=ev["macro_f1"], rbf_f1=f1r,
                             kta=kta, **spm))
            logger.info(f"ratio={ratio} seed={seed} n={len(idx)} "
                        f"AGPQK={ev['macro_f1']:.3f} RBF={f1r:.3f}")

    df = pd.DataFrame(rows); df.to_csv(os.path.join(RES, "metrics.csv"),
                                        index=False)
    _plot(df)


def _plot(df):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = [("agpqk_f1", "AGPQK Macro-F1"),
               ("kta",       "ΔKTA"),
               ("off_diag_var", "Off-diag variance")]
    for ax, (k, lab) in zip(axes, metrics):
        g = df.groupby("ratio")[k].agg(["mean", "std"]).reset_index()
        ax.errorbar(g["ratio"], g["mean"], yerr=g["std"],
                    marker="o", capsize=4, color="#9467bd")
        if k == "agpqk_f1":
            g2 = df.groupby("ratio")["rbf_f1"].agg(["mean", "std"]).reset_index()
            ax.errorbar(g2["ratio"], g2["mean"], yerr=g2["std"],
                        marker="s", capsize=4, color="#ff7f0e", label="RBF")
            ax.legend()
        ax.set_xscale("log"); ax.set_xlabel("Majority:Minority ratio")
        ax.set_ylabel(lab); ax.grid(alpha=0.3)
    fig.suptitle("E8 — class-imbalance sensitivity")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
