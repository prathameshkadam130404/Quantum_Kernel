"""
Experiment E10 — Learning-curve vs classical.

Macro-F1 as a function of training-set size N ∈ {250, 500, 1000, 2000}
for AGPQK vs tuned RBF-SVM on physics-16 features.

Uses the existing 2000-sample AGPQK Gram matrix (cached) and submatrix slicing.
RBF with CV-tuned γ is re-fit at each N on the same stratified subsample.

Output: results/learning_curve/{curve.csv, curve.pdf, curve.png}
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

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "learning_curve"))
logger = setup_logging("exp_e10", log_file=os.path.join(RES, "e10.log"))

AGPQK_FULL = os.path.join(config.RESULTS_DIR, "physics_cv",
                          "full_kernels", "K_agpqk_full.npy")
SIZES = [250, 500, 1000, 1500, 2000]
SEEDS = config.SEED_LIST


def rbf_tuned_f1(X_tr, y_tr, X_te, y_te):
    grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1.0]}
    clf = GridSearchCV(SVC(class_weight="balanced"), grid,
                       cv=3, scoring="f1_macro", n_jobs=-1)
    clf.fit(X_tr, y_tr)
    return float(f1_score(y_te, clf.predict(X_te), average="macro")), clf.best_params_


def agpqk_f1(K_full, tr, te, y):
    K_tr = K_full[np.ix_(tr, tr)]; K_te = K_full[np.ix_(te, tr)]
    clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    return float(f1_score(y[te], clf.predict(K_te), average="macro"))


def run():
    X, _, y = load_physics_16()
    if not os.path.exists(AGPQK_FULL):
        raise FileNotFoundError(AGPQK_FULL)
    K = np.load(AGPQK_FULL)

    rows = []
    for N in SIZES:
        for seed in SEEDS:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=min(N, len(y)-200),
                                         test_size=200, random_state=seed)
            (tr, te), = sss.split(np.zeros(len(y)), y)
            f1_q = agpqk_f1(K, tr, te, y)
            f1_r, best = rbf_tuned_f1(X[tr], y[tr], X[te], y[te])
            rows.append(dict(N=N, seed=seed, kernel="AGPQK", macro_f1=f1_q))
            rows.append(dict(N=N, seed=seed, kernel="RBF-tuned",
                             macro_f1=f1_r, best=str(best)))
            logger.info(f"N={N} seed={seed}  AGPQK={f1_q:.4f}  RBF={f1_r:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "curve.csv"), index=False)
    _plot(df)


def _plot(df):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, c in [("AGPQK", "#9467bd"), ("RBF-tuned", "#ff7f0e")]:
        sub = df[df.kernel == k]
        g = sub.groupby("N")["macro_f1"].agg(["mean", "std"]).reset_index()
        ax.errorbar(g["N"], g["mean"], yerr=g["std"], marker="o",
                    label=k, color=c, capsize=4)
    ax.set_xlabel("Training-set size N")
    ax.set_ylabel("Macro-F1")
    ax.set_title("E10 — AGPQK vs tuned RBF learning curve (physics-16)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"curve.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
