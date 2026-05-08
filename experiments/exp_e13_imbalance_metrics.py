"""
Experiment E13 — Imbalance-aware metrics.

For all cached kernels + tuned RBF baseline, compute:
    - Balanced accuracy
    - Cohen's κ
    - Macro AUC (OvR, decision_function or probability)
    - G-mean (geometric mean of per-class recall)

LCZ42's 67:1 majority:minority ratio makes plain Macro-F1 insufficient;
these metrics are what QMI reviewers look at for imbalanced EO work.

Output: results/imbalance_metrics/{metrics.csv, fig.pdf}
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             roc_auc_score, recall_score, f1_score)
from sklearn.preprocessing import label_binarize

import config
from src.utils import ensure_dir, setup_logging
from experiments._e_common import load_physics_16

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "imbalance_metrics"))
logger = setup_logging("exp_e13", log_file=os.path.join(RES, "e13.log"))

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


def gmean(y_true, y_pred, classes):
    recs = recall_score(y_true, y_pred, labels=classes, average=None,
                        zero_division=0)
    recs = np.clip(recs, 1e-8, None)
    return float(np.exp(np.mean(np.log(recs))))


def auc_macro(y_true, scores, classes):
    try:
        yb = label_binarize(y_true, classes=classes)
        return float(roc_auc_score(yb, scores, multi_class="ovr",
                                   average="macro"))
    except Exception:
        return float("nan")


def fit_precomputed(K, y, tr, te):
    clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced",
              decision_function_shape="ovr")
    clf.fit(K[np.ix_(tr, tr)], y[tr])
    yp = clf.predict(K[np.ix_(te, tr)])
    try:
        scr = clf.decision_function(K[np.ix_(te, tr)])
    except Exception:
        scr = None
    return yp, scr, clf.classes_


def fit_rbf(X, y, tr, te):
    grid = {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1]}
    clf = GridSearchCV(SVC(class_weight="balanced",
                           decision_function_shape="ovr"),
                       grid, cv=3, scoring="f1_macro", n_jobs=-1)
    clf.fit(X[tr], y[tr])
    est = clf.best_estimator_
    yp = est.predict(X[te])
    try:
        scr = est.decision_function(X[te])
    except Exception:
        scr = None
    return yp, scr, est.classes_


def run():
    X, _, y = load_physics_16()
    classes = np.unique(y)

    kernels = {name: np.load(p) for name, p in KERNELS if os.path.exists(p)}

    rows = []
    for seed in config.SEED_LIST:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3,
                                     random_state=seed)
        (tr, te), = sss.split(np.zeros(len(y)), y)
        y_te = y[te]

        for name, K in kernels.items():
            yp, scr, cls = fit_precomputed(K, y, tr, te)
            rows.append(dict(
                kernel=name, seed=seed,
                macro_f1=f1_score(y_te, yp, average="macro", zero_division=0),
                balanced_acc=balanced_accuracy_score(y_te, yp),
                cohen_kappa=cohen_kappa_score(y_te, yp),
                gmean=gmean(y_te, yp, classes),
                macro_auc=auc_macro(y_te, scr, cls) if scr is not None else float("nan")
            ))
        # RBF baseline
        yp, scr, cls = fit_rbf(X, y, tr, te)
        rows.append(dict(
            kernel="RBF-tuned", seed=seed,
            macro_f1=f1_score(y_te, yp, average="macro", zero_division=0),
            balanced_acc=balanced_accuracy_score(y_te, yp),
            cohen_kappa=cohen_kappa_score(y_te, yp),
            gmean=gmean(y_te, yp, classes),
            macro_auc=auc_macro(y_te, scr, cls) if scr is not None else float("nan")
        ))
        logger.info(f"seed={seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "metrics.csv"), index=False)
    _plot(df)


def _plot(df):
    import matplotlib.pyplot as plt
    metrics = ["macro_f1", "balanced_acc", "cohen_kappa", "gmean", "macro_auc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
    for ax, m in zip(axes, metrics):
        g = df.groupby("kernel")[m].agg(["mean", "std"]).reset_index()
        g = g.sort_values("mean")
        ax.barh(g["kernel"], g["mean"], xerr=g["std"], capsize=4,
                color="#6a5acd")
        ax.set_xlabel(m); ax.grid(alpha=0.3, axis="x")
    fig.suptitle("E13 — Imbalance-aware metrics")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
