"""
Per-Class F1 Analysis for Physics Regime.

Generates per-class F1 bar charts comparing AGPQK vs RBF-on-AGPQK-features
on the full 2000-sample physics test set.

Uses pre-computed AGPQK kernel (already cached) — no recomputation needed.

Output:
    results/physics_per_class_f1/per_class_f1.png
    results/physics_per_class_f1/per_class_f1_table.csv

Runtime: ~5 minutes.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.classifiers import train_precomputed_svm
from src.utils import ensure_dir, setup_logging

logger = setup_logging(
    "per_class_f1",
    log_file=os.path.join(
        config.RESULTS_DIR, "physics_per_class_f1", "per_class_f1.log"
    ),
)

RESULTS_DIR = ensure_dir(os.path.join(config.RESULTS_DIR, "physics_per_class_f1"))


def load_physics_data():
    phys16_path = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
    data = np.load(phys16_path)
    X_train = data["X_train"][: config.SUBSAMPLE_TRAIN]
    y_train = data["y_train"][: config.SUBSAMPLE_TRAIN]
    X_test = data["X_test"][: config.SUBSAMPLE_TEST]
    y_test = data["y_test"][: config.SUBSAMPLE_TEST]
    return X_train, y_train, X_test, y_test


def load_agpqk_config():
    cfg_path = os.path.join(config.RESULTS_DIR, "physics", "agpqk_config.json")
    with open(cfg_path) as f:
        return json.load(f)


def run_analysis():
    X_train, y_train, X_test, y_test = load_physics_data()
    agpqk_cfg = load_agpqk_config()
    sel_idx = agpqk_cfg["selected_feature_indices"]

    # Load pre-computed AGPQK kernel
    K_agpqk_tr = np.load(
        os.path.join(
            config.RESULTS_DIR, "physics", "fused", "K_agpqk_physics_train.npy"
        )
    )
    K_agpqk_te = np.load(
        os.path.join(config.RESULTS_DIR, "physics", "fused", "K_agpqk_physics_test.npy")
    )

    # Train AGPQK-SVM
    clf_agpqk = train_precomputed_svm(K_agpqk_tr, y_train)
    y_pred_agpqk = clf_agpqk.predict(K_agpqk_te)

    # Train RBF-on-AGPQK-features SVM
    X_tr_sel = X_train[:, sel_idx]
    X_te_sel = X_test[:, sel_idx]
    from sklearn.svm import SVC

    clf_rbf_sel = SVC(
        kernel="rbf", class_weight="balanced", random_state=config.RANDOM_SEED
    )
    clf_rbf_sel.fit(X_tr_sel, y_train)
    y_pred_rbf_sel = clf_rbf_sel.predict(X_te_sel)

    # Per-class F1
    from sklearn.metrics import f1_score

    n_classes = config.LCZ_N_CLASSES
    f1_agpqk = f1_score(y_test, y_pred_agpqk, average=None, labels=range(n_classes))
    f1_rbf_sel = f1_score(y_test, y_pred_rbf_sel, average=None, labels=range(n_classes))

    # Per-class support
    from collections import Counter

    support = [Counter(y_test).get(i, 0) for i in range(n_classes)]

    # Table
    logger.info(f"\n{'=' * 60}")
    logger.info("  PER-CLASS F1 (PHYSICS REGIME)")
    logger.info(f"{'=' * 60}")
    logger.info(f"{'Class':<25s} {'Support':>8s} {'AGPQK':>8s} {'RBF-AGPQK':>10s}")
    logger.info("-" * 55)
    for i in range(n_classes):
        logger.info(
            f"{config.LCZ_CLASS_NAMES[i]:<25s} {support[i]:>8d} "
            f"{f1_agpqk[i]:>8.4f} {f1_rbf_sel[i]:>10.4f}"
        )

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(n_classes)
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, f1_agpqk, width, label="AGPQK", color="steelblue", alpha=0.85
    )
    bars2 = ax.bar(
        x + width / 2,
        f1_rbf_sel,
        width,
        label="RBF-on-AGPQK-features",
        color="coral",
        alpha=0.85,
    )

    ax.set_xlabel("Local Climate Zone Class", fontsize=12)
    ax.set_ylabel("Per-Class F1 Score", fontsize=12)
    ax.set_title(
        "Per-Class F1: AGPQK vs RBF-on-AGPQK-features (Physics Features, N=2000)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [config.LCZ_CLASS_NAMES[i][:12] for i in range(n_classes)],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    # Add support annotations
    for i, s in enumerate(support):
        ax.annotate(
            f"n={s}",
            xy=(i, 0),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            fontsize=6,
            color="gray",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "per_class_f1.png"), dpi=200, bbox_inches="tight"
    )
    plt.close()

    # Save data
    import csv

    csv_path = os.path.join(RESULTS_DIR, "per_class_f1_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "class_id",
                "class_name",
                "support",
                "f1_agpqk",
                "f1_rbf_on_agpqk_features",
            ]
        )
        for i in range(n_classes):
            writer.writerow(
                [
                    i,
                    config.LCZ_CLASS_NAMES[i],
                    support[i],
                    round(f1_agpqk[i], 4),
                    round(f1_rbf_sel[i], 4),
                ]
            )

    logger.info(f"\nPlot saved: {os.path.join(RESULTS_DIR, 'per_class_f1.png')}")
    logger.info(f"Table saved: {csv_path}")


if __name__ == "__main__":
    run_analysis()
