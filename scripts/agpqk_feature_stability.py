"""
AGPQK Feature Selection Stability Analysis.

Reports which 8 features are selected by Fisher-ratio in each of the 5 CV folds.
If different features are selected across folds, this indicates instability
in the feature selection process.

Uses existing physics_features_16.npz — no raw data needed.

Output:
    results/agpqk_stability/agpqk_stability_results.json
    results/agpqk_stability/stability_table.csv

Runtime: ~30 minutes.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
from sklearn.model_selection import StratifiedKFold

import config
from src.attention_kernel import (
    select_features_by_fisher,
    FEATURE_NAMES_16,
    FEATURE_MODALITY_16,
)
from src.utils import save_results, setup_logging, ensure_dir

logger = setup_logging(
    "agpqk_stability",
    log_file=os.path.join(config.RESULTS_DIR, "agpqk_stability", "agpqk_stability.log"),
)

RESULTS_DIR = ensure_dir(os.path.join(config.RESULTS_DIR, "agpqk_stability"))
N_SELECT = 8


def load_physics_data():
    phys16_path = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
    data = np.load(phys16_path)
    X_all = data["X_train"][: config.SUBSAMPLE_TRAIN]
    X_all_raw = data["X_train_raw"][: config.SUBSAMPLE_TRAIN]
    y_all = data["y_train"][: config.SUBSAMPLE_TRAIN]
    return X_all_raw, y_all


def run_analysis():
    results_file = os.path.join(RESULTS_DIR, "agpqk_stability_results.json")
    if os.path.exists(results_file):
        logger.info(f"[SKIP] Results exist: {results_file}")
        return

    np.random.seed(config.RANDOM_SEED)
    X_raw, y = load_physics_data()

    skf = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.CV_SEED
    )
    fold_selections = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
        X_tr_raw = X_raw[train_idx]
        y_tr = y[train_idx]

        selected_indices, fisher = select_features_by_fisher(
            X_tr_raw, y_tr, n_select=N_SELECT, min_sar=4, min_opt=4
        )

        fold_sel = {
            "fold": fold_idx,
            "selected_indices": selected_indices.tolist(),
            "selected_names": [FEATURE_NAMES_16[i] for i in selected_indices],
            "selected_modalities": [FEATURE_MODALITY_16[i] for i in selected_indices],
            "fisher_scores": fisher.tolist(),
        }
        fold_selections.append(fold_sel)

        logger.info(f"  Fold {fold_idx}: {fold_sel['selected_names']}")

    # ---- Stability analysis ----
    # How many times was each feature selected?
    feature_counts = np.zeros(16, dtype=int)
    for fs in fold_selections:
        for idx in fs["selected_indices"]:
            feature_counts[idx] += 1

    # Jaccard similarity between all fold pairs
    from itertools import combinations

    jaccard_scores = []
    for i, j in combinations(range(config.CV_FOLDS), 2):
        set_i = set(fold_selections[i]["selected_indices"])
        set_j = set(fold_selections[j]["selected_indices"])
        jaccard = len(set_i & set_j) / len(set_i | set_j)
        jaccard_scores.append({"fold_i": i, "fold_j": j, "jaccard": jaccard})

    avg_jaccard = np.mean([j["jaccard"] for j in jaccard_scores])

    # Features selected in ALL folds
    all_folds_set = set(fold_selections[0]["selected_indices"])
    for fs in fold_selections[1:]:
        all_folds_set &= set(fs["selected_indices"])
    stable_features = [FEATURE_NAMES_16[i] for i in sorted(all_folds_set)]

    # Features selected in NO folds
    never_selected = [i for i in range(16) if feature_counts[i] == 0]

    logger.info(f"\n{'=' * 60}")
    logger.info("  AGPQK FEATURE SELECTION STABILITY")
    logger.info(f"{'=' * 60}")
    logger.info(f"\n  Feature selection frequency across {config.CV_FOLDS} folds:")
    for i in range(16):
        bar = "█" * feature_counts[i]
        logger.info(
            f"    {FEATURE_NAMES_16[i]:<15s} ({FEATURE_MODALITY_16[i]:>3s}): "
            f"{feature_counts[i]}/{config.CV_FOLDS}  {bar}"
        )

    logger.info(f"\n  Features selected in ALL folds: {stable_features}")
    logger.info(
        f"  Features never selected: {[FEATURE_NAMES_16[i] for i in never_selected]}"
    )
    logger.info(f"  Mean Jaccard similarity: {avg_jaccard:.4f}")

    if avg_jaccard >= 0.8:
        logger.info("  ✓ Feature selection is STABLE (Jaccard ≥ 0.8)")
    elif avg_jaccard >= 0.5:
        logger.info("  ⚠ Feature selection is MODERATELY STABLE (0.5 ≤ Jaccard < 0.8)")
    else:
        logger.info("  ✗ Feature selection is UNSTABLE (Jaccard < 0.5)")

    save_results(
        {
            "fold_selections": fold_selections,
            "feature_counts": {
                FEATURE_NAMES_16[i]: int(feature_counts[i]) for i in range(16)
            },
            "jaccard_scores": jaccard_scores,
            "mean_jaccard": float(avg_jaccard),
            "stable_features": stable_features,
            "never_selected": [FEATURE_NAMES_16[i] for i in never_selected],
            "stability": "stable"
            if avg_jaccard >= 0.8
            else "moderate"
            if avg_jaccard >= 0.5
            else "unstable",
        },
        results_file,
    )


if __name__ == "__main__":
    run_analysis()
