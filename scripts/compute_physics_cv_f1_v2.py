"""
Compute Physics CV F1 V2 — Upgraded 5-fold CV for physics regime.

This is a lightweight wrapper that runs 5-fold CV specifically for
the physics regime, focusing on the key comparison:
  AGPQK vs RBF-on-AGPQK-features vs RBF-CV

Uses pre-computed AGPQK config for feature selection.
Results are saved to results/reviewer_fixes/physics_cv_f1_v2.json

This supersedes the original compute_physics_cv_f1.py.

Runtime: ~2-3 hours (recomputes kernels per fold).
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import config
from src.quantum_kernels import (
    compute_fqk_kernel_matrix,
    compute_pqk_kernel_matrix,
)
from src.classical_kernels import compute_rbf_kernel, compute_rbf_kernel_cv
from src.data_loader import load_modality
from src.classifiers import train_precomputed_svm, evaluate_classifier
from src.kernel_target_alignment import compute_centered_kta
from src.utils import (
    save_results,
    setup_logging,
    Timer,
    ensure_dir,
    compute_full_metrics,
)

logger = setup_logging(
    "physics_cv_f1_v2",
    log_file=os.path.join(config.RESULTS_DIR, "reviewer_fixes", "physics_cv_f1_v2.log"),
)

RESULTS_DIR = ensure_dir(os.path.join(config.RESULTS_DIR, "reviewer_fixes"))
PHYS16_PATH = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")


def load_data():
    data = np.load(PHYS16_PATH)
    X_all = data["X_train"][: config.SUBSAMPLE_TRAIN]
    X_all_raw = data["X_train_raw"][: config.SUBSAMPLE_TRAIN]
    y_all = data["y_train"][: config.SUBSAMPLE_TRAIN]
    X_test = data["X_test"][: config.SUBSAMPLE_TEST]
    y_test = data["y_test"][: config.SUBSAMPLE_TEST]
    return X_all, X_all_raw, y_all, X_test, y_test


def run_cv():
    results_file = os.path.join(RESULTS_DIR, "physics_cv_f1_v2.json")
    checkpoint_file = results_file.replace(".json", "_checkpoint.json")
    if os.path.exists(results_file):
        logger.info(f"[SKIP] Results exist: {results_file}")
        return

    np.random.seed(config.RANDOM_SEED)
    X_all, X_all_raw, y_all, X_test, y_test = load_data()

    skf = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.CV_SEED
    )
    folds = list(skf.split(X_all, y_all))

    # Resume from checkpoint
    fold_results = {}
    start_fold = 0
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file) as f:
                ckpt = json.load(f)
            fold_results = ckpt.get("fold_results", {})
            start_fold = ckpt.get("completed", 0)
            logger.info(
                f"[RESUME] Loaded checkpoint: {start_fold}/{config.CV_FOLDS} folds complete"
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[RESUME] Checkpoint corrupt ({e}), starting from fold 0")

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if fold_idx < start_fold:
            logger.info(f"  [SKIP] Fold {fold_idx + 1} already in checkpoint")
            continue
        X_tr = X_all[train_idx]
        X_val = X_all[val_idx]
        y_tr = y_all[train_idx]
        y_val = y_all[val_idx]

        fold_dir = ensure_dir(
            os.path.join(RESULTS_DIR, f"physics_cv_f1_v2_fold{fold_idx}")
        )
        fr = {}

        # FQK
        with Timer(f"FQK fold{fold_idx}", logger=logger):
            K_fqk_tr = compute_fqk_kernel_matrix(
                X_tr, save_path=os.path.join(fold_dir, "K_fqk.npy")
            )
            K_fqk_val = compute_fqk_kernel_matrix(
                X_tr, X2=X_val, save_path=os.path.join(fold_dir, "K_fqk_val.npy")
            )
        clf = train_precomputed_svm(K_fqk_tr, y_tr)
        m = evaluate_classifier(clf, K_fqk_val, y_val, f"FQK fold{fold_idx}")
        fr["FQK"] = {
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
            "kta": float(compute_centered_kta(K_fqk_tr, y_tr, class_weighted=True)),
        }

        # PQK
        with Timer(f"PQK fold{fold_idx}", logger=logger):
            K_pqk_tr = compute_pqk_kernel_matrix(
                X_tr,
                save_path=os.path.join(fold_dir, "K_pqk.npy"),
                bloch_save_path=os.path.join(fold_dir, "bloch.npy"),
            )
            K_pqk_val = compute_pqk_kernel_matrix(
                X_tr,
                X2=X_val,
                save_path=os.path.join(fold_dir, "K_pqk_val.npy"),
                bloch_save_path=os.path.join(fold_dir, "bloch.npy"),
            )
        clf = train_precomputed_svm(K_pqk_tr, y_tr)
        m = evaluate_classifier(clf, K_pqk_val, y_val, f"PQK fold{fold_idx}")
        fr["PQK"] = {
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
            "kta": float(compute_centered_kta(K_pqk_tr, y_tr, class_weighted=True)),
        }

        # RBF-CV
        K_rbf_cv_tr, K_rbf_cv_val, best_gamma, _ = compute_rbf_kernel_cv(
            X_tr,
            y_tr,
            X_val,
            save_path_train=os.path.join(fold_dir, "K_rbf_cv.npy"),
            save_path_test=os.path.join(fold_dir, "K_rbf_cv_val.npy"),
        )
        clf = train_precomputed_svm(K_rbf_cv_tr, y_tr)
        m = evaluate_classifier(clf, K_rbf_cv_val, y_val, f"RBF-CV fold{fold_idx}")
        fr["RBF-CV"] = {
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
            "kta": float(compute_centered_kta(K_rbf_cv_tr, y_tr, class_weighted=True)),
            "best_gamma": float(best_gamma) if best_gamma else None,
        }

        # RBF-Tuned (GridSearchCV)
        param_grid = {
            "C": [0.1, 1.0, 10.0, 100.0],
            "gamma": ["scale", "auto", 0.01, 0.1],
        }
        inner_cv = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=config.CV_SEED + fold_idx
        )
        grid = GridSearchCV(
            SVC(kernel="rbf", class_weight="balanced", random_state=config.RANDOM_SEED),
            param_grid,
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_tr, y_tr)
        y_pred = grid.predict(X_val)
        m = compute_full_metrics(y_val, y_pred)
        fr["RBF-Tuned"] = {
            "macro_f1": m["macro_f1"],
            "accuracy": m["accuracy"],
            "best_params": grid.best_params_,
            "cv_score": float(grid.best_score_),
        }

        # RandomForest
        rf = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_val)
        m = compute_full_metrics(y_val, y_pred)
        fr["RandomForest"] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}

        fold_results[f"fold_{fold_idx}"] = fr
        logger.info(
            f"  Fold {fold_idx} complete: FQK={fr['FQK']['macro_f1']:.4f}, "
            f"PQK={fr['PQK']['macro_f1']:.4f}, RBF-CV={fr['RBF-CV']['macro_f1']:.4f}"
        )

        # Checkpoint
        ckpt = {"fold_results": fold_results, "completed": fold_idx + 1}
        with open(checkpoint_file, "w") as f:
            json.dump(ckpt, f, indent=2)
        logger.info(f"  [CHECKPOINT] Saved after fold {fold_idx + 1}")

    # Aggregate
    methods = ["FQK", "PQK", "RBF-CV", "RBF-Tuned", "RandomForest"]
    summary = {}
    for method in methods:
        f1s = [
            fold_results[fk][method]["macro_f1"]
            for fk in fold_results
            if isinstance(
                fold_results[fk].get(method, {}).get("macro_f1"), (int, float)
            )
        ]
        summary[method] = {
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)) if len(f1s) > 1 else None,
            "n_folds": len(f1s),
        }

    logger.info(f"\n{'=' * 60}")
    logger.info("  PHYSICS CV F1 V2 SUMMARY")
    logger.info(f"{'=' * 60}")
    for method in methods:
        s = summary[method]
        logger.info(
            f"  {method:<15s}: {s['macro_f1_mean']:.4f} ± {s['macro_f1_std']:.4f}"
        )

    save_results(
        {
            "summary": summary,
            "fold_results": fold_results,
            "n_folds": config.CV_FOLDS,
            "n_train": config.SUBSAMPLE_TRAIN,
        },
        results_file,
    )


if __name__ == "__main__":
    run_cv()
