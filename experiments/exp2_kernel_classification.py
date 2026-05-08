"""
Experiment 2: Kernel Classification.

Full classification comparison: quantum kernel SVMs, classical SVMs, CNN baseline.
Quantum kernels: FQK, PQK, TFK (trained), CM-FQK (cross-modal).
Classical baselines: untuned SVMs + tuned RBF-SVM (GridSearchCV), Random Forest,
Gradient Boosting. Tuned baselines added to close reviewer fairness objection.
All 17 LCZ classes. Macro-F1 is PRIMARY metric (Rule I3).

Phase 3 of the pipeline. Expected runtime: ~1 hour.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import config
from src.quantum_kernels import (
    compute_fqk_kernel_matrix,
    compute_pqk_kernel_matrix,
)
from src.classical_kernels import compute_rbf_kernel
from src.data_loader import load_modality, load_subsample
from src.classifiers import (
    train_precomputed_svm, evaluate_classifier,
    train_classical_svms, mcnemars_test,
)
from src.utils import (
    save_results, setup_logging, Timer, check_and_skip, ensure_dir,
    plot_confusion_matrix, plot_per_class_f1, plot_per_class_f1_comparison,
    compute_full_metrics, validate_test_kernel_shape,
)

logger = setup_logging("exp2", log_file=os.path.join(config.RESULTS_DIR, "exp2.log"))


def load_precomputed_kernel(train_path, test_path, name):
    """Load a precomputed kernel from disk. Raises clear error if missing."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"{name} train kernel not found: {train_path}\n"
            f"Run the kernel computation script before exp2."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"{name} test kernel not found: {test_path}\n"
            f"Run the kernel computation script before exp2."
        )
    K_train = np.load(train_path)
    K_test  = np.load(test_path)
    logger.info(f"  Loaded {name}: train={K_train.shape}, test={K_test.shape}")
    return K_train, K_test


def run_experiment():
    """Full kernel classification experiment."""
    results_dir = ensure_dir(os.path.join(config.RESULTS_DIR, "classification"))
    results_file = os.path.join(results_dir, "exp2_results.json")

    if check_and_skip(results_file, logger):
        return

    np.random.seed(config.RANDOM_SEED)

    # Load subsample
    with Timer("Loading data", logger=logger):
        subsample = load_subsample()
        X_train = subsample["fused_X_train"]
        X_test = subsample["fused_X_test"]
        y_train = subsample["y_train"]
        y_test = subsample["y_test"]

    all_results = {}

    # ---- Quantum kernel SVMs ----
    logger.info("\n--- Quantum Kernel SVMs ---")
    # Load FQK and PQK from exp1's cached output — do NOT recompute.
    # compute_fqk_kernel_matrix and compute_pqk_kernel_matrix check
    # save_path first and return the cached file if it exists.
    # Pointing save_path at exp1's output directory means exp2 loads
    # in seconds instead of recomputing over 20 hours.
    fused_gd_dir = os.path.join(config.RESULTS_DIR,
                                 "geometric_difference", "fused")

    K_fqk = compute_fqk_kernel_matrix(X_train,
        save_path=os.path.join(fused_gd_dir, "K_fqk_train.npy"))
    K_fqk_test = compute_fqk_kernel_matrix(X_train, X2=X_test,
        save_path=os.path.join(fused_gd_dir, "K_fqk_test.npy"))

    K_pqk = compute_pqk_kernel_matrix(X_train,
        save_path=os.path.join(fused_gd_dir, "K_pqk_train.npy"),
        bloch_save_path=os.path.join(fused_gd_dir, "bloch_train.npy"))
    K_pqk_test = compute_pqk_kernel_matrix(X_train, X2=X_test,
        save_path=os.path.join(fused_gd_dir, "K_pqk_test.npy"),
        bloch_save_path=os.path.join(fused_gd_dir, "bloch_train.npy"))
    validate_test_kernel_shape(K_pqk_test, X_train, X_test, "PQK")


    for name, K_tr, K_te in [("FQK-SVM", K_fqk, K_fqk_test), ("PQK-SVM", K_pqk, K_pqk_test)]:
        clf = train_precomputed_svm(K_tr, y_train)
        metrics = evaluate_classifier(clf, K_te, y_test, name)
        y_pred = metrics.pop("predictions")
        all_results[name] = metrics

        plot_confusion_matrix(y_test, y_pred, os.path.join(results_dir, f"cm_{name}"), title=f"Confusion Matrix — {name}")
        plot_per_class_f1(y_test, y_pred, os.path.join(results_dir, f"f1_{name}"), title=f"Per-Class F1 — {name}")

    # ---- TFK and CM_FQK ----
    logger.info("\n--- Trained & Cross-Modal Quantum Kernel SVMs ---")

    fused_gd_dir = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
    tfk_dir      = os.path.join(fused_gd_dir, "tfk")
    cmfqk_dir    = fused_gd_dir

    K_tfk, K_tfk_test = load_precomputed_kernel(
        os.path.join(tfk_dir,   "K_tfk_train.npy"),
        os.path.join(tfk_dir,   "K_tfk_test.npy"),
        "TFK",
    )
    K_cm_fqk, K_cm_fqk_test = load_precomputed_kernel(
        os.path.join(cmfqk_dir, "K_cm_fqk_train.npy"),
        os.path.join(cmfqk_dir, "K_cm_fqk_test.npy"),
        "CM-FQK",
    )

    for name, K_tr, K_te in [("TFK-SVM", K_tfk, K_tfk_test), ("CM-FQK-SVM", K_cm_fqk, K_cm_fqk_test)]:
        clf = train_precomputed_svm(K_tr, y_train)
        metrics = evaluate_classifier(clf, K_te, y_test, name)
        y_pred = metrics.pop("predictions")
        all_results[name] = metrics

        plot_confusion_matrix(y_test, y_pred, os.path.join(results_dir, f"cm_{name}"), title=f"Confusion Matrix — {name}")
        plot_per_class_f1(y_test, y_pred, os.path.join(results_dir, f"f1_{name}"), title=f"Per-Class F1 — {name}")

    # ---- Classical SVMs ----
    logger.info("\n--- Classical SVMs ---")
    classical_results = train_classical_svms(X_train, y_train, X_test, y_test)
    for name, metrics in classical_results.items():
        y_pred = metrics.pop("predictions")
        all_results[name] = metrics
        plot_confusion_matrix(y_test, y_pred, os.path.join(results_dir, f"cm_{name}"), title=f"Confusion Matrix — {name}")

    # ---- Tuned Classical Baselines ----
    # Added to close the "unfair classical comparison" reviewer objection.
    # All models operate on raw PCA features (X_train, X_test).
    # Results are reported honestly regardless of direction.
    logger.info("\n--- Tuned Classical Baselines ---")
    tuned_dir = ensure_dir(os.path.join(results_dir, "tuned_classical"))
    tuned_results_path = os.path.join(tuned_dir, "tuned_classical_results.json")

    if os.path.exists(tuned_results_path):
        logger.info("[SKIP] Tuned classical results already exist. Loading...")
        import json
        with open(tuned_results_path) as _f:
            tuned_classical = json.load(_f)
        for name, metrics in tuned_classical.items():
            all_results[name] = metrics
            logger.info(f"  Loaded {name}: macro_f1={metrics.get('macro_f1', 'N/A'):.4f}")
    else:
        tuned_classical = {}

        # 1. Tuned RBF-SVM via GridSearchCV
        logger.info("\n  Tuned RBF-SVM (GridSearchCV 5-fold)...")
        from sklearn.svm import SVC
        param_grid = {
            'C':     [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto', 0.01, 0.1],
        }
        grid_search = GridSearchCV(
            SVC(kernel='rbf', class_weight='balanced',
                random_state=config.RANDOM_SEED),
            param_grid,
            scoring='f1_macro',
            cv=5,
            n_jobs=-1,
            verbose=1,
            refit=True,
        )
        with Timer("GridSearchCV RBF-SVM", logger=logger):
            grid_search.fit(X_train, y_train)

        best_rbf = grid_search.best_estimator_
        y_pred_rbf_tuned = best_rbf.predict(X_test)
        rbf_tuned_metrics = compute_full_metrics(y_test, y_pred_rbf_tuned)
        rbf_tuned_metrics['best_params'] = grid_search.best_params_
        rbf_tuned_metrics['cv_best_score'] = float(grid_search.best_score_)
        tuned_classical['RBF-SVM-Tuned'] = rbf_tuned_metrics
        all_results['RBF-SVM-Tuned'] = rbf_tuned_metrics

        logger.info(f"  Best params : {grid_search.best_params_}")
        logger.info(f"  CV score    : {grid_search.best_score_:.4f}")
        logger.info(f"  Test macro-F1: {rbf_tuned_metrics['macro_f1']:.4f}")

        plot_confusion_matrix(
            y_test, y_pred_rbf_tuned,
            os.path.join(results_dir, "cm_RBF-SVM-Tuned"),
            title="Confusion Matrix — RBF-SVM (Tuned)",
        )

        # 2. Random Forest
        logger.info("\n  Random Forest (500 trees, balanced)...")
        rf = RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
        with Timer("Random Forest", logger=logger):
            rf.fit(X_train, y_train)

        y_pred_rf = rf.predict(X_test)
        rf_metrics = compute_full_metrics(y_test, y_pred_rf)
        rf_metrics['n_estimators'] = 500
        tuned_classical['RandomForest'] = rf_metrics
        all_results['RandomForest'] = rf_metrics

        logger.info(f"  RF macro-F1: {rf_metrics['macro_f1']:.4f}")

        plot_confusion_matrix(
            y_test, y_pred_rf,
            os.path.join(results_dir, "cm_RandomForest"),
            title="Confusion Matrix — Random Forest",
        )

        # 3. Gradient Boosting
        logger.info("\n  Gradient Boosting (200 estimators)...")
        gb = GradientBoostingClassifier(
            n_estimators=200,
            random_state=config.RANDOM_SEED,
            verbose=0,
        )
        with Timer("Gradient Boosting", logger=logger):
            gb.fit(X_train, y_train)

        y_pred_gb = gb.predict(X_test)
        gb_metrics = compute_full_metrics(y_test, y_pred_gb)
        gb_metrics['n_estimators'] = 200
        tuned_classical['GradientBoosting'] = gb_metrics
        all_results['GradientBoosting'] = gb_metrics

        logger.info(f"  GB macro-F1: {gb_metrics['macro_f1']:.4f}")

        plot_confusion_matrix(
            y_test, y_pred_gb,
            os.path.join(results_dir, "cm_GradientBoosting"),
            title="Confusion Matrix — Gradient Boosting",
        )

        # Save tuned classical results independently for fast reload
        import json
        with open(tuned_results_path, 'w') as _f:
            # Strip non-serializable entries before saving
            serializable = {}
            for k, v in tuned_classical.items():
                serializable[k] = {
                    sk: sv for sk, sv in v.items()
                    if isinstance(sv, (int, float, str, bool, list, dict))
                    and sk != 'predictions'
                }
            json.dump(serializable, _f, indent=2)
        logger.info(f"\n  Saved tuned classical results: {tuned_results_path}")

    # ---- McNemar's test ----
    logger.info("\n--- McNemar's Test ---")
    best_q = max(
        ["FQK-SVM", "PQK-SVM", "TFK-SVM", "CM-FQK-SVM"],
        key=lambda x: all_results[x]["macro_f1"],
    )
    # Include tuned classical models in best_c selection
    all_classical_names = (
        list(classical_results.keys())
        + ['RBF-SVM-Tuned', 'RandomForest', 'GradientBoosting']
    )
    # Filter to only those that completed successfully
    all_classical_names = [
        n for n in all_classical_names
        if n in all_results and isinstance(all_results[n].get('macro_f1'), float)
    ]
    best_c = max(all_classical_names, key=lambda x: all_results[x]["macro_f1"])

    # Re-compute predictions for McNemar
    kernel_map = {
        "FQK-SVM":    (K_fqk,    K_fqk_test),
        "PQK-SVM":    (K_pqk,    K_pqk_test),
        "TFK-SVM":    (K_tfk,    K_tfk_test),
        "CM-FQK-SVM": (K_cm_fqk, K_cm_fqk_test),
    }
    K_tr_q, K_te_q = kernel_map[best_q]
    clf_q = train_precomputed_svm(K_tr_q, y_train)
    y_pred_q = clf_q.predict(K_te_q)

    from sklearn.svm import SVC
    # Recompute predictions for best classical model
    if best_c == 'RBF-SVM-Tuned':
        # Refit with best params from grid search
        _bp = all_results['RBF-SVM-Tuned'].get('best_params', {})
        clf_c = SVC(kernel='rbf', class_weight='balanced',
                    random_state=config.RANDOM_SEED,
                    C=_bp.get('C', 1.0), gamma=_bp.get('gamma', 'scale'))
        clf_c.fit(X_train, y_train)
        y_pred_c = clf_c.predict(X_test)
    elif best_c == 'RandomForest':
        clf_c = RandomForestClassifier(
            n_estimators=500, class_weight='balanced',
            random_state=config.RANDOM_SEED, n_jobs=-1)
        clf_c.fit(X_train, y_train)
        y_pred_c = clf_c.predict(X_test)
    elif best_c == 'GradientBoosting':
        clf_c = GradientBoostingClassifier(
            n_estimators=200, random_state=config.RANDOM_SEED)
        clf_c.fit(X_train, y_train)
        y_pred_c = clf_c.predict(X_test)
    else:
        # Original fallback for untuned classical SVMs
        clf_c = SVC(kernel="rbf", class_weight="balanced",
                    random_state=config.RANDOM_SEED)
        clf_c.fit(X_train, y_train)
        y_pred_c = clf_c.predict(X_test)

    mcnemar = mcnemars_test(y_test, y_pred_q, y_pred_c, best_q, best_c)
    all_results["mcnemars_test"] = mcnemar

    # ---- Per-class F1 comparison ----
    plot_per_class_f1_comparison(
        y_test, y_pred_q, y_pred_c,
        os.path.join(results_dir, "fig8_per_class_f1_comparison"),
        quantum_label=best_q, classical_label=best_c,
        title=f"Per-Class F1: Best Quantum ({best_q}) vs Best Classical ({best_c})",
    )

    # ---- Summary table ----
    table_rows = []
    for name, metrics in all_results.items():
        if name == "mcnemars_test":
            continue
        table_rows.append({
            "Model": name,
            "Macro-F1": metrics.get("macro_f1", ""),
            "Weighted-F1": metrics.get("weighted_f1", ""),
            "Accuracy": metrics.get("accuracy", ""),
            "Cohen κ": metrics.get("cohen_kappa", ""),
        })

    df = pd.DataFrame(table_rows).sort_values("Macro-F1", ascending=False)
    df.to_csv(os.path.join(results_dir, "classification_results.csv"), index=False)
    logger.info(f"\n{df.to_string()}")

    save_results(all_results, results_file)
    logger.info(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    run_experiment()
