"""
Bootstrap Confidence Intervals for Classification Results.

Computes bootstrap CIs for all Table II (PCA regime) and Table V (physics regime)
classification results using the pre-computed kernel matrices.

Uses pre-computed kernels — no quantum circuit recomputation needed.

Output:
    results/bootstrap_ci/bootstrap_ci_results.json
    results/bootstrap_ci/table_ci.csv

Kaggle GPU estimate: 2-3 hours (CPU-bound, SVM training).
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import config
from src.utils import save_results, setup_logging, ensure_dir

logger = setup_logging(
    "bootstrap_ci",
    log_file=os.path.join(config.RESULTS_DIR, "bootstrap_ci", "bootstrap_ci.log"),
)

RESULTS_DIR = ensure_dir(os.path.join(config.RESULTS_DIR, "bootstrap_ci"))

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42


def load_pca_results():
    """Load pre-computed PCA regime results."""
    exp1_path = os.path.join(
        config.RESULTS_DIR, "geometric_difference", "exp1_results.json"
    )
    if not os.path.exists(exp1_path):
        logger.warning(f"exp1_results.json not found: {exp1_path}")
        return None

    with open(exp1_path) as f:
        return json.load(f)


def load_physics_results():
    """Load pre-computed physics regime results."""
    phys_path = os.path.join(config.RESULTS_DIR, "physics", "exp1_physics_results.json")
    if not os.path.exists(phys_path):
        logger.warning(f"exp1_physics_results.json not found: {phys_path}")
        return None

    with open(phys_path) as f:
        return json.load(f)


def load_kernel_matrices(regime):
    """Load pre-computed kernel matrices for bootstrap resampling."""
    kernels = {}

    if regime == "pca":
        base = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
        kernel_map = {
            "FQK": ("K_fqk_train.npy", "K_fqk_test.npy"),
            "PQK": ("K_pqk_train.npy", "K_pqk_test.npy"),
            "RBF": ("K_rbf_train.npy", "K_rbf_test.npy"),
        }
        tfk_dir = os.path.join(base, "tfk")
        if os.path.exists(os.path.join(tfk_dir, "K_tfk_train.npy")):
            kernel_map["TFK"] = (
                os.path.join("tfk", "K_tfk_train.npy"),
                os.path.join("tfk", "K_tfk_test.npy"),
            )
        if os.path.exists(os.path.join(base, "K_cm_fqk_train.npy")):
            kernel_map["CM-FQK"] = ("K_cm_fqk_train.npy", "K_cm_fqk_test.npy")

        for name, (tr_name, te_name) in kernel_map.items():
            tr_path = (
                os.path.join(base, tr_name)
                if not tr_name.startswith("tfk")
                else os.path.join(base, tr_name)
            )
            te_path = (
                os.path.join(base, te_name)
                if not te_name.startswith("tfk")
                else os.path.join(base, te_name)
            )
            if os.path.exists(tr_path) and os.path.exists(te_path):
                kernels[name] = (np.load(tr_path), np.load(te_path))

    elif regime == "physics":
        base = os.path.join(config.RESULTS_DIR, "physics", "fused")
        kernel_map = {
            "FQK": ("K_fqk_physics_train.npy", "K_fqk_physics_test.npy"),
            "PQK": ("K_pqk_physics_train.npy", "K_pqk_physics_test.npy"),
            "RBF": ("K_rbf_physics_train.npy", None),  # RBF computed on-the-fly
            "AGPQK": ("K_agpqk_physics_train.npy", "K_agpqk_physics_test.npy"),
        }
        for name, (tr_name, te_name) in kernel_map.items():
            tr_path = os.path.join(base, tr_name)
            if os.path.exists(tr_path):
                K_tr = np.load(tr_path)
                K_te = (
                    np.load(os.path.join(base, te_name))
                    if te_name and os.path.exists(os.path.join(base, te_name))
                    else None
                )
                kernels[name] = (K_tr, K_te)

    return kernels


def load_labels(regime):
    """Load train/test labels."""
    if regime == "pca":
        subsample_path = os.path.join(config.PROCESSED_DIR, "subsample_2000.npz")
        data = np.load(subsample_path)
        return data["y_train"], data["y_test"]
    else:
        phys16_path = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
        data = np.load(phys16_path)
        return data["y_train"][: config.SUBSAMPLE_TRAIN], data["y_test"][
            : config.SUBSAMPLE_TEST
        ]


def bootstrap_kernel_svm(
    K_tr, K_te, y_train, y_test, n_bootstrap=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
):
    """
    Bootstrap CI for precomputed kernel SVM.

    Resamples test set with replacement, retrains SVM on full training set,
    and computes metric distribution.
    """
    from src.classifiers import train_precomputed_svm
    from sklearn.metrics import f1_score, accuracy_score

    rng = np.random.RandomState(seed)
    n_test = len(y_test)
    f1_vals = []
    acc_vals = []

    clf = train_precomputed_svm(K_tr, y_train)

    for i in range(n_bootstrap):
        # Resample test indices with replacement
        idx = rng.choice(n_test, size=n_test, replace=True)
        K_te_boot = K_te[idx]
        y_boot = y_test[idx]

        y_pred = clf.predict(K_te_boot)
        f1_vals.append(f1_score(y_boot, y_pred, average="macro"))
        acc_vals.append(accuracy_score(y_boot, y_pred))

    return {
        "macro_f1_mean": float(np.mean(f1_vals)),
        "macro_f1_std": float(np.std(f1_vals)),
        "macro_f1_ci_lower": float(np.percentile(f1_vals, 2.5)),
        "macro_f1_ci_upper": float(np.percentile(f1_vals, 97.5)),
        "accuracy_mean": float(np.mean(acc_vals)),
        "accuracy_std": float(np.std(acc_vals)),
        "accuracy_ci_lower": float(np.percentile(acc_vals, 2.5)),
        "accuracy_ci_upper": float(np.percentile(acc_vals, 97.5)),
        "n_bootstrap": n_bootstrap,
    }


def run_analysis():
    results_file = os.path.join(RESULTS_DIR, "bootstrap_ci_results.json")
    checkpoint_file = results_file.replace(".json", "_checkpoint.json")
    if os.path.exists(results_file):
        logger.info(f"[SKIP] Results exist: {results_file}")
        return

    # Resume from checkpoint
    all_results = {}
    completed_regimes = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file) as f:
                ckpt = json.load(f)
            all_results = ckpt.get("regime_results", {})
            completed_regimes = set(all_results.keys())
            logger.info(
                f"[RESUME] Loaded checkpoint: {len(completed_regimes)} regimes complete: {sorted(completed_regimes)}"
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[RESUME] Checkpoint corrupt ({e}), starting from scratch")

    for regime in ["pca", "physics"]:
        if regime in completed_regimes:
            logger.info(f"  [SKIP] {regime.upper()} regime already in checkpoint")
            continue
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  BOOTSTRAP CI: {regime.upper()} REGIME")
        logger.info(f"{'=' * 60}")

        kernels = load_kernel_matrices(regime)
        y_train, y_test = load_labels(regime)

        regime_results = {}
        for name, (K_tr, K_te) in kernels.items():
            if K_te is None:
                logger.info(f"  [SKIP] {name}: no test kernel")
                continue

            logger.info(f"  {name}: bootstrapping ({K_tr.shape}, {K_te.shape})...")
            ci = bootstrap_kernel_svm(K_tr, K_te, y_train, y_test)
            regime_results[name] = ci

            logger.info(
                f"    Macro-F1: {ci['macro_f1_mean']:.4f} "
                f"[{ci['macro_f1_ci_lower']:.4f}, {ci['macro_f1_ci_upper']:.4f}]"
            )
            logger.info(
                f"    Accuracy: {ci['accuracy_mean']:.4f} "
                f"[{ci['accuracy_ci_lower']:.4f}, {ci['accuracy_ci_upper']:.4f}]"
            )

        all_results[regime] = regime_results

        # Checkpoint after each regime
        ckpt = {"regime_results": all_results}
        with open(checkpoint_file, "w") as f:
            json.dump(ckpt, f, indent=2)
        logger.info(f"  [CHECKPOINT] {regime.upper()} regime complete")

    # Summary table
    logger.info(f"\n{'=' * 60}")
    logger.info("  BOOTSTRAP CI SUMMARY (95% CI, 1000 resamples)")
    logger.info(f"{'=' * 60}")

    for regime in ["pca", "physics"]:
        if regime not in all_results:
            continue
        logger.info(f"\n  --- {regime.upper()} ---")
        logger.info(f"{'Method':<15s} {'Macro-F1':>20s} {'Accuracy':>20s}")
        logger.info("-" * 58)
        for name, ci in all_results[regime].items():
            f1_str = f"{ci['macro_f1_mean']:.4f} [{ci['macro_f1_ci_lower']:.4f}, {ci['macro_f1_ci_upper']:.4f}]"
            acc_str = f"{ci['accuracy_mean']:.4f} [{ci['accuracy_ci_lower']:.4f}, {ci['accuracy_ci_upper']:.4f}]"
            logger.info(f"{name:<15s} {f1_str:>20s} {acc_str:>20s}")

    save_results(all_results, results_file)
    logger.info(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    run_analysis()
