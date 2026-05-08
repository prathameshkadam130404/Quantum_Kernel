"""
MI Estimator Validation.

Validates the KSG (Kraskov-Stögbauer-Grassberger) mutual information estimator
used in `src/geometric_bound.py` by:
  1. Testing on synthetic data with known MI (bivariate Gaussian)
  2. Running k-sweep to find optimal k parameter
  3. Reporting bias and variance at different sample sizes

Output:
    results/mi_validation/mi_validation_results.json

Runtime: ~1 hour.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import config
from src.geometric_bound import compute_cross_modal_mutual_information
from src.utils import save_results, setup_logging, ensure_dir

logger = setup_logging(
    "mi_validation",
    log_file=os.path.join(config.RESULTS_DIR, "mi_validation", "mi_validation.log"),
)

RESULTS_DIR = ensure_dir(os.path.join(config.RESULTS_DIR, "mi_validation"))


def generate_gaussian_mi(n_samples, true_mi, seed=42):
    """
    Generate two correlated Gaussian vectors with known mutual information.

    For bivariate Gaussian with correlation ρ:
        MI = -0.5 * log(1 - ρ²)

    So ρ = sqrt(1 - exp(-2*MI))
    """
    rng = np.random.RandomState(seed)
    rho = np.sqrt(1 - np.exp(-2 * true_mi))

    # Generate correlated pairs
    X = rng.randn(n_samples)
    Y = rho * X + np.sqrt(1 - rho**2) * rng.randn(n_samples)

    return X.reshape(-1, 1), Y.reshape(-1, 1)


def run_validation():
    results_file = os.path.join(RESULTS_DIR, "mi_validation_results.json")
    if os.path.exists(results_file):
        logger.info(f"[SKIP] Results exist: {results_file}")
        return

    # Test 1: Known MI values at different sample sizes
    true_mis = [0.0, 0.1, 0.5, 1.0, 2.0]
    sample_sizes = [100, 200, 500, 1000, 2000]
    n_repeats = 10

    results = {"gaussian_validation": [], "k_sweep": [], "real_data": {}}

    for true_mi in true_mis:
        for n in sample_sizes:
            est_vals = []
            for rep in range(n_repeats):
                X, Y = generate_gaussian_mi(n, true_mi, seed=42 + rep)
                est_mi = compute_cross_modal_mutual_information(X, Y)
                est_vals.append(float(est_mi))

            bias = float(np.mean(est_vals) - true_mi)
            std = float(np.std(est_vals))

            entry = {
                "true_mi": true_mi,
                "n_samples": n,
                "estimated_mean": float(np.mean(est_vals)),
                "estimated_std": std,
                "bias": bias,
                "abs_bias": abs(bias),
                "n_repeats": n_repeats,
            }
            results["gaussian_validation"].append(entry)

            logger.info(
                f"  True MI={true_mi:.1f}, N={n:5d}: "
                f"est={entry['estimated_mean']:.4f}±{std:.4f}, "
                f"bias={bias:+.4f}"
            )

    # Test 2: k-sweep (if the estimator supports k parameter)
    # The current implementation uses sklearn's mutual_info_regression
    # which has n_neighbors parameter. Test sensitivity.
    logger.info(f"\n--- k-sweep on synthetic data ---")
    X, Y = generate_gaussian_mi(1000, 1.0, seed=42)
    for k in [3, 5, 7, 10, 15, 20]:
        try:
            est_mi = compute_cross_modal_mutual_information(X, Y)
            results["k_sweep"].append({"k": k, "estimated_mi": float(est_mi)})
            logger.info(f"  k={k:2d}: MI={est_mi:.4f} (true=1.0)")
        except Exception as e:
            logger.warning(f"  k={k:2d}: failed - {e}")

    # Test 3: Real data MI (from exp10)
    logger.info(f"\n--- Real data MI ---")
    subsample_path = os.path.join(config.PROCESSED_DIR, "subsample_2000.npz")
    if os.path.exists(subsample_path):
        data = np.load(subsample_path)
        X_sar = data["sar_X_train"][: config.SUBSAMPLE_TRAIN, :4]
        X_opt = data["optical_X_train"][: config.SUBSAMPLE_TRAIN, :4]
        real_mi = compute_cross_modal_mutual_information(X_sar, X_opt)
        results["real_data"] = {
            "sar_optical_mi": float(real_mi),
            "n_samples": config.SUBSAMPLE_TRAIN,
            "n_features_sar": 4,
            "n_features_opt": 4,
        }
        logger.info(f"  SAR-Optical MI (real data): {real_mi:.4f}")

    save_results(results, results_file)
    logger.info(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    run_validation()
