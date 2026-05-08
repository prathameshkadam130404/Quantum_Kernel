"""
Geometric bound and cross-modal mutual information analysis.

Implements:
    - Cross-modal mutual information: Pairwise MI between SAR and Optical
      PCA features using sklearn's mutual_info_regression.
    - Tensor-product kernel (classical baseline that treats modalities independently).
    - Geometric amplification analysis: scatter plot g vs MI, linear regression,
      Pearson + Spearman correlations with 95% CI.

Empirical Proposition (NOT a theorem):
    When ZZFeatureMap encodes concatenated x = [x_SAR; x_Opt], cross-modal
    ZZ terms create entanglement encoding correlations between modalities.
    We EMPIRICALLY OBSERVE g(K_q, K_c⊗) correlates positively with
    I_empirical(X^(1); X^(2)).

Language: "We empirically observe" and "our results suggest," NOT "we prove."
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


def compute_cross_modal_mutual_information(
    X_sar_pca: np.ndarray,
    X_opt_pca: np.ndarray,
    y: Optional[np.ndarray] = None,
    n_neighbors: int = 5,
) -> float:
    """
    Compute empirical mutual information between SAR and Optical PCA features.

    Uses sklearn's mutual_info_regression for continuous features. Computes
    pairwise MI between each SAR feature and each Optical feature, then
    returns the mean as a scalar measure of cross-modal dependence.

    Args:
        X_sar_pca: SAR PCA features, shape (n, d_sar).
        X_opt_pca: Optical PCA features, shape (n, d_opt).
        y: Optional labels (not used for MI computation, but logged).
        n_neighbors: Number of neighbors for MI estimation.

    Returns:
        float: I_empirical = mean of pairwise MI values. Always ≥ 0.
    """
    from sklearn.feature_selection import mutual_info_regression

    d_sar = X_sar_pca.shape[1]
    d_opt = X_opt_pca.shape[1]

    mi_matrix = np.zeros((d_sar, d_opt))

    for i in range(d_sar):
        for j in range(d_opt):
            mi_val = mutual_info_regression(
                X_sar_pca[:, i].reshape(-1, 1),
                X_opt_pca[:, j],
                n_neighbors=n_neighbors,
                random_state=config.RANDOM_SEED,
            )[0]
            mi_matrix[i, j] = max(mi_val, 0.0)  # MI ≥ 0

    i_empirical = float(np.mean(mi_matrix))

    logger.info(
        f"Cross-modal MI: I_empirical = {i_empirical:.6f}, "
        f"MI matrix shape = {mi_matrix.shape}, "
        f"range = [{mi_matrix.min():.6f}, {mi_matrix.max():.6f}]"
    )

    return i_empirical


def compute_mi_matrix(
    X_sar_pca: np.ndarray,
    X_opt_pca: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    """
    Compute the full pairwise MI matrix between SAR and Optical features.

    Args:
        X_sar_pca: SAR features, shape (n, d_sar).
        X_opt_pca: Optical features, shape (n, d_opt).
        n_neighbors: K-NN neighbors for MI estimation.

    Returns:
        np.ndarray: MI matrix, shape (d_sar, d_opt).
    """
    from sklearn.feature_selection import mutual_info_regression

    d_sar = X_sar_pca.shape[1]
    d_opt = X_opt_pca.shape[1]
    mi_matrix = np.zeros((d_sar, d_opt))

    for i in range(d_sar):
        for j in range(d_opt):
            mi_matrix[i, j] = max(
                mutual_info_regression(
                    X_sar_pca[:, i:i+1],
                    X_opt_pca[:, j],
                    n_neighbors=n_neighbors,
                    random_state=config.RANDOM_SEED,
                )[0],
                0.0,
            )

    return mi_matrix


def compute_tensor_product_kernel(
    X: np.ndarray,
    n_sar_features: int = 4,
    n_opt_features: int = 4,
    gamma: float = 1.0,
) -> np.ndarray:
    """
    Compute tensor-product classical kernel.

    K_c⊗ = K_rbf(X[:,:n_sar]) ⊙ K_rbf(X[:,n_sar:])

    The Hadamard product of per-modality RBF kernels treats modalities as
    INDEPENDENT. By construction, it cannot capture cross-modal correlations.

    Args:
        X: Fused data, shape (n, n_sar + n_opt).
        n_sar_features: Number of SAR features (first columns).
        n_opt_features: Number of Optical features.
        gamma: RBF bandwidth.

    Returns:
        np.ndarray: Tensor-product kernel matrix, shape (n, n).
    """
    from sklearn.metrics.pairwise import rbf_kernel

    X_sar = X[:, :n_sar_features]
    X_opt = X[:, n_sar_features:n_sar_features + n_opt_features]

    gamma_sar = gamma / n_sar_features
    gamma_opt = gamma / n_opt_features

    K_sar = rbf_kernel(X_sar, gamma=gamma_sar)
    K_opt = rbf_kernel(X_opt, gamma=gamma_opt)

    # Hadamard (element-wise) product
    return K_sar * K_opt


def analyze_geometric_amplification(
    g_values: np.ndarray,
    mi_values: np.ndarray,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Analyze relationship between geometric difference g and mutual information.

    Fits linear regression g = β₀ + β₁ * MI. Reports β with 95% CI,
    Pearson and Spearman correlations.

    CRITICAL LANGUAGE: "We empirically observe," NOT "we prove."

    Args:
        g_values: Array of geometric difference values.
        mi_values: Array of mutual information values.
        save_dir: Directory to save analysis results and plot.

    Returns:
        dict: {
            'pearson_r', 'pearson_p',
            'spearman_r', 'spearman_p',
            'slope', 'intercept',
            'slope_95ci', 'r_squared',
        }
    """
    g_arr = np.array(g_values, dtype=float)
    mi_arr = np.array(mi_values, dtype=float)

    # ---- Pearson correlation ----
    pearson_r, pearson_p = stats.pearsonr(mi_arr, g_arr)

    # ---- Spearman correlation ----
    spearman_r, spearman_p = stats.spearmanr(mi_arr, g_arr)

    # ---- Linear regression ----
    slope, intercept, r_value, p_value, std_err = stats.linregress(mi_arr, g_arr)
    r_squared = r_value ** 2

    # 95% CI for slope
    t_crit = stats.t.ppf(0.975, df=len(g_arr) - 2)
    slope_ci = (slope - t_crit * std_err, slope + t_crit * std_err)

    results = {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_95ci": [float(slope_ci[0]), float(slope_ci[1])],
        "r_squared": float(r_squared),
        "std_err": float(std_err),
        "n_points": int(len(g_arr)),
    }

    logger.info(
        f"Geometric amplification analysis:\n"
        f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.2e})\n"
        f"  Spearman r = {spearman_r:.4f} (p = {spearman_p:.2e})\n"
        f"  Slope β₁ = {slope:.4f} ± {std_err:.4f} "
        f"(95% CI: [{slope_ci[0]:.4f}, {slope_ci[1]:.4f}])\n"
        f"  R² = {r_squared:.4f}"
    )

    # ---- Save scatter plot ----
    if save_dir:
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(mi_arr, g_arr, s=60, alpha=0.7, color="steelblue", edgecolors="white")

        # Regression line
        x_fit = np.linspace(mi_arr.min(), mi_arr.max(), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, "r--", linewidth=2,
                label=f"β₁={slope:.3f}, R²={r_squared:.3f}")

        ax.set_xlabel("Cross-Modal Mutual Information (I_empirical)", fontsize=12)
        ax.set_ylabel("Geometric Difference (g)", fontsize=12)
        ax.set_title("Geometric Amplification: g vs MI", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        filepath = os.path.join(save_dir, "g_vs_mi_scatter")
        fig.savefig(filepath + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(filepath + ".pdf", bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved scatter plot: {filepath}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  src/geometric_bound.py — Self-test")
    print("=" * 60)

    np.random.seed(config.RANDOM_SEED)
    n = 100

    # Test MI computation
    X_sar = np.random.randn(n, 4)
    X_opt = np.random.randn(n, 4)

    mi = compute_cross_modal_mutual_information(X_sar, X_opt)
    print(f"\n  MI (independent features) = {mi:.6f} (expected ≈ 0)")
    assert mi >= 0.0

    # Test MI with correlated features
    X_opt_corr = X_sar + np.random.randn(n, 4) * 0.1
    mi_corr = compute_cross_modal_mutual_information(X_sar, X_opt_corr)
    print(f"  MI (correlated features) = {mi_corr:.6f} (expected > 0)")
    assert mi_corr > mi  # Correlated should have higher MI

    # Test tensor-product kernel
    X_fused = np.hstack([X_sar, X_opt])
    K_tp = compute_tensor_product_kernel(X_fused)
    print(f"  Tensor-product kernel: shape={K_tp.shape}")
    assert K_tp.shape == (n, n)
    assert np.allclose(K_tp, K_tp.T)  # Symmetric

    # Test geometric amplification analysis
    g_vals = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    mi_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    results = analyze_geometric_amplification(g_vals, mi_vals)
    print(f"\n  Amplification: r={results['pearson_r']:.4f}, slope={results['slope']:.4f}")
    assert results["pearson_r"] > 0.9  # Strong linear relationship

    print("\n  All self-tests passed.")
