"""
Generate paper figures with updated 5-fold CV results from optimized experiments.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import json

os.makedirs("results/paper_figures", exist_ok=True)

# Load the 5-fold CV results
with open("results/pca_cv/exp1_pca_cv_results.json", "r") as f:
    pca_results = json.load(f)["summary"]

with open("results/physics_cv/exp1_physics_cv_results.json", "r") as f:
    physics_results = json.load(f)["summary"]


# -------------------------------------------------------------
# Figure 7: Feature regime comparison (UPDATED with 5-fold CV)
# -------------------------------------------------------------
def plot_fig7():
    kernels = ["CM-FQK", "FQK", "PQK", "AGPQK"]

    # New 5-fold CV results
    pca_f1 = [
        pca_results["CM-FQK"]["macro_f1_mean"],
        pca_results["FQK"]["macro_f1_mean"],
        pca_results["PQK"]["macro_f1_mean"],
        0.0,
    ]  # AGPQK not in PCA regime

    pca_std = [
        pca_results["CM-FQK"]["macro_f1_std"],
        pca_results["FQK"]["macro_f1_std"],
        pca_results["PQK"]["macro_f1_std"],
        0.0,
    ]

    phys_f1 = [
        physics_results["CM-FQK"]["macro_f1_mean"],
        physics_results["FQK"]["macro_f1_mean"],
        physics_results["PQK"]["macro_f1_mean"],
        physics_results["AGPQK"]["macro_f1_mean"],
    ]

    phys_std = [
        physics_results["CM-FQK"]["macro_f1_std"],
        physics_results["FQK"]["macro_f1_std"],
        physics_results["PQK"]["macro_f1_std"],
        physics_results["AGPQK"]["macro_f1_std"],
    ]

    x = np.arange(len(kernels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out AGPQK for PCA
    rects1 = ax.bar(
        x[:3] - width / 2,
        pca_f1[:3],
        width,
        yerr=pca_std[:3],
        label="PCA Features (5-fold CV)",
        color="#3498db",
        edgecolor="black",
        capsize=3,
    )
    rects2 = ax.bar(
        x + width / 2,
        phys_f1,
        width,
        yerr=phys_std,
        label="Physics Features (5-fold CV)",
        color="#e67e22",
        edgecolor="black",
        capsize=3,
    )

    # Add tuned RBF ceiling
    ax.axhline(
        y=physics_results["RBF"]["macro_f1_mean"],
        color="#27ae60",
        linestyle="--",
        linewidth=2,
        label=f"RBF Tuned ({physics_results['RBF']['macro_f1_mean']:.3f} ± {physics_results['RBF']['macro_f1_std']:.3f})",
    )

    # Add Random Forest baseline
    ax.axhline(
        y=physics_results["RandomForest"]["macro_f1_mean"],
        color="#9b59b6",
        linestyle=":",
        linewidth=2,
        label=f"RandomForest ({physics_results['RandomForest']['macro_f1_mean']:.3f})",
    )

    ax.set_ylabel("Macro-F1 Score", fontsize=12)
    ax.set_title("Macro-F1 Across Feature Regimes (5-fold CV)", fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(kernels, fontsize=11)
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(
        "results/paper_figures/fig7_feature_regime.png", dpi=300, bbox_inches="tight"
    )
    print("Saved fig7_feature_regime.png")


# -------------------------------------------------------------
# Figure 3: Geometric and alignment analysis (UPDATED)
# -------------------------------------------------------------
def plot_fig3():
    kernels = ["CM-FQK", "FQK", "PQK", "RBF"]

    # Updated KTA from 5-fold CV
    kta_values = [
        pca_results["CM-FQK"]["kta_mean"],
        pca_results["FQK"]["kta_mean"],
        pca_results["PQK"]["kta_mean"],
        pca_results["RBF"]["kta_mean"],
    ]

    rbf_kta = pca_results["RBF"]["kta_mean"]
    delta_kta = [f"+{k - rbf_kta:.3f}" for k in kta_values]
    delta_kta[-1] = ""  # RBF has no delta

    # Geometric difference from old analysis (g values)
    g_values = [406.12, 384.76, 327.02, 0]  # RBF has no g

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: g values (keep old for comparison)
    colors_g = ["#3498db", "#2ecc71", "#e74c3c", "#95a5a6"]
    bars = ax1.bar(kernels[:3], g_values[:3], color=colors_g[:3], edgecolor="black")
    ax1.set_title("Geometric Difference ($g$) vs RBF", fontsize=14, pad=15)
    ax1.set_ylabel("$g$ value", fontsize=12)
    ax1.set_ylim(0, 500)
    for bar in bars:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    ax1.axhline(y=0, color="gray", linestyle="-", zorder=0)

    # Right subplot: KTA (updated from 5-fold CV)
    colors_kta = ["#3498db", "#2ecc71", "#e74c3c", "#95a5a6"]
    bars2 = ax2.bar(kernels, kta_values, color=colors_kta, edgecolor="black")
    ax2.set_title(r"Task Alignment ($\mathrm{KTA}_c$) - 5-fold CV", fontsize=14, pad=15)
    ax2.set_ylabel(r"$\mathrm{KTA}_c$", fontsize=12)
    ax2.set_ylim(0, 0.30)

    for i, bar in enumerate(bars2):
        # Base KTA text
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
        # Delta KTA annotation (if not RBF)
        if delta_kta[i]:
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                rf"$\Delta${delta_kta[i]}",
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )

    # Draw horizontal line for RBF baseline
    ax2.axhline(y=rbf_kta, color="#7f8c8d", linestyle="--", zorder=0)

    plt.suptitle(
        "Figure 3: Geometric and Alignment Analysis (Fused PCA, 5-fold CV)",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        "results/paper_figures/fig3_geometric_alignment.png",
        dpi=300,
        bbox_inches="tight",
    )
    print("Saved fig3_geometric_alignment.png")


# -------------------------------------------------------------
# Figure 8: Per-class F1 comparison (check if needs update)
# -------------------------------------------------------------
def plot_fig8_per_class():
    """Generate per-class F1 comparison figure."""

    # Get fold results
    with open("results/pca_cv/exp1_pca_cv_results.json", "r") as f:
        pca_folds = json.load(f)["fold_results"]

    with open("results/physics_cv/exp1_physics_cv_results.json", "r") as f:
        phys_folds = json.load(f)["fold_results"]

    # Class names (LCZ 1-17)
    classes = [f"LCZ{i}" for i in range(1, 18)]

    # Extract per-class F1 from fold results (use fold 0 for simplicity)
    # This would need actual per-class computation from the results

    # For now, create a placeholder - this needs actual per-class data
    fig, ax = plt.subplots(figsize=(14, 6))

    # This figure may need more detailed data extraction
    print("Note: fig8_per_class_f1_comparison.png needs per-class F1 data extraction")
    plt.close()
    print("Skipped fig8 - needs per-class data")


# -------------------------------------------------------------
# Physics regime performance comparison
# -------------------------------------------------------------
def plot_physics_comparison():
    """Bar chart comparing all methods in physics regime."""

    methods = [
        "RBF\nTuned",
        "RBF\nCV",
        "Random\nForest",
        "AGPQK",
        "CM-FQK",
        "FQK",
        "PQK",
    ]
    f1_values = [
        physics_results["RBF"]["macro_f1_mean"],
        physics_results["RBF-CV"]["macro_f1_mean"],
        physics_results["RandomForest"]["macro_f1_mean"],
        physics_results["AGPQK"]["macro_f1_mean"],
        physics_results["CM-FQK"]["macro_f1_mean"],
        physics_results["FQK"]["macro_f1_mean"],
        physics_results["PQK"]["macro_f1_mean"],
    ]
    f1_stds = [
        physics_results["RBF"]["macro_f1_std"],
        physics_results["RBF-CV"]["macro_f1_std"],
        physics_results["RandomForest"]["macro_f1_std"],
        physics_results["AGPQK"]["macro_f1_std"],
        physics_results["CM-FQK"]["macro_f1_std"],
        physics_results["FQK"]["macro_f1_std"],
        physics_results["PQK"]["macro_f1_std"],
    ]

    colors = [
        "#27ae60",
        "#27ae60",
        "#9b59b6",
        "#e74c3c",
        "#3498db",
        "#3498db",
        "#3498db",
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(
        methods, f1_values, yerr=f1_stds, color=colors, edgecolor="black", capsize=4
    )

    ax.set_ylabel("Macro-F1 Score", fontsize=12)
    ax.set_title("Physics Feature Regime: 5-fold CV Results", fontsize=14)
    ax.set_ylim(0, 0.65)

    for bar, val, std in zip(bars, f1_values, f1_stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + std + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(
        "results/paper_figures/fig_physics_comparison.png", dpi=300, bbox_inches="tight"
    )
    print("Saved fig_physics_comparison.png")


# -------------------------------------------------------------
# PCA regime performance comparison
# -------------------------------------------------------------
def plot_pca_comparison():
    """Bar chart comparing all methods in PCA regime."""

    methods = ["Random\nForest", "CM-FQK", "FQK", "RBF\nTuned", "PQK", "RBF"]
    f1_values = [
        pca_results["RandomForest"]["macro_f1_mean"],
        pca_results["CM-FQK"]["macro_f1_mean"],
        pca_results["FQK"]["macro_f1_mean"],
        pca_results["RBF-Tuned"]["macro_f1_mean"],
        pca_results["PQK"]["macro_f1_mean"],
        pca_results["RBF"]["macro_f1_mean"],
    ]
    f1_stds = [
        pca_results["RandomForest"]["macro_f1_std"],
        pca_results["CM-FQK"]["macro_f1_std"],
        pca_results["FQK"]["macro_f1_std"],
        pca_results["RBF-Tuned"]["macro_f1_std"],
        pca_results["PQK"]["macro_f1_std"],
        pca_results["RBF"]["macro_f1_std"],
    ]

    colors = ["#9b59b6", "#3498db", "#3498db", "#27ae60", "#3498db", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        methods, f1_values, yerr=f1_stds, color=colors, edgecolor="black", capsize=4
    )

    ax.set_ylabel("Macro-F1 Score", fontsize=12)
    ax.set_title("PCA Feature Regime: 5-fold CV Results", fontsize=14)
    ax.set_ylim(0, 0.35)

    for bar, val, std in zip(bars, f1_values, f1_stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + std + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(
        "results/paper_figures/fig_pca_comparison.png", dpi=300, bbox_inches="tight"
    )
    print("Saved fig_pca_comparison.png")


if __name__ == "__main__":
    print("Generating updated paper figures with 5-fold CV results...")
    plot_fig7()
    plot_fig3()
    plot_physics_comparison()
    plot_pca_comparison()
    print("\nAll figures generated successfully!")
    print("\nSummary of updated results:")
    print("=" * 60)
    print("PCA REGIME (5-fold CV):")
    print(
        f"  CM-FQK: {pca_results['CM-FQK']['macro_f1_mean']:.3f} ± {pca_results['CM-FQK']['macro_f1_std']:.3f}"
    )
    print(
        f"  FQK:    {pca_results['FQK']['macro_f1_mean']:.3f} ± {pca_results['FQK']['macro_f1_std']:.3f}"
    )
    print(
        f"  PQK:    {pca_results['PQK']['macro_f1_mean']:.3f} ± {pca_results['PQK']['macro_f1_std']:.3f}"
    )
    print(
        f"  RBF:    {pca_results['RBF']['macro_f1_mean']:.3f} ± {pca_results['RBF']['macro_f1_std']:.3f}"
    )
    print(
        f"  RandomForest: {pca_results['RandomForest']['macro_f1_mean']:.3f} ± {pca_results['RandomForest']['macro_f1_std']:.3f}"
    )
    print("\nPHYSICS REGIME (5-fold CV):")
    print(
        f"  AGPQK:  {physics_results['AGPQK']['macro_f1_mean']:.3f} ± {physics_results['AGPQK']['macro_f1_std']:.3f}"
    )
    print(
        f"  CM-FQK: {physics_results['CM-FQK']['macro_f1_mean']:.3f} ± {physics_results['CM-FQK']['macro_f1_std']:.3f}"
    )
    print(
        f"  FQK:    {physics_results['FQK']['macro_f1_mean']:.3f} ± {physics_results['FQK']['macro_f1_std']:.3f}"
    )
    print(
        f"  PQK:    {physics_results['PQK']['macro_f1_mean']:.3f} ± {physics_results['PQK']['macro_f1_std']:.3f}"
    )
    print(
        f"  RBF:    {physics_results['RBF']['macro_f1_mean']:.3f} ± {physics_results['RBF']['macro_f1_std']:.3f}"
    )
    print(
        f"  RandomForest: {physics_results['RandomForest']['macro_f1_mean']:.3f} ± {physics_results['RandomForest']['macro_f1_std']:.3f}"
    )
    print("=" * 60)
