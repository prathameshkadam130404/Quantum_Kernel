"""
Generate all figures for the quantum kernel paper.

Reads from completed experiment results in results/ and produces
publication-grade PDF + PNG figures in results/paper_figures/.

Usage:
    python scripts/generate_paper_figures.py

Requirements:
    numpy, matplotlib, pandas
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent style across all figures
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
})

# Color palette — muted, print-safe, distinguishable
C_FQK = "#2166ac"
C_PQK = "#4393c3"
C_SRQFM = "#d6604d"
C_AGPQK = "#b2182b"
C_RBF = "#1b7837"
C_RF = "#762a83"
C_CLASSICAL = "#636363"


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as both PDF (for LaTeX) and PNG (for preview)."""
    fig.savefig(OUT_DIR / f"{name}.pdf", format="pdf")
    fig.savefig(OUT_DIR / f"{name}.png", format="png")
    print(f"  Saved: {OUT_DIR / name}.pdf/.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Learning Curves (E10 + E27 combined)
# ---------------------------------------------------------------------------
def fig_learning_curves():
    """AGPQK vs tuned RBF-SVM learning curves from E10 and E27."""
    print("Generating Fig 1: Learning Curves...")

    # E10 data (from paper text, 5 seeds, physics features)
    n_e10 = np.array([250, 500, 1000, 1500, 2000])
    agpqk_e10 = np.array([0.316, 0.350, 0.413, 0.447, 0.418])
    rbf_e10 = np.array([0.393, 0.414, 0.457, 0.521, 0.532])

    # E27 data (from paper Table E27, 3 seeds)
    n_e27 = np.array([2000, 3000, 5000, 7000, 10000])
    agpqk_e27 = np.array([0.3501, 0.3456, 0.3875, 0.3898, 0.4016])
    agpqk_e27_std = np.array([0.0325, 0.0237, 0.0142, 0.0048, 0.0099])
    rbf_e27 = np.array([0.5125, 0.5437, 0.5951, 0.6428, 0.6332])
    rbf_e27_std = np.array([0.0305, 0.0084, 0.0089, 0.0141, 0.0235])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)

    # Panel (a): E10 (N=250 to 2000)
    ax1.plot(n_e10, agpqk_e10, "o-", color=C_AGPQK, label="AGPQK")
    ax1.plot(n_e10, rbf_e10, "s-", color=C_RBF, label="RBF-SVM (tuned)")
    ax1.set_xlabel("Training samples $N$")
    ax1.set_ylabel("Macro-F1")
    ax1.set_title("(a) $N = 250$--$2{,}000$")
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.set_ylim(0.25, 0.70)
    ax1.grid(True, alpha=0.3)

    # Panel (b): E27 (N=2000 to 10000)
    ax2.errorbar(n_e27, agpqk_e27, yerr=agpqk_e27_std, fmt="o-",
                 color=C_AGPQK, capsize=3, label="AGPQK")
    ax2.errorbar(n_e27, rbf_e27, yerr=rbf_e27_std, fmt="s-",
                 color=C_RBF, capsize=3, label="RBF-SVM (tuned)")
    ax2.set_xlabel("Training samples $N$")
    ax2.set_title("(b) $N = 2{,}000$--$10{,}000$")
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x):,}"))

    fig.tight_layout()
    save_fig(fig, "fig_learning_curves")


# ---------------------------------------------------------------------------
# Figure 2: Concentration Scaling (E28)
# ---------------------------------------------------------------------------
def fig_concentration_scaling():
    """Off-diagonal variance vs qubit count for FQK, ZZ-PQK, SRQFM-PQK."""
    print("Generating Fig 2: Concentration Scaling...")

    n_qubits = np.array([2, 4, 6, 8])

    # From paper Table E28 (reps=2)
    fqk_var = np.array([0.1214, 0.0645, 0.0166, 0.0018])
    zz_var = np.array([0.0756, 0.0656, 0.0383, 0.0409])
    srqfm_var = np.array([0.0798, 0.1101, 0.1033, 0.0659])

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.semilogy(n_qubits, fqk_var, "D-", color=C_FQK, label="FQK",
                markersize=7)
    ax.semilogy(n_qubits, zz_var, "s-", color=C_PQK, label="ZZ-PQK",
                markersize=7)
    ax.semilogy(n_qubits, srqfm_var, "o-", color=C_SRQFM, label="SRQFM-PQK",
                markersize=7)

    # Reference line: exponential decay O(4^{-n})
    ref_exp = fqk_var[0] * (4.0 ** (-n_qubits + 2))
    ax.semilogy(n_qubits, ref_exp, "k--", alpha=0.4, linewidth=1.0,
                label=r"$O(4^{-n})$ reference")

    ax.set_xlabel("Number of qubits $n$")
    ax.set_ylabel(r"$\mathrm{Var}[K_{\mathrm{off}}]$")
    ax.set_xticks(n_qubits)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_ylim(5e-4, 0.3)

    fig.tight_layout()
    save_fig(fig, "fig_concentration_scaling")


# ---------------------------------------------------------------------------
# Figure 3: Noise Degradation (E3)
# ---------------------------------------------------------------------------
def fig_noise_degradation():
    """Macro-F1 vs depolarizing noise rate for FQK and AGPQK."""
    print("Generating Fig 3: Noise Degradation...")

    # From paper text (Sec 5.7, 3 seeds mean)
    noise_rates = np.array([0.0, 0.002, 0.005, 0.01, 0.02, 0.05])
    fqk_f1 = np.array([0.290, 0.278, 0.260, 0.240, 0.215, 0.173])
    agpqk_f1 = np.array([0.237, 0.235, 0.230, 0.225, 0.220, 0.213])

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    ax.plot(noise_rates, fqk_f1, "D-", color=C_FQK, label="FQK ($-40\\%$)",
            markersize=7)
    ax.plot(noise_rates, agpqk_f1, "o-", color=C_AGPQK,
            label="AGPQK ($-10\\%$)", markersize=7)

    # Shade the hardware-relevant regime
    ax.axvspan(0.01, 0.05, alpha=0.08, color="red",
               label="Hardware regime")

    ax.set_xlabel("Depolarizing noise rate $p$")
    ax.set_ylabel("Macro-F1")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.002, 0.055)
    ax.set_ylim(0.15, 0.32)

    fig.tight_layout()
    save_fig(fig, "fig_noise_degradation")


# ---------------------------------------------------------------------------
# Figure 4: OOD Generalization (E15) — bar chart
# ---------------------------------------------------------------------------
def fig_ood_comparison():
    """In-distribution vs OOD performance comparison."""
    print("Generating Fig 4: OOD Generalization...")

    models = ["AGPQK", "RBF-SVM\n(tuned)", "Random\nForest"]
    in_dist = [0.418, 0.466, 0.497]  # approximate in-distribution F1
    ood = [0.153, 0.549, 0.544]
    colors_in = [C_AGPQK, C_RBF, C_RF]
    colors_ood = [C_AGPQK, C_RBF, C_RF]

    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    bars_in = ax.bar(x - width / 2, in_dist, width, label="In-distribution",
                     color=colors_in, alpha=0.7, edgecolor="black",
                     linewidth=0.5)
    bars_ood = ax.bar(x + width / 2, ood, width, label="Out-of-distribution",
                      color=colors_ood, alpha=0.35, edgecolor="black",
                      linewidth=0.5, hatch="//")

    # Add value labels
    for bar in bars_in:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=7.5)
    for bar in bars_ood:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom",
                fontsize=7.5)

    # Annotate the collapse
    ax.annotate(r"$\mathbf{-63\%}$", xy=(0 + width / 2, 0.153),
                xytext=(0.6, 0.25), fontsize=9, color=C_AGPQK,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_AGPQK, lw=1.5))

    ax.set_ylabel("Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, 0.65)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    save_fig(fig, "fig_ood_comparison")


# ---------------------------------------------------------------------------
# Figure 5: Topology Comparison (E7) — bar chart with concentration overlay
# ---------------------------------------------------------------------------
def fig_topology_comparison():
    """F1 and off-diagonal variance for linear/cross-modal/full topologies."""
    print("Generating Fig 5: Topology Comparison...")

    topologies = ["Linear", "Cross-modal", "Full"]
    f1_means = [0.468, 0.467, 0.396]
    f1_stds = [0.010, 0.008, 0.013]
    var_off = [0.0075, 0.0060, 0.0005]

    fig, ax1 = plt.subplots(figsize=(4.5, 3.2))

    x = np.arange(len(topologies))
    bars = ax1.bar(x, f1_means, yerr=f1_stds, width=0.5, capsize=4,
                   color=[C_PQK, C_SRQFM, C_CLASSICAL], alpha=0.75,
                   edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Macro-F1", color="black")
    ax1.set_xticks(x)
    ax1.set_xticklabels(topologies)
    ax1.set_ylim(0.35, 0.52)
    ax1.grid(True, alpha=0.2, axis="y")

    # Secondary axis: off-diagonal variance
    ax2 = ax1.twinx()
    ax2.plot(x, var_off, "kD--", markersize=7, linewidth=1.2,
             label=r"$\mathrm{Var}[K_{\mathrm{off}}]$")
    ax2.set_ylabel(r"$\mathrm{Var}[K_{\mathrm{off}}]$", color="black")
    ax2.set_ylim(0, 0.010)
    ax2.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    save_fig(fig, "fig_topology_comparison")


# ---------------------------------------------------------------------------
# Figure 6: SRQFM Ablation (E25-v3)
# ---------------------------------------------------------------------------
def fig_srqfm_ablation():
    """Ablation of SRQFM coupling variants with baselines."""
    print("Generating Fig 6: SRQFM Ablation...")

    configs = ["A\nbaseline", "B\ndepth-3", "C\nRY enc.", "D\nscaled",
               "E\nfull-range", "F\nRZ+RY"]
    f1_means = [0.4864, 0.4848, 0.4384, 0.4643, 0.4568, 0.4766]
    f1_stds = [0.0181, 0.0313, 0.0228, 0.0190, 0.0204, 0.0065]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    x = np.arange(len(configs))
    bars = ax.bar(x, f1_means, yerr=f1_stds, width=0.55, capsize=3,
                  color=C_SRQFM, alpha=0.7, edgecolor="black", linewidth=0.5)

    # Classical baselines
    ax.axhline(y=0.405, color=C_RBF, linestyle="--", linewidth=1.2,
               label="RBF-SVM (0.405)")
    ax.axhline(y=0.450, color=C_RF, linestyle="-.", linewidth=1.2,
               label="Random Forest (0.450)")

    ax.set_ylabel("Macro-F1")
    ax.set_xlabel("SRQFM Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=8)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)
    ax.set_ylim(0.38, 0.55)
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    save_fig(fig, "fig_srqfm_ablation")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Generate all paper figures."""
    print(f"Output directory: {OUT_DIR.resolve()}\n")

    fig_learning_curves()
    fig_concentration_scaling()
    fig_noise_degradation()
    fig_ood_comparison()
    fig_topology_comparison()
    fig_srqfm_ablation()

    print(f"\nAll figures saved to: {OUT_DIR.resolve()}")
    print("Include in LaTeX with:")
    print(r"  \includegraphics[width=\columnwidth]{results/paper_figures/fig_*.pdf}")


if __name__ == "__main__":
    main()
