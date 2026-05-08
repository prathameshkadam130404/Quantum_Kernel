"""
Diagnostic 3: Kernel Similarity Analysis
  (a) Centered Kernel Alignment (CKA) matrix between all kernels
  (b) Frobenius distance matrix between all kernels
  (c) Spectral analysis: eigenvalue distributions

ANSWERS THREE QUESTIONS:

(a) CKA: "How similar are the quantum kernels to each other and to classical kernels
    geometrically? Is CM_FQK genuinely different from FQK?"
    CKA = HSIC(K1,K2) / sqrt(HSIC(K1,K1) * HSIC(K2,K2))
    CKA=1.0: identical geometry. CKA≈0: completely different geometry.
    Reference: Kornblith et al. NeurIPS 2019.

(b) Frobenius: "How numerically different are the kernel matrices entry-by-entry?"
    Normalized Frobenius: ||K1-K2||_F / sqrt(||K1||_F * ||K2||_F)
    This is a simple sanity check — large CKA with small Frobenius distance
    would mean the kernels are nearly identical matrices, not just geometrically
    similar.

(c) Spectral: "What is the effective RKHS dimensionality of each kernel?
    How does the eigenvalue decay compare across kernels?"
    This explains why PQK has lower eff_rank despite high g.

Runtime: ~5 minutes (pure linear algebra on cached matrices)
"""

import sys, os
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
import json
import config

# ---- Paths ----
BASE    = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
TFK_DIR = os.path.join(BASE, "tfk")
OUT_DIR = os.path.join(config.RESULTS_DIR, "diagnostics", "kernel_similarity")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load kernels ----
print("Loading kernels...")
K_fqk = np.load(os.path.join(BASE, "K_fqk_train.npy"))
K_pqk = np.load(os.path.join(BASE, "K_pqk_train.npy"))
K_tfk = np.load(os.path.join(TFK_DIR, "K_tfk_train.npy"))
K_rbf = np.load(os.path.join(BASE, "K_rbf_train.npy"))

cm_paths = [os.path.join(BASE, "K_cm_fqk_train.npy"),
            os.path.join(BASE, "cm_fqk", "K_cm_fqk_train.npy")]
K_cm = None
for p in cm_paths:
    if os.path.exists(p):
        K_cm = np.load(p)
        break

kernels = {"FQK": K_fqk, "PQK": K_pqk, "TFK": K_tfk, "RBF": K_rbf}
if K_cm is not None:
    kernels["CM_FQK"] = K_cm

knames = list(kernels.keys())
n = len(K_fqk)
print(f"n={n}, kernels: {knames}")

# ============================================================
# (a) CKA Matrix
# ============================================================
def centering_matrix(n):
    return np.eye(n) - np.ones((n, n)) / n

def hsic(K1, K2, H):
    """Unbiased HSIC estimator (Kornblith et al. 2019)."""
    return float(np.sum((H @ K1 @ H) * (H @ K2 @ H))) / ((n - 1) ** 2)

def cka(K1, K2, H):
    h12 = hsic(K1, K2, H)
    h11 = hsic(K1, K1, H)
    h22 = hsic(K2, K2, H)
    return h12 / (np.sqrt(h11 * h22) + 1e-15)

print("\n--- (a) CKA Matrix ---")
H = centering_matrix(n)
cka_matrix = np.zeros((len(knames), len(knames)))
for i, k1 in enumerate(knames):
    for j, k2 in enumerate(knames):
        cka_matrix[i, j] = cka(kernels[k1], kernels[k2], H)

print(f"\n{'':>10}", end="")
for k in knames:
    print(f"{k:>10}", end="")
print()
for i, k1 in enumerate(knames):
    print(f"{k1:>10}", end="")
    for j in range(len(knames)):
        print(f"{cka_matrix[i,j]:>10.4f}", end="")
    print()

# ============================================================
# (b) Frobenius Distance Matrix
# ============================================================
print("\n--- (b) Normalized Frobenius Distance Matrix ---")
frob_matrix = np.zeros((len(knames), len(knames)))
for i, k1 in enumerate(knames):
    for j, k2 in enumerate(knames):
        diff = kernels[k1] - kernels[k2]
        frob_dist = np.linalg.norm(diff, 'fro')
        norm_denom = np.sqrt(np.linalg.norm(kernels[k1], 'fro') *
                             np.linalg.norm(kernels[k2], 'fro'))
        frob_matrix[i, j] = frob_dist / (norm_denom + 1e-15)

print(f"\n{'':>10}", end="")
for k in knames:
    print(f"{k:>10}", end="")
print()
for i, k1 in enumerate(knames):
    print(f"{k1:>10}", end="")
    for j in range(len(knames)):
        print(f"{frob_matrix[i,j]:>10.4f}", end="")
    print()

# ============================================================
# (c) Spectral Analysis
# ============================================================
print("\n--- (c) Spectral Analysis ---")
spectral_stats = {}
eigenvalues_all = {}
for kname, K in kernels.items():
    K_sym = (K + K.T) / 2
    eigs = np.linalg.eigvalsh(K_sym)
    eigs = np.sort(eigs)[::-1]   # descending

    # Effective rank (participation ratio)
    eigs_pos = np.clip(eigs, 0, None)
    total = eigs_pos.sum() + 1e-15
    probs = eigs_pos / total
    eff_rank_pr = float(np.exp(-np.sum(probs * np.log(probs + 1e-15))))  # entropy-based

    # Effective rank (trace / max_eig)
    eff_rank_simple = float(np.trace(K_sym) / max(eigs[0], 1e-10))

    # Spectral entropy
    spec_entropy = float(-np.sum(probs * np.log(probs + 1e-15)))

    # Participation ratio
    part_ratio = float((np.sum(eigs_pos) ** 2) / (np.sum(eigs_pos ** 2) + 1e-15))

    # Top eigenvalue contribution
    top1_ratio = float(eigs[0] / (total))
    top5_ratio = float(eigs[:5].sum() / total)

    eigenvalues_all[kname] = eigs
    spectral_stats[kname] = {
        "eff_rank_entropy": eff_rank_pr,
        "eff_rank_simple": eff_rank_simple,
        "spectral_entropy": spec_entropy,
        "participation_ratio": part_ratio,
        "top1_eigenvalue_fraction": top1_ratio,
        "top5_eigenvalue_fraction": top5_ratio,
        "n_positive_eigenvalues": int(np.sum(eigs > 1e-6)),
        "max_eigenvalue": float(eigs[0]),
        "min_eigenvalue": float(eigs[-1]),
    }

    print(f"\n  {kname}:")
    print(f"    Eff rank (entropy)  : {eff_rank_pr:.2f}")
    print(f"    Eff rank (simple)   : {eff_rank_simple:.2f}")
    print(f"    Spectral entropy    : {spec_entropy:.4f}")
    print(f"    Participation ratio : {part_ratio:.2f}")
    print(f"    Top-1 eig fraction  : {top1_ratio:.4f}")
    print(f"    Top-5 eig fraction  : {top5_ratio:.4f}")
    print(f"    N positive eigs     : {int(np.sum(eigs > 1e-6))}")

# ============================================================
# Plots
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: CKA heatmap
import matplotlib
im1 = axes[0].imshow(cka_matrix, cmap="RdYlGn", vmin=0, vmax=1)
axes[0].set_xticks(range(len(knames)))
axes[0].set_yticks(range(len(knames)))
axes[0].set_xticklabels(knames, rotation=45, ha="right")
axes[0].set_yticklabels(knames)
for i in range(len(knames)):
    for j in range(len(knames)):
        axes[0].text(j, i, f"{cka_matrix[i,j]:.2f}", ha="center", va="center",
                     fontsize=9, color="black")
plt.colorbar(im1, ax=axes[0])
axes[0].set_title("CKA Matrix\n(1.0 = identical geometry)", fontsize=11)

# Plot 2: Frobenius distance heatmap
im2 = axes[1].imshow(frob_matrix, cmap="RdYlGn_r", vmin=0, vmax=frob_matrix.max())
axes[1].set_xticks(range(len(knames)))
axes[1].set_yticks(range(len(knames)))
axes[1].set_xticklabels(knames, rotation=45, ha="right")
axes[1].set_yticklabels(knames)
for i in range(len(knames)):
    for j in range(len(knames)):
        axes[1].text(j, i, f"{frob_matrix[i,j]:.2f}", ha="center", va="center",
                     fontsize=9, color="black")
plt.colorbar(im2, ax=axes[1])
axes[1].set_title("Normalised Frobenius Distance\n(0.0 = identical matrices)", fontsize=11)

# Plot 3: Eigenvalue decay curves (top 50 eigenvalues, normalized)
colors_sp = {"FQK": "steelblue", "PQK": "coral", "TFK": "green",
             "RBF": "gray", "CM_FQK": "darkorange"}
for kname, eigs in eigenvalues_all.items():
    eigs_pos = np.clip(eigs[:50], 0, None)
    eigs_norm = eigs_pos / (eigs_pos.max() + 1e-15)
    axes[2].plot(range(1, len(eigs_norm)+1), eigs_norm, "-",
                 linewidth=2, label=kname, color=colors_sp.get(kname, "black"), alpha=0.85)

axes[2].set_xlabel("Eigenvalue rank", fontsize=12)
axes[2].set_ylabel("Normalised eigenvalue", fontsize=12)
axes[2].set_title("Eigenvalue Decay (top 50)\n(Faster decay = lower effective rank)", fontsize=11)
axes[2].legend(fontsize=10)
axes[2].set_yscale("log")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "kernel_similarity.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "kernel_similarity.png"), dpi=150, bbox_inches="tight")

# ---- Save ----
out = {
    "cka_matrix": {knames[i]: {knames[j]: float(cka_matrix[i,j])
                                for j in range(len(knames))}
                   for i in range(len(knames))},
    "frobenius_matrix": {knames[i]: {knames[j]: float(frob_matrix[i,j])
                                      for j in range(len(knames))}
                         for i in range(len(knames))},
    "spectral_stats": spectral_stats,
}
with open(os.path.join(OUT_DIR, "kernel_similarity_results.json"), "w") as f:
    json.dump(out, f, indent=2)

print(f"\nPlot + results saved: {OUT_DIR}/")
print("\n=== PAPER GUIDANCE ===")
print("CKA(FQK, RBF): if < 0.5 → quantum kernel is genuinely different geometry")
print("CKA(CM_FQK, FQK): if > 0.9 → CM_FQK improvement is marginal (same geometry)")
print("                  if < 0.8 → cross-modal topology created a genuinely different kernel")
print("Frobenius(CM_FQK, FQK): if < 0.1 → kernels are nearly identical matrices")
print("Spectral: lower eff_rank for PQK confirms its RKHS is more concentrated")
