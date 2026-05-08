"""
Diagnostic 4: Per-Class KTA Breakdown

ANSWERS: "Which LCZ urban morphology types does each quantum kernel align
better with than the classical RBF? Is the quantum advantage uniform across
classes or concentrated in specific urban types?"

This is novel for a remote sensing paper. It connects the geometric analysis
to the domain — showing that quantum kernels specifically improve separation
of certain urban structure types (e.g., compact high-rise vs open low-rise)
has direct practical relevance for urban planning and climate modelling.

Method: For each pair of classes (c_i, c_j), extract the submatrix of K
indexed by samples from c_i and c_j, then compute centered weighted KTA
on that submatrix. This gives a 17×17 class-pair KTA matrix per kernel.

Summary statistics:
- Per-class "separability": mean KTA over all pairs involving that class
- Quantum advantage per class: KTA_FQK - KTA_RBF per class
- Which classes benefit most from CM_FQK vs FQK

Runtime: ~15-30 minutes (17×17 = 289 pairs × 5 kernels × KTA computation)
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
OUT_DIR = os.path.join(config.RESULTS_DIR, "diagnostics", "perclass_kta")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load ----
print("Loading kernels and labels...")
data    = np.load("data/processed/subsample_2000.npz")
y_train = data["y_train"]
n       = len(y_train)
classes = np.unique(y_train)
n_classes = len(classes)
print(f"n={n}, n_classes={n_classes}, classes={classes.tolist()}")

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

# LCZ class names for So2Sat (standard LCZ42 naming)
LCZ_NAMES = {
    0: "LCZ1 Compact High-rise",
    1: "LCZ2 Compact Mid-rise",
    2: "LCZ3 Compact Low-rise",
    3: "LCZ4 Open High-rise",
    4: "LCZ5 Open Mid-rise",
    5: "LCZ6 Open Low-rise",
    6: "LCZ7 Lightweight Low-rise",
    7: "LCZ8 Large Low-rise",
    8: "LCZ9 Sparsely Built",
    9: "LCZ10 Heavy Industry",
    10: "LCZA Dense Trees",
    11: "LCZB Scattered Trees",
    12: "LCZC Bush/Scrub",
    13: "LCZD Low Plants",
    14: "LCZE Bare Rock/Paved",
    15: "LCZF Bare Soil/Sand",
    16: "LCZG Water",
}

# ---- Per-pair KTA computation ----
def pair_kta(K, y, ci, cj):
    """
    Compute centered weighted KTA on the submatrix for classes ci and cj only.
    If ci == cj: compute within-class KTA (all samples from class ci).
    """
    if ci == cj:
        idx = np.where(y == ci)[0]
    else:
        idx = np.where((y == ci) | (y == cj))[0]

    if len(idx) < 4:
        return np.nan

    K_sub = K[np.ix_(idx, idx)]
    y_sub = y[idx]
    n_s   = len(y_sub)

    # Ideal kernel
    classes_sub, counts_sub = np.unique(y_sub, return_counts=True)
    cdict = dict(zip(classes_sub.tolist(), counts_sub.tolist()))
    y_ideal = np.zeros((n_s, n_s))
    for i in range(n_s):
        for j in range(n_s):
            if y_sub[i] == y_sub[j]:
                y_ideal[i, j] = 1.0 / cdict[y_sub[i]]

    H = np.eye(n_s) - np.ones((n_s, n_s)) / n_s
    Kc = H @ K_sub @ H
    yc = H @ y_ideal @ H
    num = np.sum(Kc * yc)
    den = np.sqrt(np.sum(Kc**2) * np.sum(yc**2) + 1e-15)
    return float(num / den)

print(f"\nComputing {n_classes}×{n_classes} per-class KTA matrices...")
print("(One matrix per kernel — this takes ~15-30 minutes total)")

from tqdm import tqdm
all_kta_matrices = {}
for kname, K in kernels.items():
    print(f"\n  Processing {kname}...")
    kta_mat = np.zeros((n_classes, n_classes))
    for i, ci in enumerate(tqdm(classes, desc=f"{kname} pairs")):
        for j, cj in enumerate(classes):
            if j < i:
                kta_mat[i, j] = kta_mat[j, i]
            else:
                kta_mat[i, j] = pair_kta(K, y_train, ci, cj)
    all_kta_matrices[kname] = kta_mat

# ---- Per-class separability: mean KTA over all pairs involving class c ----
print("\n--- Per-Class Separability (mean pairwise KTA involving each class) ---")
class_sep = {}
for kname, mat in all_kta_matrices.items():
    sep = []
    for i in range(n_classes):
        row = mat[i, :]
        row_valid = row[~np.isnan(row)]
        sep.append(float(np.nanmean(row_valid)))
    class_sep[kname] = sep

# Print table: class | count | FQK | CM_FQK | RBF | Δ(FQK-RBF) | Δ(CM_FQK-RBF)
print(f"\n{'Class':<30} {'N':>5}", end="")
for kname in kernels:
    print(f" {kname:>8}", end="")
print(f" {'Δ(FQK-RBF)':>12} {'Δ(CM-RBF)':>10}")
print("-" * 100)

class_counts = {c: int(np.sum(y_train == c)) for c in classes}
for i, c in enumerate(classes):
    name = LCZ_NAMES.get(int(c), f"Class {c}")
    row = f"  {name:<28} {class_counts[c]:>5}"
    for kname in kernels:
        row += f" {class_sep[kname][i]:>8.4f}"
    d_fqk = class_sep["FQK"][i] - class_sep["RBF"][i]
    d_cm  = (class_sep["CM_FQK"][i] - class_sep["RBF"][i]) if "CM_FQK" in class_sep else float('nan')
    row  += f" {d_fqk:>+12.4f} {d_cm:>+10.4f}"
    print(row)

# ---- Find classes where quantum >> classical ----
print("\n--- Top 5 Classes with Largest FQK Advantage over RBF ---")
deltas = [(class_sep["FQK"][i] - class_sep["RBF"][i], classes[i]) for i in range(n_classes)]
deltas.sort(reverse=True)
for delta, c in deltas[:5]:
    print(f"  {LCZ_NAMES.get(int(c), f'Class {c}'):<35} ΔKTA(FQK-RBF) = {delta:+.4f}")

print("\n--- Top 5 Classes where RBF >> FQK (classical advantage) ---")
for delta, c in deltas[-5:]:
    print(f"  {LCZ_NAMES.get(int(c), f'Class {c}'):<35} ΔKTA(FQK-RBF) = {delta:+.4f}")

# ---- Plots ----
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
short_names = [LCZ_NAMES.get(int(c), f"C{c}").split()[0] for c in classes]

# Plot 1: FQK vs RBF per-class separability bar chart
x = np.arange(n_classes)
w = 0.35
axes[0].bar(x - w/2, class_sep["FQK"], w, label="FQK", color="steelblue", alpha=0.8)
axes[0].bar(x + w/2, class_sep["RBF"], w, label="RBF", color="coral", alpha=0.8)
if "CM_FQK" in class_sep:
    axes[0].plot(x, class_sep["CM_FQK"], "D-", color="darkorange",
                 markersize=5, linewidth=1.5, label="CM_FQK")
axes[0].set_xticks(x)
axes[0].set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("Mean pairwise KTA", fontsize=12)
axes[0].set_title("Per-Class KTA Separability: Quantum vs Classical", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis="y")

# Plot 2: FQK - RBF quantum advantage per class
delta_fqk = [class_sep["FQK"][i] - class_sep["RBF"][i] for i in range(n_classes)]
bar_colors = ["steelblue" if d > 0 else "salmon" for d in delta_fqk]
axes[1].bar(x, delta_fqk, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5)
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set_xticks(x)
axes[1].set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("ΔKTA (FQK − RBF)", fontsize=12)
axes[1].set_title("Quantum Advantage per LCZ Class\n(blue=quantum better, red=classical better)", fontsize=12)
axes[1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "perclass_kta.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "perclass_kta.png"), dpi=150, bbox_inches="tight")

# ---- Save ----
out = {
    "class_separability": {kname: class_sep[kname] for kname in kernels},
    "class_labels": [LCZ_NAMES.get(int(c), f"Class {c}") for c in classes],
    "class_counts": {str(c): class_counts[c] for c in classes},
}
with open(os.path.join(OUT_DIR, "perclass_kta_results.json"), "w") as f:
    json.dump(out, f, indent=2)

print(f"\nPlot + results saved: {OUT_DIR}/")
print("\n=== PAPER GUIDANCE ===")
print("Classes where ΔKTA(FQK-RBF) > 0 consistently = quantum kernel is better")
print("suited for that urban morphology type.")
print("If concentrated in structurally complex types (LCZ1, LCZ7) → supports")
print("the hypothesis that quantum advantage correlates with structural complexity.")
print("If random across classes → no domain-specific advantage pattern.")
