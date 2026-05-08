"""Per-class one-vs-rest Fisher analysis for the BSCM lacking classes."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.attention_kernel import FEATURE_NAMES_16, FEATURE_MODALITY_16

d = np.load("data/processed/physics_features_16.npz")
X = d["X_train_raw"][:2000]
y = d["y_train"][:2000]

class_names = ['CompactHR','CompactMR','CompactLR','OpenHR','OpenMR','OpenLR',
                'LightLR','LargeLR','Sparse','HeavyInd','DenseTrees','ScattTrees',
                'Bush','LowPlants','BareRock','BareSoil','Water']

# Worst-performing classes from BSCM eval (F1 < 0.40)
lacking = [9, 4, 0, 6, 11, 3, 1]   # HeavyInd, OpenMR, CompactHR, LightLR, ScattTrees, OpenHR, CompactMR
all_classes = list(range(17))

CURRENT_8 = {0, 2, 4, 5, 6, 7, 12, 13}  # NDVI, NDWI, SAVI, NDRE, MNDWI, EVI, VH_dB, VV_dB

def one_vs_rest_fisher(X, y, c, fi):
    mask = (y == c)
    if mask.sum() < 2 or (~mask).sum() < 2:
        return 0.0
    x_pos = X[mask, fi]
    x_neg = X[~mask, fi]
    grand = X[:, fi].mean()
    np_ = mask.sum(); nn_ = (~mask).sum()
    between = np_ * (x_pos.mean() - grand)**2 + nn_ * (x_neg.mean() - grand)**2
    within = ((x_pos - x_pos.mean())**2).sum() + ((x_neg - x_neg.mean())**2).sum()
    return between / max(within, 1e-12)

print("=" * 110)
print("ONE-VS-REST FISHER RATIO PER FEATURE FOR LACKING CLASSES")
print("=" * 110)
header = f"{'Feature':14s}{'Mod':5s}{'Sel':4s}"
for c in lacking:
    header += f" {class_names[c]:>9s}"
header += f" {'MEAN':>7s}"
print(header)
print("-" * 110)

rows = []
for fi in range(16):
    fishers = [one_vs_rest_fisher(X, y, c, fi) for c in lacking]
    mean_f = float(np.mean(fishers))
    sel_str = " * " if fi in CURRENT_8 else "   "
    line = f"{FEATURE_NAMES_16[fi]:14s}{FEATURE_MODALITY_16[fi]:5s}{sel_str:4s}"
    for v in fishers:
        line += f" {v:9.4f}"
    line += f" {mean_f:7.4f}"
    rows.append((fi, mean_f, line))
    print(line)

print()
print("=" * 110)
print("RANKED BY MEAN ONE-VS-REST FISHER ACROSS LACKING CLASSES")
print("=" * 110)
rows.sort(key=lambda x: -x[1])
for rank, (fi, mean_f, _) in enumerate(rows, 1):
    sel = "ALREADY USED" if fi in CURRENT_8 else "MISSING"
    print(f"  #{rank:2d}  [{fi:2d}] {FEATURE_NAMES_16[fi]:14s} ({FEATURE_MODALITY_16[fi]})  mean_F={mean_f:.4f}  --> {sel}")

# Also: per-class top-feature analysis
print()
print("=" * 110)
print("TOP-3 BEST DISCRIMINATING FEATURES PER LACKING CLASS")
print("=" * 110)
for c in lacking:
    fishers_c = [(fi, one_vs_rest_fisher(X, y, c, fi)) for fi in range(16)]
    fishers_c.sort(key=lambda x: -x[1])
    line = f"  {class_names[c]:12s} (n={int((y==c).sum()):4d}): "
    for rank, (fi, v) in enumerate(fishers_c[:3]):
        used = "*" if fi in CURRENT_8 else " "
        line += f"  {used}{FEATURE_NAMES_16[fi]:10s}({v:.3f})"
    print(line)

# Summary: how many of the top discriminators per lacking class are CURRENTLY used?
print()
print("=" * 110)
print("COVERAGE: How many of each class's top-3 discriminators are in the current 8?")
print("=" * 110)
for c in lacking:
    fishers_c = [(fi, one_vs_rest_fisher(X, y, c, fi)) for fi in range(16)]
    fishers_c.sort(key=lambda x: -x[1])
    top3 = [fi for fi, _ in fishers_c[:3]]
    in_current = sum(1 for fi in top3 if fi in CURRENT_8)
    missing = [FEATURE_NAMES_16[fi] for fi in top3 if fi not in CURRENT_8]
    print(f"  {class_names[c]:12s}: {in_current}/3 covered.  Missing top-3: {missing}")
