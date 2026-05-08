"""
Quick concentration test for physics-FQK with [0,π/4] rescaling.

Tests n=200 stratified samples — takes ~4 minutes.
Run BEFORE the full exp1_physics.py run to confirm rescaling fixed
the exponential concentration problem.

Pass criteria (based on PCA-FQK baseline):
  off_diag_mean  > 0.05      (was 0.021 with γ=1.0 encoding, target >0.05)
  eff_rank       < 100       (was 460 with γ=1.0, PCA-FQK was 33)
  ΔKTA(FQK-RBF) > -0.05      (was -0.267 with γ=1.0)
  CV             in [1, 8]   (PCA-FQK was 3.17)
  γ used:        0.75        (CV-selected, Shaydulin & Wild 2022)

Usage:
    python scripts/test_physics_concentration.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.model_selection import train_test_split

import config
from src.quantum_kernels import compute_fqk_kernel_matrix
from src.classical_kernels import compute_rbf_kernel

# ============ CONFIG ============
N_TEST     = 200       # small n for speed — ~4 min on GPU

from src.bandwidth import load_bandwidth_gamma
GAMMA_PHYS = load_bandwidth_gamma("physics")   # CV-selected: 0.75
ENCODE_MAX = GAMMA_PHYS * np.pi               # [0, 0.75π]

FEATURE_INDICES = [0, 5, 6, 3, 10, 15, 9, 11]
FEATURE_NAMES   = ["NDVI", "NDRE", "MNDWI", "BSI",
                   "CrossPol", "NDVI_tex", "SAR_total", "PolCoh"]

# PCA-FQK baselines for comparison
PCA_FQK_KTA      = 0.1840
PCA_FQK_EFF_RANK = 33.24
PCA_FQK_CV       = 3.17
PCA_RBF_KTA      = 0.1309

# ============ LOAD FEATURES ============
phys16_path = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
phys8_path  = os.path.join(config.PROCESSED_DIR, "physics_features_piqfm.npz")

print("=" * 60)
print(f"Physics-FQK Concentration Test  (n={N_TEST})")
print(f"Encoding range: [0, γπ] = [0, {ENCODE_MAX:.4f}]  (γ={GAMMA_PHYS:.3f}, CV-selected)")
print("=" * 60)

# Load raw [0,π] features then rescale
if os.path.exists(phys8_path):
    print(f"WARNING: {phys8_path} exists — delete it to use v5 features")
    print("Run: rm data/processed/physics_features_piqfm.npz")
    sys.exit(1)

if not os.path.exists(phys16_path):
    print(f"ERROR: {phys16_path} not found. Run extract_physics_features.py first.")
    sys.exit(1)

data = np.load(phys16_path)
X16  = data["X_train"]   # (2000, 16), normalized to [0,π]
y    = data["y_train"]

# Extract 8 features and apply CV-selected bandwidth (γ=0.75 → [0, 0.75π])
from src.bandwidth import apply_bandwidth
X8 = apply_bandwidth(X16[:, FEATURE_INDICES], "physics")

print(f"\nFeature stats after [0,π/4] rescaling:")
print(f"  Range:  [{X8.min():.4f}, {X8.max():.4f}]  "
      f"(expected: ~[0, {ENCODE_MAX:.4f}] = [0, {GAMMA_PHYS:.3f}π])")
print(f"  Stds:   {np.std(X8, axis=0).round(4)}")

# Stratified subsample
_, X_sub, _, y_sub = train_test_split(
    X8, y, test_size=N_TEST,
    stratify=y, random_state=config.RANDOM_SEED
)

classes, counts = np.unique(y_sub, return_counts=True)
print(f"\nSubsample: {N_TEST} samples, {len(classes)} classes")
print(f"  Class counts: min={counts.min()}, max={counts.max()}")

# ============ COMPUTE FQK ============
print(f"\nComputing FQK kernel ({N_TEST}×{N_TEST})...")
print("Expected runtime: ~4 minutes on GPU")

K_fqk = compute_fqk_kernel_matrix(X_sub)

# ============ DIAGNOSTICS ============
diag     = np.diag(K_fqk)
off_idx  = np.triu_indices(N_TEST, k=1)
off      = K_fqk[off_idx]

eff_rank_val = None
eigvals = np.linalg.eigvalsh(K_fqk)
eigvals_pos = np.maximum(eigvals, 0)
total = eigvals_pos.sum()
if total > 0:
    probs = eigvals_pos / total
    eff_rank_val = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
top1_frac = float(eigvals_pos[-1] / (total + 1e-12))

cv = float(off.std() / (off.mean() + 1e-12))

print(f"\n--- Kernel Diagnostics ---")
print(f"  Diagonal:    mean={diag.mean():.6f}  (should be 1.0)")
print(f"  Off-diag:    mean={off.mean():.4f}, std={off.std():.4f}")
print(f"  CV:          {cv:.4f}         PCA-FQK: {PCA_FQK_CV:.2f}")
print(f"  Eff rank:    {eff_rank_val:.2f}          PCA-FQK: {PCA_FQK_EFF_RANK:.2f}")
print(f"  Top-1 frac:  {top1_frac:.4f}")
print(f"  Min eigval:  {eigvals.min():.6f}  (≥0 = PSD)")

# ============ KTA ============
def centered_kta(K, y):
    n = len(y)
    col_mean   = K.mean(axis=0)
    row_mean   = K.mean(axis=1)
    grand_mean = K.mean()
    Kc = K - col_mean[None,:] - row_mean[:,None] + grand_mean

    classes, counts = np.unique(y, return_counts=True)
    count_map = dict(zip(classes, counts))
    Ki = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if y[i] == y[j]:
                Ki[i,j] = 1.0 / count_map[y[i]]
    col_m = Ki.mean(axis=0); row_m = Ki.mean(axis=1); gm = Ki.mean()
    Kic = Ki - col_m[None,:] - row_m[:,None] + gm

    return float(np.sum(Kc * Kic) / (
        np.linalg.norm(Kc,'fro') * np.linalg.norm(Kic,'fro') + 1e-12))

K_rbf = compute_rbf_kernel(X_sub)
kta_fqk = centered_kta(K_fqk, y_sub)
kta_rbf = centered_kta(K_rbf, y_sub)
delta_kta = kta_fqk - kta_rbf

print(f"\n--- KTA (centered, class-weighted) ---")
print(f"  Physics-FQK KTA:    {kta_fqk:.4f}   PCA-FQK: {PCA_FQK_KTA:.4f}")
print(f"  Physics-RBF KTA:    {kta_rbf:.4f}   PCA-RBF: {PCA_RBF_KTA:.4f}")
print(f"  ΔKTA (FQK - RBF):   {delta_kta:+.4f}  PCA ΔKTA: +0.0530")

# ============ PASS/FAIL ============
print(f"\n{'='*60}")
print("PASS/FAIL CRITERIA")
print(f"{'='*60}")

checks = [
    ("off_diag_mean > 0.05",   off.mean() > 0.05,          f"{off.mean():.4f}"),
    ("eff_rank < 100",         eff_rank_val < 100,          f"{eff_rank_val:.1f}"),
    ("ΔKTA(FQK-RBF) > -0.05", delta_kta > -0.05,           f"{delta_kta:+.4f}"),
    ("CV in [1, 8]",           1.0 < cv < 8.0,              f"{cv:.2f}"),
]

all_pass = True
for name, passed, value in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}]  {name:<30}  got {value}")

print(f"\n{'='*60}")
if all_pass:
    print("ALL CHECKS PASSED — safe to run full exp1_physics.py")
    print("Expected full run time: ~6-8 hours")
else:
    print("SOME CHECKS FAILED — do NOT run full exp1_physics.py yet")
    print("Review failed checks and adjust encoding range or circuit depth")
print(f"{'='*60}")
