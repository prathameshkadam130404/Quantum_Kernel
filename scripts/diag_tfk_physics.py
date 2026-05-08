"""
diag_tfk_physics.py — TFK training diagnostic for physics features.

Analyses the saved trained theta and K_tfk kernel to assess:
  1. Theta parameter quality (saturation, range, per-qubit contribution)
  2. Kernel concentration (eff_rank, off-diag mean, CV)
  3. KTA vs FQK and RBF on physics features
  4. Training convergence (if kta_history is saved)

Reads from:
  results/physics/fused/tfk_physics/theta_physics_trained.npy
  results/physics/fused/tfk_physics/K_tfk_physics_train.npy
  results/physics/fused/K_fqk_physics_train.npy        (for comparison)
  results/physics/fused/K_rbf_physics_train.npy        (for comparison)
  data/processed/physics_features_16.npz               (for labels)

Usage:
    python scripts/diag_tfk_physics.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import config
from src.bandwidth import apply_bandwidth

# ── Paths ──────────────────────────────────────────────────────────────────
PHYSICS_DIR = os.path.join(config.RESULTS_DIR, "physics", "fused")
TFK_DIR     = os.path.join(PHYSICS_DIR, "tfk_physics")

THETA_PATH  = os.path.join(TFK_DIR, "theta_physics_trained.npy")
K_TFK_PATH  = os.path.join(TFK_DIR, "K_tfk_physics_train.npy")
K_FQK_PATH  = os.path.join(PHYSICS_DIR, "K_fqk_physics_train.npy")
K_RBF_PATH  = os.path.join(PHYSICS_DIR, "K_rbf_physics_train.npy")
PHYS16_PATH = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")

FEATURE_INDICES = [0, 5, 6, 3, 10, 15, 9, 11]
FEATURE_NAMES   = ["NDVI", "NDRE", "MNDWI", "BSI",
                   "CrossPol", "NDVI_tex", "SAR_total", "PolCoh"]

# PCA-TFK baselines for comparison
PCA_TFK_KTA      = 0.1825
PCA_TFK_EFF_RANK = 137.81
PCA_TFK_CV       = None   # not recorded — will show as N/A
PCA_FQK_KTA      = 0.1840
PCA_RBF_KTA      = 0.1309
PHYS_FQK_KTA     = 0.2345   # from n=2000 FQK diagnostic
PHYS_RBF_KTA     = 0.2956   # from n=2000 FQK diagnostic

print("=" * 65)
print("  TFK Physics Diagnostic")
print("=" * 65)

# ── 1. Load and check theta ────────────────────────────────────────────────
print("\n--- Section 1: Trained Theta Parameters ---")

if not os.path.exists(THETA_PATH):
    print(f"ERROR: theta not found at {THETA_PATH}")
    print("TFK training may not have completed or saved correctly.")
    sys.exit(1)

theta = np.load(THETA_PATH)
print(f"  theta shape: {theta.shape}  (expected: depends on TFK circuit)")
print(f"  theta values: {np.round(theta, 4)}")
print(f"  theta range:  [{theta.min():.4f}, {theta.max():.4f}]")
print(f"  theta mean:   {theta.mean():.4f}")
print(f"  theta std:    {theta.std():.4f}")

# Saturation check — theta near 0 or π may indicate dead/suppressed parameters
PI = np.pi
near_zero   = np.sum(np.abs(theta) < 0.05)
near_pi     = np.sum(np.abs(np.abs(theta) - PI) < 0.05)
near_halfpi = np.sum(np.abs(np.abs(theta) - PI/2) < 0.05)
print(f"\n  Saturation check:")
print(f"    Near 0       (|θ| < 0.05):     {near_zero}/{len(theta)} params")
print(f"    Near π       (|θ-π| < 0.05):   {near_pi}/{len(theta)} params")
print(f"    Near π/2     (|θ-π/2| < 0.05): {near_halfpi}/{len(theta)} params")

if near_zero > len(theta) // 4:
    print("  WARNING: Many parameters near 0 — possible suppression/dead params")
elif near_pi > len(theta) // 4:
    print("  WARNING: Many parameters near π — possible saturation")
else:
    print("  OK: No widespread saturation detected")

# Per-feature interpretation
# TFK uses parameterized rotations — theta values correspond to rotation angles
# per qubit or per gate depending on the circuit architecture.
# Values in [0.5, 2.5] are in the sensitive regime.
sensitive = np.sum((theta > 0.5) & (theta < 2.5))
print(f"\n  Params in sensitive regime [0.5, 2.5]: {sensitive}/{len(theta)}")
# Note: KTA from user log was 0.5811
print(f"  Final training KTA: 0.5811  (from training log)")


# ── 2. Load and analyse K_tfk ─────────────────────────────────────────────
print("\n--- Section 2: K_tfk Kernel Diagnostics ---")

if not os.path.exists(K_TFK_PATH):
    print(f"  K_tfk not found at {K_TFK_PATH}")
    print("  Kernel computation still running — check 'ps aux'")
    print("  Re-run this diagnostic after exp1_physics.py completes.")
    K_tfk = None
else:
    K_tfk = np.load(K_TFK_PATH)
    n = K_tfk.shape[0]
    print(f"  Shape: {K_tfk.shape}")

    diag = np.diag(K_tfk)
    off_idx = np.triu_indices(n, k=1)
    off = K_tfk[off_idx]

    print(f"  Diagonal: mean={diag.mean():.6f}, std={diag.std():.6f}  (should be ~1.0)")
    print(f"  Off-diag: mean={off.mean():.4f}, std={off.std():.4f}")
    print(f"  Min value: {K_tfk.min():.6f}  (should be ≥ 0)")
    print(f"  Symmetric: {np.allclose(K_tfk, K_tfk.T, atol=1e-5)}")

    cv = float(off.std() / (off.mean() + 1e-12))
    print(f"  CV (off-diag std/mean): {cv:.4f}  (PCA-FQK was ~3.17)")

    # Eigenvalue / effective rank
    eigvals = np.linalg.eigvalsh(K_tfk)
    eigvals_pos = np.maximum(eigvals, 0)
    total = eigvals_pos.sum()
    probs = eigvals_pos / (total + 1e-12)
    eff_rank = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
    top1_frac = float(eigvals_pos[-1] / (total + 1e-12))

    print(f"  Effective rank: {eff_rank:.2f}  "
          f"(PCA-TFK={PCA_TFK_EFF_RANK:.1f}, PCA-FQK=33.24, RBF=2.39)")
    print(f"  Top-1 eig frac: {top1_frac:.4f}")
    print(f"  Min eigenvalue: {eigvals.min():.6f}  (≥0 = PSD confirmed)")

    # Concentration verdict
    print(f"\n  Concentration assessment:")
    if off.mean() < 0.01:
        print("  WARN: off-diag mean very low — possible concentration/collapse")
    elif off.mean() > 0.5:
        print("  WARN: off-diag mean very high — kernel may be too smooth")
    else:
        print(f"  OK: off-diag mean {off.mean():.4f} in healthy range [0.01, 0.5]")

    if eff_rank > 500:
        print("  WARN: eff_rank very high — kernel may be near-identity (collapsed)")
    elif eff_rank < 5:
        print("  WARN: eff_rank very low — kernel is rank-deficient")
    else:
        print(f"  OK: eff_rank {eff_rank:.1f} in reasonable range")


# ── 3. KTA comparison ─────────────────────────────────────────────────────
print("\n--- Section 3: KTA Comparison ---")

# Load labels
if not os.path.exists(PHYS16_PATH):
    print(f"  ERROR: {PHYS16_PATH} not found. Cannot compute KTA.")
    y_train = None
else:
    data16  = np.load(PHYS16_PATH)
    y_train = data16["y_train"]
    n_train = len(y_train)
    print(f"  Labels loaded: {n_train} samples, {len(np.unique(y_train))} classes")

def centered_kta_weighted(K, y):
    """Centered class-weighted KTA."""
    n = len(y)
    col_mean   = K.mean(axis=0)
    row_mean   = K.mean(axis=1)
    grand_mean = K.mean()
    Kc = K - col_mean[None, :] - row_mean[:, None] + grand_mean

    classes, counts = np.unique(y, return_counts=True)
    count_map = dict(zip(classes, counts))
    Ki = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if y[i] == y[j]:
                Ki[i, j] = 1.0 / count_map[y[i]]
    col_m = Ki.mean(axis=0)
    row_m = Ki.mean(axis=1)
    gm    = Ki.mean()
    Kic   = Ki - col_m[None, :] - row_m[:, None] + gm

    denom = np.linalg.norm(Kc, 'fro') * np.linalg.norm(Kic, 'fro')
    return float(np.sum(Kc * Kic) / (denom + 1e-12))

if K_tfk is not None and y_train is not None:
    print(f"\n  Computing KTA for K_tfk ({K_tfk.shape[0]}×{K_tfk.shape[0]})...")
    kta_tfk = centered_kta_weighted(K_tfk, y_train)
    print(f"  Physics-TFK  KTA = {kta_tfk:.4f}   (PCA-TFK  = {PCA_TFK_KTA:.4f})")
    print(f"  Physics-FQK  KTA = {PHYS_FQK_KTA:.4f}   (PCA-FQK  = {PCA_FQK_KTA:.4f})")
    print(f"  Physics-RBF  KTA = {PHYS_RBF_KTA:.4f}   (PCA-RBF  = {PCA_RBF_KTA:.4f})")

    delta_vs_fqk = kta_tfk - PHYS_FQK_KTA
    delta_vs_rbf = kta_tfk - PHYS_RBF_KTA
    print(f"\n  ΔKTA(TFK - FQK_physics) = {delta_vs_fqk:+.4f}")
    print(f"  ΔKTA(TFK - RBF_physics) = {delta_vs_rbf:+.4f}")

    if delta_vs_rbf > 0:
        print("  TFK > RBF on physics features  ← strong result")
    elif delta_vs_fqk > 0:
        print("  TFK > FQK but < RBF on physics  ← training improved over vanilla FQK")
    else:
        print("  TFK < FQK and < RBF on physics  ← training did not help")

    # Also compute KTA for FQK and RBF if available, for full table
    print(f"\n  Full KTA table (physics features, n=2000):")
    print(f"  {'Kernel':<12} {'KTA':>8}  {'ΔKTA vs RBF':>14}")
    print(f"  {'-'*38}")
    for kname, kta_val in [
        ("TFK",    kta_tfk),
        ("FQK",    PHYS_FQK_KTA),
        ("RBF",    PHYS_RBF_KTA),
    ]:
        delta = kta_val - PHYS_RBF_KTA
        marker = " ←" if kname == "TFK" else ""
        print(f"  {kname:<12} {kta_val:>8.4f}  {delta:>+14.4f}{marker}")


# ── 4. Theta vs PCA-TFK comparison ────────────────────────────────────────
print("\n--- Section 4: Theta vs PCA-TFK Comparison ---")
print("  PCA-TFK final theta (from exp1):  not stored numerically")
# Updated from user log
print("  Physics-TFK final theta (min/max from tqdm): [1.48, 2.51]")
print(f"  Physics-TFK theta loaded from file:      {np.round(theta, 4)}")
print()
print("  Interpretation:")
print("  - theta[i] controls rotation angle for qubit i in the TFK circuit")
print("  - values near π/2 ≈ 1.57 are at max sensitivity")
print("  - values [1.48, 2.51] span π/2, suggesting qubits are active")

if len(theta) == 2:
    print(f"\n  2-parameter TFK confirmed (matches PIQFM bridge count)")
    print(f"  theta[0] = {theta[0]:.4f}  ({'near π/2' if abs(theta[0]-PI/2)<0.3 else 'away from π/2'})")
    print(f"  theta[1] = {theta[1]:.4f}  ({'near π/2' if abs(theta[1]-PI/2)<0.3 else 'away from π/2'})")
else:
    print(f"\n  {len(theta)}-parameter TFK")
    for i, t in enumerate(theta):
        label = "near π/2" if abs(abs(t) - PI/2) < 0.3 else (
                "near 0"  if abs(t) < 0.3 else
                "near π"  if abs(abs(t) - PI) < 0.3 else "mid-range")
        feat = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"q{i}"
        print(f"  theta[{i}] = {t:.4f}  ({label})  [{feat}]")


# ── 5. Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)

print(f"\n  Theta: {np.round(theta, 4)}")
print(f"  Training KTA (from log): 0.5811")
print(f"  No suppressed parameters: {'YES' if near_zero == 0 else f'NO ({near_zero} near 0)'}")

if K_tfk is not None and y_train is not None:
    print(f"  K_tfk eff_rank:     {eff_rank:.2f}")
    print(f"  K_tfk off-diag mean:{off.mean():.4f}")
    print(f"  Physics-TFK KTA:    {kta_tfk:.4f}")
    print(f"  ΔKTA(TFK - RBF):    {delta_vs_rbf:+.4f}")

    verdict_lines = []
    if kta_tfk > PHYS_RBF_KTA:
        verdict_lines.append("TFK beats RBF on physics features — strong result for paper")
    elif kta_tfk > PHYS_FQK_KTA:
        verdict_lines.append("TFK beats FQK but not RBF — training improved encoding")
    else:
        verdict_lines.append("TFK below FQK and RBF — theta training did not help on physics")

    if eff_rank > PCA_TFK_EFF_RANK * 1.5:
        verdict_lines.append(f"eff_rank={eff_rank:.0f} much higher than PCA-TFK={PCA_TFK_EFF_RANK:.0f} — richer geometry")
    elif eff_rank < PCA_TFK_EFF_RANK * 0.5:
        verdict_lines.append(f"eff_rank={eff_rank:.0f} much lower than PCA-TFK={PCA_TFK_EFF_RANK:.0f} — concentrated")

    print(f"\n  Verdict:")
    for v in verdict_lines:
        print(f"    - {v}")
else:
    print("\n  K_tfk not yet available — rerun after exp1_physics.py completes")

print()
