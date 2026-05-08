"""
verify_d4_invariance_fast.py — Fast D4OrbitPCA invariance check.

Does NOT require loading HDF5 raw patches.
Uses synthetic raw patches to verify the orbit-based PCA invariance property
mathematically. If the orbit construction is correct, invariance holds for
ANY input including synthetic — the guarantee is structural, not data-dependent.

Also verifies that standard FQK on D4OrbitPCA features equals the group-averaged
D4_FQK analytically (no circuits needed), proving the 8x computation is redundant.

Runtime: < 2 minutes (pure numpy, no quantum circuits).

Usage:
    python scripts/verify_d4_invariance_fast.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import joblib
import config

from src.d4_augmented_pca import (
    D4OrbitPCA, verify_invariance,
    apply_d4_to_flat_patch, D4_ELEMENTS,
    fit_d4_augmented_pca,
)
from src.d4_representation import get_d4_pixel_permutation

RAW_DIM   = 32 * 32 * 8   # 8192 — flat SAR patch
N_QUBITS  = config.N_QUBITS  # 8
HEIGHT    = 32
WIDTH     = 32
N_CHAN    = 8

PCA_PATH  = os.path.join(config.RESULTS_DIR, "d4_standalone", "sar_d4_pca_model.joblib")

print("=" * 65)
print("  D4OrbitPCA Invariance Verification (Synthetic Patches)")
print("=" * 65)


# ── Step 1: Load or fit D4OrbitPCA ────────────────────────────────────────
print("\n--- Step 1: Load or Fit D4OrbitPCA ---")

if os.path.exists(PCA_PATH):
    pca = joblib.load(PCA_PATH)
    print(f"  Loaded: {PCA_PATH}")
    print(f"  Components shape: {pca.components_.shape}  "
          f"(expected (8, 8192))")

    if pca.components_.shape != (N_QUBITS, RAW_DIM):
        print(f"  ERROR: unexpected shape {pca.components_.shape}")
        sys.exit(1)
else:
    print(f"  PCA model not found at {PCA_PATH}")
    print(f"  Fitting from synthetic data (100 samples)...")

    rng = np.random.RandomState(42)
    X_synthetic = rng.uniform(0, 1, (100, RAW_DIM)).astype(np.float32)
    pca, _, _ = fit_d4_augmented_pca(
        X_synthetic, n_components=N_QUBITS,
        height=HEIGHT, width=WIDTH, n_channels=N_CHAN,
        normalize_output=False,
    )
    print(f"  Fitted synthetic D4OrbitPCA: {pca.components_.shape}")


# ── Step 2: Generate synthetic raw patches ────────────────────────────────
print("\n--- Step 2: Generating Synthetic Raw Patches ---")

rng = np.random.RandomState(42)
N_TEST = 10
X_raw = rng.uniform(0, 1, (N_TEST, RAW_DIM)).astype(np.float64)
print(f"  {N_TEST} synthetic patches, shape {X_raw.shape}")

# Compute normalization params from synthetic data
X_pca_raw = pca.transform(X_raw)
x_min   = X_pca_raw.min(axis=0)
x_max   = X_pca_raw.max(axis=0)
x_range = x_max - x_min
x_range[x_range < 1e-10] = 1.0


# ── Step 3: Verify D4 invariance ──────────────────────────────────────────
print("\n--- Step 3: D4 Invariance Test ---")
print("  Testing: theta(g·x) == theta(x) for all g in D4")
print(f"  D4 elements: {D4_ELEMENTS}")
print()

passed, max_error = verify_invariance(
    pca_components=pca.components_,
    X_sample_raw=X_raw,
    height=HEIGHT,
    width=WIDTH,
    n_channels=N_CHAN,
    tol=1e-4,
    x_min=x_min,
    x_range=x_range,
)

print(f"  Max invariance error across all D4 elements: {max_error:.2e}")
print(f"  Tolerance: 1e-4")
print(f"  Result: {'PASS ✓' if passed else 'FAIL ✗'}")


# ── Step 4: Prove standard FQK == group-averaged FQK analytically ─────────
print("\n--- Step 4: Analytical Equivalence Proof ---")
print("  Claim: K_FQK(theta(x), theta(y)) == K_D4(x_raw, y_raw)")
print("  where K_D4 = (1/8) Σ_g K_FQK(theta(g·x_raw), theta(y_raw))")
print()
print("  Proof:")
print("  Since theta(g·x_raw) = theta(x_raw) for all g (verified above),")
print("  K_D4(x_raw, y_raw)")
print("    = (1/8) Σ_g K_FQK(theta(g·x_raw), theta(y_raw))")
print("    = (1/8) Σ_g K_FQK(theta(x_raw), theta(y_raw))")
print("    = (1/8) × 8 × K_FQK(theta(x_raw), theta(y_raw))")
print("    = K_FQK(theta(x_raw), theta(y_raw))")
print()

if passed:
    print("  CONCLUSION: The 8× group averaging is REDUNDANT.")
    print("  Standard FQK on D4OrbitPCA features IS the D4-covariant kernel.")
    print("  Compute savings: 8 kernel computations → 1 kernel computation.")
    print("  Time savings: ~192 hours → ~24 hours.")
else:
    print("  WARNING: Invariance failed — analytical equivalence does NOT hold.")
    print("  Group averaging may still be needed. Investigate error source.")


# ── Step 5: Verify orbit basis orthonormality ─────────────────────────────
print("\n--- Step 5: Orbit Basis Properties ---")

C = pca.components_   # (8, 8192)

# Check rows are unit norm
row_norms = np.linalg.norm(C, axis=1)
print(f"  Row norms (should be 1.0): min={row_norms.min():.6f}, "
      f"max={row_norms.max():.6f}")
norm_ok = np.allclose(row_norms, 1.0, atol=1e-6)
print(f"  Unit norm: {'OK ✓' if norm_ok else 'FAIL ✗'}")

# Check rows are orthogonal (approximate — orbit basis may not be exactly orthogonal)
gram = C @ C.T   # (8, 8)
off_diag = gram - np.diag(np.diag(gram))
max_off = np.abs(off_diag).max()
print(f"  Max off-diagonal of C·C.T: {max_off:.6f}  "
      f"({'near-orthogonal ✓' if max_off < 0.1 else 'not orthogonal ✗'})")

# Check D4 invariance of each component vector
print(f"\n  Per-component D4 invariance (||C[i, perm_g] - C[i, :]||):")
for i in range(N_QUBITS):
    errors = []
    for elem in D4_ELEMENTS[1:]:   # skip identity
        spatial_perm = get_d4_pixel_permutation(elem, HEIGHT, WIDTH)
        full_perm = np.concatenate([
            spatial_perm + ch * (HEIGHT * WIDTH) for ch in range(N_CHAN)
        ])
        err = float(np.max(np.abs(C[i, full_perm] - C[i, :])))
        errors.append(err)
    max_comp_err = max(errors)
    status = "✓" if max_comp_err < 1e-10 else "✗"
    print(f"    Component {i}: max_err={max_comp_err:.2e}  {status}")


# ── Step 6: Runtime comparison ────────────────────────────────────────────
print("\n--- Step 6: Compute Cost Analysis ---")
print()

gpu_speed_its = 69   # it/s from your RTX 4050 measurements
n_train = 2000
n_test  = 2000
n_pairs_train = n_train * (n_train - 1) // 2 + n_train  # upper triangle + diag
n_pairs_test  = n_train * n_test

time_one_kernel_train = n_pairs_train / gpu_speed_its / 3600   # hours
time_one_kernel_test  = n_pairs_test  / gpu_speed_its / 3600

print(f"  GPU speed: ~{gpu_speed_its} it/s (RTX 4050, from FQK logs)")
print(f"  n_train={n_train}, n_test={n_test}")
print()
print(f"  One FQK (train):      ~{time_one_kernel_train:.1f} hours")
print(f"  One FQK (test):       ~{time_one_kernel_test:.1f} hours")
print()
print(f"  Option A — Standard FQK on D4OrbitPCA features (RECOMMENDED):")
print(f"    Train + Test: ~{(time_one_kernel_train + time_one_kernel_test):.1f} hours")
print(f"    D4-covariant: YES (by orbit construction, max_err={max_error:.2e})")
print()
print(f"  Option B — Full D4 group-averaged FQK (current plan):")
print(f"    Train: 8 × {time_one_kernel_train:.1f} = ~{8*time_one_kernel_train:.0f} hours")
print(f"    Test:  8 × {time_one_kernel_test:.1f}  = ~{8*time_one_kernel_test:.0f} hours")
print(f"    Total: ~{8*(time_one_kernel_train + time_one_kernel_test):.0f} hours")
print(f"    D4-covariant: YES (mathematically identical to Option A if invariance holds)")
print()

if passed:
    savings_pct = (1 - 1/8) * 100
    print(f"  RECOMMENDATION: Use Option A.")
    print(f"  Savings: {savings_pct:.0f}% compute reduction, zero loss of correctness.")
    print(f"  Paper claim: 'D4-invariant PCA encoding achieves group-covariant kernel")
    print(f"  without explicit group averaging (O(n²) vs O(|G|n²)).'")
    print(f"  This is itself a contribution: efficient covariant kernel construction.")
else:
    print(f"  RECOMMENDATION: Use Option B — invariance failed (max_err={max_error:.2e}).")
    print(f"  Investigate why orbit construction did not achieve machine-precision invariance.")


# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  SUMMARY")
print(f"{'='*65}")
print(f"  D4 invariance: {'PASS' if passed else 'FAIL'}  (max_error={max_error:.2e})")
print(f"  Group averaging redundant: {'YES' if passed else 'NO'}")
print(f"  Recommended approach: {'Option A (1 FQK)' if passed else 'Option B (8 FQK)'}")
print(f"  Estimated time saved: "
      f"{'~{:.0f} hours'.format(7*(time_one_kernel_train+time_one_kernel_test)) if passed else '0'}")
print()
