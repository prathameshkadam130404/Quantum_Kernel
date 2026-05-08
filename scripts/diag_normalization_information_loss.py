"""
diag_normalization_information_loss.py

Checks whether per-feature [0,π] normalization destroys discriminative
information that should be used for feature selection and attention.

Three-part analysis:
  Part 1: What do raw features look like? Natural scales, natural ranges.
  Part 2: What is lost in per-feature normalization?
           Fisher ratios raw vs normalized, correlation structure raw vs normalized.
  Part 3: What should be done differently?
           Recommendation on normalization strategy for each pipeline stage.

Key question: should feature selection, Fisher computation, and attention
use raw features (preserving natural distinctiveness) while only quantum
encoding uses normalized features?

Usage:
    python scripts/diag_normalization_information_loss.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import config

# ── Load data ──────────────────────────────────────────────────────────────
PHYS16_PATH  = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
PHYS8_PATH   = os.path.join(config.PROCESSED_DIR, "physics_features.npz")

FEATURE_NAMES_16 = [
    "NDVI", "NDBI", "NDWI", "BSI", "SAVI", "NDRE", "MNDWI", "EVI",
    "VV/VH", "SAR_total", "CrossPol", "PolCoh",
    "VH_dB", "VV_dB", "VV_tex", "NDVI_tex"
]
FEATURE_MODALITY_16 = [
    "OPT","OPT","OPT","OPT","OPT","OPT","OPT","OPT",
    "SAR","SAR","SAR","SAR","SAR","SAR","SAR","SAR"
]

print("=" * 70)
print("  Normalization Information Loss Diagnostic")
print("=" * 70)

# ── Check what keys exist in each file ────────────────────────────────────
print("\n--- File Contents ---")

data16 = np.load(PHYS16_PATH)
print(f"\n  physics_features_16.npz keys: {list(data16.files)}")

has_raw_16 = "X_train_raw" in data16.files
has_norm_lo = "normalization_lo" in data16.files

if os.path.exists(PHYS8_PATH):
    data8 = np.load(PHYS8_PATH)
    print(f"  physics_features.npz keys:    {list(data8.files)}")
    has_raw_8 = "X_train_raw" in data8.files
else:
    data8 = None
    has_raw_8 = False

# Load normalized 16-feature set (always available)
X16_norm = data16["X_train"]   # (2000, 16), in [0, π]
y_train  = data16["y_train"]
n, d     = X16_norm.shape

print(f"\n  Normalized features: shape={X16_norm.shape}, "
      f"range=[{X16_norm.min():.4f}, {X16_norm.max():.4f}]")

# Try to load raw features
X16_raw = None
if has_raw_16:
    X16_raw = data16["X_train_raw"]
    print(f"  Raw features (16):   shape={X16_raw.shape}, "
          f"range=[{X16_raw.min():.4f}, {X16_raw.max():.4f}]")
elif has_raw_8 and data8 is not None:
    print(f"  Raw features (8):    available in physics_features.npz")
    X8_raw = data8["X_train_raw"]
    print(f"  Raw 8-feature range: [{X8_raw.min():.4f}, {X8_raw.max():.4f}]")
elif has_norm_lo:
    # Reconstruct approximate raw from normalization parameters
    lo = data16["normalization_lo"]
    hi = data16["normalization_hi"]
    # Reverse: raw ≈ (norm / π) * (hi - lo) + lo
    X16_raw = (X16_norm / np.pi) * (hi - lo) + lo
    print(f"  Raw features (reconstructed from normalization params)")
    print(f"  Reconstructed range: [{X16_raw.min():.4f}, {X16_raw.max():.4f}]")
else:
    print("  WARNING: Raw features not available in any file.")
    print("  Will only analyze normalized features.")
    print("  To get raw features, re-run extract_physics_features.py")
    print("  (it saves X_train_raw alongside X_train)")


# ── Part 1: Natural scales of raw features ────────────────────────────────
print("\n" + "=" * 70)
print("  PART 1: Natural Feature Scales (Raw vs Normalized)")
print("=" * 70)

print(f"\n  {'Feature':<12} {'Mod':>4}  "
      f"{'Raw Min':>9} {'Raw Max':>9} {'Raw Std':>9} {'Raw Range':>10}  "
      f"{'Norm Std':>9}")
print("  " + "-" * 70)

for i in range(d):
    norm_std = X16_norm[:, i].std()
    if X16_raw is not None:
        raw_min = X16_raw[:, i].min()
        raw_max = X16_raw[:, i].max()
        raw_std = X16_raw[:, i].std()
        raw_range = raw_max - raw_min
        print(f"  {FEATURE_NAMES_16[i]:<12} {FEATURE_MODALITY_16[i]:>4}  "
              f"{raw_min:>9.4f} {raw_max:>9.4f} {raw_std:>9.4f} {raw_range:>10.4f}  "
              f"{norm_std:>9.4f}")
    else:
        print(f"  {FEATURE_NAMES_16[i]:<12} {FEATURE_MODALITY_16[i]:>4}  "
              f"{'N/A':>9} {'N/A':>9} {'N/A':>9} {'N/A':>10}  "
              f"{norm_std:>9.4f}")

print(f"\n  After per-feature normalization to [0,π]:")
print(f"  Every feature has the same range [0, {np.pi:.4f}].")
print(f"  Natural scale differences between features are ERASED.")


# ── Part 2: Fisher ratio raw vs normalized ────────────────────────────────
print("\n" + "=" * 70)
print("  PART 2: Fisher Ratio — Raw vs Normalized")
print("=" * 70)

def fisher_per_feature(X, y):
    classes    = np.unique(y)
    grand_mean = X.mean(axis=0)
    between    = np.zeros(X.shape[1])
    within     = np.zeros(X.shape[1])
    for c in classes:
        mask = y == c
        nc   = mask.sum()
        mu_c = X[mask].mean(axis=0)
        between += nc * (mu_c - grand_mean) ** 2
        within  += ((X[mask] - mu_c) ** 2).sum(axis=0)
    return between / np.maximum(within, 1e-12)

fisher_norm = fisher_per_feature(X16_norm, y_train)

if X16_raw is not None:
    fisher_raw = fisher_per_feature(X16_raw, y_train)
    print(f"\n  {'Feature':<12} {'Mod':>4}  "
          f"{'Fisher Raw':>12} {'Fisher Norm':>12} {'Difference':>12}  Change")
    print("  " + "-" * 60)
    for i in range(d):
        diff   = fisher_norm[i] - fisher_raw[i]
        change = diff / (fisher_raw[i] + 1e-12) * 100
        marker = " ↑" if change > 5 else (" ↓" if change < -5 else "  ≈")
        print(f"  {FEATURE_NAMES_16[i]:<12} {FEATURE_MODALITY_16[i]:>4}  "
              f"{fisher_raw[i]:>12.4f} {fisher_norm[i]:>12.4f} "
              f"{diff:>+12.4f} {change:>+6.1f}%{marker}")
    print(f"\n  Mean Fisher (raw):  {fisher_raw.mean():.4f}")
    print(f"  Mean Fisher (norm): {fisher_norm.mean():.4f}")
    print(f"  Top feature raw:    {FEATURE_NAMES_16[np.argmax(fisher_raw)]} "
          f"({fisher_raw.max():.4f})")
    print(f"  Top feature norm:   {FEATURE_NAMES_16[np.argmax(fisher_norm)]} "
          f"({fisher_norm.max():.4f})")
else:
    print(f"\n  Fisher (normalized only):")
    for i in range(d):
        print(f"  {FEATURE_NAMES_16[i]:<12} {FEATURE_MODALITY_16[i]:>4}  "
              f"{fisher_norm[i]:.4f}")


# ── Part 3: Correlation structure raw vs normalized ───────────────────────
print("\n" + "=" * 70)
print("  PART 3: Cross-Feature Correlation Structure")
print("=" * 70)

def compute_correlation(X):
    mu  = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xz  = (X - mu) / std
    return (Xz.T @ Xz) / len(X)

corr_norm = compute_correlation(X16_norm)

print(f"\n  Correlation matrix stats (normalized features):")
off_diag_norm = corr_norm[np.triu_indices(d, k=1)]
print(f"    Off-diagonal: mean={off_diag_norm.mean():.4f}, "
      f"std={off_diag_norm.std():.4f}, "
      f"max={off_diag_norm.max():.4f}")

# Highlight strongest cross-modal correlations
print(f"\n  Strongest cross-modal correlations (normalized):")
cross_pairs = []
for i in range(d):
    for j in range(i+1, d):
        if FEATURE_MODALITY_16[i] != FEATURE_MODALITY_16[j]:
            cross_pairs.append((i, j, corr_norm[i, j]))
cross_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for i, j, c in cross_pairs[:6]:
    print(f"    {FEATURE_NAMES_16[i]:<10} ↔ {FEATURE_NAMES_16[j]:<10}  "
          f"corr={c:+.4f}")

if X16_raw is not None:
    corr_raw = compute_correlation(X16_raw)
    off_diag_raw = corr_raw[np.triu_indices(d, k=1)]
    print(f"\n  Correlation matrix stats (raw features):")
    print(f"    Off-diagonal: mean={off_diag_raw.mean():.4f}, "
          f"std={off_diag_raw.std():.4f}, "
          f"max={off_diag_raw.max():.4f}")

    print(f"\n  Strongest cross-modal correlations (raw):")
    cross_raw = []
    for i in range(d):
        for j in range(i+1, d):
            if FEATURE_MODALITY_16[i] != FEATURE_MODALITY_16[j]:
                cross_raw.append((i, j, corr_raw[i, j]))
    cross_raw.sort(key=lambda x: abs(x[2]), reverse=True)
    for i, j, c in cross_raw[:6]:
        print(f"    {FEATURE_NAMES_16[i]:<10} ↔ {FEATURE_NAMES_16[j]:<10}  "
              f"corr={c:+.4f}")

    # Check if correlation structure changed significantly
    corr_diff = np.abs(corr_norm - corr_raw)
    print(f"\n  Correlation structure change (|norm_corr - raw_corr|):")
    print(f"    Mean change:  {corr_diff.mean():.4f}")
    print(f"    Max change:   {corr_diff.max():.4f}")
    print(f"    Significantly different (>0.1): "
          f"{(corr_diff > 0.1).sum()} of {d*d} entries")


# ── Part 4: Recommendation ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  PART 4: Recommendation")
print("=" * 70)

print("""
  Per-feature normalization to [0,π] is CORRECT for quantum encoding.
  The circuit needs features in a bounded range — this cannot change.

  But per-feature normalization is WRONG for these uses:
    1. Fisher ratio computation — should use raw features
       (raw Fisher reflects true class discriminability)
    2. Attention matrix — should use raw or z-scored features
       (raw correlation reflects true physical relationships)
    3. Feature selection — should use raw Fisher, not normalized Fisher

  The fix: use a TWO-STAGE pipeline:
    Stage A (analysis): raw features → Fisher → feature selection → attention
    Stage B (encoding): selected features → per-feature normalize to [0,γπ]

  This ensures:
    - Feature selection is based on true physical discriminability
    - Attention captures true cross-feature physical relationships
    - Quantum encoding still operates in the correct bounded range
    - Per-qubit γ from raw Fisher has correct relative magnitudes

  Concretely in src/attention_kernel.py:
    select_features_by_fisher(X_raw, y)  ← use raw features
    compute_attention_matrix(X_raw)       ← use raw features
    get_attention_entanglement_pairs(...)  ← from raw attention

  And in scripts/compute_agpqk_kernel.py:
    X_for_analysis = X16_raw[:, SELECTED_INDICES]   ← raw
    X_for_encoding = normalize_selected(X_for_analysis, gamma_per_qubit)  ← encode
""")

# ── Part 5: Does physics_features_16.npz have raw? ────────────────────────
print("=" * 70)
print("  PART 5: Action Required")
print("=" * 70)

if X16_raw is not None and not has_raw_16:
    print("""
  Raw features were reconstructed from normalization parameters.
  Quality of reconstruction: approximate (round-trip through π scaling).
  
  For exact raw features: physics_features_16.npz needs to be regenerated
  with X_train_raw saved alongside X_train.
  
  Modify scripts/extract_physics_features_16.py (or equivalent) to add:
    np.savez(output_file,
        X_train=features_norm,
        X_train_raw=features_raw,   ← ADD THIS
        X_test=...,
        ...)
""")
elif has_raw_16:
    print("""
  Raw 16-feature data is already saved in physics_features_16.npz.
  No regeneration needed.
  
  Update src/attention_kernel.py and scripts/compute_agpqk_kernel.py to:
    1. Load X_train_raw from physics_features_16.npz
    2. Use X_raw for Fisher + attention computation
    3. Use X_norm for quantum circuit encoding only
""")
else:
    print("""
  Raw 16-feature data is NOT available in physics_features_16.npz.
  
  To fix this properly, re-run the 16-feature extraction script and
  ensure it saves X_train_raw.
  
  In the meantime, check if physics_features.npz (8-feature) has raw
  data — it does save X_train_raw (confirmed in extract_physics_features.py).
  
  Short-term workaround: use physics_features.npz X_train_raw for the
  8 features that overlap with AGPQK's selected features.
""")

print("\n  Run this to check what normalization parameters are saved:")
print("  python -c \"import numpy as np; d=np.load('data/processed/"
      "physics_features_16.npz'); print(d.files)\"")
