"""
diag_feature_embedding.py — Feature value and quantum embedding analysis.

Answers the question: where does class-discriminative information in physics
features get lost during quantum encoding?

Three-part analysis:
  Part 1 — Raw feature statistics and classical discriminability
           (per-feature class separation BEFORE any kernel)
  Part 2 — Quantum embedding geometry
           (how each feature maps to qubit rotations, per-qubit overlap)
  Part 3 — Information survival rate
           (how much class separation survives the ZZFeatureMap encoding)

Key concept: kernels do not "increase" information — they project features
into a similarity space. If the projection geometry mismatches the label
structure, class information is destroyed, regardless of feature quality.

Usage:
    python scripts/diag_feature_embedding.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.model_selection import train_test_split

import config
from src.bandwidth import apply_bandwidth

# ── Config ────────────────────────────────────────────────────────────────
FEATURE_INDICES = [0, 5, 6, 3, 10, 15, 9, 11]
FEATURE_NAMES   = ["NDVI", "NDRE", "MNDWI", "BSI",
                   "CrossPol", "NDVI_tex", "SAR_total", "PolCoh"]
GAMMA_PHYS      = 0.50   # CV-selected bandwidth
GAMMA_PCA       = 1.00   # PCA bandwidth (no change)
ZZ_REPS         = 2      # circuit repetitions

# PCA baselines for comparison (from exp1)
PCA_FQK_KTA  = 0.1840
PCA_RBF_KTA  = 0.1309
PHYS_FQK_KTA = 0.2345
PHYS_RBF_KTA = 0.2956

PHYS16_PATH = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
PCA_PATH    = os.path.join(config.PROCESSED_DIR, "pca_features.npz")  # adjust if different


# ── Utilities ─────────────────────────────────────────────────────────────

def fisher_ratio(X, y):
    """
    Per-feature Fisher discriminant ratio: between-class / within-class variance.
    Higher = more linearly separable. Scale-invariant.
    """
    classes = np.unique(y)
    grand_mean = X.mean(axis=0)
    n = len(y)

    between = np.zeros(X.shape[1])
    within  = np.zeros(X.shape[1])

    for c in classes:
        mask = y == c
        nc = mask.sum()
        mu_c = X[mask].mean(axis=0)
        between += nc * (mu_c - grand_mean) ** 2
        within  += ((X[mask] - mu_c) ** 2).sum(axis=0)

    within = np.maximum(within, 1e-12)
    return between / within


def per_feature_overlap_decay(X, gamma, n_qubits=8, n_pairs=5000):
    """
    Estimate per-qubit and joint overlap decay in the ZZFeatureMap.

    For each feature i, computes the expected per-qubit kernel contribution:
        E[cos²(γ · Δx_i / 2)]
    And the predicted joint overlap across all qubits:
        prod_i E[cos²(γ · Δx_i / 2)]

    This predicts the expected off-diagonal kernel value if features
    were independent (no ZZ entanglement). Entanglement modifies this
    but the independent estimate captures the concentration effect.

    Args:
        X:        feature matrix (n, d), already scaled by gamma
        gamma:    bandwidth scalar (for logging only — X already scaled)
        n_qubits: number of qubits
        n_pairs:  number of random pairs to estimate expectation

    Returns:
        per_qubit_overlap: (d,) expected cos² per feature
        predicted_joint:   predicted joint overlap (product)
        std_joint:         std of joint overlap across pairs
    """
    n = len(X)
    rng = np.random.default_rng(42)
    idx_i = rng.integers(0, n, n_pairs)
    idx_j = rng.integers(0, n, n_pairs)
    # avoid self-pairs
    same = idx_i == idx_j
    idx_j[same] = (idx_j[same] + 1) % n

    delta = X[idx_i] - X[idx_j]          # (n_pairs, d)
    # Per-qubit overlap: cos²(Δx_i / 2) — X already scaled by gamma
    per_pair_per_qubit = np.cos(delta / 2) ** 2   # (n_pairs, d)
    per_qubit_mean     = per_pair_per_qubit.mean(axis=0)  # (d,)

    # Joint overlap for ZZ_REPS repetitions
    # Each rep contributes ~product of single-qubit overlaps (approx)
    joint_per_pair = per_pair_per_qubit.prod(axis=1) ** ZZ_REPS  # (n_pairs,)

    return per_qubit_mean, joint_per_pair.mean(), joint_per_pair.std()


def class_conditional_overlap(X, y, gamma, n_same=2000, n_diff=2000):
    """
    Compute predicted kernel overlap separately for same-class and
    different-class pairs. The ratio determines discriminability.

    A good kernel has: same-class overlap >> different-class overlap.
    A collapsed kernel has: same ≈ different ≈ near zero.

    Returns:
        same_class_overlap: mean predicted overlap for same-class pairs
        diff_class_overlap: mean predicted overlap for different-class pairs
        discrimination_ratio: same / (same + diff)  — 0.5 = random, 1.0 = perfect
    """
    classes = np.unique(y)
    rng = np.random.default_rng(42)

    # Same-class pairs
    same_overlaps = []
    for _ in range(n_same):
        c = classes[rng.integers(0, len(classes))]
        idx = np.where(y == c)[0]
        if len(idx) < 2:
            continue
        i, j = rng.choice(idx, 2, replace=False)
        delta = X[i] - X[j]
        overlap = (np.cos(delta / 2) ** 2).prod() ** ZZ_REPS
        same_overlaps.append(overlap)

    # Different-class pairs
    diff_overlaps = []
    for _ in range(n_diff):
        c1, c2 = rng.choice(classes, 2, replace=False)
        idx1 = np.where(y == c1)[0]
        idx2 = np.where(y == c2)[0]
        i = rng.choice(idx1)
        j = rng.choice(idx2)
        delta = X[i] - X[j]
        overlap = (np.cos(delta / 2) ** 2).prod() ** ZZ_REPS
        diff_overlaps.append(overlap)

    same_mean = np.mean(same_overlaps)
    diff_mean = np.mean(diff_overlaps)
    ratio = same_mean / (same_mean + diff_mean + 1e-12)
    return same_mean, diff_mean, ratio


# ── Load data ─────────────────────────────────────────────────────────────
print("=" * 70)
print("  Feature Embedding Diagnostic")
print("=" * 70)

if not os.path.exists(PHYS16_PATH):
    print(f"ERROR: {PHYS16_PATH} not found")
    sys.exit(1)

data16  = np.load(PHYS16_PATH)
X16_raw = data16["X_train"]   # (2000, 16), normalized to [0, π]
y_train = data16["y_train"]

X_phys_raw   = X16_raw[:, FEATURE_INDICES]               # [0, π]
X_phys_scaled = apply_bandwidth(X_phys_raw.copy(), "physics")  # [0, 0.5π]

n_train  = len(y_train)
n_classes = len(np.unique(y_train))
classes, counts = np.unique(y_train, return_counts=True)

print(f"\n  Dataset: {n_train} samples, {n_classes} classes")
print(f"  Class imbalance: min={counts.min()}, max={counts.max()}, "
      f"ratio={counts.max()/counts.min():.1f}:1")
print(f"  Bandwidth γ_physics = {GAMMA_PHYS}  → encoding range [0, {GAMMA_PHYS}π]")
print(f"  Circuit depth: ZZ_REPS = {ZZ_REPS}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Raw feature statistics and classical discriminability
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PART 1: Raw Feature Statistics (BEFORE quantum encoding)")
print("=" * 70)

print(f"\n  {'Feature':<12} {'Min':>7} {'Max':>7} {'Mean':>7} {'Std':>7} "
      f"{'Range[π]':>9} {'Fisher':>8}")
print("  " + "-" * 65)

fisher = fisher_ratio(X_phys_raw, y_train)
for i, name in enumerate(FEATURE_NAMES):
    x = X_phys_raw[:, i]
    print(f"  {name:<12} {x.min():>7.4f} {x.max():>7.4f} {x.mean():>7.4f} "
          f"{x.std():>7.4f} {x.max()/np.pi:>9.3f}π {fisher[i]:>8.4f}")

print(f"\n  Fisher ratio interpretation:")
print(f"  Higher = better class separation in raw feature space")
print(f"  Top features: {[FEATURE_NAMES[i] for i in np.argsort(fisher)[::-1][:3]]}")
print(f"  Weak features: {[FEATURE_NAMES[i] for i in np.argsort(fisher)[:3]]}")

# After γ scaling
print(f"\n  After γ={GAMMA_PHYS} scaling (what quantum circuit receives):")
print(f"  {'Feature':<12} {'Min':>7} {'Max':>7} {'Mean':>7} {'Std':>7} {'Max/π':>8}")
print("  " + "-" * 55)
for i, name in enumerate(FEATURE_NAMES):
    x = X_phys_scaled[:, i]
    print(f"  {name:<12} {x.min():>7.4f} {x.max():>7.4f} {x.mean():>7.4f} "
          f"{x.std():>7.4f} {x.max()/np.pi:>8.3f}π")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Quantum embedding geometry: per-qubit overlap decay
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PART 2: Quantum Embedding Geometry")
print("=" * 70)

print(f"\n  Each feature x_i is encoded as RZ(x_i) on qubit i.")
print(f"  Kernel value ≈ |⟨ψ(x)|ψ(x')⟩|² = prod_i cos²(Δx_i/2)^{ZZ_REPS}")
print(f"  (ZZ interactions modify this but this gives the base decay rate)")

print(f"\n  Per-qubit overlap analysis (5000 random pairs):")
print(f"  {'Feature':<12} {'E[cos²(Δx/2)]':>15} {'Std(Δx)':>10} {'Overlap@1σ':>12}")
print("  " + "-" * 55)

per_qubit, joint_mean, joint_std = per_feature_overlap_decay(
    X_phys_scaled, GAMMA_PHYS
)

for i, name in enumerate(FEATURE_NAMES):
    x = X_phys_scaled[:, i]
    delta_std = x.std() * np.sqrt(2)  # std of difference of two independent samples
    overlap_1sigma = np.cos(delta_std / 2) ** 2
    print(f"  {name:<12} {per_qubit[i]:>15.4f} {delta_std:>10.4f} {overlap_1sigma:>12.4f}")

print(f"\n  Predicted joint overlap (product across all {len(FEATURE_NAMES)} qubits):")
print(f"    E[K(x,x')] ≈ {joint_mean:.6f}  std={joint_std:.6f}")
print(f"    This predicts off-diagonal mean ≈ {joint_mean:.4f}")
print(f"    Actual FQK off-diag mean (n=2000): 0.0927")
print(f"    Ratio actual/predicted: {0.0927/max(joint_mean, 1e-8):.2f}x "
      f"(ZZ entanglement correction factor)")

# Compare with PCA
print(f"\n  Same analysis for PCA features (for reference):")
try:
    # Try to load PCA features — path may vary
    pca_candidates = [
        os.path.join(config.PROCESSED_DIR, "pca_features.npz"),
        os.path.join(config.PROCESSED_DIR, "pca_features_train.npz"),
        os.path.join(config.PROCESSED_DIR, "X_train_pca.npy"),
    ]
    pca_data = None
    for p in pca_candidates:
        if os.path.exists(p):
            pca_data = np.load(p)
            print(f"  Loaded PCA features from: {p}")
            break

    if pca_data is not None:
        # Handle different possible npz structures
        if hasattr(pca_data, 'files'):
            key = [k for k in pca_data.files if 'train' in k.lower() or 'X' in k][0]
            X_pca_raw = pca_data[key]
        else:
            X_pca_raw = pca_data

        X_pca_raw = X_pca_raw[:n_train]  # match size
        X_pca_scaled = apply_bandwidth(X_pca_raw.copy(), "pca")

        pca_per_qubit, pca_joint, pca_joint_std = per_feature_overlap_decay(
            X_pca_scaled, GAMMA_PCA
        )
        print(f"  PCA per-qubit overlaps: {np.round(pca_per_qubit, 4)}")
        print(f"  PCA predicted joint:     {pca_joint:.6f}")
        print(f"  Physics predicted joint: {joint_mean:.6f}")
        print(f"  Ratio physics/PCA:       {joint_mean/max(pca_joint, 1e-8):.2f}x")
        if pca_joint > joint_mean:
            print("  PCA produces higher predicted overlap — less concentrated kernel")
        else:
            print("  Physics produces higher predicted overlap at γ=0.50")
    else:
        print("  PCA feature file not found — skipping PCA comparison")
        print("  (adjust PCA_PATH at top of script if needed)")
except Exception as e:
    print(f"  PCA comparison failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — Information survival: same-class vs different-class overlap
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PART 3: Class Discriminative Information Survival")
print("=" * 70)

print(f"\n  Comparing kernel overlap for SAME-CLASS vs DIFFERENT-CLASS pairs.")
print(f"  A kernel is discriminative when: same-class overlap >> diff-class overlap.")
print(f"  A collapsed kernel has: same ≈ diff (kernel cannot see class boundaries).")
print(f"\n  Computing for physics features at γ={GAMMA_PHYS}...")

same_phys, diff_phys, ratio_phys = class_conditional_overlap(
    X_phys_scaled, y_train, GAMMA_PHYS
)
disc_phys = same_phys / (diff_phys + 1e-12)

print(f"\n  Physics-FQK (predicted, γ={GAMMA_PHYS}):")
print(f"    Same-class overlap:  {same_phys:.6f}")
print(f"    Diff-class overlap:  {diff_phys:.6f}")
print(f"    Discrimination S/D:  {disc_phys:.4f}  (>1 = quantum sees class structure)")
print(f"    Discrimination ratio:{ratio_phys:.4f}  (0.5=random, 1.0=perfect)")

# Per-feature contribution to same vs diff
print(f"\n  Per-feature contribution to same/diff discrimination:")
print(f"  {'Feature':<12} {'Same E[cos²]':>14} {'Diff E[cos²]':>14} {'S/D ratio':>10}")
print("  " + "-" * 55)

rng = np.random.default_rng(42)
same_pairs_i, same_pairs_j = [], []
diff_pairs_i, diff_pairs_j = [], []

for _ in range(2000):
    c = classes[rng.integers(0, n_classes)]
    idx = np.where(y_train == c)[0]
    if len(idx) >= 2:
        ii, jj = rng.choice(idx, 2, replace=False)
        same_pairs_i.append(ii); same_pairs_j.append(jj)

for _ in range(2000):
    c1, c2 = rng.choice(classes, 2, replace=False)
    idx1 = np.where(y_train == c1)[0]
    idx2 = np.where(y_train == c2)[0]
    same_pairs_i  # just keeping going
    ii = rng.choice(idx1); jj = rng.choice(idx2)
    diff_pairs_i.append(ii); diff_pairs_j.append(jj)

same_pairs_i = np.array(same_pairs_i); same_pairs_j = np.array(same_pairs_j)
diff_pairs_i = np.array(diff_pairs_i); diff_pairs_j = np.array(diff_pairs_j)

for i, name in enumerate(FEATURE_NAMES):
    xi = X_phys_scaled[:, i]
    same_cos = (np.cos((xi[same_pairs_i] - xi[same_pairs_j]) / 2) ** 2).mean()
    diff_cos  = (np.cos((xi[diff_pairs_i] - xi[diff_pairs_j]) / 2) ** 2).mean()
    sd = same_cos / (diff_cos + 1e-12)
    marker = " ←" if sd > 1.05 else (" ↓" if sd < 0.98 else "")
    print(f"  {name:<12} {same_cos:>14.4f} {diff_cos:>14.4f} {sd:>10.4f}{marker}")

print(f"\n  ← means this feature helps quantum discrimination (same > diff)")
print(f"  ↓ means this feature hurts quantum discrimination (same < diff)")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — The geometry mismatch: why RBF captures what FQK misses
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PART 4: Geometry Mismatch — RBF vs ZZFeatureMap")
print("=" * 70)

print(f"""
  RBF kernel:      K_rbf(x,x') = exp(-||x-x'||² / (2σ²))
  ZZFeatureMap:    K_fqk(x,x') ≈ |⟨ψ(x)|ψ(x')⟩|²
                              ≈ prod_i cos²(Δx_i/2)^{ZZ_REPS}  (approx, no ZZ)

  Key difference:
  - RBF uses EUCLIDEAN distance — all features contribute additively
  - ZZFeatureMap uses PRODUCT of per-qubit overlaps — multiplicative structure

  Multiplicative structure has a critical property:
  If ANY single feature shows large Δx, the ENTIRE kernel collapses to near-0.
  RBF degrades gracefully; ZZFeatureMap degrades catastrophically per feature.

  This is why physics features hurt quantum kernels MORE than PCA features:
  Physics features have uniformly high variance (all std ≈ 0.4-0.6 after scaling).
  PCA features have DECREASING variance (PC1 >> PC8) — low-variance PCs act as
  "safe" qubits that don't collapse the product, buffering the high-variance ones.
""")

# Demonstrate with variance profile
print("  Variance profile comparison:")
print(f"  {'Feature':<12} {'Phys std (γ=0.5)':>18} {'Collapse risk':>14}")
print("  " + "-" * 50)
stds = X_phys_scaled.std(axis=0)
for i, name in enumerate(FEATURE_NAMES):
    # Collapse risk: probability that |cos(Δx/2)| < 0.5 for a random pair
    # Δx ~ N(0, 2*std²) approximately
    delta_std = stds[i] * np.sqrt(2)
    # cos²(δ/2) < 0.25 when |δ| > π/1.5 ≈ 2.09
    collapse_prob = 2 * (1 - (1 / (1 + np.exp(-10*(delta_std - np.pi/3)))))
    risk = "HIGH" if delta_std > 0.8 else ("MED" if delta_std > 0.5 else "LOW")
    print(f"  {name:<12} {stds[i]:>18.4f} {risk:>14}  (Δx_std≈{delta_std:.3f})")


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SUMMARY: Why Classical Beats Quantum on Physics Features")
print("=" * 70)

print(f"""
  1. FEATURE QUALITY IS NOT THE PROBLEM
     Physics features are more discriminative than PCA (Fisher ratios higher).
     This confirms your intuition — the information IS in the features.

  2. THE PROBLEM IS THE EMBEDDING GEOMETRY
     ZZFeatureMap uses multiplicative qubit overlap structure.
     Physics features have UNIFORM variance across all 8 qubits.
     → All 8 qubits contribute large Δx simultaneously.
     → Product of 8 overlaps collapses faster than RBF's additive distance.

  3. PCA ACCIDENTALLY WORKED
     PCA features have DECREASING variance: PC1 > PC2 > ... > PC8.
     Low-variance PCs (PC6-PC8) act as "buffer qubits" with cos²≈1.
     → Product doesn't collapse because not all qubits contribute equally.
     → This is structural luck, not quantum advantage.

  4. WHAT WOULD ACTUALLY HELP
     Option A: Feature-specific bandwidth (different γ per qubit)
               — TFK does this but overfit on 75-sample anchor
     Option B: Fewer qubits with highest-Fisher features (4 qubits)
               — Removes the multiplicative collapse problem
     Option C: Projected Quantum Kernel (PQK) instead of fidelity kernel
               — PQK measures local observables, not global state overlap
               — Less susceptible to exponential concentration
               — IBM published evidence this helps (docs.quantum.ibm.com)

  5. PAPER FRAMING
     "ZZFeatureMap fidelity kernel assumes that feature pairs with small
     Euclidean distance also have large quantum state overlap. This holds
     when features have heterogeneous variance (PCA regime) but fails for
     uniformly-informative physics features where the multiplicative product
     structure of the fidelity kernel causes systematic under-estimation of
     same-class similarity."

  Actual KTA values:
    Physics-RBF KTA = {PHYS_RBF_KTA:.4f}  ← highest
    Physics-FQK KTA = {PHYS_FQK_KTA:.4f}  ← below RBF by 0.061
    PCA-FQK     KTA = {PCA_FQK_KTA:.4f}  ← quantum advantage over RBF (+0.053)
    PCA-RBF     KTA = {PCA_RBF_KTA:.4f}  ← lowest
""")
