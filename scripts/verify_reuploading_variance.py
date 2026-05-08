"""
verify_reuploading_variance.py — Check if pure data re-uploading
adds meaningful per-image variance before running full circuits.

The approach: instead of softmax attention (which collapsed to 0.125),
use pure Pérez-Salinas data re-uploading. Layer 2 injects raw features
via RY rotations (non-commuting with Layer 1's RZ). No normalization.
No attention matrix. Direct feature re-injection.

This script analytically estimates the Bloch vector components after
the re-uploading layer and checks:
  1. Does per-image variance increase over standard AGPQK?
  2. Are the re-uploading Bloch components class-structured (Fisher > 0)?
  3. Is the rotation angle range meaningful (>0.5 rad spread)?
  4. Do different images get genuinely different quantum states?

Analytical approximation used here (no quantum circuit needed):
  After RZ(θ_z) then RY(θ_y) on |0⟩:
    ⟨Z⟩ ≈ cos(θ_z) · cos(θ_y)
    ⟨X⟩ ≈ sin(θ_z) · sin(θ_y)
    ⟨Y⟩ ≈ sin(θ_z) · cos(θ_y)  [simplified, ignores ZZ]
This is an approximation that captures the variance structure
without running quantum circuits.

Usage:
    python scripts/verify_reuploading_variance.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np

import config

# ── Load confirmed AGPQK config ────────────────────────────────────────────
CONFIG_PATH = os.path.join(config.RESULTS_DIR, "physics", "agpqk_config.json")

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        agpqk_cfg = json.load(f)
    SELECTED_INDICES   = np.array(agpqk_cfg["selected_feature_indices"])
    GAMMA_PER_QUBIT    = np.array(agpqk_cfg["gamma_per_qubit"])
    ENTANGLEMENT_PAIRS = agpqk_cfg["entanglement_pairs_qubit_space"]
    FEATURE_NAMES_SEL  = agpqk_cfg["selected_feature_names"]
    MODALITY_SEL       = agpqk_cfg["selected_feature_modalities"]
    print(f"  Loaded AGPQK config: {CONFIG_PATH}")
else:
    print(f"  WARNING: config not found, using hardcoded values")
    SELECTED_INDICES   = np.array([0, 2, 5, 6, 9, 11, 12, 13])
    GAMMA_PER_QUBIT    = np.array([0.57, 0.80, 0.54, 0.80, 0.20, 0.20, 0.46, 0.36])
    ENTANGLEMENT_PAIRS = [(2, 6), (1, 5), (0, 7), (3, 4)]
    FEATURE_NAMES_SEL  = ["NDVI","NDWI","NDRE","MNDWI",
                           "SAR_total","PolCoh","VH_dB","VV_dB"]
    MODALITY_SEL       = ["OPT","OPT","OPT","OPT","SAR","SAR","SAR","SAR"]

N_QUBITS = len(SELECTED_INDICES)

FEATURE_NAMES_16 = [
    "NDVI","NDBI","NDWI","BSI","SAVI","NDRE","MNDWI","EVI",
    "VV/VH","SAR_total","CrossPol","PolCoh","VH_dB","VV_dB","VV_tex","NDVI_tex"
]

# ── Load data ──────────────────────────────────────────────────────────────
PHYS16_PATH = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
data    = np.load(PHYS16_PATH)
X16     = data["X_train"]    # (2000, 16), in [0, π]
y_train = data["y_train"]

X8_raw = X16[:, SELECTED_INDICES]                        # (2000, 8) in [0, π]
X8_enc = X8_raw * GAMMA_PER_QUBIT[np.newaxis, :]        # Layer 1 angles
X8_l2  = X8_raw                                          # Layer 2 angles (raw, no scaling)

n = len(X8_enc)

print(f"\n  Dataset: {n} samples, {N_QUBITS} selected features")
print(f"  Layer 1 (RZ): γ-scaled features, range "
      f"[{X8_enc.min():.3f}, {X8_enc.max():.3f}]")
print(f"  Layer 2 (RY): raw features, range "
      f"[{X8_l2.min():.3f}, {X8_l2.max():.3f}]")


# ── Analysis 1: Raw feature range and variance per qubit ──────────────────
print("\n" + "="*65)
print("  ANALYSIS 1: Layer 2 Rotation Angle Range (RY angles = raw features)")
print("="*65)

print(f"\n  {'Qubit':<5} {'Feature':<12} {'Mod':>4}  "
      f"{'Min':>7} {'Max':>7} {'Mean':>7} {'Std':>7} {'Range':>7}")
print("  " + "-" * 60)

for q in range(N_QUBITS):
    angles = X8_l2[:, q]
    print(f"  q{q:<4} {FEATURE_NAMES_SEL[q]:<12} {MODALITY_SEL[q]:>4}  "
          f"{angles.min():>7.3f} {angles.max():>7.3f} "
          f"{angles.mean():>7.3f} {angles.std():>7.3f} "
          f"{(angles.max()-angles.min()):>7.3f}")

print(f"\n  Compare to softmax attention scores: range was only [0.108, 0.141]")
print(f"  Raw feature range: [{X8_l2.min():.3f}, {X8_l2.max():.3f}]  "
      f"(~{(X8_l2.max()-X8_l2.min())/0.033:.0f}× more diverse)")


# ── Analysis 2: Analytical Bloch vector approximation ────────────────────
print("\n" + "="*65)
print("  ANALYSIS 2: Analytical Bloch Vector Approximation After Re-uploading")
print("="*65)
print("  (Approximate: RZ(θ_z) then RY(θ_y), ignoring ZZ entanglement)")
print("  ⟨Z⟩ ≈ cos(θ_z)·cos(θ_y),  ⟨X⟩ ≈ sin(θ_z)·sin(θ_y)")

# Standard AGPQK: only Layer 1
bloch_l1_z = np.cos(X8_enc)   # (2000, 8) — ⟨Z⟩ after Layer 1 only
bloch_l1_x = np.sin(X8_enc)   # (2000, 8) — ⟨X⟩ after Layer 1 only

# After Layer 2 re-uploading: RZ(γx) then RY(x)
bloch_reu_z = np.cos(X8_enc) * np.cos(X8_l2)   # (2000, 8)
bloch_reu_x = np.sin(X8_enc) * np.sin(X8_l2)   # (2000, 8)
bloch_reu_y = np.sin(X8_enc) * np.cos(X8_l2)   # (2000, 8)

print(f"\n  Per-qubit Bloch ⟨Z⟩ standard deviation comparison:")
print(f"  {'Qubit':<5} {'Feature':<12}  "
      f"{'L1-only std':>12} {'Re-upload std':>14} {'Improvement':>12}")
print("  " + "-" * 58)

for q in range(N_QUBITS):
    std_l1  = bloch_l1_z[:, q].std()
    std_reu = bloch_reu_z[:, q].std()
    improvement = (std_reu - std_l1) / (std_l1 + 1e-12) * 100
    marker = " ↑" if improvement > 5 else (" ↓" if improvement < -5 else "  ≈")
    print(f"  q{q:<4} {FEATURE_NAMES_SEL[q]:<12}  "
          f"{std_l1:>12.4f} {std_reu:>14.4f} "
          f"{improvement:>+11.1f}%{marker}")

# Combined Bloch vector (flatten across qubits)
bloch_l1_flat  = np.concatenate([bloch_l1_z, bloch_l1_x], axis=1)      # (2000, 16)
bloch_reu_flat = np.concatenate([bloch_reu_z, bloch_reu_x,
                                  bloch_reu_y], axis=1)                   # (2000, 24)

print(f"\n  Overall Bloch vector variance:")
print(f"    L1-only (16-dim):      mean_std = {bloch_l1_flat.std(axis=0).mean():.4f}")
print(f"    Re-uploading (24-dim): mean_std = {bloch_reu_flat.std(axis=0).mean():.4f}")


# ── Analysis 3: Fisher ratio of re-uploaded Bloch components ─────────────
print("\n" + "="*65)
print("  ANALYSIS 3: Fisher Ratio — Class Structure in Re-uploaded Bloch Vectors")
print("="*65)

def fisher_ratio_per_dim(B, y):
    """Fisher ratio per dimension of Bloch vector matrix B."""
    classes   = np.unique(y)
    grand_mean = B.mean(axis=0)
    between   = np.zeros(B.shape[1])
    within    = np.zeros(B.shape[1])
    for c in classes:
        mask = y == c
        nc   = mask.sum()
        mu_c = B[mask].mean(axis=0)
        between += nc * (mu_c - grand_mean) ** 2
        within  += ((B[mask] - mu_c) ** 2).sum(axis=0)
    return between / np.maximum(within, 1e-12)

fisher_l1  = fisher_ratio_per_dim(bloch_l1_flat,  y_train)
fisher_reu = fisher_ratio_per_dim(bloch_reu_flat, y_train)

print(f"\n  L1-only  — mean Fisher: {fisher_l1.mean():.4f}  "
      f"max: {fisher_l1.max():.4f}  "
      f"dims>0.1: {(fisher_l1 > 0.1).sum()}/{len(fisher_l1)}")
print(f"  Re-upload — mean Fisher: {fisher_reu.mean():.4f}  "
      f"max: {fisher_reu.max():.4f}  "
      f"dims>0.1: {(fisher_reu > 0.1).sum()}/{len(fisher_reu)}")

print(f"\n  Per-qubit Fisher ratio (⟨Z⟩ component):")
print(f"  {'Qubit':<5} {'Feature':<12}  "
      f"{'Fisher L1':>10} {'Fisher Reu-Z':>13} {'Fisher Reu-X':>13} {'Best':>8}")
print("  " + "-" * 60)

for q in range(N_QUBITS):
    f_l1  = fisher_l1[q]
    f_z   = fisher_ratio_per_dim(bloch_reu_z[:, q:q+1], y_train)[0]
    f_x   = fisher_ratio_per_dim(bloch_reu_x[:, q:q+1], y_train)[0]
    best  = max(f_l1, f_z, f_x)
    winner = "L1" if f_l1 == best else ("Reu-Z" if f_z == best else "Reu-X")
    print(f"  q{q:<4} {FEATURE_NAMES_SEL[q]:<12}  "
          f"{f_l1:>10.4f} {f_z:>13.4f} {f_x:>13.4f} {winner:>8}")


# ── Analysis 4: Cross-modal interaction terms ─────────────────────────────
print("\n" + "="*65)
print("  ANALYSIS 4: Cross-Modal Interaction Terms")
print("  (Re-uploading creates x_i · x_j product terms unavailable in L1)")
print("="*65)

print(f"\n  For each entanglement pair, the re-uploading adds terms:")
print(f"  sin(γ_i·x_i)·sin(x_i) and cos(γ_i·x_i)·cos(x_i) per qubit,")
print(f"  plus cross-qubit terms from ZZ entanglement.\n")

print(f"  Cross-pair product variance (sin(γ_i·x_i)·x_j — cross-modal signal):")
print(f"  {'Pair':<40} {'Std of product':>16} {'Fisher':>8}")
print("  " + "-" * 68)

for qi, qj in ENTANGLEMENT_PAIRS:
    # Cross-modal product term: sin(γ_i·x_i) * x_j (approximation of ZZ effect)
    product = np.sin(X8_enc[:, qi]) * X8_l2[:, qj]
    f_prod  = fisher_ratio_per_dim(product.reshape(-1, 1), y_train)[0]
    label   = (f"q{qi}({FEATURE_NAMES_SEL[qi]}) ↔ "
               f"q{qj}({FEATURE_NAMES_SEL[qj]})")
    print(f"  {label:<40} {product.std():>16.4f} {f_prod:>8.4f}")

print(f"\n  Compare to softmax attention: all pairs had Fisher < 0.10")


# ── Analysis 5: Sample diversity ──────────────────────────────────────────
print("\n" + "="*65)
print("  ANALYSIS 5: Per-Sample Diversity")
print("  (How different are random image pairs in re-uploaded Bloch space?)")
print("="*65)

rng = np.random.default_rng(42)
n_pairs_test = 1000
idx_i = rng.integers(0, n, n_pairs_test)
idx_j = rng.integers(0, n, n_pairs_test)
same_class  = y_train[idx_i] == y_train[idx_j]
diff_class  = ~same_class

# L1 only distance
dist_l1_same = np.linalg.norm(
    bloch_l1_flat[idx_i[same_class]] - bloch_l1_flat[idx_j[same_class]], axis=1)
dist_l1_diff = np.linalg.norm(
    bloch_l1_flat[idx_i[diff_class]] - bloch_l1_flat[idx_j[diff_class]], axis=1)

# Re-uploading distance
dist_reu_same = np.linalg.norm(
    bloch_reu_flat[idx_i[same_class]] - bloch_reu_flat[idx_j[same_class]], axis=1)
dist_reu_diff = np.linalg.norm(
    bloch_reu_flat[idx_i[diff_class]] - bloch_reu_flat[idx_j[diff_class]], axis=1)

print(f"\n  Pairwise Bloch-space distances ({n_pairs_test} random pairs):")
print(f"  {'Method':<16} {'Same-class dist':>16} {'Diff-class dist':>16} "
      f"{'Ratio D/S':>10}")
print("  " + "-" * 62)

r_l1  = dist_l1_diff.mean()  / (dist_l1_same.mean()  + 1e-12)
r_reu = dist_reu_diff.mean() / (dist_reu_same.mean() + 1e-12)

print(f"  {'L1-only':<16} {dist_l1_same.mean():>16.4f} "
      f"{dist_l1_diff.mean():>16.4f} {r_l1:>10.4f}")
print(f"  {'Re-uploading':<16} {dist_reu_same.mean():>16.4f} "
      f"{dist_reu_diff.mean():>16.4f} {r_reu:>10.4f}")

print(f"\n  Higher D/S ratio = better class separation in Bloch space")
print(f"  (RBF achieves high D/S by weighting features by discriminability)")


# ── Verdict ────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  VERDICT: Is pure data re-uploading worth running?")
print("="*65)

mean_fisher_reu = fisher_reu.mean()
mean_fisher_l1  = fisher_l1.mean()
fisher_improvement = (mean_fisher_reu - mean_fisher_l1) / (mean_fisher_l1 + 1e-12) * 100
diversity_improvement = (r_reu - r_l1) / (r_l1 + 1e-12) * 100

print(f"\n  Fisher improvement:    {fisher_improvement:+.1f}%  "
      f"({mean_fisher_l1:.4f} → {mean_fisher_reu:.4f})")
print(f"  D/S ratio improvement: {diversity_improvement:+.1f}%  "
      f"({r_l1:.4f} → {r_reu:.4f})")
print(f"  Angle range (Layer 2): [{X8_l2.min():.3f}, {X8_l2.max():.3f}]  "
      f"vs softmax [{0.108:.3f}, {0.141:.3f}]")

if fisher_improvement > 10 and diversity_improvement > 5:
    print(f"\n  GO: Re-uploading adds meaningful class-structured information.")
    print(f"  Expected KTA improvement over AGPQK (0.2518): likely +0.01 to +0.04")
    print(f"  Implement Layer 2 as RY(x_i) after Layer 1's ZZ entanglement.")
elif fisher_improvement > 0 and diversity_improvement > 0:
    print(f"\n  MARGINAL: Some improvement predicted but modest.")
    print(f"  Worth running — the circuit cost is only ~1.5× AGPQK.")
    print(f"  If KTA improves by >0.005, re-uploading is validated.")
else:
    print(f"\n  STOP: Re-uploading not predicted to improve over AGPQK.")
    print(f"  The Fisher-weighted outer kernel fix is a better next step.")

print(f"\n  NOTE: This is an analytical approximation (ignores ZZ entanglement).")
print(f"  The actual circuit will show different values, likely higher Fisher")
print(f"  because ZZ entanglement creates cross-qubit terms not captured here.")
print()
