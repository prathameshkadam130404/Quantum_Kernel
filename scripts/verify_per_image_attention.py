"""
verify_per_image_attention.py — Check if per-image attention scores
vary meaningfully across samples before implementing re-uploading.

For each training image x, we compute a per-image attention matrix A(x)
from x's own 16 feature values. The question is: do the attention scores
for the 4 selected cross-modal entanglement pairs vary enough across images
to be informative as rotation angles in a re-uploading layer?

If variance is near zero → re-uploading adds constant angles → no benefit.
If variance is high and class-structured → re-uploading encodes per-image
  feature relationships → genuine additional discriminative signal.

Confirmed circuit decisions (from AGPQK run):
  Selected features:  [ 0  2  5  6  9 11 12 13]
    q0: NDVI  [OPT]  q1: NDWI  [OPT]  q2: NDRE  [OPT]  q3: MNDWI [OPT]
    q4: SAR_total [SAR]  q5: PolCoh [SAR]  q6: VH_dB [SAR]  q7: VV_dB [SAR]
  Entanglement pairs (qubit space):
    q2(NDRE) ↔ q6(VH_dB)
    q1(NDWI) ↔ q5(PolCoh)
    q0(NDVI) ↔ q7(VV_dB)
    q3(MNDWI) ↔ q4(SAR_total)

NOTE: Update SELECTED_INDICES and ENTANGLEMENT_PAIRS below to match
your actual agpqk_config.json if they differ from the above.

Usage:
    python scripts/verify_per_image_attention.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np

import config

# ── Load confirmed AGPQK config from file if available ────────────────────
CONFIG_PATH = os.path.join(config.RESULTS_DIR, "physics", "agpqk_config.json")

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH) as f:
        agpqk_cfg = json.load(f)
    SELECTED_INDICES   = np.array(agpqk_cfg["selected_feature_indices"])
    ENTANGLEMENT_PAIRS = agpqk_cfg["entanglement_pairs_qubit_space"]
    FEATURE_NAMES_SEL  = agpqk_cfg["selected_feature_names"]
    MODALITY_SEL       = agpqk_cfg["selected_feature_modalities"]
    print(f"  Loaded AGPQK config from: {CONFIG_PATH}")
else:
    # Fallback: hardcode from verification output above
    print(f"  WARNING: {CONFIG_PATH} not found — using hardcoded values")
    SELECTED_INDICES   = np.array([0, 2, 5, 6, 9, 11, 12, 13])
    ENTANGLEMENT_PAIRS = [(2, 6), (1, 5), (0, 7), (3, 4)]
    FEATURE_NAMES_SEL  = ["NDVI","NDWI","NDRE","MNDWI",
                           "SAR_total","PolCoh","VH_dB","VV_dB"]
    MODALITY_SEL       = ["OPT","OPT","OPT","OPT","SAR","SAR","SAR","SAR"]

FEATURE_NAMES_16 = [
    "NDVI","NDBI","NDWI","BSI","SAVI","NDRE","MNDWI","EVI",
    "VV/VH","SAR_total","CrossPol","PolCoh","VH_dB","VV_dB","VV_tex","NDVI_tex"
]

# ── Load data ─────────────────────────────────────────────────────────────
PHYS16_PATH = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
data    = np.load(PHYS16_PATH)
X16     = data["X_train"]    # (2000, 16), in [0, π]
y_train = data["y_train"]

n, d = X16.shape
print(f"\n  Dataset: {n} samples, {d} features")
print(f"  Selected features: {SELECTED_INDICES.tolist()}")
print(f"  Entanglement pairs: {ENTANGLEMENT_PAIRS}")


# ── Per-image attention function ──────────────────────────────────────────
def compute_per_image_attention(x_full: np.ndarray,
                                 selected_idx: np.ndarray,
                                 temperature: float = None) -> np.ndarray:
    """
    Compute per-image attention matrix from a single sample's 8 selected features.

    For a single image x with 8 selected features, the 'correlation' between
    feature i and feature j is simply x[i] * x[j] (outer product), normalized
    by the feature norms. This captures: if two features are both high for this
    image, they are mutually attending.

    This is fundamentally different from the global A computed over 2000 samples:
    - Global A: captures co-variance structure of the DATASET
    - Per-image A(x): captures which features are simultaneously active IN THIS IMAGE

    Args:
        x_full:       Full 16-feature vector for one image, shape (16,)
        selected_idx: Indices of 8 selected features
        temperature:  Softmax temperature (default: sqrt(8) = 2.83)

    Returns:
        A_img: Per-image attention matrix, shape (8, 8), row-stochastic
    """
    x8 = x_full[selected_idx]   # (8,) — selected features only
    n_sel = len(x8)

    if temperature is None:
        temperature = float(np.sqrt(n_sel))  # √8 ≈ 2.83

    # Outer product: scores[i,j] = x8[i] * x8[j]
    # High when both features are simultaneously large for this image
    scores = np.outer(x8, x8)  # (8, 8)

    # Normalize by feature magnitudes to get cosine-like similarity
    norms = np.abs(x8) + 1e-8   # (8,)
    norm_mat = np.outer(norms, norms)  # (8, 8)
    scores = scores / norm_mat         # (8, 8), in [-1, 1]

    # Scale by temperature and apply softmax (row-wise)
    scores = scores / temperature
    scores = scores - scores.max(axis=1, keepdims=True)  # numerical stability
    exp_s  = np.exp(scores)
    A_img  = exp_s / exp_s.sum(axis=1, keepdims=True)    # row-stochastic

    return A_img


def get_pair_attention_scores(A_img: np.ndarray,
                               pairs: list) -> np.ndarray:
    """
    Extract attention scores for the 4 entanglement pairs from a per-image A.

    Args:
        A_img: Per-image attention matrix, shape (8, 8)
        pairs: List of (qi, qj) tuples in qubit space 0-7

    Returns:
        scores: shape (n_pairs,), attention score per pair
    """
    scores = np.array([(A_img[qi, qj] + A_img[qj, qi]) / 2.0
                       for qi, qj in pairs])
    return scores


# ── Compute per-image attention scores for all 2000 samples ───────────────
print("\n--- Computing per-image attention scores for all 2000 samples ---")

n_pairs = len(ENTANGLEMENT_PAIRS)
pair_scores_all = np.zeros((n, n_pairs), dtype=np.float64)

for i in range(n):
    A_img = compute_per_image_attention(X16[i], SELECTED_INDICES)
    pair_scores_all[i] = get_pair_attention_scores(A_img, ENTANGLEMENT_PAIRS)

print(f"  Done. Shape: {pair_scores_all.shape}  (samples × pairs)")


# ── Analysis 1: Overall variance per pair ────────────────────────────────
print("\n--- Analysis 1: Variance of Per-Image Attention Scores ---")
print(f"\n  {'Pair':<35} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'CV':>8}")
print("  " + "-" * 75)

pair_labels = []
for qi, qj in ENTANGLEMENT_PAIRS:
    label = f"q{qi}({FEATURE_NAMES_SEL[qi]}) ↔ q{qj}({FEATURE_NAMES_SEL[qj]})"
    pair_labels.append(label)

for p, label in enumerate(pair_labels):
    scores = pair_scores_all[:, p]
    cv = scores.std() / (scores.mean() + 1e-12)
    print(f"  {label:<35} {scores.mean():>8.4f} {scores.std():>8.4f} "
          f"{scores.min():>8.4f} {scores.max():>8.4f} {cv:>8.4f}")

print(f"\n  Interpretation:")
print(f"  CV > 0.3: meaningful variance → re-uploading will add information")
print(f"  CV < 0.1: near-constant → re-uploading adds noise, not signal")


# ── Analysis 2: Class-conditional mean per pair ───────────────────────────
print("\n--- Analysis 2: Class-Conditional Attention Scores (per pair) ---")
print("  (Do different LCZ classes get different per-image attention?)")

classes = np.unique(y_train)
class_means = np.zeros((len(classes), n_pairs))

for ci, c in enumerate(classes):
    mask = y_train == c
    class_means[ci] = pair_scores_all[mask].mean(axis=0)

# Fisher ratio of attention scores across classes
print(f"\n  Fisher ratio of attention score (between-class / within-class var):")
print(f"  {'Pair':<35} {'Fisher':>8}  Interpretation")
print("  " + "-" * 60)

grand_mean = pair_scores_all.mean(axis=0)
for p, label in enumerate(pair_labels):
    between = 0.0
    within  = 0.0
    for ci, c in enumerate(classes):
        mask = y_train == c
        nc   = mask.sum()
        mu_c = pair_scores_all[mask, p].mean()
        between += nc * (mu_c - grand_mean[p]) ** 2
        within  += ((pair_scores_all[mask, p] - mu_c) ** 2).sum()
    fisher = between / max(within, 1e-12)
    quality = ("STRONG" if fisher > 0.5 else
               "MODERATE" if fisher > 0.1 else "WEAK")
    print(f"  {label:<35} {fisher:>8.4f}  {quality}")


# ── Analysis 3: Per-class attention score heatmap ────────────────────────
print("\n--- Analysis 3: Per-Class Mean Attention Scores ---")
print("  (Higher = this pair is more simultaneously active for this class)")
print()

header = f"  {'Class':<22}"
for p in range(n_pairs):
    qi, qj = ENTANGLEMENT_PAIRS[p]
    header += f"  {FEATURE_NAMES_SEL[qi][:4]}↔{FEATURE_NAMES_SEL[qj][:4]:>4}"
print(header)
print("  " + "-" * (22 + 12 * n_pairs))

lcz_names = config.LCZ_CLASS_NAMES

for ci, c in enumerate(classes):
    row = f"  {lcz_names[c]:<22}"
    for p in range(n_pairs):
        val = class_means[ci, p]
        # Mark high values with *
        mark = "*" if val > class_means[:, p].mean() + class_means[:, p].std() else " "
        row += f"  {val:.3f}{mark}    "
    print(row)

print(f"\n  * = above mean + 1std for that pair")


# ── Analysis 4: Correlation with class labels ────────────────────────────
print("\n--- Analysis 4: Attention Score Distribution by Modality Split ---")
print("  (SAR-heavy vs optical-heavy images — key question for cross-modal)")

# Identify "optical-heavy" vs "SAR-heavy" samples by comparing feature magnitudes
x8_all = X16[:, SELECTED_INDICES]   # (2000, 8)
opt_cols = [i for i, m in enumerate(MODALITY_SEL) if m == "OPT"]
sar_cols = [i for i, m in enumerate(MODALITY_SEL) if m == "SAR"]

opt_strength = x8_all[:, opt_cols].mean(axis=1)   # (2000,)
sar_strength = x8_all[:, sar_cols].mean(axis=1)   # (2000,)

opt_dominant = opt_strength > sar_strength
sar_dominant = ~opt_dominant

print(f"\n  Optical-dominant samples: {opt_dominant.sum()} "
      f"({100*opt_dominant.mean():.1f}%)")
print(f"  SAR-dominant samples:     {sar_dominant.sum()} "
      f"({100*sar_dominant.mean():.1f}%)")

print(f"\n  {'Pair':<35} {'OPT-dom mean':>14} {'SAR-dom mean':>14} {'Δ':>8}")
print("  " + "-" * 75)

for p, label in enumerate(pair_labels):
    opt_mean = pair_scores_all[opt_dominant, p].mean()
    sar_mean = pair_scores_all[sar_dominant, p].mean()
    delta    = opt_mean - sar_mean
    print(f"  {label:<35} {opt_mean:>14.4f} {sar_mean:>14.4f} {delta:>+8.4f}")

print(f"\n  Large |Δ| means optical vs SAR dominant images get different")
print(f"  attention scores → re-uploading encodes modality-specific structure")


# ── Summary verdict ───────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  VERDICT: Is per-image attention useful for re-uploading?")
print(f"{'='*65}")

cv_values    = [pair_scores_all[:, p].std() / (pair_scores_all[:, p].mean() + 1e-12)
                for p in range(n_pairs)]
fisher_vals  = []
for p in range(n_pairs):
    between, within = 0.0, 0.0
    gm = pair_scores_all[:, p].mean()
    for c in classes:
        mask = y_train == c
        nc   = mask.sum()
        mu_c = pair_scores_all[mask, p].mean()
        between += nc * (mu_c - gm) ** 2
        within  += ((pair_scores_all[mask, p] - mu_c) ** 2).sum()
    fisher_vals.append(between / max(within, 1e-12))

mean_cv     = np.mean(cv_values)
mean_fisher = np.mean(fisher_vals)
max_fisher  = np.max(fisher_vals)

print(f"\n  Mean CV across pairs:     {mean_cv:.4f}")
print(f"  Mean Fisher across pairs: {mean_fisher:.4f}")
print(f"  Max Fisher across pairs:  {max_fisher:.4f}")

print()
if mean_cv > 0.3 and mean_fisher > 0.1:
    print(f"  GO: Per-image attention varies meaningfully AND is class-structured.")
    print(f"  Re-uploading with per-image attention scores is worth implementing.")
    print(f"  Expected benefit: the re-uploading layer encodes per-image SAR-optical")
    print(f"  feature relationships, creating instance-adaptive quantum states.")
elif mean_cv > 0.3 and mean_fisher <= 0.1:
    print(f"  PARTIAL: Attention varies across images but not along class boundaries.")
    print(f"  Re-uploading may add noise rather than signal.")
    print(f"  Consider using only the highest-Fisher pair for re-uploading.")
elif mean_cv <= 0.3:
    print(f"  STOP: Per-image attention scores have low variance (CV={mean_cv:.3f}).")
    print(f"  All images produce similar attention scores → re-uploading adds")
    print(f"  near-constant rotation angles → no benefit over current AGPQK.")
    print(f"  The outer-kernel Fisher weighting fix is a better next step.")
else:
    print(f"  MARGINAL: Mixed signals — review per-pair analysis above.")

print(f"\n  Rotation angle range if used in re-uploading:")
for p, label in enumerate(pair_labels):
    scores = pair_scores_all[:, p]
    angle_min = scores.min() * np.pi
    angle_max = scores.max() * np.pi
    print(f"    {label:<35}  [{angle_min:.3f}, {angle_max:.3f}] rad")
print()
