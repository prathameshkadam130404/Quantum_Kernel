"""
attention_kernel.py — Attention-guided feature selection and bandwidth for AGPQK.

Computes a classical attention matrix from the full 16-feature physics set,
then derives:
  - Top-8 feature selection by attention importance
  - Per-qubit bandwidth γ_i modulated by feature importance
  - Top-k entanglement pairs from cross-feature attention scores

All computations use TRAINING DATA ONLY — no label information, no test leakage.

The attention matrix A[i,j] measures the co-variance alignment between features
i and j across training samples. High A[i,j] means features i and j carry
correlated structural information about the dataset — they should be entangled.

Reference:
  Inspired by QKSAN (Quantum Kernel Self-Attention Network, 2023) but adapted
  for tabular SAR-optical fusion data rather than sequential NLP inputs.
  Key difference: attention is computed from feature co-variance, not from
  learned query/key/value matrices, making it training-data-grounded and
  free of gradient-based optimization on small anchors.
"""

import numpy as np
from typing import Tuple, List


# Full 16-feature index map (from physics_features_16.npz)
# 0:NDVI  1:NDBI  2:NDWI  3:BSI  4:SAVI  5:NDRE  6:MNDWI  7:EVI
# 8:VV/VH 9:SAR_total 10:CrossPol 11:PolCoh 12:VH_dB 13:VV_dB
# 14:VV_tex 15:NDVI_tex
FEATURE_NAMES_16 = [
    "NDVI", "NDBI", "NDWI", "BSI", "SAVI", "NDRE", "MNDWI", "EVI",
    "VV/VH", "SAR_total", "CrossPol", "PolCoh",
    "VH_dB", "VV_dB", "VV_tex", "NDVI_tex"
]

# Modality tags for each of the 16 features (for reporting cross-modal pairs)
FEATURE_MODALITY_16 = [
    "OPT", "OPT", "OPT", "OPT", "OPT", "OPT", "OPT", "OPT",
    "SAR", "SAR", "SAR", "SAR", "SAR", "SAR", "SAR", "SAR"
]


def compute_attention_matrix(
    X: np.ndarray,
    temperature: float = None,
) -> np.ndarray:
    """
    Compute the 16×16 classical attention matrix from feature co-variance.

    Steps:
      1. Standardize X to zero mean, unit variance per feature
      2. Compute correlation matrix C = X_std.T @ X_std / n  (16×16)
      3. Apply softmax with temperature scaling: A = softmax(C / sqrt(d))

    The result A[i,j] is the normalized attention weight between feature i
    and feature j. High values indicate features that co-vary strongly across
    training samples — candidates for quantum entanglement.

    Args:
        X:           Feature matrix, shape (n, 16), values in [0, π]
        temperature: Softmax temperature. Default: sqrt(n_features) = sqrt(16) = 4.0

    Returns:
        A:  Attention matrix, shape (16, 16), row-stochastic (rows sum to 1)
    """
    n, d = X.shape
    assert d == 16, f"Expected 16 features, got {d}"

    if temperature is None:
        temperature = float(np.sqrt(d))  # √16 = 4.0

    # Step 1: Standardize (zero mean, unit std per feature)
    mu  = X.mean(axis=0)             # (16,)
    std = X.std(axis=0) + 1e-8       # (16,)
    X_std = (X - mu) / std           # (n, 16)

    # Step 2: Correlation matrix — feature × feature co-variance
    C = (X_std.T @ X_std) / n        # (16, 16)

    # Step 3: Scaled dot-product attention with softmax (row-wise)
    scores = C / temperature          # (16, 16)
    # Numerically stable softmax
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_s  = np.exp(scores)
    A      = exp_s / exp_s.sum(axis=1, keepdims=True)  # (16, 16), row-stochastic

    return A


def compute_fisher_ratio(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Per-feature Fisher discriminant ratio: between-class / within-class variance.
    Computed on TRAINING DATA ONLY. Standard supervised feature selection.
    Higher = more linearly class-separable.
    """
    classes = np.unique(y)
    grand_mean = X.mean(axis=0)
    between = np.zeros(X.shape[1])
    within  = np.zeros(X.shape[1])
    for c in classes:
        mask = y == c
        nc   = mask.sum()
        mu_c = X[mask].mean(axis=0)
        between += nc * (mu_c - grand_mean) ** 2
        within  += ((X[mask] - mu_c) ** 2).sum(axis=0)
    return between / np.maximum(within, 1e-12)


def select_features_by_fisher(
    X: np.ndarray,
    y: np.ndarray,
    n_select: int = 8,
    min_sar: int = 2,
    min_opt: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top-n features by Fisher ratio with modality balance constraint.

    min_sar and min_opt enforce at least 2 SAR and 2 optical features,
    ensuring cross-modal entanglement is meaningful.

    Returns:
        selected_indices: shape (n_select,), sorted
        fisher:           shape (16,), Fisher ratio per feature
    """
    fisher = compute_fisher_ratio(X, y)

    # Separate optical (0-7) and SAR (8-15) indices
    opt_idx = np.arange(0, 8)
    sar_idx = np.arange(8, 16)

    # Sort each modality by Fisher ratio descending
    opt_ranked = opt_idx[np.argsort(fisher[opt_idx])[::-1]]
    sar_ranked = sar_idx[np.argsort(fisher[sar_idx])[::-1]]

    # Guarantee minimums
    selected = list(opt_ranked[:min_opt]) + list(sar_ranked[:min_sar])
    remaining_slots = n_select - len(selected)

    # Fill remaining slots from global Fisher ranking, excluding already selected
    all_ranked = np.argsort(fisher)[::-1]
    for idx in all_ranked:
        if len(selected) >= n_select:
            break
        if idx not in selected:
            selected.append(int(idx))

    selected = np.sort(np.array(selected))
    return selected, fisher

def compute_per_qubit_gamma(
    importance: np.ndarray,
    selected_indices: np.ndarray,
    gamma_base: float = 0.50,
    gamma_min:  float = 0.20,
    gamma_max:  float = 0.80,
) -> np.ndarray:
    """
    Compute per-qubit bandwidth from attention importance scores.

    γ_i = γ_base × normalize(importance[selected_i])
    where normalize maps selected importances to mean=1.0.

    Features with higher attention importance get larger bandwidth
    (more encoding range), analogous to RBF's natural feature weighting
    by Euclidean distance variance.

    Args:
        importance:       Full 16-feature importance vector
        selected_indices: Indices of selected 8 features
        gamma_base:       Base bandwidth (CV-selected = 0.50)
        gamma_min:        Minimum per-qubit gamma (safety floor)
        gamma_max:        Maximum per-qubit gamma (concentration ceiling)

    Returns:
        gamma_per_qubit:  np.ndarray, shape (n_select,), per-qubit γ values
    """
    sel_importance = importance[selected_indices]   # (8,)
    # Normalize so mean = 1.0 → multiply by gamma_base
    norm = sel_importance / (sel_importance.mean() + 1e-12)
    gamma_i = gamma_base * norm
    # Clip to safe range
    gamma_i = np.clip(gamma_i, gamma_min, gamma_max)
    return gamma_i


def get_attention_entanglement_pairs(
    A: np.ndarray,
    selected_indices: np.ndarray,
    top_k: int = 4,
    require_cross_modal: bool = True,
    max_appearances: int = 1,       # ADD THIS PARAMETER
) -> List[Tuple[int, int]]:
    """
    ...same docstring...
    max_appearances: max times any single qubit can appear across all pairs.
                     Set to 1 for fully diverse pairs (no shared nodes).
                     Set to 2 to allow one shared node.
    """
    n_sel = len(selected_indices)
    A_sub = A[np.ix_(selected_indices, selected_indices)]
    A_sym = (A_sub + A_sub.T) / 2.0

    candidates = []
    for i in range(n_sel):
        for j in range(i + 1, n_sel):
            if require_cross_modal:
                mod_i = FEATURE_MODALITY_16[selected_indices[i]]
                mod_j = FEATURE_MODALITY_16[selected_indices[j]]
                if not ((mod_i == "SAR") != (mod_j == "SAR")):
                    continue
            candidates.append((i, j, A_sym[i, j]))

    if not candidates:
        for i in range(n_sel):
            for j in range(i + 1, n_sel):
                candidates.append((i, j, A_sym[i, j]))

    candidates.sort(key=lambda x: x[2], reverse=True)

    # ADD: diversity constraint — track appearances per qubit
    pairs = []
    appearances = {}
    for (i, j, score) in candidates:
        if len(pairs) >= top_k:
            break
        count_i = appearances.get(i, 0)
        count_j = appearances.get(j, 0)
        if count_i < max_appearances and count_j < max_appearances:
            pairs.append((i, j))
            appearances[i] = count_i + 1
            appearances[j] = count_j + 1

    return pairs


def print_attention_summary(
    A: np.ndarray,
    selected_indices: np.ndarray,
    importance: np.ndarray,
    gamma_per_qubit: np.ndarray,
    entanglement_pairs: List[Tuple[int, int]],
) -> None:
    """Print a human-readable summary of all attention decisions."""

    print("\n" + "=" * 65)
    print("  ATTENTION MATRIX SUMMARY")
    print("=" * 65)

    print(f"\n  16-feature importance scores (row sums of A):")
    print(f"  {'#':>3}  {'Feature':<12} {'Modality':>8}  {'Importance':>11}  {'Selected':>9}")
    print("  " + "-" * 50)
    for i in range(16):
        sel = "  ← SELECTED" if i in selected_indices else ""
        print(f"  {i:>3}  {FEATURE_NAMES_16[i]:<12} {FEATURE_MODALITY_16[i]:>8}"
              f"  {importance[i]:>11.4f}{sel}")

    print(f"\n  Selected features ({len(selected_indices)}):")
    for q, idx in enumerate(selected_indices):
        print(f"    qubit {q}: feature {idx:>2} ({FEATURE_NAMES_16[idx]:<12}"
              f" [{FEATURE_MODALITY_16[idx]}])  γ_{q} = {gamma_per_qubit[q]:.4f}")

    print(f"\n  Attention-selected entanglement pairs (qubit space 0-7):")
    for qi, qj in entanglement_pairs:
        fi = selected_indices[qi]
        fj = selected_indices[qj]
        print(f"    qubit {qi} ↔ qubit {qj}  "
              f"({FEATURE_NAMES_16[fi]} [{FEATURE_MODALITY_16[fi]}] ↔ "
              f"{FEATURE_NAMES_16[fj]} [{FEATURE_MODALITY_16[fj]}])"
              f"  A={A[np.ix_(selected_indices, selected_indices)][qi,qj]:.4f}")

    print()
