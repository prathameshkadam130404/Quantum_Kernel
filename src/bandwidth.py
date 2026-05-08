"""
bandwidth.py — CV-selected quantum kernel bandwidth utilities.

Bandwidth γ is selected by 5-fold cross-validation on the training set,
following Shaydulin & Wild (2022), Phys. Rev. A 107, 042113.

γ controls the encoding range: features are scaled as X_encoded = γ * X_raw,
where X_raw ∈ [0, π] (normalised by extract_physics_features.py or setup_data.py).

CV-selected values (results/bandwidth_cv/selected_gamma.json):
  PCA features:     γ* = 1.000  →  range [0, π]      (default, no change)
  Physics features: γ* = 0.750  →  range [0, 0.75π]
"""

import os
import json
import numpy as np

# Path relative to project root
_GAMMA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "bandwidth_cv", "selected_gamma.json"
)

# Fallback values if file is missing (should not happen in normal runs)
_FALLBACK = {
    "pca":     1.000,
    "physics": 0.750,
}


def load_bandwidth_gamma(feature_set: str) -> float:
    """
    Load the CV-selected bandwidth scalar γ for the given feature set.

    Args:
        feature_set: one of "pca" or "physics"

    Returns:
        γ (float): bandwidth scalar. Features should be multiplied by γ
                   after [0,π] normalisation and before kernel computation.

    Raises:
        ValueError: if feature_set is not recognised
    """
    if feature_set not in ("pca", "physics"):
        raise ValueError(
            f"feature_set must be 'pca' or 'physics', got '{feature_set}'"
        )

    if not os.path.exists(_GAMMA_FILE):
        gamma = _FALLBACK[feature_set]
        print(
            f"[bandwidth] WARNING: {_GAMMA_FILE} not found. "
            f"Using fallback γ={gamma} for '{feature_set}'."
        )
        return gamma

    with open(_GAMMA_FILE) as f:
        data = json.load(f)

    key = f"gamma_{feature_set}"
    if key not in data:
        gamma = _FALLBACK[feature_set]
        print(
            f"[bandwidth] WARNING: '{key}' not in gamma file. "
            f"Using fallback γ={gamma}."
        )
        return gamma

    gamma = float(data[key])
    return gamma


def apply_bandwidth(X: np.ndarray, feature_set: str) -> np.ndarray:
    """
    Apply CV-selected bandwidth scaling to feature matrix X.

    X is assumed to already be normalised to [0, π] (as produced by
    extract_physics_features.py and setup_data.py). This function
    scales X to [0, γ*π] using the CV-selected γ.

    Args:
        X:            feature matrix, shape (n, d), values in [0, π]
        feature_set:  "pca" or "physics"

    Returns:
        X_scaled:     X * γ, shape (n, d), values in [0, γ*π]
    """
    gamma = load_bandwidth_gamma(feature_set)
    if abs(gamma - 1.0) < 1e-9:
        # γ=1.0 means no change — return original to avoid unnecessary copy
        return X
    return X * gamma


def log_bandwidth_info(X: np.ndarray, feature_set: str, logger=None) -> float:
    """
    Load γ, log the encoding range info, and return γ. Convenience wrapper.

    Args:
        X:           feature matrix after bandwidth scaling (used only for range logging)
        feature_set: "pca" or "physics"
        logger:      optional logger; uses print() if None

    Returns:
        γ (float)
    """
    gamma = load_bandwidth_gamma(feature_set)
    encoding_max = gamma * np.pi
    msg = (
        f"  Bandwidth γ = {gamma:.4f}  "
        f"(CV-selected, Shaydulin & Wild 2022)  "
        f"Encoding range: [0, {encoding_max:.4f}] = [0, {gamma:.3f}π]  "
        f"Feature range after scaling: [{X.min():.4f}, {X.max():.4f}]"
    )
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
    return gamma
