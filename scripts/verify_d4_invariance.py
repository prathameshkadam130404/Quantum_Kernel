"""
Standalone script to verify D4 invariance of the SAR-D4 PCA model.
Loads existing results and runs the mathematical consistency check.
"""
import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.utils import setup_logging
from src.d4_augmented_pca import verify_invariance

logger = setup_logging("verify_invariance")

# Paths (matching exp_d4_standalone.py)
D4_DIR           = os.path.join("results", "d4_standalone")
PCA_SAVE_PATH    = os.path.join(D4_DIR, "sar_d4_pca_model.joblib")
X_TRAIN_PATH     = os.path.join(D4_DIR, "X_sar_d4_train.npy")

D4_N_CHANNELS    = 8
D4_PATCH_HEIGHT  = 32
D4_PATCH_WIDTH   = 32

def main():
    if not os.path.exists(PCA_SAVE_PATH):
        print(f"Error: PCA model not found at {PCA_SAVE_PATH}")
        return

    # Load raw SAR flat patches — NOT the PCA-projected X_train
    # We need 8192-dim raw vectors to apply D4 pixel permutations
    RAW_TRAIN_PATH = os.path.join(D4_DIR, "X_sar_d4_raw_train.npy")
    if not os.path.exists(RAW_TRAIN_PATH):
        print(f"Error: Raw SAR patches not found at {RAW_TRAIN_PATH}")
        print("Re-run exp_d4_standalone.py to generate raw patch cache.")
        print("Or set RAW_TRAIN_PATH to the correct path of the raw")
        print("channel-first flat SAR patches (shape: n x 8192).")
        return

    print(f"Loading PCA model: {PCA_SAVE_PATH}")
    pca = joblib.load(PCA_SAVE_PATH)

    print(f"Loading raw SAR patches from: {RAW_TRAIN_PATH}")
    X_raw = np.load(RAW_TRAIN_PATH)   # shape (n, 8192), channel-first flat

    # Also load normalisation params fitted on training data
    X_MIN_PATH   = os.path.join(D4_DIR, "x_min.npy")
    X_RANGE_PATH = os.path.join(D4_DIR, "x_range.npy")
    if not os.path.exists(X_MIN_PATH) or not os.path.exists(X_RANGE_PATH):
        print("Error: x_min.npy or x_range.npy not found. Rerun exp_d4_standalone.py.")
        return

    x_min   = np.load(X_MIN_PATH)
    x_range = np.load(X_RANGE_PATH)

    # Take first 20 raw samples for the check
    X_sample_raw = X_raw[:20]   # shape (20, 8192)

    # Define the full pipeline: raw patch -> PCA -> normalise -> [0, pi]
    def theta(x_raw_batch):
        """
        x_raw_batch: shape (n, 8192), channel-first flat
        Returns: shape (n, 8), normalised to [0, pi]
        """
        x_pca = pca.transform(x_raw_batch)           # (n, 8)
        x_norm = (x_pca - x_min) / x_range * np.pi   # (n, 8)
        return np.clip(x_norm, 0, np.pi)

    # D4 group elements and their pixel permutations
    from src.d4_representation import get_d4_pixel_permutation

    D4_ELEMENTS = ['e', 'r', 'r2', 'r3', 's', 'sr', 'sr2', 'sr3']
    n_spatial = D4_PATCH_HEIGHT * D4_PATCH_WIDTH   # 1024

    def apply_d4_to_raw(x_raw, elem):
        """
        Apply D4 element to a raw channel-first flat patch.
        x_raw: shape (8192,) — channel-first: ch0[0..1023], ch1[1024..2047],...
        Returns permuted patch shape (8192,)
        """
        spatial_perm = get_d4_pixel_permutation(elem, D4_PATCH_HEIGHT, D4_PATCH_WIDTH)
        # Build full permutation: each channel's pixels are permuted identically
        full_perm = np.concatenate([
            spatial_perm + ch * n_spatial
            for ch in range(D4_N_CHANNELS)
        ])
        return x_raw[full_perm]

    print(f"\nRunning Raw-Space Invariance Verification (correct test)...")
    print(f"Testing: theta(x_raw) ≈ theta(g * x_raw) for each g in D4")
    print(f"Samples: {len(X_sample_raw)}, Tolerance: 1e-4")
    print("-" * 60)

    max_err_overall = 0.0
    passed = True
    tol = 1e-4

    # Reference: theta(original raw patches)
    theta_orig = theta(X_sample_raw)   # (20, 8)

    for elem in D4_ELEMENTS[1:]:  # Skip identity 'e'
        # Apply D4 to each raw patch
        X_rotated = np.array([
            apply_d4_to_raw(X_sample_raw[i], elem)
            for i in range(len(X_sample_raw))
        ])   # (20, 8192)

        # Project through pipeline
        theta_rotated = theta(X_rotated)   # (20, 8)

        # Compute per-sample inf-norm error
        for i in range(len(X_sample_raw)):
            err = float(np.max(np.abs(theta_orig[i] - theta_rotated[i])))
            max_err_overall = max(max_err_overall, err)
            if err > tol:
                passed = False
                print(f"  FAIL g={elem}, sample {i}: "
                      f"||theta(x) - theta(g*x)||_inf = {err:.8f} > tol={tol}")

    print("-" * 60)
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print(f"Max Error: {max_err_overall:.8e}")
    print(f"Tolerance: {tol}")
    print("-" * 60)

    if passed:
        print("Success: orbit-augmented PCA is correctly D4-invariant.")
        print("theta(x) ≈ theta(g*x) for all g in D4 within tolerance 1e-4.")
    else:
        print("Failure: theta(x) != theta(g*x) for some g in D4.")
        print("The orbit-augmented PCA subspace is not D4-invariant.")
        print("Check: n_augmentation_samples, patch layout (channel-first),")
        print("       and that fit_d4_augmented_pca used all 8 D4 transforms.")

if __name__ == "__main__":
    main()
