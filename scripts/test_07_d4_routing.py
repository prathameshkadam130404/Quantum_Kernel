import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.d4_representation import compute_m_g_matrices, extract_orbits_from_m_g, _is_approx_permutation

def run_test():
    print("=" * 60)
    print("TEST 07: D4 Routing Verification (SAR Only)")
    print("=" * 60)

    try:
        # Priority: D4-augmented PCA from standalone experiment
        d4_pca_path = os.path.join("results", "d4_standalone", "sar_d4_pca_model.joblib")
        std_pca_path = os.path.join(config.PROCESSED_DIR, "sar_pca_model.joblib")
        
        if os.path.exists(d4_pca_path):
            print(f"Using D4-AUGMENTED PCA: {d4_pca_path}")
            pca_path = d4_pca_path
        elif os.path.exists(std_pca_path):
            print(f"WARNING: D4-augmented PCA not found. Falling back to STANDARD SAR PCA: {std_pca_path}")
            print("NOTE: Proper routing check expects D4-augmented PCA from exp_d4_standalone.py.")
            pca_path = std_pca_path
        else:
            print(f"FAIL: Neither {d4_pca_path} nor {std_pca_path} found.")
            return

        pca = joblib.load(pca_path)
        M_g_dict = compute_m_g_matrices(
            pca.components_, 32, 32, 8 # Standalone uses n_channels=8
        )

        orbits, is_clean = extract_orbits_from_m_g(M_g_dict)

        print("M_g Array Diagnostics:")
        n_dense = 0
        for g, m in M_g_dict.items():
            max_offdiag = 0.0
            if m.shape[0] > 1:
                offdiags = np.abs(m - np.diag(np.diagonal(m)))
                max_offdiag = np.max(offdiags)
            
            is_perm = _is_approx_permutation(m)
            if not is_perm: n_dense += 1
            print(f"  g={g:2s} | Max Off-Diag: {max_offdiag:.4f} | Is Permutation: {is_perm}")

        print("\nSummary:")
        print(f"  Number of Dense Matrices (non-permutation): {n_dense} / 8")
        print(f"  is_clean result from extract_orbits_from_m_g: {is_clean}")
        
        routing_decision = "DISCRETE (Orbit-sharing)" if is_clean else "CONTINUOUS (Group-averaging)"
        print(f"  Routing Decision: {routing_decision}")
        
        if is_clean:
            print("\nWARNING: is_clean=True is unexpected for SAR data due to spatial mixing in PCA components.")
            print("Please manually inspect the PCA transformation limits.")
        else:
            print("\nPASS: Routing logic behaves correctly for mixed spatial inputs.")
            print("Confirm continuous path: theta(x) = (1/8) * sum M_g @ x")

    except Exception as e:
        print(f"\nFAIL: An exception occurred: {e}")

if __name__ == "__main__":
    run_test()
