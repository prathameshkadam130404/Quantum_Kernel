import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.d4_representation import compute_m_g_matrices, _is_approx_permutation, extract_orbits_from_m_g

def run_test():
    print("=" * 60)
    print("TEST 03: D4 M_e Identity Check (SAR Only)")
    print("=" * 60)

    try:
        sar_pca_path = os.path.join(config.PROCESSED_DIR, "sar_pca_model.joblib")
        if not os.path.exists(sar_pca_path):
            print(f"FAIL: {sar_pca_path} not found.")
            return
        
        pca = joblib.load(sar_pca_path)
        pca_components = pca.components_
        
        M_g_dict = compute_m_g_matrices(
            pca_components=pca_components,
            patch_height=config.D4_PATCH_HEIGHT,
            patch_width=config.D4_PATCH_WIDTH,
            n_channels=config.D4_N_CHANNELS
        )
        
        M_e = M_g_dict['e']
        I = np.eye(config.N_QUBITS)
        err = np.max(np.abs(M_e - I))
        
        print(f"[CHECK] ||M_e - I||_inf < 1e-4 — Actual: {err:.2e}")
        if err >= 1e-4:
            print("FAIL: Identity matrix error is too large.")
            return
            
        print("\nAll 8 M_g matrix characteristics:")
        n_dense = 0
        is_clean_all = True
        
        for g, m in M_g_dict.items():
            norm = np.linalg.norm(m)
            max_offdiag = 0.0
            if m.shape[0] > 1:
                offdiags = np.abs(m - np.diag(np.diagonal(m)))
                max_offdiag = np.max(offdiags)
            
            is_perm = _is_approx_permutation(m)
            if not is_perm:
                n_dense += 1
                is_clean_all = False
                
            print(f"  g={g:2s} | max_offdiag: {max_offdiag:.4f} | is_permutation: {is_perm}")

        print(f"\n[INFO] n_dense matrices: {n_dense}/8")
        
        orbits, is_clean = extract_orbits_from_m_g(M_g_dict)
        print(f"[INFO] Reported is_clean: {is_clean}")
        
        if is_clean:
            print("WARNING: is_clean is True. For SAR+PCA, we expect continuous group-averaging (is_clean=False) because PCA mixes spatial features. Verify if this is correct.")
            print("\nPASS: Identity check passed, but is_clean=True is unexpected.")
        else:
            print("\nPASS: Identity check passed. Continuous (group-averaging) path will be used (expected).")

    except Exception as e:
        print(f"\nFAIL: An exception occurred: {e}")

if __name__ == "__main__":
    run_test()
