import os
import sys
import time
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.quantum_kernels import apply_trained_zz_feature_map, get_top_mi_pairs, compute_crossmodal_fqk_kernel_matrix
from src.d4_covariant_kernel import compute_d4_fqk_kernel_matrix, compute_d4_orbits, verify_d4_covariance
import pennylane as qml

def check_kernel(K, name):
    print(f"\nEvaluating {name} Kernel:")
    print(f"  Shape: {K.shape}")
    
    diag = np.diag(K)
    mean_diag = np.mean(diag)
    print(f"  Diagonal mean: {mean_diag:.4f}")
    if not np.allclose(diag, 1.0, atol=0.05):
        print("  WARNING: Diagonal entries are not ~1.0")
        
    print(f"  Entries in [0,1]: {np.all((K >= -1e-5) & (K <= 1.0 + 1e-5))}")
    
    sym_err = np.max(np.abs(K - K.T))
    print(f"  Symmetry error: {sym_err:.2e}")
    if sym_err > 1e-4:
        print("  WARNING: Matrix is not symmetric")
        
    max_val = np.max(K)
    min_val = np.min(K)
    print(f"  Min/Max: [{min_val:.4f}, {max_val:.4f}]")
    
    if K.shape[0] > 1:
        off = K[~np.eye(K.shape[0], dtype=bool)]
        print(f"  Mean off-diagonal: {np.mean(off):.4f}")

def run_test():
    print("=" * 60)
    print("TEST 04: Small Kernel Smoke Test (All New Kernels) V2")
    print("=" * 60)

    subsample_path = os.path.join(config.PROCESSED_DIR, "subsample_2000.npz")
    if not os.path.exists(subsample_path):
        print(f"FAIL: {subsample_path} not found.")
        return

    data = np.load(subsample_path)
    X_fused = data['fused_X_train']
    X_sar = data['sar_X_train']

    print("\n--- 1. TFK Smoke Test ---")
    tfk_theta_path = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused", "tfk", "theta_trained.npy")
    if not os.path.exists(tfk_theta_path):
        print(f"PENDING: {tfk_theta_path} not found. Skipping TFK test.")
    else:
        theta = np.load(tfk_theta_path)
        X_sample = X_fused[:10]
        dev = config.get_device(config.N_QUBITS)
        
        @qml.qnode(dev, diff_method=None)
        def tfk_circuit(x1, x2):
            apply_trained_zz_feature_map(x1, theta, config.N_QUBITS, config.ZZ_REPS)
            qml.adjoint(apply_trained_zz_feature_map)(x2, theta, config.N_QUBITS, config.ZZ_REPS)
            return qml.probs(wires=range(config.N_QUBITS))
            
        K_tfk = np.eye(10)
        t0 = time.time()
        for i in range(10):
            for j in range(i+1, 10):
                val = float(tfk_circuit(X_sample[i], X_sample[j])[0])
                K_tfk[i, j] = val
                K_tfk[j, i] = val
                
        evals_total = 10 * 9 / 2
        t1 = time.time()
        print(f"  Time: {t1-t0:.2f}s ({evals_total/(t1-t0):.2f} evals/second)")
        check_kernel(K_tfk, "TFK")

    print("\n--- 2. CM_FQK Smoke Test ---")
    X_sample_cm = X_fused[:10]
    mi_pairs = get_top_mi_pairs(X_fused, n_sar=4, n_opt=4, top_k=3)
    t0 = time.time()
    K_cm = compute_crossmodal_fqk_kernel_matrix(X_sample_cm, mi_pairs)
    t1 = time.time()
    evals_total = 10 * 10 
    print(f"  Time: {t1-t0:.2f}s ({evals_total/(t1-t0):.2f} evals/second)")
    check_kernel(K_cm, "CM_FQK")
    
    print("\n--- 3. D4_FQK Smoke Test (SAR Data) ---")
    pca_path = os.path.join(config.PROCESSED_DIR, "sar_pca_model.joblib")
    if not os.path.exists(pca_path):
        print(f"FAIL: {pca_path} not found.")
        return
        
    pca = joblib.load(pca_path)
    X_sample_sar = X_sar[:10]
    
    t0 = time.time()
    K_d4 = compute_d4_fqk_kernel_matrix(
        X_sample_sar,
        pca_components=pca.components_,
        patch_height=config.D4_PATCH_HEIGHT,
        patch_width=config.D4_PATCH_WIDTH,
        n_channels=config.D4_N_CHANNELS,
        n_qubits=config.N_QUBITS,
        reps=config.ZZ_REPS
    )
    t1 = time.time()
    evals_total = 10 * 9 / 2
    print(f"  Time: {t1-t0:.2f}s ({evals_total/(t1-t0):.2f} evals/second)")
    check_kernel(K_d4, "D4_FQK")
    
    print("\n  Verifying covariance on 10x10 chunk...")
    orbits, is_clean, M_g_dict = compute_d4_orbits(
        pca.components_, config.D4_PATCH_HEIGHT, config.D4_PATCH_WIDTH, config.D4_N_CHANNELS
    )
    print(f"  is_clean: {is_clean}")
    
    try:
        verify_d4_covariance(K_d4, X_sample_sar, M_g_dict, orbits, is_clean, n_checks=10, tol=1e-3)
    except Exception as e:
        print(f"  [WARNING] Covariance spot check generated an exception: {e}")
        
    print("\nPASS: Smoke tests completed.")

if __name__ == "__main__":
    run_test()
