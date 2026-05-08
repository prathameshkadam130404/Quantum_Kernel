import os
import sys
import numpy as np
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.d4_covariant_kernel import compute_d4_fqk_kernel_matrix, verify_d4_covariance, compute_d4_orbits
from src.d4_representation import compute_m_g_matrices
from src.quantum_kernels import apply_trained_zz_feature_map
import pennylane as qml

def run_test():
    print("=" * 60)
    print("TEST 06: D4 Covariance Spot Check (SAR Only)")
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
            print("NOTE: Mathematical invariance is expected to fail (~0.04 error) until exp_d4_standalone.py runs.")
            pca_path = std_pca_path
        else:
            print(f"FAIL: Neither {d4_pca_path} nor {std_pca_path} found.")
            return

        pca = joblib.load(pca_path)
        M_g_dict = compute_m_g_matrices(
            pca.components_, 32, 32, 8 # Standalone uses n_channels=8
        )

        subsample_path = os.path.join(config.PROCESSED_DIR, "subsample_2000.npz")
        standalone_x_path = os.path.join("results", "d4_standalone", "X_sar_d4_train.npy")
        
        if os.path.exists(standalone_x_path):
            print(f"Using STANDALONE D4 FEATURES: {standalone_x_path}")
            X_sar = np.load(standalone_x_path)
        elif os.path.exists(subsample_path):
            print(f"Using SUBSAMPLE SAR DATA: {subsample_path}")
            data = np.load(subsample_path)
            X_sar = data['sar_X_train']
        else:
            print(f"FAIL: No data found at {standalone_x_path} or {subsample_path}")
            return
        
        np.random.seed(config.RANDOM_SEED)
        indices = np.random.choice(len(X_sar), 5, replace=False)
        X_sample = X_sar[indices]

        print("Computing 5x5 Reference Kernel...")
        K_ref = compute_d4_fqk_kernel_matrix(
            X_sample,
            pca_components=pca.components_,
            patch_height=32,
            patch_width=32,
            n_channels=8,
            n_qubits=config.N_QUBITS,
            reps=config.ZZ_REPS
        )

        d4_elements = list(M_g_dict.keys())
        orbits, is_clean, _ = compute_d4_orbits(
            pca.components_, 32, 32, 8
        )
        
        dev = config.get_device(config.N_QUBITS)
        
        if is_clean:
            from src.d4_covariant_kernel import apply_d4_covariant_feature_map_discrete
            @qml.qnode(dev, diff_method=None)
            def check_circuit(x1, x2):
                apply_d4_covariant_feature_map_discrete(x1, orbits, config.N_QUBITS, config.ZZ_REPS)
                qml.adjoint(apply_d4_covariant_feature_map_discrete)(x2, orbits, config.N_QUBITS, config.ZZ_REPS)
                return qml.probs(wires=range(config.N_QUBITS))
        else:
            from src.d4_covariant_kernel import apply_d4_covariant_feature_map_continuous
            @qml.qnode(dev, diff_method=None)
            def check_circuit(x1, x2):
                apply_d4_covariant_feature_map_continuous(x1, M_g_dict, config.N_QUBITS, config.ZZ_REPS)
                qml.adjoint(apply_d4_covariant_feature_map_continuous)(x2, M_g_dict, config.N_QUBITS, config.ZZ_REPS)
                return qml.probs(wires=range(config.N_QUBITS))

        print("\nManual Spot Checks (5 random pair/group combinations):")
        local_max_err = 0.0
        for i in range(5):
            pair_i, pair_j = np.random.choice(5, 2, replace=False)
            g_elem = np.random.choice(d4_elements)
            M_g = M_g_dict[g_elem]
            
            x_i_trans = np.clip(M_g @ X_sample[pair_i], 0.0, np.pi)
            x_j_trans = np.clip(M_g @ X_sample[pair_j], 0.0, np.pi)
            
            k_orig = K_ref[pair_i, pair_j]
            k_trans = float(check_circuit(x_i_trans, x_j_trans)[0])
            
            err = abs(k_orig - k_trans)
            local_max_err = max(local_max_err, err)
            print(f"  Check {i+1}: g={g_elem:2s}, pair=({pair_i},{pair_j}) | K_orig: {k_orig:.4f}, K_trans: {k_trans:.4f} | Err: {err:.4e}")
            if err > 1e-3:
                print(f"FAIL: Spot check {i+1} exceeded error tolerance.")
                return 

        print(f"\nManual max error: {local_max_err:.4e} (tol: 1e-3)")

        print("\nRunning module verify_d4_covariance()...")
        res = verify_d4_covariance(
            K_ref, X_sample, M_g_dict, orbits, is_clean, n_checks=10, tol=1e-3
        )
        
        if not res:
            print("FAIL: verify_d4_covariance() failed.")
        else:
            print("\nPASS: D4 Covariance verified correctly.")

    except Exception as e:
        print(f"\nFAIL: An exception occurred: {e}")

if __name__ == "__main__":
    run_test()
