import os
import sys
import time
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.quantum_kernels import apply_trained_zz_feature_map, get_top_mi_pairs, compute_crossmodal_fqk_kernel_matrix, compute_fqk_kernel_matrix
from src.d4_covariant_kernel import compute_d4_fqk_kernel_matrix

def run_test():
    print("=" * 60)
    print("TEST 11: Compute Time Estimate (V2 - Separate Modalities)")
    print("=" * 60)

    try:
        subsample_path = os.path.join(config.PROCESSED_DIR, "subsample_2000.npz")
        if not os.path.exists(subsample_path):
            print(f"FAIL: {subsample_path} not found.")
            return

        data = np.load(subsample_path)
        X_fused = data['fused_X_train']
        X_sar = data['sar_X_train']
        
        n_train = len(X_fused)
        n_test = 2000 # typical size
        
        evals_train = n_train * (n_train - 1) / 2
        evals_test = n_test * n_train
        
        print(f"Settings: N_TRAIN={n_train}, N_TEST={n_test}")
        print(f"Evals/Train: {evals_train:,.0f} | Evals/Test: {evals_test:,.0f}\n")

        timings = {}

        # 1. FQK (Fused)
        t0 = time.time()
        # Compute 20 evaluations means computing a small rect: 4x5 = 20 evals, or loop. Loop is closer to reality
        X1 = X_fused[:4]
        X2 = X_fused[4:9]
        compute_fqk_kernel_matrix(X1, X2=X2)
        t1 = time.time()
        timings['FQK'] = (t1-t0) / 20.0
        print(f"FQK Base Time Measured.")

        # 2. TFK (Fused)
        tfk_theta_path = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused", "tfk", "theta_trained.npy")
        if os.path.exists(tfk_theta_path):
            theta = np.load(tfk_theta_path)
            t0 = time.time()
            import pennylane as qml
            dev = config.get_device(config.N_QUBITS)
            @qml.qnode(dev, diff_method=None)
            def tfk_c(x1, x2):
                apply_trained_zz_feature_map(x1, theta, config.N_QUBITS, config.ZZ_REPS)
                qml.adjoint(apply_trained_zz_feature_map)(x2, theta, config.N_QUBITS, config.ZZ_REPS)
                return qml.probs(wires=range(config.N_QUBITS))
            for i in range(20):
                tfk_c(X_fused[0], X_fused[min(i, len(X_fused)-1)])
            t1 = time.time()
            timings['TFK'] = (t1-t0) / 20.0
            print(f"TFK Time Measured.")
        else:
            timings['TFK'] = float('nan')
            print("TFK Theta missing. Skipped.")
            
        # 3. CM_FQK (Fused)
        mi_pairs = get_top_mi_pairs(X_fused, n_sar=4, n_opt=4, top_k=3)
        t0 = time.time()
        compute_crossmodal_fqk_kernel_matrix(X1, mi_pairs, X2=X2)
        t1 = time.time()
        timings['CM_FQK'] = (t1-t0) / 20.0
        print(f"CM_FQK Time Measured.")
        
        # 4. D4_FQK (SAR)
        sar_pca_path = os.path.join(config.PROCESSED_DIR, "sar_pca_model.joblib")
        if os.path.exists(sar_pca_path):
            pca_sar = joblib.load(sar_pca_path)
            X1_sar = X_sar[:4]
            X2_sar = X_sar[4:9]
            t0 = time.time()
            compute_d4_fqk_kernel_matrix(
                X1_sar,
                pca_components=pca_sar.components_,
                patch_height=32,
                patch_width=32,
                n_channels=8,
                X2=X2_sar
            )
            t1 = time.time()
            timings['D4_FQK'] = (t1-t0) / 20.0
            print(f"D4_FQK Time Measured.")
        else:
            timings['D4_FQK'] = float('nan')
            print("SAR PCA missing. Skipped.")

        print("\n" + "="*70)
        print(f"{'Kernel':<10} | {'Modality':<8} | {'secs/eval':<10} | {'Train (hrs)':<12} | {'Test (hrs)':<12}")
        print("-" * 70)
        
        mods = {'FQK': 'fused', 'TFK': 'fused', 'CM_FQK': 'fused', 'D4_FQK': 'SAR'}
        for k in ['FQK', 'TFK', 'CM_FQK', 'D4_FQK']:
            sec_eval = timings[k]
            if np.isnan(sec_eval):
                print(f"{k:<10} | {mods[k]:<8} | {'N/A':<10} | {'N/A':<12} | {'N/A':<12}")
            else:
                train_hrs = (sec_eval * evals_train) / 3600.0
                test_hrs = (sec_eval * evals_test) / 3600.0
                print(f"{k:<10} | {mods[k]:<8} | {sec_eval:<10.5f} | {train_hrs:<12.2f} | {test_hrs:<12.2f}")
        
        print("=" * 70)
        print("\nPASS: Compute time estimation completed.")

    except Exception as e:
        print(f"\nFAIL: An exception occurred: {e}")

if __name__ == "__main__":
    run_test()
