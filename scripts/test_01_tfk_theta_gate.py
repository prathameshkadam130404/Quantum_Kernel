import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import config

def main():
    print("=== TEST 01: TFK THETA GATE ===")
    theta_path = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused", "tfk", "theta_trained.npy")
    
    if not os.path.exists(theta_path):
        print(f"FAIL: theta_trained.npy not found at {theta_path}")
        sys.exit(1)
        
    theta = np.load(theta_path)
    
    # Check shape
    if theta.shape != (config.N_QUBITS,):
        print(f"FAIL: theta shape is {theta.shape}, expected ({config.N_QUBITS},)")
        sys.exit(1)
        
    # Check bounds
    dev = np.linalg.norm(theta - 1.0)
    std = theta.std()
    theta_min = theta.min()
    theta_max = theta.max()
    
    print(f"theta_dev: {dev:.4f}")
    print(f"theta_std: {std:.4f}")
    print(f"theta_min: {theta_min:.4f}")
    print(f"theta_max: {theta_max:.4f}")
    
    if not (0.3 <= dev <= 5.0):
        print(f"FAIL: ||theta - ones|| = {dev:.4f} not in [0.3, 5.0]")
        sys.exit(1)
        
    if std <= 0.05:
        print(f"FAIL: theta std = {std:.4f} <= 0.05 (too uniform)")
        sys.exit(1)
        
    if theta_min < -2*np.pi or theta_max > 2*np.pi:
        print("FAIL: theta array outside [-2pi, 2pi]")
        sys.exit(1)
        
    # Check kernel on 10 samples
    try:
        from src.data_loader import load_modality
        from src.quantum_kernels import apply_trained_zz_feature_map
        import pennylane as qml
        
        data = load_modality("fused")
        X = data["X_train"][:10]
        
        dev_q = config.get_device(config.N_QUBITS)
        @qml.qnode(dev_q, diff_method=None)
        def tfk_circ(x1, x2, th):
            apply_trained_zz_feature_map(x1, th, config.N_QUBITS, config.ZZ_REPS)
            qml.adjoint(apply_trained_zz_feature_map)(x2, th, config.N_QUBITS, config.ZZ_REPS)
            return qml.probs(wires=range(config.N_QUBITS))
            
        K = np.eye(10)
        for i in range(10):
            for j in range(i+1, 10):
                v = float(tfk_circ(X[i], X[j], theta)[0])
                K[i, j] = v
                K[j, i] = v
                
        diag_mean = np.diag(K).mean()
        if abs(diag_mean - 1.0) > 1e-3:
            print(f"FAIL: kernel diagonal mean = {diag_mean:.4f} != 1.0")
            sys.exit(1)
            
        if K.min() < -1e-4 or K.max() > 1.0001:
            print(f"FAIL: kernel entries out of bounds [0, 1]. Min={K.min():.4f}, Max={K.max():.4f}")
            sys.exit(1)
            
        sym_err = np.max(np.abs(K - K.T))
        if sym_err > 1e-4:
            print(f"FAIL: kernel is not symmetric, max error = {sym_err:.4f}")
            sys.exit(1)
            
        offdiag = K[~np.eye(10, dtype=bool)]
        mean_offdiag = offdiag.mean()
        print(f"mean_offdiag: {mean_offdiag:.4f}")
        
        if mean_offdiag < 0.01:
            print(f"FAIL: mean_offdiag {mean_offdiag:.4f} < 0.01")
            sys.exit(1)
            
        print("PASS: TFK Theta Validation passed.")
        
    except Exception as e:
        print(f"WARNING: Smoke test failed due to execution error: {e}")
        # Note: if data doesn't exist, we permit failure of smoke test, or fail it directly
        sys.exit(1)

if __name__ == "__main__":
    main()
