import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import config

def compute_effective_rank(K):
    eigvals = np.linalg.eigvalsh(K)
    max_eig = eigvals.max()
    if max_eig < 1e-10:
        return 0.0
    return np.trace(K) / max_eig

def check_file(filepath, is_train):
    if not os.path.exists(filepath):
        return {"status": "MISSING", "msg": "File not found"}
        
    try:
        K = np.load(filepath)
    except Exception as e:
        return {"status": "FAIL (Corrupt)", "msg": str(e)}
        
    shape = K.shape
    has_nan_inf = np.isnan(K).any() or np.isinf(K).any()
    
    if has_nan_inf:
        return {"status": "FAIL", "msg": "Contains NaN or Inf"}
        
    if K.min() < -0.01 or K.max() > 1.01:
        return {"status": "FAIL", "msg": f"Bounds violated: [{K.min():.4f}, {K.max():.4f}]"}
        
    msg = f"Shape: {shape}"
    if is_train:
        diag = np.diag(K).mean()
        if abs(diag - 1.0) > 0.01:
            return {"status": "FAIL", "msg": f"Diag off {diag:.4f}"}
        sym_err = np.max(np.abs(K - K.T))
        if sym_err > 1e-4:
            return {"status": "FAIL", "msg": f"Asymmetric (err {sym_err:.4f})"}
            
        er = compute_effective_rank(K)
        msg += f", ERank: {er:.2f}"
        
    return {"status": "PASS", "msg": msg}

def main():
    print("=== TEST 12: PRECOMPUTED KERNEL INTEGRITY CHECK ===")
    res_dir = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
    
    files = {
        "K_fqk_train": os.path.join(res_dir, "K_fqk_train.npy"),
        "K_fqk_test": os.path.join(res_dir, "K_fqk_test.npy"),
        "K_pqk_train": os.path.join(res_dir, "K_pqk_train.npy"),
        "K_pqk_test": os.path.join(res_dir, "K_pqk_test.npy"),
    }
    
    print(f"{'Kernel Type':<15} | {'Status':<15} | {'Details'}")
    print("-" * 75)
    
    all_pass = True
    for name, filepath in files.items():
        is_train = "train" in name
        res = check_file(filepath, is_train)
        status = res["status"]
        if "FAIL" in status or "MISSING" in status:
            all_pass = False
        print(f"{name:<15} | {status:<15} | {res['msg']}")
        
    if all_pass:
        print("\nVerdict: PASS")
    else:
        print("\nVerdict: PENDING/FAIL")

if __name__ == "__main__":
    main()
