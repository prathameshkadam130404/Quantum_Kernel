import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import glob
from sklearn.svm import SVC
import config

def main():
    print("=== TEST 10: SVM TRAINABILITY ON EXISTING KERNELS ===")
    
    y_path = os.path.join(config.PROCESSED_DIR, "subsample_2000.npz")
    if not os.path.exists(y_path):
        print("FAIL: Cannot run SVM trainability test without labels.")
        sys.exit(1)
        
    y_train = np.load(y_path)['y_train']
    res_dir = os.path.join(config.RESULTS_DIR, "geometric_difference")
    search_pattern = os.path.join(res_dir, "**", "*.npy")
    files = glob.glob(search_pattern, recursive=True)
    kernel_files = [f for f in files if "K_" in os.path.basename(f) and "train" in os.path.basename(f)]
    
    if not kernel_files:
        print("WARNING: No predefined kernel matrices found to check.")
        print("Verdict: PENDING")
        sys.exit(0)
        
    print(f"\nFound {len(kernel_files)} kernels to check.")
    sub_size = min(200, len(y_train))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_train), size=sub_size, replace=False)
    y_sub = y_train[idx]
    
    print(f"{'Filename':<35} | {'Min Eig':<10} | {'SVs':<5} | {'Classes':<8} | {'Status'}")
    print("-" * 80)
    
    fail_count = 0
    warning_count = 0
    
    for f in kernel_files:
        name = os.path.basename(f)
        try:
            K = np.load(f)
            K_sub = K[np.ix_(idx, idx)]
            eigvals = np.linalg.eigvalsh(K_sub)
            min_eig = eigvals.min()
            
            clf = SVC(kernel="precomputed", C=1.0)
            clf.fit(K_sub, y_sub)
            n_sv = sum(clf.n_support_)
            preds = clf.predict(K_sub)
            u_classes = len(np.unique(preds))
            
            status = "PASS"
            if min_eig < -0.01:
                status = "FAIL (Not PSD)"
                fail_count += 1
            elif u_classes <= 1:
                status = "WARNING (Collapse)"
                warning_count += 1
                
            print(f"{name:<35} | {min_eig:<10.4f} | {n_sv:<5d} | {u_classes:<8d} | {status}")
            
        except Exception as e:
            print(f"{name:<35} | {'Error':<10} | {'-':<5} | {'-':<8} | FAIL ({e})")
            fail_count += 1
            
    print(f"\nSummary: {fail_count} Fails, {warning_count} Warnings")
    if fail_count > 0:
        print("Verdict: FAIL")
        sys.exit(1)
    elif warning_count > 0:
        print("Verdict: WARNING")
    else:
        print("Verdict: PASS")

if __name__ == "__main__":
    main()
