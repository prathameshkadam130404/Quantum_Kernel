import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import glob
import config

def main():
    print("=== TEST 09: LABEL CONSISTENCY ACROSS KERNEL MATRICES ===")
    y_path = os.path.join(config.PROCESSED_DIR, "subsample_2000.npz")
    if not os.path.exists(y_path):
        print(f"FAIL: subsample_2000.npz not found")
        sys.exit(1)
        
    y_train = np.load(y_path)['y_train']
    n_samples = len(y_train)
    print(f"Loaded y_train from: {y_path}")
    print(f"Expected dimension: {n_samples}")
    
    res_dir = os.path.join(config.RESULTS_DIR, "geometric_difference")
    search_pattern = os.path.join(res_dir, "**", "*.npy")
    files = glob.glob(search_pattern, recursive=True)
    
    kernel_files = [f for f in files if "K_" in os.path.basename(f) and "train" in os.path.basename(f)]
    
    if not kernel_files:
        print("\nNo kernel matrices found. PENDING.")
        sys.exit(0)
    
    all_ok = True
    print("\nKernel Matrix Shapes:")
    print(f"{'Filename':<40} | {'Shape':<15} | {'Status'}")
    print("-" * 65)
    for f in kernel_files:
        try:
            K = np.load(f)
            sh = K.shape
            if len(sh) == 2 and sh[0] == n_samples and sh[1] == n_samples:
                # Square matrix check
                diag = np.diag(K).mean()
                if abs(diag - 1.0) > 0.01:
                    status = f"FAIL (diag={diag:.4f})"
                    all_ok = False
                else:
                    status = "PASS"
            else:
                status = "FAIL (shape mismatch)"
                all_ok = False
                
            print(f"{os.path.basename(f):<40} | {str(sh):<15} | {status}")
        except Exception as e:
            print(f"{os.path.basename(f):<40} | {'Error':<15} | FAIL ({e})")
            all_ok = False
            
    if not all_ok:
        print("\nVerdict: FAIL")
        sys.exit(1)
        
    print("\nVerdict: PASS")
    
if __name__ == "__main__":
    main()
