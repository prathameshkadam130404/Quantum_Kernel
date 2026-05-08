import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import config
from src.data_loader import load_modality
from src.quantum_kernels import get_top_mi_pairs
from src.geometric_bound import compute_cross_modal_mutual_information

def main():
    print("=== TEST 05: CM_FQK MI PAIRS VALIDATION ===")
    
    try:
        data = load_modality("fused")
        X = data["X_train"]
    except Exception as e:
        print(f"FAIL: Could not load fused data: {e}")
        sys.exit(1)
        
    print(f"X_train shape: {X.shape}")
    if X.shape[1] != 8:
        print(f"FAIL: Expected X_train to have 8 features, got {X.shape[1]}")
        sys.exit(1)
        
    mi_pairs = get_top_mi_pairs(X, n_sar=4, n_opt=4, top_k=3)
    print(f"Selected MI pairs: {mi_pairs}")
    
    mi_matrix = np.zeros((4, 4))
    for s in range(4):
        for o in range(4):
            sar_feat = X[:, s:s+1]
            opt_feat = X[:, o+4:o+5]
            val = compute_cross_modal_mutual_information(sar_feat, opt_feat)
            mi_matrix[s, o] = val
            
    print("\nFull MI Matrix (Rows=SAR, Cols=Opt):")
    print(np.round(mi_matrix, 4))
    
    for i, j in mi_pairs:
        if not (0 <= i <= 3):
            print(f"FAIL: Pair ({i},{j}) has invalid SAR index {i}")
            sys.exit(1)
        if not (4 <= j <= 7):
            print(f"FAIL: Pair ({i},{j}) has invalid Opt index {j}")
            sys.exit(1)
            
        mi_val = mi_matrix[i, j - 4]
        if mi_val <= 0:
            print(f"FAIL: Pair ({i},{j}) has non-positive MI: {mi_val:.4f}")
            sys.exit(1)
            
    top_mi = np.max(mi_matrix)
    print(f"\nTop MI value: {top_mi:.4f}")
    
    if top_mi < 0.01:
        print("WARNING: Top MI is < 0.01. Cross-modal signal might be weak.")
        print("Verdict: WARNING")
    else:
        print("Verdict: PASS")
        
if __name__ == "__main__":
    main()
