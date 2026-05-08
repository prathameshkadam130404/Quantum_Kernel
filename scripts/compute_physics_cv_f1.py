"""
5-fold CV F1 on physics kernel matrices.
Verifies that FQK F1=0.469 and RBF_CV F1=0.466 are not split-dependent.
Uses saved kernel matrices — no circuits needed.
"""

import sys, os
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.classifiers import train_precomputed_svm
from src.utils import ensure_dir

N_FOLDS = 5
RANDOM_SEED = 42

def cv_f1_from_kernel(K_train, y_train, n_folds=5, seed=42):
    """
    Stratified k-fold CV macro-F1 from a precomputed kernel matrix.
    For each fold: train SVM on K[train_idx, train_idx],
                   predict on K[test_idx, train_idx].
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_f1s = []
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_train)), y_train)):
        K_tr  = K_train[np.ix_(tr_idx, tr_idx)]
        K_val = K_train[np.ix_(val_idx, tr_idx)]
        y_tr  = y_train[tr_idx]
        y_val = y_train[val_idx]
        
        clf = train_precomputed_svm(K_tr, y_tr)
        y_pred = clf.predict(K_val)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        fold_f1s.append(float(f1))
        print(f"  Fold {fold+1}/{n_folds}: F1={f1:.4f}")
    
    return {
        "fold_f1s": fold_f1s,
        "mean_f1": float(np.mean(fold_f1s)),
        "std_f1":  float(np.std(fold_f1s)),
        "ci_95_low":  float(np.mean(fold_f1s) - 1.96*np.std(fold_f1s)/np.sqrt(n_folds)),
        "ci_95_high": float(np.mean(fold_f1s) + 1.96*np.std(fold_f1s)/np.sqrt(n_folds)),
    }

def main():
    phys_npz_path = "data/processed/physics_features_16.npz"
    if not os.path.exists(phys_npz_path):
        print(f"Error: {phys_npz_path} not found.")
        return
        
    y_phys = np.load(phys_npz_path)['y_train']

    kernels = {
        "FQK_physics":    "results/physics/fused/K_fqk_physics_train.npy",
        "AGPQK_physics":  "results/physics/fused/K_agpqk_physics_train.npy",
        "RBF_CV_physics": "results/physics/fused/K_rbf_cv_physics_train.npy",
        "PQK_physics":    "results/physics/fused/K_pqk_physics_train.npy",
        "CM_FQK_physics": "results/physics/fused/K_cm_fqk_physics_train.npy",
    }

    cv_results = {}
    for name, path in kernels.items():
        if not os.path.exists(path):
            if name == "RBF_CV_physics":
                print(f"\n{name}: Computing RBF_CV kernel...")
                X_phys = np.load(phys_npz_path)['X_train']
                from src.bandwidth import load_bandwidth_gamma
                from src.classical_kernels import compute_rbf_kernel
                gamma_cv = load_bandwidth_gamma("physics")
                K = compute_rbf_kernel(X_phys, gamma=gamma_cv)
                np.save(path, K)
            else:
                print(f"  {name}: kernel not found at {path}. Skipping.")
                continue
        else:
            K = np.load(path)
        
        print(f"\n{name}: 5-fold CV F1")
        cv_results[name] = cv_f1_from_kernel(K, y_phys, N_FOLDS, RANDOM_SEED)
        r = cv_results[name]
        print(f"  Mean F1 = {r['mean_f1']:.4f} ± {r['std_f1']:.4f}  "
              f"(95% CI: [{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}])")

    print("\n=== CV F1 SUMMARY ===")
    for name, r in cv_results.items():
        print(f"  {name:<25}: {r['mean_f1']:.4f} ± {r['std_f1']:.4f}")

    ensure_dir("results/reviewer_fixes")
    out_path = "results/reviewer_fixes/physics_cv_f1.json"
    with open(out_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()
