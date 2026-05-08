"""
Permutation test for KTA significance.
Tests H0: ΔKTA(quantum - classical) <= 0 using label permutations.
No circuit recomputation — uses saved kernel matrices only.
"""

import sys, os
import numpy as np
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.classical_kernels import compute_rbf_kernel
from src.bandwidth import load_bandwidth_gamma
from src.utils import setup_logging, ensure_dir

N_PERMUTATIONS = 1000
RANDOM_SEED    = 42

def centered_kta_fast(K, y, class_weighted=True):
    """Fast centered class-weighted KTA from kernel matrix and labels."""
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    
    # Center kernel
    col_mean = K.mean(axis=0)
    row_mean = K.mean(axis=1)
    grand    = K.mean()
    Kc = K - col_mean[None,:] - row_mean[:,None] + grand
    
    # Build ideal kernel
    if class_weighted:
        count_map = dict(zip(classes, counts))
        Ki = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if y[i] == y[j]:
                    Ki[i,j] = 1.0 / count_map[y[i]]
    else:
        Ki = (y[:,None] == y[None,:]).astype(float)
    
    col_m = Ki.mean(axis=0); row_m = Ki.mean(axis=1); gm = Ki.mean()
    Kic = Ki - col_m[None,:] - row_m[:,None] + gm
    
    num  = np.sum(Kc * Kic)
    denom = np.linalg.norm(Kc,'fro') * np.linalg.norm(Kic,'fro')
    return float(num / (denom + 1e-12))


def run_permutation_test(K1, K2, y, label1, label2, n_perm=1000, seed=42):
    """
    Permutation test for H0: KTA(K1) <= KTA(K2).
    
    Shuffles y_train n_perm times and measures how often
    the permuted ΔKTA exceeds the observed ΔKTA.
    
    Returns dict with observed values, p-value, CI.
    """
    rng = np.random.default_rng(seed)
    
    # Observed ΔKTA
    kta1_obs = centered_kta_fast(K1, y)
    kta2_obs = centered_kta_fast(K2, y)
    delta_obs = kta1_obs - kta2_obs
    
    print(f"\nComparing {label1} vs {label2}:")
    print(f"  {label1} KTA = {kta1_obs:.6f}")
    print(f"  {label2} KTA = {kta2_obs:.6f}")
    print(f"  Observed ΔKTA = {delta_obs:+.6f}")
    print(f"  Running {n_perm} permutations...")
    
    # Permutation distribution
    delta_perm = np.zeros(n_perm)
    for i in tqdm(range(n_perm), desc="Permuting"):
        y_perm = rng.permutation(y)
        kta1_p = centered_kta_fast(K1, y_perm)
        kta2_p = centered_kta_fast(K2, y_perm)
        delta_perm[i] = kta1_p - kta2_p
    
    # Two-sided p-value for H0: ΔKTA = 0
    p_val = float(np.mean(np.abs(delta_perm) >= np.abs(delta_obs)))
    # One-sided p-value for H0: ΔKTA <= 0
    p_one = float(np.mean(delta_perm >= delta_obs))
    
    # 95% CI of observed ΔKTA under permutation null
    ci_low  = float(np.percentile(delta_perm, 2.5))
    ci_high = float(np.percentile(delta_perm, 97.5))
    
    print(f"  p-value (two-sided) = {p_val:.4f}")
    print(f"  p-value (one-sided, H0: ΔKTA<=0) = {p_one:.4f}")
    print(f"  Null distribution 95% CI: [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"  Result: {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'} at α=0.05")
    
    return {
        "kernel_1": label1,
        "kernel_2": label2,
        "kta_1_observed": float(kta1_obs),
        "kta_2_observed": float(kta2_obs),
        "delta_kta_observed": float(delta_obs),
        "p_value_two_sided": p_val,
        "p_value_one_sided": p_one,
        "null_ci_95_low": ci_low,
        "null_ci_95_high": ci_high,
        "n_permutations": n_perm,
        "significant_alpha_005": p_val < 0.05,
        "permutation_delta_mean": float(delta_perm.mean()),
        "permutation_delta_std": float(delta_perm.std()),
    }

def main():
    ensure_dir("results/reviewer_fixes")
    
    # 1. PCA Fused experiment
    print("\n>>> PCA Fused Experiment <<<")
    K_fqk_pca   = np.load("results/geometric_difference/fused/K_fqk_train.npy")
    K_rbf_pca   = np.load("results/geometric_difference/fused/K_rbf_train.npy")
    K_cmfqk_pca = np.load("results/geometric_difference/fused/K_cm_fqk_train.npy")
    K_pqk_pca   = np.load("results/geometric_difference/fused/K_pqk_train.npy")
    y_pca       = np.load("data/processed/subsample_2000.npz")['y_train']
    
    results_pca_fqk = run_permutation_test(K_fqk_pca, K_rbf_pca, y_pca, "FQK_PCA", "RBF_PCA")
    results_pca_cmfqk = run_permutation_test(K_cmfqk_pca, K_rbf_pca, y_pca, "CM_FQK_PCA", "RBF_PCA")
    results_pca_pqk = run_permutation_test(K_pqk_pca, K_rbf_pca, y_pca, "PQK_PCA", "RBF_PCA")
    
    # 2. Physics experiment
    print("\n>>> Physics Experiment <<<")
    K_fqk_phys    = np.load("results/physics/fused/K_fqk_physics_train.npy")
    K_agpqk_phys  = np.load("results/physics/fused/K_agpqk_physics_train.npy")
    y_phys        = np.load("data/processed/physics_features_16.npz")['y_train']
    
    rbf_cv_path = "results/physics/fused/K_rbf_cv_physics_train.npy"
    if not os.path.exists(rbf_cv_path):
        print("  Computing RBF_CV physics kernel...")
        gamma_cv = load_bandwidth_gamma("physics")
        X_phys = np.load("data/processed/physics_features_16.npz")["X_train"]
        K_rbf_cv_phys = compute_rbf_kernel(X_phys, gamma=gamma_cv)
        np.save(rbf_cv_path, K_rbf_cv_phys)
    else:
        K_rbf_cv_phys = np.load(rbf_cv_path)
        
    results_phys_fqk = run_permutation_test(K_fqk_phys, K_rbf_cv_phys, y_phys, "FQK_Phys", "RBF_CV_Phys")
    results_phys_agpqk = run_permutation_test(K_agpqk_phys, K_rbf_cv_phys, y_phys, "AGPQK_Phys", "RBF_CV_Phys")
    
    # Save all results
    all_results = {
        "pca_fqk_vs_rbf": results_pca_fqk,
        "pca_cmfqk_vs_rbf": results_pca_cmfqk,
        "pca_pqk_vs_rbf": results_pca_pqk,
        "phys_fqk_vs_rbf_cv": results_phys_fqk,
        "phys_agpqk_vs_rbf_cv": results_phys_agpqk
    }
    
    with open("results/reviewer_fixes/permutation_test_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    print("\nSaved: results/reviewer_fixes/permutation_test_results.json")

if __name__ == "__main__":
    main()
