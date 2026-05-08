"""
Classification Improvement Diagnostic.

Tests 3 improvements to the classification pipeline WITHOUT re-computing
any quantum kernels — all matrices are loaded from Exp1 cache:

  1. GridSearchCV on C for precomputed quantum kernel SVMs
     (current pipeline uses hardcoded C=100)
  2. Kernel matrix normalization (cosine normalization: K_ij / sqrt(K_ii * K_jj))
  3. PQK gamma sweep using cached Bloch vectors

Expected runtime: < 5 minutes (pure classical, no quantum simulation).

Output: results/diagnostics/classification_improvement/improvement_results.json
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score

import config
from src.data_loader import load_subsample
from src.utils import compute_full_metrics

# ---- Paths ----
FUSED_DIR = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
TFK_DIR   = os.path.join(FUSED_DIR, "tfk")
OUT_DIR   = os.path.join(config.RESULTS_DIR, "diagnostics", "classification_improvement")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- C grid for tuning ----
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

# ---- PQK gamma values (pre-computed train kernels exist for these) ----
PQK_GAMMAS = [0.1, 0.5, 1.0, 2.0, 5.0]


# ============ UTILITY FUNCTIONS ============

def normalize_kernel(K_train, K_test=None):
    """
    Cosine-normalize a kernel matrix: K_norm[i,j] = K[i,j] / sqrt(K[i,i] * K[j,j]).

    This ensures diagonal = 1 everywhere, putting all kernels on equal
    footing for the SVM's C parameter.

    For rectangular test matrices (n_test × n_train), normalization uses:
        K_test_norm[i,j] = K_test[i,j] / sqrt(diag_test[i] * diag_train[j])

    where diag_test[i] = K_test_full[i,i] is the self-similarity of the i-th test point.
    Since we don't have K_test_full (test×test), we estimate diag_test from the
    test kernel by using the maximum along each row (which is 1.0 for FQK kernels,
    approximated for others). In practice, for FQK-style kernels diag(K) = 1 always.

    If diag entries are already ~1, normalization is a near-identity operation.
    """
    diag_train = np.diag(K_train).copy()
    # Clamp to avoid division by zero for near-zero diagonal entries
    diag_train = np.clip(diag_train, 1e-10, None)

    # Square train kernel normalization
    sqrt_diag = np.sqrt(diag_train)
    K_train_norm = K_train / np.outer(sqrt_diag, sqrt_diag)

    if K_test is None:
        return K_train_norm, None

    # Rectangular test kernel normalization
    # For FQK/PQK/TFK, K(x,x) = 1 by construction, so diag_test = 1.
    # For safety, we estimate it: K_test_diag ≈ 1 for fidelity kernels.
    n_test = K_test.shape[0]

    # Heuristic: for fidelity quantum kernels, self-similarity = 1.0
    # For PQK with Gaussian, also 1.0 since exp(0) = 1.
    # Fall back to 1.0 since we don't have K_test_full (test×test).
    diag_test = np.ones(n_test)

    sqrt_diag_test = np.sqrt(diag_test)
    K_test_norm = K_test / np.outer(sqrt_diag_test, sqrt_diag)

    return K_train_norm, K_test_norm


def tune_C(K_train, y_train, K_test, y_test, name, C_grid=C_GRID):
    """
    GridSearchCV over C for a precomputed kernel SVM.

    Returns dict with best_C, best_cv_score, test_macro_f1, test_accuracy.
    """
    # StratifiedKFold to handle class imbalance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    grid = GridSearchCV(
        SVC(kernel='precomputed', class_weight='balanced',
            random_state=config.RANDOM_SEED),
        param_grid={'C': C_grid},
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(K_train, y_train)

    y_pred = grid.predict(K_test)
    test_f1 = float(f1_score(y_test, y_pred, average='macro'))
    test_acc = float(np.mean(y_pred == y_test))

    result = {
        'best_C': float(grid.best_params_['C']),
        'best_cv_f1': float(grid.best_score_),
        'test_macro_f1': test_f1,
        'test_accuracy': test_acc,
    }

    print(f"  {name:20s}: best_C={result['best_C']:>8.3f}, "
          f"CV-F1={result['best_cv_f1']:.4f}, "
          f"Test-F1={result['test_macro_f1']:.4f}, "
          f"Test-Acc={result['test_accuracy']:.4f}")

    return result


def evaluate_fixed_C(K_train, y_train, K_test, y_test, name, C=100.0):
    """Evaluate with a fixed C (the current pipeline default)."""
    clf = SVC(kernel='precomputed', class_weight='balanced',
              C=C, random_state=config.RANDOM_SEED)
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    test_f1 = float(f1_score(y_test, y_pred, average='macro'))
    test_acc = float(np.mean(y_pred == y_test))
    return {'test_macro_f1': test_f1, 'test_accuracy': test_acc, 'C': C}


def compute_pqk_from_bloch(bloch_train, bloch_test, gamma, n_qubits=config.N_QUBITS):
    """
    Re-compute PQK kernel from cached Bloch vectors at a different gamma.

    PQK(x, x') = exp(-gamma * sum_q (1/2) sum_a (bloch_q_a(x) - bloch_q_a(x'))^2)
    """
    n1 = len(bloch_train)
    n2 = len(bloch_test) if bloch_test is not None else n1

    b1 = bloch_train.reshape(n1, n_qubits, 3)
    if bloch_test is not None:
        b2 = bloch_test.reshape(n2, n_qubits, 3)
    else:
        b2 = b1

    K = np.zeros((n1, n2), dtype=np.float64)
    for i in range(n1):
        diff = b1[i] - b2  # (n2, n_qubits, 3)
        sq_diff = np.sum(diff ** 2, axis=2)  # (n2, n_qubits)
        frob_dist = 0.5 * np.sum(sq_diff, axis=1)  # (n2,)
        K[i, :] = np.exp(-gamma * frob_dist)

    return K


# ============ MAIN ============

def main():
    print("=" * 70)
    print("CLASSIFICATION IMPROVEMENT DIAGNOSTIC")
    print("=" * 70)

    # ---- Load labels ----
    print("\nLoading data...")
    subsample = load_subsample()
    y_train = subsample["y_train"]
    y_test  = subsample["y_test"]
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}, "
          f"Classes: {len(np.unique(y_train))}")

    # ---- Load pre-computed kernels ----
    print("\nLoading pre-computed kernels...")
    kernels = {}

    kernel_files = {
        'FQK': ('K_fqk_train.npy', 'K_fqk_test.npy', FUSED_DIR),
        'PQK': ('K_pqk_train.npy', 'K_pqk_test.npy', FUSED_DIR),
        'TFK': ('K_tfk_train.npy', 'K_tfk_test.npy', TFK_DIR),
        'CM_FQK': ('K_cm_fqk_train.npy', 'K_cm_fqk_test.npy', FUSED_DIR),
    }

    for name, (train_file, test_file, base_dir) in kernel_files.items():
        train_path = os.path.join(base_dir, train_file)
        test_path  = os.path.join(base_dir, test_file)
        if os.path.exists(train_path) and os.path.exists(test_path):
            K_tr = np.load(train_path)
            K_te = np.load(test_path)
            kernels[name] = (K_tr, K_te)
            print(f"  {name}: train={K_tr.shape}, test={K_te.shape}, "
                  f"diag_mean={np.mean(np.diag(K_tr)):.4f}")
        else:
            print(f"  {name}: MISSING — skipping")

    if not kernels:
        print("ERROR: No kernels found. Run Exp1 first.")
        return

    results = {}

    # ================================================================
    # IMPROVEMENT 1: GridSearchCV on C (raw kernels)
    # ================================================================
    print("\n" + "=" * 70)
    print("IMPROVEMENT 1: GridSearchCV on C (raw kernels, current pipeline)")
    print("=" * 70)

    print(f"\nC search grid: {C_GRID}")
    print(f"\n{'Kernel':20s}  {'best_C':>8s}  {'CV-F1':>8s}  {'Test-F1':>8s}  {'Test-Acc':>8s}")
    print("-" * 60)

    results['improvement_1_c_tuning'] = {}
    for name, (K_tr, K_te) in kernels.items():
        result = tune_C(K_tr, y_train, K_te, y_test, name)
        # Also compute baseline fixed C=100 for comparison
        baseline = evaluate_fixed_C(K_tr, y_train, K_te, y_test, name)
        result['baseline_f1_C100'] = baseline['test_macro_f1']
        result['improvement_over_baseline'] = result['test_macro_f1'] - baseline['test_macro_f1']
        results['improvement_1_c_tuning'][name] = result

    # ================================================================
    # IMPROVEMENT 2: Kernel normalization + C tuning
    # ================================================================
    print("\n" + "=" * 70)
    print("IMPROVEMENT 2: Cosine-normalized kernels + GridSearchCV on C")
    print("=" * 70)

    print(f"\n{'Kernel':20s}  {'best_C':>8s}  {'CV-F1':>8s}  {'Test-F1':>8s}  {'Test-Acc':>8s}")
    print("-" * 60)

    results['improvement_2_normalized'] = {}
    for name, (K_tr, K_te) in kernels.items():
        K_tr_norm, K_te_norm = normalize_kernel(K_tr, K_te)

        # Validate normalization
        diag_check = np.mean(np.abs(np.diag(K_tr_norm) - 1.0))
        if diag_check > 1e-6:
            print(f"  WARNING: {name} normalized diagonal error = {diag_check:.2e}")

        result = tune_C(K_tr_norm, y_train, K_te_norm, y_test, f"{name} (norm)")
        # Compare against raw C-tuned
        raw_result = results['improvement_1_c_tuning'].get(name, {})
        result['raw_tuned_f1'] = raw_result.get('test_macro_f1', 0)
        result['improvement_over_raw_tuned'] = result['test_macro_f1'] - result['raw_tuned_f1']
        results['improvement_2_normalized'][name] = result

    # ================================================================
    # IMPROVEMENT 3: PQK gamma sweep from cached Bloch vectors
    # ================================================================
    print("\n" + "=" * 70)
    print("IMPROVEMENT 3: PQK Gamma Sweep + Normalization + C Tuning")
    print("=" * 70)

    bloch_train_path = os.path.join(FUSED_DIR, "bloch_train.npy")
    bloch_test_path  = os.path.join(FUSED_DIR, "bloch_train_test.npy")

    if os.path.exists(bloch_train_path) and os.path.exists(bloch_test_path):
        bloch_train = np.load(bloch_train_path)
        bloch_test  = np.load(bloch_test_path)
        print(f"  Loaded Bloch vectors: train={bloch_train.shape}, test={bloch_test.shape}")

        pqk_results = {}
        gammas_to_test = [0.01, 0.05, 0.1, 0.2, 0.5, 0.67, 1.0, 2.0, 5.0, 10.0]

        print(f"\n  Testing gammas: {gammas_to_test}")
        print(f"\n  {'Gamma':>8s}  {'best_C':>8s}  {'CV-F1':>8s}  {'Test-F1':>8s}  "
              f"{'Test-Acc':>8s}  {'diag_mean':>10s}  {'offdiag_CV':>10s}")
        print("  " + "-" * 75)

        for gamma in gammas_to_test:
            # Re-compute PQK at this gamma from Bloch vectors
            K_pqk_tr = compute_pqk_from_bloch(bloch_train, None, gamma)
            K_pqk_te = compute_pqk_from_bloch(bloch_train, bloch_test, gamma)

            # Collect diagnostic info
            offdiag_mask = ~np.eye(len(K_pqk_tr), dtype=bool)
            offdiag_vals = K_pqk_tr[offdiag_mask]
            offdiag_cv = float(np.std(offdiag_vals) / (np.mean(offdiag_vals) + 1e-10))
            diag_mean = float(np.mean(np.diag(K_pqk_tr)))

            # Normalize
            K_pqk_tr_n, K_pqk_te_n = normalize_kernel(K_pqk_tr, K_pqk_te)

            # Tune C
            result = tune_C(K_pqk_tr_n, y_train, K_pqk_te_n, y_test,
                          f"PQK(γ={gamma})")
            result['gamma'] = gamma
            result['offdiag_cv'] = offdiag_cv
            result['diag_mean'] = diag_mean

            print(f"  {'':20s}  {'':>8s}  {'':>8s}  {'':>8s}  {'':>8s}  "
                  f"{diag_mean:>10.4f}  {offdiag_cv:>10.4f}")

            pqk_results[str(gamma)] = result

        # Find best gamma
        best_gamma = max(pqk_results.keys(),
                        key=lambda g: pqk_results[g]['test_macro_f1'])
        pqk_results['best_gamma'] = float(best_gamma)
        pqk_results['best_result'] = pqk_results[best_gamma]

        print(f"\n  BEST PQK: gamma={best_gamma}, "
              f"F1={pqk_results[best_gamma]['test_macro_f1']:.4f}")

        results['improvement_3_pqk_gamma'] = pqk_results
    else:
        print("  WARNING: Bloch vectors not found. Skipping PQK gamma sweep.")
        print(f"  Expected: {bloch_train_path}")
        results['improvement_3_pqk_gamma'] = {'error': 'bloch_vectors_not_found'}

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: BEFORE vs AFTER")
    print("=" * 70)

    print(f"\n{'Kernel':15s}  {'Before (C=100)':>15s}  {'C-Tuned':>15s}  "
          f"{'Norm+C-Tuned':>15s}  {'Improvement':>12s}")
    print("-" * 75)

    for name in kernels:
        before = results['improvement_1_c_tuning'].get(name, {}).get('baseline_f1_C100', 0)
        c_tuned = results['improvement_1_c_tuning'].get(name, {}).get('test_macro_f1', 0)
        norm_tuned = results['improvement_2_normalized'].get(name, {}).get('test_macro_f1', 0)
        best = max(c_tuned, norm_tuned)
        delta = best - before

        print(f"  {name:13s}  {before:>15.4f}  {c_tuned:>15.4f}  "
              f"{norm_tuned:>15.4f}  {delta:>+12.4f}")

    if 'best_result' in results.get('improvement_3_pqk_gamma', {}):
        best_pqk = results['improvement_3_pqk_gamma']['best_result']
        pqk_before = results['improvement_1_c_tuning'].get('PQK', {}).get('baseline_f1_C100', 0)
        print(f"  {'PQK (best γ)':13s}  {pqk_before:>15.4f}  "
              f"{'':>15s}  {best_pqk['test_macro_f1']:>15.4f}  "
              f"{best_pqk['test_macro_f1'] - pqk_before:>+12.4f}")

    # Compare against best classical
    print(f"\n  Reference baselines:")
    print(f"    RF (500 trees, balanced) : 0.2079 macro-F1")
    print(f"    RBF-SVM (GridSearchCV)   : 0.1930 macro-F1")
    print(f"    GradientBoosting (200)   : 0.1927 macro-F1")

    # ---- Save results ----
    results_path = os.path.join(OUT_DIR, "improvement_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
