"""
Direct Kernel Classification — No SVM.

Uses the quantum kernel matrix DIRECTLY as the classifier, bypassing the SVM.
The kernel IS the classifier: classify by kernel similarity, not by learned hyperplanes.

Methods:
  1. Kernel k-NN: majority vote among k most similar training samples
  2. Kernel Density (Parzen window): weighted sum of kernel similarities per class
  3. Class-Weighted Kernel Density: accounts for class imbalance (Rule I2)
  4. Kernel Combination Density: α·K_quantum + (1-α)·K_classical

All use pre-computed kernel matrices from Exp1. Zero quantum simulation needed.
Expected runtime: < 30 seconds.

Output: results/diagnostics/kernel_direct/direct_classify_results.json
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report
from collections import Counter

import config
from src.data_loader import load_subsample

# ---- Paths ----
FUSED_DIR = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
TFK_DIR   = os.path.join(FUSED_DIR, "tfk")
OUT_DIR   = os.path.join(config.RESULTS_DIR, "diagnostics", "kernel_direct")
os.makedirs(OUT_DIR, exist_ok=True)


# ============ DIRECT KERNEL CLASSIFIERS ============

def kernel_knn(K_test, y_train, k=5):
    """
    Kernel k-NN: for each test sample, find k training samples with
    highest kernel similarity, majority vote.

    K_test[i, j] = K(x_test_i, x_train_j)
    Higher kernel value = more similar.
    """
    n_test = K_test.shape[0]
    y_pred = np.zeros(n_test, dtype=y_train.dtype)

    for i in range(n_test):
        # Find k training samples most similar to test sample i
        top_k_idx = np.argsort(K_test[i])[-k:]  # highest K values
        top_k_labels = y_train[top_k_idx]
        # Majority vote
        counter = Counter(top_k_labels)
        y_pred[i] = counter.most_common(1)[0][0]

    return y_pred


def kernel_weighted_knn(K_test, y_train, k=5):
    """
    Kernel weighted k-NN: same as k-NN but votes are weighted by
    kernel similarity (closer neighbors count more).
    """
    n_test = K_test.shape[0]
    classes = np.unique(y_train)
    y_pred = np.zeros(n_test, dtype=y_train.dtype)

    for i in range(n_test):
        top_k_idx = np.argsort(K_test[i])[-k:]
        top_k_labels = y_train[top_k_idx]
        top_k_weights = K_test[i, top_k_idx]

        # Weighted vote per class
        class_scores = {}
        for c in classes:
            mask = top_k_labels == c
            class_scores[c] = np.sum(top_k_weights[mask])

        y_pred[i] = max(class_scores, key=class_scores.get)

    return y_pred


def kernel_density(K_test, y_train):
    """
    Kernel Density Classification (Parzen window):
      score(x, class c) = (1/n_c) * Σ_{i: y_i=c} K(x, x_i)
      predict = argmax_c score(x, c)

    The kernel directly determines the class — no training, no hyperplane.
    Normalization by n_c handles class size differences.
    """
    classes = np.unique(y_train)
    n_test = K_test.shape[0]
    scores = np.zeros((n_test, len(classes)))

    for idx, c in enumerate(classes):
        mask = y_train == c
        n_c = np.sum(mask)
        # Average kernel similarity to all training samples of class c
        scores[:, idx] = np.mean(K_test[:, mask], axis=1)

    y_pred = classes[np.argmax(scores, axis=1)]
    return y_pred


def kernel_density_weighted(K_test, y_train):
    """
    Class-Weighted Kernel Density: weight each class inversely by frequency.
    Equivalent to class_weight='balanced' for SVMs (Rule I2).

      weight_c = N_total / (n_classes * n_c)
      score(x, c) = weight_c * mean(K(x, x_i) for i in class c)
    """
    classes = np.unique(y_train)
    n_test = K_test.shape[0]
    n_total = len(y_train)
    n_classes = len(classes)
    scores = np.zeros((n_test, len(classes)))

    for idx, c in enumerate(classes):
        mask = y_train == c
        n_c = np.sum(mask)
        weight = n_total / (n_classes * n_c)
        scores[:, idx] = weight * np.mean(K_test[:, mask], axis=1)

    y_pred = classes[np.argmax(scores, axis=1)]
    return y_pred


def kernel_combination_density(K_test_q, K_test_c, y_train, alpha):
    """
    Kernel Combination Density:
      K_combined = α * K_quantum + (1-α) * K_classical
    Then apply class-weighted kernel density on the combined kernel.
    """
    K_test_combined = alpha * K_test_q + (1 - alpha) * K_test_c
    return kernel_density_weighted(K_test_combined, y_train)


# ============ EVALUATION ============

def evaluate(y_test, y_pred, name):
    """Compute macro-F1, accuracy, and per-class F1."""
    macro_f1 = float(f1_score(y_test, y_pred, average='macro'))
    accuracy = float(accuracy_score(y_test, y_pred))
    weighted_f1 = float(f1_score(y_test, y_pred, average='weighted'))
    return {
        'macro_f1': macro_f1,
        'accuracy': accuracy,
        'weighted_f1': weighted_f1,
        'model': name,
    }


# ============ MAIN ============

def main():
    print("=" * 70)
    print("DIRECT KERNEL CLASSIFICATION — NO SVM")
    print("The kernel IS the classifier")
    print("=" * 70)

    # ---- Load labels ----
    print("\nLoading data...")
    subsample = load_subsample()
    y_train = subsample["y_train"]
    y_test  = subsample["y_test"]
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}, "
          f"Classes: {len(np.unique(y_train))}")

    class_counts = Counter(y_train)
    print(f"  Min class: {min(class_counts.values())} samples, "
          f"Max class: {max(class_counts.values())} samples")

    # ---- Load kernels ----
    print("\nLoading pre-computed kernels...")
    kernels = {}
    kernel_files = {
        'FQK':    ('K_fqk_train.npy',    'K_fqk_test.npy',    FUSED_DIR),
        'PQK':    ('K_pqk_train.npy',    'K_pqk_test.npy',    FUSED_DIR),
        'TFK':    ('K_tfk_train.npy',    'K_tfk_test.npy',    TFK_DIR),
        'CM_FQK': ('K_cm_fqk_train.npy', 'K_cm_fqk_test.npy', FUSED_DIR),
        'RBF':    ('K_rbf_train.npy',    None,                 FUSED_DIR),
    }

    # For RBF, we need the test kernel too — check if it exists
    rbf_test_path = os.path.join(FUSED_DIR, "K_rbf_test.npy")
    if not os.path.exists(rbf_test_path):
        # Compute RBF test kernel from raw features
        print("  Computing RBF test kernel from features...")
        X_train = subsample["fused_X_train"]
        X_test  = subsample["fused_X_test"]
        from sklearn.metrics.pairwise import rbf_kernel
        # Use same gamma as sklearn default ('scale'): 1 / (n_features * X.var())
        gamma = 1.0 / (X_train.shape[1] * X_train.var())
        K_rbf_test = rbf_kernel(X_test, X_train, gamma=gamma)
        K_rbf_train = np.load(os.path.join(FUSED_DIR, "K_rbf_train.npy"))
        kernels['RBF'] = (K_rbf_train, K_rbf_test)
        print(f"  RBF: train={K_rbf_train.shape}, test={K_rbf_test.shape}, gamma={gamma:.6f}")
    else:
        K_rbf_train = np.load(os.path.join(FUSED_DIR, "K_rbf_train.npy"))
        K_rbf_test = np.load(rbf_test_path)
        kernels['RBF'] = (K_rbf_train, K_rbf_test)

    for name, (train_file, test_file, base_dir) in kernel_files.items():
        if name == 'RBF':
            continue  # already handled
        train_path = os.path.join(base_dir, train_file)
        test_path  = os.path.join(base_dir, test_file)
        if os.path.exists(train_path) and os.path.exists(test_path):
            K_tr = np.load(train_path)
            K_te = np.load(test_path)
            kernels[name] = (K_tr, K_te)
            print(f"  {name}: train={K_tr.shape}, test={K_te.shape}")
        else:
            print(f"  {name}: MISSING — skipping")

    results = {}

    # ================================================================
    # TEST 1: Kernel k-NN at various k
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Kernel k-NN (majority vote among k highest K values)")
    print("=" * 70)

    k_values = [1, 3, 5, 7, 11, 21, 51]

    print(f"\n{'Kernel':10s}", end="")
    for k in k_values:
        print(f"  k={k:<4d}", end="")
    print()
    print("-" * (10 + 8 * len(k_values)))

    results['knn'] = {}
    for name, (K_tr, K_te) in kernels.items():
        results['knn'][name] = {}
        print(f"{name:10s}", end="")
        for k in k_values:
            y_pred = kernel_knn(K_te, y_train, k=k)
            r = evaluate(y_test, y_pred, f"{name}-kNN(k={k})")
            results['knn'][name][f'k={k}'] = r
            print(f"  {r['macro_f1']:.4f}", end="")
        print()

    # ================================================================
    # TEST 2: Kernel Weighted k-NN
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Kernel Weighted k-NN (similarity-weighted votes)")
    print("=" * 70)

    print(f"\n{'Kernel':10s}", end="")
    for k in k_values:
        print(f"  k={k:<4d}", end="")
    print()
    print("-" * (10 + 8 * len(k_values)))

    results['weighted_knn'] = {}
    for name, (K_tr, K_te) in kernels.items():
        results['weighted_knn'][name] = {}
        print(f"{name:10s}", end="")
        for k in k_values:
            y_pred = kernel_weighted_knn(K_te, y_train, k=k)
            r = evaluate(y_test, y_pred, f"{name}-WkNN(k={k})")
            results['weighted_knn'][name][f'k={k}'] = r
            print(f"  {r['macro_f1']:.4f}", end="")
        print()

    # ================================================================
    # TEST 3: Kernel Density Classification
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Kernel Density Classification (Parzen window)")
    print("=" * 70)

    print(f"\n{'Kernel':10s}  {'Density':>10s}  {'Weighted':>10s}")
    print("-" * 35)

    results['density'] = {}
    for name, (K_tr, K_te) in kernels.items():
        y_pred_d = kernel_density(K_te, y_train)
        y_pred_w = kernel_density_weighted(K_te, y_train)
        r_d = evaluate(y_test, y_pred_d, f"{name}-Density")
        r_w = evaluate(y_test, y_pred_w, f"{name}-WeightedDensity")
        results['density'][name] = {
            'unweighted': r_d,
            'weighted': r_w,
        }
        print(f"{name:10s}  {r_d['macro_f1']:>10.4f}  {r_w['macro_f1']:>10.4f}")

    # ================================================================
    # TEST 4: Kernel Combination Density (α·FQK + (1-α)·RBF)
    # ================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Kernel Combination Density (α·K_quantum + (1-α)·K_RBF)")
    print("=" * 70)

    if 'RBF' in kernels:
        _, K_rbf_te = kernels['RBF']
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        quantum_kernels = {k: v for k, v in kernels.items() if k != 'RBF'}

        results['combination'] = {}
        for q_name, (K_tr_q, K_te_q) in quantum_kernels.items():
            results['combination'][q_name] = {}
            print(f"\n  {q_name} + RBF:")
            print(f"  {'α':>6s}  {'Macro-F1':>10s}  {'Accuracy':>10s}")
            print("  " + "-" * 30)

            best_f1 = 0
            best_alpha = 0
            for alpha in alphas:
                y_pred = kernel_combination_density(
                    K_te_q, K_rbf_te, y_train, alpha)
                r = evaluate(y_test, y_pred,
                           f"{q_name}+RBF(α={alpha})")
                results['combination'][q_name][f'alpha={alpha}'] = r
                print(f"  {alpha:>6.1f}  {r['macro_f1']:>10.4f}  {r['accuracy']:>10.4f}")

                if r['macro_f1'] > best_f1:
                    best_f1 = r['macro_f1']
                    best_alpha = alpha

            results['combination'][q_name]['best_alpha'] = best_alpha
            results['combination'][q_name]['best_f1'] = best_f1
            print(f"  BEST: α={best_alpha}, F1={best_f1:.4f}")
    else:
        print("  RBF test kernel not available. Skipping.")

    # ================================================================
    # FINAL COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("FINAL COMPARISON: Best Direct Method vs SVM Baseline")
    print("=" * 70)

    # Find best result across all methods
    all_scores = []

    for name in kernels:
        # Best k-NN
        for k_str, r in results['knn'].get(name, {}).items():
            all_scores.append((r['macro_f1'], f"{name} kNN({k_str})", r))
        # Best weighted k-NN
        for k_str, r in results['weighted_knn'].get(name, {}).items():
            all_scores.append((r['macro_f1'], f"{name} WkNN({k_str})", r))
        # Density
        if name in results['density']:
            r = results['density'][name]['weighted']
            all_scores.append((r['macro_f1'], f"{name} WeightedDensity", r))

    # Combination
    for q_name in results.get('combination', {}):
        for k, v in results['combination'][q_name].items():
            if isinstance(v, dict) and 'macro_f1' in v:
                all_scores.append((v['macro_f1'], f"{q_name}+RBF({k})", v))

    all_scores.sort(reverse=True)

    print(f"\n  Top 10 results:")
    print(f"  {'Rank':>4s}  {'Method':40s}  {'Macro-F1':>10s}  {'Accuracy':>10s}")
    print("  " + "-" * 68)

    for i, (f1, method, r) in enumerate(all_scores[:10]):
        print(f"  {i+1:>4d}  {method:40s}  {f1:>10.4f}  {r['accuracy']:>10.4f}")

    print(f"\n  SVM Baselines (for comparison):")
    print(f"    FQK-SVM (C=1000)         : 0.1896 macro-F1")
    print(f"    CM_FQK-SVM (C=100)       : 0.1858 macro-F1")
    print(f"    RBF-SVM (GridSearchCV)   : 0.1930 macro-F1")
    print(f"    RF (500 trees, balanced) : 0.2079 macro-F1")

    # ---- Save ----
    results_path = os.path.join(OUT_DIR, "direct_classify_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
