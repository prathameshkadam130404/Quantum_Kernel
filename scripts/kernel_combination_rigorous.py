"""
Rigorous Kernel Combination — Nested CV + Bootstrap CI.

Fixes two methodological issues from the initial kernel combination script:

1. DATA LEAKAGE FIX: α was selected by looking at test F1 (mild overfitting).
   Now uses NESTED CV: outer 5-fold selects (α, C) jointly via inner 3-fold CV.
   Test set is NEVER seen during hyperparameter selection.

2. STATISTICAL SIGNIFICANCE: Bootstrap confidence intervals on test F1
   to determine whether TRIPLE vs RF margin is real or noise.

Output: results/diagnostics/kernel_combination/rigorous_results.json
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from itertools import product

import config
from src.data_loader import load_subsample

# ---- Paths ----
FUSED_DIR = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
TFK_DIR   = os.path.join(FUSED_DIR, "tfk")
OUT_DIR   = os.path.join(config.RESULTS_DIR, "diagnostics", "kernel_combination")
os.makedirs(OUT_DIR, exist_ok=True)

N_BOOTSTRAP = 1000
RANDOM_SEED = config.RANDOM_SEED


def bootstrap_f1_ci(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED,
                    confidence=0.95):
    """
    Bootstrap confidence interval on macro-F1.
    Resamples (y_true, y_pred) pairs with replacement.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    f1_scores = []

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_pred[idx]
        # Skip degenerate resamples
        if len(np.unique(y_t)) < 2:
            continue
        f1_scores.append(f1_score(y_t, y_p, average='macro'))

    f1_scores = np.array(f1_scores)
    alpha = (1 - confidence) / 2
    lo = np.percentile(f1_scores, 100 * alpha)
    hi = np.percentile(f1_scores, 100 * (1 - alpha))
    mean = np.mean(f1_scores)
    std = np.std(f1_scores)

    return {
        'mean': float(mean),
        'std': float(std),
        'ci_lo': float(lo),
        'ci_hi': float(hi),
        'n_bootstrap': len(f1_scores),
    }


def paired_bootstrap_test(y_true, y_pred_a, y_pred_b,
                          n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED):
    """
    Paired bootstrap test: is model A significantly better than model B?
    Returns p-value (proportion of bootstraps where B ≥ A).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    a_wins = 0

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        if len(np.unique(y_t)) < 2:
            continue
        f1_a = f1_score(y_t, y_pred_a[idx], average='macro')
        f1_b = f1_score(y_t, y_pred_b[idx], average='macro')
        if f1_a > f1_b:
            a_wins += 1

    p_value = 1.0 - (a_wins / n_bootstrap)
    return float(p_value)


def nested_cv_kernel_combination(K_q_train, K_rbf_train, y_train,
                                  alphas, C_values, n_outer=5, n_inner=3):
    """
    Nested CV for kernel combination.

    Outer loop: 5-fold stratified CV for unbiased performance estimate.
    Inner loop: 3-fold CV within each outer train fold to select best (α, C).
    Test set is NEVER seen during selection.

    Returns: y_pred for all training samples (via cross_val_predict style),
             best (α, C) per fold, and mean CV-F1.
    """
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True,
                               random_state=RANDOM_SEED)
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True,
                               random_state=RANDOM_SEED + 1)

    y_pred_all = np.full_like(y_train, -1)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(
            np.zeros(len(y_train)), y_train)):

        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]

        best_inner_f1 = -1
        best_alpha = None
        best_C = None

        # Inner CV: try all (α, C) combinations
        for alpha in alphas:
            # Build combined kernel for this fold's training data
            K_comb_fold = (alpha * K_q_train[np.ix_(train_idx, train_idx)] +
                          (1 - alpha) * K_rbf_train[np.ix_(train_idx, train_idx)])
            K_comb_fold = (K_comb_fold + K_comb_fold.T) / 2

            for C in C_values:
                # Inner CV on the outer training fold
                inner_f1s = []
                for in_train, in_val in inner_cv.split(np.zeros(len(y_tr)), y_tr):
                    K_in_train = K_comb_fold[np.ix_(in_train, in_train)]
                    K_in_val = K_comb_fold[np.ix_(in_val, in_train)]

                    clf = SVC(kernel='precomputed', class_weight='balanced',
                             C=C, random_state=RANDOM_SEED)
                    clf.fit(K_in_train, y_tr[in_train])
                    y_p = clf.predict(K_in_val)
                    inner_f1s.append(f1_score(y_tr[in_val], y_p, average='macro'))

                mean_inner_f1 = np.mean(inner_f1s)
                if mean_inner_f1 > best_inner_f1:
                    best_inner_f1 = mean_inner_f1
                    best_alpha = alpha
                    best_C = C

        # Train on full outer train fold with best (α, C), predict outer val fold
        K_comb_train = (best_alpha * K_q_train[np.ix_(train_idx, train_idx)] +
                       (1 - best_alpha) * K_rbf_train[np.ix_(train_idx, train_idx)])
        K_comb_train = (K_comb_train + K_comb_train.T) / 2

        K_comb_val = (best_alpha * K_q_train[np.ix_(val_idx, train_idx)] +
                     (1 - best_alpha) * K_rbf_train[np.ix_(val_idx, train_idx)])

        clf = SVC(kernel='precomputed', class_weight='balanced',
                 C=best_C, random_state=RANDOM_SEED)
        clf.fit(K_comb_train, y_tr)
        y_pred_all[val_idx] = clf.predict(K_comb_val)

        fold_f1 = f1_score(y_val, y_pred_all[val_idx], average='macro')
        fold_results.append({
            'fold': fold_idx,
            'best_alpha': best_alpha,
            'best_C': best_C,
            'inner_cv_f1': float(best_inner_f1),
            'outer_val_f1': float(fold_f1),
        })

        print(f"    Fold {fold_idx+1}: α={best_alpha:.1f}, C={best_C:.0f}, "
              f"inner-F1={best_inner_f1:.4f}, outer-F1={fold_f1:.4f}")

    overall_f1 = f1_score(y_train, y_pred_all, average='macro')
    return y_pred_all, fold_results, overall_f1


def main():
    print("=" * 70)
    print("RIGOROUS KERNEL COMBINATION — NESTED CV + BOOTSTRAP CI")
    print("=" * 70)

    # ---- Load data ----
    print("\nLoading data...")
    subsample = load_subsample()
    y_train = subsample["y_train"]
    y_test  = subsample["y_test"]
    X_train = subsample["fused_X_train"]
    X_test  = subsample["fused_X_test"]
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")

    # ---- Load quantum kernels ----
    print("\nLoading quantum kernels...")
    K_fqk_train = np.load(os.path.join(FUSED_DIR, "K_fqk_train.npy"))
    K_fqk_test  = np.load(os.path.join(FUSED_DIR, "K_fqk_test.npy"))
    K_cm_train  = np.load(os.path.join(FUSED_DIR, "K_cm_fqk_train.npy"))
    K_cm_test   = np.load(os.path.join(FUSED_DIR, "K_cm_fqk_test.npy"))
    print("  FQK, CM_FQK loaded")

    # ---- Compute RBF kernels (consistent gamma) ----
    print("\nComputing RBF kernels...")
    gamma_scale = 1.0 / (X_train.shape[1] * X_train.var())
    # Use best gamma from previous experiment or scale default
    K_rbf_train = rbf_kernel(X_train, X_train, gamma=gamma_scale)
    K_rbf_test  = rbf_kernel(X_test, X_train, gamma=gamma_scale)
    K_rbf_train = (K_rbf_train + K_rbf_train.T) / 2
    print(f"  gamma={gamma_scale:.6f}")

    # ---- Hyperparameter grid ----
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    C_values = [1.0, 10.0, 100.0, 1000.0, 10000.0]

    results = {}

    # ================================================================
    # PART 1: NESTED CV (no data leakage)
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 1: NESTED CV — No Data Leakage")
    print("  Outer: 5-fold, Inner: 3-fold, selects (α, C) without test set")
    print("=" * 70)

    # --- FQK + RBF ---
    print("\n  FQK + RBF:")
    _, fqk_folds, fqk_cv_f1 = nested_cv_kernel_combination(
        K_fqk_train, K_rbf_train, y_train, alphas, C_values)
    print(f"  → Nested CV F1: {fqk_cv_f1:.4f}")
    results['nested_cv_fqk_rbf'] = {
        'folds': fqk_folds,
        'nested_cv_f1': float(fqk_cv_f1),
    }

    # --- CM_FQK + RBF ---
    print("\n  CM_FQK + RBF:")
    _, cm_folds, cm_cv_f1 = nested_cv_kernel_combination(
        K_cm_train, K_rbf_train, y_train, alphas, C_values)
    print(f"  → Nested CV F1: {cm_cv_f1:.4f}")
    results['nested_cv_cm_rbf'] = {
        'folds': cm_folds,
        'nested_cv_f1': float(cm_cv_f1),
    }

    # --- TRIPLE: use combined quantum kernel = 0.5*FQK + 0.5*CM_FQK ---
    # For triple, define K_quantum = β*FQK + (1-β)*CM_FQK, then combine with RBF
    print("\n  TRIPLE (FQK+CM_FQK+RBF) via nested CV:")
    # Use the consensus alpha split from previous results
    betas = [0.3, 0.5, 0.7]  # ratio of FQK within quantum component
    best_triple_cv = 0
    best_beta = 0.5

    for beta in betas:
        K_q_train = beta * K_fqk_train + (1 - beta) * K_cm_train
        print(f"\n    β={beta} (quantum = {beta}·FQK + {1-beta}·CM_FQK):")
        _, t_folds, t_cv_f1 = nested_cv_kernel_combination(
            K_q_train, K_rbf_train, y_train, alphas, C_values)
        print(f"    → Nested CV F1: {t_cv_f1:.4f}")
        if t_cv_f1 > best_triple_cv:
            best_triple_cv = t_cv_f1
            best_beta = beta
            best_triple_folds = t_folds

    results['nested_cv_triple'] = {
        'best_beta': best_beta,
        'folds': best_triple_folds,
        'nested_cv_f1': float(best_triple_cv),
    }
    print(f"\n  Best TRIPLE β={best_beta}, Nested CV F1: {best_triple_cv:.4f}")

    # ================================================================
    # PART 2: FINAL MODEL — train on all train data, test on held-out test
    # Use the most common (α, C) selected across nested CV folds
    # ================================================================
    print("\n" + "=" * 70)
    print("PART 2: FINAL TEST SET EVALUATION")
    print("  Using consensus (α, C) from nested CV folds")
    print("=" * 70)

    predictions = {}

    # --- FQK + RBF: use consensus α, C from folds ---
    fqk_alphas = [f['best_alpha'] for f in fqk_folds]
    fqk_Cs = [f['best_C'] for f in fqk_folds]
    # Use median as consensus (robust to outlier folds)
    consensus_alpha_fqk = float(np.median(fqk_alphas))
    consensus_C_fqk = float(np.median(fqk_Cs))

    K_comb_train = consensus_alpha_fqk * K_fqk_train + (1 - consensus_alpha_fqk) * K_rbf_train
    K_comb_test  = consensus_alpha_fqk * K_fqk_test  + (1 - consensus_alpha_fqk) * K_rbf_test
    K_comb_train = (K_comb_train + K_comb_train.T) / 2

    clf = SVC(kernel='precomputed', class_weight='balanced',
             C=consensus_C_fqk, random_state=RANDOM_SEED)
    clf.fit(K_comb_train, y_train)
    y_pred_fqk_comb = clf.predict(K_comb_test)
    f1_fqk_comb = f1_score(y_test, y_pred_fqk_comb, average='macro')
    print(f"\n  FQK+RBF: α={consensus_alpha_fqk:.1f}, C={consensus_C_fqk:.0f} "
          f"→ Test F1={f1_fqk_comb:.4f}")
    predictions['FQK+RBF'] = y_pred_fqk_comb

    # --- TRIPLE: consensus from best β ---
    K_q_train_best = best_beta * K_fqk_train + (1 - best_beta) * K_cm_train
    K_q_test_best  = best_beta * K_fqk_test  + (1 - best_beta) * K_cm_test

    triple_alphas = [f['best_alpha'] for f in best_triple_folds]
    triple_Cs = [f['best_C'] for f in best_triple_folds]
    consensus_alpha_triple = float(np.median(triple_alphas))
    consensus_C_triple = float(np.median(triple_Cs))

    K_triple_train = consensus_alpha_triple * K_q_train_best + (1 - consensus_alpha_triple) * K_rbf_train
    K_triple_test  = consensus_alpha_triple * K_q_test_best  + (1 - consensus_alpha_triple) * K_rbf_test
    K_triple_train = (K_triple_train + K_triple_train.T) / 2

    clf = SVC(kernel='precomputed', class_weight='balanced',
             C=consensus_C_triple, random_state=RANDOM_SEED)
    clf.fit(K_triple_train, y_train)
    y_pred_triple = clf.predict(K_triple_test)
    f1_triple = f1_score(y_test, y_pred_triple, average='macro')
    print(f"  TRIPLE: β={best_beta}, α={consensus_alpha_triple:.1f}, C={consensus_C_triple:.0f} "
          f"→ Test F1={f1_triple:.4f}")
    predictions['TRIPLE'] = y_pred_triple

    # --- RBF-SVM baseline ---
    clf_rbf = SVC(kernel='precomputed', class_weight='balanced',
                 C=1000.0, random_state=RANDOM_SEED)
    clf_rbf.fit(K_rbf_train, y_train)
    y_pred_rbf = clf_rbf.predict(K_rbf_test)
    f1_rbf = f1_score(y_test, y_pred_rbf, average='macro')
    print(f"  RBF-SVM (C=1000): Test F1={f1_rbf:.4f}")
    predictions['RBF-SVM'] = y_pred_rbf

    # --- FQK-SVM alone ---
    clf_fqk = SVC(kernel='precomputed', class_weight='balanced',
                 C=1000.0, random_state=RANDOM_SEED)
    clf_fqk.fit(K_fqk_train, y_train)
    y_pred_fqk = clf_fqk.predict(K_fqk_test)
    f1_fqk = f1_score(y_test, y_pred_fqk, average='macro')
    print(f"  FQK-SVM (C=1000): Test F1={f1_fqk:.4f}")
    predictions['FQK-SVM'] = y_pred_fqk

    # --- RF baseline ---
    rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    print(f"  RF (500 trees): Test F1={f1_rf:.4f}")
    predictions['RF'] = y_pred_rf

    # ================================================================
    # PART 3: BOOTSTRAP CONFIDENCE INTERVALS + PAIRED TESTS
    # ================================================================
    print("\n" + "=" * 70)
    print(f"PART 3: BOOTSTRAP CONFIDENCE INTERVALS ({N_BOOTSTRAP} resamples)")
    print("=" * 70)

    print(f"\n  {'Method':25s}  {'F1':>8s}  {'95% CI':>20s}  {'Std':>8s}")
    print("  " + "-" * 65)

    ci_results = {}
    for name, y_pred in predictions.items():
        ci = bootstrap_f1_ci(y_test, y_pred)
        ci_results[name] = ci
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"  {name:25s}  {f1:>8.4f}  [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]  "
              f"{ci['std']:>8.4f}")

    results['bootstrap_ci'] = ci_results

    # ---- Paired bootstrap tests ----
    print(f"\n  Paired Bootstrap Tests (p-value: probability that B ≥ A):")
    print(f"  {'A vs B':35s}  {'ΔF1':>8s}  {'p-value':>8s}  {'Significant?':>12s}")
    print("  " + "-" * 68)

    comparisons = [
        ('FQK+RBF', 'RBF-SVM', "FQK+RBF beats RBF-SVM?"),
        ('FQK+RBF', 'FQK-SVM', "FQK+RBF beats FQK-SVM?"),
        ('TRIPLE', 'RBF-SVM', "TRIPLE beats RBF-SVM?"),
        ('TRIPLE', 'RF', "TRIPLE beats RF?"),
        ('TRIPLE', 'FQK+RBF', "TRIPLE beats FQK+RBF?"),
    ]

    paired_results = {}
    for name_a, name_b, desc in comparisons:
        p = paired_bootstrap_test(y_test, predictions[name_a], predictions[name_b])
        f1_a = f1_score(y_test, predictions[name_a], average='macro')
        f1_b = f1_score(y_test, predictions[name_b], average='macro')
        delta = f1_a - f1_b
        sig = "YES (p<0.05)" if p < 0.05 else "NO"
        print(f"  {desc:35s}  {delta:>+8.4f}  {p:>8.4f}  {sig:>12s}")
        paired_results[f"{name_a}_vs_{name_b}"] = {
            'delta_f1': float(delta), 'p_value': p,
            'significant_at_005': p < 0.05,
        }

    results['paired_tests'] = paired_results

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  Nested CV (no leakage):")
    print(f"    FQK+RBF:  {results['nested_cv_fqk_rbf']['nested_cv_f1']:.4f}")
    print(f"    CM+RBF:   {results['nested_cv_cm_rbf']['nested_cv_f1']:.4f}")
    print(f"    TRIPLE:   {results['nested_cv_triple']['nested_cv_f1']:.4f}")

    print(f"\n  Final test (consensus hyperparams from nested CV):")
    for name, y_pred in predictions.items():
        f1 = f1_score(y_test, y_pred, average='macro')
        ci = ci_results[name]
        print(f"    {name:20s}: {f1:.4f} [{ci['ci_lo']:.4f}, {ci['ci_hi']:.4f}]")

    # ---- Save ----
    results_path = os.path.join(OUT_DIR, "rigorous_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
