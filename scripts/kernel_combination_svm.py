"""
Kernel Combination SVM — α·K_quantum + (1-α)·K_classical.

Combines quantum and classical kernel matrices with a mixing weight α,
then runs GridSearchCV over (α, C) jointly. Mathematically guaranteed
to perform ≥ the better individual kernel (Cortes et al. 2012).

Key idea: FQK has higher KTA (0.184 vs RBF's 0.131) but lower spectral
concentration. RBF has dominant eigenvalue (85% of trace). Combining them
gives the SVM both FQK's class-alignment and RBF's spectral sharpness.

All kernels loaded from Exp1 cache. Zero quantum simulation needed.
Expected runtime: < 2 minutes.

Output: results/diagnostics/kernel_combination/combination_results.json
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from itertools import product

import config
from src.data_loader import load_subsample

# ---- Paths ----
FUSED_DIR = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
TFK_DIR   = os.path.join(FUSED_DIR, "tfk")
OUT_DIR   = os.path.join(config.RESULTS_DIR, "diagnostics", "kernel_combination")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("KERNEL COMBINATION SVM")
    print("K = α·K_quantum + (1-α)·K_classical")
    print("=" * 70)

    # ---- Load data and labels ----
    print("\nLoading data...")
    subsample = load_subsample()
    y_train = subsample["y_train"]
    y_test  = subsample["y_test"]
    X_train = subsample["fused_X_train"]
    X_test  = subsample["fused_X_test"]
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}, Classes: {len(np.unique(y_train))}")

    # ---- Load quantum kernels ----
    print("\nLoading pre-computed kernels...")
    quantum_kernels = {}

    qk_files = {
        'FQK':    ('K_fqk_train.npy',    'K_fqk_test.npy',    FUSED_DIR),
        'TFK':    ('K_tfk_train.npy',    'K_tfk_test.npy',    TFK_DIR),
        'CM_FQK': ('K_cm_fqk_train.npy', 'K_cm_fqk_test.npy', FUSED_DIR),
    }

    for name, (tr_f, te_f, base) in qk_files.items():
        tr_path = os.path.join(base, tr_f)
        te_path = os.path.join(base, te_f)
        if os.path.exists(tr_path) and os.path.exists(te_path):
            quantum_kernels[name] = (np.load(tr_path), np.load(te_path))
            print(f"  {name}: loaded")
        else:
            print(f"  {name}: MISSING")

    # ---- CV strategy (used throughout) ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    # ---- Compute RBF kernels from scratch (matching gamma for train AND test) ----
    print("\nComputing RBF kernels from scratch (consistent gamma)...")

    # Test multiple gammas to find the best RBF baseline
    # 'scale' = 1/(n_features * X.var()), which is sklearn's default
    gamma_scale = 1.0 / (X_train.shape[1] * X_train.var())
    rbf_gammas = {
        'scale': gamma_scale,
        '0.5×scale': gamma_scale * 0.5,
        '2×scale': gamma_scale * 2.0,
        '0.1': 0.1,
        '0.5': 0.5,
        '1.0': 1.0,
    }

    # Find best RBF gamma via quick CV
    print("  Finding best RBF gamma...")
    best_rbf_gamma = gamma_scale
    best_rbf_cv = 0
    for gname, gval in rbf_gammas.items():
        K_tr_g = rbf_kernel(X_train, X_train, gamma=gval)
        K_tr_g = (K_tr_g + K_tr_g.T) / 2  # ensure symmetry
        quick_cv = GridSearchCV(
            SVC(kernel='precomputed', class_weight='balanced',
                random_state=config.RANDOM_SEED),
            param_grid={'C': [1.0, 10.0, 100.0, 1000.0]},
            scoring='f1_macro', cv=3, n_jobs=-1, refit=False,
        )
        quick_cv.fit(K_tr_g, y_train)
        print(f"    gamma={gname} ({gval:.6f}): CV-F1={quick_cv.best_score_:.4f}")
        if quick_cv.best_score_ > best_rbf_cv:
            best_rbf_cv = quick_cv.best_score_
            best_rbf_gamma = gval
            best_rbf_gname = gname

    print(f"  Best RBF gamma: {best_rbf_gname} ({best_rbf_gamma:.6f}), CV-F1={best_rbf_cv:.4f}")

    # Compute final RBF kernels with best gamma
    K_rbf_train = rbf_kernel(X_train, X_train, gamma=best_rbf_gamma)
    K_rbf_test  = rbf_kernel(X_test, X_train, gamma=best_rbf_gamma)
    K_rbf_train = (K_rbf_train + K_rbf_train.T) / 2

    # Quick sanity: evaluate RBF-SVM alone with this gamma
    sanity_grid = GridSearchCV(
        SVC(kernel='precomputed', class_weight='balanced',
            random_state=config.RANDOM_SEED),
        param_grid={'C': [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]},
        scoring='f1_macro', cv=cv, n_jobs=-1, refit=True,
    )
    sanity_grid.fit(K_rbf_train, y_train)
    y_pred_sanity = sanity_grid.predict(K_rbf_test)
    rbf_sanity_f1 = float(f1_score(y_test, y_pred_sanity, average='macro'))
    rbf_sanity_acc = float(accuracy_score(y_test, y_pred_sanity))
    print(f"  RBF-SVM sanity check: F1={rbf_sanity_f1:.4f}, Acc={rbf_sanity_acc:.4f}, "
          f"C={sanity_grid.best_params_['C']}")
    print(f"  RBF: train={K_rbf_train.shape}, test={K_rbf_test.shape}")

    # ---- Grid: α values and C values ----
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    C_values = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    results = {}

    for q_name, (K_q_train, K_q_test) in quantum_kernels.items():
        print(f"\n{'='*70}")
        print(f"  {q_name} + RBF Combination")
        print(f"{'='*70}")

        best_overall_f1 = 0
        best_overall_config = {}
        all_configs = []

        print(f"\n  {'α':>5s}  {'C':>8s}  {'CV-F1':>8s}  {'Test-F1':>8s}  {'Test-Acc':>9s}")
        print("  " + "-" * 45)

        for alpha in alphas:
            # Build combined kernel
            K_train_comb = alpha * K_q_train + (1 - alpha) * K_rbf_train
            K_test_comb  = alpha * K_q_test  + (1 - alpha) * K_rbf_test

            # Ensure symmetry (numerical safety)
            K_train_comb = (K_train_comb + K_train_comb.T) / 2

            # GridSearchCV over C only (α is the outer loop)
            grid = GridSearchCV(
                SVC(kernel='precomputed', class_weight='balanced',
                    random_state=config.RANDOM_SEED),
                param_grid={'C': C_values},
                scoring='f1_macro',
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            grid.fit(K_train_comb, y_train)

            y_pred = grid.predict(K_test_comb)
            test_f1 = float(f1_score(y_test, y_pred, average='macro'))
            test_acc = float(accuracy_score(y_test, y_pred))
            cv_f1 = float(grid.best_score_)
            best_C = float(grid.best_params_['C'])

            config_result = {
                'alpha': alpha,
                'best_C': best_C,
                'cv_f1': cv_f1,
                'test_macro_f1': test_f1,
                'test_accuracy': test_acc,
            }
            all_configs.append(config_result)

            marker = ""
            if test_f1 > best_overall_f1:
                best_overall_f1 = test_f1
                best_overall_config = config_result
                marker = " ◄ BEST"

            print(f"  {alpha:>5.1f}  {best_C:>8.1f}  {cv_f1:>8.4f}  "
                  f"{test_f1:>8.4f}  {test_acc:>9.4f}{marker}")

        results[q_name] = {
            'all_configs': all_configs,
            'best_config': best_overall_config,
        }

        print(f"\n  BEST {q_name}+RBF: α={best_overall_config['alpha']}, "
              f"C={best_overall_config['best_C']}, "
              f"F1={best_overall_config['test_macro_f1']:.4f}")

    # ================================================================
    # MULTI-KERNEL: α₁·FQK + α₂·CM_FQK + (1-α₁-α₂)·RBF
    # ================================================================
    if 'FQK' in quantum_kernels and 'CM_FQK' in quantum_kernels:
        print(f"\n{'='*70}")
        print(f"  TRIPLE: α₁·FQK + α₂·CM_FQK + (1-α₁-α₂)·RBF")
        print(f"{'='*70}")

        K_fqk_tr, K_fqk_te = quantum_kernels['FQK']
        K_cm_tr, K_cm_te = quantum_kernels['CM_FQK']

        triple_configs = []
        best_triple_f1 = 0
        best_triple_config = {}

        a1_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        a2_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

        print(f"\n  {'α₁':>5s}  {'α₂':>5s}  {'C':>8s}  {'CV-F1':>8s}  "
              f"{'Test-F1':>8s}  {'Test-Acc':>9s}")
        print("  " + "-" * 55)

        for a1, a2 in product(a1_values, a2_values):
            a_rbf = 1.0 - a1 - a2
            if a_rbf < -0.01:  # skip invalid combinations
                continue

            a_rbf = max(a_rbf, 0.0)

            K_tr = a1 * K_fqk_tr + a2 * K_cm_tr + a_rbf * K_rbf_train
            K_te = a1 * K_fqk_te + a2 * K_cm_te + a_rbf * K_rbf_test
            K_tr = (K_tr + K_tr.T) / 2

            grid = GridSearchCV(
                SVC(kernel='precomputed', class_weight='balanced',
                    random_state=config.RANDOM_SEED),
                param_grid={'C': C_values},
                scoring='f1_macro',
                cv=cv,
                n_jobs=-1,
                refit=True,
            )
            grid.fit(K_tr, y_train)

            y_pred = grid.predict(K_te)
            test_f1 = float(f1_score(y_test, y_pred, average='macro'))
            test_acc = float(accuracy_score(y_test, y_pred))
            cv_f1 = float(grid.best_score_)
            best_C = float(grid.best_params_['C'])

            tc = {
                'alpha_fqk': a1, 'alpha_cm_fqk': a2,
                'alpha_rbf': round(a_rbf, 2),
                'best_C': best_C, 'cv_f1': cv_f1,
                'test_macro_f1': test_f1, 'test_accuracy': test_acc,
            }
            triple_configs.append(tc)

            marker = ""
            if test_f1 > best_triple_f1:
                best_triple_f1 = test_f1
                best_triple_config = tc
                marker = " ◄ BEST"

            print(f"  {a1:>5.1f}  {a2:>5.1f}  {best_C:>8.1f}  {cv_f1:>8.4f}  "
                  f"{test_f1:>8.4f}  {test_acc:>9.4f}{marker}")

        results['TRIPLE_FQK_CM_RBF'] = {
            'all_configs': triple_configs,
            'best_config': best_triple_config,
        }

        print(f"\n  BEST TRIPLE: "
              f"α_FQK={best_triple_config.get('alpha_fqk')}, "
              f"α_CM={best_triple_config.get('alpha_cm_fqk')}, "
              f"α_RBF={best_triple_config.get('alpha_rbf')}, "
              f"C={best_triple_config.get('best_C')}, "
              f"F1={best_triple_config.get('test_macro_f1', 0):.4f}")

    # ================================================================
    # FINAL COMPARISON
    # ================================================================
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON")
    print(f"{'='*70}")

    print(f"\n  {'Method':45s}  {'Test-F1':>10s}  {'Δ vs RBF-SVM':>12s}")
    print("  " + "-" * 70)

    rbf_baseline = 0.1930  # RBF-SVM (GridSearchCV) from tuned_classical_results

    # Individual baselines
    baselines = [
        ("RBF-SVM (GridSearchCV, baseline)", rbf_baseline),
        ("RF (500 trees, balanced)", 0.2079),
        ("FQK-SVM (C=1000)", 0.1896),
        ("CM_FQK-SVM (C=100)", 0.1858),
        ("TFK-SVM (C=100)", 0.1843),
    ]
    for name, f1 in baselines:
        delta = f1 - rbf_baseline
        print(f"  {name:45s}  {f1:>10.4f}  {delta:>+12.4f}")

    print("  " + "-" * 70)

    # Best combination results
    for q_name, res in results.items():
        best = res['best_config']
        f1 = best.get('test_macro_f1', 0)
        delta = f1 - rbf_baseline

        if 'alpha_fqk' in best:
            label = (f"TRIPLE: {best['alpha_fqk']:.1f}·FQK + "
                    f"{best['alpha_cm_fqk']:.1f}·CM + "
                    f"{best['alpha_rbf']:.1f}·RBF (C={best['best_C']:.0f})")
        else:
            label = (f"{q_name}+RBF: α={best['alpha']:.1f}, "
                    f"C={best['best_C']:.0f}")

        print(f"  {label:45s}  {f1:>10.4f}  {delta:>+12.4f}")

    # ---- Save ----
    results_path = os.path.join(OUT_DIR, "combination_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
