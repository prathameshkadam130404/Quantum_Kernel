"""
Bandwidth Cross-Validation for Quantum Kernel Encoding.

Selects the optimal encoding bandwidth scalar γ for each feature set via
5-fold cross-validation on the training set, following Shaydulin & Wild (2022).

The quantum kernel encodes features as RZ(γ · θ_bw · x[i]) where x ∈ [0,π].
γ controls how much of the Hilbert space is utilized. Too large → exponential
concentration. Too small → kernel near-constant. CV finds the sweet spot.

Outputs:
    results/bandwidth_cv/selected_gamma.json   — γ* for each feature set
    results/bandwidth_cv/cv_curves.pdf         — γ vs mean KTA curve
    results/bandwidth_cv/cv_details.json       — full CV results per fold/γ

Usage:
    python scripts/select_bandwidth_cv.py
    python scripts/select_bandwidth_cv.py --fast   # n=100 per fold, faster

Reference:
    Shaydulin & Wild, Phys. Rev. A 106, 042407 (2022)
    "Importance of kernel bandwidth in quantum machine learning"
"""

import sys, os, json, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

import config
from src.quantum_kernels import compute_fqk_kernel_matrix

# ============ CONFIG ============
# γ grid — log-spaced from 0.1 to 1.0
# Rationale:
#   γ=1.0 = current default [0,π]
#   γ=0.25 = ad-hoc fix tried earlier [0,π/4]
#   Intermediate values give the full CV curve
GAMMA_GRID = [0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0]

N_FOLDS    = 5
N_CV       = 200   # samples per fold evaluation (speed vs accuracy tradeoff)
                   # Use --fast for n=100 (15 min) or default n=200 (30-45 min)

# PCA v5 feature indices
PHYSICS_FEATURE_INDICES = [0, 5, 6, 3, 10, 15, 9, 11]
# = NDVI, NDRE, MNDWI, BSI, CrossPol, NDVI_tex, SAR_total, PolCoh

OUTPUT_DIR = os.path.join(config.RESULTS_DIR, "bandwidth_cv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def centered_class_weighted_kta(K, y):
    """Centered class-weighted KTA. Primary metric per Cortes et al. (2012)."""
    n = len(y)
    # Center kernel
    col_mean   = K.mean(axis=0)
    row_mean   = K.mean(axis=1)
    grand_mean = K.mean()
    Kc = K - col_mean[None, :] - row_mean[:, None] + grand_mean

    # Build ideal kernel with class-weighting for imbalance
    classes, counts = np.unique(y, return_counts=True)
    count_map = dict(zip(classes, counts))
    Ki = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if y[i] == y[j]:
                Ki[i, j] = 1.0 / count_map[y[i]]

    col_m = Ki.mean(axis=0); row_m = Ki.mean(axis=1); gm = Ki.mean()
    Kic = Ki - col_m[None, :] - row_m[:, None] + gm

    return float(
        np.sum(Kc * Kic) /
        (np.linalg.norm(Kc, 'fro') * np.linalg.norm(Kic, 'fro') + 1e-12)
    )


def run_cv_for_feature_set(X, y, name, n_cv=N_CV):
    """
    Run 5-fold CV over gamma grid for a single feature set.

    X: features already in [0,π] (raw normalized, γ not yet applied)
    y: labels
    name: string label for logging

    Returns: dict mapping γ → mean CV KTA
    """
    print(f"\n{'='*60}")
    print(f"  CV: {name} features  (n_total={len(y)}, n_cv={n_cv})")
    print(f"  γ grid: {GAMMA_GRID}")
    print(f"  5-fold × {len(GAMMA_GRID)} γ values = {5*len(GAMMA_GRID)} kernel computations")
    print(f"  Each kernel: {n_cv//2}×{n_cv//2} ≈ {(n_cv//2)**2 // 2} evaluations")
    print(f"{'='*60}")

    # Stratified subsample to n_cv for speed
    from sklearn.model_selection import train_test_split
    if len(y) > n_cv:
        _, X_cv, _, y_cv = train_test_split(
            X, y, test_size=n_cv,
            stratify=y, random_state=config.RANDOM_SEED
        )
    else:
        X_cv, y_cv = X, y

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                          random_state=config.RANDOM_SEED)

    gamma_results = {}  # γ → list of fold KTAs

    for gamma in GAMMA_GRID:
        X_encoded = X_cv * gamma   # Apply γ scaling
        fold_ktas = []

        print(f"\n  γ={gamma:.3f}  (encoding range [0, {gamma*np.pi:.3f}])")

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_encoded, y_cv)):
            X_tr = X_encoded[train_idx]
            y_tr = y_cv[train_idx]

            t0 = time.time()
            # Compute symmetric train kernel only — KTA uses training data only
            K_tr = compute_fqk_kernel_matrix(X_tr)
            elapsed = time.time() - t0

            kta = centered_class_weighted_kta(K_tr, y_tr)
            fold_ktas.append(kta)
            print(f"    fold {fold_idx+1}/{N_FOLDS}: KTA={kta:.4f}  [{elapsed:.0f}s]")

        mean_kta = float(np.mean(fold_ktas))
        std_kta  = float(np.std(fold_ktas))
        gamma_results[gamma] = {
            "fold_ktas": fold_ktas,
            "mean_kta": mean_kta,
            "std_kta": std_kta,
        }
        print(f"  γ={gamma:.3f}  →  mean KTA = {mean_kta:.4f} ± {std_kta:.4f}")

    # Select γ* = argmax mean KTA
    gamma_star = max(gamma_results, key=lambda g: gamma_results[g]["mean_kta"])
    best_kta   = gamma_results[gamma_star]["mean_kta"]

    print(f"\n  {'='*40}")
    print(f"  SELECTED γ* = {gamma_star:.3f}  (mean KTA = {best_kta:.4f})")
    print(f"  Encoding range: [0, {gamma_star*np.pi:.4f}]  = [0, {gamma_star:.3f}π]")
    print(f"  {'='*40}")

    return gamma_star, gamma_results


def plot_cv_curves(pca_results, phys_results, gamma_star_pca, gamma_star_phys):
    """Save CV curves for both feature sets."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, results, gamma_star, title in [
        (axes[0], pca_results,  gamma_star_pca,  "PCA Features"),
        (axes[1], phys_results, gamma_star_phys, "Physics Features"),
    ]:
        gammas   = sorted(results.keys())
        means    = [results[g]["mean_kta"] for g in gammas]
        stds     = [results[g]["std_kta"]  for g in gammas]

        ax.errorbar(gammas, means, yerr=stds, marker='o', capsize=4,
                    linewidth=2, color='steelblue', label='CV KTA ± std')
        ax.axvline(gamma_star, color='red', linestyle='--', linewidth=1.5,
                   label=f'γ* = {gamma_star:.3f}')
        ax.set_xlabel('γ (encoding bandwidth scalar)')
        ax.set_ylabel('Mean 5-fold KTA')
        ax.set_title(f'Bandwidth CV — {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "cv_curves.pdf")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\n  CV curves saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true',
                        help='Use n=100 per fold instead of n=200 (faster, less precise)')
    args = parser.parse_args()

    n_cv = 100 if args.fast else N_CV

    print("=" * 60)
    print("BANDWIDTH CROSS-VALIDATION")
    print("Following Shaydulin & Wild, Phys. Rev. A 106, 042407 (2022)")
    print(f"γ grid: {GAMMA_GRID}")
    print(f"n_cv per fold: {n_cv}, folds: {N_FOLDS}")
    print("=" * 60)

    # ============ LOAD PCA FEATURES ============
    pca_path = os.path.join(config.PROCESSED_DIR, "subsample_2000.npz")
    if not os.path.exists(pca_path):
        print(f"ERROR: PCA features not found: {pca_path}")
        sys.exit(1)

    pca_data  = np.load(pca_path)
    X_pca     = pca_data["fused_X_train"]   # (2000, 8), already in [0,π]
    y_pca     = pca_data["y_train"]
    print(f"\nPCA features loaded: {X_pca.shape}, range [{X_pca.min():.3f}, {X_pca.max():.3f}]")

    # If keys differ, try common alternatives
    if "fused_X_train" not in pca_data:
        # Try to find the right key
        available = list(pca_data.keys())
        print(f"  Keys in subsample_2000.npz: {available}")
        # Look for fused or X_train key
        for key in ["X_train", "fused_train", "X_fused_train"]:
            if key in pca_data:
                X_pca = pca_data[key]
                break
        for key in ["y_train", "y"]:
            if key in pca_data:
                y_pca = pca_data[key]
                break

    # ============ LOAD PHYSICS FEATURES ============
    phys16_path = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
    if not os.path.exists(phys16_path):
        print(f"ERROR: Physics features not found: {phys16_path}")
        sys.exit(1)

    phys_data  = np.load(phys16_path)
    X_phys16   = phys_data["X_train"]   # (2000, 16), in [0,π]
    y_phys     = phys_data["y_train"]
    X_phys     = X_phys16[:, PHYSICS_FEATURE_INDICES]   # (2000, 8)
    print(f"Physics features loaded: {X_phys.shape}, range [{X_phys.min():.3f}, {X_phys.max():.3f}]")

    # Sanity check: features should be in [0, π]
    assert X_pca.max() <= np.pi + 0.01,   f"PCA features out of [0,π]: max={X_pca.max():.3f}"
    assert X_phys.max() <= np.pi + 0.01,  f"Physics features out of [0,π]: max={X_phys.max():.3f}"

    # ============ RUN CV ============
    gamma_star_pca, pca_cv_results = run_cv_for_feature_set(
        X_pca, y_pca, "PCA", n_cv=n_cv
    )
    gamma_star_phys, phys_cv_results = run_cv_for_feature_set(
        X_phys, y_phys, "Physics", n_cv=n_cv
    )

    # ============ SAVE RESULTS ============
    selected = {
        "gamma_pca":     gamma_star_pca,
        "gamma_physics": gamma_star_phys,
        "n_cv":          n_cv,
        "n_folds":       N_FOLDS,
        "gamma_grid":    GAMMA_GRID,
        "reference":     "Shaydulin & Wild, PRA 106, 042407 (2022)",
        "note": (
            "gamma is applied as X_encoded = gamma * X_normalized where X_normalized in [0,pi]. "
            "Selected by argmax of 5-fold CV centered class-weighted KTA on training set only."
        )
    }

    selected_path = os.path.join(OUTPUT_DIR, "selected_gamma.json")
    with open(selected_path, 'w') as f:
        json.dump(selected, f, indent=2)
    print(f"\n  Selected γ saved: {selected_path}")

    details = {
        "pca": {
            str(g): v for g, v in pca_cv_results.items()
        },
        "physics": {
            str(g): v for g, v in phys_cv_results.items()
        }
    }
    details_path = os.path.join(OUTPUT_DIR, "cv_details.json")
    with open(details_path, 'w') as f:
        json.dump(details, f, indent=2)
    print(f"  CV details saved: {details_path}")

    # ============ PLOT ============
    try:
        plot_cv_curves(pca_cv_results, phys_cv_results,
                       gamma_star_pca, gamma_star_phys)
    except Exception as e:
        print(f"  [WARN] Plot failed: {e}")

    # ============ SUMMARY ============
    print(f"\n{'='*60}")
    print("BANDWIDTH CV SUMMARY")
    print(f"{'='*60}")
    print(f"  PCA features:     γ* = {gamma_star_pca:.3f}  "
          f"(range [0, {gamma_star_pca*np.pi:.4f}] = [0, {gamma_star_pca:.3f}π])")
    print(f"  Physics features: γ* = {gamma_star_phys:.3f}  "
          f"(range [0, {gamma_star_phys*np.pi:.4f}] = [0, {gamma_star_phys:.3f}π])")
    print(f"\n  If γ_pca = 1.0:   PCA exp1 results VALID as-is, no rerun needed")
    print(f"  If γ_pca < 1.0:   PCA exp1 must be RERUN with γ={gamma_star_pca}")
    print(f"  Physics exp1_physics: ALWAYS apply γ={gamma_star_phys}")
    print(f"\n  Next step: add load_bandwidth_gamma() call to both experiments")
    print(f"  See: results/bandwidth_cv/selected_gamma.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
