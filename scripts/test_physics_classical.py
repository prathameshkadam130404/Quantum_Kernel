"""
16 Physics-Informed Feature Extraction + Classical Baseline Comparison.

Extracts 16 physics features (8 optical + 8 SAR), then compares RF + RBF-SVM
on three feature sets: 8 PCA (existing), 8 physics, 16 physics.

This is the go/no-go gate before computing quantum kernels.
"""

import os, sys, gc, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.metrics.pairwise import rbf_kernel

import config
from src.data_loader import load_subsample

EPS = 1e-8


# ============ FEATURE EXTRACTION ============

def safe_ratio(a, b):
    return (a - b) / (a + b + EPS)


def extract_16_features(sar_patches, opt_patches):
    """
    Extract 16 physics-informed features from raw SAR+optical patches.

    Args:
        sar_patches: (N, 32, 32, 8) Sentinel-1
        opt_patches: (N, 32, 32, 10) Sentinel-2

    Returns:
        features: (N, 16) array
        names: list of 16 feature names
    """
    N = sar_patches.shape[0]

    # ---- Optical: spatial means per band ----
    opt_mean = opt_patches.mean(axis=(1, 2))  # (N, 10)
    blue  = opt_mean[:, 0]   # B2
    green = opt_mean[:, 1]   # B3
    red   = opt_mean[:, 2]   # B4
    vre1  = opt_mean[:, 3]   # B5
    nir   = opt_mean[:, 6]   # B8
    swir1 = opt_mean[:, 8]   # B11

    # ---- SAR: spatial means per channel ----
    sar_mean = sar_patches.mean(axis=(1, 2))  # (N, 8)
    vh_lee = np.abs(sar_mean[:, 4]) + EPS
    vv_lee = np.abs(sar_mean[:, 5]) + EPS
    cov_re = sar_mean[:, 6]
    cov_im = sar_mean[:, 7]

    # ---- 8 Optical features ----
    f1_ndvi  = safe_ratio(nir, red)
    f2_ndbi  = safe_ratio(swir1, nir)
    f3_ndwi  = safe_ratio(green, nir)
    f4_bsi   = safe_ratio(swir1 + red, nir + blue)
    f5_savi  = 1.5 * (nir - red) / (nir + red + 0.5 + EPS)
    f6_ndre  = safe_ratio(nir, vre1)
    f7_mndwi = safe_ratio(green, swir1)
    f8_evi   = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + EPS)

    # ---- 8 SAR features ----
    f9_vv_vh    = vv_lee / (vh_lee + EPS)
    f10_total   = vh_lee + vv_lee
    f11_xpol    = vh_lee / (vv_lee + EPS)
    cov_mag     = np.sqrt(cov_re**2 + cov_im**2)
    f12_coh     = cov_mag / (np.sqrt(vv_lee * vh_lee) + EPS)
    f13_vh_db   = 10 * np.log10(vh_lee + EPS)
    f14_vv_db   = 10 * np.log10(vv_lee + EPS)

    # Spatial texture features (std over 32x32 patch)
    f15_vv_tex  = sar_patches[:, :, :, 5].std(axis=(1, 2))   # VV texture

    # Per-pixel NDVI then spatial std
    pixel_nir  = opt_patches[:, :, :, 6]  # (N, 32, 32)
    pixel_red  = opt_patches[:, :, :, 2]
    pixel_ndvi = (pixel_nir - pixel_red) / (pixel_nir + pixel_red + EPS)
    f16_ndvi_tex = pixel_ndvi.std(axis=(1, 2))

    features = np.stack([
        f1_ndvi, f2_ndbi, f3_ndwi, f4_bsi,
        f5_savi, f6_ndre, f7_mndwi, f8_evi,
        f9_vv_vh, f10_total, f11_xpol, f12_coh,
        f13_vh_db, f14_vv_db, f15_vv_tex, f16_ndvi_tex,
    ], axis=1)

    names = [
        "NDVI", "NDBI", "NDWI", "BSI",
        "SAVI", "NDRE", "MNDWI", "EVI",
        "VV/VH", "SAR_total", "CrossPol", "PolCoherence",
        "VH_dB", "VV_dB", "VV_texture", "NDVI_texture",
    ]

    return features, names


def normalize_to_pi(features, lo=None, hi=None):
    """Robust [0, π] normalization. Returns (normalized, lo, hi)."""
    if lo is None:
        lo = np.percentile(features, 1, axis=0)
        hi = np.percentile(features, 99, axis=0)

    result = np.zeros_like(features)
    for i in range(features.shape[1]):
        if abs(hi[i] - lo[i]) < EPS:
            result[:, i] = np.pi / 2
        else:
            scaled = np.clip((features[:, i] - lo[i]) / (hi[i] - lo[i]), 0.0, 1.0)
            result[:, i] = scaled * np.pi
    return result, lo, hi


# ============ CLASSICAL BASELINE ============

def run_classical_comparison(X_train, X_test, y_train, y_test, label):
    """Run RF and RBF-SVM on given features, return results dict."""
    print(f"\n  [{label}] shape: train={X_train.shape}, test={X_test.shape}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)

    # ---- Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=500, class_weight='balanced',
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    f1_rf = f1_score(y_test, y_pred_rf, average='macro')
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"    RF: F1={f1_rf:.4f}, Acc={acc_rf:.4f}")

    # ---- RBF-SVM with GridSearchCV ----
    C_values = [1, 10, 100, 1000, 10000]
    gamma_values = ['scale', 'auto']

    svm = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced', random_state=config.RANDOM_SEED),
        param_grid={'C': C_values, 'gamma': gamma_values},
        scoring='f1_macro', cv=cv, n_jobs=-1, refit=True,
    )
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    f1_svm = f1_score(y_test, y_pred_svm, average='macro')
    acc_svm = accuracy_score(y_test, y_pred_svm)
    print(f"    SVM: F1={f1_svm:.4f}, Acc={acc_svm:.4f} "
          f"(best C={svm.best_params_['C']}, γ={svm.best_params_['gamma']})")

    return {
        'label': label,
        'rf_f1': f1_rf, 'rf_acc': acc_rf,
        'svm_f1': f1_svm, 'svm_acc': acc_svm,
        'svm_best_params': svm.best_params_,
        'rf_predictions': y_pred_rf,
        'svm_predictions': y_pred_svm,
    }


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--skip-extraction", action="store_true",
                       help="Skip feature extraction, load from saved .npz")
    args = parser.parse_args()

    print("=" * 70)
    print("CLASSICAL BASELINE: 8 PCA vs 8 Physics vs 16 Physics")
    print("=" * 70)

    # ==== Load existing 8 PCA features ====
    print("\n--- Loading 8 PCA features (existing baseline) ---")
    subsample = load_subsample()
    X_train_pca = subsample["fused_X_train"]
    X_test_pca  = subsample["fused_X_test"]
    y_train     = subsample["y_train"]
    y_test      = subsample["y_test"]
    print(f"  PCA: train={X_train_pca.shape}, test={X_test_pca.shape}")

    # ==== Extract 16 physics features ====
    phys16_file = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")

    if args.skip_extraction and os.path.exists(phys16_file):
        print("\n--- Loading saved physics features ---")
        data = np.load(phys16_file)
        X_train_phys16 = data["X_train_raw"]
        X_test_phys16  = data["X_test_raw"]
        X_train_phys16_norm = data["X_train"]
        X_test_phys16_norm  = data["X_test"]
        y_train_phys = data["y_train"]
        y_test_phys  = data["y_test"]
        feature_names = list(data["feature_names"])
    else:
        print("\n--- Extracting 16 physics features from raw data ---")
        from setup_data import (find_local_h5_files, load_labels_only,
                               load_h5_by_indices, get_stratified_indices)

        h5_files = find_local_h5_files(override_path=args.data_path)
        if not h5_files["train"] or not h5_files["test"]:
            print("ERROR: HDF5 files not found. Use --data-path.")
            sys.exit(1)

        y_train_full = load_labels_only(h5_files["train"])
        y_test_full  = load_labels_only(h5_files["test"])

        n_train_load = min(config.PCA_FIT_SAMPLES, len(y_train_full))
        n_test_load  = min(config.SUBSAMPLE_TEST * 4, len(y_test_full))

        train_indices = get_stratified_indices(y_train_full, n_train_load)
        test_indices  = get_stratified_indices(y_test_full, n_test_load)

        # Load raw patches
        sar_train, _, _ = load_h5_by_indices(h5_files["train"], train_indices, load_only="sar")
        sar_test, _, _  = load_h5_by_indices(h5_files["test"], test_indices, load_only="sar")
        _, opt_train, _ = load_h5_by_indices(h5_files["train"], train_indices, load_only="optical")
        _, opt_test, _  = load_h5_by_indices(h5_files["test"], test_indices, load_only="optical")

        y_train_loaded = y_train_full[train_indices]
        y_test_loaded  = y_test_full[test_indices]

        # Extract features
        print("  Extracting 16 features...")
        features_train, feature_names = extract_16_features(sar_train, opt_train)
        features_test, _              = extract_16_features(sar_test, opt_test)
        del sar_train, sar_test, opt_train, opt_test
        gc.collect()

        # Normalize
        features_train_norm, lo, hi = normalize_to_pi(features_train)
        features_test_norm, _, _    = normalize_to_pi(features_test, lo, hi)

        # Subsample to 2000/2000
        if config.SUBSAMPLE_TRAIN < len(features_train):
            _, sub_train, _, y_sub_train = train_test_split(
                features_train, y_train_loaded,
                test_size=config.SUBSAMPLE_TRAIN, stratify=y_train_loaded,
                random_state=config.RANDOM_SEED,
            )
            _, sub_train_norm, _, _ = train_test_split(
                features_train_norm, y_train_loaded,
                test_size=config.SUBSAMPLE_TRAIN, stratify=y_train_loaded,
                random_state=config.RANDOM_SEED,
            )
        else:
            sub_train, y_sub_train = features_train, y_train_loaded
            sub_train_norm = features_train_norm

        if config.SUBSAMPLE_TEST < len(features_test):
            _, sub_test, _, y_sub_test = train_test_split(
                features_test, y_test_loaded,
                test_size=config.SUBSAMPLE_TEST, stratify=y_test_loaded,
                random_state=config.RANDOM_SEED,
            )
            _, sub_test_norm, _, _ = train_test_split(
                features_test_norm, y_test_loaded,
                test_size=config.SUBSAMPLE_TEST, stratify=y_test_loaded,
                random_state=config.RANDOM_SEED,
            )
        else:
            sub_test, y_sub_test = features_test, y_test_loaded
            sub_test_norm = features_test_norm

        X_train_phys16 = sub_train
        X_test_phys16  = sub_test
        X_train_phys16_norm = sub_train_norm
        X_test_phys16_norm  = sub_test_norm
        y_train_phys = y_sub_train
        y_test_phys  = y_sub_test

        # Save
        np.savez_compressed(
            phys16_file,
            X_train=sub_train_norm, X_test=sub_test_norm,
            X_train_raw=sub_train, X_test_raw=sub_test,
            y_train=y_sub_train, y_test=y_sub_test,
            feature_names=np.array(feature_names),
            normalization_lo=lo, normalization_hi=hi,
        )
        print(f"  Saved: {phys16_file}")

    # 8 physics = first 8 of 16
    X_train_phys8 = X_train_phys16[:, :8]
    X_test_phys8  = X_test_phys16[:, :8]

    # ==== Feature statistics ====
    print(f"\n{'='*70}")
    print("FEATURE STATISTICS (16 physics)")
    print(f"{'='*70}")
    for i, name in enumerate(feature_names):
        col = X_train_phys16[:, i]
        print(f"  {name:15s}: min={col.min():+8.4f}  max={col.max():+8.4f}  "
              f"mean={col.mean():+8.4f}  std={col.std():.4f}")

    # ==== Run comparisons ====
    print(f"\n{'='*70}")
    print("CLASSICAL BASELINE COMPARISON")
    print(f"{'='*70}")

    results = []

    # Use same labels for all comparisons
    # (PCA subsample should match physics subsample)
    r1 = run_classical_comparison(X_train_pca, X_test_pca, y_train, y_test,
                                   "8 PCA (fused)")
    results.append(r1)

    r2 = run_classical_comparison(X_train_phys8, X_test_phys8, y_train_phys, y_test_phys,
                                   "8 Physics")
    results.append(r2)

    r3 = run_classical_comparison(X_train_phys16, X_test_phys16, y_train_phys, y_test_phys,
                                   "16 Physics")
    results.append(r3)

    # ==== Summary table ====
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Feature Set':20s}  {'RF F1':>8s}  {'RF Acc':>8s}  {'SVM F1':>8s}  {'SVM Acc':>8s}")
    print("  " + "-" * 56)
    for r in results:
        print(f"  {r['label']:20s}  {r['rf_f1']:>8.4f}  {r['rf_acc']:>8.4f}  "
              f"{r['svm_f1']:>8.4f}  {r['svm_acc']:>8.4f}")

    # Best vs PCA
    best_phys_rf = max(r2['rf_f1'], r3['rf_f1'])
    delta_rf = best_phys_rf - r1['rf_f1']
    best_phys_svm = max(r2['svm_f1'], r3['svm_f1'])
    delta_svm = best_phys_svm - r1['svm_f1']

    print(f"\n  Physics improvement over PCA:")
    print(f"    RF:  {delta_rf:+.4f} ({'BETTER' if delta_rf > 0 else 'WORSE'})")
    print(f"    SVM: {delta_svm:+.4f} ({'BETTER' if delta_svm > 0 else 'WORSE'})")

    if delta_rf > 0 or delta_svm > 0:
        print(f"\n  ✅ GO: Physics features improve at least one baseline.")
        print(f"  → Proceed to quantum kernel computation.")
    else:
        print(f"\n  ❌ NO-GO: Physics features don't improve classical baselines.")
        print(f"  → Investigate before computing quantum kernels.")


if __name__ == "__main__":
    main()
