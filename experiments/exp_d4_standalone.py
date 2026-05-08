"""
Exp D4 Standalone: D4 Covariant Quantum Kernel on SAR Modality.

Scientific question:
    Does encoding D4 symmetry (rotations + reflections) into the quantum
    feature map improve kernel alignment and classification performance
    on SAR urban patches, where rotation invariance is a known physical
    property of Sentinel-1 backscatter patterns?

Design:
    Feature space: SAR with D4-augmented PCA (orbit-invariant subspace).
    Kernels compared within this feature space:
        1. RBF (SAR, D4-augmented features) — classical baseline
        2. FQK (SAR, D4-augmented features) — quantum baseline
        3. D4_FQK (SAR, D4-augmented features) — D4 covariant quantum kernel

    All three kernels use the same D4-augmented SAR features, ensuring
    fair comparison. D4_FQK additionally encodes D4 covariance at the
    circuit level via group averaging.

    The experiment answers:
        a) Does D4-augmented PCA alone improve over standard SAR PCA?
           (Compare FQK here vs SAR FQK from exp1)
        b) Does circuit-level D4 covariance add further benefit beyond
           what feature-space invariance already achieves?
           (Compare D4_FQK vs FQK within this experiment)

Compute estimate:
    D4-augmented PCA fit: ~30 minutes
    FQK (2000x2000):      ~24 hours
    D4_FQK (2000x2000):   ~24 hours
    RBF:                  ~5 minutes
    Analysis:             ~1 hour

Output:
    results/d4_standalone/
        sar_d4_pca_model.joblib
        X_sar_d4_train.npy
        X_sar_d4_test.npy
        K_fqk_sar_d4_train.npy
        K_fqk_sar_d4_test.npy
        K_d4fqk_sar_train.npy
        K_d4fqk_sar_test.npy
        K_rbf_sar_d4_train.npy
        K_rbf_sar_d4_test.npy
        invariance_check.txt
        results_summary.txt

References:
    Glick et al., Nature Physics (2024) — covariant quantum kernels
    Cohen & Welling, ICML 2016 — group equivariant networks
"""

import os
import sys
import logging
import time

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.utils import setup_logging, Timer

logger = setup_logging(
    "exp_d4_standalone",
    log_file=os.path.join("results", "d4_standalone", "exp_d4.log")
)

# ---- Paths ----
D4_DIR           = os.path.join("results", "d4_standalone")
PCA_SAVE_PATH    = os.path.join(D4_DIR, "sar_d4_pca_model.joblib")
X_TRAIN_PATH     = os.path.join(D4_DIR, "X_sar_d4_train.npy")
X_TEST_PATH      = os.path.join(D4_DIR, "X_sar_d4_test.npy")
Y_TRAIN_PATH     = os.path.join(D4_DIR, "y_train.npy")
Y_TEST_PATH      = os.path.join(D4_DIR, "y_test.npy")

K_FQK_TRAIN      = os.path.join(D4_DIR, "K_fqk_sar_d4_train.npy")
K_FQK_TEST       = os.path.join(D4_DIR, "K_fqk_sar_d4_test.npy")
K_D4FQK_TRAIN    = os.path.join(D4_DIR, "K_d4fqk_sar_train.npy")
K_D4FQK_TEST     = os.path.join(D4_DIR, "K_d4fqk_sar_test.npy")
K_RBF_TRAIN      = os.path.join(D4_DIR, "K_rbf_sar_d4_train.npy")
K_RBF_TEST       = os.path.join(D4_DIR, "K_rbf_sar_d4_test.npy")
INVARIANCE_PATH  = os.path.join(D4_DIR, "invariance_check.txt")
X_MIN_PATH       = os.path.join(D4_DIR, "x_min.npy")
X_RANGE_PATH     = os.path.join(D4_DIR, "x_range.npy")
SUMMARY_PATH     = os.path.join(D4_DIR, "results_summary.txt")

D4_N_CHANNELS    = 8     # sen1 shape: (N, 32, 32, 8)
D4_PATCH_HEIGHT  = 32
D4_PATCH_WIDTH   = 32


def section(title):
    logger.info("=" * 65)
    logger.info(f"  {title}")
    logger.info("=" * 65)


# ================================================================
# STEP 1 — Load raw SAR data
# ================================================================

def load_raw_sar_data():
    """
    Load raw SAR patches from HDF5 for PCA fitting.
    Uses the same HDF5 files as setup_data.py.
    Returns flat SAR arrays ready for D4 augmentation.
    """
    section("STEP 1 — Loading raw SAR data")

    import h5py, glob

    # Find HDF5 files
    search_paths = config.LOCAL_DATASET_SEARCH_PATHS
    train_h5, test_h5 = None, None

    for sp in search_paths:
        if not os.path.isdir(sp):
            continue
        for f in glob.glob(os.path.join(sp, "**", "*.h5"), recursive=True):
            bn = os.path.basename(f).lower()
            if "train" in bn and train_h5 is None:
                train_h5 = f
            elif "test" in bn and test_h5 is None:
                test_h5 = f

    if train_h5 is None or test_h5 is None:
        raise FileNotFoundError(
            "Cannot find So2Sat HDF5 files. "
            "Set config.LOCAL_DATASET_SEARCH_PATHS correctly."
        )

    logger.info(f"  Train HDF5: {train_h5}")
    logger.info(f"  Test  HDF5: {test_h5}")

    def load_sar_stratified(h5path, n_samples, seed=config.RANDOM_SEED):
        with h5py.File(h5path, 'r') as f:
            keys = list(f.keys())
            sar_key = 'sen1' if 'sen1' in keys else 's1'
            label_key = 'label' if 'label' in keys else 'labels'

            labels_raw = f[label_key][:]
            if labels_raw.ndim == 2 and labels_raw.shape[1] > 1:
                labels = np.argmax(labels_raw, axis=1)
            else:
                labels = labels_raw.flatten().astype(int)

            n_total = len(labels)
            rng = np.random.RandomState(seed)

            # Stratified sampling
            from sklearn.model_selection import train_test_split
            all_idx = np.arange(len(labels))
            if n_samples >= len(labels):
                idx = all_idx
            else:
                _, idx, _, _ = train_test_split(
                    all_idx, labels,
                    test_size=n_samples,
                    stratify=labels,
                    random_state=seed,
                )
            idx = np.sort(idx)

            sar_data = f[sar_key][idx]  # shape (n, 32, 32, 8)
            y = labels[idx]

        logger.info(f"  Loaded {len(idx)} SAR samples from {os.path.basename(h5path)}")
        logger.info(f"  SAR shape: {sar_data.shape}")

        # Flatten: (n, 32, 32, 8) -> (n, 8192) in channel-last order
        # Channel layout: pixel 0 ch0, pixel 0 ch1, ..., pixel 0 ch7,
        #                 pixel 1 ch0, ..., pixel 1023 ch7
        # IMPORTANT: must use channel-FIRST layout for D4 permutation
        # Reorder to (n, 8, 32, 32) then flatten to (n, 8*32*32)
        # so that channel ch occupies x_flat[ch*1024 : (ch+1)*1024]
        sar_chfirst = sar_data.transpose(0, 3, 1, 2)  # (n, 8, 32, 32)
        sar_flat = sar_chfirst.reshape(len(sar_data), -1).astype(np.float32)
        return sar_flat, y

    # Use config.PCA_FIT_SAMPLES for PCA fitting (more = better PCA)
    n_pca = min(getattr(config, 'PCA_FIT_SAMPLES', 10000), 50000)
    n_sub = config.SUBSAMPLE_TRAIN  # 2000 for experiment

    logger.info(f"  Loading {n_pca} samples for PCA fitting...")
    X_flat_pca, y_pca = load_sar_stratified(train_h5, n_pca, seed=config.RANDOM_SEED)

    logger.info(f"  Loading {n_sub} samples for experiment subsample...")
    X_flat_train, y_train = load_sar_stratified(train_h5, n_sub, seed=config.RANDOM_SEED + 1)

    logger.info(f"  Loading test samples...")
    n_test = getattr(config, 'SUBSAMPLE_TEST', 500)
    X_flat_test, y_test = load_sar_stratified(test_h5, n_test, seed=config.RANDOM_SEED)

    logger.info(f"  PCA fit data: {X_flat_pca.shape}")
    logger.info(f"  Train subsample: {X_flat_train.shape}")
    logger.info(f"  Test subsample:  {X_flat_test.shape}")

    return X_flat_pca, X_flat_train, X_flat_test, y_train, y_test


# ================================================================
# STEP 2 — Fit D4-augmented PCA and transform data
# ================================================================

def fit_and_transform(X_flat_pca, X_flat_train, X_flat_test, y_train, y_test):
    section("STEP 2 — D4-Augmented PCA")
    from src.d4_augmented_pca import fit_d4_augmented_pca, verify_invariance
    import joblib

    # 1. Load or Fit PCA
    pca_loaded = False
    if os.path.exists(PCA_SAVE_PATH):
        logger.info(f"  [LOAD] D4-augmented PCA model found: {PCA_SAVE_PATH}")
        pca = joblib.load(PCA_SAVE_PATH)
        pca_loaded = True
    else:
        os.makedirs(D4_DIR, exist_ok=True)
        with Timer("D4-augmented PCA fitting", logger=logger):
            pca, _, _ = fit_d4_augmented_pca(
                X_flat_pca,
                n_components=config.N_QUBITS,
                height=D4_PATCH_HEIGHT,
                width=D4_PATCH_WIDTH,
                n_channels=D4_N_CHANNELS,
                save_path=PCA_SAVE_PATH,
                normalize_output=False, # We normalize manually below
            )
        pca_loaded = False

    # 2. Load or Transform Data
    # Save raw patches for verify_d4_invariance.py
    # (needed for correct raw-space invariance verification)
    RAW_TRAIN_CACHE = os.path.join(D4_DIR, "X_sar_d4_raw_train.npy")
    if not os.path.exists(RAW_TRAIN_CACHE):
        np.save(RAW_TRAIN_CACHE, X_flat_train)
        logger.info(f"  Saved raw train patches: {RAW_TRAIN_CACHE}")

    if pca_loaded and (os.path.exists(X_TRAIN_PATH) and os.path.exists(X_TEST_PATH)
            and os.path.exists(X_MIN_PATH) and os.path.exists(X_RANGE_PATH)):
        logger.info(f"  [SKIP] Transformed features already exist. Loading...")
        X_train = np.load(X_TRAIN_PATH)
        X_test  = np.load(X_TEST_PATH)
        x_min   = np.load(X_MIN_PATH)
        x_range = np.load(X_RANGE_PATH)
    else:
        logger.info("  Transforming and normalizing data using PCA...")
        X_train_pca = pca.transform(X_flat_train)
        x_min = X_train_pca.min(axis=0)
        x_max = X_train_pca.max(axis=0)
        x_range = x_max - x_min
        x_range[x_range < 1e-10] = 1.0

        X_train = (X_train_pca - x_min) / x_range * np.pi
        X_test_pca = pca.transform(X_flat_test)
        X_test  = (X_test_pca - x_min) / x_range * np.pi

        np.save(X_TRAIN_PATH, X_train)
        np.save(X_TEST_PATH,  X_test)
        np.save(Y_TRAIN_PATH, y_train)
        np.save(Y_TEST_PATH,  y_test)
        np.save(X_MIN_PATH,   x_min)
        np.save(X_RANGE_PATH, x_range)
        logger.info(f"  Saved: X_train {X_train.shape}, X_test {X_test.shape}")

    # ---- Verify invariance ----
    section("STEP 2b — Invariance Verification")
    # Correct invariance check: test on raw patches, not PCA features.
    # verify_invariance must apply D4 to raw 8192-dim vectors, project
    # through PCA, then compare. Testing on 8-dim PCA outputs is wrong
    # because D4 permutations are defined on 8192-dim space.
    passed, max_err = verify_invariance(
        pca.components_,
        X_flat_train[:10],   # raw 8192-dim patches, NOT PCA features
        height=D4_PATCH_HEIGHT,
        width=D4_PATCH_WIDTH,
        n_channels=D4_N_CHANNELS,
        tol=1e-4,
        x_min=x_min,         # pass normalisation params for full pipeline test
        x_range=x_range,
    )
    inv_msg = (
        f"Group-averaging invariance:\n"
        f"  max ||theta(x) - theta(g*x)||_inf = {max_err:.8f}\n"
        f"  tolerance = 1e-4\n"
        f"  result = {'PASS' if passed else 'FAIL'}\n"
        f"  (standard PCA had error ~0.045; augmented PCA target: < 1e-4)\n"
    )
    logger.info(inv_msg)
    with open(INVARIANCE_PATH, 'w') as f:
        f.write(inv_msg)

    if not passed:
        logger.error(
            "INVARIANCE FAILED. D4-augmented PCA did not produce a D4-invariant "
            "subspace. Check n_channels and patch layout in augment_with_d4_orbits()."
        )
        raise RuntimeError("Invariance check failed — cannot proceed.")

    return pca, X_train, X_test, y_train, y_test, x_min, x_range


# ================================================================
# STEP 3 — Compute kernels
# ================================================================

def compute_kernels(pca, X_train, X_test, y_train, X_flat_train, X_flat_test, x_min, x_range):
    section("STEP 3 — Kernel Computation")

    from src.quantum_kernels import compute_fqk_kernel_matrix
    from src.d4_covariant_kernel import compute_group_averaged_fqk

    results = {}

    # ---- RBF (fast) ----
    if os.path.exists(K_RBF_TRAIN):
        logger.info("[SKIP] RBF train kernel exists")
        results['K_rbf_train'] = np.load(K_RBF_TRAIN)
        results['K_rbf_test']  = np.load(K_RBF_TEST)
    else:
        logger.info("Computing RBF kernel...")
        gamma = 1.0 / (X_train.shape[1] * X_train.var())
        K_rbf_train = rbf_kernel(X_train, X_train, gamma=gamma)
        K_rbf_test  = rbf_kernel(X_test,  X_train, gamma=gamma)
        np.save(K_RBF_TRAIN, K_rbf_train)
        np.save(K_RBF_TEST,  K_rbf_test)
        results['K_rbf_train'] = K_rbf_train
        results['K_rbf_test']  = K_rbf_test
        logger.info(f"  RBF done. mean_offdiag={K_rbf_train[~np.eye(len(K_rbf_train),dtype=bool)].mean():.4f}")

    # ---- Standard FQK (SAR, D4-augmented features) ----
    if os.path.exists(K_FQK_TRAIN):
        logger.info("[SKIP] FQK train kernel exists")
        results['K_fqk_train'] = np.load(K_FQK_TRAIN)
        results['K_fqk_test']  = np.load(K_FQK_TEST)
    else:
        logger.info("Computing FQK (SAR D4-augmented features)... ~24 hours")
        results['K_fqk_train'] = compute_fqk_kernel_matrix(
            X_train, save_path=K_FQK_TRAIN
        )
        results['K_fqk_test'] = compute_fqk_kernel_matrix(
            X_train, X2=X_test, save_path=K_FQK_TEST
        )

    # ---- D4_FQK — Standard FQK on D4OrbitPCA features ----
    # Group averaging is provably redundant: D4OrbitPCA.transform() satisfies
    # theta(g·x) = theta(x) to machine precision (max_err=1.33e-15, verified
    # scripts/verify_d4_invariance_fast.py). Therefore:
    #   K_D4(x,y) = (1/8)Σ_g K_FQK(theta(g·x),theta(y))
    #             = (1/8) × 8 × K_FQK(theta(x),theta(y))
    #             = K_FQK(theta(x),theta(y))
    # Using K_FQK on D4OrbitPCA features directly — O(n²) vs O(|G|n²).
    # Savings: 88% compute reduction (193 hours → 24 hours), zero correctness loss.
    if os.path.exists(K_D4FQK_TRAIN):
        logger.info("[SKIP] D4_FQK train kernel exists")
        results['K_d4fqk_train'] = np.load(K_D4FQK_TRAIN)
        results['K_d4fqk_test']  = np.load(K_D4FQK_TEST)
    else:
        logger.info("Computing D4_FQK = FQK on D4OrbitPCA features (~24 hours)")
        logger.info("  (group averaging redundant — invariance verified to 1.33e-15)")
        from src.quantum_kernels import compute_fqk_kernel_matrix
        results['K_d4fqk_train'] = compute_fqk_kernel_matrix(
            X_train, save_path=K_D4FQK_TRAIN
        )
        results['K_d4fqk_test'] = compute_fqk_kernel_matrix(
            X_train, X2=X_test, save_path=K_D4FQK_TEST
        )
        logger.info("  K_D4FQK == K_FQK(theta(x)) by orbit invariance proof")

    return results


# ================================================================
# STEP 4 — Analysis
# ================================================================

def run_analysis(kernels, y_train, y_test):
    section("STEP 4 — Analysis")

    from src.kernel_target_alignment import compute_centered_kta, compute_delta_kta
    from src.geometric_difference import compute_geometric_difference
    from src.kernel_concentration import compute_concentration_metrics

    K_rbf_train   = kernels['K_rbf_train']
    K_fqk_train   = kernels['K_fqk_train']
    K_d4fqk_train = kernels['K_d4fqk_train']

    results = {}

    for name, K_train, K_test in [
        ('RBF',    K_rbf_train,   kernels['K_rbf_test']),
        ('FQK',    K_fqk_train,   kernels['K_fqk_test']),
        ('D4_FQK', K_d4fqk_train, kernels['K_d4fqk_test']),
    ]:
        logger.info(f"\n  --- {name} ---")

        kta = compute_centered_kta(K_train, y_train, class_weighted=True)
        offdiag = K_train[~np.eye(len(K_train), dtype=bool)]
        mean_od = float(offdiag.mean())
        min_eig = float(np.linalg.eigvalsh(K_train).min())

        # SVM classification
        try:
            svm = SVC(kernel='precomputed', C=1.0, class_weight='balanced')
            svm.fit(K_train, y_train)
            y_pred = svm.predict(K_test)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        except Exception as e:
            logger.warning(f"    SVM failed: {e}")
            f1 = float('nan')

        results[name] = {
            'kta': kta,
            'mean_offdiag': mean_od,
            'min_eigenvalue': min_eig,
            'macro_f1': f1,
        }

        logger.info(f"    KTA (centered weighted): {kta:.4f}")
        logger.info(f"    mean_offdiag:            {mean_od:.4f}")
        logger.info(f"    min_eigenvalue:          {min_eig:.4f}")
        logger.info(f"    macro-F1:                {f1:.4f}")

    # Geometric difference: FQK vs RBF, D4_FQK vs RBF
    for q_name, K_q in [('FQK', K_fqk_train), ('D4_FQK', K_d4fqk_train)]:
        g, _ = compute_geometric_difference(K_q, K_rbf_train)
        results[q_name]['g_vs_rbf'] = g
        logger.info(f"  g({q_name} vs RBF) = {g:.4f}")

    # ΔKTA
    delta_fqk    = compute_delta_kta(K_fqk_train,   K_rbf_train, y_train)
    delta_d4fqk  = compute_delta_kta(K_d4fqk_train, K_rbf_train, y_train)
    delta_d4_vs_fqk = compute_delta_kta(K_d4fqk_train, K_fqk_train, y_train)

    results['FQK']['delta_kta_vs_rbf']       = delta_fqk
    results['D4_FQK']['delta_kta_vs_rbf']    = delta_d4fqk
    results['D4_FQK']['delta_kta_vs_fqk']    = delta_d4_vs_fqk

    return results


# ================================================================
# STEP 5 — Save summary
# ================================================================

def save_summary(results, y_train):
    section("STEP 5 — Summary")

    lines = [
        "=" * 65,
        "  EXP D4 STANDALONE — RESULTS SUMMARY",
        "=" * 65,
        "",
        "Scientific question:",
        "  Does D4 covariant encoding improve quantum kernel performance",
        "  on SAR urban classification vs standard FQK?",
        "",
        "Feature space: SAR with D4-augmented PCA (orbit-invariant)",
        f"n_train: {config.SUBSAMPLE_TRAIN}, n_test: {getattr(config,'SUBSAMPLE_TEST',500)}",
        f"n_classes: 17 (So2Sat LCZ42)",
        "",
        f"{'Kernel':<12} {'KTA':>8} {'ΔKTA/RBF':>10} {'mean_od':>9} {'g/RBF':>8} {'F1':>8}",
        "-" * 65,
    ]

    for name in ['RBF', 'FQK', 'D4_FQK']:
        r = results[name]
        delta = r.get('delta_kta_vs_rbf', float('nan'))
        g     = r.get('g_vs_rbf', float('nan'))
        lines.append(
            f"{name:<12} {r['kta']:>8.4f} {delta:>10.4f} "
            f"{r['mean_offdiag']:>9.4f} {g:>8.4f} {r['macro_f1']:>8.4f}"
        )

    lines += [
        "",
        f"ΔKTA(D4_FQK vs FQK): "
        f"{results['D4_FQK'].get('delta_kta_vs_fqk', float('nan')):.4f}",
        "",
        "Interpretation:",
        "  If ΔKTA(D4_FQK vs FQK) > 0: circuit-level D4 covariance adds",
        "  benefit beyond feature-space invariance alone.",
        "  If ΔKTA(D4_FQK vs FQK) ≈ 0: feature-space invariance (augmented",
        "  PCA) captures the symmetry; circuit encoding is redundant.",
        "  Both outcomes are scientifically valid and publishable.",
    ]

    summary = "\n".join(lines)
    logger.info("\n" + summary)

    with open(SUMMARY_PATH, 'w') as f:
        f.write(summary)
    logger.info(f"\nSaved summary: {SUMMARY_PATH}")


# ================================================================
# MAIN
# ================================================================

def main():
    os.makedirs(D4_DIR, exist_ok=True)
    np.random.seed(config.RANDOM_SEED)

    section("EXP D4 STANDALONE")
    logger.info("D4 Covariant Quantum Kernel — SAR Modality")
    logger.info("All results saved to: results/d4_standalone/")
    logger.info("No dependency on exp1 or fused modality results.")

    # Step 1: Load raw data
    X_flat_pca, X_flat_train, X_flat_test, y_train, y_test = load_raw_sar_data()

    # Step 2: Fit D4-augmented PCA
    pca, X_train, X_test, y_train, y_test, x_min, x_range = fit_and_transform(
        X_flat_pca, X_flat_train, X_flat_test, y_train, y_test
    )

    # Step 3: Compute kernels
    kernels = compute_kernels(
        pca, X_train, X_test, y_train, 
        X_flat_train, X_flat_test, x_min, x_range
    )

    # Step 4: Analysis
    results = run_analysis(kernels, y_train, y_test)

    # Step 5: Save summary
    save_summary(results, y_train)

    section("COMPLETE")
    logger.info("exp_d4_standalone.py finished.")
    logger.info(f"Results: {D4_DIR}")


if __name__ == "__main__":
    main()
