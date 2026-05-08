import sys, os
import json
import logging
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from setup_data import find_local_h5_files, load_labels_only, \
                       load_h5_by_indices, get_stratified_indices
from extract_physics_features import extract_optical_indices, extract_sar_indices
from src.kernel_target_alignment import compute_centered_kta
from src.kernel_concentration import compute_concentration_metrics
from src.classifiers import train_precomputed_svm, evaluate_classifier
from src.classical_kernels import compute_rbf_kernel
from src.utils import setup_logging

logger = setup_logging(
    "compute_agpqk_qovk",
    log_file=os.path.join("results", "physics", "qovk", "qovk.log")
)

# ---- Constants ----
N_QUBITS    = 8
N_CLASSES   = 17
BLOCH_DIM   = 24
RANDOM_SEED = 42
N_VALUES    = [2000, 3500, 5000, 7000]
TEST_SIZE   = 2000

SELECTED_INDICES  = [0, 2, 5, 6, 11, 12, 13, 15]
FEATURE_NAMES_SEL = ["NDVI","NDWI","NDRE","MNDWI","PolCoh","VH_dB","VV_dB","NDVI_tex"]
GAMMA_PER_QUBIT   = np.array([0.5801, 0.8000, 0.5525, 0.8000,
                               0.2000, 0.4462, 0.3584, 0.2000])
ENTANGLEMENT_PAIRS = [(2, 5), (1, 4), (0, 7), (3, 6)]
OUTER_GAMMA        = 0.67
ZZ_REPS            = 2
EPS                = 1e-8

LCZ_CLASS_NAMES = [
    "Compact High Rise","Compact Mid Rise","Compact Low Rise",
    "Open High Rise","Open Mid Rise","Open Low Rise",
    "Lightweight Low Rise","Large Low Rise","Sparsely Built",
    "Heavy Industry","Dense Trees","Scattered Trees",
    "Bush/Scrub","Low Plants","Bare Rock/Paved",
    "Bare Soil/Sand","Water"
]

BASELINES_N2000 = {
    "RBF": 0.2956, "FQK": 0.2345, "PQK": 0.2329,
    "TFK": 0.1947, "AGPQK": 0.2665,
}

# ---- File Locations ----
PHYS16_PATH      = "data/processed/physics_features_16.npz"
AGPQK_CFG        = "results/physics/agpqk_config.json"
BLOCH_TRAIN_2000 = "results/physics/fused/bloch_agpqk_physics_train.npy"
BLOCH_TEST_2000  = "results/physics/fused/bloch_agpqk_physics_test.npy"
K_AGPQK_TRAIN    = "results/physics/fused/K_agpqk_physics_train.npy"
K_AGPQK_TEST     = "results/physics/fused/K_agpqk_physics_test.npy"
QOVK_DIR         = "results/physics/qovk"


def section(title):
    logger.info("=" * 65)
    logger.info(f"  {title}")
    logger.info("=" * 65)


# ---- Stage 1: Load AGPQK Config and Verify ----
def load_and_verify_config():
    with open(AGPQK_CFG, 'r') as f:
        cfg = json.load(f)
    if cfg["selected_feature_indices"] != SELECTED_INDICES:
        raise ValueError("SELECTED_INDICES mismatch")
    if not np.allclose(cfg["gamma_per_qubit"], GAMMA_PER_QUBIT, atol=1e-3):
        raise ValueError("GAMMA_PER_QUBIT mismatch")
    cfg_pairs = [tuple(p) for p in cfg["entanglement_pairs_qubit_space"]]
    if cfg_pairs != ENTANGLEMENT_PAIRS:
        raise ValueError("ENTANGLEMENT_PAIRS mismatch")
    logger.info("  AGPQK config verified.")


# ---- Stage 2: Data Loading Function ----
def extract_all_16_features(sar_patches, opt_patches):
    """
    Extract all 16 physics features from raw HDF5 patches.
    Returns: (N, 16) raw (unnormalized) features
    """
    N = len(sar_patches)

    opt_mean = opt_patches.mean(axis=(1, 2))
    blue  = opt_mean[:, 0]
    green = opt_mean[:, 1]
    red   = opt_mean[:, 2]
    vre1  = opt_mean[:, 3]
    nir   = opt_mean[:, 6]
    swir1 = opt_mean[:, 8]

    sar_mean = sar_patches.mean(axis=(1, 2))
    vh_lee = np.abs(sar_mean[:, 4]) + EPS
    vv_lee = np.abs(sar_mean[:, 5]) + EPS
    cov_re = sar_mean[:, 6]
    cov_im = sar_mean[:, 7]

    def safe_ratio(a, b):
        return (a - b) / (a + b + EPS)

    ndvi = safe_ratio(nir, red)
    ndbi = safe_ratio(swir1, nir)
    ndwi = safe_ratio(green, nir)
    bsi = safe_ratio(swir1 + red, nir + blue)
    savi = 1.5 * (nir - red) / (nir + red + 0.5 + EPS)
    ndre = safe_ratio(nir, vre1)
    mndwi = safe_ratio(green, swir1)
    evi = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1.0 + EPS)
    
    vv_vh = vv_lee / (vh_lee + EPS)
    sar_total = vh_lee + vv_lee
    crosspol = vh_lee / (vv_lee + EPS)
    
    cov_mag = np.sqrt(cov_re**2 + cov_im**2)
    polcoh = cov_mag / (np.sqrt(vv_lee * vh_lee) + EPS)
    
    vh_db = 10.0 * np.log10(vh_lee)
    vv_db = 10.0 * np.log10(vv_lee)
    
    vv_lee_2d = np.abs(sar_patches[:, :, :, 5])
    vv_tex = vv_lee_2d.std(axis=(1, 2))
    
    nir_2d = opt_patches[:, :, :, 6]
    red_2d = opt_patches[:, :, :, 2]
    ndvi_2d = (nir_2d - red_2d) / (nir_2d + red_2d + EPS)
    ndvi_tex = ndvi_2d.std(axis=(1, 2))

    return np.stack([
        ndvi, ndbi, ndwi, bsi, savi, ndre, mndwi, evi,
        vv_vh, sar_total, crosspol, polcoh, vh_db, vv_db, vv_tex, ndvi_tex
    ], axis=1).astype(np.float32)

def normalize_with_saved_params(X_raw_16, lo, hi, eps=1e-8):
    result = np.zeros_like(X_raw_16, dtype=np.float64)
    for i in range(16):
        if abs(hi[i] - lo[i]) < eps:
            result[:, i] = np.pi / 2
        else:
            scaled = np.clip((X_raw_16[:, i] - lo[i]) / (hi[i] - lo[i]), 0.0, 1.0)
            result[:, i] = scaled * np.pi
    return result

def load_data_for_n(n, phys16_data, h5_files, norm_lo, norm_hi):
    X_norm_base = phys16_data["X_train"][:, SELECTED_INDICES]
    y_base      = phys16_data["y_train"]
    X_norm_test = phys16_data["X_test"][:, SELECTED_INDICES]
    y_test      = phys16_data["y_test"]

    if n == 2000:
        X_enc      = X_norm_base * GAMMA_PER_QUBIT
        X_enc_test = X_norm_test * GAMMA_PER_QUBIT
        return X_enc, X_norm_base, y_base, X_enc_test, X_norm_test, y_test, None

    n_extra = n - 2000
    logger.info(f"  Loading {n_extra} additional samples from HDF5...")

    y_train_full = load_labels_only(h5_files["train"])
    n_total      = len(y_train_full)

    extra_idx  = get_stratified_indices(y_train_full, n_extra + 1000, seed=RANDOM_SEED + 1)
    
    # Deduplicate
    base_indices = set(phys16_data.get("train_indices", np.array([])).tolist())
    extra_idx = np.array([i for i in extra_idx if i not in base_indices])
    
    if len(extra_idx) < n_extra:
        raise ValueError("Not enough unique indices found for extra samples.")
        
    extra_idx  = extra_idx[:n_extra]
    y_extra    = y_train_full[extra_idx]

    sar_extra, opt_extra, _ = load_h5_by_indices(
        h5_files["train"], extra_idx, load_only="both"
    )

    X_raw_extra_16 = extract_all_16_features(sar_extra, opt_extra)
    del sar_extra, opt_extra

    X_norm_extra_16 = normalize_with_saved_params(X_raw_extra_16, norm_lo, norm_hi)
    X_norm_extra = X_norm_extra_16[:, SELECTED_INDICES]

    X_norm_all = np.vstack([X_norm_base, X_norm_extra])
    y_all      = np.concatenate([y_base, y_extra])

    rng  = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(n)
    X_norm_all = X_norm_all[perm]
    y_all      = y_all[perm]

    X_enc      = X_norm_all * GAMMA_PER_QUBIT
    X_enc_test = X_norm_test * GAMMA_PER_QUBIT

    logger.info(f"  Total: {n} train ({2000} from npz + {n_extra} from HDF5)")
    classes, counts = np.unique(y_all, return_counts=True)
    logger.info(f"  Class balance: min={counts.min()}, max={counts.max()}, "
          f"ratio={counts.max()/counts.min():.1f}:1")

    return X_enc, X_norm_all, y_all, X_enc_test, X_norm_test, y_test, perm


# ---- Stage 3: AGPQK Circuit ----
def apply_agpqk_encoding(x, entangle_pairs, n_qubits=8, reps=2):
    import pennylane as qml
    for _ in range(reps):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RZ(float(x[i]), wires=i)
        for (qi, qj) in entangle_pairs:
            zz_angle = (np.pi - float(x[qi])) * (np.pi - float(x[qj]))
            qml.CNOT(wires=[qi, qj])
            qml.RZ(zz_angle, wires=qj)
            qml.CNOT(wires=[qi, qj])

def compute_bloch_vectors(X_enc, entangle_pairs, cache_path=None, n_qubits=8, reps=2):
    import pennylane as qml
    from tqdm import tqdm

    if cache_path and os.path.exists(cache_path):
        logger.info(f"  [SKIP] Loading cached Bloch vectors: {cache_path}")
        return np.load(cache_path)

    dev = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def mX(x):
        apply_agpqk_encoding(x, entangle_pairs, n_qubits, reps)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def mY(x):
        apply_agpqk_encoding(x, entangle_pairs, n_qubits, reps)
        return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def mZ(x):
        apply_agpqk_encoding(x, entangle_pairs, n_qubits, reps)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    n = len(X_enc)
    bloch = np.zeros((n, 3 * n_qubits), dtype=np.float64)

    for i in tqdm(range(n), desc="AGPQK Bloch"):
        x = X_enc[i]
        bx = np.array(mX(x))
        by = np.array(mY(x))
        bz = np.array(mZ(x))
        for q in range(n_qubits):
            bloch[i, 3*q]   = bx[q]
            bloch[i, 3*q+1] = by[q]
            bloch[i, 3*q+2] = bz[q]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, bloch)
        logger.info(f"  Saved: {cache_path}")

    return bloch


# ---- Stage 4: Scalar AGPQK Kernel ----
def compute_scalar_kernel(bloch1, bloch2, gamma=0.67, n_qubits=8, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        logger.info(f"  [SKIP] Loading cached kernel: {cache_path}")
        return np.load(cache_path)

    n1 = len(bloch1)
    n2 = len(bloch2)
    B1 = bloch1.reshape(n1, n_qubits, 3)
    B2 = bloch2.reshape(n2, n_qubits, 3)
    K = np.zeros((n1, n2), dtype=np.float64)

    for i in range(n1):
        diff = B1[i] - B2              # (n2, n_qubits, 3)
        sq   = np.sum(diff**2, axis=2) # (n2, n_qubits)
        frob = 0.5 * np.sum(sq, axis=1)
        K[i] = np.exp(-gamma * frob)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, K)
    return K


# ---- Stage 5: Quantum Class Gram Matrix (A_out) ----
def compute_quantum_class_gram(bloch_train, y_train, gamma=0.67, n_qubits=8):
    n_classes = 17
    prototypes = np.zeros((n_classes, 3 * n_qubits))
    counts = np.zeros(n_classes, dtype=int)

    for c in range(n_classes):
        mask = y_train == c
        if mask.sum() > 0:
            prototypes[c] = bloch_train[mask].mean(axis=0)
            counts[c] = mask.sum()
        else:
            logger.warning(f"  WARNING: Class {c} ({LCZ_CLASS_NAMES[c]}) has 0 samples!")

    A_out = compute_scalar_kernel(prototypes, prototypes, gamma, n_qubits)

    eigvals = np.linalg.eigvalsh(A_out)
    if eigvals.min() < -1e-8:
        A_out += (-eigvals.min() + 1e-8) * np.eye(n_classes)
        logger.info(f"  A_out: clipped negative eigenvalue {eigvals.min():.2e}")

    logger.info(f"  A_out eigenvalues: min={eigvals.min():.4f}, max={eigvals.max():.4f}")
    logger.info(f"  A_out diagonal (class self-similarity):")
    for c in range(n_classes):
        bar = '█' * int(A_out[c,c] * 20)
        logger.info(f"    {LCZ_CLASS_NAMES[c]:<22}: {A_out[c,c]:.4f}  {bar}  (n={counts[c]})")

    return A_out, prototypes


# ---- Stage 6: Class Similarity Signatures ----
def compute_class_signatures(bloch_samples, prototypes, gamma=0.67, n_qubits=8):
    return compute_scalar_kernel(bloch_samples, prototypes, gamma, n_qubits)


# ---- Stage 7: QOVK Augmented Kernel ----
def compute_qovk_augmented(K_scalar_train, K_scalar_test,
                            sigs_train, sigs_test, A_out, beta):
    coupling_train = sigs_train @ A_out @ sigs_train.T  # (n, n)
    coupling_test  = sigs_test  @ A_out @ sigs_train.T  # (n_test, n)

    K_aug_train = K_scalar_train + beta * coupling_train
    K_aug_test  = K_scalar_test  + beta * coupling_test

    d = np.sqrt(np.diag(K_aug_train))
    d = np.maximum(d, 1e-10)
    K_aug_train_norm = K_aug_train / np.outer(d, d)
    K_aug_test_norm  = K_aug_test  / d[np.newaxis, :]

    K_aug_train_norm = np.clip(K_aug_train_norm, -1, 1)
    K_aug_test_norm  = np.clip(K_aug_test_norm, -1, 1)

    return K_aug_train_norm, K_aug_test_norm


# ---- Stage 8: Beta CV Selection ----
def select_beta_cv(K_scalar_train, sigs_train, A_out, y_train, beta_candidates):
    n = len(y_train)
    fold_size = n // 3
    best_beta = 0.5
    best_kta  = -np.inf

    for beta in beta_candidates:
        fold_ktas = []
        for fold in range(3):
            v_start = fold * fold_size
            v_end   = (fold + 1) * fold_size
            v_idx   = np.arange(v_start, v_end)

            K_fold_aug, _ = compute_qovk_augmented(
                K_scalar_train[np.ix_(v_idx, v_idx)],
                K_scalar_train[np.ix_(v_idx, v_idx)],
                sigs_train[v_idx], sigs_train[v_idx], A_out, beta
            )
            kta = compute_centered_kta(K_fold_aug, y_train[v_idx],
                                        class_weighted=True)
            fold_ktas.append(kta)

        mean_kta = float(np.mean(fold_ktas))
        logger.info(f"    beta={beta:.2f}  cv_KTA={mean_kta:.4f}")

        if mean_kta > best_kta:
            best_kta  = mean_kta
            best_beta = beta

    logger.info(f"  Best beta={best_beta}  (cv_KTA={best_kta:.4f})")
    return best_beta


# ---- Main Pipeline ----
def main():
    os.makedirs(QOVK_DIR, exist_ok=True)
    section("AGPQK-QOVK Scaling Experiment")

    load_and_verify_config()

    phys16_data = np.load(PHYS16_PATH)
    norm_lo = phys16_data["normalization_lo"]
    norm_hi = phys16_data["normalization_hi"]
    
    h5_files = find_local_h5_files()
    if "train" not in h5_files:
        raise FileNotFoundError("So2Sat train HDF5 not found!")

    results_by_n = {}

    for n in N_VALUES:
        section(f"Evaluating n = {n}")
        n_dir = os.path.join(QOVK_DIR, f"n{n}")
        os.makedirs(n_dir, exist_ok=True)

        # 1. Load Data
        X_enc, X_norm, y_train, X_enc_test, X_norm_test, y_test, perm = load_data_for_n(
            n, phys16_data, h5_files, norm_lo, norm_hi)

        # 2. Compute Bloch Vectors
        bloch_test_cache = os.path.join(QOVK_DIR, "n2000", "bloch_test.npy")
        if n == 2000:
            import shutil
            if os.path.exists(BLOCH_TRAIN_2000) and not os.path.exists(os.path.join(n_dir, "bloch_train.npy")):
                shutil.copy(BLOCH_TRAIN_2000, os.path.join(n_dir, "bloch_train.npy"))
            if os.path.exists(BLOCH_TEST_2000) and not os.path.exists(bloch_test_cache):
                shutil.copy(BLOCH_TEST_2000, bloch_test_cache)
        
        bloch_train_cache = os.path.join(n_dir, "bloch_train.npy")
        
        if n == 2000:
            bloch_train = compute_bloch_vectors(X_enc, ENTANGLEMENT_PAIRS, bloch_train_cache)
        else:
            if os.path.exists(bloch_train_cache):
                bloch_train = np.load(bloch_train_cache)
            else:
                bloch_base = np.load(os.path.join(QOVK_DIR, "n2000", "bloch_train.npy"))
                X_enc_extra = X_enc[2000:]
                bloch_extra = compute_bloch_vectors(X_enc_extra, ENTANGLEMENT_PAIRS, None)
                bloch_train_unshuffled = np.vstack([bloch_base, bloch_extra])
                bloch_train = bloch_train_unshuffled[perm]
                np.save(bloch_train_cache, bloch_train)
                
        bloch_test = compute_bloch_vectors(X_enc_test, ENTANGLEMENT_PAIRS, bloch_test_cache)

        # 3. Scalar Kernel
        k_scalar_train_cache = os.path.join(n_dir, "K_scalar_train.npy")
        k_scalar_test_cache  = os.path.join(n_dir, "K_scalar_test.npy")
        
        if n == 2000:
            import shutil
            if os.path.exists(K_AGPQK_TRAIN) and not os.path.exists(k_scalar_train_cache):
                shutil.copy(K_AGPQK_TRAIN, k_scalar_train_cache)
            if os.path.exists(K_AGPQK_TEST) and not os.path.exists(k_scalar_test_cache):
                shutil.copy(K_AGPQK_TEST, k_scalar_test_cache)

        K_scalar_train = compute_scalar_kernel(bloch_train, bloch_train, OUTER_GAMMA, N_QUBITS, k_scalar_train_cache)
        K_scalar_test  = compute_scalar_kernel(bloch_test, bloch_train, OUTER_GAMMA, N_QUBITS, k_scalar_test_cache)

        # 4. A_out and Signatures
        A_out, prototypes = compute_quantum_class_gram(bloch_train, y_train, OUTER_GAMMA, N_QUBITS)
        sigs_train = compute_class_signatures(bloch_train, prototypes, OUTER_GAMMA, N_QUBITS)
        sigs_test  = compute_class_signatures(bloch_test, prototypes, OUTER_GAMMA, N_QUBITS)

        # 5. Beta CV
        beta_cands = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
        best_beta = select_beta_cv(K_scalar_train, sigs_train, A_out, y_train, beta_cands)

        # 6. Augmented Kernel
        K_aug_train, K_aug_test = compute_qovk_augmented(
            K_scalar_train, K_scalar_test, sigs_train, sigs_test, A_out, best_beta)

        # 7. RBF Kernel
        K_rbf_train = compute_rbf_kernel(X_norm)
        K_rbf_test  = compute_rbf_kernel(X_norm_test, X_norm)

        # 8. Evaluate all 3
        res = {}
        for k_name, K_tr, K_te in [
            ("scalar_agpqk", K_scalar_train, K_scalar_test),
            ("qovk_augmented", K_aug_train, K_aug_test),
            ("rbf", K_rbf_train, K_rbf_test)
        ]:
            kta = float(compute_centered_kta(K_tr, y_train, class_weighted=True))
            conc = compute_concentration_metrics(K_tr)
            
            try:
                clf = train_precomputed_svm(K_tr, y_train)
                metrics = evaluate_classifier(clf, K_te, y_test, k_name)
                macro_f1 = float(metrics['macro_f1'])
            except Exception as e:
                logger.error(f"SVM failed for {k_name}: {e}")
                macro_f1 = float('nan')
                
            res[k_name] = {
                "kta": kta,
                "macro_f1": macro_f1,
                "eff_rank": float(conc["effective_rank"])
            }
            if k_name == "qovk_augmented":
                res[k_name]["beta"] = float(best_beta)
                
        # Include A_out eigenvalues
        res["a_out_eigenvalues"] = np.linalg.eigvalsh(A_out).tolist()
        
        counts = np.bincount(y_train, minlength=17)
        res["n_per_class"] = {LCZ_CLASS_NAMES[c]: int(counts[c]) for c in range(17)}
        
        results_by_n[str(n)] = res

    # ---- Summary Document ----
    section("Crossover Analysis")
    crossover_n = None
    for n in N_VALUES:
        q_kta = results_by_n[str(n)]['qovk_augmented']['kta']
        r_kta = results_by_n[str(n)]['rbf']['kta']
        delta = q_kta - r_kta
        logger.info(f"n={n}: QOVK={q_kta:.4f}, RBF={r_kta:.4f}, Δ={delta:+.4f}")
        if delta > 0 and crossover_n is None:
            logger.info(f"  ★ CROSSOVER AT n={n}: QOVK beats RBF!")
            crossover_n = n

    print("\n========================================================================")
    print("  AGPQK-QOVK: Sample Scaling + Operator-Valued Kernel")
    print("  Physics Features | Real HDF5 data | RTX 4050")
    print("========================================================================")
    print("\n  KTA (centered weighted, class-balanced):")
    print("  n       AGPQK-scalar  QOVK-augmented  RBF(same n)  QOVK beats RBF?")
    print("  ----    ------------  --------------  -----------  ---------------")
    for n in N_VALUES:
        r = results_by_n[str(n)]
        s_kta = r['scalar_agpqk']['kta']
        q_kta = r['qovk_augmented']['kta']
        r_kta = r['rbf']['kta']
        beats = "YES" if q_kta > r_kta else "NO"
        print(f"  {n:<6}  {s_kta:.4f}        {q_kta:.4f}          {r_kta:.4f}       {beats}")

    print("\n  Macro-F1:")
    print("  n       AGPQK-scalar  QOVK-augmented  RBF(same n)")
    print("  ----    ------------  --------------  -----------")
    for n in N_VALUES:
        r = results_by_n[str(n)]
        s_f1 = r['scalar_agpqk']['macro_f1']
        q_f1 = r['qovk_augmented']['macro_f1']
        r_f1 = r['rbf']['macro_f1']
        print(f"  {n:<6}  {s_f1:.4f}        {q_f1:.4f}          {r_f1:.4f}")

    print("\n  Effective rank:")
    print("  n       AGPQK-scalar  QOVK-augmented  RBF(same n)")
    for n in N_VALUES:
        r = results_by_n[str(n)]
        s_er = r['scalar_agpqk']['eff_rank']
        q_er = r['qovk_augmented']['eff_rank']
        r_er = r['rbf']['eff_rank']
        print(f"  {n:<4}    {s_er:.2f}         {q_er:.2f}           {r_er:.2f}")

    print(f"\n  Theoretical crossover prediction: n≈6800 (scalar), n≈2000 (QOVK)")
    print(f"  Observed crossover: n={crossover_n if crossover_n else 'NONE'} for QOVK-augmented")
    print("========================================================================")

    # Save JSON
    summary_data = {
        "n_values": N_VALUES,
        "results_by_n": results_by_n,
        "crossover_n": crossover_n,
        "theoretical_crossover_scalar": 6800,
        "theoretical_crossover_qovk": 2000,
        "prediction_validated": crossover_n is not None,
        "data_source": "real HDF5 for n>2000, no bootstrap",
        "baselines_n2000": BASELINES_N2000
    }
    with open(os.path.join(QOVK_DIR, "qovk_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

    # Save CSV
    import csv
    csv_path = os.path.join(QOVK_DIR, "qovk_summary.csv")
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["n", "kernel", "kta", "macro_f1", "eff_rank", "vs_rbf_delta", "beats_rbf"])
        for n in N_VALUES:
            r = results_by_n[str(n)]
            
            vs_rbf_s = r['scalar_agpqk']['kta'] - r['rbf']['kta']
            writer.writerow([n, "scalar_agpqk", r['scalar_agpqk']['kta'], r['scalar_agpqk']['macro_f1'], 
                             r['scalar_agpqk']['eff_rank'], vs_rbf_s, vs_rbf_s > 0])
                             
            vs_rbf_q = r['qovk_augmented']['kta'] - r['rbf']['kta']
            writer.writerow([n, "qovk_augmented", r['qovk_augmented']['kta'], r['qovk_augmented']['macro_f1'], 
                             r['qovk_augmented']['eff_rank'], vs_rbf_q, vs_rbf_q > 0])
                             
            writer.writerow([n, "rbf", r['rbf']['kta'], r['rbf']['macro_f1'], 
                             r['rbf']['eff_rank'], 0.0, False])

if __name__ == "__main__":
    main()
