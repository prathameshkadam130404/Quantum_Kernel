import os
import sys
import json
import logging
import numpy as np
from typing import Dict, Any

# Ensure we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import config
from src.kernel_target_alignment import compute_centered_kta
from src.kernel_concentration import compute_concentration_metrics
from src.classifiers import train_precomputed_svm, evaluate_classifier
from src.utils import setup_logging, ensure_dir

# Set up logging
logger = setup_logging(
    "compute_pqkme",
    log_file=os.path.join("results", "physics", "pqkme", "pqkme.log")
)

# ---- Confirmed constants ----
N_QUBITS   = 8
N_CLASSES  = 17
N_TRAIN    = 2000
N_TEST     = 2000
BLOCH_DIM  = 24   # 8 qubits * 3 (X, Y, Z per qubit)
RANDOM_SEED = 42

LCZ_CLASS_NAMES = [
    "Compact High Rise",    # 0
    "Compact Mid Rise",     # 1
    "Compact Low Rise",     # 2
    "Open High Rise",       # 3
    "Open Mid Rise",        # 4
    "Open Low Rise",        # 5
    "Lightweight Low Rise", # 6
    "Large Low Rise",       # 7
    "Sparsely Built",       # 8
    "Heavy Industry",       # 9
    "Dense Trees",          # 10
    "Scattered Trees",      # 11
    "Bush/Scrub",           # 12
    "Low Plants",           # 13
    "Bare Rock/Paved",      # 14
    "Bare Soil/Sand",       # 15
    "Water",                # 16
]

BASELINES = {
    "RBF":         0.2956,
    "FQK":         0.2345,
    "PQK":         0.2329,
    "TFK":         0.1947,
    "AGPQK":       0.2665,
}

# ---- File locations ----
BLOCH_TRAIN = os.path.join("results", "physics", "fused", "bloch_agpqk_physics_train.npy")
BLOCH_TEST  = os.path.join("results", "physics", "fused", "bloch_agpqk_physics_test.npy")
PHYS16_PATH = os.path.join("data", "processed", "physics_features_16.npz")
AGPQK_CFG = os.path.join("results", "physics", "agpqk_config.json")

OUT_DIR = os.path.join("results", "physics", "pqkme")
ensure_dir(OUT_DIR)

K_PQKME_TRAIN = os.path.join(OUT_DIR, "K_pqkme_train.npy")
K_PQKME_TEST  = os.path.join(OUT_DIR, "K_pqkme_test.npy")
K_HS_TRAIN    = os.path.join(OUT_DIR, "K_hs_train.npy")
K_HS_TEST     = os.path.join(OUT_DIR, "K_hs_test.npy")
K_HYBRID_TRAIN = os.path.join(OUT_DIR, "K_hybrid_train.npy")
K_HYBRID_TEST  = os.path.join(OUT_DIR, "K_hybrid_test.npy")

RESULTS_JSON = os.path.join(OUT_DIR, "pqkme_results.json")
PROTO_PATH   = os.path.join(OUT_DIR, "class_prototypes.npy")
SIGS_TRAIN   = os.path.join(OUT_DIR, "signatures_train.npy")
SIGS_TEST    = os.path.join(OUT_DIR, "signatures_test.npy")

def section(title):
    logger.info("=" * 65)
    logger.info(f"  {title}")
    logger.info("=" * 65)

def compute_signatures(B_samples, PROTO, n_qubits=N_QUBITS, n_classes=N_CLASSES):
    """
    Compute class signatures for all samples.
    """
    dots = np.einsum('nqi,cqi->ncq', B_samples, PROTO)  # (n, C, Q)
    hs_per_qubit = (1.0 + dots) / 2.0  # (n, C, Q), each in [0, 1]
    sigs = np.prod(hs_per_qubit, axis=2)  # (n, C)
    return sigs

def compute_pqkme_kernel(sigs1, sigs2, weights):
    """
    Weighted cosine similarity of class signature vectors.
    """
    w_sqrt = np.sqrt(weights)
    S1 = sigs1 * w_sqrt[np.newaxis, :]   # (n1, 17)
    S2 = sigs2 * w_sqrt[np.newaxis, :]   # (n2, 17)
    
    norms1 = np.linalg.norm(S1, axis=1, keepdims=True)  # (n1, 1)
    norms2 = np.linalg.norm(S2, axis=1, keepdims=True)  # (n2, 1)
    
    S1_norm = S1 / (norms1 + 1e-12)
    S2_norm = S2 / (norms2 + 1e-12)
    
    K = S1_norm @ S2_norm.T   # (n1, n2)
    return K

def compute_hs_kernel(B1, B2, n_qubits=N_QUBITS):
    """
    Product Hilbert-Schmidt kernel across all qubits.
    """
    dots = np.einsum('iql,jql->ijq', B1, B2)   # (n1, n2, N_QUBITS)
    hs_per_qubit = (1.0 + dots) / 2.0           # (n1, n2, N_QUBITS)
    K = np.prod(hs_per_qubit, axis=2)            # (n1, n2)
    return K

def run_diagnostics_and_svm(kernel_name, K_train, K_test, y_train, y_test):
    """Compute concentration metrics, KTA, and train SVM."""
    logger.info(f"--- Diagnostics for {kernel_name} ---")
    
    # 1. Concentration metrics (diagonal, off-diag, CV, eff rank, min eig)
    conc = compute_concentration_metrics(K_train)
    logger.info(f"  Off-diag mean:   {conc['mean_offdiag']:.4f}")
    logger.info(f"  Off-diag std:    {conc['std_offdiag']:.4f}")
    logger.info(f"  CV:              {conc['cv']:.4f}")
    logger.info(f"  Effective Rank:  {conc['effective_rank']:.4f}")
    logger.info(f"  Min Eigenvalue:  {conc['eigenvalue_spectrum'][-1]:g}")
    
    # 2. KTA
    kta = compute_centered_kta(K_train, y_train, class_weighted=True)
    logger.info(f"  KTA (weighted):  {kta:.4f}")
    
    # 3. SVM
    try:
        clf = train_precomputed_svm(K_train, y_train)
        metrics = evaluate_classifier(clf, K_test, y_test, kernel_name)
        macro_f1 = metrics['macro_f1']
        logger.info(f"  macro_F1:        {macro_f1:.4f}")
    except Exception as e:
        logger.error(f"  SVM training failed: {e}")
        macro_f1 = float('nan')
        
    conc['eigenvalue_spectrum'] = conc['eigenvalue_spectrum'].tolist()
    
    return {
        "kta": float(kta),
        "macro_f1": float(macro_f1),
        "concentration": conc
    }

def main():
    np.random.seed(RANDOM_SEED)

    # --- Stage 1: Load Bloch vectors and labels ---
    section("Stage 1: Load Data")
    
    if not os.path.exists(BLOCH_TRAIN) or not os.path.exists(BLOCH_TEST):
        logger.error(f"Bloch vectors not found at {BLOCH_TRAIN}")
        return
        
    bloch_train = np.load(BLOCH_TRAIN)
    bloch_test  = np.load(BLOCH_TEST)
    data = np.load(PHYS16_PATH)
    y_train = data["y_train"]
    y_test  = data["y_test"]
    
    B_train = bloch_train.reshape(N_TRAIN, N_QUBITS, 3)
    B_test  = bloch_test.reshape(N_TEST, N_QUBITS, 3)
    
    norms = np.linalg.norm(B_train, axis=2)
    assert norms.max() <= 1.0 + 1e-6, f"Bloch norm violation: max={norms.max():.6f}"
    logger.info(f"Loaded Bloch vectors. Train norms in [{norms.min():.4f}, {norms.max():.4f}]")
    
    # --- Stage 2: Compute class prototypes ---
    section("Stage 2: Compute Prototypes")
    
    classes, counts = np.unique(y_train, return_counts=True)
    n_c = dict(zip(classes, counts))
    w_c = {c: N_TRAIN / (N_CLASSES * n_c.get(c, 1)) for c in range(N_CLASSES)}
    weights = np.array([w_c[c] for c in range(N_CLASSES)])
    weights /= weights.sum()
    
    PROTO = np.zeros((N_CLASSES, N_QUBITS, 3), dtype=np.float64)
    for c in range(N_CLASSES):
        mask = y_train == c
        if mask.sum() > 0:
            PROTO[c] = B_train[mask].mean(axis=0)
            
    np.save(PROTO_PATH, PROTO.reshape(N_CLASSES, BLOCH_DIM))
    proto_norms = np.linalg.norm(PROTO, axis=2)
    logger.info(f"Prototype norms: min={proto_norms.min():.4f}, max={proto_norms.max():.4f}")
    
    # --- Stage 3: Compute signatures ---
    section("Stage 3: Compute Signatures")
    
    sigs_train = compute_signatures(B_train, PROTO)
    sigs_test  = compute_signatures(B_test, PROTO)
    
    np.save(SIGS_TRAIN, sigs_train)
    np.save(SIGS_TEST, sigs_test)
    
    logger.info(f"Signature stats: min={sigs_train.min():.6f}, max={sigs_train.max():.6f}")
    logger.info("Mean signature per class:")
    for c in range(N_CLASSES):
        mask = y_train == c
        if mask.sum() == 0: continue
        mean_s = sigs_train[mask].mean(axis=0)
        top_class = np.argmax(mean_s)
        logger.info(f"  {LCZ_CLASS_NAMES[c]:<22} top_response=LCZ_{top_class} ({LCZ_CLASS_NAMES[top_class]}), n={mask.sum()}")
        
    # --- Stage 4: Compute PQKME kernel ---
    section("Stage 4: PQKME Kernel")
    K_pqkme_train = compute_pqkme_kernel(sigs_train, sigs_train, weights)
    K_pqkme_test  = compute_pqkme_kernel(sigs_test, sigs_train, weights)
    
    np.save(K_PQKME_TRAIN, K_pqkme_train)
    np.save(K_PQKME_TEST, K_pqkme_test)
    logger.info("Saved PQKME kernels.")
    
    # --- Stage 5: Compute HS kernel ---
    section("Stage 5: HS Kernel")
    K_hs_train = compute_hs_kernel(B_train, B_train)
    K_hs_test  = compute_hs_kernel(B_test, B_train)
    
    np.save(K_HS_TRAIN, K_hs_train)
    np.save(K_HS_TEST, K_hs_test)
    logger.info("Saved HS kernels.")
    
    # --- Stage 6: Hybrid Kernel (CV) ---
    section("Stage 6: Hybrid Kernel CV")
    ALPHA_CANDIDATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_results = {}
    fold_size = N_TRAIN // 3
    
    for alpha in ALPHA_CANDIDATES:
        fold_ktas = []
        for fold in range(3):
            val_start = fold * fold_size
            val_end   = (fold + 1) * fold_size
            val_idx   = np.arange(val_start, val_end)
            
            K_hybrid_fold = (alpha * K_pqkme_train[np.ix_(val_idx, val_idx)] +
                            (1 - alpha) * K_hs_train[np.ix_(val_idx, val_idx)])
            
            kta_fold = compute_centered_kta(K_hybrid_fold, y_train[val_idx], class_weighted=True)
            fold_ktas.append(kta_fold)
            
        alpha_results[alpha] = float(np.mean(fold_ktas))
        logger.info(f"  alpha={alpha:.1f}  cv_KTA={alpha_results[alpha]:.4f}")
        
    best_alpha = max(alpha_results, key=alpha_results.get)
    logger.info(f"Best alpha = {best_alpha:.1f} (cv_KTA={alpha_results[best_alpha]:.4f})")
    
    K_hybrid_train = best_alpha * K_pqkme_train + (1 - best_alpha) * K_hs_train
    K_hybrid_test  = best_alpha * K_pqkme_test  + (1 - best_alpha) * K_hs_test
    
    np.save(K_HYBRID_TRAIN, K_hybrid_train)
    np.save(K_HYBRID_TEST, K_hybrid_test)
    logger.info("Saved Hybrid kernels.")
    
    # --- Stage 7 & 8: Diagnostics & SVM ---
    section("Stage 7 & 8: Diagnostics & SVM")
    
    res_hs = run_diagnostics_and_svm("K_HS", K_hs_train, K_hs_test, y_train, y_test)
    res_pqkme = run_diagnostics_and_svm("K_PQKME", K_pqkme_train, K_pqkme_test, y_train, y_test)
    res_hybrid = run_diagnostics_and_svm(f"K_Hybrid(alpha={best_alpha:.1f})", K_hybrid_train, K_hybrid_test, y_train, y_test)
    
    # --- Stage 9: Output and Save ---
    section("Stage 9: Results")
    
    print("\n=================================================================")
    print("  PQKME Results — Physics Features (n=2000)")
    print("=================================================================")
    print("  Kernel              KTA      vs RBF   macro_F1   Note")
    print("  ---------------------------------------------------------")
    print(f"  RBF               {BASELINES['RBF']:.4f}   +0.0000    -        classical")
    print(f"  AGPQK (fixed)     {BASELINES['AGPQK']:.4f}   {BASELINES['AGPQK'] - BASELINES['RBF']:+.4f}    -        best quantum so far")
    print(f"  K_HS (product)    {res_hs['kta']:.4f}   {res_hs['kta'] - BASELINES['RBF']:+.4f}     {res_hs['macro_f1']:.4f}   quantum-native HS")
    print(f"  K_PQKME           {res_pqkme['kta']:.4f}   {res_pqkme['kta'] - BASELINES['RBF']:+.4f}     {res_pqkme['macro_f1']:.4f}   class-prototype HS")
    print(f"  K_hybrid (α={best_alpha:.1f})  {res_hybrid['kta']:.4f}   {res_hybrid['kta'] - BASELINES['RBF']:+.4f}     {res_hybrid['macro_f1']:.4f}   PQKME + HS")
    print("  =================================================================\n")
    
    results_json = {
        "K_hs": res_hs,
        "K_pqkme": res_pqkme,
        "K_hybrid": res_hybrid,
        "hybrid_best_alpha": best_alpha,
        "alpha_cv_results": alpha_results,
        "class_prototype_norms": proto_norms.tolist(),
        "signature_stats": {"min": float(sigs_train.min()), "max": float(sigs_train.max()), "mean": float(sigs_train.mean())},
        "baselines": BASELINES,
        "bloch_source": os.path.basename(BLOCH_TRAIN),
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "n_classes": N_CLASSES,
        "n_qubits": N_QUBITS
    }
    
    with open(RESULTS_JSON, "w") as f:
        json.dump(results_json, f, indent=4)
    logger.info(f"Saved results to {RESULTS_JSON}")

if __name__ == "__main__":
    main()
