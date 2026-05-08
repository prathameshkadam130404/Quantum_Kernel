"""
Experiment 10 V2 (Redesigned): Controlled Mutual Information.

Design:
  - N=300 train, N=300 test (top-4 PCA per modality → 8 features for 8 qubits)
  - 3 kernels: FQK (concentrated), PQK (anti-concentration), AGPQK (per-qubit bandwidth)
  - 5 noise seeds (42-46)
  - 11 alpha values: [0.0, 0.1, ..., 1.0]
  - Permutation-based noise (preserves marginals, destroys cross-modal structure)
  - 5-fold CV at each (alpha, seed) combination
  - Global kernel computed ONCE per (alpha, seed), sliced for CV folds (12x speedup)

N=300 matches Bowles et al. (2024, N=250) and exceeds Hubregtsen et al.
(2022, N=30-60). 5 seeds provide ~0.95 statistical power for Spearman
trend test. Total: 55 evaluation points (11 alphas × 5 seeds).

Output:
    results/controlled_mi_v2/exp10_v2_results.json
    results/controlled_mi_v2/summary_fqk.csv
    results/controlled_mi_v2/summary_pqk.csv
    results/controlled_mi_v2/summary_agpqk.csv

Kaggle GPU estimate: ~7 hours per kernel (fits within 9-hour session).

Usage:
  python experiments/exp10_controlled_mi_v2.py              # all kernels
  python experiments/exp10_controlled_mi_v2.py --kernel fqk  # FQK only
  python experiments/exp10_controlled_mi_v2.py --kernel pqk  # PQK only
  python experiments/exp10_controlled_mi_v2.py --kernel agpqk  # AGPQK only
  python experiments/exp10_controlled_mi_v2.py --force       # rerun
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import config
from src.quantum_kernels import (
    compute_fqk_kernel_matrix,
    compute_pqk_kernel_matrix,
    get_top_mi_pairs,
    compute_crossmodal_fqk_kernel_matrix,
)
from src.classical_kernels import compute_rbf_kernel, compute_tensor_product_kernel
from src.geometric_difference import compute_geometric_difference
from src.kernel_target_alignment import compute_centered_kta
from src.geometric_bound import compute_cross_modal_mutual_information
from src.data_loader import load_subsample
from src.classifiers import train_precomputed_svm, evaluate_classifier
from src.feature_extraction import normalize_to_pi
from src.utils import (
    save_results,
    setup_logging,
    Timer,
    ensure_dir,
    compute_full_metrics,
)

logger = setup_logging(
    "exp10_v2",
    log_file=os.path.join(config.RESULTS_DIR, "controlled_mi_v2", "exp10_v2.log"),
)

RESULTS_DIR = ensure_dir(os.path.join(config.RESULTS_DIR, "controlled_mi_v2"))
PLOTS_DIR = ensure_dir(os.path.join(RESULTS_DIR, "plots"))

N_TRAIN = config.EXP10_V2_N_TRAIN
N_TEST = config.EXP10_V2_N_TEST
ALPHAS = config.EXP10_V2_ALPHAS
NOISE_SEEDS = config.EXP10_V2_NOISE_SEEDS
N_FOLDS = config.CV_FOLDS


def load_data():
    """Load genuinely separate SAR and optical PCA features."""
    subsample = load_subsample()
    X_sar_full = subsample["sar_X_train"][:N_TRAIN, :4]
    X_opt_full = subsample["optical_X_train"][:N_TRAIN, :4]
    y_train = subsample["y_train"][:N_TRAIN]
    X_sar_test = subsample["sar_X_test"][:N_TEST, :4]
    X_opt_test = subsample["optical_X_test"][:N_TEST, :4]
    y_test = subsample["y_test"][:N_TEST]
    return X_sar_full, X_opt_full, y_train, X_sar_test, X_opt_test, y_test


def create_mixed_data(
    X_sar, X_opt_real, X_sar_test, X_opt_test_real, alpha, noise_seed
):
    """Create mixed optical data with permutation-based noise.

    Permutation noise: shuffle each feature column independently.
    This preserves marginal distributions but destroys cross-feature
    and cross-modal structure.
    """
    rng = np.random.RandomState(noise_seed)

    # Permutation noise: shuffle each column independently
    X_noise = np.zeros_like(X_opt_real)
    for col in range(X_opt_real.shape[1]):
        X_noise[:, col] = rng.permutation(X_opt_real[:, col])

    # Mix: alpha * real + (1-alpha) * noise
    X_opt_mixed = alpha * X_opt_real + (1 - alpha) * X_noise
    X_opt_mixed, _, _ = normalize_to_pi(X_opt_mixed)

    X_fused = np.hstack([X_sar, X_opt_mixed])

    # Same for test
    X_noise_test = np.zeros_like(X_opt_test_real)
    for col in range(X_opt_test_real.shape[1]):
        X_noise_test[:, col] = rng.permutation(X_opt_test_real[:, col])

    X_opt_test_mixed = alpha * X_opt_test_real + (1 - alpha) * X_noise_test
    X_opt_test_mixed, _, _ = normalize_to_pi(X_opt_test_mixed)
    X_fused_test = np.hstack([X_sar_test, X_opt_test_mixed])

    return X_fused, X_fused_test


def compute_full_kernel_cached(X_fused, kernel_type, combo_dir):
    """
    OPTIMIZATION: Compute full NxN kernel ONCE per alpha/seed combintion.
    Returns the global kernel matrix which can be mathematically sliced for cross-validation.
    """
    K_full_path = os.path.join(combo_dir, f"K_{kernel_type}_full_train.npy")
    if os.path.exists(K_full_path):
        return np.load(K_full_path)

    if kernel_type == "fqk":
        K_full = compute_fqk_kernel_matrix(X_fused, save_path=K_full_path)
    elif kernel_type == "pqk":
        bloch_path = os.path.join(combo_dir, f"bloch_pqk_full_train.npy")
        K_full = compute_pqk_kernel_matrix(
            X_fused,
            save_path=K_full_path,
            bloch_save_path=bloch_path,
        )
    elif kernel_type == "agpqk":
        import pennylane as qml

        agpqk_cfg_path = os.path.join(
            config.RESULTS_DIR, "physics", "agpqk_config.json"
        )
        if os.path.exists(agpqk_cfg_path):
            with open(agpqk_cfg_path) as f:
                agpqk_cfg = json.load(f)
            gamma_per_qubit = np.array(agpqk_cfg["gamma_per_qubit"])
            entangle_pairs = [
                tuple(p) for p in agpqk_cfg["entanglement_pairs_qubit_space"]
            ]
        else:
            gamma_per_qubit = np.array([0.5] * config.N_QUBITS)
            entangle_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]

        X_fused_enc = X_fused * gamma_per_qubit[np.newaxis, :]
        dev = config.get_device(config.N_QUBITS)

        def apply_agpqk(x, n_q=config.N_QUBITS, reps=config.ZZ_REPS):
            for _ in range(reps):
                for i in range(n_q):
                    qml.Hadamard(wires=i)
                for i in range(n_q):
                    qml.RZ(x[i], wires=i)
                for qi, qj in entangle_pairs:
                    if qi < n_q and qj < n_q:
                        zz = (np.pi - x[qi]) * (np.pi - x[qj])
                        qml.CNOT(wires=[qi, qj])
                        qml.RZ(zz, wires=qj)
                        qml.CNOT(wires=[qi, qj])

        @qml.qnode(dev, diff_method=None)
        def mx(x):
            apply_agpqk(x)
            return [qml.expval(qml.PauliX(i)) for i in range(config.N_QUBITS)]

        @qml.qnode(dev, diff_method=None)
        def my(x):
            apply_agpqk(x)
            return [qml.expval(qml.PauliY(i)) for i in range(config.N_QUBITS)]

        @qml.qnode(dev, diff_method=None)
        def mz(x):
            apply_agpqk(x)
            return [qml.expval(qml.PauliZ(i)) for i in range(config.N_QUBITS)]

        def compute_bloch(X_enc):
            n = len(X_enc)
            b = np.zeros((n, 3 * config.N_QUBITS))
            for i in range(n):
                ex = np.array(mx(X_enc[i]))
                ey = np.array(my(X_enc[i]))
                ez = np.array(mz(X_enc[i]))
                for q in range(config.N_QUBITS):
                    b[i, 3 * q] = ex[q]
                    b[i, 3 * q + 1] = ey[q]
                    b[i, 3 * q + 2] = ez[q]
            return b

        bloch_path = os.path.join(combo_dir, f"bloch_agpqk_full_train.npy")
        if os.path.exists(bloch_path):
            bloch_full = np.load(bloch_path)
        else:
            bloch_full = compute_bloch(X_fused_enc)
            np.save(bloch_path, bloch_full)

        def bloch_kernel(b1, b2, gamma=config.PQK_GAMMA):
            n1 = len(b1)
            b1r = b1.reshape(n1, config.N_QUBITS, 3)
            b2r = b2.reshape(len(b2), config.N_QUBITS, 3)
            K = np.zeros((n1, len(b2)))
            for i in range(n1):
                d = b1r[i] - b2r
                sq = np.sum(d**2, axis=2)
                f = 0.5 * np.sum(sq, axis=1)
                K[i, :] = np.exp(-gamma * f)
            return K

        K_full = bloch_kernel(bloch_full, bloch_full)
        np.save(K_full_path, K_full)
    elif kernel_type == "kdca_pqk":
        from src.kdca_kernel import compute_kdca_pqk_full_pipeline
        # X_fused has 4 SAR then 4 OPT features. Map to indices that reflect this modality split:
        synthetic_indices = np.array([8, 9, 10, 11, 0, 1, 2, 3])
        
        pipeline_res = compute_kdca_pqk_full_pipeline(
            X_train=X_fused,
            selected_indices=synthetic_indices,
            gamma_kernel=config.PQK_GAMMA,
            top_k_pairs=4,
            save_dir=combo_dir
        )
        K_full = pipeline_res["K_train"]
        np.save(K_full_path, K_full)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    return K_full

def run_single_alpha_seed(
    X_sar,
    X_opt,
    y_train,
    X_sar_test,
    X_opt_test,
    y_test,
    alpha,
    seed,
    kernel_type,
    results_dir,
):
    """Run 5-fold CV for a single (alpha, seed, kernel) combination."""
    X_fused, X_fused_test = create_mixed_data(
        X_sar, X_opt, X_sar_test, X_opt_test, alpha, seed
    )

    # Compute MI
    mi = compute_cross_modal_mutual_information(X_fused[:, :4], X_fused[:, 4:])

    # NEW OPTIMIZATION: Compute global kernel only ONCE per alpha/seed combo
    combo_dir = ensure_dir(os.path.join(results_dir, f"alpha{alpha:.1f}_seed{seed}"))
    K_full_tr = compute_full_kernel_cached(X_fused, kernel_type, combo_dir)

    # 5-fold CV
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_f1s, fold_accs, fold_ktas = [], [], []

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_fused, y_train)):
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        # Extract matrices via mathematical slicing instead of re-evaluation
        K_tr = K_full_tr[np.ix_(tr_idx, tr_idx)]
        K_val = K_full_tr[np.ix_(val_idx, tr_idx)]

        clf = train_precomputed_svm(K_tr, y_tr)
        m = evaluate_classifier(
            clf, K_val, y_val, f"{kernel_type} α={alpha:.1f} seed={seed} fold{fold_idx}"
        )

        fold_f1s.append(m["macro_f1"])
        fold_accs.append(m["accuracy"])
        fold_ktas.append(float(compute_centered_kta(K_tr, y_tr, class_weighted=True)))

    # Geometric difference on full training set
    g, _ = compute_geometric_difference(
        K_full_tr, compute_tensor_product_kernel(X_fused)
    )
    kta_full = compute_centered_kta(K_full_tr, y_train, class_weighted=True)

    return {
        "alpha": alpha,
        "seed": seed,
        "kernel": kernel_type,
        "mi": mi,
        "cv_macro_f1_mean": float(np.mean(fold_f1s)),
        "cv_macro_f1_std": float(np.std(fold_f1s)),
        "cv_accuracy_mean": float(np.mean(fold_accs)),
        "cv_accuracy_std": float(np.std(fold_accs)),
        "cv_kta_mean": float(np.mean(fold_ktas)),
        "cv_kta_std": float(np.std(fold_ktas)),
        "g_tensor": g,
        "kta_full": kta_full,
        "n_folds": N_FOLDS,
    }


def run_experiment(kernel_types=None, force=False):
    if kernel_types is None:
        kernel_types = ["fqk", "pqk", "agpqk", "kdca_pqk"]

    results_file = os.path.join(RESULTS_DIR, "exp10_v2_results.json")
    if not force and os.path.exists(results_file):
        logger.info(f"[SKIP] Results exist: {results_file}. Use --force to rerun.")
        return

    np.random.seed(config.RANDOM_SEED)

    with Timer("Loading data", logger=logger):
        X_sar, X_opt, y_train, X_sar_test, X_opt_test, y_test = load_data()
    logger.info(
        f"Loaded: SAR={X_sar.shape}, OPT={X_opt.shape}, "
        f"y_train={y_train.shape}, y_test={y_test.shape}"
    )

    all_results = {
        "kernel_types": kernel_types,
        "alphas": ALPHAS,
        "noise_seeds": NOISE_SEEDS,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "cv_folds": N_FOLDS,
    }

    for kernel_type in kernel_types:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"  KERNEL: {kernel_type.upper()}")
        logger.info(f"{'=' * 60}")

        kernel_results = []
        for seed in NOISE_SEEDS:
            for alpha in ALPHAS:
                # Check if already computed
                cache_path = os.path.join(
                    RESULTS_DIR, f"{kernel_type}_alpha{alpha:.1f}_seed{seed}.json"
                )
                if not force and os.path.exists(cache_path):
                    with open(cache_path) as f:
                        r = json.load(f)
                    logger.info(f"  [SKIP] α={alpha:.1f} seed={seed} ({kernel_type})")
                    kernel_results.append(r)
                    continue

                logger.info(f"  α={alpha:.1f} seed={seed} ({kernel_type})")
                r = run_single_alpha_seed(
                    X_sar,
                    X_opt,
                    y_train,
                    X_sar_test,
                    X_opt_test,
                    y_test,
                    alpha,
                    seed,
                    kernel_type,
                    RESULTS_DIR,
                )
                kernel_results.append(r)

                # Save per-combination result (checkpointing)
                with open(cache_path, "w") as f:
                    json.dump(r, f, indent=2)

            logger.info(
                f"  [SEED COMPLETE] {kernel_type} seed={seed}: "
                f"{len(ALPHAS)}/{len(ALPHAS)} alphas done"
            )

        all_results[kernel_type] = kernel_results

        # Summary CSV
        df = pd.DataFrame(kernel_results)
        summary = df.select_dtypes(include=[np.number]).groupby("alpha").agg(["mean", "std"]).reset_index()
        summary.columns = [
            "_".join(str(c) for c in col).strip("_") for col in summary.columns
        ]
        csv_path = os.path.join(RESULTS_DIR, f"summary_{kernel_type}.csv")
        summary.to_csv(csv_path, index=False)
        logger.info(f"  Summary saved: {csv_path}")

    save_results(all_results, results_file)
    logger.info(f"\nAll results saved: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel", choices=["fqk", "pqk", "agpqk", "kdca_pqk"], nargs="+", default=None
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_experiment(kernel_types=args.kernel, force=args.force)
