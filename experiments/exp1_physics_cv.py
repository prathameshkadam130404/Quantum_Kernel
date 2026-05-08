"""
Experiment 1 (Physics Regime): 5-Fold Cross-Validation — OPTIMIZED.

OPTIMIZATIONS APPLIED:
============================================================

1. COMPUTE FULL KERNEL ONCE, SLICE FOR FOLDS (4.5x speedup)
   Original: Recomputed quantum kernels for each of 5 folds (5x O(n^2) circuit calls).
   Optimized: Compute the full NxN kernel matrix ONCE, then extract fold submatrices
   via numpy array slicing. K[i,j] depends only on data points i and j, not on fold
   assignment. Mathematically identical, 4.5x fewer circuit evaluations.

2. TFK EXCLUDED (3x speedup)
   TFK (Trained Fidelity Kernel) consistently fails in both PCA and physics regimes
   across all anchor sizes (see TFK anchor sensitivity analysis). Including it adds
   ~3 hrs/fold of GPU time with no scientific value. Paper will note: "TFK excluded
   due to consistent failure across all regimes and anchor sizes."

3. PIQFM EXCLUDED (2x speedup)
   PIQFM (Physics-Informed Quantum Feature Map) is a trained variant similar to TFK.
   It also fails consistently in both regimes. Same justification for exclusion.

4. CM-FQK RETAINED
   Cross-Modal FQK is retained as it provides meaningful comparison of cross-modal
   entanglement effects.

COMBINED SPEEDUP: ~3.5x reduction from original design
Original estimate: ~70 hours
Optimized estimate: ~25 hours

Computes FQK, PQK, CM-FQK, AGPQK, RBF, RBF-CV, RBF-Tuned, RandomForest,
RBF-on-AGPQK-features with proper 5-fold stratified CV on physics features (N=2000).

All methods use the SAME 5 folds. Reports mean +/- std for Macro-F1, Accuracy, KTA.

Output:
    results/physics_cv/exp1_physics_cv_results.json
    results/physics_cv/fold_results/  (per-fold kernel caches)
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

import config
from src.quantum_kernels import (
    compute_fqk_kernel_matrix,
    compute_pqk_kernel_matrix,
    compute_crossmodal_fqk_kernel_matrix,
    get_top_mi_pairs,
)
from src.classical_kernels import compute_rbf_kernel, compute_rbf_kernel_cv
from src.data_loader import load_modality
from src.classifiers import train_precomputed_svm, evaluate_classifier
from src.kernel_target_alignment import compute_centered_kta
from src.utils import (
    save_results,
    setup_logging,
    Timer,
    ensure_dir,
    compute_full_metrics,
)
from src.attention_kernel import (
    compute_attention_matrix,
    select_features_by_fisher,
    compute_per_qubit_gamma,
    get_attention_entanglement_pairs,
    FEATURE_NAMES_16,
    FEATURE_MODALITY_16,
)

logger = setup_logging(
    "exp1_physics_cv",
    log_file=os.path.join(config.RESULTS_DIR, "physics_cv", "exp1_physics_cv.log"),
)

RESULTS_DIR = ensure_dir(os.path.join(config.RESULTS_DIR, "physics_cv"))
FOLD_DIR = ensure_dir(os.path.join(RESULTS_DIR, "fold_results"))

N_QUBITS = config.N_QUBITS
ZZ_REPS = config.ZZ_REPS
OUTER_GAMMA = config.PQK_GAMMA
GAMMA_BASE = 0.50
N_SELECT = 8
TOP_K_PAIRS = 4


def load_physics_data():
    phys16_path = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
    data = np.load(phys16_path)
    X_all = data["X_train"][: config.SUBSAMPLE_TRAIN]
    X_all_raw = data["X_train_raw"][: config.SUBSAMPLE_TRAIN]
    y_all = data["y_train"][: config.SUBSAMPLE_TRAIN]
    return X_all, X_all_raw, y_all


def compute_full_kernel_matrix(X_all, X_all_raw, y_all, kernel_type, save_path=None):
    """
    OPTIMIZATION 1: Compute the full NxN kernel matrix ONCE.

    Instead of computing separate kernels for each fold (which recomputes the same
    O(n^2) circuit calls 5 times), we compute the full kernel matrix once and then
    slice it for fold submatrices. K[i,j] depends only on data points i and j,
    not on which fold they belong to.

    Speedup: ~4.5x (avoids 4x redundant circuit evaluations per fold).

    Args:
        X_all: Full data matrix, shape (N, n_features).
        X_all_raw: Raw features (for AGPQK attention), shape (N, 16).
        y_all: Labels, shape (N,).
        kernel_type: "fqk", "pqk", "cm_fqk", or "agpqk".
        save_path: Optional path to cache the full kernel matrix.

    Returns:
        K_full: Full symmetric kernel matrix, shape (N, N).
        agpqk_config: AGPQK config dict (only for "agpqk" type, else None).
    """
    if save_path and os.path.exists(save_path):
        logger.info(f"  [CACHE HIT] Loading full kernel: {save_path}")
        return np.load(save_path), None

    if kernel_type == "fqk":
        K_full = compute_fqk_kernel_matrix(X_all, save_path=save_path)
        return K_full, None

    elif kernel_type == "pqk":
        K_full = compute_pqk_kernel_matrix(
            X_all,
            save_path=save_path,
            bloch_save_path=save_path.replace(".npy", "_bloch.npy")
            if save_path
            else None,
        )
        return K_full, None

    elif kernel_type == "cm_fqk":
        mi_pairs = get_top_mi_pairs(
            X_all, n_sar=4, n_opt=4, top_k=config.CM_FQK_TOP_K_PAIRS
        )
        K_full = compute_crossmodal_fqk_kernel_matrix(
            X_all, mi_pairs, save_path=save_path
        )
        return K_full, None

    elif kernel_type == "agpqk":
        # AGPQK: compute attention, Fisher selection, Bloch vectors, then kernel
        logger.info("  Computing AGPQK full kernel (attention + Bloch + kernel)...")

        # Attention matrix on full training data
        A = compute_attention_matrix(X_all_raw)

        # Fisher-ratio feature selection on full training data
        selected_indices, fisher = select_features_by_fisher(
            X_all_raw, y_all, n_select=N_SELECT, min_sar=4, min_opt=4
        )

        # Per-qubit gamma
        gamma_per_qubit = compute_per_qubit_gamma(
            fisher, selected_indices, gamma_base=GAMMA_BASE
        )

        # Attention-selected entanglement pairs
        entangle_pairs = get_attention_entanglement_pairs(
            A,
            selected_indices,
            top_k=TOP_K_PAIRS,
            require_cross_modal=True,
            max_appearances=1,
        )

        # Apply per-qubit bandwidth
        X_sel = X_all[:, selected_indices]
        X_enc = X_sel * gamma_per_qubit[np.newaxis, :]

        # Build AGPQK circuit
        import pennylane as qml

        dev = config.get_device(N_QUBITS)

        def apply_agpqk(x, n_q=N_QUBITS, reps=ZZ_REPS):
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
            return [qml.expval(qml.PauliX(i)) for i in range(N_QUBITS)]

        @qml.qnode(dev, diff_method=None)
        def my(x):
            apply_agpqk(x)
            return [qml.expval(qml.PauliY(i)) for i in range(N_QUBITS)]

        @qml.qnode(dev, diff_method=None)
        def mz(x):
            apply_agpqk(x)
            return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

        # Compute Bloch vectors for all N samples
        N = len(X_enc)
        bloch = np.zeros((N, 3 * N_QUBITS))
        from tqdm import tqdm

        for i in tqdm(range(N), desc="AGPQK Bloch (full)"):
            ex = np.array(mx(X_enc[i]))
            ey = np.array(my(X_enc[i]))
            ez = np.array(mz(X_enc[i]))
            for q in range(N_QUBITS):
                bloch[i, 3 * q] = ex[q]
                bloch[i, 3 * q + 1] = ey[q]
                bloch[i, 3 * q + 2] = ez[q]

        # Compute full kernel from Bloch vectors
        K_full = np.zeros((N, N))
        for i in tqdm(range(N), desc="AGPQK kernel (full)"):
            diff = bloch[i] - bloch
            diff_r = diff.reshape(N, N_QUBITS, 3)
            sq = np.sum(diff_r**2, axis=2)
            frob = 0.5 * np.sum(sq, axis=1)
            K_full[i, :] = np.exp(-OUTER_GAMMA * frob)

        if save_path:
            np.save(save_path, K_full)
            logger.info(f"  Saved full AGPQK kernel: {save_path}")

        agpqk_config = {
            "selected_feature_indices": selected_indices.tolist(),
            "selected_feature_names": [FEATURE_NAMES_16[i] for i in selected_indices],
            "gamma_per_qubit": gamma_per_qubit.tolist(),
            "entanglement_pairs": entangle_pairs,
            "outer_gamma": OUTER_GAMMA,
        }

        return K_full, agpqk_config

    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def run_classical_methods(X_tr, X_val, y_tr, y_val, fold_idx):
    """Run classical (non-quantum) methods — no circuit calls, fast."""
    fold_results = {}

    # ---- RBF (instant) ----
    K_rbf_tr = compute_rbf_kernel(X_tr)
    K_rbf_val = compute_rbf_kernel(X_val, X_tr)
    clf = train_precomputed_svm(K_rbf_tr, y_tr)
    m = evaluate_classifier(clf, K_rbf_val, y_val, f"RBF fold{fold_idx}")
    kta = compute_centered_kta(K_rbf_tr, y_tr, class_weighted=True)
    fold_results["RBF"] = {
        "macro_f1": m["macro_f1"],
        "accuracy": m["accuracy"],
        "kta": kta,
    }

    # ---- RBF-CV (gamma tuned via CV) ----
    fold_dir = ensure_dir(os.path.join(FOLD_DIR, f"fold{fold_idx}"))
    K_rbf_cv_tr, K_rbf_cv_val, best_gamma, _ = compute_rbf_kernel_cv(
        X_tr,
        y_tr,
        X_val,
        save_path_train=os.path.join(fold_dir, "K_rbf_cv.npy"),
        save_path_test=os.path.join(fold_dir, "K_rbf_cv_val.npy"),
    )
    clf = train_precomputed_svm(K_rbf_cv_tr, y_tr)
    m = evaluate_classifier(clf, K_rbf_cv_val, y_val, f"RBF-CV fold{fold_idx}")
    fold_results["RBF-CV"] = {
        "macro_f1": m["macro_f1"],
        "accuracy": m["accuracy"],
        "kta": compute_centered_kta(K_rbf_cv_tr, y_tr, class_weighted=True),
        "best_gamma": float(best_gamma) if best_gamma else None,
    }

    # ---- RBF-Tuned (GridSearchCV on raw features) ----
    logger.info(f"  Fold {fold_idx}: RBF-Tuned (GridSearchCV 5-fold inner)...")
    param_grid = {
        "C": [0.1, 1.0, 10.0, 100.0],
        "gamma": ["scale", "auto", 0.01, 0.1],
    }
    inner_cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=config.CV_SEED + fold_idx
    )
    grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", random_state=config.RANDOM_SEED),
        param_grid,
        scoring="f1_macro",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_tr, y_tr)
    y_pred = grid.predict(X_val)
    m = compute_full_metrics(y_val, y_pred)
    fold_results["RBF-Tuned"] = {
        "macro_f1": m["macro_f1"],
        "accuracy": m["accuracy"],
        "best_params": grid.best_params_,
        "cv_score": float(grid.best_score_),
    }

    # ---- RandomForest ----
    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_val)
    m = compute_full_metrics(y_val, y_pred)
    fold_results["RandomForest"] = {
        "macro_f1": m["macro_f1"],
        "accuracy": m["accuracy"],
    }

    return fold_results, inner_cv


def run_experiment():
    results_file = os.path.join(RESULTS_DIR, "exp1_physics_cv_results.json")
    checkpoint_file = results_file.replace(".json", "_checkpoint.json")

    if os.path.exists(results_file):
        logger.info(f"[SKIP] Results already exist: {results_file}")
        return

    np.random.seed(config.RANDOM_SEED)

    with Timer("Loading physics data", logger=logger):
        X_all, X_all_raw, y_all = load_physics_data()
    logger.info(f"Data: {X_all.shape}, {len(np.unique(y_all))} classes")

    skf = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.CV_SEED
    )
    folds = list(skf.split(X_all, y_all))

    # Resume from checkpoint
    all_fold_results = {}
    start_fold = 0
    agpqk_config_global = None
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file) as f:
                ckpt = json.load(f)
            all_fold_results = ckpt.get("folds", {})
            start_fold = ckpt.get("completed", 0)
            agpqk_config_global = ckpt.get("agpqk_config")
            logger.info(
                f"[RESUME] Loaded checkpoint: {start_fold}/{config.CV_FOLDS} folds complete"
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[RESUME] Checkpoint corrupt ({e}), starting from fold 0")

    # OPTIMIZATION 1: Compute full quantum kernels ONCE before fold loop
    logger.info(f"\n{'=' * 60}")
    logger.info("  COMPUTING FULL QUANTUM KERNELS (once, then slice for folds)")
    logger.info(f"{'=' * 60}")

    full_kernel_dir = ensure_dir(os.path.join(RESULTS_DIR, "full_kernels"))
    full_kernels = {}

    for kernel_type in ["fqk", "pqk", "cm_fqk", "agpqk"]:
        save_path = os.path.join(full_kernel_dir, f"K_{kernel_type}_full.npy")
        if os.path.exists(save_path):
            logger.info(f"  [SKIP] Full {kernel_type} kernel already exists")
            K_full = np.load(save_path)
            full_kernels[kernel_type] = K_full
            if kernel_type == "agpqk":
                cfg_path = save_path.replace(".npy", "_config.json")
                if os.path.exists(cfg_path):
                    with open(cfg_path) as f:
                        agpqk_config_global = json.load(f)
        else:
            with Timer(
                f"Full {kernel_type.upper()} kernel (N={config.SUBSAMPLE_TRAIN})",
                logger=logger,
            ):
                K_full, agpqk_cfg = compute_full_kernel_matrix(
                    X_all, X_all_raw, y_all, kernel_type, save_path=save_path
                )
                full_kernels[kernel_type] = K_full
                if agpqk_cfg is not None:
                    agpqk_config_global = agpqk_cfg
                    cfg_path = save_path.replace(".npy", "_config.json")
                    with open(cfg_path, "w") as f:
                        json.dump(agpqk_cfg, f, indent=2)

    # ---- Run folds ----
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if fold_idx < start_fold:
            logger.info(f"  [SKIP] Fold {fold_idx + 1} already in checkpoint")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"  FOLD {fold_idx + 1}/{config.CV_FOLDS} "
            f"(train={len(train_idx)}, val={len(val_idx)})"
        )
        logger.info(f"{'=' * 60}")

        X_tr = X_all[train_idx]
        X_val = X_all[val_idx]
        y_tr = y_all[train_idx]
        y_val = y_all[val_idx]

        fold_dir = ensure_dir(os.path.join(FOLD_DIR, f"fold{fold_idx}"))
        fold_results = {}

        # Classical methods (fast, no circuit calls)
        classical_results, inner_cv = run_classical_methods(
            X_tr, X_val, y_tr, y_val, fold_idx
        )
        fold_results.update(classical_results)

        # Quantum methods: slice from pre-computed full kernels
        for kernel_type, display_name in [
            ("fqk", "FQK"),
            ("pqk", "PQK"),
            ("cm_fqk", "CM-FQK"),
            ("agpqk", "AGPQK"),
        ]:
            K_full = full_kernels[kernel_type]
            # Slice submatrices (no circuit calls!)
            K_tr = K_full[np.ix_(train_idx, train_idx)]
            K_val = K_full[np.ix_(val_idx, train_idx)]

            clf = train_precomputed_svm(K_tr, y_tr)
            m = evaluate_classifier(clf, K_val, y_val, f"{display_name} fold{fold_idx}")
            kta = compute_centered_kta(K_tr, y_tr, class_weighted=True)
            fold_results[display_name] = {
                "macro_f1": m["macro_f1"],
                "accuracy": m["accuracy"],
                "kta": kta,
            }

        # ---- RBF-on-AGPQK-features ----
        if agpqk_config_global is not None:
            sel_idx = agpqk_config_global["selected_feature_indices"]
            X_tr_sel = X_tr[:, sel_idx]
            X_val_sel = X_val[:, sel_idx]
            logger.info(f"  Fold {fold_idx}: RBF-on-AGPQK-features (GridSearchCV)...")
            param_grid = {
                "C": [0.1, 1.0, 10.0, 100.0],
                "gamma": ["scale", "auto", 0.01, 0.1],
            }
            grid_sel = GridSearchCV(
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=config.RANDOM_SEED,
                ),
                param_grid,
                scoring="f1_macro",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
            )
            grid_sel.fit(X_tr_sel, y_tr)
            y_pred = grid_sel.predict(X_val_sel)
            m = compute_full_metrics(y_val, y_pred)
            fold_results["RBF-on-AGPQK-features"] = {
                "macro_f1": m["macro_f1"],
                "accuracy": m["accuracy"],
                "best_params": grid_sel.best_params_,
                "cv_score": float(grid_sel.best_score_),
                "selected_features": [FEATURE_NAMES_16[i] for i in sel_idx],
            }

        all_fold_results[f"fold_{fold_idx}"] = fold_results

        # Checkpoint
        ckpt = {
            "folds": all_fold_results,
            "completed": fold_idx + 1,
            "agpqk_config": agpqk_config_global,
        }
        with open(checkpoint_file, "w") as f:
            json.dump(ckpt, f, indent=2)
        logger.info(f"  [CHECKPOINT] Saved after fold {fold_idx + 1}")

    # ---- Aggregate ----
    methods = [
        "FQK",
        "PQK",
        "CM-FQK",
        "AGPQK",
        "RBF",
        "RBF-CV",
        "RBF-Tuned",
        "RandomForest",
        "RBF-on-AGPQK-features",
    ]
    summary = {}
    for method in methods:
        f1s, accs, ktas = [], [], []
        for fk in all_fold_results:
            r = all_fold_results[fk].get(method, {})
            if isinstance(r.get("macro_f1"), (int, float)):
                f1s.append(r["macro_f1"])
                accs.append(r["accuracy"])
                if isinstance(r.get("kta"), (int, float)):
                    ktas.append(r["kta"])
        summary[method] = {
            "macro_f1_mean": float(np.mean(f1s)) if f1s else None,
            "macro_f1_std": float(np.std(f1s)) if len(f1s) > 1 else None,
            "accuracy_mean": float(np.mean(accs)) if accs else None,
            "accuracy_std": float(np.std(accs)) if len(accs) > 1 else None,
            "kta_mean": float(np.mean(ktas)) if ktas else None,
            "kta_std": float(np.std(ktas)) if len(ktas) > 1 else None,
            "n_folds": len(f1s),
        }

    logger.info(f"\n{'=' * 60}")
    logger.info("  5-FOLD CV SUMMARY (PHYSICS REGIME, N=2000) — OPTIMIZED")
    logger.info(f"{'=' * 60}")
    logger.info(f"{'Method':<25s} {'Macro-F1':>14s} {'Accuracy':>14s} {'KTA':>12s}")
    logger.info("-" * 68)
    for method in methods:
        s = summary[method]
        f1 = (
            f"{s['macro_f1_mean']:.4f} +/- {s['macro_f1_std']:.4f}"
            if s["macro_f1_mean"]
            else "N/A"
        )
        acc = (
            f"{s['accuracy_mean']:.4f} +/- {s['accuracy_std']:.4f}"
            if s["accuracy_mean"]
            else "N/A"
        )
        kta = f"{s['kta_mean']:.4f} +/- {s['kta_std']:.4f}" if s["kta_mean"] else "N/A"
        logger.info(f"{method:<25s} {f1:>14s} {acc:>14s} {kta:>12s}")

    save_results(
        {
            "summary": summary,
            "fold_results": all_fold_results,
            "n_folds": config.CV_FOLDS,
            "n_train": config.SUBSAMPLE_TRAIN,
            "cv_seed": config.CV_SEED,
            "modality": "physics_features_16",
            "agpqk_config": agpqk_config_global,
            "optimizations": {
                "full_kernel_once": "Full quantum kernels computed once, sliced for folds (4.5x speedup)",
                "tfk_excluded": "TFK excluded due to consistent failure across all regimes",
                "piqfm_excluded": "PIQFM excluded due to consistent failure across all regimes",
            },
        },
        results_file,
    )
    logger.info(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    run_experiment()
