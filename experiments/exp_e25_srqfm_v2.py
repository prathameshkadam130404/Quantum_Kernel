"""
E25-v3: Systematic SRQFM Circuit Ablation with SVM C-Optimization.

Root cause analysis revealed that the v2 failure was caused by three
simultaneous uncontrolled changes (pi scaling, RY encoding, depth 4) that
destroyed the v1's optimal rank-30 kernel geometry (KTA=0.379).

This experiment isolates each variable and adds proper SVM C-tuning
via inner cross-validation, which was the actual bottleneck.

Circuit Configurations (one variable changed at a time from v1 baseline):
    A: H->RZ, sin^2(d/2),       reps=2  (v1 baseline)
    B: H->RZ, sin^2(d/2),       reps=3  (deeper)
    C: H->RY, sin^2(d/2),       reps=2  (amplitude encoding)
    D: H->RZ, (pi/2)*sin^2(d/2), reps=2  (moderate scaling)
    E: H->RZ, sin^2(d),          reps=2  (wider coupling window)
    F: H->RZ->RY, sin^2(d/2),   reps=2  (phase+amplitude)

All measured via Fidelity Quantum Kernel (FQK) - no classical gamma.
SVM C optimized via inner 3-fold CV for each train split.

Author: Prathamesh Kadam et al.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Callable, Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
import pennylane as qml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._e_common import spectral_metrics, cohens_d
from experiments.exp_e22_srqfm_benchmark import evaluate_classical_baselines
from src.srqfm_kernel import compute_kta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("exp_e25v3")

# ============ CONSTANTS ============
N_QUBITS = 8
SEEDS = config.SEED_LIST
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
SAVE_DIR = os.path.join(config.RESULTS_DIR, "srqfm_ablation_v3")

dev = qml.device("lightning.qubit", wires=N_QUBITS)


# ============ CIRCUIT FACTORY ============

def _make_statevector_qnode(encoding: str, coupling_fn: str, reps: int):
    """
    Build a PennyLane QNode returning the full statevector for a given
    circuit configuration.

    Args:
        encoding: One of 'H_RZ', 'H_RY', 'H_RZ_RY'.
        coupling_fn: One of 'sin2_half', 'sin2_full', 'sin2_half_scaled'.
        reps: Number of repetitions.

    Returns:
        A callable QNode f(x) -> statevector.
    """
    @qml.qnode(dev, diff_method=None)
    def circuit(x):
        for _ in range(reps):
            # --- Single-qubit encoding ---
            for i in range(N_QUBITS):
                qml.Hadamard(wires=i)

            if encoding == "H_RZ":
                for i in range(N_QUBITS):
                    qml.RZ(x[i], wires=i)
            elif encoding == "H_RY":
                for i in range(N_QUBITS):
                    qml.RY(x[i], wires=i)
            elif encoding == "H_RZ_RY":
                for i in range(N_QUBITS):
                    qml.RZ(x[i], wires=i)
                    qml.RY(x[i], wires=i)
            else:
                raise ValueError(f"Unknown encoding: {encoding}")

            # --- Entanglement ---
            for i in range(N_QUBITS):
                for j in range(i + 1, N_QUBITS):
                    delta = float(x[i] - x[j])

                    if coupling_fn == "sin2_half":
                        # Original v1: sin^2(delta/2), max=1.0
                        J = np.sin(delta / 2.0) ** 2
                    elif coupling_fn == "sin2_full":
                        # Wider window: sin^2(delta), max=1.0
                        J = np.sin(delta) ** 2
                    elif coupling_fn == "sin2_half_scaled":
                        # Moderate scaling: (pi/2) * sin^2(delta/2), max~1.57
                        J = (np.pi / 2.0) * np.sin(delta / 2.0) ** 2
                    else:
                        raise ValueError(f"Unknown coupling: {coupling_fn}")

                    if J > 0.01:
                        qml.IsingZZ(J, wires=[i, j])

        return qml.state()

    return circuit


def compute_fqk_matrix(X: np.ndarray, qnode_fn) -> Tuple[np.ndarray, float]:
    """Compute exact FQK kernel matrix K_ij = |<psi_i|psi_j>|^2."""
    N = len(X)
    dim = 2 ** N_QUBITS

    logger.info(f"  Extracting {N} statevectors (dim={dim})...")
    states = np.zeros((N, dim), dtype=np.complex128)

    t0 = time.time()
    for i in tqdm(range(N), desc="Statevector", leave=False):
        states[i] = qnode_fn(X[i])
    dt_state = time.time() - t0

    logger.info(f"  Computing fidelity matrix ({N}x{N})...")
    t0 = time.time()
    K = np.abs(states @ states.conj().T) ** 2
    dt_kernel = time.time() - t0

    np.fill_diagonal(K, 1.0)
    K = np.clip(K, 0.0, 1.0)

    total_time = dt_state + dt_kernel
    logger.info(f"  Done in {total_time:.1f}s")
    return K, total_time


# ============ SVM WITH C-TUNING ============

def svm_eval_with_c_tuning(
    K_full: np.ndarray, y: np.ndarray, seed: int,
    test_frac: float = 0.3, c_grid: List[float] = C_GRID,
) -> Dict:
    """
    Evaluate SVM on precomputed kernel with inner CV for C selection.

    The outer split defines train/test. Within the training fold,
    3-fold stratified CV selects the optimal C from the grid.
    The best C is then used to train on the full training fold
    and predict on the test fold.

    Returns dict with macro_f1, accuracy, best_C.
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac,
                                random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)

    K_tr = K_full[np.ix_(tr, tr)]
    K_te = K_full[np.ix_(te, tr)]
    y_tr, y_te = y[tr], y[te]

    # Inner CV for C selection
    clf_cv = GridSearchCV(
        SVC(kernel="precomputed", class_weight="balanced"),
        param_grid={"C": c_grid},
        cv=3,
        scoring="f1_macro",
        refit=True,
        n_jobs=1,
    )
    clf_cv.fit(K_tr, y_tr)
    best_c = clf_cv.best_params_["C"]

    yp = clf_cv.predict(K_te)
    return {
        "macro_f1": float(f1_score(y_te, yp, average="macro")),
        "accuracy": float(accuracy_score(y_te, yp)),
        "best_C": float(best_c),
    }


# ============ MAIN EXPERIMENT ============

def run_experiment():
    os.makedirs(SAVE_DIR, exist_ok=True)

    fh = logging.FileHandler(os.path.join(SAVE_DIR, "exp_e25v3.log"), mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    ))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 70)
    logger.info("  E25-v3: Systematic SRQFM Ablation + SVM C-Optimization")
    logger.info("=" * 70)
    logger.info(f"  Date: {datetime.now().isoformat()}")
    logger.info(f"  C grid: {C_GRID}")

    # 1. Load raw band data
    data_path = os.path.join(config.PROCESSED_DIR, "raw_band_features_8.npz")
    data = np.load(data_path)
    X_norm = data["X_norm"]
    X_raw = data["X_raw"]
    y = data["y"]
    names = list(data["feature_names"])

    logger.info(f"  N={len(X_norm)}, features={len(names)}")
    logger.info(f"  Classes: {len(np.unique(y))}, distribution: "
                f"{dict(zip(*np.unique(y, return_counts=True)))}")

    # 2. Define circuit configurations
    configs = {
        "A_v1_baseline": {"encoding": "H_RZ",    "coupling": "sin2_half",        "reps": 2},
        "B_depth3":      {"encoding": "H_RZ",    "coupling": "sin2_half",        "reps": 3},
        "C_RY_encode":   {"encoding": "H_RY",    "coupling": "sin2_half",        "reps": 2},
        "D_scaled":      {"encoding": "H_RZ",    "coupling": "sin2_half_scaled", "reps": 2},
        "E_wider":       {"encoding": "H_RZ",    "coupling": "sin2_full",        "reps": 2},
        "F_RZ_RY":       {"encoding": "H_RZ_RY", "coupling": "sin2_half",        "reps": 2},
    }

    # 3. Compute all kernel matrices
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 1: Kernel Matrix Computation")
    logger.info("=" * 70)

    kernels = {}
    timings = {}

    for name, cfg in configs.items():
        logger.info(f"\n--- Config {name}: encoding={cfg['encoding']}, "
                    f"coupling={cfg['coupling']}, reps={cfg['reps']} ---")

        qnode = _make_statevector_qnode(cfg["encoding"], cfg["coupling"], cfg["reps"])
        K, dt = compute_fqk_matrix(X_norm, qnode)
        kernels[name] = K
        timings[name] = dt

        np.save(os.path.join(SAVE_DIR, f"K_{name}.npy"), K)

    # 4. Kernel Diagnostics
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 2: Kernel Diagnostics")
    logger.info("=" * 70)

    diagnostics = {}
    logger.info(f"\n  {'Config':<20} {'KTA':>8} {'Eff Rank':>10} "
                f"{'Off-diag mu':>12} {'Off-diag sd':>12}")
    logger.info("  " + "-" * 66)

    for name, K in kernels.items():
        spec = spectral_metrics(K)
        kta = compute_kta(K, y)
        diagnostics[name] = {**spec, "kta": kta}
        logger.info(f"  {name:<20} {kta:>8.4f} {spec['eff_rank_shannon']:>10.1f} "
                    f"{spec['off_diag_mean']:>12.4f} {spec['off_diag_var']**0.5:>12.4f}")

    # 5. Classification with C-tuning
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 3: Classification (SVM C-tuned + Classical Baselines)")
    logger.info("=" * 70)

    all_methods = list(kernels.keys()) + ["RBF-SVM", "RandomForest"]
    cv_results = {m: [] for m in all_methods}

    for seed in SEEDS:
        logger.info(f"\n  --- Seed {seed} ---")

        for name, K in kernels.items():
            res = svm_eval_with_c_tuning(K, y, seed)
            cv_results[name].append(res)
            logger.info(f"    {name:<20}: F1={res['macro_f1']:.4f}  "
                        f"C={res['best_C']:.2f}")

        cl_res = evaluate_classical_baselines(X_raw, y, seed)
        for cl_name, cl_metrics in cl_res.items():
            cv_results[cl_name].append(cl_metrics)
            logger.info(f"    {cl_name:<20}: F1={cl_metrics['macro_f1']:.4f}")

    # 6. Summary
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 4: Final Summary")
    logger.info("=" * 70)

    summary = {}
    logger.info(f"\n  {'Method':<20} {'F1 Mean':>8} {'F1 Std':>8} "
                f"{'Acc':>8} {'KTA':>8} {'Best C':>8}")
    logger.info("  " + "-" * 72)

    for name in all_methods:
        f1s = [r["macro_f1"] for r in cv_results[name]]
        accs = [r["accuracy"] for r in cv_results[name]]
        best_cs = [r.get("best_C", None) for r in cv_results[name]]
        kta_val = diagnostics.get(name, {}).get("kta", None)

        summary[name] = {
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "f1_scores": f1s,
            "acc_mean": float(np.mean(accs)),
            "kta": kta_val,
            "best_cs": best_cs,
        }

        kta_str = f"{kta_val:.4f}" if kta_val is not None else "   N/A"
        c_str = f"{np.mean([c for c in best_cs if c is not None]):.2f}" \
                if any(c is not None for c in best_cs) else "   N/A"
        logger.info(f"  {name:<20} {np.mean(f1s):>8.4f} {np.std(f1s):>8.4f} "
                    f"{np.mean(accs):>8.4f} {kta_str:>8} {c_str:>8}")

    # 7. Statistical tests (best quantum vs RF)
    logger.info("\n--- Statistical Significance ---")
    best_quantum = max(
        [n for n in kernels.keys()],
        key=lambda n: summary[n]["f1_mean"]
    )
    logger.info(f"  Best quantum config: {best_quantum} "
                f"(F1={summary[best_quantum]['f1_mean']:.4f})")

    stat_tests = {}
    bq_f1s = summary[best_quantum]["f1_scores"]

    for name in all_methods:
        if name == best_quantum:
            continue
        other_f1s = summary[name]["f1_scores"]
        if len(bq_f1s) != len(other_f1s):
            continue

        t, p = ttest_rel(bq_f1s, other_f1s)
        d = cohens_d(np.array(bq_f1s), np.array(other_f1s))
        direction = "wins" if np.mean(bq_f1s) > np.mean(other_f1s) else "loses"
        sig = "YES" if p < 0.05 else "no"

        stat_tests[name] = {
            "p_value": float(p), "cohens_d": float(d),
            "significant": p < 0.05,
        }
        logger.info(f"  {best_quantum} vs {name:<20}: p={p:.4f} d={d:+.3f} "
                    f"({direction}, {sig})")

    # 8. Save
    results = {
        "experiment": "E25_v3_Systematic_Ablation",
        "timestamp": datetime.now().isoformat(),
        "configurations": configs,
        "c_grid": C_GRID,
        "summary": {k: {kk: vv for kk, vv in v.items()
                        if not isinstance(vv, np.ndarray)}
                    for k, v in summary.items()},
        "diagnostics": {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()}
            for k, v in diagnostics.items()
        },
        "statistical_tests": stat_tests,
        "timings_s": timings,
        "best_quantum_config": best_quantum,
    }

    res_path = os.path.join(SAVE_DIR, "e25v3_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved: {res_path}")
    return results


if __name__ == "__main__":
    run_experiment()
