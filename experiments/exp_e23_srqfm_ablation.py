"""
E23 — SRQFM Ablation Study: Isolating the Coupling Function's Contribution.

Tests four coupling function variants to prove that the fidelity-distance
coupling is the key innovation, not just "any data-dependent coupling":

    Condition A: SRQFM coupling      sin^2((x_i - x_j) / 2)  [fidelity distance]
    Condition B: Standard ZZ coupling (pi - x_i)(pi - x_j)    [Havlicek]
    Condition C: Constant coupling    0.5 (always moderate)    [control]
    Condition D: Product state        0.0 (no entanglement)    [control]

All conditions use the same circuit structure (H -> RZ -> entanglement)
and the same PQK Bloch extraction, differing ONLY in the coupling function.

If SRQFM > B > C >= D, this proves:
    1. Self-regulating coupling adds value beyond standard ZZ
    2. Entanglement itself provides benefit (B, A > D)
    3. Data-dependent coupling is better than constant (A, B > C)

Author: Prathamesh Kadam et al.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import ttest_rel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._e_common import (
    load_physics_16,
    spectral_metrics,
    cohens_d,
)
from src.attention_kernel import select_features_by_fisher
from src.srqfm_kernel import compute_kta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("exp_e23")

# Configuration
N_QUBITS = config.N_QUBITS
ZZ_REPS = config.ZZ_REPS
PQK_GAMMA = config.PQK_GAMMA
SEEDS = config.SEED_LIST
SAVE_DIR = os.path.join(config.RESULTS_DIR, "srqfm_ablation")


# ============ COUPLING FUNCTIONS ============

def coupling_fidelity(x_i, x_j):
    """SRQFM: fidelity distance. Self-regulating."""
    return np.sin((x_i - x_j) / 2.0) ** 2

def coupling_standard_zz(x_i, x_j):
    """Standard Havlicek ZZ. Not self-regulating."""
    return (np.pi - x_i) * (np.pi - x_j)

def coupling_constant(x_i, x_j):
    """Constant coupling. Control condition."""
    return 0.5

def coupling_zero(x_i, x_j):
    """No coupling. Product state control."""
    return 0.0


CONDITIONS = {
    "A_SRQFM": {
        "coupling_fn": coupling_fidelity,
        "description": "Fidelity distance: sin^2((x_i-x_j)/2)",
        "use_ising": True,
    },
    "B_StandardZZ": {
        "coupling_fn": coupling_standard_zz,
        "description": "Standard Havlicek: (pi-x_i)(pi-x_j)",
        "use_ising": False,  # Use CNOT-RZ-CNOT decomposition
    },
    "C_Constant": {
        "coupling_fn": coupling_constant,
        "description": "Constant coupling: 0.5",
        "use_ising": True,
    },
    "D_Product": {
        "coupling_fn": coupling_zero,
        "description": "No entanglement (product state)",
        "use_ising": True,
    },
}


# ============ GENERIC FEATURE MAP WITH PLUGGABLE COUPLING ============

def compute_bloch_with_coupling(X, coupling_fn, use_ising=True,
                                 n_qubits=N_QUBITS, reps=ZZ_REPS):
    """
    Compute PQK Bloch vectors using a generic coupling function.

    All conditions share the same circuit skeleton:
        H -> RZ(x_i) -> [entanglement with coupling_fn] -> measure Bloch

    Args:
        X: Input data, shape (N, n_qubits). Values in [0, pi].
        coupling_fn: Callable(x_i, x_j) -> float. Coupling strength.
        use_ising: If True, use IsingZZ(angle). If False, use CNOT-RZ-CNOT.
        n_qubits: Number of qubits.
        reps: Feature map repetitions.

    Returns:
        np.ndarray: Bloch vectors, shape (N, 3*n_qubits).
    """
    import pennylane as qml

    dev = config.get_device(n_qubits)
    pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    def apply_fm(x):
        for _ in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for i in range(n_qubits):
                qml.RZ(x[i], wires=i)
            for qi, qj in pairs:
                c = coupling_fn(x[qi], x[qj])
                if abs(c) < 0.001:
                    continue  # Skip negligible coupling
                if use_ising:
                    qml.IsingZZ(c, wires=[qi, qj])
                else:
                    # CNOT-RZ-CNOT decomposition (standard ZZ)
                    qml.CNOT(wires=[qi, qj])
                    qml.RZ(c, wires=qj)
                    qml.CNOT(wires=[qi, qj])

    @qml.qnode(dev, diff_method=None)
    def mx(x):
        apply_fm(x)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def my(x):
        apply_fm(x)
        return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def mz(x):
        apply_fm(x)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    N = len(X)
    bloch = np.zeros((N, 3 * n_qubits), dtype=np.float64)
    for idx in tqdm(range(N), desc="Bloch extraction", leave=False):
        ex = np.array(mx(X[idx]))
        ey = np.array(my(X[idx]))
        ez = np.array(mz(X[idx]))
        for q in range(n_qubits):
            bloch[idx, 3 * q] = ex[q]
            bloch[idx, 3 * q + 1] = ey[q]
            bloch[idx, 3 * q + 2] = ez[q]

    return bloch


def bloch_to_kernel(bloch, gamma=PQK_GAMMA, n_qubits=N_QUBITS):
    """Convert Bloch vectors to PQK RBF kernel matrix."""
    N = len(bloch)
    b = bloch.reshape(N, n_qubits, 3)
    K = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        diff = b[i] - b
        sq = np.sum(diff ** 2, axis=2)
        frob = 0.5 * np.sum(sq, axis=1)
        K[i, :] = np.exp(-gamma * frob)
    return K


# ============ MAIN ABLATION ============

def run_ablation():
    """Execute the E23 ablation study."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    fh = logging.FileHandler(os.path.join(SAVE_DIR, "exp_e23.log"), mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    ))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 70)
    logger.info("  E23: SRQFM Ablation — Coupling Function Analysis")
    logger.info("=" * 70)
    logger.info(f"  Date: {datetime.now().isoformat()}")

    # Load data
    X_norm, X_raw, y = load_physics_16()
    sel_idx, _ = select_features_by_fisher(X_raw, y)
    X_sel = X_norm[:, sel_idx]

    feature_names_all = list(np.load(
        os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
    )["feature_names"])
    sel_names = [feature_names_all[i] for i in sel_idx]
    logger.info(f"  Features: {sel_names}")
    logger.info(f"  N={len(X_sel)}, classes={len(np.unique(y))}")

    # ========= Compute Bloch vectors for each condition =========
    all_bloch = {}
    all_kernels = {}
    all_diag = {}

    for cond_name, cond_cfg in CONDITIONS.items():
        logger.info(f"\n--- Condition {cond_name}: {cond_cfg['description']} ---")

        t0 = time.time()
        bloch = compute_bloch_with_coupling(
            X_sel, cond_cfg["coupling_fn"], cond_cfg["use_ising"],
        )
        dt = time.time() - t0

        # Bloch magnitudes
        mags = []
        for q in range(N_QUBITS):
            bx = bloch[:, 3 * q]
            by = bloch[:, 3 * q + 1]
            bz = bloch[:, 3 * q + 2]
            mags.append(float(np.sqrt(bx ** 2 + by ** 2 + bz ** 2).mean()))

        logger.info(f"  Bloch mags: {[f'{m:.3f}' for m in mags]}")
        logger.info(f"  Mean Bloch mag: {np.mean(mags):.4f}")
        logger.info(f"  Time: {dt:.1f}s")

        # Kernel
        K = bloch_to_kernel(bloch)
        spec = spectral_metrics(K)
        kta = compute_kta(K, y)

        logger.info(f"  KTA: {kta:.4f}")
        logger.info(f"  Eff rank: {spec['eff_rank_shannon']:.1f}")
        logger.info(f"  Off-diag mean: {spec['off_diag_mean']:.4f}")

        all_bloch[cond_name] = bloch
        all_kernels[cond_name] = K
        all_diag[cond_name] = {
            **spec,
            "kta": kta,
            "bloch_magnitudes": mags,
            "mean_bloch_mag": float(np.mean(mags)),
            "time_s": dt,
        }

        np.save(os.path.join(SAVE_DIR, f"K_{cond_name}.npy"), K)

    # ========= CV Evaluation =========
    logger.info("\n" + "=" * 70)
    logger.info("  CV Classification")
    logger.info("=" * 70)

    cv_results = {cond: [] for cond in CONDITIONS}

    for seed in SEEDS:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3,
                                    random_state=seed)
        (tr, te), = sss.split(np.zeros(len(y)), y)

        for cond_name, K in all_kernels.items():
            K_tr = K[np.ix_(tr, tr)]
            K_te = K[np.ix_(te, tr)]
            clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
            clf.fit(K_tr, y[tr])
            yp = clf.predict(K_te)
            f1 = float(f1_score(y[te], yp, average="macro"))
            acc = float(accuracy_score(y[te], yp))
            cv_results[cond_name].append({"macro_f1": f1, "accuracy": acc})

    # ========= Summary =========
    logger.info("\n" + "=" * 70)
    logger.info("  ABLATION RESULTS")
    logger.info("=" * 70)

    summary = {}
    logger.info(f"\n  {'Condition':<20} {'F1 Mean':>8} {'F1 Std':>8} "
                f"{'KTA':>8} {'Bloch Mag':>10}")
    logger.info("  " + "-" * 60)

    for cond_name in CONDITIONS:
        f1s = [r["macro_f1"] for r in cv_results[cond_name]]
        accs = [r["accuracy"] for r in cv_results[cond_name]]
        kta = all_diag[cond_name]["kta"]
        bmag = all_diag[cond_name]["mean_bloch_mag"]

        summary[cond_name] = {
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "f1_scores": f1s,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "kta": kta,
            "mean_bloch_mag": bmag,
            "description": CONDITIONS[cond_name]["description"],
        }

        logger.info(f"  {cond_name:<20} {np.mean(f1s):>8.4f} {np.std(f1s):>8.4f} "
                    f"{kta:>8.4f} {bmag:>10.4f}")

    # Statistical tests: A vs each other
    logger.info("\n--- Statistical Tests (A_SRQFM vs others) ---")
    stat_tests = {}
    a_f1s = summary["A_SRQFM"]["f1_scores"]

    for cond_name in ["B_StandardZZ", "C_Constant", "D_Product"]:
        other_f1s = summary[cond_name]["f1_scores"]
        t_stat, p_val = ttest_rel(a_f1s, other_f1s)
        d = cohens_d(np.array(a_f1s), np.array(other_f1s))
        direction = "SRQFM wins" if np.mean(a_f1s) > np.mean(other_f1s) else "SRQFM loses"
        sig = "p<0.05" if p_val < 0.05 else "n.s."

        stat_tests[cond_name] = {
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": float(d),
            "significant": p_val < 0.05,
        }

        logger.info(f"  A vs {cond_name:<20}: t={t_stat:+.3f}, p={p_val:.4f}, "
                    f"d={d:+.3f} ({direction}, {sig})")

    # ========= Scientific Interpretation =========
    logger.info("\n" + "=" * 70)
    logger.info("  INTERPRETATION")
    logger.info("=" * 70)

    a_f1 = summary["A_SRQFM"]["f1_mean"]
    b_f1 = summary["B_StandardZZ"]["f1_mean"]
    c_f1 = summary["C_Constant"]["f1_mean"]
    d_f1 = summary["D_Product"]["f1_mean"]

    if a_f1 > b_f1:
        logger.info("  [+] SRQFM coupling > Standard ZZ: self-regulation helps")
    else:
        logger.info("  [-] Standard ZZ >= SRQFM: self-regulation does not help")

    if max(a_f1, b_f1) > d_f1:
        logger.info("  [+] Entanglement > Product state: entanglement IS useful")
    else:
        logger.info("  [-] Product state >= entangled: entanglement hurts")

    if max(a_f1, b_f1) > c_f1:
        logger.info("  [+] Data-dependent > Constant: coupling should be adaptive")
    else:
        logger.info("  [-] Constant >= data-dependent: no benefit to adaptation")

    # Save
    results = {
        "experiment": "E23_SRQFM_Ablation",
        "timestamp": datetime.now().isoformat(),
        "conditions": {k: v["description"] for k, v in CONDITIONS.items()},
        "summary": summary,
        "diagnostics": {
            k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                for kk, vv in v.items()}
            for k, v in all_diag.items()
        },
        "statistical_tests": stat_tests,
    }

    results_path = os.path.join(SAVE_DIR, "e23_srqfm_ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n  Results saved: {results_path}")

    return results


if __name__ == "__main__":
    run_ablation()
