"""
IBM Quantum hardware validation for the Quantum-Sat Classification pipeline.

Two-stage budget approach (10 min QPU free tier):
    Stage 1 (pilot): 10 samples, 55 kernel evals (~1 min). Verify and save.
    Stage 2 (full):  15 train + 10 test, ~270 evals (~4-5 min). Save after each batch.

Analysis (local): Frobenius error, element-wise heatmap, g_hardware vs g_simulator,
SVM accuracy comparison.
"""

import os
import sys
import logging
import time
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)


def compute_hardware_kernel_entry(
    x1: np.ndarray,
    x2: np.ndarray,
    hw_circuit,
) -> float:
    """
    Compute a single kernel entry on IBM Quantum hardware.

    Args:
        x1: First data point.
        x2: Second data point.
        hw_circuit: Hardware-connected QNode.

    Returns:
        float: Kernel value.
    """
    probs = hw_circuit(x1, x2)
    return float(probs[0])  # Probability of all-zeros state


def build_hardware_circuit(n_qubits: int, reps: int):
    """
    Build the FQK circuit connected to IBM Quantum hardware.

    Args:
        n_qubits: Number of qubits.
        reps: ZZFeatureMap repetitions.

    Returns:
        QNode connected to IBM hardware.
    """
    import pennylane as qml
    from src.quantum_kernels import apply_zz_feature_map

    dev = config.get_ibm_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def hw_circuit(x1, x2):
        apply_zz_feature_map(x1, n_qubits, reps)
        qml.adjoint(apply_zz_feature_map)(x2, n_qubits, reps)
        return qml.probs(wires=range(n_qubits))

    return hw_circuit


def run_pilot_stage(
    X_train: np.ndarray,
    n_pilot: int = config.IBM_PILOT_N_TRAIN,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Stage 1: Pilot validation with small kernel matrix.

    10 samples → 55 unique evaluations → ~1 min QPU time.
    Saves immediately after computation.

    Args:
        X_train: Training data (pilot subset).
        n_pilot: Number of pilot samples.
        n_qubits: Number of qubits.
        reps: Feature map repetitions.
        save_dir: Directory to save results.

    Returns:
        np.ndarray: Pilot kernel matrix.
    """
    pilot_file = os.path.join(save_dir, "pilot_kernel.npy") if save_dir else None
    if pilot_file and os.path.exists(pilot_file):
        logger.info(f"[SKIP] Pilot kernel exists: {pilot_file}")
        return np.load(pilot_file)

    X_pilot = X_train[:n_pilot]
    n = len(X_pilot)
    n_evals = n * (n + 1) // 2

    logger.info(f"Stage 1 — Pilot: {n} samples, {n_evals} kernel evaluations")
    logger.info(f"Estimated QPU time: ~{n_evals * 1.0:.0f} seconds")

    hw_circuit = build_hardware_circuit(n_qubits, reps)

    K = np.eye(n, dtype=np.float64)
    with tqdm(total=n_evals - n, desc="Pilot kernel (HW)") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                val = compute_hardware_kernel_entry(X_pilot[i], X_pilot[j], hw_circuit)
                K[i, j] = val
                K[j, i] = val
                pbar.update(1)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(pilot_file, K)
        logger.info(f"Saved pilot kernel: {pilot_file}")

    return K


def run_full_stage(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_train: int = config.IBM_FULL_N_TRAIN,
    n_test: int = config.IBM_FULL_N_TEST,
    n_qubits: int = config.N_QUBITS,
    reps: int = config.ZZ_REPS,
    save_dir: Optional[str] = None,
    batch_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stage 2: Full hardware validation.

    15 train + 10 test → training kernel (15×15) + test kernel (10×15).
    ~270 evaluations → ~4-5 min QPU time.
    Saves after each batch for crash recovery.

    Args:
        X_train: Training data.
        X_test: Test data.
        n_train: Training samples for hardware.
        n_test: Test samples for hardware.
        n_qubits: Number of qubits.
        reps: Feature map repetitions.
        save_dir: Save directory.
        batch_size: Batch size for saving intermediate results.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (K_train_hw, K_test_hw).
    """
    train_file = os.path.join(save_dir, "full_train_kernel.npy") if save_dir else None
    test_file = os.path.join(save_dir, "full_test_kernel.npy") if save_dir else None

    if train_file and os.path.exists(train_file) and os.path.exists(test_file):
        logger.info("[SKIP] Full hardware kernels exist")
        return np.load(train_file), np.load(test_file)

    X_tr = X_train[:n_train]
    X_te = X_test[:n_test]

    hw_circuit = build_hardware_circuit(n_qubits, reps)

    # Training kernel (symmetric)
    n = len(X_tr)
    K_train = np.eye(n, dtype=np.float64)
    n_evals = n * (n - 1) // 2

    logger.info(f"Stage 2 — Full training kernel: {n}×{n} ({n_evals} evals)")

    with tqdm(total=n_evals, desc="Full train kernel (HW)") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                val = compute_hardware_kernel_entry(X_tr[i], X_tr[j], hw_circuit)
                K_train[i, j] = val
                K_train[j, i] = val
                pbar.update(1)

            # Save periodically
            if save_dir and (i + 1) % batch_size == 0:
                np.save(train_file, K_train)

    if save_dir:
        np.save(train_file, K_train)

    # Test kernel (rectangular)
    m = len(X_te)
    K_test = np.zeros((m, n), dtype=np.float64)
    n_evals_test = m * n

    logger.info(f"Stage 2 — Full test kernel: {m}×{n} ({n_evals_test} evals)")

    with tqdm(total=n_evals_test, desc="Full test kernel (HW)") as pbar:
        for i in range(m):
            for j in range(n):
                K_test[i, j] = compute_hardware_kernel_entry(X_te[i], X_tr[j], hw_circuit)
                pbar.update(1)

            if save_dir and (i + 1) % batch_size == 0:
                np.save(test_file, K_test)

    if save_dir:
        np.save(test_file, K_test)

    return K_train, K_test


def analyze_hardware_results(
    K_sim: np.ndarray,
    K_hw: np.ndarray,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Compare simulator and hardware kernel matrices.

    Computes: Frobenius error, element-wise statistics, correlation.

    Args:
        K_sim: Simulator kernel matrix.
        K_hw: Hardware kernel matrix.
        save_dir: Directory to save analysis plots.

    Returns:
        dict: Comparison metrics.
    """
    # Ensure same size
    n = min(K_sim.shape[0], K_hw.shape[0])
    K_s = K_sim[:n, :n]
    K_h = K_hw[:n, :n]

    # Frobenius error
    frob_err = np.linalg.norm(K_s - K_h, "fro") / np.linalg.norm(K_s, "fro")

    # Element-wise statistics
    diff = np.abs(K_s - K_h)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    results = {
        "frobenius_error": float(frob_err),
        "max_element_error": float(diff[mask].max()),
        "mean_element_error": float(diff[mask].mean()),
        "std_element_error": float(diff[mask].std()),
        "correlation": float(np.corrcoef(K_s[mask], K_h[mask])[0, 1]),
        "n_samples": n,
    }

    logger.info(
        f"Hardware vs Simulator:\n"
        f"  Frobenius error: {results['frobenius_error']:.4f}\n"
        f"  Max element error: {results['max_element_error']:.4f}\n"
        f"  Mean element error: {results['mean_element_error']:.4f}\n"
        f"  Correlation: {results['correlation']:.4f}"
    )

    # Save heatmap
    if save_dir:
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs(save_dir, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.heatmap(K_s, ax=axes[0], cmap="viridis", vmin=0, vmax=1, square=True)
        axes[0].set_title("Simulator Kernel")

        sns.heatmap(K_h, ax=axes[1], cmap="viridis", vmin=0, vmax=1, square=True)
        axes[1].set_title("Hardware Kernel")

        sns.heatmap(diff, ax=axes[2], cmap="Reds", vmin=0, square=True)
        axes[2].set_title(f"|Δ| (Frob error={frob_err:.3f})")

        fig.suptitle("Simulator vs Hardware Kernel Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()

        filepath = os.path.join(save_dir, "hw_vs_sim_comparison")
        fig.savefig(filepath + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(filepath + ".pdf", bbox_inches="tight")
        plt.close(fig)

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  src/hardware_validation.py — Self-test")
    print("=" * 60)

    np.random.seed(config.RANDOM_SEED)
    n = 10

    # Test analysis with synthetic data (no actual IBM connection)
    K_sim = np.eye(n) + np.random.rand(n, n) * 0.3
    K_sim = (K_sim + K_sim.T) / 2
    np.fill_diagonal(K_sim, 1.0)

    # Simulate hardware noise
    noise = np.random.randn(n, n) * 0.05
    K_hw = K_sim + (noise + noise.T) / 2
    K_hw = np.clip(K_hw, 0, 1)
    np.fill_diagonal(K_hw, 1.0)

    results = analyze_hardware_results(K_sim, K_hw)
    print(f"\n  Frobenius error: {results['frobenius_error']:.4f}")
    print(f"  Correlation: {results['correlation']:.4f}")

    assert results["frobenius_error"] < 0.5, "Error too high for mild noise"
    assert results["correlation"] > 0.5, "Correlation too low"

    print("\n  All self-tests passed.")
