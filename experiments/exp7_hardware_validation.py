"""
Experiment 7: IBM Hardware Validation.

Two-stage approach: pilot (10 samples) → full (15+10 samples).
Compares g_hardware vs g_simulator, SVM accuracy.

Phase 8. Expected runtime: 30-60 min (IBM queue).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import config
from src.quantum_kernels import compute_fqk_kernel_matrix
from src.hardware_validation import run_pilot_stage, run_full_stage, analyze_hardware_results
from src.geometric_difference import compute_geometric_difference
from src.classical_kernels import compute_rbf_kernel
from src.classifiers import train_precomputed_svm, evaluate_classifier
from src.data_loader import load_modality, load_subsample
from src.utils import save_results, setup_logging, Timer, check_and_skip, ensure_dir

logger = setup_logging("exp7", log_file=os.path.join(config.RESULTS_DIR, "exp7.log"))


def run_experiment():
    """Hardware validation experiment."""
    results_dir = ensure_dir(os.path.join(config.RESULTS_DIR, "hardware"))
    results_file = os.path.join(results_dir, "exp7_results.json")

    if check_and_skip(results_file, logger):
        return

    np.random.seed(config.RANDOM_SEED)

    # Check IBM credentials
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(channel=config.IBM_QUANTUM_CHANNEL)
        logger.info("IBM Quantum credentials verified")
    except Exception as e:
        logger.error(f"IBM Quantum not available: {e}")
        logger.error("Skipping hardware experiment. Run offline analysis only.")

        # Create placeholder results
        results = {"status": "SKIPPED", "reason": str(e)}
        save_results(results, results_file)
        return

    # Load data
    with Timer("Loading data", logger=logger):
        subsample = load_subsample()
        X_train = subsample["fused_X_train"]
        X_test = subsample["fused_X_test"]
        y_train = subsample["y_train"]
        y_test = subsample["y_test"]

    results = {}

    # Stage 1: Pilot
    logger.info("\n--- Stage 1: Pilot ---")
    with Timer("Pilot stage", logger=logger):
        K_pilot_hw = run_pilot_stage(X_train, save_dir=results_dir)

    # Simulator pilot
    K_pilot_sim = compute_fqk_kernel_matrix(
        X_train[:config.IBM_PILOT_N_TRAIN],
        save_path=os.path.join(results_dir, "pilot_kernel_sim.npy")
    )

    pilot_analysis = analyze_hardware_results(K_pilot_sim, K_pilot_hw, save_dir=results_dir)
    results["pilot"] = pilot_analysis
    logger.info(f"Pilot Frobenius error: {pilot_analysis['frobenius_error']:.4f}")

    # Stage 2: Full (only if pilot looks reasonable)
    if pilot_analysis["frobenius_error"] < 0.5:
        logger.info("\n--- Stage 2: Full ---")
        with Timer("Full stage", logger=logger):
            K_full_train_hw, K_full_test_hw = run_full_stage(
                X_train, X_test, save_dir=results_dir
            )

        # Simulator full
        n_tr = config.IBM_FULL_N_TRAIN
        n_te = config.IBM_FULL_N_TEST
        K_full_train_sim = compute_fqk_kernel_matrix(
            X_train[:n_tr],
            save_path=os.path.join(results_dir, "full_train_kernel_sim.npy")
        )

        full_analysis = analyze_hardware_results(K_full_train_sim, K_full_train_hw, save_dir=results_dir)
        results["full"] = full_analysis

        # g comparison
        K_rbf = compute_rbf_kernel(X_train[:n_tr])
        g_sim, _ = compute_geometric_difference(K_full_train_sim, K_rbf)
        g_hw, _ = compute_geometric_difference(K_full_train_hw, K_rbf)

        results["g_simulator"] = g_sim
        results["g_hardware"] = g_hw
        results["g_relative_error"] = abs(g_sim - g_hw) / g_sim if g_sim > 0 else 0

        logger.info(f"g_simulator = {g_sim:.4f}, g_hardware = {g_hw:.4f}")
        logger.info(f"g relative error = {results['g_relative_error']:.4f}")

        # SVM accuracy comparison
        clf_hw = train_precomputed_svm(K_full_train_hw, y_train[:n_tr])
        m_hw = evaluate_classifier(clf_hw, K_full_test_hw, y_test[:n_te], "HW-FQK-SVM")

        clf_sim = train_precomputed_svm(K_full_train_sim, y_train[:n_tr])
        K_sim_test = compute_fqk_kernel_matrix(X_train[:n_tr], X2=X_test[:n_te])
        m_sim = evaluate_classifier(clf_sim, K_sim_test, y_test[:n_te], "Sim-FQK-SVM")

        results["svm_hw_f1"] = m_hw["macro_f1"]
        results["svm_sim_f1"] = m_sim["macro_f1"]
    else:
        logger.warning(f"Pilot error too high ({pilot_analysis['frobenius_error']:.4f}). Skipping full stage.")
        results["full"] = {"status": "SKIPPED", "reason": "Pilot error too high"}

    save_results(results, results_file)
    logger.info(f"\nResults saved: {results_file}")


if __name__ == "__main__":
    run_experiment()
