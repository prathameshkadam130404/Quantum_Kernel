"""
Master orchestration script: runs all experiments sequentially.

Usage:
    python run_all.py                 # Run everything
    python run_all.py --skip 7        # Skip exp7 (hardware)
    python run_all.py --only 1 2 3    # Run only exp1, exp2, exp3
"""

import os
import sys
import argparse
import time
import logging
import importlib

sys.path.insert(0, os.path.dirname(__file__))
import config
from src.utils import setup_logging

logger = setup_logging(
    "run_all", log_file=os.path.join(config.RESULTS_DIR, "run_all.log")
)

EXPERIMENT_MODULES = {
    1: "experiments.exp1_geometric_analysis",
    2: "experiments.exp2_kernel_classification",
    3: "experiments.exp3_fewshot_curves",
    4: "experiments.exp4_topological_boost",
    5: "experiments.exp5_equivariant_comparison",
    6: "experiments.exp6_dequantization_defense",
    7: "experiments.exp7_hardware_validation",
    8: "experiments.exp8_multidataset",
    9: "experiments.exp9_full_ablation",
    10: "experiments.exp10_controlled_mi",
    11: "experiments.exp11_expressibility",
    # CV experiments (new)
    12: "experiments.exp1_pca_cv",
    13: "experiments.exp1_physics_cv",
    14: "experiments.exp10_controlled_mi_v2_opt",
}


def main():
    parser = argparse.ArgumentParser(description="Run quantum-sat experiments.")
    parser.add_argument(
        "--skip", nargs="*", type=int, default=[], help="Experiments to skip"
    )
    parser.add_argument(
        "--only", nargs="*", type=int, default=[], help="Run only these experiments"
    )
    args = parser.parse_args()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if args.only:
        exp_ids = args.only
    else:
        exp_ids = [i for i in sorted(EXPERIMENT_MODULES.keys()) if i not in args.skip]

    logger.info("=" * 60)
    logger.info("  Quantum-Sat Classification — Full Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Experiments to run: {exp_ids}")
    logger.info(f"  Results directory: {config.RESULTS_DIR}")

    results = {}
    for exp_id in exp_ids:
        module_name = EXPERIMENT_MODULES.get(exp_id)
        if not module_name:
            logger.warning(f"Unknown experiment: {exp_id}")
            continue

        logger.info(f"\n{'=' * 60}")
        logger.info(f"  Experiment {exp_id}: {module_name}")
        logger.info(f"{'=' * 60}")

        t0 = time.time()
        try:
            mod = importlib.import_module(module_name)
            mod.run_experiment()
            elapsed = time.time() - t0
            results[exp_id] = {"status": "SUCCESS", "time_s": elapsed}
            logger.info(f"  ✓ Exp {exp_id} completed in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            results[exp_id] = {"status": "FAILED", "error": str(e), "time_s": elapsed}
            logger.error(f"  ✗ Exp {exp_id} FAILED after {elapsed:.1f}s: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("  Pipeline Summary")
    logger.info(f"{'=' * 60}")
    total_time = sum(r["time_s"] for r in results.values())
    for eid, r in sorted(results.items()):
        logger.info(f"  Exp {eid:2d}: {r['status']:8s} ({r['time_s']:.1f}s)")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time / 3600:.1f}h)")

    n_ok = sum(1 for r in results.values() if r["status"] == "SUCCESS")
    n_fail = sum(1 for r in results.values() if r["status"] == "FAILED")
    logger.info(f"  Success: {n_ok}/{len(results)}, Failed: {n_fail}/{len(results)}")


if __name__ == "__main__":
    main()
