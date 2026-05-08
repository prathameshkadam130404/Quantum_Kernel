#!/usr/bin/env python3
"""
EuroSAT FQK + PQK Fixed-Pool Comparison
========================================

Companion to ``scripts/bscm_eurosat_fixed_pool.py`` — runs the same
fixed-pool protocol (POOL_SEED=0, N_POOL=1500, five eval seeds 42..46)
on FQK (Havlicek ZZFeatureMap fidelity) and PQK (SRQFM Bloch-vector
RBF) so all four methods (BSCM-uniform / SRQFM-fid / FQK / PQK) are
evaluated on identical splits with identical SVM-tuning and statistical
protocols.

Kernel topologies
-----------------
  FQK : ZZFeatureMap, CNOT-RZ((pi-x_i)(pi-x_j))-CNOT, **linear** (i,i+1)
        — matches scripts/run_eurosat_experiment.py:build_fqk_kernel.
  PQK : SRQFM Bloch-vector projected kernel
        K(x,x') = exp(-gamma * sum_q ||b_q(x) - b_q(x')||^2 / 2)
        with **linear** connectivity (matches the SRQFM-fid topology in
        bscm_eurosat_fixed_pool.py and the published EuroSAT reference).
        Gamma = config.PQK_GAMMA = 0.67 (project default; same as
        run_eurosat_experiment).

PQK note: PQK is projected, not fidelity, so we DO NOT compute an
N_POOL x N_POOL fidelity gram.  Instead we measure 3*n_q Bloch
expectation values once per pool sample (1500 quantum invocations
total per feature set, vs ~1.1 M for FQK), then build the RBF gram
classically.

Statistics & analyses
---------------------
Identical to bscm_eurosat_fixed_pool.py — same KTA (class-frequency-
normalised), same per-class F1, same pooled stratified bootstrap CIs
(B=1000), same per-seed McNemar (vs RBF-SVM and vs RF), same Wilcoxon
+ Holm-Bonferroni on planned pairs, same permutation-KTA test (n=1000)
on the full 1500-sample pool.

Outputs
-------
  results/eurosat_fqk_pqk_fixed/
      kernels/K_pool_fqk_linear_pca8.npy       (1500 x 1500, float64)
      kernels/K_pool_fqk_linear_physics8.npy
      kernels/bloch_pqk_pca8.npy               (1500, 24)  — 3*n_q Bloch
      kernels/bloch_pqk_physics8.npy
      eurosat_fqk_pqk_fixed_results.json
      run.log

Usage
-----
  python scripts/fqk_pqk_eurosat_fixed_pool.py
  python scripts/fqk_pqk_eurosat_fixed_pool.py --pca-only
  python scripts/fqk_pqk_eurosat_fixed_pool.py --physics-only
  python scripts/fqk_pqk_eurosat_fixed_pool.py --skip-pqk
  python scripts/fqk_pqk_eurosat_fixed_pool.py --skip-fqk
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
from eurosat_data import (  # noqa: E402
    PHYSICS_FEATURE_NAMES,
    load_eurosat_allbands,
)

# Re-use all shared utilities from the BSCM fixed-pool script so
# protocols stay byte-identical.
from bscm_eurosat_fixed_pool import (  # noqa: E402
    POOL_SEED,
    N_TRAIN,
    N_TEST,
    N_POOL,
    EVAL_SEEDS,
    C_GRID,
    ZZ_REPS,
    PCA_DIM,
    _get_device,
    compute_gram_pool,
    evaluate_feature_set,
    perm_kta_pvalue,
    prepare_pool_features,
    spectral_metrics,
    _save as _save_results_helper,  # noqa: F401  (not used; we write our own)
)
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ── Constants specific to this script ─────────────────────────────────────────
FQK_CONNECTIVITY = "linear"             # matches run_eurosat_experiment.py
PQK_CONNECTIVITY = "linear"             # matches SRQFM-fid linear in BSCM fixed-pool
PQK_COUP_THRESH  = 0.0                  # no threshold gating (paper-spec PQK)
PQK_GAMMA        = config.PQK_GAMMA     # 0.67

RESULTS_DIR = Path("results/eurosat_fqk_pqk_fixed")
KERNEL_DIR  = RESULTS_DIR / "kernels"

log = logging.getLogger(__name__)


# ── FQK circuit (Havlicek ZZ feature map, linear connectivity) ────────────────

def make_fqk_circuit(n_q: int):
    """
    Build the FQK fidelity circuit.  Layout matches
    scripts/run_eurosat_experiment.py:build_fqk_kernel exactly:

        for _ in range(REPS):
            H on all qubits
            RZ(x[i]) on qubit i
            for i in 0..n_q-2:
                CNOT(i, i+1); RZ((pi-x[i])(pi-x[i+1]), i+1); CNOT(i, i+1)
        # adjoint of x2

    Returns a QNode taking (x1, x2) and returning probs(0).
    """
    import pennylane as qml
    dev = _get_device(n_q)

    @qml.qnode(dev, diff_method=None)
    def circuit(x1, x2):
        for _ in range(ZZ_REPS):
            for i in range(n_q):
                qml.Hadamard(wires=i)
                qml.RZ(x1[i], wires=i)
            for i in range(n_q - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ((np.pi - x1[i]) * (np.pi - x1[i + 1]), wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
        for _ in range(ZZ_REPS):
            for i in range(n_q - 2, -1, -1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(-(np.pi - x2[i]) * (np.pi - x2[i + 1]), wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_q - 1, -1, -1):
                qml.RZ(-x2[i], wires=i)
                qml.Hadamard(wires=i)
        return qml.probs(wires=range(n_q))

    return circuit


# ── PQK Bloch-vector extraction + RBF gram ────────────────────────────────────

def _resolve_pairs_linear(n_q: int) -> List[Tuple[int, int]]:
    return [(i, i + 1) for i in range(n_q - 1)]


def _apply_srqfm_linear(x, n_q: int):
    """
    Inline SRQFM feature map matching the linear-connectivity SRQFM
    used elsewhere in this protocol (no threshold gating).
    """
    import pennylane as qml
    pairs = _resolve_pairs_linear(n_q)
    for _ in range(ZZ_REPS):
        for i in range(n_q):
            qml.Hadamard(wires=i)
        for i in range(n_q):
            qml.RZ(x[i], wires=i)
        for qi, qj in pairs:
            c = float(np.sin((x[qi] - x[qj]) / 2.0) ** 2)
            if c > PQK_COUP_THRESH:
                qml.IsingZZ(c, wires=[qi, qj])


def compute_pqk_bloch_pool(X_pool: np.ndarray, cache_path: Path) -> np.ndarray:
    """
    Extract per-qubit Bloch vectors (<X>, <Y>, <Z> per qubit) for all
    pool samples.  Returns array of shape (N_POOL, 3 * n_q).
    Cached to .npy.
    """
    import pennylane as qml

    n_q = X_pool.shape[1]
    if cache_path.exists():
        bloch = np.load(cache_path)
        if bloch.shape == (len(X_pool), 3 * n_q):
            log.info("  Loaded cached Bloch vectors from %s", cache_path)
            return bloch
        log.warning("  Cached Bloch has wrong shape %s — recomputing.",
                    bloch.shape)

    dev = _get_device(n_q)

    @qml.qnode(dev, diff_method=None)
    def measure_x(x):
        _apply_srqfm_linear(x, n_q)
        return [qml.expval(qml.PauliX(i)) for i in range(n_q)]

    @qml.qnode(dev, diff_method=None)
    def measure_y(x):
        _apply_srqfm_linear(x, n_q)
        return [qml.expval(qml.PauliY(i)) for i in range(n_q)]

    @qml.qnode(dev, diff_method=None)
    def measure_z(x):
        _apply_srqfm_linear(x, n_q)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

    N = len(X_pool)
    bloch = np.zeros((N, 3 * n_q), dtype=np.float64)
    t0 = time.time()
    for idx in tqdm(range(N), desc=f"  PQK Bloch (N={N}, n_q={n_q})", unit="sample"):
        bx = np.asarray(measure_x(X_pool[idx]))
        by = np.asarray(measure_y(X_pool[idx]))
        bz = np.asarray(measure_z(X_pool[idx]))
        for q in range(n_q):
            bloch[idx, 3 * q]     = bx[q]
            bloch[idx, 3 * q + 1] = by[q]
            bloch[idx, 3 * q + 2] = bz[q]
    elapsed = time.time() - t0
    log.info("  PQK Bloch extraction (%d samples) in %.0f s (%.1f min)",
             N, elapsed, elapsed / 60)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, bloch)

    # Diagnostic: per-qubit Bloch magnitudes
    mags = np.zeros(n_q)
    for q in range(n_q):
        mags[q] = np.sqrt(
            bloch[:, 3 * q] ** 2 +
            bloch[:, 3 * q + 1] ** 2 +
            bloch[:, 3 * q + 2] ** 2
        ).mean()
    log.info("  Bloch magnitudes per qubit: %s (mean=%.3f)",
             np.round(mags, 3).tolist(), float(mags.mean()))
    return bloch


def pqk_gram_from_bloch(bloch: np.ndarray, n_q: int, gamma: float) -> np.ndarray:
    """
    K(x,x') = exp(-gamma * 0.5 * sum_q ||b_q(x) - b_q(x')||^2)
    Vectorised; symmetric N x N output with K[i,i] = 1.
    """
    N = len(bloch)
    b = bloch.reshape(N, n_q, 3)
    K = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        diff = b[i] - b              # (N, n_q, 3)
        sq   = (diff * diff).sum(axis=2)   # (N, n_q)
        d    = 0.5 * sq.sum(axis=1)        # (N,)
        K[i] = np.exp(-gamma * d)
    np.fill_diagonal(K, 1.0)
    K = (K + K.T) / 2.0  # numerical symmetrisation
    return K


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EuroSAT FQK + PQK Fixed-Pool Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default="data/raw/eurosat/EuroSATallBands")
    parser.add_argument("--pca-only",     action="store_true")
    parser.add_argument("--physics-only", action="store_true")
    parser.add_argument("--skip-fqk",     action="store_true")
    parser.add_argument("--skip-pqk",     action="store_true")
    args = parser.parse_args()

    if args.skip_fqk and args.skip_pqk:
        raise SystemExit("Nothing to do — both --skip-fqk and --skip-pqk set.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(RESULTS_DIR / "run.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), fh],
    )

    t_wall = time.time()
    log.info("=" * 72)
    log.info("  EuroSAT FQK + PQK Fixed-Pool Comparison")
    log.info("=" * 72)

    # Data + fixed pool
    band_means, labels, class_names = load_eurosat_allbands(
        args.data_dir,
        cache_path="data/processed/eurosat_band_means.npz",
    )
    log.info("EuroSAT: %d samples, %d classes", len(labels), len(class_names))

    pool_sss = StratifiedShuffleSplit(
        n_splits=1, train_size=N_TRAIN, test_size=N_TEST,
        random_state=POOL_SEED,
    )
    canonical_tr_full, canonical_te_full = next(pool_sss.split(band_means, labels))
    pool_idx           = np.concatenate([canonical_tr_full, canonical_te_full])
    pool_labels        = labels[pool_idx]
    bm_pool            = band_means[pool_idx]
    canonical_tr_local = np.arange(N_TRAIN)

    dist = dict(zip(class_names, np.bincount(pool_labels).tolist()))
    log.info(
        "Fixed pool: %d samples (pool_seed=%d)\n  Class distribution: %s",
        N_POOL, POOL_SEED, dist,
    )

    log.info("\nPreparing pool features...")
    feats = prepare_pool_features(bm_pool, canonical_tr_local)
    log.info("  PCA-8 explained variance: %.3f", feats["pca_explained_var"])
    log.info("  Physics-8: %s", PHYSICS_FEATURE_NAMES)

    feat_sets: List[Tuple[str, np.ndarray]] = []
    if not args.physics_only:
        feat_sets.append(("pca8",     feats["pca8"]))
    if not args.pca_only:
        feat_sets.append(("physics8", feats["physics8"]))

    run_fqk = not args.skip_fqk
    run_pqk = not args.skip_pqk

    all_results = {
        "experiment":   "EuroSAT FQK + PQK Fixed-Pool Comparison",
        "dataset":      "EuroSAT (Helber et al. 2019) — 13 Sentinel-2 bands",
        "n_classes":    len(class_names),
        "class_names":  class_names,
        "pool_config": {
            "pool_seed":    POOL_SEED,
            "n_pool":       N_POOL,
            "n_train":      N_TRAIN,
            "n_test":       N_TEST,
            "eval_seeds":   EVAL_SEEDS,
            "feature_fit":  "fitted on canonical training portion (pool[:1000])",
            "shared_with":  "results/eurosat_bscm_fixed/ (identical pool & splits)",
        },
        "kernel_config": {
            "fqk_connectivity":  FQK_CONNECTIVITY,
            "pqk_connectivity":  PQK_CONNECTIVITY,
            "pqk_gamma":         PQK_GAMMA,
            "pqk_coup_thresh":   PQK_COUP_THRESH,
            "zz_reps":           ZZ_REPS,
            "c_grid":            C_GRID,
            "topology_note":     (
                "FQK uses linear (i,i+1) ZZ connectivity matching "
                "scripts/run_eurosat_experiment.py.  PQK uses linear "
                "SRQFM connectivity matching the SRQFM-fid topology "
                "in bscm_eurosat_fixed_pool.py for fairness."
            ),
        },
        "physics_features": PHYSICS_FEATURE_NAMES,
        "feature_sets": {},
    }

    for feat_name, X_pool in feat_sets:
        n_q = X_pool.shape[1]
        log.info("\n%s\n  FEATURE SET: %s  (n_qubits=%d)\n%s",
                 "=" * 72, feat_name.upper(), n_q, "=" * 72)

        gram_matrices: Dict[str, np.ndarray] = {}
        global_specs: Dict[str, dict] = {}

        # FQK gram (fidelity, full pairwise — slow)
        if run_fqk:
            log.info("\n  [FQK] ZZFeatureMap fidelity kernel (linear)...")
            circ_fqk = make_fqk_circuit(n_q)
            K_fqk = compute_gram_pool(
                circ_fqk, X_pool,
                cache_path=KERNEL_DIR /
                    f"K_pool_fqk_{FQK_CONNECTIVITY}_{feat_name}.npy",
                label=f"FQK|{feat_name}",
            )
            gram_matrices["FQK"] = K_fqk
            log.info("  Global FQK spectral metrics (N=%d):", N_POOL)
            global_specs["FQK"] = spectral_metrics(K_fqk)
            for k, v in global_specs["FQK"].items():
                log.info("    %s = %.4f", k, v)

        # PQK gram (Bloch-vector RBF — fast: only 1500 quantum invocations)
        if run_pqk:
            log.info("\n  [PQK] SRQFM Bloch-vector projected kernel "
                     "(linear, gamma=%.3f)...", PQK_GAMMA)
            bloch = compute_pqk_bloch_pool(
                X_pool,
                cache_path=KERNEL_DIR / f"bloch_pqk_{feat_name}.npy",
            )
            K_pqk = pqk_gram_from_bloch(bloch, n_q, PQK_GAMMA)
            gram_matrices["PQK"] = K_pqk
            log.info("  Global PQK spectral metrics (N=%d):", N_POOL)
            global_specs["PQK"] = spectral_metrics(K_pqk)
            for k, v in global_specs["PQK"].items():
                log.info("    %s = %.4f", k, v)

        # Permutation KTA on full pool
        log.info("\n  Permutation KTA test (n_perm=1000) on full pool...")
        perm_results: Dict[str, dict] = {}
        for kname, K_g in gram_matrices.items():
            t_p = time.time()
            perm = perm_kta_pvalue(
                K_g, pool_labels, n_perm=1000, rng_seed=0, sub_n=None,
            )
            log.info(
                "    %-5s  KTA(cc)=%.4f  null=%.4f±%.4f  p=%.4f  (%.1fs)",
                kname, perm["kta_cc_observed"],
                perm["null_mean"], perm["null_std"], perm["p_value"],
                time.time() - t_p,
            )
            perm_results[kname] = perm

        # Per-seed evaluation (re-uses the BSCM-fixed-pool harness; pass
        # run_srqfm=False so the cross-kernel BSCM-vs-SRQFM block is skipped).
        # Build planned comparison pairs dynamically from actual kernel names
        # so that Wilcoxon and bootstrap use the correct keys (FQK/PQK, not
        # the BSCM-specific defaults).
        _knames = list(gram_matrices)
        _pairs: List[Tuple[str, str]] = []
        for _kn in _knames:
            _pairs += [(_kn, "RBF-SVM"), (_kn, "RF")]
        if len(_knames) == 2:
            _pairs.append((_knames[0], _knames[1]))

        log.info("\n  Evaluating on %d within-pool splits...", len(EVAL_SEEDS))
        feat_entry = evaluate_feature_set(
            feat_name, X_pool, pool_labels, gram_matrices,
            run_srqfm=False,
            planned_pairs=_pairs,
            bootstrap_pairs=_pairs,
        )
        feat_entry["global_spectral"] = global_specs
        feat_entry["permutation_kta"] = perm_results

        all_results["feature_sets"][feat_name] = feat_entry
        all_results["wall_clock_s"] = round(time.time() - t_wall, 1)
        _save(all_results)

    all_results["wall_clock_s"] = round(time.time() - t_wall, 1)
    _save(all_results)

    log.info("\n%s", "=" * 72)
    log.info("  DONE.  Wall clock: %.1f min", (time.time() - t_wall) / 60)
    log.info("  Results: %s", RESULTS_DIR / "eurosat_fqk_pqk_fixed_results.json")
    log.info("%s", "=" * 72)


def _save(results: dict):
    out = RESULTS_DIR / "eurosat_fqk_pqk_fixed_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
