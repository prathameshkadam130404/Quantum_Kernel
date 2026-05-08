#!/usr/bin/env python3
"""
EuroSAT Cross-Dataset Validation (13-Band + Physics Features)
=============================================================

Validates quantum kernel findings from So2Sat LCZ42 on EuroSAT (10-class
Sentinel-2 satellite imagery) using both PCA-8 and physics-derived features
from the full 13-band multispectral data.

This experiment directly addresses the "single dataset" limitation noted in
the paper (Section 6, Limitations). It tests Claims 1, 2, 5, and 8.

Usage:
    python scripts/run_eurosat_experiment.py [--n-train 1000] [--n-seeds 5]
    python scripts/run_eurosat_experiment.py --skip-quantum   # classical only
    python scripts/run_eurosat_experiment.py --n-train 50 --n-seeds 1  # quick test

Output:
    results/eurosat/eurosat_cross_dataset_results.json
    results/eurosat/kernels/  (checkpointed kernel matrices)
"""

import argparse
import json
import os
import sys
import time
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import f1_score, balanced_accuracy_score
from scipy.stats import wilcoxon, chi2

warnings.filterwarnings("ignore", category=UserWarning)

# Add scripts dir to path
sys.path.insert(0, os.path.dirname(__file__))

from eurosat_data import (
    load_eurosat_allbands, prepare_split_features,
    EUROSAT_CLASSES, PHYSICS_FEATURE_NAMES,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

RESULTS_DIR = Path("results/eurosat")
KERNEL_CACHE_DIR = RESULTS_DIR / "kernels"
FQK_REPS = 2


# ── Quantum Kernels ─────────────────────────────────────────────────────────

def build_fqk_kernel(X_train, X_test):
    """FQK: ZZFeatureMap fidelity kernel with tqdm progress."""
    import pennylane as qml

    n_q = X_train.shape[1]
    dev = qml.device("default.qubit", wires=n_q)

    @qml.qnode(dev)
    def circuit(x1, x2):
        for _ in range(FQK_REPS):
            for i in range(n_q):
                qml.Hadamard(wires=i)
                qml.RZ(x1[i], wires=i)
            for i in range(n_q - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ((np.pi - x1[i]) * (np.pi - x1[i + 1]), wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
        for _ in range(FQK_REPS):
            for i in range(n_q - 2, -1, -1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(-(np.pi - x2[i]) * (np.pi - x2[i + 1]), wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_q - 1, -1, -1):
                qml.RZ(-x2[i], wires=i)
                qml.Hadamard(wires=i)
        return qml.probs(wires=range(n_q))

    def _gram(Xa, Xb, desc="FQK"):
        na, nb = len(Xa), len(Xb)
        K = np.zeros((na, nb))
        pbar = tqdm(total=na * nb, desc=f"    {desc} ({na}x{nb})", unit="eval")
        for i in range(na):
            for j in range(nb):
                K[i, j] = circuit(Xa[i], Xb[j])[0]
                pbar.update(1)
        pbar.close()
        return K

    K_train = _gram(X_train, X_train, desc="FQK K_train")
    K_test = _gram(X_test, X_train, desc="FQK K_test")
    return K_train, K_test


def build_srqfm_kernel(X_train, X_test):
    """SRQFM: sin²-coupled fidelity kernel with tqdm progress."""
    import pennylane as qml

    n_q = X_train.shape[1]
    dev = qml.device("default.qubit", wires=n_q)

    @qml.qnode(dev)
    def circuit(x1, x2):
        for _ in range(FQK_REPS):
            for i in range(n_q):
                qml.Hadamard(wires=i)
                qml.RZ(x1[i], wires=i)
            for i in range(n_q - 1):
                c = np.sin((x1[i] - x1[i + 1]) / 2.0) ** 2
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(c, wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
        for _ in range(FQK_REPS):
            for i in range(n_q - 2, -1, -1):
                c = np.sin((x2[i] - x2[i + 1]) / 2.0) ** 2
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(-c, wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_q - 1, -1, -1):
                qml.RZ(-x2[i], wires=i)
                qml.Hadamard(wires=i)
        return qml.probs(wires=range(n_q))

    def _gram(Xa, Xb, desc="SRQFM"):
        na, nb = len(Xa), len(Xb)
        K = np.zeros((na, nb))
        pbar = tqdm(total=na * nb, desc=f"    {desc} ({na}x{nb})", unit="eval")
        for i in range(na):
            for j in range(nb):
                K[i, j] = circuit(Xa[i], Xb[j])[0]
                pbar.update(1)
        pbar.close()
        return K

    K_train = _gram(X_train, X_train, desc="SRQFM K_train")
    K_test = _gram(X_test, X_train, desc="SRQFM K_test")
    return K_train, K_test


# ── Classical Baselines ──────────────────────────────────────────────────────

def run_rbf_svm(X_tr, y_tr, X_te, y_te):
    """RBF-SVM with grid-searched C and gamma."""
    grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", random_state=42),
        {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1.0]},
        cv=3, scoring="f1_macro", n_jobs=-1, refit=True,
    )
    grid.fit(X_tr, y_tr)
    y_pred = grid.predict(X_te)
    return {
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_te, y_pred)),
        "best_params": {k: str(v) for k, v in grid.best_params_.items()},
        "predictions": y_pred.tolist(),
    }


def run_rf(X_tr, y_tr, X_te, y_te):
    """Random Forest baseline."""
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=5,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    return {
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_te, y_pred)),
        "predictions": y_pred.tolist(),
    }


def run_quantum_svm(K_tr, y_tr, K_te, y_te):
    """Precomputed-kernel SVM with C-tuning."""
    best_f1, best = -1, None
    for C in [0.1, 1, 10, 100]:
        svm = SVC(kernel="precomputed", C=C, class_weight="balanced")
        svm.fit(K_tr, y_tr)
        yp = svm.predict(K_te)
        f1 = f1_score(y_te, yp, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best = {
                "f1_macro": float(f1),
                "balanced_acc": float(balanced_accuracy_score(y_te, yp)),
                "best_C": float(C),
                "predictions": yp.tolist(),
            }
    return best


# ── Statistical Tests ────────────────────────────────────────────────────────

def mcnemar_test(y_true, y_a, y_b):
    """McNemar's test with continuity correction."""
    ca, cb = (y_a == y_true), (y_b == y_true)
    b01 = int(np.sum(ca & ~cb))
    b10 = int(np.sum(~ca & cb))
    if b01 + b10 == 0:
        return {"chi2": 0.0, "p_value": 1.0}
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p = 1 - chi2.cdf(stat, df=1)
    return {"chi2": float(stat), "p_value": float(p)}


def compute_kta(K, y):
    """Centered kernel-target alignment."""
    Y = (y[:, None] == y[None, :]).astype(float)
    Y_c = Y - Y.mean()
    K_c = K - K.mean()
    num = np.sum(K_c * Y_c)
    denom = np.sqrt(np.sum(K_c ** 2) * np.sum(Y_c ** 2))
    return float(num / max(denom, 1e-12))


def spectral_metrics(K):
    """Off-diagonal variance and effective rank."""
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = K[mask]
    Ksym = (K + K.T) / 2.0
    eig = np.clip(np.linalg.eigvalsh(Ksym), 1e-12, None)
    p = eig / eig.sum()
    return {
        "off_diag_mean": float(off.mean()),
        "off_diag_var": float(off.var()),
        "eff_rank_shannon": float(np.exp(-np.sum(p * np.log(p + 1e-12)))),
        "eff_rank_huang": float(eig.sum() / eig.max()),
    }


# ── Kernel Checkpointing ────────────────────────────────────────────────────

def save_kernel(K, name, seed, feat_set):
    """Save kernel matrix to disk for crash recovery."""
    KERNEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = KERNEL_CACHE_DIR / f"{name}_{feat_set}_seed{seed}.npy"
    np.save(path, K)
    return str(path)


def load_kernel(name, seed, feat_set):
    """Load cached kernel matrix if it exists."""
    path = KERNEL_CACHE_DIR / f"{name}_{feat_set}_seed{seed}.npy"
    if path.exists():
        return np.load(path)
    return None


# ── Main Experiment ──────────────────────────────────────────────────────────

def run_feature_set(
    feat_name, X_tr, y_tr, X_te, y_te, seed, run_quantum=True,
):
    """Run all methods on one feature set for one seed."""
    logger = logging.getLogger(__name__)
    sr = {}

    # Classical baselines
    logger.info(f"  [{feat_name}] Classical baselines...")
    t0 = time.time()
    sr["RBF-SVM"] = run_rbf_svm(X_tr, y_tr, X_te, y_te)
    sr["RF"] = run_rf(X_tr, y_tr, X_te, y_te)
    sr["classical_time_s"] = round(time.time() - t0, 1)
    logger.info(f"    RBF-SVM: F1={sr['RBF-SVM']['f1_macro']:.4f}  RF: F1={sr['RF']['f1_macro']:.4f}")

    if not run_quantum:
        return sr

    # FQK
    K_cached = load_kernel("fqk_train", seed, feat_name)
    K_cached_te = load_kernel("fqk_test", seed, feat_name)
    if K_cached is not None and K_cached_te is not None:
        logger.info(f"  [{feat_name}] FQK: loaded from cache")
        K_tr_fqk, K_te_fqk = K_cached, K_cached_te
    else:
        logger.info(f"  [{feat_name}] Computing FQK kernel...")
        t0 = time.time()
        K_tr_fqk, K_te_fqk = build_fqk_kernel(X_tr, X_te)
        save_kernel(K_tr_fqk, "fqk_train", seed, feat_name)
        save_kernel(K_te_fqk, "fqk_test", seed, feat_name)
        logger.info(f"    FQK computed in {time.time()-t0:.0f}s (checkpointed)")

    sr["FQK"] = run_quantum_svm(K_tr_fqk, y_tr, K_te_fqk, y_te)
    sr["FQK"]["kta"] = compute_kta(K_tr_fqk, y_tr)
    sr["FQK"]["spectral"] = spectral_metrics(K_tr_fqk)
    logger.info(f"    FQK-SVM:   F1={sr['FQK']['f1_macro']:.4f}  KTA={sr['FQK']['kta']:.3f}")

    # SRQFM
    K_cached = load_kernel("srqfm_train", seed, feat_name)
    K_cached_te = load_kernel("srqfm_test", seed, feat_name)
    if K_cached is not None and K_cached_te is not None:
        logger.info(f"  [{feat_name}] SRQFM: loaded from cache")
        K_tr_sr, K_te_sr = K_cached, K_cached_te
    else:
        logger.info(f"  [{feat_name}] Computing SRQFM kernel...")
        t0 = time.time()
        K_tr_sr, K_te_sr = build_srqfm_kernel(X_tr, X_te)
        save_kernel(K_tr_sr, "srqfm_train", seed, feat_name)
        save_kernel(K_te_sr, "srqfm_test", seed, feat_name)
        logger.info(f"    SRQFM computed in {time.time()-t0:.0f}s (checkpointed)")

    sr["SRQFM"] = run_quantum_svm(K_tr_sr, y_tr, K_te_sr, y_te)
    sr["SRQFM"]["kta"] = compute_kta(K_tr_sr, y_tr)
    sr["SRQFM"]["spectral"] = spectral_metrics(K_tr_sr)
    logger.info(f"    SRQFM-SVM: F1={sr['SRQFM']['f1_macro']:.4f}  KTA={sr['SRQFM']['kta']:.3f}")

    # McNemar tests vs RBF-SVM
    rbf_p = np.array(sr["RBF-SVM"]["predictions"])
    for qk in ["FQK", "SRQFM"]:
        qk_p = np.array(sr[qk]["predictions"])
        sr[f"McNemar_{qk}_vs_RBF"] = mcnemar_test(y_te, qk_p, rbf_p)

    return sr


def main():
    parser = argparse.ArgumentParser(
        description="EuroSAT Cross-Dataset Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--data-dir", default="data/raw/eurosat/EuroSATallBands")
    parser.add_argument("--skip-quantum", action="store_true")
    parser.add_argument("--pca-only", action="store_true")
    parser.add_argument("--physics-only", action="store_true")
    args = parser.parse_args()

    # Setup file + console logging
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(RESULTS_DIR / "experiment.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("EuroSAT Cross-Dataset Validation Experiment")
    logger.info("=" * 60)
    logger.info(f"Config: n_train={args.n_train}, n_test={args.n_test}, "
                f"n_seeds={args.n_seeds}, skip_quantum={args.skip_quantum}")

    t_start = time.time()
    seeds = [args.seed_base + i for i in range(args.n_seeds)]

    # Load data (with caching)
    cache_path = "data/processed/eurosat_band_means.npz"
    band_means, labels, class_names = load_eurosat_allbands(
        args.data_dir, cache_path=cache_path,
    )

    all_results = {
        "experiment": "EuroSAT Cross-Dataset Validation (13-Band + Physics)",
        "dataset": "EuroSAT (Helber et al., 2019) — 13 Sentinel-2 bands",
        "n_classes": len(class_names),
        "class_names": class_names,
        "config": vars(args),
        "physics_features": PHYSICS_FEATURE_NAMES,
    }

    run_quantum = not args.skip_quantum
    feat_sets = []
    if not args.physics_only:
        feat_sets.append("pca8")
    if not args.pca_only:
        feat_sets.append("physics8")

    for feat_name in feat_sets:
        logger.info(f"\n{'='*60}")
        logger.info(f"FEATURE SET: {feat_name.upper()}")
        logger.info(f"{'='*60}")

        method_f1s = {"FQK": [], "SRQFM": [], "RBF-SVM": [], "RF": []}
        seed_results = []

        for seed_idx, seed in enumerate(seeds):
            logger.info(f"\n--- Seed {seed_idx+1}/{len(seeds)} (seed={seed}) ---")

            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=args.n_train,
                test_size=args.n_test, random_state=seed,
            )
            tr_idx, te_idx = next(splitter.split(band_means, labels))

            # Prepare features with TRAIN-ONLY fitting (no data leakage)
            feats = prepare_split_features(
                band_means[tr_idx], band_means[te_idx],
                pca_dim=8, seed=seed,
            )

            if feat_name == "pca8":
                X_tr = feats["pca8_train"]
                X_te = feats["pca8_test"]
                logger.info(f"  PCA explained var: {feats['pca_explained_var']:.3f}")
            else:
                X_tr = feats["physics8_train"]
                X_te = feats["physics8_test"]

            y_tr, y_te = labels[tr_idx], labels[te_idx]
            logger.info(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

            sr = run_feature_set(feat_name, X_tr, y_tr, X_te, y_te, seed, run_quantum)
            sr["seed"] = seed

            # Collect F1 values
            for m in method_f1s:
                if m in sr:
                    method_f1s[m].append(sr[m]["f1_macro"])

            # Strip predictions for JSON size
            for m in ["FQK", "SRQFM", "RBF-SVM", "RF"]:
                if m in sr and "predictions" in sr[m]:
                    del sr[m]["predictions"]

            seed_results.append(sr)

            # Save incremental results after each seed
            all_results[feat_name] = {"seeds": seed_results}
            _save_results(all_results, t_start)

        # Aggregate summary
        summary = {}
        for m, f1s in method_f1s.items():
            if f1s:
                arr = np.array(f1s)
                summary[m] = {
                    "f1_mean": round(float(arr.mean()), 4),
                    "f1_std": round(float(arr.std()), 4),
                    "f1_values": [round(v, 4) for v in arr.tolist()],
                }

        # Wilcoxon tests (need >= 5 seeds)
        if len(method_f1s.get("RBF-SVM", [])) >= 5 and run_quantum:
            rbf_arr = np.array(method_f1s["RBF-SVM"])
            for qk in ["FQK", "SRQFM"]:
                if len(method_f1s.get(qk, [])) >= 5:
                    try:
                        stat, p = wilcoxon(np.array(method_f1s[qk]), rbf_arr)
                        summary[f"Wilcoxon_{qk}_vs_RBF"] = {
                            "statistic": float(stat), "p_value": float(p),
                        }
                    except ValueError:
                        pass

        all_results[feat_name]["summary"] = summary

        logger.info(f"\n  Summary ({feat_name.upper()}):")
        for m, s in summary.items():
            if isinstance(s, dict) and "f1_mean" in s:
                logger.info(f"    {m:15s}: F1 = {s['f1_mean']:.4f} ± {s['f1_std']:.4f}")

    all_results["total_runtime_s"] = round(time.time() - t_start, 1)
    _save_results(all_results, t_start)

    logger.info(f"\nTotal runtime: {all_results['total_runtime_s']:.0f}s")
    logger.info(f"Results saved to: {RESULTS_DIR / 'eurosat_cross_dataset_results.json'}")


def _save_results(results, t_start):
    """Save results JSON (called after each seed for crash resilience)."""
    results["total_runtime_s"] = round(time.time() - t_start, 1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "eurosat_cross_dataset_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
