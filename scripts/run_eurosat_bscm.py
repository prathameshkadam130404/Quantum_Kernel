#!/usr/bin/env python3
"""
EuroSAT Cross-Dataset Validation -- BSCM Edition (READY, NOT RUN)

Mirrors scripts/run_eurosat_experiment.py but adds the Bell-Spectrum
Coupling Map (BSCM) fidelity kernel as a quantum method alongside FQK
and SRQFM.  Uses the locked-tau hyperparameter from
results/bscm/locked_tau.json (which was selected on a So2Sat held-out
pool that has zero overlap with EuroSAT, so applying it to EuroSAT is
a genuine cross-dataset transfer test).

Differences vs the original EuroSAT script:
    - All-to-all connectivity for both BSCM and SRQFM (the original
      script used linear connectivity for SRQFM; we standardise so the
      cross-dataset comparison is BSCM-fidelity vs SRQFM-fidelity at
      the same topology used in So2Sat).
    - BSCM uses uniform Bell-prior weights and the locked tau.
    - Output goes to a separate JSON to not clobber the existing run.

Usage (when ready):
    python scripts/run_eurosat_bscm.py [--n-train 1000] [--n-seeds 5]
    python scripts/run_eurosat_bscm.py --pca-only       # PCA8 only
    python scripts/run_eurosat_bscm.py --physics-only   # physics8 only
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
from eurosat_data import (  # noqa: E402
    EUROSAT_CLASSES,
    PHYSICS_FEATURE_NAMES,
    load_eurosat_allbands,
    prepare_split_features,
)
from src.bscm_kernel import (  # noqa: E402
    BELL_WEIGHT_PRESETS,
    apply_bscm_feature_map,
    apply_bscm_feature_map_adjoint,
)
from src.srqfm_fidelity_kernel import (  # noqa: E402
    _apply_srqfm,
    _apply_srqfm_adjoint,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # noqa: ARG001
        return it

RESULTS_DIR = Path("results/eurosat_bscm")
KERNEL_CACHE_DIR = RESULTS_DIR / "kernels"
LOCKED_TAU_PATH = Path("results/bscm/locked_tau.json")


# ----- Locked tau lookup ---------------------------------------------------

def get_locked_tau() -> float:
    if not LOCKED_TAU_PATH.exists():
        raise RuntimeError(
            f"locked_tau.json not found at {LOCKED_TAU_PATH}.  "
            f"Run scripts/bscm_phase2_lockdown.py first."
        )
    with open(LOCKED_TAU_PATH) as f:
        return float(json.load(f)["locked"]["tau"])


# ----- Quantum kernels (true fidelity, paired QNode) ------------------------

def _build_paired_circuit(n_q, apply_forward, apply_adjoint):
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_q)

    @qml.qnode(dev, diff_method=None)
    def circuit(x1, x2):
        apply_forward(x1)
        apply_adjoint(x2)
        return qml.probs(wires=range(n_q))

    return circuit


def build_bscm_kernel(X_train, X_test, tau, prior="uniform"):
    n_q = X_train.shape[1]
    w = BELL_WEIGHT_PRESETS[prior]
    fwd = lambda x: apply_bscm_feature_map(
        x, n_qubits=n_q, reps=config.ZZ_REPS, tau=tau,
        coupling_threshold=1e-4, connectivity="all", bell_weights=w,
    )
    adj = lambda x: apply_bscm_feature_map_adjoint(
        x, n_qubits=n_q, reps=config.ZZ_REPS, tau=tau,
        coupling_threshold=1e-4, connectivity="all", bell_weights=w,
    )
    circuit = _build_paired_circuit(n_q, fwd, adj)

    def gram(Xa, Xb, label):
        na, nb = len(Xa), len(Xb)
        K = np.zeros((na, nb))
        pbar = tqdm(total=na * nb, desc=f"    BSCM-{prior} {label} ({na}x{nb})",
                     unit="eval")
        for i in range(na):
            for j in range(nb):
                K[i, j] = float(circuit(Xa[i], Xb[j])[0])
                pbar.update(1)
        pbar.close()
        return K

    return gram(X_train, X_train, "K_train"), gram(X_test, X_train, "K_test")


def build_srqfm_fidelity_kernel(X_train, X_test):
    """All-to-all SRQFM-fidelity for apples-to-apples comparison with BSCM."""
    n_q = X_train.shape[1]
    fwd = lambda x: _apply_srqfm(x, n_q, config.ZZ_REPS, 0.01, "all")
    adj = lambda x: _apply_srqfm_adjoint(x, n_q, config.ZZ_REPS, 0.01, "all")
    circuit = _build_paired_circuit(n_q, fwd, adj)

    def gram(Xa, Xb, label):
        na, nb = len(Xa), len(Xb)
        K = np.zeros((na, nb))
        pbar = tqdm(total=na * nb, desc=f"    SRQFM-fid {label} ({na}x{nb})",
                     unit="eval")
        for i in range(na):
            for j in range(nb):
                K[i, j] = float(circuit(Xa[i], Xb[j])[0])
                pbar.update(1)
        pbar.close()
        return K

    return gram(X_train, X_train, "K_train"), gram(X_test, X_train, "K_test")


def build_fqk_kernel(X_train, X_test):
    """ZZFeatureMap fidelity kernel (Havlicek 2019).  Linear connectivity."""
    import pennylane as qml

    n_q = X_train.shape[1]
    REPS = 2
    dev = qml.device("default.qubit", wires=n_q)

    @qml.qnode(dev, diff_method=None)
    def circuit(x1, x2):
        for _ in range(REPS):
            for i in range(n_q):
                qml.Hadamard(wires=i)
                qml.RZ(x1[i], wires=i)
            for i in range(n_q - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ((np.pi - x1[i]) * (np.pi - x1[i + 1]), wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
        for _ in range(REPS):
            for i in range(n_q - 2, -1, -1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(-(np.pi - x2[i]) * (np.pi - x2[i + 1]), wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_q - 1, -1, -1):
                qml.RZ(-x2[i], wires=i)
                qml.Hadamard(wires=i)
        return qml.probs(wires=range(n_q))

    def gram(Xa, Xb, label):
        na, nb = len(Xa), len(Xb)
        K = np.zeros((na, nb))
        pbar = tqdm(total=na * nb, desc=f"    FQK {label} ({na}x{nb})", unit="eval")
        for i in range(na):
            for j in range(nb):
                K[i, j] = float(circuit(Xa[i], Xb[j])[0])
                pbar.update(1)
        pbar.close()
        return K

    return gram(X_train, X_train, "K_train"), gram(X_test, X_train, "K_test")


# ----- Classical baselines + utilities --------------------------------------

def run_rbf_svm(X_tr, y_tr, X_te, y_te):
    grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced", random_state=42),
        {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1.0]},
        cv=3, scoring="f1_macro", n_jobs=-1, refit=True,
    )
    grid.fit(X_tr, y_tr)
    yp = grid.predict(X_te)
    return {"f1_macro": float(f1_score(y_te, yp, average="macro")),
            "balanced_acc": float(balanced_accuracy_score(y_te, yp)),
            "best_params": {k: str(v) for k, v in grid.best_params_.items()},
            "predictions": yp.tolist()}


def run_rf(X_tr, y_tr, X_te, y_te):
    rf = RandomForestClassifier(
        n_estimators=200, min_samples_split=5,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    rf.fit(X_tr, y_tr)
    yp = rf.predict(X_te)
    return {"f1_macro": float(f1_score(y_te, yp, average="macro")),
            "balanced_acc": float(balanced_accuracy_score(y_te, yp)),
            "predictions": yp.tolist()}


def run_quantum_svm(K_tr, y_tr, K_te, y_te):
    best_f1, best = -1, None
    for C in [0.1, 1, 10, 100]:
        clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
        clf.fit(K_tr, y_tr)
        yp = clf.predict(K_te)
        f1 = f1_score(y_te, yp, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best = {"f1_macro": float(f1),
                     "balanced_acc": float(balanced_accuracy_score(y_te, yp)),
                     "best_C": float(C),
                     "predictions": yp.tolist()}
    return best


def mcnemar_test(y_true, y_a, y_b):
    ca = (y_a == y_true)
    cb = (y_b == y_true)
    b01 = int(np.sum(ca & ~cb))
    b10 = int(np.sum(~ca & cb))
    if b01 + b10 == 0:
        return {"chi2": 0.0, "p_value": 1.0}
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    return {"chi2": float(stat), "p_value": float(1 - chi2.cdf(stat, df=1))}


def compute_kta(K, y):
    Y = (y[:, None] == y[None, :]).astype(float)
    Yc, Kc = Y - Y.mean(), K - K.mean()
    den = np.sqrt(np.sum(Kc ** 2) * np.sum(Yc ** 2))
    return float(np.sum(Kc * Yc) / max(den, 1e-12))


def spectral_metrics(K):
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = K[mask]
    eig = np.clip(np.linalg.eigvalsh((K + K.T) / 2.0), 1e-12, None)
    p = eig / eig.sum()
    return {
        "off_diag_mean": float(off.mean()),
        "off_diag_var": float(off.var()),
        "eff_rank_shannon": float(np.exp(-np.sum(p * np.log(p + 1e-12)))),
        "eff_rank_huang": float(eig.sum() / eig.max()),
    }


# ----- Caching --------------------------------------------------------------

def save_kernel(K, name, seed, feat_set):
    KERNEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = KERNEL_CACHE_DIR / f"{name}_{feat_set}_seed{seed}.npy"
    np.save(p, K)
    return str(p)


def load_kernel(name, seed, feat_set):
    p = KERNEL_CACHE_DIR / f"{name}_{feat_set}_seed{seed}.npy"
    return np.load(p) if p.exists() else None


# ----- Per-feature-set runner -----------------------------------------------

def run_feature_set(feat_name, X_tr, y_tr, X_te, y_te, seed,
                     tau_locked, run_quantum=True):
    log = logging.getLogger(__name__)
    sr = {}

    log.info("  [%s] Classical baselines...", feat_name)
    sr["RBF-SVM"] = run_rbf_svm(X_tr, y_tr, X_te, y_te)
    sr["RF"] = run_rf(X_tr, y_tr, X_te, y_te)
    log.info("    RBF-SVM F1=%.4f  RF F1=%.4f",
              sr["RBF-SVM"]["f1_macro"], sr["RF"]["f1_macro"])

    if not run_quantum:
        return sr

    quantum_specs = [
        ("FQK", build_fqk_kernel),
        ("SRQFM-fid", build_srqfm_fidelity_kernel),
        ("BSCM-uniform", lambda a, b: build_bscm_kernel(a, b, tau_locked, "uniform")),
    ]
    for name, builder in quantum_specs:
        K_tr = load_kernel(f"{name}_train", seed, feat_name)
        K_te = load_kernel(f"{name}_test", seed, feat_name)
        if K_tr is None or K_te is None:
            log.info("  [%s] Computing %s kernel...", feat_name, name)
            t0 = time.time()
            K_tr, K_te = builder(X_tr, X_te)
            save_kernel(K_tr, f"{name}_train", seed, feat_name)
            save_kernel(K_te, f"{name}_test", seed, feat_name)
            log.info("    %s computed in %.0fs", name, time.time() - t0)
        else:
            log.info("  [%s] %s loaded from cache", feat_name, name)
        sr[name] = run_quantum_svm(K_tr, y_tr, K_te, y_te)
        sr[name]["kta"] = compute_kta(K_tr, y_tr)
        sr[name]["spectral"] = spectral_metrics(K_tr)
        log.info("    %s F1=%.4f  KTA=%.3f", name,
                  sr[name]["f1_macro"], sr[name]["kta"])

    rbf_p = np.array(sr["RBF-SVM"]["predictions"])
    for q in ["FQK", "SRQFM-fid", "BSCM-uniform"]:
        if q in sr:
            sr[f"McNemar_{q}_vs_RBF"] = mcnemar_test(y_te, np.array(sr[q]["predictions"]), rbf_p)

    return sr


# ----- Main -----------------------------------------------------------------

def _save_results(all_results, t_start):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results["wall_clock_s"] = round(time.time() - t_start, 1)
    with open(RESULTS_DIR / "eurosat_bscm_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(
        description="EuroSAT Cross-Dataset Validation -- BSCM",
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(RESULTS_DIR / "experiment.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.StreamHandler(sys.stdout), fh,
    ])
    log = logging.getLogger(__name__)

    tau_locked = get_locked_tau()
    log.info("=" * 60)
    log.info("EuroSAT BSCM Cross-Dataset Validation")
    log.info("=" * 60)
    log.info("Locked tau (from So2Sat holdout, transferred):  %.3f", tau_locked)
    log.info("Config: %s", vars(args))

    t_start = time.time()
    seeds = [args.seed_base + i for i in range(args.n_seeds)]

    band_means, labels, class_names = load_eurosat_allbands(
        args.data_dir, cache_path="data/processed/eurosat_band_means.npz",
    )

    all_results = {
        "experiment": "EuroSAT BSCM Cross-Dataset Validation (13-Band + Physics)",
        "dataset": "EuroSAT (Helber et al., 2019) -- 13 Sentinel-2 bands",
        "n_classes": len(class_names),
        "class_names": class_names,
        "config": vars(args),
        "physics_features": PHYSICS_FEATURE_NAMES,
        "tau_locked": tau_locked,
        "tau_source": "So2Sat held-out pool (200 samples, KTA-argmax)",
    }

    run_quantum = not args.skip_quantum
    feat_sets = []
    if not args.physics_only:
        feat_sets.append("pca8")
    if not args.pca_only:
        feat_sets.append("physics8")

    for feat_name in feat_sets:
        log.info("\n%s\nFEATURE SET: %s\n%s", "=" * 60, feat_name.upper(), "=" * 60)
        method_f1s = {m: [] for m in ["FQK", "SRQFM-fid", "BSCM-uniform", "RBF-SVM", "RF"]}
        seed_results = []

        for i, seed in enumerate(seeds):
            log.info("\n--- Seed %d/%d (seed=%d) ---", i + 1, len(seeds), seed)
            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=args.n_train,
                test_size=args.n_test, random_state=seed,
            )
            tr_idx, te_idx = next(splitter.split(band_means, labels))
            feats = prepare_split_features(
                band_means[tr_idx], band_means[te_idx], pca_dim=8, seed=seed,
            )
            if feat_name == "pca8":
                X_tr, X_te = feats["pca8_train"], feats["pca8_test"]
            else:
                X_tr, X_te = feats["physics8_train"], feats["physics8_test"]
            y_tr, y_te = labels[tr_idx], labels[te_idx]
            log.info("  Train %s, Test %s", X_tr.shape, X_te.shape)

            sr = run_feature_set(feat_name, X_tr, y_tr, X_te, y_te,
                                  seed, tau_locked, run_quantum)
            sr["seed"] = seed
            for m in method_f1s:
                if m in sr:
                    method_f1s[m].append(sr[m]["f1_macro"])
            for m in ["FQK", "SRQFM-fid", "BSCM-uniform", "RBF-SVM", "RF"]:
                if m in sr and "predictions" in sr[m]:
                    del sr[m]["predictions"]
            seed_results.append(sr)
            all_results[feat_name] = {"seeds": seed_results}
            _save_results(all_results, t_start)

        summary = {}
        for m, arr in method_f1s.items():
            if arr:
                a = np.array(arr)
                summary[m] = {"mean": float(a.mean()), "std": float(a.std()),
                              "n": len(arr)}
        all_results[feat_name]["summary"] = summary
        log.info("\nSummary for %s:", feat_name)
        for m, s in summary.items():
            log.info("  %-15s mean=%.4f  std=%.4f", m, s["mean"], s["std"])

    _save_results(all_results, t_start)
    log.info("\nDone.  Total wall clock %.1f s", time.time() - t_start)


if __name__ == "__main__":
    main()
