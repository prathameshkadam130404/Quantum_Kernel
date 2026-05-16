"""
audit_e38_significance.py
=========================
Recomputes the E38 (16-qubit, N=10,000, depth=6) maximum-capacity benchmark
*from cached Gram matrices*, storing the per-split F1 vector for every method
and emitting Wilcoxon signed-rank p-values (Holm-Bonferroni corrected) against
Standard-ZZ-PQK.  No quantum simulation is rerun.

What this script does
---------------------
1. Loads each cached precomputed kernel ``cache_<dataset>_<method>.npz``
   from ``results/e38_max_data/``.  Each file contains the Gram matrix ``K``
   and the label vector ``y`` produced by the upstream E38 scripts
   (``exp_e38_max_data_pqk.py``, ``exp_e38_max_data_standard_pqk.py``,
   ``exp_e38_max_data_baselines.py``).
2. Runs the *identical* evaluation pipeline used in those scripts:
     ``StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)``
   with ``GridSearchCV(SVC(kernel='precomputed', class_weight='balanced'),
                        {'C': [0.1, 1, 10, 100, 1000]}, cv=3,
                        scoring='f1_macro', n_jobs=-1)``.
   This reproduces the f1_mean values in the per-method E38 JSONs to within
   REPRO_TOL=1e-4; mismatches raise RuntimeError so no inconsistent result
   is ever written.
3. Saves the 10-element per-split vector for every method.
4. Computes Wilcoxon signed-rank tests of every other method's per-split
   vector against Standard-ZZ-PQK's, then applies Holm-Bonferroni correction
   over the family of (#methods - 1) tests per dataset.
5. Optionally re-evaluates the classical baselines (RBF-SVM, RandomForest)
   from the raw feature arrays if the user passes ``--with-classical``.
   This requires loading the raw datasets and adds ~5--10 minutes; default
   is False (kernel-vs-kernel significance is the headline claim).

Output: ``results/e38_max_data/e38_significance.json``.

Usage
-----
    python experiments/audit_e38_significance.py [--with-classical]

This script intentionally never touches any cached ``K`` matrix.  If the
cached means do not reproduce the published numbers, the script aborts
without writing a result.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# --------------------------------------------------------------------------- #
# Constants -- locked to the upstream E38 pipeline.
# --------------------------------------------------------------------------- #
N_SPLITS = 10
TEST_SIZE = 0.3
SPLIT_SEED = 42
CV_FOLDS = 3
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]

E38_DIR = os.path.join(config.RESULTS_DIR, "e38_max_data")
OUT_PATH = os.path.join(E38_DIR, "e38_significance.json")

# Map (dataset, method) -> cache file basename.
CACHE_MAP: Dict[Tuple[str, str], str] = {
    ("so2sat", "SRQFM-PQK"):          "cache_so2sat_srqfm.npz",
    ("so2sat", "Standard-ZZ-PQK"):    "cache_so2sat_standard_pqk.npz",
    ("so2sat", "BSCM-uniform-PQK"):   "cache_so2sat_bscm_uniform.npz",
    ("so2sat", "BSCM-phi-PQK"):       "cache_so2sat_bscm_phi_only.npz",
    ("so2sat", "BSCM-psi-PQK"):       "cache_so2sat_bscm_psi_only.npz",
    ("eurosat", "SRQFM-PQK"):         "cache_eurosat_srqfm.npz",
    ("eurosat", "Standard-ZZ-PQK"):   "cache_eurosat_standard_pqk.npz",
    ("eurosat", "BSCM-uniform-PQK"):  "cache_eurosat_bscm_uniform.npz",
    ("eurosat", "BSCM-phi-PQK"):      "cache_eurosat_bscm_phi_only.npz",
    ("eurosat", "BSCM-psi-PQK"):      "cache_eurosat_bscm_psi_only.npz",
}

# Reference means from the published per-method JSONs.  We assert that the
# recomputed mean is within REPRO_TOL of the published value before trusting
# the per-split vector for the Wilcoxon test.
REPRO_TOL = 1e-4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(E38_DIR, "audit_e38_significance.log"),
                            mode="w"),
    ],
)
log = logging.getLogger("audit_e38")


# --------------------------------------------------------------------------- #
# Per-split SVM evaluation on a precomputed kernel.
# --------------------------------------------------------------------------- #
def _per_split_f1_precomputed(K: np.ndarray, y: np.ndarray) -> List[float]:
    """Run the locked 10-split protocol on a precomputed Gram matrix."""
    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SPLIT_SEED,
    )
    scores: List[float] = []
    for tr, te in sss.split(K, y):
        K_tr = K[np.ix_(tr, tr)]
        K_te = K[np.ix_(te, tr)]
        clf = GridSearchCV(
            SVC(kernel="precomputed", class_weight="balanced"),
            {"C": C_GRID}, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1,
        )
        clf.fit(K_tr, y[tr])
        scores.append(float(f1_score(y[te], clf.predict(K_te), average="macro")))
    return scores


def _per_split_f1_rbf(X: np.ndarray, y: np.ndarray) -> List[float]:
    """Re-run the classical RBF baseline (same protocol as the E38 baselines)."""
    from sklearn.preprocessing import StandardScaler
    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SPLIT_SEED,
    )
    scores: List[float] = []
    for tr, te in sss.split(X, y):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        clf = GridSearchCV(
            SVC(kernel="rbf", class_weight="balanced"),
            {"C": C_GRID, "gamma": ["scale", "auto", 0.01, 0.1, 1.0]},
            cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1,
        )
        clf.fit(X_tr, y[tr])
        scores.append(float(f1_score(y[te], clf.predict(X_te), average="macro")))
    return scores


def _per_split_f1_rf(X: np.ndarray, y: np.ndarray) -> List[float]:
    from sklearn.ensemble import RandomForestClassifier
    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SPLIT_SEED,
    )
    scores: List[float] = []
    for tr, te in sss.split(X, y):
        rf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=SPLIT_SEED, n_jobs=-1,
        )
        rf.fit(X[tr], y[tr])
        scores.append(float(f1_score(y[te], rf.predict(X[te]), average="macro")))
    return scores


# --------------------------------------------------------------------------- #
# Statistics.
# --------------------------------------------------------------------------- #
def _holm_correct(pvals_by_key: Dict[str, float]) -> Dict[str, float]:
    """Return Holm-Bonferroni-corrected p-values, preserving keys."""
    keys = list(pvals_by_key.keys())
    raw = np.asarray([pvals_by_key[k] for k in keys], dtype=float)
    m = len(raw)
    order = np.argsort(raw)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        # Holm step: adjusted = max over k<=rank of (m - k) * p_(k)
        scaled = (m - rank) * raw[idx]
        running_max = max(running_max, scaled)
        adj[idx] = min(running_max, 1.0)
    return {k: float(adj[i]) for i, k in enumerate(keys)}


def _wilcoxon_vs_ref(v: List[float], ref: List[float]) -> float:
    """Two-sided Wilcoxon signed-rank against a reference vector."""
    diffs = np.asarray(v) - np.asarray(ref)
    if np.all(np.abs(diffs) < 1e-12):
        return 1.0
    try:
        stat, p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
        return float(p)
    except ValueError:
        # All zeros after dropping zeros: undefined; treat as no evidence.
        return 1.0


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def _load_published_means() -> Dict[Tuple[str, str], Optional[float]]:
    """Load the published f1_mean values from the per-method JSONs.

    Used as a sanity check that the recomputed per-split vectors are
    consistent with what the paper currently cites.  Mismatches abort the
    audit so we never store inconsistent results.
    """
    def _load(path: str) -> dict:
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)

    baselines = _load(os.path.join(E38_DIR, "baselines_summary.json"))
    standard = _load(os.path.join(E38_DIR, "standard_pqk_summary.json"))
    bscm = _load(os.path.join(E38_DIR, "max_summary_bscm_sectors.json"))

    out: Dict[Tuple[str, str], Optional[float]] = {}
    for dset in ("so2sat", "eurosat"):
        out[(dset, "SRQFM-PQK")] = baselines.get(dset, {}).get("srqfm_f1_mean")
        out[(dset, "Standard-ZZ-PQK")] = (
            standard.get(dset, {}).get("standard_pqk_f1_mean")
        )
        out[(dset, "BSCM-uniform-PQK")] = bscm.get(f"{dset}_bscm_uniform_f1_mean")
        out[(dset, "BSCM-phi-PQK")] = bscm.get(f"{dset}_bscm_phi_only_f1_mean")
        out[(dset, "BSCM-psi-PQK")] = bscm.get(f"{dset}_bscm_psi_only_f1_mean")
    return out


def _evaluate_dataset(
    dataset: str,
    with_classical: bool,
    published_means: Dict[Tuple[str, str], Optional[float]],
) -> Dict[str, dict]:
    """Run all available methods for one dataset and return per-split records."""
    methods = [
        "SRQFM-PQK", "Standard-ZZ-PQK",
        "BSCM-uniform-PQK", "BSCM-phi-PQK", "BSCM-psi-PQK",
    ]
    per_split: Dict[str, List[float]] = {}

    for method in methods:
        cache_name = CACHE_MAP.get((dataset, method))
        if cache_name is None:
            continue
        cache_path = os.path.join(E38_DIR, cache_name)
        if not os.path.exists(cache_path):
            log.warning("Missing cache for %s/%s -- skipped", dataset, method)
            continue
        log.info("[%s/%s] loading %s", dataset, method, cache_name)
        d = np.load(cache_path)
        K, y = d["K"], d["y"]
        t0 = time.time()
        v = _per_split_f1_precomputed(K, y)
        dt = time.time() - t0
        mean = float(np.mean(v))
        std = float(np.std(v))
        log.info("[%s/%s] f1=%.4f +/- %.4f over %d splits (%.0fs)",
                 dataset, method, mean, std, len(v), dt)

        # Reproduction check.
        pub = published_means.get((dataset, method))
        if pub is not None and abs(pub - mean) > REPRO_TOL:
            raise RuntimeError(
                f"Reproduction mismatch for {dataset}/{method}: "
                f"published={pub:.6f}, recomputed={mean:.6f}, "
                f"delta={mean - pub:.6f} (tol={REPRO_TOL}). "
                "The cached kernel and the published summary disagree; "
                "investigate before publishing p-values."
            )
        per_split[method] = v

    # Optionally run classical baselines from raw features.
    if with_classical:
        log.info("[%s] re-running classical baselines for per-split CIs/p", dataset)
        X_raw, y_clf = _load_raw_features(dataset)
        t0 = time.time()
        per_split["RBF-SVM"] = _per_split_f1_rbf(X_raw, y_clf)
        log.info("[%s/RBF-SVM] f1=%.4f +/- %.4f (%.0fs)",
                 dataset, np.mean(per_split["RBF-SVM"]),
                 np.std(per_split["RBF-SVM"]), time.time() - t0)
        t0 = time.time()
        per_split["RandomForest"] = _per_split_f1_rf(X_raw, y_clf)
        log.info("[%s/RF] f1=%.4f +/- %.4f (%.0fs)",
                 dataset, np.mean(per_split["RandomForest"]),
                 np.std(per_split["RandomForest"]), time.time() - t0)

    # Wilcoxon vs Standard-ZZ-PQK.
    ref = per_split.get("Standard-ZZ-PQK")
    if ref is None:
        log.warning("No Standard-ZZ-PQK reference for %s; skipping p-values",
                    dataset)
        return {"per_split_f1": per_split}

    raw_p = {
        m: _wilcoxon_vs_ref(per_split[m], ref)
        for m in per_split if m != "Standard-ZZ-PQK"
    }
    holm_p = _holm_correct(raw_p)

    return {
        "per_split_f1": per_split,
        "summary": {
            m: {
                "f1_mean": float(np.mean(per_split[m])),
                "f1_std": float(np.std(per_split[m])),
                "p_wilcoxon_vs_ZZ": raw_p.get(m),
                "p_holm_vs_ZZ": holm_p.get(m),
                "delta_over_ZZ": float(np.mean(per_split[m]) - np.mean(ref)),
            }
            for m in per_split
        },
        "reference_method": "Standard-ZZ-PQK",
        "wilcoxon_min_achievable_raw_p": float(1.0 / (2 ** (N_SPLITS - 1))),
    }


def _load_raw_features(dataset: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the raw feature matrix used by the E38 classical baselines."""
    # We import lazily because these loaders are slow; only the
    # --with-classical path needs them.
    from experiments.exp_e38_max_data_pqk import (
        _load_so2sat_max, _load_eurosat_10k,
    )
    if dataset == "so2sat":
        _, X_raw, y = _load_so2sat_max()
    elif dataset == "eurosat":
        _, X_raw, y = _load_eurosat_10k()
    else:
        raise ValueError(dataset)
    return X_raw, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-classical", action="store_true",
        help=("Also re-evaluate RBF-SVM and Random Forest from raw features "
              "to obtain per-split vectors and p-values for the classical "
              "baselines.  Adds ~5--10 minutes."),
    )
    args = parser.parse_args()

    log.info("=" * 72)
    log.info("  E38 significance audit -- per-split + Holm-corrected p-values")
    log.info("=" * 72)

    published_means = _load_published_means()

    out = {
        "protocol": {
            "n_splits": N_SPLITS,
            "test_size": TEST_SIZE,
            "split_seed": SPLIT_SEED,
            "inner_cv_folds": CV_FOLDS,
            "C_grid": C_GRID,
            "scoring": "f1_macro",
            "reference_method": "Standard-ZZ-PQK",
        },
        "datasets": {},
    }
    for dataset in ("so2sat", "eurosat"):
        log.info("")
        log.info("[%s] starting", dataset)
        out["datasets"][dataset] = _evaluate_dataset(
            dataset, args.with_classical, published_means,
        )

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    log.info("")
    log.info("Wrote %s", OUT_PATH)

    # Console summary.
    for dataset in ("so2sat", "eurosat"):
        d = out["datasets"][dataset]
        if "summary" not in d:
            continue
        log.info("")
        log.info("[%s] significance vs Standard-ZZ-PQK (10 splits, Holm-corrected):",
                 dataset)
        log.info("%-22s %12s %12s %12s %12s",
                 "method", "f1_mean", "delta_ZZ", "p_wilcoxon", "p_holm")
        for m, s in d["summary"].items():
            log.info("%-22s %12.4f %12.4f %12s %12s",
                     m, s["f1_mean"], s["delta_over_ZZ"],
                     f"{s['p_wilcoxon_vs_ZZ']:.4f}" if s["p_wilcoxon_vs_ZZ"] is not None else "---",
                     f"{s['p_holm_vs_ZZ']:.4f}"      if s["p_holm_vs_ZZ"]      is not None else "---")


if __name__ == "__main__":
    main()
