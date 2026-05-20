"""
audit_e38_perclass_mcnemar.py
=============================
Companion audit to ``audit_e38_significance.py``: re-runs the E38
(16-qubit, N=10,000, depth=6) maximum-capacity pipeline from cached Gram
matrices and produces two extra families of statistics that the original
significance audit did not emit:

  1. **Per-class macro-F1 with bootstrap 95% CI.**  For every method and
     dataset, computes per-class F1 on each of the 10 stratified shuffle
     splits, then a percentile bootstrap CI (B=10,000 resamples with
     replacement over the 10 per-split values).

  2. **McNemar test of best-BSCM vs Standard-ZZ-PQK on the test
     predictions.**  Per split, builds the 2x2 contingency on the test
     fold predictions of the two kernels; reports per-split exact
     McNemar p (binomial on the smaller off-diagonal), plus a
     pooled-across-splits exact McNemar p over the full union of test
     samples (the test folds are disjoint by construction of
     ``StratifiedShuffleSplit`` only in expectation, so we record both
     pooled and Stouffer-combined per-split p as alternatives).

The script does NOT touch any cached ``K`` matrix.  Re-running the SVM
is deterministic given the seed and grid, so the per-split macro-F1 we
observe here must match the per-split vectors in
``e38_significance.json`` to ``REPRO_TOL``; mismatches abort.

Output: ``results/e38_max_data/e38_perclass_mcnemar.json``
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import binom, combine_pvalues
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# --------------------------------------------------------------------------- #
# Locked protocol constants -- identical to audit_e38_significance.py.
# --------------------------------------------------------------------------- #
N_SPLITS = 10
TEST_SIZE = 0.3
SPLIT_SEED = 42
CV_FOLDS = 3
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 42
REPRO_TOL = 1e-4

E38_DIR = os.path.join(config.RESULTS_DIR, "e38_max_data")
OUT_PATH = os.path.join(E38_DIR, "e38_perclass_mcnemar.json")
SIG_PATH = os.path.join(E38_DIR, "e38_significance.json")  # for reproduction check

CACHE_MAP: Dict[Tuple[str, str], str] = {
    ("so2sat",  "SRQFM-PQK"):         "cache_so2sat_srqfm.npz",
    ("so2sat",  "Standard-ZZ-PQK"):   "cache_so2sat_standard_pqk.npz",
    ("so2sat",  "BSCM-uniform-PQK"):  "cache_so2sat_bscm_uniform.npz",
    ("so2sat",  "BSCM-phi-PQK"):      "cache_so2sat_bscm_phi_only.npz",
    ("so2sat",  "BSCM-psi-PQK"):      "cache_so2sat_bscm_psi_only.npz",
    ("eurosat", "SRQFM-PQK"):         "cache_eurosat_srqfm.npz",
    ("eurosat", "Standard-ZZ-PQK"):   "cache_eurosat_standard_pqk.npz",
    ("eurosat", "BSCM-uniform-PQK"):  "cache_eurosat_bscm_uniform.npz",
    ("eurosat", "BSCM-phi-PQK"):      "cache_eurosat_bscm_phi_only.npz",
    ("eurosat", "BSCM-psi-PQK"):      "cache_eurosat_bscm_psi_only.npz",
}

# The kernel we use as the "best BSCM" in the McNemar contrast against
# Standard-ZZ-PQK; per Tab.~11 BSCM-phi has the highest macro-F1 mean on
# both So2Sat and EuroSAT, so we use it as the headline contrast.
BSCM_REFERENCE = "BSCM-phi-PQK"
ZZ_REFERENCE = "Standard-ZZ-PQK"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(E38_DIR, "audit_e38_perclass_mcnemar.log"),
            mode="w",
        ),
    ],
)
log = logging.getLogger("audit_perclass")


# --------------------------------------------------------------------------- #
# Core: per-split SVM that also returns y_true, y_pred so we can do
# per-class F1 and McNemar.
# --------------------------------------------------------------------------- #
def _per_split_predict(
    K: np.ndarray, y: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return list of (test_indices, y_true, y_pred) per split."""
    sss = StratifiedShuffleSplit(
        n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=SPLIT_SEED,
    )
    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for tr, te in sss.split(K, y):
        K_tr = K[np.ix_(tr, tr)]
        K_te = K[np.ix_(te, tr)]
        clf = GridSearchCV(
            SVC(kernel="precomputed", class_weight="balanced"),
            {"C": C_GRID}, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1,
        )
        clf.fit(K_tr, y[tr])
        y_pred = clf.predict(K_te)
        out.append((te.astype(np.int64), y[te].astype(np.int64),
                    y_pred.astype(np.int64)))
    return out


# --------------------------------------------------------------------------- #
# Statistics.
# --------------------------------------------------------------------------- #
def _percentile_ci(values: np.ndarray, rng: np.random.Generator,
                   B: int = BOOTSTRAP_B, alpha: float = 0.05,
                   ) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean over ``values`` (1-D)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(B, values.size))
    means = values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _per_class_f1_table(
    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    classes: np.ndarray,
) -> Dict[str, dict]:
    """Per-class F1 mean ± std + bootstrap 95% CI across the 10 splits."""
    K_classes = len(classes)
    # f1_per_split[c, s] = F1 of class classes[c] on split s.
    f1_per_split = np.zeros((K_classes, len(splits)), dtype=float)
    support = np.zeros((K_classes, len(splits)), dtype=int)
    for s, (_, y_true, y_pred) in enumerate(splits):
        per = f1_score(y_true, y_pred, labels=classes,
                       average=None, zero_division=0)
        f1_per_split[:, s] = per
        for ci, c in enumerate(classes):
            support[ci, s] = int(np.sum(y_true == c))

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out: Dict[str, dict] = {}
    for ci, c in enumerate(classes):
        vals = f1_per_split[ci]
        lo, hi = _percentile_ci(vals, rng)
        out[str(int(c))] = {
            "f1_mean": float(vals.mean()),
            "f1_std": float(vals.std()),
            "f1_ci95": [lo, hi],
            "per_split_f1": [float(v) for v in vals.tolist()],
            "support_mean": float(support[ci].mean()),
        }
    # Macro F1 across classes per split, plus bootstrap CI across splits.
    macro_per_split = f1_per_split.mean(axis=0)
    lo, hi = _percentile_ci(macro_per_split, rng)
    out["__macro__"] = {
        "f1_mean": float(macro_per_split.mean()),
        "f1_std": float(macro_per_split.std()),
        "f1_ci95": [lo, hi],
        "per_split_f1": [float(v) for v in macro_per_split.tolist()],
    }
    return out


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p from the off-diagonal counts (b, c).

    Under H0 each discordant pair is +/- with prob 1/2.  Returns the
    cumulative two-tailed binomial probability of an outcome at least as
    extreme as min(b, c) successes in n=b+c trials.
    """
    n = int(b + c)
    if n == 0:
        return 1.0
    k = int(min(b, c))
    # Two-sided: 2 * P(X <= k); cap at 1.0.
    p = float(2.0 * binom.cdf(k, n, 0.5))
    return min(1.0, p)


def _mcnemar_per_split(
    splits_a: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    splits_b: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[str, object]:
    """McNemar contingency between two methods over the matched splits.

    Both methods are evaluated on identical splits (same seed, same
    StratifiedShuffleSplit object), so test indices align by position.
    """
    assert len(splits_a) == len(splits_b)
    per_split: List[dict] = []
    total_b = 0
    total_c = 0
    per_split_p: List[float] = []
    for (te_a, y_a, yp_a), (te_b, y_b, yp_b) in zip(splits_a, splits_b):
        # Sanity: identical test indices and labels.
        assert np.array_equal(te_a, te_b), "split indices diverged"
        assert np.array_equal(y_a, y_b), "labels diverged"
        a_correct = (yp_a == y_a)
        b_correct = (yp_b == y_b)
        b_count = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
        c_count = int(np.sum(~a_correct & b_correct))   # A wrong, B right
        n_disc = b_count + c_count
        p = _mcnemar_exact_p(b_count, c_count)
        per_split.append({
            "n_test": int(len(y_a)),
            "n_concordant_both_right": int(np.sum(a_correct & b_correct)),
            "n_concordant_both_wrong": int(np.sum(~a_correct & ~b_correct)),
            "b_A_right_B_wrong": b_count,
            "c_A_wrong_B_right": c_count,
            "n_discordant": n_disc,
            "mcnemar_p_exact": p,
        })
        per_split_p.append(p)
        total_b += b_count
        total_c += c_count

    pooled_p = _mcnemar_exact_p(total_b, total_c)
    # Stouffer combine (z-transform), as an alternative to pooling.
    # combine_pvalues with method='stouffer' returns (statistic, pvalue).
    try:
        _, stouffer_p = combine_pvalues(per_split_p, method="stouffer")
        stouffer_p = float(stouffer_p)
    except Exception:
        stouffer_p = float("nan")
    return {
        "per_split": per_split,
        "pooled": {
            "b_A_right_B_wrong_total": int(total_b),
            "c_A_wrong_B_right_total": int(total_c),
            "n_discordant_total": int(total_b + total_c),
            "mcnemar_p_exact_pooled": float(pooled_p),
        },
        "stouffer_combined_p": stouffer_p,
    }


# --------------------------------------------------------------------------- #
# Reproduction check against the existing significance audit.
# --------------------------------------------------------------------------- #
def _load_published_per_split() -> Dict[Tuple[str, str], List[float]]:
    if not os.path.exists(SIG_PATH):
        log.warning("no e38_significance.json found; skipping repro check")
        return {}
    with open(SIG_PATH) as f:
        sig = json.load(f)
    out: Dict[Tuple[str, str], List[float]] = {}
    for dset, blob in sig.get("datasets", {}).items():
        for method, vec in blob.get("per_split_f1", {}).items():
            out[(dset, method)] = [float(v) for v in vec]
    return out


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #
def _evaluate_dataset(dataset: str,
                      published: Dict[Tuple[str, str], List[float]],
                      ) -> Dict[str, object]:
    methods = [
        "SRQFM-PQK", ZZ_REFERENCE,
        "BSCM-uniform-PQK", "BSCM-phi-PQK", "BSCM-psi-PQK",
    ]
    per_method_splits: Dict[
        str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ] = {}
    classes_ref: np.ndarray = np.array([], dtype=np.int64)

    for method in methods:
        cache_name = CACHE_MAP.get((dataset, method))
        if cache_name is None:
            continue
        cache_path = os.path.join(E38_DIR, cache_name)
        if not os.path.exists(cache_path):
            log.warning("missing cache for %s/%s -- skipped", dataset, method)
            continue
        log.info("[%s/%s] loading %s", dataset, method, cache_name)
        d = np.load(cache_path)
        K, y = d["K"], d["y"].astype(np.int64)
        if classes_ref.size == 0:
            classes_ref = np.unique(y)
        t0 = time.time()
        splits = _per_split_predict(K, y)
        log.info("[%s/%s] 10 splits done (%.0fs)", dataset, method,
                 time.time() - t0)

        # Sanity: check that the per-split macro-F1 reproduces the published
        # significance audit to REPRO_TOL.  This guards against any
        # protocol drift between the two scripts.
        ref = published.get((dataset, method))
        if ref is not None:
            recomputed = [
                float(f1_score(yt, yp, average="macro")) for _, yt, yp in splits
            ]
            for s_idx, (r, v) in enumerate(zip(ref, recomputed)):
                if abs(r - v) > REPRO_TOL:
                    raise RuntimeError(
                        f"split {s_idx} F1 mismatch for {dataset}/{method}: "
                        f"published={r:.6f}, recomputed={v:.6f}, "
                        f"delta={v - r:.6f} (tol={REPRO_TOL})."
                    )
        per_method_splits[method] = splits

    # Per-class F1 tables.
    per_class: Dict[str, Dict[str, dict]] = {}
    for method, splits in per_method_splits.items():
        per_class[method] = _per_class_f1_table(splits, classes_ref)

    # McNemar: BSCM-phi (or whichever BSCM_REFERENCE we picked) vs Standard-ZZ.
    mcnemar: Dict[str, object] = {}
    if BSCM_REFERENCE in per_method_splits and ZZ_REFERENCE in per_method_splits:
        log.info("[%s] McNemar %s vs %s", dataset, BSCM_REFERENCE, ZZ_REFERENCE)
        mcnemar[f"{BSCM_REFERENCE}_vs_{ZZ_REFERENCE}"] = _mcnemar_per_split(
            per_method_splits[BSCM_REFERENCE],
            per_method_splits[ZZ_REFERENCE],
        )
    # Also report the headline-uniform contrast, since the paper text
    # leads with BSCM-uniform on UCI and E48.
    if "BSCM-uniform-PQK" in per_method_splits and ZZ_REFERENCE in per_method_splits:
        log.info("[%s] McNemar BSCM-uniform-PQK vs %s", dataset, ZZ_REFERENCE)
        mcnemar[f"BSCM-uniform-PQK_vs_{ZZ_REFERENCE}"] = _mcnemar_per_split(
            per_method_splits["BSCM-uniform-PQK"],
            per_method_splits[ZZ_REFERENCE],
        )

    return {
        "classes": [int(c) for c in classes_ref.tolist()],
        "per_class_f1": per_class,
        "mcnemar": mcnemar,
    }


def main() -> None:
    log.info("=" * 72)
    log.info("  E38 per-class + McNemar audit")
    log.info("=" * 72)

    published = _load_published_per_split()

    out: Dict[str, object] = {
        "protocol": {
            "n_splits": N_SPLITS,
            "test_size": TEST_SIZE,
            "split_seed": SPLIT_SEED,
            "inner_cv_folds": CV_FOLDS,
            "C_grid": C_GRID,
            "scoring": "f1_macro",
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_ci_alpha": 0.05,
            "bscm_reference_for_mcnemar": BSCM_REFERENCE,
            "zz_reference": ZZ_REFERENCE,
        },
        "datasets": {},
    }
    for dataset in ("so2sat", "eurosat"):
        log.info("")
        log.info("[%s] starting", dataset)
        out["datasets"][dataset] = _evaluate_dataset(dataset, published)

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    log.info("")
    log.info("Wrote %s", OUT_PATH)

    # Console summary.
    for dataset in ("so2sat", "eurosat"):
        d = out["datasets"][dataset]
        log.info("")
        log.info("[%s] McNemar (best BSCM vs Standard-ZZ-PQK) summaries:",
                 dataset)
        for contrast, blob in d.get("mcnemar", {}).items():
            pooled = blob["pooled"]
            log.info("  %-45s pooled_p=%.3e  (n_disc=%d, b=%d, c=%d)  "
                     "stouffer_p=%.3e",
                     contrast,
                     pooled["mcnemar_p_exact_pooled"],
                     pooled["n_discordant_total"],
                     pooled["b_A_right_B_wrong_total"],
                     pooled["c_A_wrong_B_right_total"],
                     blob["stouffer_combined_p"])


if __name__ == "__main__":
    main()
