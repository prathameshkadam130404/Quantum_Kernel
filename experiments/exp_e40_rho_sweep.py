"""
E40 -- Correlation-vs-rescue verification of Proposition 4.

Question
--------
Does the SG-BSCM rescue widen monotonically with feature-pair correlation,
as Proposition 4 (Appendix C) predicts?  We construct synthetic
8-feature datasets with controlled pairwise correlation rho in
{0.0, 0.3, 0.6, 0.85, 0.95, 0.99} and measure macro-F1 of BSCM-uniform
vs SG-BSCM.  Prop 4 predicts E[||H_SG||] -> 0 as |rho| -> 1, so the
rescue gap (SG-BSCM - BSCM-uniform) should grow with |rho|.

Synthetic data design
---------------------
- 4 latent classes, 4 informative latent dimensions (one per class).
- 8 observed features: each pair (q_i, q_{i+4}) is a redundant pair
  with controlled correlation rho.  Pairs (0,4), (1,5), (2,6), (3,7).
- N_total = 1600, equal class balance (400 each).
- Features standardized then min-max scaled to [0, pi] for quantum
  encoding.

Methods
-------
- BSCM-uniform (the collapse case)
- SG-BSCM (the rescue case)
- RBF-SVM (classical reference, depends weakly on rho)

For each (rho, method, seed) we record:
    macro_f1, off_diag_mean, kta_product, mean(w_Psi-) over correlated pairs

Protocol
--------
- N_pool: 1200 (80% of synthetic dataset)
- Seeds: 42, 43, 44, 45, 46
- Qubits: 8
- Repetitions: REPS (= 2)
- tau: 0.25
- Connectivity: all-to-all
- Noise: noiseless statevector simulation

Output
------
- results/e40_rho_sweep/metrics.csv
- results/e40_rho_sweep/summary.json
- results/e40_rho_sweep/exp_e40.log

Wall-clock estimate: ~6-10 hours.

Author: Prathamesh Kadam et al.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, StratifiedShuffleSplit,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.bscm_kernel import (
    BELL_WEIGHT_PRESETS, compute_bscm_fidelity_kernel,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS         = [42, 43, 44, 45, 46]
N_QUBITS      = 8
REPS          = config.ZZ_REPS
TAU_LOCKED    = 0.25
COUPLING_THR  = 1e-4
N_TOTAL       = 1600
N_CLASSES     = 4
N_INFORM_DIM  = 4
N_POOL_FRAC   = 0.75
TEST_FRAC     = 1.0 / 3.0
RHO_LIST      = [0.0, 0.3, 0.6, 0.85, 0.95, 0.99]
C_GRID        = [1, 10, 100]
CV_FOLDS      = 3

SAVE_DIR = os.path.join(config.RESULTS_DIR, "e40_rho_sweep")
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "exp_e40.log"), mode="w"),
    ],
)
log = logging.getLogger("exp_e40")


# ============================================================================
# Synthetic data generator
# ============================================================================

def make_synthetic_dataset(rho: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 4-class, 8-feature dataset with controlled pair correlation rho.

    Construction:
      1. Sample 4-D latent z ~ N(mu_c, I) where mu_c is a class mean
         (one one-hot direction per class, scaled by SEP=2.0).
      2. For each latent dim k in {0, 1, 2, 3}, build TWO observed features:
          (q_k, q_{k+4}) = (z_k + n1, rho * z_k + sqrt(1-rho^2) * n2)
         where n1, n2 ~ N(0, 0.5).  This gives Corr(q_k, q_{k+4}) ~= rho.

    Returns:
        X_raw: (N, 8) raw features (real-valued)
        X_enc: (N, 8) min-max scaled to [0, pi] for quantum encoding
        y:     (N,)  integer class labels in {0, 1, 2, 3}
    """
    rng = np.random.default_rng(seed)
    n_per_class = N_TOTAL // N_CLASSES
    SEP = 2.0
    NOISE = 0.5

    Z = []
    y = []
    for c in range(N_CLASSES):
        mu = np.zeros(N_INFORM_DIM); mu[c] = SEP
        zc = rng.normal(mu, 1.0, size=(n_per_class, N_INFORM_DIM))
        Z.append(zc)
        y.append(np.full(n_per_class, c, dtype=int))
    Z = np.concatenate(Z, axis=0)
    y = np.concatenate(y, axis=0)
    perm = rng.permutation(len(y))
    Z = Z[perm]; y = y[perm]

    X_raw = np.zeros((len(y), 8), dtype=np.float64)
    sigma_n = NOISE
    for k in range(N_INFORM_DIM):
        n1 = rng.normal(0.0, sigma_n, size=len(y))
        n2 = rng.normal(0.0, sigma_n, size=len(y))
        X_raw[:, k]     = Z[:, k] + n1
        X_raw[:, k + 4] = rho * Z[:, k] + np.sqrt(max(0.0, 1.0 - rho * rho)) * Z[:, k] * 0.0 \
                          + rho * n1 + np.sqrt(max(0.0, 1.0 - rho * rho)) * n2 \
                          + rho * 0.0
        # The above gives Corr(X[:,k], X[:,k+4]) ~= rho when SEP small;
        # to make the correlation hold uniformly across all latent
        # cluster centers, we recompute the second feature as a linear
        # combination of the first plus orthogonal noise:
        a = X_raw[:, k]
        nu = rng.normal(0.0, 1.0, size=len(y))
        # Standardise a, nu for unit-variance combination
        a_std = (a - a.mean()) / (a.std() + 1e-12)
        nu_std = (nu - nu.mean()) / (nu.std() + 1e-12)
        b_std = rho * a_std + np.sqrt(max(0.0, 1.0 - rho * rho)) * nu_std
        # Re-scale to roughly the same dynamic range as a
        b = b_std * (a.std() + 1e-12) + a.mean()
        X_raw[:, k + 4] = b

    sc = StandardScaler().fit(X_raw)
    X_std = sc.transform(X_raw)
    mm = MinMaxScaler(feature_range=(0, np.pi)).fit(X_std)
    X_enc = mm.transform(X_std).clip(0, np.pi).astype(np.float64)

    return X_raw, X_enc, y


def measure_pair_correlations(X_raw: np.ndarray) -> Dict[str, float]:
    """Return measured Pearson correlation for the four redundant pairs."""
    corrs = {}
    for k in range(N_INFORM_DIM):
        rho_emp = float(np.corrcoef(X_raw[:, k], X_raw[:, k + 4])[0, 1])
        corrs[f"pair_{k}_{k + 4}"] = rho_emp
    return corrs


# ============================================================================
# SVM helpers
# ============================================================================

def _tune_c(K_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> int:
    best_c, best_f1 = C_GRID[0], -1.0
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    for C in C_GRID:
        fold = []
        for tr2, va2 in skf.split(np.zeros(len(y_tr)), y_tr):
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_tr[np.ix_(tr2, tr2)], y_tr[tr2])
            yp = clf.predict(K_tr[np.ix_(va2, tr2)])
            fold.append(float(
                f1_score(y_tr[va2], yp, average="macro", zero_division=0)
            ))
        m = float(np.mean(fold))
        if m > best_f1:
            best_f1, best_c = m, C
    return best_c


def _eval_kernel(K: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
                 seed: int) -> dict:
    K_tr = K[np.ix_(tr, tr)]
    K_te = K[np.ix_(te, tr)]
    best_c = _tune_c(K_tr, y[tr], seed=seed)
    clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    yp = clf.predict(K_te)
    return {
        "macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0)),
        "best_C": int(best_c),
    }


def _eval_rbf(X_raw: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> dict:
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_raw[tr])
    X_te = sc.transform(X_raw[te])
    grid = {"C": C_GRID, "gamma": ["scale", 0.01, 0.1, 1.0]}
    clf = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced"),
        grid, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1,
    )
    clf.fit(X_tr, y[tr])
    yp = clf.predict(X_te)
    return {
        "macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0)),
        "best_C": int(clf.best_params_["C"]),
        "best_gamma": str(clf.best_params_["gamma"]),
    }


def _kernel_health(K: np.ndarray, y: np.ndarray) -> dict:
    n = K.shape[0]
    iu = np.triu_indices(n, k=1)
    od = K[iu]
    od_mean = float(od.mean()); od_var = float(od.var())
    Y = (y[:, None] == y[None, :]).astype(np.float64) * 2 - 1
    Yc = Y - Y.mean(); Kc = K - K.mean()
    kta = float((Kc * Yc).sum() /
                (np.sqrt((Kc * Kc).sum()) * np.sqrt((Yc * Yc).sum()) + 1e-12))
    return {"off_diag_mean": od_mean, "off_diag_var": od_var, "kta_product": kta}


def _measure_w_psi_minus(X_enc: np.ndarray) -> Dict[str, float]:
    """Empirical mean of w_{Psi-} = sin^2(Delta/2) over the four redundant pairs."""
    out = {}
    for k in range(N_INFORM_DIM):
        d = X_enc[:, k] - X_enc[:, k + 4]
        w = np.sin(d / 2.0) ** 2
        out[f"E_w_psi_minus_pair_{k}_{k + 4}"] = float(w.mean())
    return out


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    log.info("=" * 70)
    log.info("E40 -- Correlation-rescue verification of Proposition 4")
    log.info("=" * 70)
    log.info("Seeds: %s   rho values: %s", SEEDS, RHO_LIST)
    log.info("N_total=%d  n_qubits=%d  reps=%d  tau=%.2f",
             N_TOTAL, N_QUBITS, REPS, TAU_LOCKED)

    rows: List[dict] = []

    for seed in SEEDS:
        for rho in RHO_LIST:
            log.info("\n--- seed=%d  rho=%.2f ---", seed, rho)
            X_raw, X_enc, y = make_synthetic_dataset(rho, seed)
            corrs = measure_pair_correlations(X_raw)
            wmean = _measure_w_psi_minus(X_enc)
            log.info("  measured pair correlations: %s",
                     {k: round(v, 3) for k, v in corrs.items()})
            log.info("  E[w_{Psi-}] over redundant pairs: %s",
                     {k: round(v, 4) for k, v in wmean.items()})

            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=TEST_FRAC, random_state=seed,
            )
            tr, te = next(sss.split(np.zeros(len(y)), y))

            # RBF baseline
            rbf = _eval_rbf(X_raw, y, tr, te)
            rows.append({
                "seed": seed, "rho": rho, "method": "RBF-SVM",
                "macro_f1": rbf["macro_f1"], "best_C": rbf["best_C"],
                "best_gamma": rbf["best_gamma"],
                "off_diag_mean": np.nan, "off_diag_var": np.nan,
                "kta_product": np.nan,
                **{k: float(v) for k, v in corrs.items()},
                **{k: float(v) for k, v in wmean.items()},
            })
            log.info("  RBF-SVM: F1=%.4f", rbf["macro_f1"])

            for method, sg in [("BSCM-uniform", False), ("SG-BSCM", True)]:
                K_path = os.path.join(
                    SAVE_DIR,
                    f"K_{method}_seed{seed}_rho{rho:.2f}.npy",
                )
                if os.path.exists(K_path):
                    log.info("  [CACHE] %s", method)
                    K = np.load(K_path)
                else:
                    t0 = time.time()
                    K = compute_bscm_fidelity_kernel(
                        X_enc, n_qubits=N_QUBITS, reps=REPS,
                        tau=TAU_LOCKED, coupling_threshold=COUPLING_THR,
                        connectivity="all",
                        bell_weights=BELL_WEIGHT_PRESETS["uniform"],
                        kernel_save_path=K_path,
                        desc=f"{method} seed={seed} rho={rho}",
                        singlet_gated=sg,
                    )
                    log.info("  Built %s in %.0fs", method, time.time() - t0)

                health = _kernel_health(K, y)
                ek = _eval_kernel(K, y, tr, te, seed=seed)
                rows.append({
                    "seed": seed, "rho": rho, "method": method,
                    "macro_f1": ek["macro_f1"], "best_C": ek["best_C"],
                    "best_gamma": "",
                    **health,
                    **{k: float(v) for k, v in corrs.items()},
                    **{k: float(v) for k, v in wmean.items()},
                })
                log.info(
                    "  %s rho=%.2f: F1=%.4f  off=%.4f  KTA=%.3f",
                    method, rho, ek["macro_f1"],
                    health["off_diag_mean"], health["kta_product"],
                )

    # ---- Save ----
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR, "metrics.csv"), index=False)
    log.info("\nWrote metrics.csv (%d rows)", len(df))

    # Aggregate per (method, rho)
    summary: Dict = {
        "experiment": "E40_rho_sweep",
        "config": {
            "n_qubits": N_QUBITS, "reps": REPS, "tau": TAU_LOCKED,
            "n_total": N_TOTAL, "n_classes": N_CLASSES,
            "rho_list": RHO_LIST, "seeds": SEEDS,
        },
        "results": {},
    }
    for method in df["method"].unique():
        sub = df[df["method"] == method]
        per_rho = {}
        for rho in RHO_LIST:
            ss = sub[sub["rho"] == rho]
            per_rho[str(rho)] = {
                "f1_mean": float(ss["macro_f1"].mean()),
                "f1_std": float(ss["macro_f1"].std()),
                "off_diag_mean": float(ss["off_diag_mean"].mean()),
                "kta_product": float(ss["kta_product"].mean()),
            }
        summary["results"][method] = per_rho

    # Rescue gap = SG-BSCM - BSCM-uniform per rho (paired, mean over seeds)
    gap = {}
    for rho in RHO_LIST:
        a = df[(df["method"] == "SG-BSCM") & (df["rho"] == rho)]["macro_f1"]
        b = df[(df["method"] == "BSCM-uniform") & (df["rho"] == rho)]["macro_f1"]
        gap[str(rho)] = {
            "rescue_gap_mean": float((a.values - b.values).mean()),
            "rescue_gap_std": float((a.values - b.values).std()),
        }
    summary["rescue_gap"] = gap

    with open(os.path.join(SAVE_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote summary.json")
    log.info("DONE.")


if __name__ == "__main__":
    main()
