"""
Comprehensive accuracy & macro-F1 comparison across every cached kernel
in this repository, FAITHFUL TO EACH ORIGINATING EXPERIMENT'S PROTOCOL.

This is the Option-1 fix of the earlier draft.  Each kernel cache is
evaluated under the EXACT pool construction and split protocol of the
script that built it, so the comprehensive numbers reproduce each
per-experiment summary.json to <=0.001.

Per-row protocol matrix (verified against source):
  E32 (8q So2Sat phys-8 family):
      pool       = make_or_load_split(y_all).eval_pool           (deterministic, on disk)
      eval split = StratifiedShuffleSplit(test_size=0.30, random_state=seed)
                   for seed in {42, 43, 44, 45, 46}
  eurosat_bscm_fixed_pool & E33 (8q EuroSAT phys-8 family):
      pool       = StratifiedShuffleSplit(train_size=1000, test_size=500,
                                          random_state=0)  THEN concat [tr, te]
      eval split = StratifiedShuffleSplit(train_size=1000, test_size=500,
                                          random_state=seed) for seed in {42..46}
  bscm_14q_test (14q So2Sat phys-Fisher-14):
      pool       = E32 eval_pool, then stratified_subsample(y_eval, 800, seed=1234)
                   followed by np.sort()
      eval split = StratifiedShuffleSplit(test_size=0.30, random_state=seed)
                   for seed in {42..46}
  E38 (16q both datasets):
      pool       = pre-extracted physics_features_10k.npz / EuroSAT Fisher-16 builder
      eval split = single StratifiedShuffleSplit(n_splits=5, test_size=0.30,
                                                 random_state=42)

Outputs:
- results/comprehensive/comparison_table.csv
- results/comprehensive/comparison_table.json
- results/comprehensive/literature_table.csv
- results/comprehensive/run.log

Wall-clock estimate: ~30-60 minutes (no kernel rebuilds).

Author: Prathamesh Kadam
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, StratifiedShuffleSplit,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

OUT_DIR = os.path.join(config.RESULTS_DIR, "comprehensive")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUT_DIR, "run.log"), mode="w"),
    ],
)
log = logging.getLogger("comprehensive")

SEEDS    = [42, 43, 44, 45, 46]
C_GRID   = [1, 10, 100]
CV_FOLDS = 3


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


def _eval_K_on_split(K: np.ndarray, y: np.ndarray, tr: np.ndarray,
                     te: np.ndarray, seed: int) -> Tuple[float, float, int]:
    K_tr = K[np.ix_(tr, tr)]
    K_te = K[np.ix_(te, tr)]
    best_c = _tune_c(K_tr, y[tr], seed=seed)
    clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    yp = clf.predict(K_te)
    return (
        float(f1_score(y[te], yp, average="macro", zero_division=0)),
        float(accuracy_score(y[te], yp)),
        int(best_c),
    )


def _eval_classical_on_split(X_raw: np.ndarray, y: np.ndarray, tr: np.ndarray,
                             te: np.ndarray, seed: int, method: str
                             ) -> Tuple[float, float]:
    if method == "RBF-SVM":
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
    elif method == "RandomForest":
        rf = RandomForestClassifier(
            n_estimators=500, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
        rf.fit(X_raw[tr], y[tr])
        yp = rf.predict(X_raw[te])
    else:
        raise ValueError(method)
    return (
        float(f1_score(y[te], yp, average="macro", zero_division=0)),
        float(accuracy_score(y[te], yp)),
    )


# ============================================================================
# Per-row protocol implementations
# ============================================================================

def _splits_E32(y: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """E32 So2Sat physics-8 protocol: per-seed test_size=0.30."""
    out = []
    for seed in SEEDS:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=0.30, random_state=seed,
        )
        tr, te = next(sss.split(np.zeros(len(y)), y))
        out.append((seed, tr, te))
    return out


def _splits_E33_eurosat(y: np.ndarray, n_train: int = 1000,
                        n_test: int = 500
                        ) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """E33 EuroSAT protocol: per-seed StratifiedShuffleSplit with explicit
    train_size=1000, test_size=500. tr/te indices are within the 1500-pool
    (i.e., they index [0..1499])."""
    out = []
    for seed in SEEDS:
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=n_train, test_size=n_test, random_state=seed,
        )
        tr, te = next(sss.split(np.zeros(len(y)), y))
        out.append((seed, tr, te))
    return out


def _splits_E38(y: np.ndarray, test_size: float = 0.30
                ) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """E38 16q protocol: SINGLE StratifiedShuffleSplit(n_splits=5, ...) call.
    All 5 splits are produced from base random_state=42 with internal counter
    (NOT five separately-seeded calls)."""
    sss = StratifiedShuffleSplit(
        n_splits=5, test_size=test_size, random_state=42,
    )
    out = []
    for i, (tr, te) in enumerate(sss.split(np.zeros(len(y)), y)):
        out.append((42 + i, tr, te))  # synthetic seed label for logging
    return out


def _splits_bscm_14q(y: np.ndarray) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """14q protocol: per-seed test_size=0.30 within the 800-sample sub-pool."""
    return _splits_E32(y)  # same eval-split convention


PROTOCOL_FN = {
    "E32":            _splits_E32,
    "E33_eurosat":    _splits_E33_eurosat,
    "E38":            _splits_E38,
    "bscm_14q":       _splits_bscm_14q,
}


def evaluate_with_protocol(K: np.ndarray, X_raw: np.ndarray, y: np.ndarray,
                           protocol: str) -> Dict[str, Any]:
    """Run quantum-kernel SVM evals + classical (RBF, RF) under the given
    protocol's split convention."""
    splits = PROTOCOL_FN[protocol](y)
    f1s, accs, cs = [], [], []
    for seed, tr, te in splits:
        f1, acc, bc = _eval_K_on_split(K, y, tr, te, seed=seed)
        f1s.append(f1); accs.append(acc); cs.append(bc)
    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "best_C_mode": int(max(set(cs), key=cs.count)),
        "n_splits": len(splits),
    }


def evaluate_classical_with_protocol(X_raw: np.ndarray, y: np.ndarray,
                                     protocol: str, method: str) -> Dict[str, Any]:
    splits = PROTOCOL_FN[protocol](y)
    f1s, accs = [], []
    for seed, tr, te in splits:
        f1, acc = _eval_classical_on_split(X_raw, y, tr, te, seed, method)
        f1s.append(f1); accs.append(acc)
    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "best_C_mode": -1,
        "n_splits": len(splits),
    }


# ============================================================================
# Pool loaders (faithful to each kernel's build script)
# ============================================================================

def _load_so2sat_phys8() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E32 protocol: deterministic 1800-sample evaluation pool of So2Sat
    physics-8 features (top-8 by Fisher rank). Pool order matches the
    `make_or_load_split` cache exactly."""
    from experiments._e_common import load_physics_16
    from src.attention_kernel import select_features_by_fisher
    from experiments._bscm_split import make_or_load_split

    X_norm, X_raw_full, y_all = load_physics_16()
    sel_idx, _ = select_features_by_fisher(X_raw_full, y_all)
    X_raw = X_raw_full[:, sel_idx]
    split = make_or_load_split(y_all)
    return X_raw[split.eval_pool], X_norm[split.eval_pool, :8], y_all[split.eval_pool]


def _load_eurosat_phys8_E33() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E33 / eurosat_bscm_fixed_pool protocol: pool is the 1500-sample union
    [tr, te] of `StratifiedShuffleSplit(train_size=1000, test_size=500,
    random_state=0)`. ORDER MATTERS for kernel-row indexing."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from eurosat_data import load_eurosat_allbands, compute_physics_indices

    bm, labels, _ = load_eurosat_allbands(
        cache_path=os.path.join(config.PROCESSED_DIR, "eurosat_band_means.npz"),
    )
    phys_raw = compute_physics_indices(bm)

    pool_sss = StratifiedShuffleSplit(
        n_splits=1, train_size=1000, test_size=500, random_state=0,
    )
    tr, te = next(pool_sss.split(phys_raw, labels))
    pool_idx = np.concatenate([tr, te])

    X_raw = phys_raw[pool_idx]
    y = labels[pool_idx]

    bg_mask = np.ones(len(phys_raw), dtype=bool); bg_mask[pool_idx] = False
    sc = StandardScaler().fit(phys_raw[bg_mask])
    mm = MinMaxScaler(feature_range=(0, np.pi)).fit(sc.transform(phys_raw[bg_mask]))
    X_enc = mm.transform(sc.transform(X_raw)).clip(0, np.pi).astype(np.float64)
    return X_raw, X_enc, y


def _load_bscm_14q_pool() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """bscm_14q_test protocol: start from E32 eval_pool, then
    `stratified_subsample(y_eval, 800, seed=1234)` followed by `np.sort()`.
    Then keep the top-14 Fisher-ranked features."""
    from experiments._e_common import load_physics_16
    from experiments._bscm_split import make_or_load_split
    from src.attention_kernel import compute_fisher_ratio

    X_norm, X_raw_full, y_all = load_physics_16()
    split = make_or_load_split(y_all)
    eval_pool = split.eval_pool
    X_raw_eval = X_raw_full[eval_pool]
    X_norm_eval = X_norm[eval_pool]
    y_eval = y_all[eval_pool]

    # Stratified sub-sample of 800 from eval pool with seed=1234, sorted.
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=800, random_state=1234,
    )
    sub_idx, _ = next(sss.split(np.zeros(len(y_eval)), y_eval))
    sub_idx = np.sort(sub_idx)

    # Top-14 by Fisher (computed on the eval pool, not the sub-pool, to match build).
    fisher = compute_fisher_ratio(X_raw_eval, y_eval)
    top14 = np.argsort(fisher)[::-1][:14]

    return (
        X_raw_eval[sub_idx][:, top14],
        X_norm_eval[sub_idx][:, top14],
        y_eval[sub_idx],
    )


def _load_so2sat_max() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E38 16q So2Sat: pre-extracted physics_features_10k.npz."""
    p = os.path.join(config.PROCESSED_DIR, "physics_features_10k.npz")
    d = np.load(p)
    return d["X_train_raw"], d["X_train"], d["y_train"]


def _load_eurosat_max() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E38 16q EuroSAT: replicate the loader inside exp_e38_max_data_pqk."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from eurosat_data import load_eurosat_allbands, compute_physics_indices
    from src.attention_kernel import compute_fisher_ratio

    bm, labels, _ = load_eurosat_allbands(
        cache_path=os.path.join(config.PROCESSED_DIR, "eurosat_band_means.npz"),
    )
    phys_raw = compute_physics_indices(bm)
    bands_raw = bm.astype(np.float64)
    X_raw_all = np.hstack([phys_raw, bands_raw])

    pool_sss = StratifiedShuffleSplit(
        n_splits=1, train_size=6700, test_size=3300, random_state=0,
    )
    tr, te = next(pool_sss.split(X_raw_all, labels))
    pool_idx = np.concatenate([tr, te])
    bg_mask = np.ones(len(X_raw_all), dtype=bool); bg_mask[pool_idx] = False
    fisher = compute_fisher_ratio(X_raw_all[bg_mask], labels[bg_mask])
    top16 = np.argsort(fisher)[::-1][:16]
    return (
        X_raw_all[pool_idx][:, top16],
        None,
        labels[pool_idx],
    )


POOL_LOADER = {
    "so2sat_phys8":      _load_so2sat_phys8,
    "eurosat_phys8_E33": _load_eurosat_phys8_E33,
    "bscm_14q":          _load_bscm_14q_pool,
    "so2sat_max":        _load_so2sat_max,
    "eurosat_max":       _load_eurosat_max,
}


# ============================================================================
# Manifest: every cached kernel + its faithful protocol
# ============================================================================

def _safe_load(p: str) -> Optional[np.ndarray]:
    if not os.path.exists(p):
        return None
    if p.endswith(".npz"):
        d = np.load(p)
        for key in ["K", "kernel", d.files[0]]:
            if key in d.files:
                return d[key]
    return np.load(p)


def build_manifest() -> List[Dict]:
    R = config.RESULTS_DIR
    return [
        # ---- 8q So2Sat phys-8 (E32) ----
        dict(scale="8q", dataset="So2Sat", features="physics-8", N_pool=1800,
             method="BSCM-uniform (Fid)",
             kernel_path=os.path.join(R, "bscm", "K_bscm_physics8_uniform.npy"),
             pool_loader="so2sat_phys8", protocol="E32"),
        dict(scale="8q", dataset="So2Sat", features="physics-8", N_pool=1800,
             method="BSCM-phi_only (Fid)",
             kernel_path=os.path.join(R, "bscm", "K_bscm_physics8_phi_only.npy"),
             pool_loader="so2sat_phys8", protocol="E32"),
        dict(scale="8q", dataset="So2Sat", features="physics-8", N_pool=1800,
             method="BSCM-psi_only (Fid)",
             kernel_path=os.path.join(R, "bscm", "K_bscm_physics8_psi_only.npy"),
             pool_loader="so2sat_phys8", protocol="E32"),
        dict(scale="8q", dataset="So2Sat", features="physics-8", N_pool=1800,
             method="SRQFM-fid",
             kernel_path=os.path.join(R, "bscm", "K_srqfm_fid_physics8.npy"),
             pool_loader="so2sat_phys8", protocol="E32"),
        dict(scale="8q", dataset="So2Sat", features="physics-8", N_pool=1800,
             method="SRQFM-PQK",
             kernel_path=os.path.join(R, "bscm", "K_srqfm_pqk_physics8.npy"),
             pool_loader="so2sat_phys8", protocol="E32"),

        # ---- 8q EuroSAT phys-8 (eurosat_bscm_fixed) ----
        dict(scale="8q", dataset="EuroSAT", features="physics-8", N_pool=1500,
             method="BSCM-uniform (Fid)",
             kernel_path=os.path.join(R, "eurosat_bscm_fixed", "kernels", "K_pool_bscm_physics8.npy"),
             pool_loader="eurosat_phys8_E33", protocol="E33_eurosat"),
        dict(scale="8q", dataset="EuroSAT", features="physics-8", N_pool=1500,
             method="SRQFM-fid",
             kernel_path=os.path.join(R, "eurosat_bscm_fixed", "kernels", "K_pool_srqfm_linear_physics8.npy"),
             pool_loader="eurosat_phys8_E33", protocol="E33_eurosat"),

        # ---- 8q EuroSAT phys-8 SG-BSCM family (E33) ----
        dict(scale="8q", dataset="EuroSAT", features="physics-8", N_pool=1500,
             method="SG-BSCM (Fid)",
             kernel_path=os.path.join(R, "sg_bscm", "K_SG-BSCM_eurosat_physics8.npy"),
             pool_loader="eurosat_phys8_E33", protocol="E33_eurosat"),

        # ---- 8q EuroSAT phys-8 SG-BSCM-PQK family (E36) ----
        dict(scale="8q", dataset="EuroSAT", features="physics-8", N_pool=1500,
             method="BSCM-uniform-PQK",
             kernel_path=os.path.join(R, "sg_bscm_pqk", "K_BSCM-PQK_eurosat_physics8.npy"),
             pool_loader="eurosat_phys8_E33", protocol="E33_eurosat"),
        dict(scale="8q", dataset="EuroSAT", features="physics-8", N_pool=1500,
             method="SG-BSCM-PQK",
             kernel_path=os.path.join(R, "sg_bscm_pqk", "K_SG-BSCM-PQK_eurosat_physics8.npy"),
             pool_loader="eurosat_phys8_E33", protocol="E33_eurosat"),
        dict(scale="8q", dataset="EuroSAT", features="physics-8", N_pool=1500,
             method="SRQFM-PQK",
             kernel_path=os.path.join(R, "sg_bscm_pqk", "K_SRQFM-PQK_eurosat_physics8.npy"),
             pool_loader="eurosat_phys8_E33", protocol="E33_eurosat"),

        # ---- 14q So2Sat physics-Fisher-14 ----
        dict(scale="14q", dataset="So2Sat", features="physics-Fisher-14", N_pool=800,
             method="BSCM-uniform (Fid)",
             kernel_path=os.path.join(R, "bscm_14q_test", "K_bscm_14q_phys14_uniform_tau0.25_N800.npy"),
             pool_loader="bscm_14q", protocol="bscm_14q"),

        # ---- 16q E38 ----
        dict(scale="16q", dataset="So2Sat", features="physics-16", N_pool=10000,
             method="BSCM-PQK",
             kernel_path=os.path.join(R, "e38_max_data", "cache_so2sat.npz"),
             pool_loader="so2sat_max", protocol="E38"),
        dict(scale="16q", dataset="So2Sat", features="physics-16", N_pool=10000,
             method="SRQFM-PQK",
             kernel_path=os.path.join(R, "e38_max_data", "cache_so2sat_srqfm.npz"),
             pool_loader="so2sat_max", protocol="E38"),
        dict(scale="16q", dataset="So2Sat", features="physics-16", N_pool=10000,
             method="Standard-PQK (Havlicek)",
             kernel_path=os.path.join(R, "e38_max_data", "cache_so2sat_standard_pqk.npz"),
             pool_loader="so2sat_max", protocol="E38"),
        dict(scale="16q", dataset="EuroSAT", features="Fisher-16", N_pool=10000,
             method="BSCM-PQK",
             kernel_path=os.path.join(R, "e38_max_data", "cache_eurosat.npz"),
             pool_loader="eurosat_max", protocol="E38"),
        dict(scale="16q", dataset="EuroSAT", features="Fisher-16", N_pool=10000,
             method="SRQFM-PQK",
             kernel_path=os.path.join(R, "e38_max_data", "cache_eurosat_srqfm.npz"),
             pool_loader="eurosat_max", protocol="E38"),
        dict(scale="16q", dataset="EuroSAT", features="Fisher-16", N_pool=10000,
             method="Standard-PQK (Havlicek)",
             kernel_path=os.path.join(R, "e38_max_data", "cache_eurosat_standard_pqk.npz"),
             pool_loader="eurosat_max", protocol="E38"),
    ]


# ============================================================================
# Literature reference table (curated from published papers)
# ============================================================================

LITERATURE = [
    dict(dataset="So2Sat", paper="Sen2LCZ-Net (Qiu et al. 2020, ISPRS-J)",
         model="recurrent residual CNN", input="raw 32x32 image cube",
         metric="OA (test)", value=0.61,
         note="Sentinel-2 only; 17 classes, official cultural-10 test split"),
    dict(dataset="So2Sat", paper="ResNeXt baseline (Zhu et al. 2020, IEEE-GRS-Mag)",
         model="ResNeXt-29", input="raw 32x32 S1+S2 cube",
         metric="OA (test)", value=0.63,
         note="multi-modal CNN; canonical baseline"),
    dict(dataset="So2Sat", paper="LCZ Transformer ensembles (2023+)",
         model="SwinT / ViT", input="raw 32x32 image cube",
         metric="OA (test)", value=0.68,
         note="state-of-the-art deep models, range 0.65-0.70"),
    dict(dataset="EuroSAT", paper="Helber et al. 2019 (IEEE-JSTARS)",
         model="ResNet-50 fine-tuned", input="64x64 RGB",
         metric="OA", value=0.985,
         note="canonical EuroSAT result"),
    dict(dataset="EuroSAT", paper="EfficientNet ensembles 2022+",
         model="EfficientNet-B0/B3", input="64x64 multispectral",
         metric="OA", value=0.99, note="near-saturated; all-bands"),
    dict(dataset="EuroSAT", paper="Classical RF/SVM with NDVI/NDBI features",
         model="RF / SVM", input="hand-crafted spectral indices",
         metric="OA", value=0.85, note="typical range 0.60-0.90"),
    dict(dataset="EuroSAT (binary)", paper="Rodriguez-Grasa et al. MLST 2025",
         model="Neural Quantum Kernel + U-Net",
         input="satellite patches", metric="OA (binary solar)", value=0.90,
         note="binary solar-panel detection; not directly comparable"),
    dict(dataset="multiple", paper="Schnabel & Roth QMI 2025",
         model="FQK / PQK 9 encodings", input="64 datasets",
         metric="best F1 vs RBF",  value=-0.01,
         note="no consistent advantage; mostly synthetic, N <= 500"),
    dict(dataset="multiple", paper="Bowles et al. 2024",
         model="12 QML models", input="160 binary tasks",
         metric="vs classical", value=-0.02,
         note="classical >= QML in most cases"),
]


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    log.info("=" * 70)
    log.info("Comprehensive accuracy & macro-F1 comparison (Option-1, faithful protocols)")
    log.info("=" * 70)

    rows: List[Dict] = []
    classical_cache: Dict[Tuple[str, str, int, str], Dict[str, dict]] = {}

    manifest = build_manifest()
    log.info("Manifest: %d cached kernels", len(manifest))

    for entry in manifest:
        K = _safe_load(entry["kernel_path"])
        if K is None:
            log.warning("MISSING: %s", entry["kernel_path"])
            rows.append({**entry, "status": "missing", "n_pool_actual": -1,
                         "f1_mean": np.nan, "f1_std": np.nan,
                         "acc_mean": np.nan, "acc_std": np.nan,
                         "best_C_mode": -1, "n_splits": 0})
            continue
        try:
            X_raw, X_enc, y = POOL_LOADER[entry["pool_loader"]]()
        except Exception as e:
            log.warning("LOADER FAIL for %s: %s", entry["pool_loader"], e)
            continue

        if K.shape[0] != len(y):
            log.warning(
                "SHAPE MISMATCH for %s: K=%d, y=%d.  Skipping.",
                entry["method"], K.shape[0], len(y),
            )
            rows.append({**entry, "status": "shape_mismatch",
                         "n_pool_actual": len(y),
                         "f1_mean": np.nan, "f1_std": np.nan,
                         "acc_mean": np.nan, "acc_std": np.nan,
                         "best_C_mode": -1, "n_splits": 0})
            continue

        log.info("Evaluating %-30s | %-7s %-20s N=%5d | proto=%s",
                 entry["method"][:30], entry["dataset"], entry["features"][:20],
                 len(y), entry["protocol"])
        t0 = time.time()
        m = evaluate_with_protocol(K, X_raw, y, entry["protocol"])
        log.info(
            "    F1=%.4f +/- %.4f   ACC=%.4f +/- %.4f   bestC=%d   (%.0fs)",
            m["f1_mean"], m["f1_std"], m["acc_mean"], m["acc_std"],
            m["best_C_mode"], time.time() - t0,
        )
        rows.append({**entry, "status": "ok", "n_pool_actual": len(y), **m})

        # Classical baselines per (dataset, features, N, protocol)
        cache_key = (entry["dataset"], entry["features"],
                     entry["N_pool"], entry["protocol"])
        if cache_key not in classical_cache:
            classical_cache[cache_key] = {}
            for cm in ["RBF-SVM", "RandomForest"]:
                log.info("    Classical %s [%s] ...", cm, entry["protocol"])
                cm_m = evaluate_classical_with_protocol(
                    X_raw, y, entry["protocol"], cm,
                )
                classical_cache[cache_key][cm] = cm_m
                log.info(
                    "      %s: F1=%.4f +/- %.4f   ACC=%.4f +/- %.4f",
                    cm, cm_m["f1_mean"], cm_m["f1_std"],
                    cm_m["acc_mean"], cm_m["acc_std"],
                )

    # Append classical rows
    for (ds, fs, N, proto), classics in classical_cache.items():
        for cm, mres in classics.items():
            rows.append({
                "scale": "classical", "dataset": ds, "features": fs,
                "N_pool": N, "method": cm, "kernel_path": "(classical)",
                "pool_loader": "", "protocol": proto, "status": "ok",
                "n_pool_actual": N, **mres,
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    log.info("\nWrote %s (%d rows)", csv_path, len(df))

    summary = {"experiment": "comprehensive_comparison_option1",
               "rows": rows, "literature_reference": LITERATURE}
    with open(os.path.join(OUT_DIR, "comparison_table.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    pd.DataFrame(LITERATURE).to_csv(
        os.path.join(OUT_DIR, "literature_table.csv"), index=False,
    )

    log.info("\n%s\nFINAL TABLE (per dataset, sorted by macro-F1)\n%s",
             "=" * 70, "=" * 70)
    for ds in sorted(df["dataset"].dropna().unique()):
        sub = df[(df["dataset"] == ds) & (df["status"] == "ok")].sort_values(
            "f1_mean", ascending=False,
        )
        log.info("\n=== %s ===", ds)
        log.info("%-30s %-22s %5s %-10s %-10s %-10s %-10s %-12s",
                 "Method", "Features", "N", "F1", "F1std", "ACC", "ACCstd", "Protocol")
        for _, r in sub.iterrows():
            log.info(
                "%-30s %-22s %5d %.4f %10.4f %.4f %10.4f %-12s",
                str(r["method"])[:30], str(r["features"])[:22],
                int(r["n_pool_actual"]),
                r["f1_mean"], r["f1_std"], r["acc_mean"], r["acc_std"],
                r["protocol"],
            )

    log.info("\nLITERATURE BASELINES (saved to literature_table.csv):")
    for lit in LITERATURE:
        log.info("  %-12s %-50s %s = %s",
                 lit["dataset"], lit["paper"][:50], lit["metric"], lit["value"])
    log.info("DONE.")


if __name__ == "__main__":
    main()
