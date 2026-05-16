"""
E42: BSCM-uniform-PQK vs Standard-ZZ-PQK on So2Sat AE-16 deep features.

Tests whether the BSCM-vs-Standard-ZZ architectural gap observed on
hand-crafted Fisher-16 physics features at N=10,000 (E38) survives when
the 16-D input is the bottleneck of a convolutional autoencoder trained
on the same So2Sat background pool (scripts/extract_so2sat_ae16.py).

Scope
-----
Quantum-vs-quantum only.  Two kernels evaluated:
    BSCM-uniform-PQK    -- four-Bell-sector uniform-prior generator
    Standard-ZZ-PQK     -- canonical Havlicek ZZ feature map
on identical block-bipartite ladder topology, n=16, depth=6, tau=0.25,
5 stratified shuffle splits at N=10,000.  Within-BSCM-family priors
(phi-only, psi-only) and classical baselines (RBF, RF) are deliberately
out of scope; E38 already covers within-family variability, and the
question E42 answers is whether the architectural gap is feature-class
specific (hand-crafted vs learned), not whether quantum beats classical.

Caches
------
Gram matrices and Bloch vectors are cached under results/e42_ae16_pqk/.
The Bloch-extraction step takes ~1.3 hours single-CPU at n=16 per kernel,
so cache reuse is the default; pass --force in your shell wrapper or
delete the cache to rebuild.

Output: results/e42_ae16_pqk/summary.json
"""
import os
import sys
import json
import time
import logging
import numpy as np
from typing import Tuple

import pennylane as qml
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.bscm_kernel import BELL_WEIGHT_PRESETS, bscm_pauli_coefficients

# ============================================================================
# CONFIGURATION
# ============================================================================
N_QUBITS = 16
DEPTH = 6
CV_FOLDS = 3
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
N_SPLITS = 5
TAU_LOCKED = 0.25

OUT_DIR = os.path.join(config.RESULTS_DIR, "e42_ae16_pqk")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s:   %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUT_DIR, "exp_e42_option2.log"), mode="w"),
    ],
)
log = logging.getLogger("exp_e42")


# ============================================================================
# DATA LOADER
# ============================================================================
def load_ae16_data() -> Tuple[np.ndarray, np.ndarray]:
    """Loads the 10,000-sample So2Sat deep features (AE-16)."""
    p = os.path.join(config.PROCESSED_DIR, "ae16_features_10k.npz")
    if not os.path.exists(p):
        log.error(f"Cannot find {p}. Run scripts/extract_so2sat_ae16.py first.")
        sys.exit(1)
    d = np.load(p)
    return d["X_train"], d["y_train"]


# ============================================================================
# BLOCH EXTRACTORS
# ============================================================================
def _pair_topology(n_qubits: int):
    """Block-bipartite ladder identical to E38."""
    mid = n_qubits // 2
    pairs = []
    pairs.extend([(i, i + 1) for i in range(mid - 1)])
    pairs.extend([(i, i + 1) for i in range(mid, n_qubits - 1)])
    pairs.extend([(i, i + mid) for i in range(mid)])
    return pairs


def _build_bscm_uniform_extractor(n_qubits: int, reps: int):
    """Used only as fallback if the BSCM-uniform cache is unexpectedly missing."""
    dev = qml.device("default.qubit", wires=n_qubits)
    pairs = _pair_topology(n_qubits)
    weights = BELL_WEIGHT_PRESETS["uniform"]

    def _apply_circuit(x):
        for _ in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            for qi, qj in pairs:
                a_xx, a_yy, a_zz = bscm_pauli_coefficients(x[qi], x[qj], weights)
                c_xx = 2.0 * TAU_LOCKED * a_xx
                c_yy = 2.0 * TAU_LOCKED * a_yy
                c_zz = 2.0 * TAU_LOCKED * a_zz
                if abs(c_xx) > 1e-5:
                    qml.IsingXX(c_xx, wires=[qi, qj])
                if abs(c_yy) > 1e-5:
                    qml.IsingYY(c_yy, wires=[qi, qj])
                if abs(c_zz) > 1e-5:
                    qml.IsingZZ(c_zz, wires=[qi, qj])

    @qml.qnode(dev, diff_method=None)
    def mx(x):
        _apply_circuit(x)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def my(x):
        _apply_circuit(x)
        return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def mz(x):
        _apply_circuit(x)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return mx, my, mz


def _build_standard_zz_extractor(n_qubits: int, reps: int):
    """Havlicek ZZFeatureMap. Same ladder topology as E38 standard-PQK."""
    dev = qml.device("default.qubit", wires=n_qubits)
    pairs = _pair_topology(n_qubits)

    def _apply_circuit(x):
        for _ in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            for qi, qj in pairs:
                qml.CNOT(wires=[qi, qj])
                qml.RZ((np.pi - x[qi]) * (np.pi - x[qj]), wires=qj)
                qml.CNOT(wires=[qi, qj])

    @qml.qnode(dev, diff_method=None)
    def mx(x):
        _apply_circuit(x)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def my(x):
        _apply_circuit(x)
        return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def mz(x):
        _apply_circuit(x)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return mx, my, mz


def extract_bloch(X: np.ndarray, n_qubits: int, reps: int, label: str, kind: str) -> np.ndarray:
    N = len(X)
    out = np.zeros((N, n_qubits * 3), dtype=np.float32)
    if kind == "bscm_uniform":
        mx, my, mz = _build_bscm_uniform_extractor(n_qubits, reps)
    elif kind == "standard_zz":
        mx, my, mz = _build_standard_zz_extractor(n_qubits, reps)
    else:
        raise ValueError(kind)

    t0 = time.time()
    for i in range(N):
        if i > 0 and i % 1000 == 0:
            log.info(f"  {label}: {i}/{N} simulated...")
        x_i = X[i]
        out[i, 0 * n_qubits : 1 * n_qubits] = mx(x_i)
        out[i, 1 * n_qubits : 2 * n_qubits] = my(x_i)
        out[i, 2 * n_qubits : 3 * n_qubits] = mz(x_i)

    log.info(f"  {label} simulation completed in {time.time() - t0:.1f}s")
    return out


# ============================================================================
# PQK GRAM (dynamic gamma, identical to E38)
# ============================================================================
def compute_dynamic_pqk_gram(bloch: np.ndarray) -> np.ndarray:
    N = len(bloch)
    b_flat = bloch.reshape(N, -1)
    sq_norms = np.sum(b_flat ** 2, axis=1)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * np.dot(b_flat, b_flat.T)
    sq_dists = np.clip(sq_dists, 0, None)
    d = 0.5 * sq_dists
    b_var = np.var(b_flat)
    dynamic_gamma = 1.0 / (b_flat.shape[1] * b_var) if b_var > 0 else 1.0
    K = np.exp(-dynamic_gamma * d)
    np.fill_diagonal(K, 1.0)
    K = (K + K.T) / 2.0
    return K


# ============================================================================
# EVALUATION
# ============================================================================
def eval_precomputed(K_full: np.ndarray, y: np.ndarray) -> dict:
    sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.3, random_state=42)
    f1s, accs = [], []
    for tr, te in sss.split(np.zeros(len(y)), y):
        K_tr = K_full[np.ix_(tr, tr)]
        K_te = K_full[np.ix_(te, tr)]
        clf = GridSearchCV(
            SVC(kernel="precomputed", class_weight="balanced"),
            {"C": C_GRID},
            cv=CV_FOLDS,
            scoring="f1_macro",
            n_jobs=-1,
        )
        clf.fit(K_tr, y[tr])
        yp = clf.predict(K_te)
        f1s.append(f1_score(y[te], yp, average="macro"))
        accs.append(accuracy_score(y[te], yp))
    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_per_split": [float(v) for v in f1s],
    }


def _load_or_compute_K(name: str, kind: str, X_enc: np.ndarray) -> np.ndarray:
    cache_K = os.path.join(OUT_DIR, f"cache_ae16_{name}_K.npz")
    cache_bloch = os.path.join(OUT_DIR, f"cache_ae16_{name}.npz")

    if os.path.exists(cache_K):
        log.info(f"  Loaded Gram matrix from {cache_K}.")
        return np.load(cache_K)["K"]

    if os.path.exists(cache_bloch):
        log.info(f"  Loaded Bloch vectors from {cache_bloch}, recomputing Gram...")
        d = np.load(cache_bloch)
        if "K" in d.files:
            return d["K"]
        bloch = d["bloch"] if "bloch" in d.files else d[d.files[0]]
        K = compute_dynamic_pqk_gram(bloch)
        np.savez_compressed(cache_K, K=K)
        return K

    log.info(f"  No cache found; simulating Bloch vectors for {name} ({kind})...")
    bloch = extract_bloch(X_enc, N_QUBITS, DEPTH, name, kind)
    K = compute_dynamic_pqk_gram(bloch)
    np.savez_compressed(cache_K, K=K)
    return K


# ============================================================================
# MAIN
# ============================================================================
def main():
    log.info("E42: BSCM-uniform vs Standard-ZZ on So2Sat AE-16 deep features")

    X_enc, y = load_ae16_data()
    log.info(f"Loaded N={len(y)} samples. Encoding shape={X_enc.shape}")

    results = {
        "n_qubits": N_QUBITS,
        "depth": DEPTH,
        "n_samples": int(len(y)),
        "n_splits": N_SPLITS,
        "tau_locked": TAU_LOCKED,
        "feature_set": "AE-16 (autoencoder, 16 dims)",
    }

    log.info("Processing BSCM-uniform...")
    K_bscm = _load_or_compute_K("BSCM-uniform", "bscm_uniform", X_enc)
    log.info("  Evaluating SVM...")
    results["BSCM-uniform"] = eval_precomputed(K_bscm, y)
    log.info(
        f"  -> F1: {results['BSCM-uniform']['f1_mean']:.4f} +/- {results['BSCM-uniform']['f1_std']:.4f}"
        f"   Acc: {results['BSCM-uniform']['acc_mean']:.4f} +/- {results['BSCM-uniform']['acc_std']:.4f}"
    )

    # 2. Standard-ZZ (no cache yet -- this is the ~1.3 h compute)
    log.info("Processing Standard-ZZ...")
    K_zz = _load_or_compute_K("Standard-ZZ", "standard_zz", X_enc)
    log.info("  Evaluating SVM...")
    results["Standard-ZZ"] = eval_precomputed(K_zz, y)
    log.info(
        f"  -> F1: {results['Standard-ZZ']['f1_mean']:.4f} +/- {results['Standard-ZZ']['f1_std']:.4f}"
        f"   Acc: {results['Standard-ZZ']['acc_mean']:.4f} +/- {results['Standard-ZZ']['acc_std']:.4f}"
    )

    # 3. Architectural gap on deep features
    gap_f1 = results["BSCM-uniform"]["f1_mean"] - results["Standard-ZZ"]["f1_mean"]
    gap_acc = results["BSCM-uniform"]["acc_mean"] - results["Standard-ZZ"]["acc_mean"]
    results["architectural_gap"] = {
        "delta_f1": float(gap_f1),
        "delta_acc": float(gap_acc),
        "note": "Positive => BSCM-uniform beats Standard-ZZ on AE-16 deep features.",
    }

    out_json = os.path.join(OUT_DIR, "summary.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Wrote {out_json}")

    log.info("=" * 60)
    log.info("E42 OPTION 2 -- HEADLINE COMPARISON")
    log.info("=" * 60)
    log.info(
        f"  BSCM-uniform : F1 {results['BSCM-uniform']['f1_mean']:.4f}"
        f" +/- {results['BSCM-uniform']['f1_std']:.4f}"
    )
    log.info(
        f"  Standard-ZZ  : F1 {results['Standard-ZZ']['f1_mean']:.4f}"
        f" +/- {results['Standard-ZZ']['f1_std']:.4f}"
    )
    log.info(f"  Architectural F1 gap (BSCM - ZZ): {gap_f1:+.4f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
