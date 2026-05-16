"""
E43: BSCM-PQK vs Standard-ZZ-PQK on UCI tabular datasets.

Tests whether the BSCM-vs-Standard-ZZ architectural gap observed on
Earth-observation data (E26 / E32 / E37 / E38) also holds on tabular UCI
classification problems.  Four datasets are evaluated:

    Digits         10 classes, 64-d (PCA-16 in pipeline), N = 1,797
    Breast Cancer   2 classes, 30-d (PCA-16),             N =   569
    Letter Recogn. 26 classes, 16-d (no PCA padding),     N = 20,000 -> 2,000 subsampled
    Wine            3 classes, 13-d (PCA-13 + zero pad), N =   178

Protocol per dataset
--------------------
* StandardScaler -> PCA(min(16, n_features)) -> zero-pad to 16 if needed
  -> MinMaxScaler[0, pi] for quantum encoding.  Classical baselines use the
  unpadded PCA features (no zero columns) so that StandardScaler's `scale`
  gamma heuristic is not perturbed by zero-variance dimensions.
* Bloch-vector PQK at n = 16, depth = 6, tau = 0.25, block-bipartite ladder
  connectivity identical to E38.
* Joint (C, gamma) search over the union of two grids that are searched
  identically for both quantum kernels:
    - Multiplier grid: GAMMA_MULTIPLIERS * dyn_gamma_train
    - Absolute grid:    GAMMA_ABSOLUTE (independent of training-fold variance)
  Inner 3-fold CV picks the best (C, gamma) tuple by macro-F1.
* Outer evaluation: 5 stratified shuffle splits with master seed 42.

Output
------
results/e43_generalization/generalization_results.json -- per-method
    accuracy / F1 means and standard deviations, per-split vectors,
    chosen (C, gamma_absolute, gamma_multiplier, gamma_source) per fold.
results/e43_generalization/exp_e43.log

Zero-leakage discipline: StandardScaler, PCA, MinMaxScaler, dynamic-gamma
variance, and the PQK Gram are all fit on the training fold only.
"""
import os
import sys
import json
import time
import logging
import numpy as np
from typing import Tuple, Dict

import pennylane as qml
from sklearn.datasets import (
    load_breast_cancer, load_digits, load_wine, fetch_openml,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.bscm_kernel import bscm_pauli_coefficients, BELL_WEIGHT_PRESETS

# ============================================================================
# CONFIGURATION
# ============================================================================
N_QUBITS = 16
DEPTH = 6
CV_FOLDS = 5
C_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
GAMMA_GRID = ["scale", "auto", 0.001, 0.01, 0.1, 1.0, 10.0]  # RBF-SVM only
# Quantum-kernel gamma search.  Both kernels see the union of the two grids
# below; this is the symmetric (C, gamma) search referenced in the paper.
GAMMA_MULTIPLIERS = [0.1, 0.5, 1.0, 2.0, 5.0]      # multiplied by training-fold dyn_gamma
GAMMA_ABSOLUTE    = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]  # shared absolute scale
INNER_CV_FOLDS = 3

OUT_DIR = os.path.join(config.RESULTS_DIR, "e43_generalization")
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(name)s] %(levelname)s:   %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(OUT_DIR, "exp_e43.log"), mode="w"),
    ]
)
log = logging.getLogger("exp_e43")


# ============================================================================
# BLOCH VECTOR SIMULATION (Fast PQK Extraction)
# ============================================================================

def _build_bscm_bloch_extractor(n_qubits: int, reps: int, bell_weights: tuple):
    dev = qml.device("default.qubit", wires=n_qubits)
    
    mid = n_qubits // 2
    pairs = []
    pairs.extend([(i, i + 1) for i in range(mid - 1)])
    pairs.extend([(i, i + 1) for i in range(mid, n_qubits - 1)])
    pairs.extend([(i, i + mid) for i in range(mid)])
    
    TAU_LOCKED = 0.25

    def _apply_circuit(x: np.ndarray):
        for _ in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            
            for qi, qj in pairs:
                a_xx, a_yy, a_zz = bscm_pauli_coefficients(x[qi], x[qj], bell_weights)
                c_xx = 2.0 * TAU_LOCKED * a_xx
                c_yy = 2.0 * TAU_LOCKED * a_yy
                c_zz = 2.0 * TAU_LOCKED * a_zz
                
                if abs(c_xx) > 1e-5: qml.IsingXX(c_xx, wires=[qi, qj])
                if abs(c_yy) > 1e-5: qml.IsingYY(c_yy, wires=[qi, qj])
                if abs(c_zz) > 1e-5: qml.IsingZZ(c_zz, wires=[qi, qj])

    @qml.qnode(dev, diff_method=None)
    def mx(x): _apply_circuit(x); return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def my(x): _apply_circuit(x); return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def mz(x): _apply_circuit(x); return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return mx, my, mz


def _build_standard_bloch_extractor(n_qubits: int, reps: int):
    dev = qml.device("default.qubit", wires=n_qubits)
    
    mid = n_qubits // 2
    pairs = []
    pairs.extend([(i, i + 1) for i in range(mid - 1)])
    pairs.extend([(i, i + 1) for i in range(mid, n_qubits - 1)])
    pairs.extend([(i, i + mid) for i in range(mid)])

    def _apply_circuit(x: np.ndarray):
        for _ in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            
            for qi, qj in pairs:
                theta = 2.0 * (np.pi - x[qi]) * (np.pi - x[qj])
                qml.IsingZZ(theta, wires=[qi, qj])

    @qml.qnode(dev, diff_method=None)
    def mx(x): _apply_circuit(x); return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def my(x): _apply_circuit(x); return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def mz(x): _apply_circuit(x); return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return mx, my, mz


def extract_bloch_vectors(X: np.ndarray, name: str, is_standard: bool = False) -> np.ndarray:
    N = len(X)
    out = np.zeros((N, N_QUBITS * 3), dtype=np.float32)
    
    if is_standard:
        mx, my, mz = _build_standard_bloch_extractor(N_QUBITS, DEPTH)
    else:
        mx, my, mz = _build_bscm_bloch_extractor(N_QUBITS, DEPTH, BELL_WEIGHT_PRESETS["uniform"])
    
    for i in range(N):
        x_i = X[i]
        out[i, 0*N_QUBITS : 1*N_QUBITS] = mx(x_i)
        out[i, 1*N_QUBITS : 2*N_QUBITS] = my(x_i)
        out[i, 2*N_QUBITS : 3*N_QUBITS] = mz(x_i)
        
    return out


# ============================================================================
# EVALUATION & PIPELINE
# ============================================================================

def run_experiment(dataset_name: str) -> Dict:
    log.info(f"\nEvaluating Dataset: {dataset_name}")
    if dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X_raw, y = data.data, data.target
    elif dataset_name == "digits":
        data = load_digits()
        X_raw, y = data.data, data.target
    elif dataset_name == "wine":
        # 13 features, 3 classes, 178 samples -- kept for compatibility with
        # earlier protocol versions; pads quantum encoding to n=16 with three
        # zero columns.
        data = load_wine()
        X_raw, y = data.data, data.target
    elif dataset_name == "letter":
        # UCI Letter Recognition (Frey & Slate 1991): 16 statistical features
        # computed from machine-printed letter images, 26 classes (A-Z),
        # 20,000 samples.  Feature dimension matches n=16 exactly, so no PCA
        # padding is needed.
        log.info("  Fetching UCI Letter Recognition (OpenML)...")
        data = fetch_openml("letter", version=1, as_frame=False)
        X_raw_full = data.data.astype(np.float64)
        y_str = data.target  # ASCII letters as strings
        # Convert single-character labels to 0-25 integer codes
        y_full = np.array([ord(c) - ord("A") for c in y_str], dtype=np.int64)
        # Subsample to 2000 stratified, matching the protocol scale of the
        # other UCI datasets in this experiment.
        sub = StratifiedShuffleSplit(
            n_splits=1, train_size=2000, test_size=None, random_state=42,
        )
        (sub_idx, _), = sub.split(X_raw_full, y_full)
        X_raw = X_raw_full[sub_idx]
        y = y_full[sub_idx]
        log.info("  Letter subsample: N=%d, 26 classes (A-Z), 16 features",
                 len(y))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    sss = StratifiedShuffleSplit(n_splits=CV_FOLDS, test_size=0.3, random_state=42)
    
    results = {
        "RandomForest": {"f1": [], "acc": []},
        "RBF-SVM": {"f1": [], "acc": []},
        "Standard-ZZ": {"f1": [], "acc": [],
                        "best_C": [], "best_gamma_multiplier": [],
                        "dyn_gamma_train": []},
        "BSCM-uniform": {"f1": [], "acc": [],
                         "best_C": [], "best_gamma_multiplier": [],
                         "dyn_gamma_train": []},
    }
    
    fold = 1
    for tr, te in sss.split(X_raw, y):
        log.info(f"  Fold {fold}/{CV_FOLDS}...")
        
        # Pipeline fit on training fold only.
        scaler = StandardScaler()
        X_tr_std = scaler.fit_transform(X_raw[tr])
        X_te_std = scaler.transform(X_raw[te])

        # If the dataset has fewer features than N_QUBITS, compute as many
        # PCs as available and zero-pad to N_QUBITS so the n=16 circuit has
        # a well-defined input.  Both quantum kernels see the same padding.
        max_pcs = min(N_QUBITS, X_tr_std.shape[1], X_tr_std.shape[0] - 1)
        pca = PCA(n_components=max_pcs, random_state=42)
        X_tr_pca_core = pca.fit_transform(X_tr_std)
        X_te_pca_core = pca.transform(X_te_std)
        if max_pcs < N_QUBITS:
            pad_tr = np.zeros((X_tr_pca_core.shape[0], N_QUBITS - max_pcs))
            pad_te = np.zeros((X_te_pca_core.shape[0], N_QUBITS - max_pcs))
            X_tr_pca = np.hstack([X_tr_pca_core, pad_tr])
            X_te_pca = np.hstack([X_te_pca_core, pad_te])
        else:
            X_tr_pca = X_tr_pca_core
            X_te_pca = X_te_pca_core

        # MinMax is fit on the non-padded columns only; the padded columns
        # are left at zero so trailing qubits receive RZ(0).
        mm = MinMaxScaler(feature_range=(0, np.pi))
        if max_pcs < N_QUBITS:
            mm.fit(X_tr_pca[:, :max_pcs])
            X_tr_enc_core = mm.transform(X_tr_pca[:, :max_pcs])
            X_te_enc_core = mm.transform(X_te_pca[:, :max_pcs]).clip(0, np.pi)
            X_tr_enc = np.hstack([
                X_tr_enc_core,
                np.zeros((X_tr_enc_core.shape[0], N_QUBITS - max_pcs)),
            ])
            X_te_enc = np.hstack([
                X_te_enc_core,
                np.zeros((X_te_enc_core.shape[0], N_QUBITS - max_pcs)),
            ])
        else:
            X_tr_enc = mm.fit_transform(X_tr_pca)
            X_te_enc = mm.transform(X_te_pca).clip(0, np.pi)
        
        # Classical baselines use the *unpadded* PCA features so the
        # zero-variance pad columns do not perturb sklearn's 'scale' gamma
        # heuristic via the X.var() denominator.
        X_tr_pca_clf = X_tr_pca_core
        X_te_pca_clf = X_te_pca_core

        rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_tr_pca_clf, y[tr])
        yp_rf = rf.predict(X_te_pca_clf)
        results["RandomForest"]["f1"].append(f1_score(y[te], yp_rf, average="macro"))
        results["RandomForest"]["acc"].append(accuracy_score(y[te], yp_rf))

        rbf = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), {"C": C_GRID, "gamma": GAMMA_GRID}, cv=3, n_jobs=-1)
        rbf.fit(X_tr_pca_clf, y[tr])
        yp_rbf = rbf.predict(X_te_pca_clf)
        results["RBF-SVM"]["f1"].append(f1_score(y[te], yp_rbf, average="macro"))
        results["RBF-SVM"]["acc"].append(accuracy_score(y[te], yp_rbf))
        
        # Quantum kernels: both kernels see the same (C, gamma) search.
        for q_name, is_std in [("Standard-ZZ", True), ("BSCM-uniform", False)]:
            bloch_tr = extract_bloch_vectors(X_tr_enc, q_name, is_standard=is_std)
            bloch_te = extract_bloch_vectors(X_te_enc, q_name, is_standard=is_std)

            b_flat_tr = bloch_tr.reshape(len(bloch_tr), -1)
            b_flat_te = bloch_te.reshape(len(bloch_te), -1)

            # Per-kernel dynamic gamma from training-fold Bloch variance.
            b_var_tr = np.var(b_flat_tr)
            dyn_gamma = 1.0 / (b_flat_tr.shape[1] * b_var_tr) if b_var_tr > 0 else 1.0

            # Build the squared-distance matrices once; exponentiate per gamma.
            sq_norms_tr = np.sum(b_flat_tr ** 2, axis=1)
            sq_norms_te = np.sum(b_flat_te ** 2, axis=1)
            sq_dists_tr = np.clip(
                sq_norms_tr[:, None] + sq_norms_tr[None, :]
                - 2.0 * np.dot(b_flat_tr, b_flat_tr.T),
                0, None,
            )
            sq_dists_te = np.clip(
                sq_norms_te[:, None] + sq_norms_tr[None, :]
                - 2.0 * np.dot(b_flat_te, b_flat_tr.T),
                0, None,
            )

            # Joint (gamma, C) inner-CV search over the union of:
            #   - GAMMA_MULTIPLIERS * dyn_gamma  (relative-scale tuning)
            #   - GAMMA_ABSOLUTE                 (shared absolute scale)
            # Each candidate is tagged so the chosen source is auditable.
            candidates = []  # list of (gamma_value, source_tag, tag_value)
            for gm in GAMMA_MULTIPLIERS:
                candidates.append((dyn_gamma * gm, "multiplier", float(gm)))
            for ga in GAMMA_ABSOLUTE:
                candidates.append((float(ga), "absolute", float(ga)))

            best_score = -np.inf
            best_gamma = float(dyn_gamma)
            best_source = "multiplier"
            best_tag = 1.0
            best_C = 1.0
            for gamma, source, tag in candidates:
                K_tr_g = np.exp(-gamma * 0.5 * sq_dists_tr)
                np.fill_diagonal(K_tr_g, 1.0)
                K_tr_g = (K_tr_g + K_tr_g.T) / 2.0

                inner = GridSearchCV(
                    SVC(kernel="precomputed", class_weight="balanced"),
                    {"C": C_GRID}, cv=INNER_CV_FOLDS,
                    scoring="f1_macro", n_jobs=-1,
                )
                inner.fit(K_tr_g, y[tr])
                cv_score = float(inner.best_score_)
                if cv_score > best_score:
                    best_score = cv_score
                    best_gamma = float(gamma)
                    best_source = source
                    best_tag = float(tag)
                    best_C = float(inner.best_params_["C"])

            # Fit final model at chosen (gamma, C) and predict.
            K_tr = np.exp(-best_gamma * 0.5 * sq_dists_tr)
            np.fill_diagonal(K_tr, 1.0)
            K_tr = (K_tr + K_tr.T) / 2.0
            K_te = np.exp(-best_gamma * 0.5 * sq_dists_te)

            final = SVC(kernel="precomputed", class_weight="balanced", C=best_C)
            final.fit(K_tr, y[tr])
            yp_q = final.predict(K_te)
            results[q_name]["f1"].append(f1_score(y[te], yp_q, average="macro"))
            results[q_name]["acc"].append(accuracy_score(y[te], yp_q))
            results[q_name]["best_C"].append(best_C)
            # Record the chosen point in both representations so the JSON is
            # auditable regardless of which grid the optimum came from.
            best_gm_back = (best_gamma / dyn_gamma) if dyn_gamma > 0 else None
            results[q_name]["best_gamma_multiplier"].append(
                float(best_tag) if best_source == "multiplier"
                else float(best_gm_back) if best_gm_back is not None else 0.0
            )
            results[q_name]["dyn_gamma_train"].append(float(dyn_gamma))
            results[q_name].setdefault("best_gamma_absolute", []).append(
                float(best_gamma)
            )
            results[q_name].setdefault("best_gamma_source", []).append(best_source)
            log.info(
                "    %s: best_C=%g, gamma=%.4g (source=%s, tag=%g), "
                "inner-CV f1_macro=%.4f",
                q_name, best_C, best_gamma, best_source, best_tag, best_score,
            )
            
        fold += 1
        
    # Aggregate results
    agg_res = {}
    for k, v in results.items():
        agg = {
            "acc_mean": float(np.mean(v["acc"])),
            "acc_std": float(np.std(v["acc"])),
            "f1_mean": float(np.mean(v["f1"])),
            "f1_std": float(np.std(v["f1"])),
            "f1_per_split": [float(x) for x in v["f1"]],
            "acc_per_split": [float(x) for x in v["acc"]],
        }
        if "best_C" in v:
            agg["best_C_per_split"] = [float(c) for c in v["best_C"]]
            agg["best_gamma_multiplier_per_split"] = [
                float(g) for g in v["best_gamma_multiplier"]
            ]
            agg["dyn_gamma_train_per_split"] = [
                float(g) for g in v["dyn_gamma_train"]
            ]
            agg["best_gamma_absolute_per_split"] = [
                float(g) for g in v.get("best_gamma_absolute", [])
            ]
            agg["best_gamma_source_per_split"] = list(
                v.get("best_gamma_source", [])
            )
        agg_res[k] = agg
    return agg_res


def main():
    # Letter (16 features) fills n=16 exactly; Wine (13 features) is the
    # boundary case where feature_dim < n_qubits forces zero-padding.
    datasets = ["breast_cancer", "digits", "letter", "wine"]
    final_results = {}
    
    for ds in datasets:
        t0 = time.time()
        final_results[ds] = run_experiment(ds)
        log.info(f"  Finished {ds} in {time.time()-t0:.1f}s")
        
    # Print Publication Table
    log.info("\n\n" + "="*80)
    log.info("GENERALIZATION BENCHMARK SUMMARY (ACCURACY)")
    log.info("="*80)
    log.info(f"{'Dataset':<15s} | {'RandomForest':<12s} | {'RBF-SVM':<12s} | {'Standard-ZZ':<12s} | {'BSCM-uniform':<12s}")
    log.info("-" * 80)
    for ds in datasets:
        rf = final_results[ds]["RandomForest"]["acc_mean"]
        rbf = final_results[ds]["RBF-SVM"]["acc_mean"]
        zz = final_results[ds]["Standard-ZZ"]["acc_mean"]
        bscm = final_results[ds]["BSCM-uniform"]["acc_mean"]
        log.info(f"{ds:<15s} | {rf:.4f}       | {rbf:.4f}       | {zz:.4f}       | {bscm:.4f}")
    
    with open(os.path.join(OUT_DIR, "generalization_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    main()
