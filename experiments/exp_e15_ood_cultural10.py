"""
Experiment E15 — Cross-city OOD on Cultural-10 held-out cities.

The So2Sat LCZ42 Cultural-10 split (arXiv:1912.12171) holds out 10 cities
from different cultural regions. physics_features_16.npz already contains
the test-split features (2000 stratified samples from testing.h5).

Evaluate on OOD:
    - AGPQK     : train kernel on training physics-16, eval on test features
    - RBF-tuned : classical baseline
    - Random Forest   : strong tabular baseline
    - CNN (if cached checkpoint exists): skipped w/ note otherwise.

AGPQK on OOD requires computing a cross-kernel K[test, train] via the same
AGPQK Bloch pipeline applied to OOD features. We compute:
    K_train (2000×2000)  — cached  (results/physics_cv/full_kernels)
    K_test_train (Nte×2000) — compute now with AGPQK Bloch
    K_test_test  (Nte×Nte)  — optional (not needed for SVM predict)

Output: results/ood_cultural10/{metrics.csv, fig.pdf, K_test_train.npy}
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pennylane as qml
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             cohen_kappa_score)

import config
from src.utils import ensure_dir, setup_logging
from src.attention_kernel import (
    compute_attention_matrix, select_features_by_fisher,
    compute_per_qubit_gamma, get_attention_entanglement_pairs,
)

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "ood_cultural10"))
logger = setup_logging("exp_e15", log_file=os.path.join(RES, "e15.log"))

K_AGPQK_TRAIN = os.path.join(config.RESULTS_DIR, "physics_cv",
                             "full_kernels", "K_agpqk_full.npy")
BLOCH_TRAIN   = os.path.join(config.RESULTS_DIR, "physics_cv",
                             "full_kernels", "K_agpqk_full_bloch.npy")

N_QUBITS, REPS = config.N_QUBITS, config.ZZ_REPS


def load_physics_16_train_test():
    p = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
    d = np.load(p)
    N = config.SUBSAMPLE_TRAIN
    return (d["X_train"][:N],     d["X_train_raw"][:N],     d["y_train"][:N],
            d["X_test"][:N],      d["X_test_raw"][:N],      d["y_test"][:N])


def compute_bloch(X_sel_enc, n_qubits, reps, entangle_pairs):
    dev = config.get_device(n_qubits)

    def apply(x):
        for _ in range(reps):
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
            for i in range(n_qubits):
                qml.RZ(x[i], wires=i)
            for qi, qj in entangle_pairs:
                if qi < n_qubits and qj < n_qubits:
                    zz = (np.pi - x[qi]) * (np.pi - x[qj])
                    qml.CNOT(wires=[qi, qj])
                    qml.RZ(zz, wires=qj)
                    qml.CNOT(wires=[qi, qj])

    @qml.qnode(dev, diff_method=None)
    def mx(x):
        apply(x); return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def my(x):
        apply(x); return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
    @qml.qnode(dev, diff_method=None)
    def mz(x):
        apply(x); return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    N = len(X_sel_enc)
    bloch = np.zeros((N, 3 * n_qubits))
    for i in tqdm(range(N), desc="AGPQK Bloch OOD", leave=False):
        ex = np.array(mx(X_sel_enc[i])); ey = np.array(my(X_sel_enc[i]))
        ez = np.array(mz(X_sel_enc[i]))
        for q in range(n_qubits):
            bloch[i, 3 * q]     = ex[q]
            bloch[i, 3 * q + 1] = ey[q]
            bloch[i, 3 * q + 2] = ez[q]
    return bloch


def cross_kernel_from_bloch(bloch_te, bloch_tr, n_qubits, outer_gamma):
    nt, ntr = len(bloch_te), len(bloch_tr)
    K = np.zeros((nt, ntr))
    for i in range(nt):
        diff = bloch_te[i] - bloch_tr
        sq = np.sum(diff.reshape(ntr, n_qubits, 3) ** 2, axis=2)
        frob = 0.5 * np.sum(sq, axis=1)
        K[i, :] = np.exp(-outer_gamma * frob)
    return K


def run():
    (X_tr, X_tr_raw, y_tr,
     X_te, X_te_raw, y_te) = load_physics_16_train_test()

    # ---- Reproduce AGPQK design on training data (same as cached kernel) ----
    selected, fisher = select_features_by_fisher(X_tr_raw, y_tr,
                                                 n_select=N_QUBITS,
                                                 min_sar=4, min_opt=4)
    A = compute_attention_matrix(X_tr_raw)
    pairs = get_attention_entanglement_pairs(
        A, selected, top_k=4, require_cross_modal=True, max_appearances=1)
    gamma = compute_per_qubit_gamma(fisher, selected, gamma_base=0.5)

    # Training Bloch — reuse if cached, else compute.
    if os.path.exists(BLOCH_TRAIN):
        bloch_tr = np.load(BLOCH_TRAIN)
        logger.info(f"Loaded cached train Bloch: {bloch_tr.shape}")
    else:
        logger.info("Computing train Bloch ...")
        X_tr_enc = X_tr_raw[:, selected] * gamma[np.newaxis, :]
        bloch_tr = compute_bloch(X_tr_enc, N_QUBITS, REPS, pairs)
        np.save(BLOCH_TRAIN, bloch_tr)

    # OOD test Bloch
    logger.info("Computing OOD-test Bloch ...")
    X_te_enc = X_te_raw[:, selected] * gamma[np.newaxis, :]
    bloch_te = compute_bloch(X_te_enc, N_QUBITS, REPS, pairs)

    K_train = np.load(K_AGPQK_TRAIN)
    K_test_train = cross_kernel_from_bloch(bloch_te, bloch_tr,
                                           N_QUBITS, config.PQK_GAMMA)
    np.save(os.path.join(RES, "K_test_train.npy"), K_test_train)

    # --- AGPQK SVM: train on full 2000-train kernel, evaluate on OOD test ---
    clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
    clf.fit(K_train, y_tr)
    yp_agpqk = clf.predict(K_test_train)

    # --- RBF tuned: train on physics-16 normalized train, eval on test ---
    grid = {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1]}
    rbf = GridSearchCV(SVC(class_weight="balanced"), grid, cv=3,
                       scoring="f1_macro", n_jobs=-1)
    rbf.fit(X_tr, y_tr)
    yp_rbf = rbf.predict(X_te)

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    rf.fit(X_tr, y_tr)
    yp_rf = rf.predict(X_te)

    rows = []
    for name, yp in [("AGPQK", yp_agpqk),
                     ("RBF-tuned", yp_rbf),
                     ("RandomForest", yp_rf)]:
        rows.append(dict(
            model=name,
            macro_f1=f1_score(y_te, yp, average="macro", zero_division=0),
            balanced_acc=balanced_accuracy_score(y_te, yp),
            cohen_kappa=cohen_kappa_score(y_te, yp),
        ))
        logger.info(f"{name:15s}  F1={rows[-1]['macro_f1']:.4f}  "
                    f"BA={rows[-1]['balanced_acc']:.4f}  "
                    f"κ={rows[-1]['cohen_kappa']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "metrics.csv"), index=False)
    _plot(df)


def _plot(df):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    w = 0.26
    ax.bar(x - w, df["macro_f1"],     w, label="Macro-F1",     color="#1f77b4")
    ax.bar(x,     df["balanced_acc"], w, label="Balanced Acc", color="#ff7f0e")
    ax.bar(x + w, df["cohen_kappa"],  w, label="Cohen κ",      color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(df["model"])
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    ax.set_title("E15 — Cultural-10 OOD evaluation")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
