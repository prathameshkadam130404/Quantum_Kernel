"""
Common helpers for E-series (E1..E10) experiments.
Lightweight — no circuit code lives here.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score

import config


def load_physics_16():
    """Load physics-16 features (normalised + raw) + labels."""
    p = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
    d = np.load(p)
    return (d["X_train"][: config.SUBSAMPLE_TRAIN],
            d["X_train_raw"][: config.SUBSAMPLE_TRAIN],
            d["y_train"][: config.SUBSAMPLE_TRAIN])


def load_pca_8():
    p = os.path.join(config.PROCESSED_DIR, "subsample_2000.npz")
    d = np.load(p)
    return d["fused_X_train"], d["y_train"]


def spectral_metrics(K):
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = K[mask]
    Ksym = (K + K.T) / 2.0
    eig = np.clip(np.linalg.eigvalsh(Ksym), 1e-12, None)
    p = eig / eig.sum()
    eff_rank_shannon = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    return {
        "off_diag_mean":    float(off.mean()),
        "off_diag_var":     float(off.var()),
        "polarization":     float(np.mean(off ** 2) - off.mean() ** 2),
        "eff_rank_shannon": eff_rank_shannon,
        "eff_rank_huang":   float(eig.sum() / eig.max()),
    }


def svm_eval(K_full, y, seed, test_frac=0.3, C=1.0):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac,
                                 random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)
    K_tr = K_full[np.ix_(tr, tr)]
    K_te = K_full[np.ix_(te, tr)]
    clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    yp = clf.predict(K_te)
    return {
        "macro_f1": float(f1_score(y[te], yp, average="macro")),
        "accuracy": float(accuracy_score(y[te], yp)),
        "train_idx": tr, "test_idx": te,
    }


def holm_bonferroni(pvals, alpha=0.05):
    """Return corrected significance flags (True = still significant)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(m, dtype=bool)
    for rank, idx in enumerate(order):
        thresh = alpha / (m - rank)
        corrected[idx] = pvals[idx] < thresh
        if not corrected[idx]:
            for jdx in order[rank + 1:]:
                corrected[jdx] = False
            break
    return corrected


def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s_pooled = np.sqrt(((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
    if s_pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / s_pooled)


def build_agpqk_kernel(X_raw_all, y_all, selected_indices, gamma_per_qubit,
                       entangle_pairs, n_qubits=8, reps=2,
                       outer_gamma=config.PQK_GAMMA):
    """Compute AGPQK Bloch + kernel given explicit knobs (no attention call)."""
    import pennylane as qml
    from tqdm import tqdm
    X_sel = X_raw_all[:, selected_indices]
    X_enc = X_sel * gamma_per_qubit[np.newaxis, :]

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

    N = len(X_enc)
    bloch = np.zeros((N, 3 * n_qubits))
    for i in tqdm(range(N), desc="AGPQK Bloch", leave=False):
        ex = np.array(mx(X_enc[i])); ey = np.array(my(X_enc[i]))
        ez = np.array(mz(X_enc[i]))
        for q in range(n_qubits):
            bloch[i, 3 * q]     = ex[q]
            bloch[i, 3 * q + 1] = ey[q]
            bloch[i, 3 * q + 2] = ez[q]

    K = np.zeros((N, N))
    for i in range(N):
        diff = bloch[i] - bloch
        sq = np.sum(diff.reshape(N, n_qubits, 3) ** 2, axis=2)
        frob = 0.5 * np.sum(sq, axis=1)
        K[i, :] = np.exp(-outer_gamma * frob)
    return K, bloch
