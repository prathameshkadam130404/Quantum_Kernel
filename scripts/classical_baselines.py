"""
Classical baselines for matched comparison against CPK / GCK pipelines.

Runs three baselines on the SAME stratified split used by
`cpk.run_experiment` (same HDF5 file, same seed, same N per class):

  1. RBF-SVM on hand-crafted polarimetric + spectral features.
       - Polarimetric: 2x2 coherency T entries (dual-pol Sentinel-1),
         eigenvalue-derived entropy/anisotropy/mean-angle.
       - Spectral:    per-band mean & variance on 10 Sentinel-2 bands,
         plus NDVI / NDWI / NBR.
  2. Incoherent product kernel: K = K_pol(RBF) * K_spec(RBF).  This is the
     classical analogue of the modality-tensoring done by CPK/GCK.
  3. Shallow 2-layer CNN on raw (32,32,18) stacked patches (SAR 8 + S2 10).

Output: JSON blob with {accuracy, macro_f1, weighted_f1, timing} for each
baseline, written to the same RESULTS_DIR used by the CPK experiment so the
two can be diffed directly.

Usage:
    python -m scripts.classical_baselines \
        --h5-train data/raw/training.h5 \
        --h5-test  data/raw/testing.h5  \
        --n-train 500 --n-test 200 --seed 42

The CLI args mirror `cpk.run_experiment` to keep the comparison honest.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpk import cpk_config
from cpk.data.loader import (
    _detect_h5_keys,
    _load_labels_h5,
    _load_patches_by_indices,
    _stratified_indices,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("classical_baselines")


# ── Feature extraction ────────────────────────────────────────────────────────

def polarimetric_features(sar_patches: np.ndarray) -> np.ndarray:
    """
    Hand-crafted dual-pol Sentinel-1 polarimetric features.

    Uses only channels 0-3 (Re/Im S_VV, Re/Im S_VH) — the only channels that
    empirically look like complex scattering amplitudes in SO2Sat (see
    cpk.data.scattering_matrix module docstring).

    Returns (N, 8) float64 feature matrix:
      [ log T_11, log T_22, |T_12|, arg T_12,
        entropy H, anisotropy A, mean alpha, span ]
    """
    n = sar_patches.shape[0]
    feats = np.zeros((n, 8), dtype=np.float64)
    for i in range(n):
        S_VV = sar_patches[i, :, :, 0].astype(np.float64) + 1j * sar_patches[i, :, :, 1]
        S_VH = sar_patches[i, :, :, 2].astype(np.float64) + 1j * sar_patches[i, :, :, 3]

        T11 = float(np.mean(np.abs(S_VV) ** 2))
        T22 = float(np.mean(np.abs(S_VH) ** 2))
        T12 = complex(np.mean(S_VV * np.conj(S_VH)))
        T = np.array([[T11, T12], [np.conj(T12), T22]], dtype=complex)

        eigvals = np.linalg.eigvalsh((T + T.conj().T) / 2.0)
        eigvals = np.clip(eigvals, 0.0, None)
        s = eigvals.sum()
        if s > 1e-15:
            p = eigvals / s
            H = float(-np.sum(p * np.log2(np.clip(p, 1e-15, None)))) / np.log2(2)
            A = float((p[1] - p[0]) / (p[1] + p[0] + 1e-15))
        else:
            H, A = 0.0, 0.0

        feats[i, 0] = np.log1p(T11)
        feats[i, 1] = np.log1p(T22)
        feats[i, 2] = np.abs(T12)
        feats[i, 3] = np.angle(T12)
        feats[i, 4] = H
        feats[i, 5] = A
        feats[i, 6] = float(np.mean(np.abs(S_VV) + np.abs(S_VH)))
        feats[i, 7] = np.log1p(T11 + T22)
    return feats


def spectral_features(opt_patches: np.ndarray) -> np.ndarray:
    """
    Hand-crafted Sentinel-2 spectral features.

    Returns (N, 23) float64:
      [ mean_B2 .. mean_B12,   (10)
        var_B2  .. var_B12,    (10)
        NDVI, NDWI, NBR ]      (3)
    Band order (SO2Sat sen2): B2 B3 B4 B5 B6 B7 B8 B8A B11 B12.
    """
    n = opt_patches.shape[0]
    P = opt_patches.astype(np.float64)
    if P.max() > 2.0:
        P = P / 10000.0
    P = np.clip(P, 0.0, None)

    means = P.reshape(n, -1, 10).mean(axis=1)
    vars_ = P.reshape(n, -1, 10).var(axis=1)

    red = means[:, 2]
    nir = means[:, 6]
    green = means[:, 1]
    swir1 = means[:, 8]
    swir2 = means[:, 9]
    eps = 1e-6

    ndvi = (nir - red) / (nir + red + eps)
    ndwi = (green - nir) / (green + nir + eps)
    nbr = (nir - swir2) / (nir + swir2 + eps)

    return np.concatenate(
        [means, vars_, np.stack([ndvi, ndwi, nbr], axis=1)], axis=1
    )


def raw_stack(sar_patches: np.ndarray, opt_patches: np.ndarray) -> np.ndarray:
    """
    Stack SAR (8ch) + optical (10ch) into (N, 18, 32, 32) float32 for CNN input.
    Optical is DN/10000-scaled; SAR left in its native dynamic range but divided
    by a robust scale for numerical stability.
    """
    n = sar_patches.shape[0]
    sar = sar_patches.astype(np.float32)
    sar_scale = np.quantile(np.abs(sar), 0.99) + 1e-6
    sar = sar / sar_scale

    opt = opt_patches.astype(np.float32)
    if opt.max() > 2.0:
        opt = opt / 10000.0
    opt = np.clip(opt, 0.0, 1.0)

    stacked = np.concatenate([sar, opt], axis=-1)  # (N, 32, 32, 18)
    return np.transpose(stacked, (0, 3, 1, 2))     # (N, 18, 32, 32)


# ── Data loading (matches cpk.run_experiment stratification) ──────────────────

def load_split(
    h5_path: str, n_per_class: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load raw SAR+optical patches and labels with the same stratification
    used by `cpk.data.loader.load_so2sat_h5`."""
    key_map = _detect_h5_keys(h5_path)
    labels_full = _load_labels_h5(h5_path, key_map)
    idx = _stratified_indices(labels_full, n_per_class, seed)
    sar, opt = _load_patches_by_indices(h5_path, key_map, idx,
                                        load_sar=True, load_opt=True)
    return sar, opt, labels_full[idx]


# ── Baselines ─────────────────────────────────────────────────────────────────

def run_rbf_svm(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    label: str = "rbf_svm",
) -> Dict:
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    # median heuristic for gamma
    n_sub = min(500, len(Xtr))
    sub = Xtr[np.random.RandomState(0).choice(len(Xtr), n_sub, replace=False)]
    dists = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
    med = np.median(dists[dists > 0])
    gamma = 1.0 / (2.0 * med ** 2) if med > 0 else 1.0 / Xtr.shape[1]

    clf = SVC(kernel="rbf", C=10.0, gamma=gamma, decision_function_shape="ovr")
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)

    return {
        "label": label,
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "gamma": float(gamma),
        "feature_dim": int(Xtr.shape[1]),
    }


def run_product_kernel_svm(
    pol_train: np.ndarray, pol_test: np.ndarray,
    spec_train: np.ndarray, spec_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
) -> Dict:
    """Incoherent product kernel: K = K_pol(RBF) * K_spec(RBF).
    Classical analogue of the modality-tensoring in CPK/GCK."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.metrics.pairwise import rbf_kernel

    pol_scaler = StandardScaler().fit(pol_train)
    spec_scaler = StandardScaler().fit(spec_train)
    Ptr, Pte = pol_scaler.transform(pol_train), pol_scaler.transform(pol_test)
    Str_, Ste = spec_scaler.transform(spec_train), spec_scaler.transform(spec_test)

    def median_gamma(X):
        n = min(500, len(X))
        sub = X[np.random.RandomState(0).choice(len(X), n, replace=False)]
        d = np.linalg.norm(sub[:, None] - sub[None, :], axis=-1)
        med = np.median(d[d > 0])
        return 1.0 / (2.0 * med ** 2) if med > 0 else 1.0 / X.shape[1]

    gp, gs = median_gamma(Ptr), median_gamma(Str_)

    K_train = rbf_kernel(Ptr, Ptr, gamma=gp) * rbf_kernel(Str_, Str_, gamma=gs)
    K_test = rbf_kernel(Pte, Ptr, gamma=gp) * rbf_kernel(Ste, Str_, gamma=gs)

    clf = SVC(kernel="precomputed", C=10.0, decision_function_shape="ovr")
    clf.fit(K_train, y_train)
    pred = clf.predict(K_test)

    return {
        "label": "product_kernel_svm",
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "gamma_pol": float(gp),
        "gamma_spec": float(gs),
    }


def run_shallow_cnn(
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    n_classes: int,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
) -> Dict:
    """2-conv-layer CNN on (N, 18, 32, 32) stacked patches."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import accuracy_score, f1_score

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class ShallowCNN(nn.Module):
        def __init__(self, in_ch: int, n_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.head = nn.Linear(64, n_classes)

        def forward(self, x):
            return self.head(self.features(x))

    model = ShallowCNN(X_train.shape[1], n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    ds_tr = TensorDataset(torch.from_numpy(X_train).float(),
                          torch.from_numpy(y_train).long())
    ds_te = TensorDataset(torch.from_numpy(X_test).float(),
                          torch.from_numpy(y_test).long())
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

    for ep in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * len(yb)
            pred = logits.argmax(1)
            correct += int((pred == yb).sum())
            total += len(yb)
        logger.info(f"  CNN epoch {ep+1:02d}/{epochs}: "
                    f"loss={loss_sum/total:.4f}  train_acc={correct/total:.4f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in dl_te:
            preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)

    return {
        "label": "shallow_cnn",
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted")),
        "epochs": int(epochs),
        "device": str(device),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all_baselines(
    h5_train: str, h5_test: str,
    n_train_per_class: int, n_test_per_class: int,
    seed: int = 42,
    cnn_epochs: int = 20,
    skip_cnn: bool = False,
) -> Dict:
    t0 = time.time()
    results = {
        "config": {
            "h5_train": h5_train, "h5_test": h5_test,
            "n_train_per_class": n_train_per_class,
            "n_test_per_class": n_test_per_class,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        },
        "timing": {},
    }

    logger.info("Loading train split...")
    t = time.time()
    sar_tr, opt_tr, y_tr = load_split(h5_train, n_train_per_class, seed)
    logger.info("Loading test split...")
    sar_te, opt_te, y_te = load_split(h5_test, n_test_per_class, seed + 1)
    results["timing"]["load"] = time.time() - t

    logger.info(
        f"  SAR train {sar_tr.shape}  OPT train {opt_tr.shape}  "
        f"y {np.bincount(y_tr)}"
    )

    t = time.time()
    pol_tr = polarimetric_features(sar_tr)
    pol_te = polarimetric_features(sar_te)
    spec_tr = spectral_features(opt_tr)
    spec_te = spectral_features(opt_te)
    X_concat_tr = np.concatenate([pol_tr, spec_tr], axis=1)
    X_concat_te = np.concatenate([pol_te, spec_te], axis=1)
    results["timing"]["features"] = time.time() - t

    t = time.time()
    results["rbf_svm_concat"] = run_rbf_svm(
        X_concat_tr, X_concat_te, y_tr, y_te, label="rbf_svm_concat"
    )
    logger.info(f"  RBF-SVM (concat): {results['rbf_svm_concat']['accuracy']:.4f}")
    results["timing"]["rbf_svm_concat"] = time.time() - t

    t = time.time()
    results["product_kernel_svm"] = run_product_kernel_svm(
        pol_tr, pol_te, spec_tr, spec_te, y_tr, y_te
    )
    logger.info(f"  Product-kernel SVM: {results['product_kernel_svm']['accuracy']:.4f}")
    results["timing"]["product_kernel_svm"] = time.time() - t

    if not skip_cnn:
        t = time.time()
        X_tr_cnn = raw_stack(sar_tr, opt_tr)
        X_te_cnn = raw_stack(sar_te, opt_te)
        n_classes = int(max(y_tr.max(), y_te.max()) + 1)
        results["shallow_cnn"] = run_shallow_cnn(
            X_tr_cnn, X_te_cnn, y_tr, y_te,
            n_classes=n_classes, epochs=cnn_epochs, seed=seed,
        )
        logger.info(f"  Shallow CNN: {results['shallow_cnn']['accuracy']:.4f}")
        results["timing"]["shallow_cnn"] = time.time() - t

    results["timing"]["total"] = time.time() - t0
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Classical baselines matched to CPK experiment split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--h5-train", type=str, default=None)
    p.add_argument("--h5-test", type=str, default=None)
    p.add_argument("--n-train", type=int, default=500)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cnn-epochs", type=int, default=20)
    p.add_argument("--skip-cnn", action="store_true")
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    h5_train = args.h5_train or cpk_config.H5_TRAIN
    h5_test = args.h5_test or cpk_config.H5_TEST

    out = run_all_baselines(
        h5_train=h5_train, h5_test=h5_test,
        n_train_per_class=args.n_train,
        n_test_per_class=args.n_test,
        seed=args.seed,
        cnn_epochs=args.cnn_epochs,
        skip_cnn=args.skip_cnn,
    )

    print("\n=== CLASSICAL BASELINE SUMMARY ===")
    for key in ("rbf_svm_concat", "product_kernel_svm", "shallow_cnn"):
        if key in out:
            r = out[key]
            print(f"  {key:22s}  acc={r['accuracy']:.4f}  "
                  f"macro_f1={r['macro_f1']:.4f}")

    if not args.no_save:
        results_dir = cpk_config.ensure_dir(cpk_config.RESULTS_DIR)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(results_dir, f"classical_baselines_{ts}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved: {path}")
