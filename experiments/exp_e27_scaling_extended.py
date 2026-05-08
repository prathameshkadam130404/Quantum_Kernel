"""
E27 — Extended learning curve: N ∈ {2 000, 3 000, 5 000, 7 000, 10 000}.

Extends E10 (which stopped at N=2 000) to address the criticism that
conclusions drawn at 0.5 % of the 400 K So2Sat training set may not
generalise.  AGPQK is used because it uses O(N) circuit evaluations
(Bloch vector extraction) plus O(N²) kernel assembly — feasible at N=10 k
on a modern workstation/GPU, unlike FQK which requires O(N²) circuits.

Design
------
- Kernels   : AGPQK (quantum) vs RBF-SVM (tuned, classical)
- Sizes     : N ∈ {2 000, 3 000, 5 000, 7 000, 10 000}
- Seeds     : [42, 43, 44]  (3 seeds — balances runtime vs variance)
- Test set  : fixed 600 samples (stratified), excluded from training
- AGPQK C   : C=1 (consistent with E10 baseline)
- RBF-SVM   : (C, gamma) tuned by 3-fold GridSearchCV at each N

Data loading
------------
Loads directly from data/raw/training.h5 (320 K samples) to avoid the
2 000-sample cap in physics_features_16.npz.  The same 16 physics-informed
features are recomputed on-the-fly using the normalization params stored in
physics_features_16.npz so the feature scale is identical to E22/E10.

Caching
-------
Bloch vectors and kernel matrices are saved under
results/scaling_extended/K_agpqk_N{N}_seed{seed}.npy so partial runs
can be resumed.

Output: results/scaling_extended/curve.csv, curve.pdf, curve.png
"""

import os
import sys
import gc
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.attention_kernel import select_features_by_fisher, FEATURE_MODALITY_16

# ---------------------------------------------------------------------------
SIZES     = [2_000, 3_000, 5_000, 7_000, 10_000]
SEEDS     = [42, 43, 44]
N_TEST    = 600
N_QUBITS  = config.N_QUBITS    # 8
ZZ_REPS   = config.ZZ_REPS     # 2
PQK_GAMMA = config.PQK_GAMMA   # 0.67

EPS = 1e-8

SAVE_DIR = os.path.join(config.RESULTS_DIR, "scaling_extended")
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "exp_e27.log"), mode="w"),
    ],
)
logger = logging.getLogger("exp_e27")


# ---------------------------------------------------------------------------
# Physics feature extraction (mirrors scripts/test_physics_classical.py)
# ---------------------------------------------------------------------------

def _safe_ratio(a, b):
    return (a - b) / (a + b + EPS)


def _extract_16_features(sar_patches, opt_patches):
    """Extract 16 physics features from raw (N,32,32,8)+(N,32,32,10) patches."""
    opt_mean = opt_patches.mean(axis=(1, 2))   # (N, 10)
    blue  = opt_mean[:, 0]; green = opt_mean[:, 1]; red  = opt_mean[:, 2]
    vre1  = opt_mean[:, 3]; nir   = opt_mean[:, 6]; swir1 = opt_mean[:, 8]

    sar_mean = sar_patches.mean(axis=(1, 2))   # (N, 8)
    vh_lee = np.abs(sar_mean[:, 4]) + EPS
    vv_lee = np.abs(sar_mean[:, 5]) + EPS
    cov_re = sar_mean[:, 6]; cov_im = sar_mean[:, 7]

    # 8 optical
    f1  = _safe_ratio(nir, red)
    f2  = _safe_ratio(swir1, nir)
    f3  = _safe_ratio(green, nir)
    f4  = _safe_ratio(swir1 + red, nir + blue)
    f5  = 1.5 * (nir - red) / (nir + red + 0.5 + EPS)
    f6  = _safe_ratio(nir, vre1)
    f7  = _safe_ratio(green, swir1)
    f8  = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + EPS)

    # 8 SAR
    f9  = vv_lee / (vh_lee + EPS)
    f10 = vh_lee + vv_lee
    f11 = vh_lee / (vv_lee + EPS)
    cov_mag = np.sqrt(cov_re**2 + cov_im**2)
    f12 = cov_mag / (np.sqrt(vv_lee * vh_lee) + EPS)
    f13 = 10 * np.log10(vh_lee + EPS)
    f14 = 10 * np.log10(vv_lee + EPS)
    f15 = sar_patches[:, :, :, 5].std(axis=(1, 2))
    pix_nir = opt_patches[:, :, :, 6]; pix_red = opt_patches[:, :, :, 2]
    pix_ndvi = (pix_nir - pix_red) / (pix_nir + pix_red + EPS)
    f16 = pix_ndvi.std(axis=(1, 2))

    return np.stack([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16], axis=1)


def _apply_norm(features, lo, hi):
    """Robust [0, π] normalization with pre-fitted lo/hi params."""
    result = np.zeros_like(features, dtype=np.float64)
    for i in range(features.shape[1]):
        if abs(hi[i] - lo[i]) < EPS:
            result[:, i] = np.pi / 2
        else:
            scaled = np.clip((features[:, i] - lo[i]) / (hi[i] - lo[i]), 0.0, 1.0)
            result[:, i] = scaled * np.pi
    return result


# ---------------------------------------------------------------------------
# Data loader — from raw HDF5 (bypasses 2000-sample cap)
# ---------------------------------------------------------------------------

def load_physics_16_full():
    """
    Load physics-16 features for up to max(SIZES)+N_TEST samples directly
    from data/raw/training.h5.  Normalization params are taken from the
    existing physics_features_16.npz so features are on the same scale as E10/E22.
    """
    import h5py

    N_needed = max(SIZES) + N_TEST          # 10 600
    h5_path  = os.path.join(config.RAW_DIR, "training.h5")

    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"Raw HDF5 not found: {h5_path}\n"
            "Place training.h5 in data/raw/ or adjust config.RAW_DIR."
        )

    # -- Load normalization params from existing processed file ---------------
    norm_path = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"physics_features_16.npz not found: {norm_path}\n"
            "Run scripts/test_physics_classical.py first to create it."
        )
    norm_data = np.load(norm_path)
    norm_lo = norm_data["normalization_lo"]   # (16,)
    norm_hi = norm_data["normalization_hi"]   # (16,)
    logger.info("  Normalization params loaded from physics_features_16.npz")

    # -- Load all labels from HDF5 (tiny memory footprint) -------------------
    with h5py.File(h5_path, "r") as f:
        labels_raw = f["label"][:]
    if labels_raw.ndim == 2:
        y_full = np.argmax(labels_raw, axis=1)
    else:
        y_full = labels_raw.flatten().astype(int)
    N_total = len(y_full)
    logger.info(f"  HDF5 total samples: {N_total}")

    n_load = min(N_needed, N_total)

    # Stratified subset of indices (sorted for HDF5 efficiency)
    from sklearn.model_selection import StratifiedShuffleSplit as SSS
    sss = SSS(n_splits=1, train_size=n_load, random_state=42)
    (idx_load, _), = sss.split(np.zeros(N_total), y_full)
    idx_load = np.sort(idx_load)
    y_loaded = y_full[idx_load]

    # -- Load raw SAR + optical patches in chunks ----------------------------
    chunk = 1000
    n = len(idx_load)
    logger.info(f"  Loading {n} samples (SAR+Optical) from HDF5...")

    with h5py.File(h5_path, "r") as f:
        sar_dset = f["sen1"]; opt_dset = f["sen2"]
        sar = np.empty((n, *sar_dset.shape[1:]), dtype=np.float32)
        opt = np.empty((n, *opt_dset.shape[1:]), dtype=np.float32)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            sar[start:end] = sar_dset[idx_load[start:end]].astype(np.float32)
            opt[start:end] = opt_dset[idx_load[start:end]].astype(np.float32)

    logger.info(f"  SAR {sar.shape}, Optical {opt.shape} loaded")

    # -- Compute 16 physics features -----------------------------------------
    logger.info("  Extracting 16 physics features...")
    X_raw = _extract_16_features(sar, opt).astype(np.float64)
    del sar, opt
    gc.collect()

    # -- Normalize using stored params (consistent with E10/E22) -------------
    X_norm = _apply_norm(X_raw, norm_lo, norm_hi)

    logger.info(f"  Loaded: N={n}, features=16, "
                f"norm range=[{X_norm.min():.3f}, {X_norm.max():.3f}]")

    if n < max(SIZES) + N_TEST:
        logger.warning(
            f"  Dataset has only {n} samples; "
            f"max requested N+test = {max(SIZES)+N_TEST}.  "
            "Larger sizes will be skipped automatically."
        )

    return X_norm, X_raw, y_loaded


# ---------------------------------------------------------------------------
# AGPQK Bloch + kernel
# ---------------------------------------------------------------------------

def _agpqk_config(X_raw_full, y_full):
    """Compute Fisher-based feature selection and per-qubit bandwidths once."""
    sel_idx, fisher_scores = select_features_by_fisher(X_raw_full, y_full)
    fisher_sel = fisher_scores[sel_idx]

    gamma_base = 0.50
    gamma_pq   = gamma_base * (fisher_sel / fisher_sel.mean())
    gamma_pq   = np.clip(gamma_pq, 0.20, 1.50)

    modalities = [FEATURE_MODALITY_16[i] for i in sel_idx]
    pairs = [(i, j) for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS)
             if modalities[i] != modalities[j]][:4]

    return sel_idx, gamma_pq, pairs


def compute_agpqk_kernel(X_enc_subset: np.ndarray, gamma_pq: np.ndarray,
                          pairs: list, seed: int, N: int) -> np.ndarray:
    """Compute AGPQK kernel for a given subset, with disk caching."""
    import pennylane as qml
    from tqdm import tqdm

    cache = os.path.join(SAVE_DIR, f"K_agpqk_N{N}_seed{seed}.npy")
    if os.path.exists(cache):
        logger.info(f"    [CACHE] K_agpqk N={N} seed={seed}")
        return np.load(cache)

    X_enc = X_enc_subset * gamma_pq[np.newaxis, :]

    dev = config.get_device(N_QUBITS)

    def apply_circuit(x):
        for _ in range(ZZ_REPS):
            for i in range(N_QUBITS):
                qml.Hadamard(wires=i)
            for i in range(N_QUBITS):
                qml.RZ(x[i], wires=i)
            for qi, qj in pairs:
                if qi < N_QUBITS and qj < N_QUBITS:
                    zz = (np.pi - x[qi]) * (np.pi - x[qj])
                    qml.CNOT(wires=[qi, qj])
                    qml.RZ(zz, wires=qj)
                    qml.CNOT(wires=[qi, qj])

    @qml.qnode(dev, diff_method=None)
    def mx(x): apply_circuit(x); return [qml.expval(qml.PauliX(i)) for i in range(N_QUBITS)]
    @qml.qnode(dev, diff_method=None)
    def my(x): apply_circuit(x); return [qml.expval(qml.PauliY(i)) for i in range(N_QUBITS)]
    @qml.qnode(dev, diff_method=None)
    def mz(x): apply_circuit(x); return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

    n = len(X_enc)
    bloch = np.zeros((n, 3 * N_QUBITS), dtype=np.float64)
    logger.info(f"    Computing Bloch vectors: N={n}...")
    for i in tqdm(range(n), desc=f"AGPQK N={N}", leave=False):
        ex = np.array(mx(X_enc[i]))
        ey = np.array(my(X_enc[i]))
        ez = np.array(mz(X_enc[i]))
        for q in range(N_QUBITS):
            bloch[i, 3*q]   = ex[q]
            bloch[i, 3*q+1] = ey[q]
            bloch[i, 3*q+2] = ez[q]

    logger.info(f"    Assembling {n}x{n} kernel...")
    b = bloch.reshape(n, N_QUBITS, 3)
    K = np.zeros((n, n), dtype=np.float64)
    for i in tqdm(range(n), desc="Kernel rows", leave=False):
        diff  = b[i] - b
        frob  = 0.5 * np.sum(diff**2, axis=(1,2))
        K[i, :] = np.exp(-PQK_GAMMA * frob)

    np.save(cache, K)
    logger.info(f"    Saved: {cache}")
    return K


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def eval_agpqk(K_full: np.ndarray, tr: np.ndarray, te: np.ndarray,
               y: np.ndarray) -> float:
    K_tr = K_full[np.ix_(tr, tr)]
    K_te = K_full[np.ix_(te, tr)]
    clf  = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    return float(f1_score(y[te], clf.predict(K_te), average="macro",
                          zero_division=0))


def eval_rbf_tuned(X_raw_tr: np.ndarray, y_tr: np.ndarray,
                   X_raw_te: np.ndarray, y_te: np.ndarray) -> tuple:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_raw_tr)
    X_te_s = scaler.transform(X_raw_te)

    grid = {"C": [1, 10, 100], "gamma": ["scale", 0.01, 0.1, 1.0]}
    clf  = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"),
                        grid, cv=3, scoring="f1_macro", n_jobs=-1)
    clf.fit(X_tr_s, y_tr)
    f1 = float(f1_score(y_te, clf.predict(X_te_s), average="macro",
                        zero_division=0))
    return f1, clf.best_params_


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logger.info("=" * 70)
    logger.info("  E27: Extended Learning Curve (N up to 10 000)")
    logger.info("=" * 70)
    logger.info(f"  Date:  {datetime.now().isoformat()}")
    logger.info(f"  Sizes: {SIZES}")
    logger.info(f"  Seeds: {SEEDS}  |  Test size: {N_TEST}")

    X_norm, X_raw, y = load_physics_16_full()
    N_avail = len(y)

    # Fisher config computed on full loaded set (unbiased feature selection)
    sel_idx, gamma_pq, pairs = _agpqk_config(X_raw, y)
    X_sel = X_norm[:, sel_idx]   # Normalised + feature-selected, all samples
    logger.info(f"  AGPQK features: indices={sel_idx.tolist()}")

    rows = []
    for N in SIZES:
        if N + N_TEST > N_avail:
            logger.warning(f"  Skipping N={N}: only {N_avail} samples available.")
            continue

        logger.info(f"\n--- N = {N} ---")
        t_N = time.time()

        for seed in SEEDS:
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=N, test_size=N_TEST, random_state=seed
            )
            (tr, te), = sss.split(np.zeros(N_avail), y)

            # AGPQK: build kernel on training subset
            t0 = time.time()
            K_full_tr = compute_agpqk_kernel(
                X_sel[tr], gamma_pq, pairs, seed=seed, N=N
            )
            dt_q = time.time() - t0

            # Inner split within training indices for AGPQK eval
            sss_inner = StratifiedShuffleSplit(
                n_splits=1, test_size=int(0.3 * N), random_state=seed + 100
            )
            (inner_tr, inner_te), = sss_inner.split(np.zeros(len(tr)), y[tr])
            K_in_tr = K_full_tr[np.ix_(inner_tr, inner_tr)]
            K_in_te = K_full_tr[np.ix_(inner_te, inner_tr)]
            clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
            clf.fit(K_in_tr, y[tr][inner_tr])
            f1_q = float(f1_score(y[tr][inner_te], clf.predict(K_in_te),
                                  average="macro", zero_division=0))

            # RBF-SVM on held-out test set
            f1_rbf, best_rbf = eval_rbf_tuned(
                X_raw[tr], y[tr], X_raw[te], y[te]
            )

            logger.info(
                f"  N={N} seed={seed}: AGPQK={f1_q:.4f}  "
                f"RBF={f1_rbf:.4f}  ({dt_q:.0f}s quantum)"
            )
            rows.append(dict(N=N, seed=seed, kernel="AGPQK",    macro_f1=f1_q))
            rows.append(dict(N=N, seed=seed, kernel="RBF-tuned",
                             macro_f1=f1_rbf, best_params=str(best_rbf)))

        logger.info(f"  N={N} done in {time.time()-t_N:.0f}s")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(SAVE_DIR, "curve.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"\nSaved: {out_csv}")

    if len(df) > 0:
        _plot(df)
        _summary(df)
    else:
        logger.warning("No results to plot — all sizes were skipped.")
    return df


def _summary(df: pd.DataFrame):
    logger.info("\n--- Summary (mean +/- std across seeds) ---")
    for kernel in ["AGPQK", "RBF-tuned"]:
        sub = df[df["kernel"] == kernel]
        if sub.empty:
            continue
        g = sub.groupby("N")["macro_f1"].agg(["mean", "std"]).reset_index()
        logger.info(f"\n  {kernel}:")
        for _, row in g.iterrows():
            logger.info(f"    N={int(row['N']):>6}: {row['mean']:.4f} +/- {row['std']:.4f}")


def _plot(df: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colours = {"AGPQK": "#9467bd", "RBF-tuned": "#ff7f0e"}
    for k, c in colours.items():
        sub = df[df["kernel"] == k]
        if sub.empty:
            continue
        g = sub.groupby("N")["macro_f1"].agg(["mean", "std"]).reset_index()
        ax.errorbar(g["N"], g["mean"], yerr=g["std"],
                    marker="o", label=k, color=c, capsize=4)

    ax.axvline(2000, color="gray", linestyle="--", linewidth=0.8,
               label="E10 limit (N=2 000)")

    ax.set_xlabel("Training-set size N")
    ax.set_ylabel("Macro-F1")
    ax.set_title("E27 — AGPQK vs tuned RBF: extended learning curve\n"
                 "(So2Sat LCZ42, physics-16 features)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(SAVE_DIR, f"curve.{ext}")
        fig.savefig(p, dpi=200)
        logger.info(f"  Plot saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    run()
