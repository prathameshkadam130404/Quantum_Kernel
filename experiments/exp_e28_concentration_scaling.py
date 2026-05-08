"""
E28 — Concentration scaling: metrics vs n_qubits and reps.

Measures how exponential concentration (Thanasilp et al., Nature
Communications 15, 5200, 2024) manifests for FQK, ZZ-PQK, and SRQFM-PQK
as a function of circuit depth (reps ∈ {1, 2, 3, 4}) and qubit count
(n ∈ {2, 4, 6, 8}).

The primary diagnostic is off-diagonal variance Var[K_off]:
    - Exponential concentration predicts Var → 0 as n → ∞.
    - A concentrated kernel has Var[K_off] ≈ 0 and eff_rank → 1.

Secondary metrics: mean_offdiag, CV, eff_rank_shannon, eff_rank_huang.

Key question addressed
----------------------
E22 found SRQFM eff_rank=15.1 vs ZZ-PQK eff_rank=37.0 (n=8, reps=2).
This experiment tests whether SRQFM off-diagonal variance decays *faster*
or *slower* than ZZ-PQK with n, which is the quantity relevant to
Thanasilp et al.'s concentration theorem.

Output: results/concentration_scaling/metrics.csv, plots.
"""

import os
import sys
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._e_common import load_physics_16, spectral_metrics
from src.kernel_concentration import compute_concentration_metrics

# ---------------------------------------------------------------------------
N_SAMPLE = 500    # Fixed N for kernel computation
SEED     = 42
N_QUBITS_LIST = [2, 4, 6, 8]
REPS_LIST     = [1, 2, 3, 4]
GAMMA         = config.PQK_GAMMA   # 0.67

SAVE_DIR = os.path.join(config.RESULTS_DIR, "concentration_scaling")
os.makedirs(SAVE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "exp_e28.log"), mode="w"),
    ],
)
logger = logging.getLogger("exp_e28")


# ---------------------------------------------------------------------------
# Kernel builders — thin wrappers with disk caching
# ---------------------------------------------------------------------------

def _cache_path(kernel_name: str, n_q: int, reps: int) -> str:
    return os.path.join(SAVE_DIR, f"K_{kernel_name}_n{n_q}_r{reps}.npy")


def build_fqk(X_enc: np.ndarray, n_q: int, reps: int) -> np.ndarray:
    """Fidelity Quantum Kernel via Pennylane state-vector overlap."""
    import pennylane as qml

    cache = _cache_path("fqk", n_q, reps)
    if os.path.exists(cache):
        return np.load(cache)

    dev = config.get_device(n_q)

    def feature_map(x):
        for _ in range(reps):
            for i in range(n_q):
                qml.Hadamard(wires=i)
            for i in range(n_q):
                qml.RZ(x[i], wires=i)
            for i in range(n_q):
                for j in range(i + 1, n_q):
                    zz = (np.pi - x[i]) * (np.pi - x[j])
                    qml.CNOT(wires=[i, j])
                    qml.RZ(zz, wires=j)
                    qml.CNOT(wires=[i, j])

    @qml.qnode(dev, diff_method=None)
    def kernel_circuit(x1, x2):
        feature_map(x1)
        qml.adjoint(feature_map)(x2)
        return qml.probs(wires=range(n_q))

    N = len(X_enc)
    K = np.zeros((N, N))
    for i in tqdm(range(N), desc=f"FQK n={n_q} r={reps}", leave=False):
        for j in range(i, N):
            p = kernel_circuit(X_enc[i], X_enc[j])[0]
            K[i, j] = K[j, i] = float(p)

    np.save(cache, K)
    return K


def build_pqk(X_enc: np.ndarray, n_q: int, reps: int) -> np.ndarray:
    """Standard ZZ Projected Quantum Kernel (Bloch vector RBF)."""
    import pennylane as qml

    cache = _cache_path("pqk", n_q, reps)
    if os.path.exists(cache):
        return np.load(cache)

    dev = config.get_device(n_q)

    def feature_map(x):
        for _ in range(reps):
            for i in range(n_q):
                qml.Hadamard(wires=i)
            for i in range(n_q):
                qml.RZ(x[i], wires=i)
            for i in range(n_q):
                for j in range(i + 1, n_q):
                    zz = (np.pi - x[i]) * (np.pi - x[j])
                    qml.CNOT(wires=[i, j])
                    qml.RZ(zz, wires=j)
                    qml.CNOT(wires=[i, j])

    @qml.qnode(dev, diff_method=None)
    def meas_x(x): feature_map(x); return [qml.expval(qml.PauliX(i)) for i in range(n_q)]
    @qml.qnode(dev, diff_method=None)
    def meas_y(x): feature_map(x); return [qml.expval(qml.PauliY(i)) for i in range(n_q)]
    @qml.qnode(dev, diff_method=None)
    def meas_z(x): feature_map(x); return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

    N = len(X_enc)
    bloch = np.zeros((N, 3 * n_q))
    for i in tqdm(range(N), desc=f"PQK bloch n={n_q} r={reps}", leave=False):
        ex = np.array(meas_x(X_enc[i]))
        ey = np.array(meas_y(X_enc[i]))
        ez = np.array(meas_z(X_enc[i]))
        for q in range(n_q):
            bloch[i, 3*q]   = ex[q]
            bloch[i, 3*q+1] = ey[q]
            bloch[i, 3*q+2] = ez[q]

    b = bloch.reshape(N, n_q, 3)
    K = np.zeros((N, N))
    for i in range(N):
        diff = b[i] - b
        frob = 0.5 * np.sum(diff**2, axis=(1, 2))
        K[i, :] = np.exp(-GAMMA * frob)

    np.save(cache, K)
    return K


def build_srqfm(X_enc: np.ndarray, n_q: int, reps: int) -> np.ndarray:
    """SRQFM Projected Quantum Kernel (fidelity coupling)."""
    import pennylane as qml
    from src.srqfm_kernel import fidelity_coupling

    cache = _cache_path("srqfm", n_q, reps)
    if os.path.exists(cache):
        return np.load(cache)

    dev = config.get_device(n_q)
    threshold = 0.01
    pairs = [(i, j) for i in range(n_q) for j in range(i + 1, n_q)]

    def feature_map(x):
        for _ in range(reps):
            for i in range(n_q):
                qml.Hadamard(wires=i)
            for i in range(n_q):
                qml.RZ(x[i], wires=i)
            for qi, qj in pairs:
                coupling = fidelity_coupling(x[qi], x[qj])
                if coupling > threshold:
                    qml.IsingZZ(coupling, wires=[qi, qj])

    @qml.qnode(dev, diff_method=None)
    def meas_x(x): feature_map(x); return [qml.expval(qml.PauliX(i)) for i in range(n_q)]
    @qml.qnode(dev, diff_method=None)
    def meas_y(x): feature_map(x); return [qml.expval(qml.PauliY(i)) for i in range(n_q)]
    @qml.qnode(dev, diff_method=None)
    def meas_z(x): feature_map(x); return [qml.expval(qml.PauliZ(i)) for i in range(n_q)]

    N = len(X_enc)
    bloch = np.zeros((N, 3 * n_q))
    for i in tqdm(range(N), desc=f"SRQFM bloch n={n_q} r={reps}", leave=False):
        ex = np.array(meas_x(X_enc[i]))
        ey = np.array(meas_y(X_enc[i]))
        ez = np.array(meas_z(X_enc[i]))
        for q in range(n_q):
            bloch[i, 3*q]   = ex[q]
            bloch[i, 3*q+1] = ey[q]
            bloch[i, 3*q+2] = ez[q]

    b = bloch.reshape(N, n_q, 3)
    K = np.zeros((N, N))
    for i in range(N):
        diff = b[i] - b
        frob = 0.5 * np.sum(diff**2, axis=(1, 2))
        K[i, :] = np.exp(-GAMMA * frob)

    np.save(cache, K)
    return K


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    logger.info("=" * 70)
    logger.info("  E28: Concentration Scaling vs n_qubits and reps")
    logger.info("=" * 70)
    logger.info(f"  Date:          {datetime.now().isoformat()}")
    logger.info(f"  N_SAMPLE:      {N_SAMPLE}")
    logger.info(f"  n_qubits list: {N_QUBITS_LIST}")
    logger.info(f"  reps list:     {REPS_LIST}")
    logger.info(f"  Reference: Thanasilp et al., Nat. Commun. 15, 5200 (2024)")

    # Load data — use first N_SAMPLE normalised physics-16 features
    X_norm, X_raw, y = load_physics_16()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), N_SAMPLE, replace=False)
    y_sub = y[idx]

    rows = []

    for n_q in N_QUBITS_LIST:
        # Select top-n_q features (by Fisher score) for this qubit count
        from src.attention_kernel import select_features_by_fisher
        sel_idx, fisher_scores = select_features_by_fisher(X_raw, y)
        top_sel = sel_idx[:n_q]      # Top n_q Fisher features
        X_sub = X_norm[np.ix_(idx, top_sel)]   # (N_SAMPLE, n_q)

        logger.info(f"\n--- n_qubits={n_q}  features={top_sel.tolist()} ---")

        for reps in REPS_LIST:
            logger.info(f"  reps={reps} ...")
            t0 = time.time()

            # FQK is expensive at N=500, n=8 — use smaller N for FQK
            n_fqk = min(N_SAMPLE, 200) if n_q >= 6 else N_SAMPLE
            X_fqk = X_sub[:n_fqk]

            kernels = {}

            # FQK
            try:
                kernels["FQK"] = build_fqk(X_fqk, n_q, reps)
            except Exception as e:
                logger.warning(f"    FQK failed: {e}")

            # ZZ-PQK
            try:
                kernels["ZZ-PQK"] = build_pqk(X_sub, n_q, reps)
            except Exception as e:
                logger.warning(f"    ZZ-PQK failed: {e}")

            # SRQFM-PQK
            try:
                kernels["SRQFM-PQK"] = build_srqfm(X_sub, n_q, reps)
            except Exception as e:
                logger.warning(f"    SRQFM-PQK failed: {e}")

            dt = time.time() - t0

            for name, K in kernels.items():
                N_K = K.shape[0]
                m = compute_concentration_metrics(K)
                sm = spectral_metrics(K)

                row = {
                    "kernel":          name,
                    "n_qubits":        n_q,
                    "reps":            reps,
                    "N":               N_K,
                    "off_diag_mean":   m["mean_offdiag"],
                    "off_diag_var":    m["std_offdiag"] ** 2,
                    "off_diag_std":    m["std_offdiag"],
                    "cv":              m["cv"],
                    "eff_rank_shannon": sm["eff_rank_shannon"],
                    "eff_rank_huang":   sm["eff_rank_huang"],
                    "spectral_entropy": m["spectral_entropy"],
                    "top_eig_ratio":    m["top_eigenvalue_ratio"],
                    "n_sig_eigs":       m["n_significant_eigs"],
                    "compute_s":       dt,
                }
                rows.append(row)
                logger.info(
                    f"    {name:<12} off_var={m['std_offdiag']**2:.5f}  "
                    f"CV={m['cv']:.4f}  eff_rank={sm['eff_rank_shannon']:.1f}  "
                    f"top_eig={m['top_eigenvalue_ratio']:.4f}"
                )

    df = pd.DataFrame(rows)
    out_csv = os.path.join(SAVE_DIR, "metrics.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"\nSaved: {out_csv}")

    _analysis(df)
    _plot(df)
    return df


def _analysis(df: pd.DataFrame):
    """Log concentration trend analysis — the key scientific result."""
    logger.info("\n--- Concentration trend analysis ---")
    logger.info("  Thanasilp 2024: exponential concentration ⟺ off_diag_var → 0 as n grows.")
    logger.info("")

    for kernel in ["FQK", "ZZ-PQK", "SRQFM-PQK"]:
        sub = df[(df.kernel == kernel) & (df.reps == 2)].sort_values("n_qubits")
        if sub.empty:
            continue
        logger.info(f"  {kernel} (reps=2), off_diag_var vs n_qubits:")
        for _, row in sub.iterrows():
            logger.info(f"    n={int(row.n_qubits)}: var={row.off_diag_var:.5f}  "
                        f"CV={row.cv:.4f}  eff_rank={row.eff_rank_shannon:.1f}")

    logger.info("")
    logger.info("  SRQFM vs ZZ-PQK at n=8, reps=2 (E22 baseline):")
    for kernel in ["ZZ-PQK", "SRQFM-PQK"]:
        row = df[(df.kernel == kernel) & (df.n_qubits == 8) & (df.reps == 2)]
        if not row.empty:
            r = row.iloc[0]
            logger.info(f"    {kernel}: off_diag_var={r.off_diag_var:.5f}  "
                        f"eff_rank={r.eff_rank_shannon:.1f}")
    logger.info("")
    logger.info("  Note: lower eff_rank (SRQFM=15.1 vs ZZ-PQK=37.0) indicates")
    logger.info("  a more targeted kernel — not necessarily more concentrated.")
    logger.info("  The relevant metric is off_diag_var decay with n.")


def _plot(df: pd.DataFrame):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots.")
        return

    palette = {"FQK": "#1f77b4", "ZZ-PQK": "#ff7f0e", "SRQFM-PQK": "#2ca02c"}
    reps_default = 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, (metric, ylabel, title) in zip(axes, [
        ("off_diag_var",    "Off-diagonal variance",   "Var[K_off] vs n"),
        ("eff_rank_shannon","Effective rank (Shannon)", "Eff rank vs n"),
        ("cv",              "Coefficient of variation","CV vs n"),
    ]):
        sub = df[df.reps == reps_default]
        for kernel, color in palette.items():
            ks = sub[sub.kernel == kernel].sort_values("n_qubits")
            if not ks.empty:
                ax.plot(ks["n_qubits"], ks[metric], marker="o",
                        label=kernel, color=color)
        ax.set_xlabel("n qubits")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} (reps={reps_default})")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("E28 — Concentration scaling\n"
                 "(Thanasilp et al. 2024: concentration ⟺ off_diag_var → 0)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(SAVE_DIR, f"concentration_vs_n.{ext}")
        fig.savefig(p, dpi=200)
        logger.info(f"  Plot: {p}")
    plt.close(fig)

    # Reps sweep at n=8
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sub8 = df[df.n_qubits == 8]
    for kernel, color in palette.items():
        ks = sub8[sub8.kernel == kernel].sort_values("reps")
        if not ks.empty:
            ax2.plot(ks["reps"], ks["off_diag_var"], marker="s",
                     label=kernel, color=color)
    ax2.set_xlabel("Circuit repetitions (reps)")
    ax2.set_ylabel("Off-diagonal variance")
    ax2.set_title("E28 — Off-diagonal variance vs reps (n=8)")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        p = os.path.join(SAVE_DIR, f"concentration_vs_reps.{ext}")
        fig2.savefig(p, dpi=200)
        logger.info(f"  Plot: {p}")
    plt.close(fig2)


if __name__ == "__main__":
    run()
