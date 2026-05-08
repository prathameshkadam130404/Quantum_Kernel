"""
PIQFM v3 — Physics-Informed Quantum Feature Map (Kernel Computation).

Novel quantum kernel design for multi-sensor Earth observation:
  - 8 physics-informed features (4 optical + 4 SAR, linear-scale only)
  - 3 cross-modal ZZ bridges at physics-justified positions
  - Trainable bandwidth θ with physics-grounded bounds

Circuit (2 layers, 11 trainable params):
  Layer 1: H + RZ(θ_bw[i] · x[i]) for i=0..7
  Layer 2: 3 cross-modal ZZ bridges with trainable θ_bridge[k]
    (0↔4) NDVI ↔ CrossPol      — vegetation × volume scattering (R²=0.89)
    (3↔7) BSI ↔ PolCoherence   — bare soil × polarimetric order
    (2↔6) MNDWI ↔ SAR_total    — water × total backscatter (disambiguation)

Parameter bounds (physics-grounded):
  θ_bw ∈ [0.1, π]    — beyond θ=π, RZ encoding wraps
  θ_br ∈ [-1/π, 1/π] — keeps max ZZ angle ≤ π

Usage:
  python scripts/compute_piqfm_kernel.py --sweep     # sweep + train + kernel
  python scripts/compute_piqfm_kernel.py              # train + kernel
  python scripts/compute_piqfm_kernel.py --skip-training  # kernel only
"""

import os, sys, time, argparse, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from tqdm import tqdm

import config

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# ============ CONSTANTS ============
N_QUBITS = 8
# v5: 2 bridges only — BSI↔PolCoh dropped (converged to 0.044 in v4)
CROSS_MODAL_BRIDGES = [(0, 4), (2, 6)]
BRIDGE_NAMES = [
    "NDVI↔CrossPol",   # R²=0.89, strongest SAR-optical correlation
    "MNDWI↔SAR_tot",   # water vs urban disambiguation (LCZG advantage)
]
FEATURE_NAMES = [
    "NDVI", "NDRE", "MNDWI", "BSI",             # Optical (qubits 0-3)
    "CrossPol", "NDVI_tex", "SAR_total", "PolCoh" # SAR (qubits 4-7)
]

# v5: Replace NDBI (idx 1, suppressed in v3+v4) with NDRE (idx 5).
# NDBI has low dynamic range on So2Sat — optimizer suppressed it in 3 consecutive runs.
# NDRE (Red Edge) targets canopy chlorophyll, well-distributed, non-redundant with NDVI.
# From physics_features_16.npz:
#   0:NDVI 1:NDBI 2:NDWI 3:BSI 4:SAVI 5:NDRE 6:MNDWI 7:EVI
#   8:VV/VH 9:SAR_total 10:CrossPol 11:PolCoh 12:VH_dB 13:VV_dB 14:VV_tex 15:NDVI_tex
FEATURE_INDICES_16 = [0, 5, 6, 3, 10, 15, 9, 11]  # NDVI,NDRE,MNDWI,BSI,CrossPol,NDVI_tex,SAR_total,PolCoh

# v5 bounds:
# Bandwidth: analytic formula (π/2)/std gives values in [0.5, 2π] range.
#   Prior ceiling of π was too tight — NDVI/MNDWI/CrossPol/PolCoh all saturated at π.
#   2π is the natural periodicity of the fidelity kernel for single-qubit RZ gates.
# Bridge: raised from 1/π≈0.318 to 0.5 — both active bridges saturated at 0.318 in v4.
#   Max ZZ angle at θ_br=0.5 is 0.5×π²≈4.93 rad (within first 1.57 periods, acceptable).
THETA_BW_MIN, THETA_BW_MAX = 0.5, 2 * np.pi
THETA_BR_MIN, THETA_BR_MAX = -0.5, 0.5


# ============ PIQFM CIRCUIT ============

def apply_piqfm(x, theta_bandwidth, theta_bridge, n_qubits=N_QUBITS):
    """
    Apply the PIQFM encoding circuit.

    Layer 1: H + RZ(θ_bandwidth[i] · x[i])
    Layer 2: Cross-modal ZZ bridges with trainable strength

    Must be called inside a qml.qnode context.

    Args:
        x: Input features, shape (8,), values in [0, π].
        theta_bandwidth: Trainable bandwidth params, shape (8,).
        theta_bridge: Trainable bridge strength params, shape (4,).
    """
    # Layer 1: Hadamard + trainable RZ encoding
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        qml.RZ(theta_bandwidth[i] * x[i], wires=i)

    # Layer 2: Cross-modal ZZ bridges
    for k, (i, j) in enumerate(CROSS_MODAL_BRIDGES):
        zz_angle = theta_bridge[k] * (np.pi - x[i]) * (np.pi - x[j])
        qml.CNOT(wires=[i, j])
        qml.RZ(zz_angle, wires=j)
        qml.CNOT(wires=[i, j])


def build_piqfm_kernel_circuit(n_qubits=N_QUBITS):
    """Build the PIQFM fidelity kernel circuit: U(x) U†(x')."""
    dev = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method="best")
    def circuit(x1, x2, theta_bandwidth, theta_bridge):
        apply_piqfm(x1, theta_bandwidth, theta_bridge, n_qubits)
        qml.adjoint(apply_piqfm)(x2, theta_bandwidth, theta_bridge, n_qubits)
        return qml.probs(wires=range(n_qubits))

    return circuit


# ============ KERNEL MATRIX COMPUTATION ============

def compute_piqfm_kernel_matrix(X, theta_bandwidth, theta_bridge,
                                 X2=None, desc="PIQFM", silent=False):
    """
    Compute the PIQFM kernel matrix K(X, X) or K(X2, X).

    Args:
        X: Training data, shape (n, 8).
        theta_bandwidth: shape (8,).
        theta_bridge: shape (n_bridges,).
        X2: Optional test data, shape (m, 8). If None, compute symmetric K(X,X).
        desc: Progress bar description.
        silent: If True, suppress progress bar (used during KTA training).

    Returns:
        K: Kernel matrix. Shape (n,n) if X2 is None, else (m,n).
    """
    circuit = build_piqfm_kernel_circuit()

    if X2 is None:
        n = len(X)
        K = np.eye(n, dtype=np.float64)

        if silent:
            for i in range(n):
                for j in range(i + 1, n):
                    probs = circuit(X[i], X[j], theta_bandwidth, theta_bridge)
                    val = float(probs[0])
                    K[i, j] = val
                    K[j, i] = val
        else:
            for i in tqdm(range(n), desc=desc, leave=False):
                for j in range(i + 1, n):
                    probs = circuit(X[i], X[j], theta_bandwidth, theta_bridge)
                    val = float(probs[0])
                    K[i, j] = val
                    K[j, i] = val
    else:
        m, n = len(X2), len(X)
        K = np.zeros((m, n), dtype=np.float64)

        if silent:
            for i in range(m):
                for j in range(n):
                    probs = circuit(X2[i], X[j], theta_bandwidth, theta_bridge)
                    K[i, j] = float(probs[0])
        else:
            for i in tqdm(range(m), desc=desc, leave=False):
                for j in range(n):
                    probs = circuit(X2[i], X[j], theta_bandwidth, theta_bridge)
                    K[i, j] = float(probs[0])

    return K


# ============ KTA COMPUTATION ============

def compute_kta(K, y, class_weighted=True):
    """
    Kernel-Target Alignment: KTA(K, y) = <K, K_ideal>_F / (||K||_F · ||K_ideal||_F).
    """
    n = len(y)
    K_ideal = np.zeros((n, n), dtype=np.float64)

    if class_weighted:
        classes, counts = np.unique(y, return_counts=True)
        count_map = dict(zip(classes, counts))
        for i in range(n):
            for j in range(n):
                if y[i] == y[j]:
                    K_ideal[i, j] = 1.0 / count_map[y[i]]
    else:
        for i in range(n):
            K_ideal[i, :] = (y == y[i]).astype(float)

    # Frobenius inner product
    kta = np.sum(K * K_ideal) / (np.linalg.norm(K, 'fro') * np.linalg.norm(K_ideal, 'fro') + 1e-12)
    return kta


# ============ KTA TRAINING ============

def compute_analytic_bandwidth(X, bw_min=THETA_BW_MIN, bw_max=THETA_BW_MAX):
    """
    Compute bandwidth analytically: θ_bw[i] = clip((π/2) / std(X[:, i]), bw_min, bw_max).

    Sets 1 standard deviation of feature variation = π/2 phase rotation,
    the maximum-sensitivity point of the cosine fidelity kernel.

    This is NOT trained — it is computed once from training data and frozen.
    Training bandwidth via KTA causes monotonic saturation at the ceiling
    (Shaydulin & Wild, Phys. Rev. A 2022) and must be avoided.

    Args:
        X: Training data, shape (n, 8), values in [0, π].
        bw_min: Lower clamp (default 0.5).
        bw_max: Upper clamp (default 2π).

    Returns:
        theta_bandwidth: shape (8,), analytic per-feature bandwidth.
    """
    stds = np.std(X, axis=0)
    # Guard against near-zero std (degenerate feature)
    stds = np.maximum(stds, 1e-3)
    theta_bandwidth = np.clip((np.pi / 2) / stds, bw_min, bw_max)
    return theta_bandwidth


def train_piqfm_kta(X_sub, y_sub, theta_bandwidth, n_epochs=100,
                    lr=0.03, spsa_delta=0.05, verbose=True,
                    checkpoint_path=None):
    """
    Train PIQFM bridge parameters (θ_bridge) by maximizing KTA via SPSA+AMSGrad.

    Bandwidth (θ_bandwidth) is passed in as a fixed argument — it is computed
    analytically before calling this function and is NOT updated here.

    Only the 2 bridge parameters are optimized. The optimization problem is
    2-dimensional, which guarantees convergence in 50-100 SPSA iterations.

    SPSA gradient estimator (Spall, 1992):
        grad[k] ≈ (KTA(θ + δ·Δ) - KTA(θ - δ·Δ)) / (2·δ·Δ[k])
        where Δ is a vector of independent Rademacher ±1 perturbations.

    AMSGrad optimizer (Reddi et al., 2018) — outperforms Adam with SPSA
    because it maintains the maximum of past squared gradients, preventing
    the learning rate from increasing when gradient estimates are noisy.

    Args:
        X_sub:           KTA training subset, shape (n_sub, 8).
        y_sub:           Labels, shape (n_sub,).
        theta_bandwidth: Fixed analytic bandwidth, shape (8,). NOT updated.
        n_epochs:        SPSA iterations (default 100).
        lr:              AMSGrad learning rate (default 0.03).
        spsa_delta:      SPSA perturbation magnitude (default 0.05).
        verbose:         Print progress bar.
        checkpoint_path: Optional path to save best parameters periodically.

    Returns:
        theta_bandwidth: Unchanged — same as input (returned for API consistency).
        theta_bridge:    Trained bridge params, shape (n_bridges,).
        history:         List of (epoch, kta) tuples.
    """
    n_sub = len(X_sub)
    n_bridges = len(CROSS_MODAL_BRIDGES)

    # Initialize bridges at small nonzero values to break symmetry
    theta_bridge = np.full(n_bridges, 0.05, dtype=np.float64)

    # Build ideal kernel once (class-weighted)
    K_ideal = np.zeros((n_sub, n_sub), dtype=np.float64)
    classes, counts = np.unique(y_sub, return_counts=True)
    count_map = dict(zip(classes, counts))
    for i in range(n_sub):
        for j in range(n_sub):
            if y_sub[i] == y_sub[j]:
                K_ideal[i, j] = 1.0 / count_map[y_sub[i]]
    K_ideal_norm = np.linalg.norm(K_ideal, 'fro')

    def kta_from_K(K):
        return np.sum(K * K_ideal) / (np.linalg.norm(K, 'fro') * K_ideal_norm + 1e-12)

    # AMSGrad optimizer state (for bridge params only)
    m_br = np.zeros(n_bridges)     # first moment
    v_br = np.zeros(n_bridges)     # second moment
    v_hat_max = np.zeros(n_bridges)  # AMSGrad max second moment
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    history = []
    best_kta = -1.0
    best_bridge = theta_bridge.copy()
    patience = 15
    no_improve = 0

    rng = np.random.default_rng(42)

    if verbose:
        print(f"\n  PIQFM v5 KTA Training (SPSA+AMSGrad)")
        print(f"  Anchor: {n_sub} samples, {n_epochs} epochs, lr={lr}, δ_spsa={spsa_delta}")
        print(f"  Bandwidth: ANALYTIC (frozen) = [{', '.join(f'{v:.3f}' for v in theta_bandwidth)}]")
        print(f"  Training: {n_bridges} bridge params only")
        print(f"  Bridges: {BRIDGE_NAMES}")
        if checkpoint_path:
            print(f"  Checkpoints: {checkpoint_path} (saves best θ at each improvement)")

    pbar = tqdm(range(n_epochs), desc="KTA Training", unit="epoch", disable=not verbose)
    try:
        for epoch in pbar:
            # Current KTA
            K = compute_piqfm_kernel_matrix(X_sub, theta_bandwidth, theta_bridge, silent=True)
            kta = kta_from_K(K)
            history.append((epoch, kta))

            if kta > best_kta:
                best_kta = kta
                best_bridge = theta_bridge.copy()
                no_improve = 0
                # Intermediate checkpoint save
                if checkpoint_path:
                    try:
                        np.savez(checkpoint_path, 
                                 theta_bandwidth=theta_bandwidth, 
                                 theta_bridge=best_bridge,
                                 kta=best_kta, epoch=epoch)
                    except Exception as e:
                        pbar.write(f"  [WARN] Failed to save checkpoint: {e}")
            else:
                no_improve += 1

            if verbose:
                br_str = " ".join(f"{v:+.4f}" for v in theta_bridge)
                pbar.set_postfix_str(
                    f"KTA={kta:.4f} best={best_kta:.4f} | br=[{br_str}]"
                )

            if no_improve >= patience:
                if verbose:
                    pbar.write(f"  Early stop at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

            # SPSA gradient estimate — 2 kernel evaluations total (not 2*n_params)
            delta_vec = rng.choice([-1.0, 1.0], size=n_bridges)  # Rademacher vector
            theta_plus  = np.clip(theta_bridge + spsa_delta * delta_vec, THETA_BR_MIN, THETA_BR_MAX)
            theta_minus = np.clip(theta_bridge - spsa_delta * delta_vec, THETA_BR_MIN, THETA_BR_MAX)

            K_plus  = compute_piqfm_kernel_matrix(X_sub, theta_bandwidth, theta_plus,  silent=True)
            K_minus = compute_piqfm_kernel_matrix(X_sub, theta_bandwidth, theta_minus, silent=True)

            kta_plus  = kta_from_K(K_plus)
            kta_minus = kta_from_K(K_minus)

            grad_bridge = (kta_plus - kta_minus) / (2 * spsa_delta * delta_vec)

            # AMSGrad update
            t_step = epoch + 1
            m_br = beta1 * m_br + (1 - beta1) * grad_bridge
            v_br = beta2 * v_br + (1 - beta2) * grad_bridge ** 2
            m_hat = m_br / (1 - beta1 ** t_step)
            v_hat = v_br / (1 - beta2 ** t_step)
            v_hat_max = np.maximum(v_hat_max, v_hat)  # AMSGrad key update

            theta_bridge += lr * m_hat / (np.sqrt(v_hat_max) + eps_adam)
            theta_bridge = np.clip(theta_bridge, THETA_BR_MIN, THETA_BR_MAX)
    except KeyboardInterrupt:
        if verbose:
            pbar.write(f"\n  [INTERRUPT] Training manually stopped. Returning best θ found so far (KTA={best_kta:.4f}).")
    finally:
        pbar.close()

    if verbose:
        print(f"\n  Final best KTA: {best_kta:.4f}")
        print(f"  θ_bw (analytic) = [{', '.join(f'{v:.3f}' for v in theta_bandwidth)}]")
        print(f"  θ_br (trained)  = [{', '.join(f'{v:+.4f}' for v in best_bridge)}]")

    return theta_bandwidth, best_bridge, history


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description="PIQFM v2 Kernel Computation")
    parser.add_argument("--skip-training", action="store_true",
                       help="Load previously trained θ instead of retraining")
    parser.add_argument("--kernel-size", type=int, default=2000,
                       help="Number of samples for full kernel (default: 2000)")
    parser.add_argument("--train-subset", type=int, default=102,
                       help="Samples for KTA training anchor (default: 102, ~6/class). "
                            "Must be stratified — minority classes need ≥2 samples.")
    parser.add_argument("--n-epochs", type=int, default=100,
                       help="SPSA iterations for bridge training (default: 100)")
    parser.add_argument("--lr", type=float, default=0.03,
                       help="AMSGrad learning rate for bridge params (default: 0.03)")
    parser.add_argument("--delta", type=float, default=0.05,
                       help="SPSA perturbation magnitude (default: 0.05)")
    args = parser.parse_args()

    print("=" * 70)
    print("PIQFM v5: Physics-Informed Quantum Feature Map")
    print("  8 features, 2 cross-modal bridges, 2 trainable bridge params")
    print("  Bandwidth: analytic (π/2)/std, frozen — NOT trained")
    print("  Bounds: θ_bw∈[0.5, 2π] (analytic), θ_br∈[-0.5, 0.5] (SPSA+AMSGrad)")
    print("=" * 70)

    # ---- Load physics features ----
    phys16_file = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
    phys8_file  = os.path.join(config.PROCESSED_DIR, "physics_features_piqfm.npz")

    if os.path.exists(phys8_file):
        print("\nLoading PIQFM features...")
        data = np.load(phys8_file)
        X_all = data["X_all"]
        y_all = data["y_all"]
        X_all_raw = data["X_all_raw"]
    elif os.path.exists(phys16_file):
        print("\nExtracting 8 PIQFM features from 16-feature set (fitting norm on TRAIN)...")
        data = np.load(phys16_file)
        
        # Fit normalization on TRAIN ONLY to avoid data leakage
        X_train_raw_all = data["X_train_raw"][:, FEATURE_INDICES_16]
        lo = np.percentile(X_train_raw_all, 1, axis=0)
        hi = np.percentile(X_train_raw_all, 99, axis=0)

        # Process full dataset (concatenate as before but use train-fitted lo/hi)
        X16_raw_full = np.concatenate([data["X_train_raw"], data["X_test_raw"]], axis=0)
        y_all = np.concatenate([data["y_train"], data["y_test"]], axis=0)
        X_all_raw = X16_raw_full[:, FEATURE_INDICES_16]

        # Normalize using train-fitted lo/hi
        X_all = np.zeros_like(X_all_raw)
        for i in range(8):
            if abs(hi[i] - lo[i]) < 1e-8:
                X_all[:, i] = np.pi / 2
            else:
                X_all[:, i] = np.clip((X_all_raw[:, i] - lo[i]) / (hi[i] - lo[i]), 0, 1) * np.pi

        np.savez_compressed(phys8_file, X_all=X_all, X_all_raw=X_all_raw,
                           y_all=y_all, lo=lo, hi=hi,
                           feature_names=np.array(FEATURE_NAMES),
                           feature_indices=np.array(FEATURE_INDICES_16))
        print(f"  Saved: {phys8_file}")
    else:
        print(f"ERROR: Need {phys16_file}. Run test_physics_classical.py first.")
        sys.exit(1)

    print(f"  Total samples: {len(y_all)}, Features: {X_all.shape[1]}")
    print(f"  Feature range: [{X_all.min():.3f}, {X_all.max():.3f}]")

    # ---- Create stratified subsets ----
    # Training subset for KTA
    from sklearn.model_selection import train_test_split

    n_train_sub = min(args.train_subset, len(y_all))
    _, X_kta, _, y_kta = train_test_split(
        X_all, y_all, test_size=n_train_sub,
        stratify=y_all, random_state=config.RANDOM_SEED,
    )

    # Full kernel subset
    n_kernel = min(args.kernel_size, len(y_all))
    _, X_kernel, _, y_kernel = train_test_split(
        X_all, y_all, test_size=n_kernel,
        stratify=y_all, random_state=config.RANDOM_SEED,
    )

    # Ensure KTA subset is contained in kernel subset
    # (not strictly required, but cleaner)
    print(f"\n  KTA training: {len(y_kta)} samples ({len(np.unique(y_kta))} classes)")
    print(f"  Full kernel:  {len(y_kernel)} samples ({len(np.unique(y_kernel))} classes)")

    classes, counts = np.unique(y_kta, return_counts=True)
    print(f"  KTA class distribution: min={counts.min()}, max={counts.max()}")

    # ---- Paths ----
    output_dir = os.path.join(config.RESULTS_DIR, "piqfm")
    os.makedirs(output_dir, exist_ok=True)
    theta_path = os.path.join(output_dir, "trained_theta.npz")

    # ---- KTA Training ----
    # Compute analytic bandwidth from KTA anchor data (frozen, not trained)
    print("\n" + "=" * 70)
    print("PHASE 1a: ANALYTIC BANDWIDTH COMPUTATION")
    print("=" * 70)
    theta_bandwidth = compute_analytic_bandwidth(X_kta)
    print(f"  Analytic θ_bw = [{', '.join(f'{v:.3f}' for v in theta_bandwidth)}]")
    for i, (name, bw) in enumerate(zip(FEATURE_NAMES, theta_bandwidth)):
        std_i = np.std(X_kta[:, i])
        print(f"    q{i} ({name:10s}): std={std_i:.4f}  →  θ_bw={bw:.4f}")

    if args.skip_training and os.path.exists(theta_path):
        print(f"\nLoading trained bridge θ from {theta_path}")
        theta_data = np.load(theta_path)
        theta_bridge = theta_data["theta_bridge"]
        # Recompute bandwidth from current data (do not use stale saved bandwidth)
        print(f"  NOTE: Using freshly computed analytic bandwidth, not saved value.")
    else:
        print("\n" + "=" * 70)
        print("PHASE 1b: BRIDGE KTA TRAINING (SPSA+AMSGrad)")
        print("=" * 70)

        theta_bandwidth, theta_bridge, history = train_piqfm_kta(
            X_kta, y_kta,
            theta_bandwidth=theta_bandwidth,
            n_epochs=args.n_epochs,
            lr=args.lr,
            spsa_delta=args.delta,
        )

        # Save trained params
        np.savez(theta_path,
                theta_bandwidth=theta_bandwidth,
                theta_bridge=theta_bridge,
                history_epochs=np.array([h[0] for h in history]),
                history_kta=np.array([h[1] for h in history]))
        print(f"\n  Saved trained θ: {theta_path}")

    # ---- Report trained parameters ----
    print(f"\n{'='*70}")
    print("TRAINED PARAMETERS")
    print(f"{'='*70}")
    print("\n  Bandwidth scaling (θ_bw):")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"    q{i} ({name:10s}): θ = {theta_bandwidth[i]:.4f}")
    print("\n  Cross-modal bridge strength (θ_bridge):")
    for k, name in enumerate(BRIDGE_NAMES):
        strength = abs(theta_bridge[k])
        status = "STRONG" if strength > 0.5 else ("WEAK" if strength > 0.1 else "OFF")
        print(f"    {name:20s}: θ = {theta_bridge[k]:+.4f} [{status}]")

    # ---- Full Kernel Computation ----
    print(f"\n{'='*70}")
    print(f"PHASE 2: FULL KERNEL COMPUTATION ({n_kernel}×{n_kernel})")
    print(f"{'='*70}")

    # Split kernel data into train/test for SVM evaluation
    n_split = n_kernel // 2
    X_k_train = X_kernel[:n_split]
    X_k_test  = X_kernel[n_split:]
    y_k_train = y_kernel[:n_split]
    y_k_test  = y_kernel[n_split:]

    # Compute train kernel (symmetric)
    K_train_path = os.path.join(output_dir, "K_piqfm_train.npy")
    if os.path.exists(K_train_path):
        print(f"  Loading cached train kernel: {K_train_path}")
        K_train = np.load(K_train_path)
    else:
        print(f"\n  Computing train kernel ({n_split}×{n_split})...")
        t0 = time.time()
        K_train = compute_piqfm_kernel_matrix(
            X_k_train, theta_bandwidth, theta_bridge, desc="K_train")
        elapsed = time.time() - t0
        np.save(K_train_path, K_train)
        print(f"  Saved: {K_train_path} [{elapsed:.1f}s]")

    # Compute test kernel (rectangular)
    K_test_path = os.path.join(output_dir, "K_piqfm_test.npy")
    if os.path.exists(K_test_path):
        print(f"  Loading cached test kernel: {K_test_path}")
        K_test = np.load(K_test_path)
    else:
        print(f"\n  Computing test kernel ({len(X_k_test)}×{n_split})...")
        t0 = time.time()
        K_test = compute_piqfm_kernel_matrix(
            X_k_train, theta_bandwidth, theta_bridge,
            X2=X_k_test, desc="K_test")
        elapsed = time.time() - t0
        np.save(K_test_path, K_test)
        print(f"  Saved: {K_test_path} [{elapsed:.1f}s]")

    # Save labels
    np.savez(os.path.join(output_dir, "labels.npz"),
             y_train=y_k_train, y_test=y_k_test)

    # ---- Kernel Diagnostics ----
    print(f"\n{'='*70}")
    print("KERNEL DIAGNOSTICS")
    print(f"{'='*70}")

    diag = np.diag(K_train)
    off_diag = K_train[np.triu_indices_from(K_train, k=1)]
    print(f"  Diagonal: mean={diag.mean():.4f}, std={diag.std():.6f}")
    print(f"  Off-diag: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")
    print(f"  CV (off-diag): {off_diag.std() / (off_diag.mean() + 1e-12):.4f}")
    print(f"  Symmetric: {np.allclose(K_train, K_train.T)}")
    eigvals = np.linalg.eigvalsh(K_train)
    print(f"  PSD: {eigvals.min():.6f} (min eigenvalue)")

    # ---- Quick SVM Classification ----
    print(f"\n{'='*70}")
    print("QUICK CLASSIFICATION (PIQFM kernel vs RBF baseline)")
    print(f"{'='*70}")

    # PIQFM kernel SVM
    for C in [1, 10, 100, 1000]:
        svm = SVC(kernel='precomputed', C=C, class_weight='balanced',
                  random_state=config.RANDOM_SEED)
        svm.fit(K_train, y_k_train)
        y_pred = svm.predict(K_test)
        f1 = f1_score(y_k_test, y_pred, average='macro')
        print(f"  PIQFM SVM (C={C:5d}): F1 = {f1:.4f}")

    # RBF baseline on same data
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    X_k_train_raw = X_kernel[:n_split]  # already [0,π] normalized
    X_k_test_raw  = X_kernel[n_split:]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    rbf_svm = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced', random_state=config.RANDOM_SEED),
        param_grid={'C': [1, 10, 100, 1000, 10000], 'gamma': ['scale', 'auto']},
        scoring='f1_macro', cv=cv, n_jobs=-1, refit=True,
    )
    rbf_svm.fit(X_k_train_raw, y_k_train)
    y_pred_rbf = rbf_svm.predict(X_k_test_raw)
    f1_rbf = f1_score(y_k_test, y_pred_rbf, average='macro')
    print(f"\n  RBF SVM (GridSearch): F1 = {f1_rbf:.4f} "
          f"(C={rbf_svm.best_params_['C']}, γ={rbf_svm.best_params_['gamma']})")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
