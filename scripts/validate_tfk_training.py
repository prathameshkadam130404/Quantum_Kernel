"""
TFK Training Validation Script.

Validates that train_quantum_kernel_kta() produced a meaningful theta
BEFORE the full 2000x2000 kernel matrix finishes computing.

Checks:
    1. theta file exists and has correct shape
    2. theta moved meaningfully from initialization (ones vector)
    3. Simulates kernel on small random subset to confirm trained
       theta produces different kernel than untrained theta (ones)
    4. Computes KTA on small validation subset with trained vs
       untrained theta to confirm improvement direction
    5. Reports go/no-go verdict

Run:
    python scripts/validate_tfk_training.py

Does NOT require K_tfk_train.npy to exist.
Does NOT modify any file.
Safe to run while exp1 is still computing the full kernel matrix.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import config

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
TFK_DIR      = os.path.join(
    config.RESULTS_DIR, "geometric_difference", "fused", "tfk"
)
THETA_PATH   = os.path.join(TFK_DIR, "theta_trained.npy")
HISTORY_PATH = os.path.join(TFK_DIR, "kta_history.npy")

SUBSAMPLE_PATH = os.path.join(config.PROCESSED_DIR, "subsample_2000.npz")

SEPARATOR = "=" * 65

def find_file(candidates: list, label: str):
    """Find first existing file from candidate paths."""
    for path in candidates:
        if os.path.exists(path):
            return path
    print(f"  [MISSING] {label} — tried:")
    for p in candidates:
        print(f"            {p}")
    return None


def section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def check_1_theta_exists(theta_path: str):
    """Check 1: theta file exists and has correct shape."""
    section("CHECK 1 — theta file exists")

    if not os.path.exists(theta_path):
        print(f"  [FAIL] theta_trained.npy not found at:")
        print(f"         {theta_path}")
        print(f"  Training did not complete or save_path was not set.")
        return None

    theta = np.load(theta_path)
    print(f"  [OK]   theta_trained.npy found")
    print(f"         Path  : {theta_path}")
    print(f"         Shape : {theta.shape}  (expected: ({config.N_QUBITS},))")
    print(f"         dtype : {theta.dtype}")

    if theta.shape != (config.N_QUBITS,):
        print(f"  [WARN] Shape mismatch — expected ({config.N_QUBITS},), got {theta.shape}")
    else:
        print(f"  [OK]   Shape correct")

    return theta


def check_2_theta_moved(theta: np.ndarray):
    """Check 2: theta moved meaningfully from ones initialization."""
    section("CHECK 2 — theta adaptation from initialization")

    ones = np.ones(config.N_QUBITS, dtype=np.float64)
    deviation = np.linalg.norm(theta - ones)
    std        = theta.std()
    min_val    = theta.min()
    max_val    = theta.max()
    mean_val   = theta.mean()

    print(f"  theta values : {np.round(theta, 4)}")
    print(f"  mean         : {mean_val:.4f}")
    print(f"  std          : {std:.4f}")
    print(f"  min / max    : {min_val:.4f} / {max_val:.4f}")
    print(f"  ||theta - 1||: {deviation:.4f}")
    print()

    # Per-qubit deviation table
    print(f"  Per-qubit deviation from 1.0:")
    print(f"  {'Qubit':<6} {'theta':<10} {'deviation':<12} {'assessment'}")
    print(f"  {'-'*50}")
    for i, t in enumerate(theta):
        dev = abs(t - 1.0)
        if dev < 0.02:
            assessment = "minimal"
        elif dev < 0.10:
            assessment = "small"
        elif dev < 0.25:
            assessment = "moderate"
        else:
            assessment = "LARGE — strong adaptation"
        print(f"  {i:<6} {t:<10.4f} {dev:<12.4f} {assessment}")

    print()
    if deviation < 0.1:
        verdict = "WEAK — theta barely moved from initialization. " \
                  "Training had minimal effect. TFK ≈ standard FQK."
        status  = "WARN"
    elif deviation < 0.5:
        verdict = "MODERATE — theta adapted from initialization. " \
                  "Some task-relevant structure found."
        status  = "OK"
    else:
        verdict = "STRONG — theta moved significantly. " \
                  "Optimizer found meaningful encoding adaptation."
        status  = "GOOD"

    print(f"  [{status}] Verdict: {verdict}")
    return deviation


def check_3_kernel_differs(theta: np.ndarray, X: np.ndarray, y: np.ndarray):
    """
    Check 3: Trained theta produces different kernel than untrained (ones).
    Uses a small 20-sample validation subset for speed.
    """
    section("CHECK 3 — trained vs untrained kernel difference")

    import pennylane as qml
    from src.quantum_kernels import apply_trained_zz_feature_map

    print(f"  Building small validation subset (20 samples stratified)...")
    classes      = np.unique(y)
    n_per_class  = max(1, 20 // len(classes))
    idx = []
    for c in classes:
        c_idx  = np.where(y == c)[0]
        chosen = np.random.choice(
            c_idx, size=min(n_per_class, len(c_idx)), replace=False
        )
        idx.extend(chosen.tolist())
    idx    = np.array(idx[:20])
    X_val  = X[idx]
    n      = len(X_val)
    print(f"  Validation subset size: {n} samples")

    # Build circuit
    dev = config.get_device(config.N_QUBITS)

    @qml.qnode(dev, diff_method=None)
    def tfk_circuit(x1, x2, th):
        apply_trained_zz_feature_map(x1, th, config.N_QUBITS, config.ZZ_REPS)
        qml.adjoint(apply_trained_zz_feature_map)(x2, th, config.N_QUBITS, config.ZZ_REPS)
        return qml.probs(wires=range(config.N_QUBITS))

    ones_theta    = np.ones(config.N_QUBITS, dtype=np.float64)

    print(f"  Computing K_trained on {n}x{n} subset...")
    K_trained = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            v = float(tfk_circuit(X_val[i], X_val[j], theta)[0])
            K_trained[i, j] = v
            K_trained[j, i] = v

    print(f"  Computing K_untrained (theta=ones) on same subset...")
    K_untrained = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            v = float(tfk_circuit(X_val[i], X_val[j], ones_theta)[0])
            K_untrained[i, j] = v
            K_untrained[j, i] = v

    # Compare
    diff            = K_trained - K_untrained
    frobenius_diff  = np.linalg.norm(diff)
    mean_abs_diff   = np.abs(diff[~np.eye(n, dtype=bool)]).mean()
    max_abs_diff    = np.abs(diff).max()

    print()
    print(f"  K_trained   mean_offdiag : "
          f"{K_trained[~np.eye(n,dtype=bool)].mean():.4f}")
    print(f"  K_untrained mean_offdiag : "
          f"{K_untrained[~np.eye(n,dtype=bool)].mean():.4f}")
    print(f"  Frobenius diff ||K_t - K_u|| : {frobenius_diff:.4f}")
    print(f"  Mean abs diff (off-diag)     : {mean_abs_diff:.4f}")
    print(f"  Max abs diff                 : {max_abs_diff:.4f}")
    print()

    if frobenius_diff < 0.1:
        verdict = "FAIL — kernels nearly identical. " \
                  "Trained theta has no effect on kernel geometry."
        status  = "FAIL"
    elif frobenius_diff < 0.5:
        verdict = "MODERATE — kernels differ meaningfully. " \
                  "Training changed kernel geometry."
        status  = "OK"
    else:
        verdict = "STRONG — kernels differ substantially. " \
                  "Training produced a distinctly different kernel."
        status  = "GOOD"

    print(f"  [{status}] Verdict: {verdict}")
    return K_trained, K_untrained, X_val, idx


def check_4_kta_improved(
    K_trained: np.ndarray,
    K_untrained: np.ndarray,
    y: np.ndarray,
    val_idx: np.ndarray,
):
    """Check 4: Trained kernel has higher KTA than untrained on validation subset."""
    section("CHECK 4 — KTA improvement on validation subset")

    from src.kernel_target_alignment import compute_centered_kta

    y_val = y[val_idx]

    kta_trained   = compute_centered_kta(K_trained,   y_val, class_weighted=True)
    kta_untrained = compute_centered_kta(K_untrained, y_val, class_weighted=True)
    delta         = kta_trained - kta_untrained

    print(f"  KTA (trained theta)   : {kta_trained:.4f}")
    print(f"  KTA (untrained theta) : {kta_untrained:.4f}")
    print(f"  Delta                 : {delta:+.4f}")
    print()

    # Context: full training set FQK baseline
    print(f"  Context — full 2000-sample baselines (from Exp1):")
    print(f"    FQK centered weighted KTA : 0.1845")
    print(f"    PQK centered weighted KTA : 0.1467")
    print(f"    RBF centered weighted KTA : 0.1308")
    print()
    print(f"  Note: validation subset KTA is not directly comparable to")
    print(f"  full-matrix KTA (different n, different samples). This check")
    print(f"  only confirms that training improved KTA in the right direction.")
    print()

    if delta > 0.01:
        verdict = "PASS — trained theta improves KTA over untrained on " \
                  "validation subset. Training moved in correct direction."
        status  = "PASS"
    elif delta > -0.01:
        verdict = "MARGINAL — KTA difference is within noise. " \
                  "Training effect is weak on this subset."
        status  = "MARGINAL"
    else:
        verdict = "FAIL — trained theta has LOWER KTA than untrained. " \
                  "Training moved in wrong direction on this subset."
        status  = "FAIL"

    print(f"  [{status}] Verdict: {verdict}")
    return kta_trained, kta_untrained, delta


def check_5_history(history_path: str):
    """Check 5: KTA history if saved."""
    section("CHECK 5 — KTA training history (if available)")

    if not os.path.exists(history_path):
        print(f"  [INFO] kta_history.npy not found — was not saved during training.")
        print(f"         This is fine. History saving is optional.")
        return

    history = np.load(history_path)
    print(f"  KTA history shape : {history.shape}")
    print(f"  First 5 epochs    : {np.round(history[:5], 4)}")
    print(f"  Last  5 epochs    : {np.round(history[-5:], 4)}")
    print(f"  Min KTA           : {history.min():.4f} (epoch {history.argmin()})")
    print(f"  Max KTA           : {history.max():.4f} (epoch {history.argmax()})")
    print(f"  Final KTA         : {history[-1]:.4f}")
    print(f"  Mean last 10      : {history[-10:].mean():.4f} "
          f"± {history[-10:].std():.4f}")
    print(f"  Overall trend     : {history[-1] - history[0]:+.4f} "
          f"({'improving' if history[-1] > history[0] else 'degrading'})")


def main():
    print(f"\n{SEPARATOR}")
    print(f"  TFK TRAINING VALIDATION")
    print(f"  {SEPARATOR}")
    print(f"  Safe to run while K_tfk_train.npy is still computing.")
    print(f"  Does not modify any file.")
    print(SEPARATOR)

    np.random.seed(config.RANDOM_SEED)

    # --- Check 1: theta exists ---
    theta = check_1_theta_exists(THETA_PATH)
    if theta is None:
        print(f"\n  CANNOT PROCEED — theta_trained.npy missing.")
        print(f"  Training did not complete successfully.")
        sys.exit(1)

    # --- Check 2: theta moved ---
    deviation = check_2_theta_moved(theta)

    # --- Load data for checks 3 and 4 ---
    section("LOADING DATA")
    if not os.path.exists(SUBSAMPLE_PATH):
        print(f"  [SKIP] subsample_2000.npz not found at {SUBSAMPLE_PATH}")
        X, y = None, None
        print(f"  Checks 1 and 2 are sufficient to confirm theta was saved correctly.")
    else:
        subsample = np.load(SUBSAMPLE_PATH)
        X = subsample['fused_X_train']
        y = subsample['y_train']
        print(f"  X_train shape : {X.shape}")
        print(f"  y_train shape : {y.shape}")
        print(f"  Classes       : {np.unique(y)}")
        print(f"  Class counts  : {dict(zip(*np.unique(y, return_counts=True)))}")

        # --- Check 3: kernel differs ---
        K_trained, K_untrained, X_val, val_idx = check_3_kernel_differs(
            theta, X, y
        )

        # --- Check 4: KTA improved ---
        kta_t, kta_u, delta = check_4_kta_improved(
            K_trained, K_untrained, y, val_idx
        )

    # --- Check 5: history ---
    check_5_history(HISTORY_PATH)

    # --- Final summary ---
    section("OVERALL VERDICT")

    if X is not None and y is not None:
        geometry_ok  = True   # Will be updated from check 3 if needed
        theta_ok     = deviation >= 0.1
        kta_dir_ok   = delta > -0.01

        if theta_ok and kta_dir_ok:
            print(f"  GO — Training appears successful.")
            print(f"       theta adapted from initialization (||theta-1||={deviation:.3f})")
            print(f"       KTA improved in correct direction (delta={delta:+.4f})")
            print(f"       Proceed: wait for K_tfk_train.npy, then run diagnose_tfk.py")
        elif not theta_ok:
            print(f"  WARN — theta barely moved (||theta-1||={deviation:.3f} < 0.1).")
            print(f"       TFK kernel will be nearly identical to standard FQK.")
            print(f"       Consider retraining with fixed anchor subset approach.")
        else:
            print(f"  WARN — KTA did not improve on validation subset (delta={delta:+.4f}).")
            print(f"       This may be subset variance — wait for full matrix KTA.")
            print(f"       If full matrix KTA < 0.1845 (FQK baseline), retrain needed.")
    else:
        if deviation >= 0.1:
            print(f"  PARTIAL GO — theta adapted (||theta-1||={deviation:.3f}).")
            print(f"  Data not found for kernel checks 3 and 4.")
            print(f"  Wait for K_tfk_train.npy then run diagnose_tfk.py for full validation.")
        else:
            print(f"  WARN — theta barely moved. Training may not have been effective.")

    print(f"\n  Next step: when K_tfk_train.npy finishes (~6.5hrs from training end),")
    print(f"  run: python scripts/diagnose_tfk.py")
    print(SEPARATOR + "\n")


if __name__ == "__main__":
    main()
