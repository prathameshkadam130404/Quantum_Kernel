"""
TFK Learning Rate Finder.

Runs a short training loop (15 epochs) for each candidate learning rate
and reports which lr produces the best combination of:
    - KTA improvement over untrained baseline
    - Bounded theta (||theta-1|| in target range)
    - Healthy kernel geometry (mean_offdiag > 0.10)

Based on Smith (2017) LR Range Test — IEEE WACV 2017.

Usage:
    python scripts/tfk_lr_finder.py

Runtime: approximately 10-15 minutes total.
Does NOT write any results files.
Does NOT modify any existing file.
Safe to run before starting exp1.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import time
import config

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
N_EPOCHS_TEST   = 15      # Short run per lr — enough to see divergence/stagnation
N_PAIR_SAMPLES  = 50      # Fewer pairs for speed — relative comparison only
SUBSET_SIZE     = 68      # Same as TFK_SUBSET_SIZE — realistic gradient noise
RANDOM_SEED     = config.RANDOM_SEED

# Candidate learning rates — log-spaced between the two failures
LR_CANDIDATES = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05]

# Target ranges for healthy training
TARGET_THETA_DEVIATION_MIN = 0.05   # ||theta-1|| after 15 epochs — minimum meaningful
TARGET_THETA_DEVIATION_MAX = 2.00   # ||theta-1|| after 15 epochs — maximum before divergence
TARGET_MEAN_OFFDIAG_MIN    = 0.10   # healthy kernel geometry threshold
TARGET_KTA_MIN             = 0.20   # must beat untrained baseline

# Data paths — try multiple candidates
DATA_CANDIDATES = [
    os.path.join(config.RESULTS_DIR, "geometric_difference", "fused", "X_train.npy"),
    os.path.join(config.RESULTS_DIR, "X_train_fused.npy"),
    os.path.join(config.PROCESSED_DIR, "X_train_fused.npy"),
]
LABEL_CANDIDATES = [
    os.path.join(config.RESULTS_DIR, "geometric_difference", "fused", "y_train.npy"),
    os.path.join(config.RESULTS_DIR, "y_train_fused.npy"),
    os.path.join(config.PROCESSED_DIR, "y_train.npy"),
]

SEPARATOR = "=" * 70


def find_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def build_stratified_subset(X, y, subset_size, seed):
    """
    Build ONE fixed stratified subset used for ALL lr tests.
    Fixed subset ensures fair comparison between lr values.
    """
    np.random.seed(seed)
    classes     = np.unique(y)
    n_per_class = subset_size // len(classes)
    idx = []
    for c in classes:
        c_idx  = np.where(y == c)[0]
        n_avail = len(c_idx)
        chosen  = np.random.choice(
            c_idx,
            size=min(n_per_class, n_avail),
            replace=(n_avail < n_per_class)
        )
        idx.extend(chosen.tolist())
    idx = np.array(idx[:subset_size])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def compute_kernel_matrix(X_sub, theta, tfk_circuit, n_qubits):
    """Full kernel matrix on subset at given theta."""
    n  = len(X_sub)
    K  = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            v        = float(tfk_circuit(X_sub[i], X_sub[j], theta)[0])
            K[i, j]  = v
            K[j, i]  = v
    return K


def compute_kernel_entry(x1, x2, theta, tfk_circuit):
    return float(tfk_circuit(x1, x2, theta)[0])


def build_ideal_kernel(y_sub):
    n            = len(y_sub)
    y_ideal      = np.zeros((n, n), dtype=np.float64)
    classes, counts = np.unique(y_sub, return_counts=True)
    count_dict   = dict(zip(classes.tolist(), counts.tolist()))
    for i in range(n):
        for j in range(n):
            if y_sub[i] == y_sub[j]:
                y_ideal[i, j] = 1.0 / count_dict[y_sub[i]]
    return y_ideal


def centered_kta(K, y_ideal):
    n   = len(K)
    H   = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    y_c = H @ y_ideal @ H
    num = np.sum(K_c * y_c)
    den = np.sqrt(np.sum(K_c ** 2) * np.sum(y_c ** 2) + 1e-15)
    return float(num / den)


def kta_weight_matrix(K, y_ideal):
    n      = len(K)
    H      = np.eye(n) - np.ones((n, n)) / n
    K_c    = H @ K @ H
    y_c    = H @ y_ideal @ H
    norm_K = np.sqrt(np.sum(K_c ** 2) + 1e-15)
    norm_y = np.sqrt(np.sum(y_c ** 2) + 1e-15)
    kta    = np.sum(K_c * y_c) / (norm_K * norm_y)
    grad_Kc = (y_c / (norm_K * norm_y)) - (kta * K_c / (norm_K ** 2))
    W       = H @ grad_Kc @ H
    sym     = 2.0 * np.ones((n, n))
    np.fill_diagonal(sym, 1.0)
    return W * sym


def stochastic_kta_gradient(X_sub, y_sub, theta, tfk_circuit, n_qubits,
                             n_pair_samples, rng):
    """Stochastic parameter-shift KTA gradient."""
    shift    = np.pi / 2.0
    n        = len(X_sub)

    K        = compute_kernel_matrix(X_sub, theta, tfk_circuit, n_qubits)
    y_ideal  = build_ideal_kernel(y_sub)
    kta_val  = centered_kta(K, y_ideal)
    W        = kta_weight_matrix(K, y_ideal)

    all_pairs   = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_available = len(all_pairs)
    n_sample    = min(n_pair_samples, n_available)
    sampled_idx = rng.choice(n_available, size=n_sample, replace=False)
    sampled     = [all_pairs[k] for k in sampled_idx]
    scale       = float(n_available) / float(n_sample)

    grad = np.zeros(n_qubits, dtype=np.float64)
    for k in range(n_qubits):
        tp  = theta.copy(); tp[k]  += shift
        tm  = theta.copy(); tm[k]  -= shift
        acc = 0.0
        for (i, j) in sampled:
            kp   = compute_kernel_entry(X_sub[i], X_sub[j], tp, tfk_circuit)
            km   = compute_kernel_entry(X_sub[i], X_sub[j], tm, tfk_circuit)
            acc += W[i, j] * (kp - km) / 2.0 * 2.0
        grad[k] = 0.5 * scale * acc

    return grad, kta_val, K


def run_lr_test(lr, X_sub, y_sub, tfk_circuit, n_qubits,
                n_epochs, n_pair_samples, seed):
    """
    Run short training with given lr.
    Returns dict of diagnostic metrics.
    """
    rng    = np.random.default_rng(seed)
    theta  = np.ones(n_qubits, dtype=np.float64)
    m      = np.zeros(n_qubits, dtype=np.float64)
    v      = np.zeros(n_qubits, dtype=np.float64)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    kta_history       = []
    theta_norm_history = []
    clipped_count     = 0

    for epoch in range(n_epochs):
        grad, kta_val, K = stochastic_kta_gradient(
            X_sub, y_sub, theta, tfk_circuit,
            n_qubits, n_pair_samples, rng
        )

        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm
            clipped_count += 1

        neg_grad  = -grad
        t         = epoch + 1
        m         = beta1 * m + (1 - beta1) * neg_grad
        v         = beta2 * v + (1 - beta2) * neg_grad ** 2
        m_hat     = m / (1 - beta1 ** t)
        v_hat     = v / (1 - beta2 ** t)
        theta     = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        theta     = np.clip(theta, -2.0 * np.pi, 2.0 * np.pi)

        kta_history.append(kta_val)
        theta_norm_history.append(np.linalg.norm(theta - 1.0))

    # Final metrics
    offdiag      = K[~np.eye(len(K), dtype=bool)]
    mean_offdiag = offdiag.mean()
    theta_dev    = np.linalg.norm(theta - 1.0)
    final_kta    = kta_history[-1]
    kta_trend    = kta_history[-1] - kta_history[0]

    return {
        "lr"           : lr,
        "final_kta"    : final_kta,
        "kta_trend"    : kta_trend,
        "theta_dev"    : theta_dev,
        "theta_min"    : theta.min(),
        "theta_max"    : theta.max(),
        "mean_offdiag" : mean_offdiag,
        "clipped_count": clipped_count,
        "theta_final"  : theta.copy(),
        "kta_history"  : kta_history,
        "theta_history": theta_norm_history,
    }


def score_lr(result):
    """
    Compute a single score for lr selection.

    A good lr must satisfy ALL of:
        1. theta_dev in [TARGET_MIN, TARGET_MAX] — moved but not diverged
        2. mean_offdiag > TARGET_MEAN_OFFDIAG_MIN — kernel not collapsed
        3. kta_trend > 0 — KTA improved over 15 epochs
        4. final_kta > TARGET_KTA_MIN — beats untrained baseline

    Score = kta_trend × mean_offdiag_bonus × theta_dev_bonus
    Penalties applied for violations.
    """
    r = result

    # Hard disqualifications
    if r["theta_dev"] > TARGET_THETA_DEVIATION_MAX:
        return -999.0, "DISQUALIFIED — theta diverged"
    if r["theta_dev"] < TARGET_THETA_DEVIATION_MIN:
        return -998.0, "DISQUALIFIED — theta did not move"
    if r["mean_offdiag"] < TARGET_MEAN_OFFDIAG_MIN:
        return -997.0, "DISQUALIFIED — kernel collapsed"
    if r["kta_trend"] <= 0:
        return -996.0, "DISQUALIFIED — KTA did not improve"

    # Score: reward KTA improvement and healthy geometry
    score = r["kta_trend"] * 10.0 + r["mean_offdiag"] * 5.0

    # Bonus for theta in sweet spot [0.1, 1.0]
    if 0.1 <= r["theta_dev"] <= 1.0:
        score += 2.0

    status = "CANDIDATE"
    return score, status


def main():
    print(f"\n{SEPARATOR}")
    print(f"  TFK LEARNING RATE FINDER")
    print(f"  Based on Smith (2017) LR Range Test — IEEE WACV 2017")
    print(SEPARATOR)
    print(f"  Candidates : {LR_CANDIDATES}")
    print(f"  Epochs/run : {N_EPOCHS_TEST}")
    print(f"  Subset size: {SUBSET_SIZE}")
    print(f"  Pair samples: {N_PAIR_SAMPLES}")
    print(f"  Estimated runtime: ~10-15 minutes total")
    print(SEPARATOR)

    # --- Load data ---
    print("\nLoading data...")
    try:
        from src.data_loader import load_subsample
        subsample = load_subsample()
        X = subsample["fused_X_train"]
        y = subsample["y_train"]
    except Exception as e:
        print(f"  [FAIL] Could not load data using src.data_loader: {e}")
        sys.exit(1)

    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    print(f"  Classes: {np.unique(y).tolist()}")

    # --- Build FIXED subset (same for all lr values) ---
    print(f"\nBuilding fixed stratified subset (size={SUBSET_SIZE})...")
    X_sub, y_sub = build_stratified_subset(X, y, SUBSET_SIZE, RANDOM_SEED)
    classes_in_sub, counts_in_sub = np.unique(y_sub, return_counts=True)
    print(f"  Subset classes: {len(classes_in_sub)}/17 represented")
    print(f"  Counts: min={counts_in_sub.min()}, max={counts_in_sub.max()}")

    # --- Build circuit (once, shared across all lr tests) ---
    print("\nBuilding TFK circuit...")
    import pennylane as qml
    from src.quantum_kernels import apply_trained_zz_feature_map

    n_qubits = config.N_QUBITS
    dev      = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def tfk_circuit(x1, x2, theta):
        apply_trained_zz_feature_map(x1, theta, n_qubits, config.ZZ_REPS)
        qml.adjoint(apply_trained_zz_feature_map)(x2, theta, n_qubits, config.ZZ_REPS)
        return qml.probs(wires=range(n_qubits))

    print(f"  Circuit built: {n_qubits} qubits, {config.ZZ_REPS} reps")

    # --- Compute untrained baseline KTA ---
    print("\nComputing untrained baseline KTA (theta=ones)...")
    ones_theta   = np.ones(n_qubits, dtype=np.float64)
    K_base       = compute_kernel_matrix(X_sub, ones_theta, tfk_circuit, n_qubits)
    y_ideal_base = build_ideal_kernel(y_sub)
    kta_baseline = centered_kta(K_base, y_ideal_base)
    offdiag_base = K_base[~np.eye(len(K_base), dtype=bool)]
    print(f"  Untrained KTA     : {kta_baseline:.4f}")
    print(f"  Untrained offdiag : {offdiag_base.mean():.4f}")

    # --- Run LR tests ---
    results = []
    print(f"\n{SEPARATOR}")
    print(f"  RUNNING LR RANGE TEST")
    print(SEPARATOR)

    for lr in LR_CANDIDATES:
        print(f"\n  lr={lr:.4f} — running {N_EPOCHS_TEST} epochs...")
        t0     = time.time()
        result = run_lr_test(
            lr, X_sub, y_sub, tfk_circuit, n_qubits,
            N_EPOCHS_TEST, N_PAIR_SAMPLES, RANDOM_SEED
        )
        elapsed = time.time() - t0

        score, status = score_lr(result)

        print(f"    KTA: {result['kta_history'][0]:.4f} → {result['final_kta']:.4f} "
              f"(trend: {result['kta_trend']:+.4f})")
        print(f"    theta_dev : {result['theta_dev']:.4f}  "
              f"[{result['theta_min']:.3f} .. {result['theta_max']:.3f}]")
        print(f"    offdiag   : {result['mean_offdiag']:.4f}")
        print(f"    clipped   : {result['clipped_count']}/{N_EPOCHS_TEST} epochs")
        print(f"    score     : {score:.3f}  [{status}]")
        print(f"    time      : {elapsed:.1f}s")

        result["score"]  = score
        result["status"] = status
        results.append(result)

    # --- Summary table ---
    print(f"\n{SEPARATOR}")
    print(f"  RESULTS SUMMARY")
    print(SEPARATOR)
    print(f"  Untrained baseline KTA : {kta_baseline:.4f}")
    print(f"  Untrained offdiag      : {offdiag_base.mean():.4f}")
    print()
    print(f"  {'lr':<8} {'KTA_final':<12} {'KTA_trend':<12} "
          f"{'theta_dev':<12} {'offdiag':<10} {'score':<10} status")
    print(f"  {'-'*80}")

    for r in results:
        print(f"  {r['lr']:<8.4f} {r['final_kta']:<12.4f} "
              f"{r['kta_trend']:<+12.4f} {r['theta_dev']:<12.4f} "
              f"{r['mean_offdiag']:<10.4f} {r['score']:<10.3f} {r['status']}")

    # --- Recommendation ---
    print(f"\n{SEPARATOR}")
    print(f"  RECOMMENDATION")
    print(SEPARATOR)

    candidates = [(r["score"], r) for r in results if r["score"] > 0]

    if not candidates:
        print("  NO VALID LR FOUND in the tested range.")
        print("  All candidates either diverged or failed to move theta.")
        print()
        print("  Diagnosis:")
        for r in results:
            print(f"    lr={r['lr']:.4f}: {r['status']}")
        print()
        print("  Suggested action: expand search range or increase N_EPOCHS_TEST")
    else:
        candidates.sort(reverse=True)
        best = candidates[0][1]
        print(f"  RECOMMENDED LR: {best['lr']}")
        print()
        print(f"  Reasoning:")
        print(f"    - KTA improved by {best['kta_trend']:+.4f} over {N_EPOCHS_TEST} epochs")
        print(f"    - theta_dev = {best['theta_dev']:.4f} (target: 0.05–2.00)")
        print(f"    - mean_offdiag = {best['mean_offdiag']:.4f} (target: > 0.10)")
        print(f"    - clipped {best['clipped_count']}/{N_EPOCHS_TEST} epochs (gradient control working)")
        print()

        # Extrapolate to 75 epochs
        rate_per_epoch  = best["theta_dev"] / N_EPOCHS_TEST
        projected_dev   = rate_per_epoch * 75
        print(f"  Projected ||theta-1|| after 75 epochs: ~{projected_dev:.2f}")
        if projected_dev < 3.0:
            print(f"  Projection is SAFE — theta should remain bounded at lr={best['lr']}")
        elif projected_dev < 8.0:
            print(f"  Projection is BORDERLINE — monitor theta during training")
            print(f"  Consider lr={best['lr'] * 0.7:.4f} as a safer alternative")
        else:
            print(f"  WARNING — projected theta_dev={projected_dev:.1f} may diverge over 75 epochs")
            print(f"  Recommend using lr={best['lr'] * 0.5:.4f} instead")

        print()
        print(f"  To apply: set TFK_LR = {best['lr']} in config.py")
        print(f"  Then delete tfk/ cache files and run exp1.")

    # --- KTA trajectory for top candidates ---
    print(f"\n{SEPARATOR}")
    print(f"  KTA TRAJECTORIES (top candidates)")
    print(SEPARATOR)
    top_candidates = sorted(
        [r for r in results if r["score"] > 0],
        key=lambda r: r["score"],
        reverse=True
    )[:4]

    if top_candidates:
        # Print every 3 epochs
        checkpoints = [0, 2, 4, 7, 9, 12, 14]
        header = f"  {'epoch':<8}" + "".join(
            f"lr={r['lr']:<9}" for r in top_candidates
        )
        print(header)
        print(f"  {'-'*60}")
        for ep in checkpoints:
            if ep < N_EPOCHS_TEST:
                row = f"  {ep:<8}"
                for r in top_candidates:
                    row += f"{r['kta_history'][ep]:<14.4f}"
                print(row)
    else:
        print("  No valid candidates to display.")

    print(f"\n{SEPARATOR}\n")


if __name__ == "__main__":
    main()
