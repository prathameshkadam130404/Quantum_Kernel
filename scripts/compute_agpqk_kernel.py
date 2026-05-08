"""
compute_agpqk_kernel.py — Attention-Guided PQK (AGPQK) kernel computation.

Novel kernel design for SAR-optical multimodal quantum classification.

Pipeline:
  Stage 1: Load all 16 physics features (training data)
  Stage 2: Compute 16×16 attention matrix from feature co-variance
  Stage 3: Select top-8 features, compute per-qubit bandwidth, select
           top-4 cross-modal entanglement pairs
  Stage 4: Build custom PQK circuit with attention-guided structure:
             - Per-qubit bandwidth: RZ(γ_i × x_i) instead of RZ(x_i)
             - Attention-selected ZZ pairs instead of adjacent default
             - Same Bloch measurement: ⟨X_i⟩,⟨Y_i⟩,⟨Z_i⟩ per qubit
  Stage 5: Compute outer RBF kernel on Bloch vectors → K_agpqk
  Stage 6: Compute KTA and compare to all existing physics kernels

Outputs:
  results/physics/fused/K_agpqk_physics_train.npy
  results/physics/fused/K_agpqk_physics_test.npy
  results/physics/fused/bloch_agpqk_physics_train.npy
  results/physics/agpqk_config.json   (stores attention decisions)

Usage:
    python scripts/compute_agpqk_kernel.py

Runtime: ~20-40 minutes (same circuit depth as PQK, same n=2000)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
from tqdm import tqdm

import pennylane as qml

import config
from src.attention_kernel import (
    compute_attention_matrix,
    select_features_by_fisher,
    compute_per_qubit_gamma,
    get_attention_entanglement_pairs,
    print_attention_summary,
    FEATURE_NAMES_16,
    FEATURE_MODALITY_16,
)
from src.kernel_target_alignment import compute_centered_kta, compute_full_kta_analysis
from src.bandwidth import apply_bandwidth

# ── Config ────────────────────────────────────────────────────────────────
N_QUBITS         = config.N_QUBITS       # 8
ZZ_REPS          = config.ZZ_REPS        # 2
OUTER_GAMMA      = config.PQK_GAMMA      # 0.67 — same as standard PQK
GAMMA_BASE       = 0.50                  # base per-qubit bandwidth (CV-selected)
N_SELECT         = 8                     # features to select from 16
TOP_K_PAIRS      = 4                     # attention-selected entanglement pairs
REQUIRE_CROSS_MODAL = True               # enforce SAR-optical pairs only

PHYSICS_DIR  = os.path.join(config.RESULTS_DIR, "physics", "fused")
AGPQK_DIR    = PHYSICS_DIR               # save alongside other physics kernels
PHYS16_PATH  = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")

os.makedirs(AGPQK_DIR, exist_ok=True)

# At the top, load both raw and normalized
data     = np.load(PHYS16_PATH)
X16_norm = data["X_train"]      # (2000, 16), in [0, π] — for encoding only
X16_raw  = data["X_train_raw"]  # (2000, 16), physical units — for analysis
X16_test_norm = data["X_test"]
X16_test_raw  = data["X_test_raw"]
y_train  = data["y_train"]
y_test   = data["y_test"]

# Known baselines for comparison
KN_KTA = {
    "RBF":    0.2956,
    "FQK":    0.2345,
    "PQK":    0.2329,
    "TFK":    0.1947,
}


# ── Stage 1: Load data ────────────────────────────────────────────────────
print("=" * 65)
print("  AGPQK — Attention-Guided Projected Quantum Kernel")
print("=" * 65)

if not os.path.exists(PHYS16_PATH):
    print(f"ERROR: {PHYS16_PATH} not found")
    sys.exit(1)

n_train  = len(y_train)
n_test   = len(y_test)

print(f"\n  Loaded: {n_train} train, {n_test} test, 16 features each")
print(f"  Norm range:  [{X16_norm.min():.4f}, {X16_norm.max():.4f}]")
print(f"  Raw  range:  [{X16_raw.min():.4f}, {X16_raw.max():.4f}]")


# ── Stage 2: Attention matrix ─────────────────────────────────────────────
print("\n--- Stage 2: Computing Attention Matrix ---")
A = compute_attention_matrix(X16_raw)
print(f"  Attention matrix shape: {A.shape}")
print(f"  Row-stochastic check (row sums): min={A.sum(1).min():.6f}, max={A.sum(1).max():.6f}")


# ── Stage 3: Feature selection, bandwidth, entanglement ───────────────────
print("\n--- Stage 3: Attention-Guided Circuit Design ---")

from src.attention_kernel import select_features_by_fisher, compute_fisher_ratio
selected_indices, fisher = select_features_by_fisher(
    X16_raw, y_train, n_select=N_SELECT, min_sar=4, min_opt=4
)
gamma_per_qubit = compute_per_qubit_gamma(fisher, selected_indices, gamma_base=GAMMA_BASE)
entanglement_pairs = get_attention_entanglement_pairs(
    A, selected_indices, top_k=TOP_K_PAIRS,
    require_cross_modal=REQUIRE_CROSS_MODAL,
    max_appearances=1,
)

print_attention_summary(A, selected_indices, fisher, gamma_per_qubit, entanglement_pairs)
# Extract selected features and apply per-qubit bandwidth
# X16 is already in [0, π]. Per-qubit bandwidth scales each feature differently.
X8_for_encode      = X16_norm[:, selected_indices]   # in [0, π]
X8_test_for_encode = X16_test_norm[:, selected_indices]

# Apply per-qubit gamma to the already-normalized features
X8_enc      = X8_for_encode      * gamma_per_qubit[np.newaxis, :]
X8_test_enc = X8_test_for_encode * gamma_per_qubit[np.newaxis, :]

print(f"  Encoded feature range after per-qubit γ scaling:")
for q in range(N_SELECT):
    feat_idx = selected_indices[q]
    print(f"    qubit {q} ({FEATURE_NAMES_16[feat_idx]:<12}): "
          f"γ={gamma_per_qubit[q]:.4f}  "
          f"range=[{X8_enc[:,q].min():.3f}, {X8_enc[:,q].max():.3f}]")

# Save attention config for paper reporting
agpqk_config = {
    "selected_feature_indices": selected_indices.tolist(),
    "selected_feature_names": [FEATURE_NAMES_16[i] for i in selected_indices],
    "selected_feature_modalities": [FEATURE_MODALITY_16[i] for i in selected_indices],
    "importance_scores": fisher.tolist(),
    "gamma_base": GAMMA_BASE,
    "gamma_per_qubit": gamma_per_qubit.tolist(),
    "entanglement_pairs_qubit_space": entanglement_pairs,
    "entanglement_pairs_feature_names": [
        (FEATURE_NAMES_16[selected_indices[i]], FEATURE_NAMES_16[selected_indices[j]])
        for i, j in entanglement_pairs
    ],
    "outer_gamma": OUTER_GAMMA,
    "n_qubits": N_QUBITS,
    "zz_reps": ZZ_REPS,
    "top_k_pairs": TOP_K_PAIRS,
    "require_cross_modal": REQUIRE_CROSS_MODAL,
}
config_path = os.path.join(config.RESULTS_DIR, "physics", "agpqk_config.json")
os.makedirs(os.path.dirname(config_path), exist_ok=True)
with open(config_path, "w") as f:
    json.dump(agpqk_config, f, indent=2)
print(f"\n  Config saved: {config_path}")


# ── Stage 4: AGPQK circuit ────────────────────────────────────────────────
print("\n--- Stage 4: Building AGPQK Circuit ---")


def apply_agpqk_encoding(x, gamma_qubit, entangle_pairs, n_qubits=N_QUBITS, reps=ZZ_REPS):
    """
    Apply attention-guided ZZFeatureMap encoding.

    Differences from standard ZZFeatureMap:
      - Per-qubit bandwidth: features already pre-scaled by gamma_per_qubit,
        so x[i] is already gamma_i * raw_feature_i
      - Entanglement: uses attention-selected pairs instead of adjacent default

    Note: x is already bandwidth-scaled (X8_enc), so this circuit
    encodes RZ(x[i]) where x[i] = gamma_i * feature_i.
    The ZZ interaction angle is (π - x[i])(π - x[j]) for each pair.

    Args:
        x:               Encoded feature vector, shape (n_qubits,)
        gamma_qubit:     Per-qubit bandwidth array (for reference/logging only)
        entangle_pairs:  List of (i, j) qubit pairs for ZZ entanglement
        n_qubits:        Number of qubits
        reps:            Circuit repetitions
    """
    for _ in range(reps):
        # Hadamard layer
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        # Single-qubit RZ encoding (x already scaled by gamma_i)
        for i in range(n_qubits):
            qml.RZ(x[i], wires=i)

        # Attention-guided ZZ entanglement (replaces adjacent default)
        for (qi, qj) in entangle_pairs:
            zz_angle = (np.pi - x[qi]) * (np.pi - x[qj])
            qml.CNOT(wires=[qi, qj])
            qml.RZ(zz_angle, wires=qj)
            qml.CNOT(wires=[qi, qj])


def build_agpqk_circuit(gamma_qubit, entangle_pairs, n_qubits=N_QUBITS, reps=ZZ_REPS):
    """
    Build three QNodes for measuring ⟨X_i⟩, ⟨Y_i⟩, ⟨Z_i⟩ per qubit.

    Returns: (measure_x, measure_y, measure_z) QNode triple
    """
    dev = config.get_device(n_qubits)

    @qml.qnode(dev, diff_method=None)
    def measure_x(x):
        apply_agpqk_encoding(x, gamma_qubit, entangle_pairs, n_qubits, reps)
        return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def measure_y(x):
        apply_agpqk_encoding(x, gamma_qubit, entangle_pairs, n_qubits, reps)
        return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]

    @qml.qnode(dev, diff_method=None)
    def measure_z(x):
        apply_agpqk_encoding(x, gamma_qubit, entangle_pairs, n_qubits, reps)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return measure_x, measure_y, measure_z


def compute_agpqk_bloch_vectors(X_enc, gamma_qubit, entangle_pairs,
                                 n_qubits=N_QUBITS, reps=ZZ_REPS,
                                 save_path=None):
    """
    Compute Bloch vectors for AGPQK circuit.

    Args:
        X_enc:    Bandwidth-scaled feature matrix, shape (n, n_qubits)
        gamma_qubit:   Per-qubit gamma (for circuit construction)
        entangle_pairs: Attention-selected entanglement pairs
        save_path: Optional path to cache result

    Returns:
        bloch: np.ndarray, shape (n, 3*n_qubits) — same layout as standard PQK
    """
    if save_path and os.path.exists(save_path):
        print(f"  [SKIP] Loading cached AGPQK Bloch vectors: {save_path}")
        return np.load(save_path)

    n = len(X_enc)
    measure_x, measure_y, measure_z = build_agpqk_circuit(
        gamma_qubit, entangle_pairs, n_qubits, reps
    )

    bloch = np.zeros((n, 3 * n_qubits), dtype=np.float64)

    for i in tqdm(range(n), desc="AGPQK Bloch vectors"):
        x = X_enc[i]
        exp_x = np.array(measure_x(x))
        exp_y = np.array(measure_y(x))
        exp_z = np.array(measure_z(x))
        for q in range(n_qubits):
            bloch[i, 3*q]   = exp_x[q]
            bloch[i, 3*q+1] = exp_y[q]
            bloch[i, 3*q+2] = exp_z[q]

    if save_path:
        np.save(save_path, bloch)
        print(f"  Saved AGPQK Bloch vectors: {save_path}")

    return bloch


def compute_agpqk_kernel_from_bloch(bloch1, bloch2, gamma, n_qubits=N_QUBITS):
    """
    Compute outer RBF kernel from Bloch vectors.

    K(x, x') = exp(-γ · Σ_i (1/2) Σ_a (b_ia - b'_ia)²)

    Identical formula to standard PQK — only the Bloch vectors differ
    (from attention-guided circuit vs standard ZZFeatureMap circuit).

    Args:
        bloch1: (n1, 3*n_qubits)
        bloch2: (n2, 3*n_qubits)
        gamma:  Outer RBF bandwidth
    """
    n1 = len(bloch1)
    n2 = len(bloch2)

    b1 = bloch1.reshape(n1, n_qubits, 3)
    b2 = bloch2.reshape(n2, n_qubits, 3)

    K = np.zeros((n1, n2), dtype=np.float64)
    for i in tqdm(range(n1), desc="AGPQK kernel"):
        diff      = b1[i] - b2          # (n2, n_qubits, 3)
        sq_diff   = np.sum(diff**2, axis=2)  # (n2, n_qubits)
        frob_dist = 0.5 * np.sum(sq_diff, axis=1)  # (n2,)
        K[i, :]   = np.exp(-gamma * frob_dist)

    return K


# ── Stage 5: Compute Bloch vectors and kernel ─────────────────────────────
print("\n--- Stage 5: Computing AGPQK Bloch Vectors and Kernel ---")
print(f"  n_train={n_train}, n_test={n_test}, n_qubits={N_QUBITS}")
print(f"  Entanglement pairs: {entanglement_pairs}")
print(f"  Per-qubit γ: {np.round(gamma_per_qubit, 4)}")
print(f"  Outer γ: {OUTER_GAMMA}")
print()

bloch_train_path = os.path.join(AGPQK_DIR, "bloch_agpqk_physics_train.npy")
bloch_test_path  = os.path.join(AGPQK_DIR, "bloch_agpqk_physics_test.npy")
K_train_path     = os.path.join(AGPQK_DIR, "K_agpqk_physics_train.npy")
K_test_path      = os.path.join(AGPQK_DIR, "K_agpqk_physics_test.npy")

print("  Computing train Bloch vectors...")
bloch_train = compute_agpqk_bloch_vectors(
    X8_enc, gamma_per_qubit, entanglement_pairs,
    save_path=bloch_train_path
)

print("  Computing test Bloch vectors...")
bloch_test = compute_agpqk_bloch_vectors(
    X8_test_enc, gamma_per_qubit, entanglement_pairs,
    save_path=bloch_test_path
)

print(f"\n  Bloch train shape: {bloch_train.shape}  "
      f"range=[{bloch_train.min():.4f}, {bloch_train.max():.4f}]")

print("\n  Computing train kernel matrix (n×n)...")
if os.path.exists(K_train_path):
    print(f"  [SKIP] Loading cached: {K_train_path}")
    K_train = np.load(K_train_path)
else:
    K_train = compute_agpqk_kernel_from_bloch(bloch_train, bloch_train, OUTER_GAMMA)
    np.save(K_train_path, K_train)
    print(f"  Saved: {K_train_path}")

print("\n  Computing test kernel matrix (n_test × n_train)...")
if os.path.exists(K_test_path):
    print(f"  [SKIP] Loading cached: {K_test_path}")
    K_test = np.load(K_test_path)
else:
    K_test = compute_agpqk_kernel_from_bloch(bloch_test, bloch_train, OUTER_GAMMA)
    np.save(K_test_path, K_test)
    print(f"  Saved: {K_test_path}")


# ── Stage 6: Diagnostics and KTA ─────────────────────────────────────────
print("\n--- Stage 6: Diagnostics ---")

# Sanity checks
diag = np.diag(K_train)
off_idx = np.triu_indices(n_train, k=1)
off = K_train[off_idx]
cv  = off.std() / (off.mean() + 1e-12)

eigvals     = np.linalg.eigvalsh(K_train)
eigvals_pos = np.maximum(eigvals, 0)
total       = eigvals_pos.sum()
probs       = eigvals_pos / (total + 1e-12)
eff_rank    = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
top1_frac   = float(eigvals_pos[-1] / (total + 1e-12))

print(f"  Diagonal:      mean={diag.mean():.6f}  (should be 1.0)")
print(f"  Off-diag:      mean={off.mean():.4f}, std={off.std():.4f}")
print(f"  CV:            {cv:.4f}  (standard PQK was 0.7357)")
print(f"  Eff rank:      {eff_rank:.2f}  (standard PQK was 30.77)")
print(f"  Top-1 frac:    {top1_frac:.4f}")
print(f"  Min eigval:    {eigvals.min():.6f}  "
      f"({'PSD OK' if eigvals.min() >= -1e-6 else 'NOT PSD'})")

# KTA
print(f"\n--- KTA Results ---")
kta_agpqk = compute_centered_kta(K_train, y_train, class_weighted=True)
full_kta   = compute_full_kta_analysis(K_train, y_train)

print(f"\n  AGPQK KTA (centered weighted, PRIMARY): {kta_agpqk:.4f}")
print(f"  AGPQK kta_weighted (standard):          {full_kta['kta_weighted']:.4f}")
print(f"  AGPQK kta_centered_unweighted:          {full_kta['kta_centered_unweighted']:.4f}")

print(f"\n--- Comparison Table (physics features, n=2000) ---")
print(f"\n  {'Kernel':<12} {'KTA':>8}  {'vs RBF':>8}  Note")
print(f"  {'-'*50}")
all_results = {**KN_KTA, "AGPQK": kta_agpqk}
for name in ["RBF", "FQK", "PQK", "TFK", "AGPQK"]:
    if name not in all_results:
        continue
    kta     = all_results[name]
    d_rbf   = kta - all_results["RBF"]
    star    = " ★ BEATS RBF" if name != "RBF" and d_rbf > 0 else ""
    note    = " ← THIS RESULT" if name == "AGPQK" else ""
    print(f"  {name:<12} {kta:>8.4f}  {d_rbf:>+8.4f}{star}{note}")

print(f"\n  ΔKTA(AGPQK - RBF)     = {kta_agpqk - KN_KTA['RBF']:+.4f}")
print(f"  ΔKTA(AGPQK - std PQK) = {kta_agpqk - KN_KTA['PQK']:+.4f}")
print(f"  ΔKTA(AGPQK - FQK)     = {kta_agpqk - KN_KTA['FQK']:+.4f}")

# Save KTA to config
agpqk_config["kta_centered_weighted"] = float(kta_agpqk)
agpqk_config["kta_full"]              = {k: float(v) for k, v in full_kta.items()}
agpqk_config["concentration"] = {
    "off_diag_mean": float(off.mean()),
    "eff_rank":      float(eff_rank),
    "cv":            float(cv),
    "min_eigval":    float(eigvals.min()),
}
with open(config_path, "w") as f:
    json.dump(agpqk_config, f, indent=2)
print(f"\n  Results saved to: {config_path}")

print(f"\n{'='*65}")
if kta_agpqk > KN_KTA["RBF"]:
    print(f"  ★ AGPQK BEATS RBF — quantum advantage on physics features!")
    print(f"  Attention-guided encoding closes the quantum-classical gap.")
elif kta_agpqk > KN_KTA["PQK"]:
    improvement = kta_agpqk - KN_KTA["PQK"]
    print(f"  AGPQK > standard PQK by {improvement:+.4f}")
    print(f"  Attention guidance improved the quantum kernel.")
    print(f"  Gap to RBF remaining: {KN_KTA['RBF'] - kta_agpqk:.4f}")
else:
    print(f"  AGPQK did not improve over standard PQK.")
    print(f"  The attention-selected features/entanglement did not help.")
    print(f"  Inspect agpqk_config.json for selected features and pairs.")
print(f"{'='*65}")
