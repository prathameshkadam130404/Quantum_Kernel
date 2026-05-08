"""
compute_qaak_kernel.py

Computes the novel Quantum Autonomous Attention Kernel (QAAK). 
Replaces the static classical covariance attention of AGPQK with a quantum-native
interference mechanism (CSWAP tests) to naturally route entanglement and dephase noise.

Produces KTA benchmark over 2000 LCZ training samples using identical 
feature splits and topological definitions as AGPQK to ensure 1:1 rigor.
"""
import sys, os
import json
import numpy as np
import time
import pennylane as qml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.kernel_target_alignment import compute_centered_kta
from src.qaak_kernel import build_qaak_circuit, extract_qaak_bloch_vectors

# Constants
N_QUBITS = 8
N_ANCILLAS = 4

# Paths
PHYSICS_DIR  = os.path.join("results", "physics", "fused")
PHYS16_PATH  = os.path.join("data", "processed", "physics_features_16.npz")
AGPQK_CONFIG = os.path.join(PHYSICS_DIR, "agpqk_config.json")
os.makedirs(PHYSICS_DIR, exist_ok=True)

print("=" * 65)
print("  QAAK — Quantum Autonomous Attention Kernel Base Execution")
print("=" * 65)

# --- 1. Load Data ---
data = np.load(PHYS16_PATH)
X_train_raw = data["X_train"] 
y_train = data["y_train"]

# --- 2. Align Features to AGPQK ---
if os.path.exists(AGPQK_CONFIG):
    with open(AGPQK_CONFIG, "r") as f:
        ag_config = json.load(f)
    selected_indices = ag_config.get("selected_feature_indices", [0, 2, 5, 6, 11, 12, 13, 15])
    print(f"Loaded strict AGPQK Feature Config: {selected_indices}")
else:
    selected_indices = [0, 2, 5, 6, 11, 12, 13, 15]

# Using normalized data [0, pi]
# Crucial structural fix: Multipling by 0.5 to prevent absolute phase cancellation 
# which causes the CSWAP ancillas to fully dephase the data register into white noise.
QAAK_BASE_BANDWIDTH = 0.5
X_train = data["X_train"][:, selected_indices] * QAAK_BASE_BANDWIDTH
X_test = data["X_test"][:, selected_indices] * QAAK_BASE_BANDWIDTH

# Pairs mapping optical (0-3) vs SAR (4-7)
pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
print(f"Topological Pairings for QAAK Interference: {pairs}")

# --- 3. Build PennyLane Framework ---
# Use default.qubit - using expval natively traces unmeasured wires automatically
dev = qml.device('default.qubit', wires=N_QUBITS + N_ANCILLAS)
measure_x, measure_y, measure_z = build_qaak_circuit(dev, pairs, zz_reps=1)

# Allow smaller sample logic for fast validation runs, default to full scale otherwise
N_SAMPLES = min(2000, len(X_train))

print("\n--- Phase 1: Generating QAAK Subspace ---")
t0 = time.time()
bloch_train = extract_qaak_bloch_vectors(X_train[:N_SAMPLES], measure_x, measure_y, measure_z, log_frequency=200)
bloch_test = extract_qaak_bloch_vectors(X_test[:N_SAMPLES], measure_x, measure_y, measure_z, log_frequency=200)
t1 = time.time()
print(f"QAAK embedding runtime: {t1 - t0:.2f} seconds.")

# Save
np.save(os.path.join(PHYSICS_DIR, "bloch_qaak_train.npy"), bloch_train)
np.save(os.path.join(PHYSICS_DIR, "bloch_qaak_test.npy"), bloch_test)

def compute_pqk_matrix_from_bloch(bloch_x, bloch_y, gamma):
    # Vectorized Frobenius outer kernel
    n_x = len(bloch_x)
    n_y = len(bloch_y)
    dist = np.zeros((n_x, n_y))
    for i in range(n_x):
        diffs = bloch_y - bloch_x[i]
        dist[i] = np.sum(diffs**2, axis=1)
    return np.exp(-gamma * dist)

print("\n--- Phase 2: Executing Projected Inner Product ---")
# Use the proven PQK_GAMMA scaling rule
gamma = config.PQK_GAMMA
K_train = compute_pqk_matrix_from_bloch(bloch_train, bloch_train, gamma)
K_test = compute_pqk_matrix_from_bloch(bloch_test, bloch_train, gamma)

np.save(os.path.join(PHYSICS_DIR, "K_qaak_physics_train.npy"), K_train)
np.save(os.path.join(PHYSICS_DIR, "K_qaak_physics_test.npy"), K_test)
print(f"Kernel matrices saved! Train Shape: {K_train.shape}, Test Shape: {K_test.shape}")

print("\n--- Phase 3: Benchmark (KTA) ---")
kta_val = compute_centered_kta(K_train, y_train[:N_SAMPLES])
print(f"QAAK KTA (Training): {kta_val:.4f}")

# Historical benchmark for Context
print("\nHistorical Context:")
print(f"  AGPQK / RBF reference: ~0.2956")
if kta_val > 0.2956:
    print("  [✓] QAAK exhibits STRONGER Kernel Target Alignment than classical RBF/AGPQK!")
else:
    print("  [!] QAAK structural coherence aligns lower than standard references in this subspace.")
