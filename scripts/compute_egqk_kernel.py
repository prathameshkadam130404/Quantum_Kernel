import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from src.egqk_kernel import extract_egqk_fidelities

print("=" * 65)
print("  EGQK — Entanglement-Gated Quantum Kernel Base Execution")
print("=" * 65)

# 1. Load the unified target dataset
# The 16 features mapping specifically pairs Optical and SAR modalities
DATA_PATH = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"[!] NPZ file missing: {DATA_PATH}")

print(f"Loading data from: {DATA_PATH}")
data = np.load(DATA_PATH, allow_pickle=True)

# 2. Extract strictly bounded subsets for computational constraint parity
X_tr_full = data["X_train"]
y_tr_full = data["y_train"]
X_te_full = data["X_test"]
y_te_full = data["y_test"]

n_train = min(config.SUBSAMPLE_TRAIN, len(X_tr_full))
n_test = min(config.SUBSAMPLE_TEST, len(X_te_full))

# Standard Feature Alignments (Indices specific to LCZ42)
# Optical: 0, 2, 5, 6
# SAR: 11, 12, 13, 15
selected_indices = [0, 2, 5, 6, 11, 12, 13, 15]

# Scale features to generic bounds to cover the phase-sphere logically
X_tr = X_tr_full[:n_train, selected_indices] * 2.0
X_te = X_te_full[:n_test, selected_indices] * 2.0
print(f"Data scaled geometrically. Train size: {n_train}, Test size: {n_test}")

# 3. Form Topological Graph Pairs based strictly on physical cross-interactions
pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
print(f"Entanglement Gate Pairings: {pairs}")

# 4. Bind Hardware Simulator
q_device = config.get_device(wires=config.N_QUBITS, shots=None)

# 5. Extract Native Quantum Fidelity Kernel
K_train, K_test = extract_egqk_fidelities(X_tr, X_te, q_device, pairs)

# 6. Save Artifacts for ML execution
print("Saving Kernel Matrices...")
save_tr_path = os.path.join(config.RESULTS_DIR, "K_EGQK_train.npy")
save_te_path = os.path.join(config.RESULTS_DIR, "K_EGQK_test.npy")

np.save(save_tr_path, K_train)
np.save(save_te_path, K_test)

print(f"[Done] Artefacts written to {config.RESULTS_DIR}")
