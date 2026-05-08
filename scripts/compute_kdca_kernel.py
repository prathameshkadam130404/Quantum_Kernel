"""
compute_kdca_kernel.py

Computes the novel Kirkwood–Dirac Cross-Basis Coherence Attention Projected 
Quantum Kernel (KDCA-PQK).

This execution logically isolates the completely quantum-native attention mechanism 
from classical routines by computing the Kirkwood-Dirac nonpositivity (two-body YX, XY, YY
expectation magnitudes) and builds an expanded 36D KDCA-PQK feature vector.

Produces KTA benchmark over 2000 LCZ training samples using identical 
feature splits and topological definitions as AGPQK to ensure 1:1 rigor.
"""
import sys, os
import json
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.kernel_target_alignment import compute_centered_kta
from src.kdca_kernel import compute_kdca_pqk_full_pipeline

# Constants
N_QUBITS = config.N_QUBITS
ZZ_REPS = config.ZZ_REPS

# Paths
PHYSICS_DIR  = os.path.join("results", "physics", "fused")
PHYS16_PATH  = os.path.join("data", "processed", "physics_features_16.npz")
AGPQK_CONFIG = os.path.join(PHYSICS_DIR, "agpqk_config.json")
os.makedirs(PHYSICS_DIR, exist_ok=True)

print("=" * 65)
print("  KDCA-PQK — Kirkwood-Dirac Attention Kernel Execution")
print("=" * 65)

# --- 1. Load Data ---
data = np.load(PHYS16_PATH)
X_train_raw = data["X_train"] 
y_train = data["y_train"]
X_test_raw = data["X_test"]

# Allow smaller sample logic for fast validation runs, default to full scale otherwise
# Using N=2000 matching formal testing scale
N_SAMPLES = min(2000, len(X_train_raw))
X_train_run = X_train_raw[:N_SAMPLES]
y_train_run = y_train[:N_SAMPLES]
X_test_run = X_test_raw[:N_SAMPLES]

# --- 2. Align Features to AGPQK ---
if os.path.exists(AGPQK_CONFIG):
    with open(AGPQK_CONFIG, "r") as f:
        ag_config = json.load(f)
    selected_indices = ag_config.get("selected_feature_indices", [0, 2, 5, 6, 11, 12, 13, 15])
    print(f"Loaded strict AGPQK Feature Config: {selected_indices}")
else:
    selected_indices = [0, 2, 5, 6, 11, 12, 13, 15]

# Subset to N_QUBITS
X_train_sub = X_train_run[:, selected_indices]
X_test_sub = X_test_run[:, selected_indices]

# Using normalized data [0, pi]
KDCA_BASE_BANDWIDTH = 1.0 # Base multiplier
X_train_sub = X_train_sub * KDCA_BASE_BANDWIDTH
X_test_sub = X_test_sub * KDCA_BASE_BANDWIDTH

print("\n--- Computing KDCA-PQK Feature Extractor and RBF Pipeline ---")
t0 = time.time()
res = compute_kdca_pqk_full_pipeline(
    X_train=X_train_sub,
    X_test=X_test_sub,
    y_train=y_train_run,
    selected_indices=np.array(selected_indices),
    gamma_kernel=config.PQK_GAMMA,
    top_k_pairs=4,
    n_qubits=N_QUBITS,
    reps=ZZ_REPS,
    save_dir=PHYSICS_DIR
)
t1 = time.time()
print(f"KDCA-PQK embedding and metrics runtime: {t1 - t0:.2f} seconds.")

K_train = res["K_train"]
print(f"Kernel matrices saved! Train Shape: {K_train.shape}")

print("\n--- Phase 3: Benchmark (KTA) ---")
kta_val = compute_centered_kta(K_train, y_train_run)
print(f"KDCA-PQK KTA (Training N={N_SAMPLES}): {kta_val:.4f}")

# Historical benchmark for Context
print("\nHistorical Context:")
print(f"  AGPQK / RBF reference: ~0.2956")
if kta_val > 0.2956:
    print("  [✓] KDCA exhibits STRONGER Kernel Target Alignment than classical RBF/AGPQK!")
else:
    print("  [!] KDCA structural coherence aligns lower than standard references in this subspace.")
