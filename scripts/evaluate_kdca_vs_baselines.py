"""
evaluate_kdca_vs_baselines.py

Definitively benchmarks the KDCA-PQK statistically-verified kernel subset against
the standard AGPQK baseline and Random Forest to evaluate its standalone classification performance.
"""
import os
import sys
import numpy as np
import json
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

print("="*65)
print("  KDCA-PQK vs AGPQK vs Random Forest Benchmark")
print("="*65)

# --- 1. Load Data Elements ---
PHYSICS_DIR = os.path.join("results", "physics", "fused")
PHYS16_PATH = os.path.join("data", "processed", "physics_features_16.npz")

K_KDCA_TRAIN = os.path.join(PHYSICS_DIR, "K_kdca_train.npy")
K_KDCA_TEST = os.path.join(PHYSICS_DIR, "K_kdca_test.npy")

K_AGPQK_TRAIN = os.path.join(PHYSICS_DIR, "K_agpqk_physics_train.npy")
K_AGPQK_TEST = os.path.join(PHYSICS_DIR, "K_agpqk_physics_test.npy")

data = np.load(PHYS16_PATH)
# Use N=2000 (which is what we evaluated KDCA at)
N_SAMPLES = min(2000, len(data["y_train"]))
y_train = data["y_train"][:N_SAMPLES]
y_test = data["y_test"][:N_SAMPLES]

# Check existing kernels
if not os.path.exists(K_KDCA_TRAIN) or not os.path.exists(K_KDCA_TEST):
    print("Precomputed KDCA kernels missing! Run compute_kdca_kernel.py first.")
    sys.exit(1)

# Format the matching X_train array for RF (8 selected indices)
AGPQK_CONFIG = os.path.join(PHYSICS_DIR, "agpqk_config.json")
if os.path.exists(AGPQK_CONFIG):
    with open(AGPQK_CONFIG, "r") as f:
        agpqk_config = json.load(f)
    selected_indices = agpqk_config.get("selected_feature_indices", [0, 2, 5, 6, 11, 12, 13, 15])
else:
    selected_indices = [0, 2, 5, 6, 11, 12, 13, 15]

# Using N=2000 for standard testing scale
X_train_rf = data["X_train"][:N_SAMPLES, selected_indices]
X_test_rf = data["X_test"][:N_SAMPLES, selected_indices]

# --- 2. Evaluate Random Forest ---
print("\n--- Training Classical Baseline: Random Forest ---")
rf = RandomForestClassifier(n_estimators=500, max_depth=None, class_weight='balanced', random_state=42)
rf.fit(X_train_rf, y_train)
y_pred_rf = rf.predict(X_test_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='macro')
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Macro-F1: {f1_rf:.4f} | Accuracy: {acc_rf:.4f}")

# --- 3. Evaluate AGPQK Baseline ---
print("\n--- Training Quantum Baseline: AGPQK ---")
if os.path.exists(K_AGPQK_TRAIN) and os.path.exists(K_AGPQK_TEST):
    K_tr_agpqk = np.load(K_AGPQK_TRAIN)[:N_SAMPLES, :N_SAMPLES]
    K_te_agpqk = np.load(K_AGPQK_TEST)[:N_SAMPLES, :N_SAMPLES]

    svm_agpqk = SVC(kernel='precomputed', C=1.0, class_weight='balanced')
    svm_agpqk.fit(K_tr_agpqk, y_train)
    y_pred_agpqk = svm_agpqk.predict(K_te_agpqk)
    f1_agpqk = f1_score(y_test, y_pred_agpqk, average='macro')
    acc_agpqk = accuracy_score(y_test, y_pred_agpqk)
    print(f"AGPQK-SVM Macro-F1: {f1_agpqk:.4f} | Accuracy: {acc_agpqk:.4f}")
else:
    f1_agpqk = 0.0
    print("AGPQK matrices were not found in the path. Skipping AGPQK evaluation.")

# --- 4. Evaluate KDCA-PQK ---
print("\n--- Training Novel Approach: KDCA-PQK ---")
K_tr_kdca = np.load(K_KDCA_TRAIN)[:N_SAMPLES, :N_SAMPLES]
K_te_kdca = np.load(K_KDCA_TEST)[:N_SAMPLES, :N_SAMPLES]

svm_kdca = SVC(kernel='precomputed', C=1.0, class_weight='balanced')
svm_kdca.fit(K_tr_kdca, y_train)
y_pred_kdca = svm_kdca.predict(K_te_kdca)
f1_kdca = f1_score(y_test, y_pred_kdca, average='macro')
acc_kdca = accuracy_score(y_test, y_pred_kdca)
print(f"KDCA-PQK SVM Macro-F1: {f1_kdca:.4f} | Accuracy: {acc_kdca:.4f}")

print("\n--- Final Conclusion ---")
if f1_kdca > f1_rf and f1_kdca > f1_agpqk:
    print(" [✓] Success! KDCA explicitly outperforms both classical RF and baseline AGPQK geometries.")
elif f1_kdca > f1_agpqk:
    print(" [✓] KDCA outperforms previous Quantum Baselines (AGPQK), despite failing to surpass un-scaled classical RF bounds.")
else:
    print(" [!] KDCA structural framework underperforms AGPQK/RF limits. Consider tuning default gamma or feature bandwidth.")
