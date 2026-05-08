"""
evaluate_qaak_vs_rf.py

Definitively benchmarks the QAAK mathematically-routed feature subspace against
Random Forest to prove quantum structural superiority in an empirical setting.
"""
import os
import sys
import numpy as np
import json
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

print("="*65)
print("  QAAK vs Random Forest Benchmark")
print("="*65)

# --- 1. Load Data Elements ---
PHYSICS_DIR = os.path.join("results", "physics", "fused")
PHYS16_PATH = os.path.join("data", "processed", "physics_features_16.npz")
K_QAAK_TRAIN = os.path.join(PHYSICS_DIR, "K_qaak_physics_train.npy")
K_QAAK_TEST = os.path.join(PHYSICS_DIR, "K_qaak_physics_test.npy")

data = np.load(PHYS16_PATH)
y_train = data["y_train"]
y_test = data["y_test"]

if not os.path.exists(K_QAAK_TRAIN):
    print("Precomputed QAAK train kernel missing! Run compute_qaak_kernel.py first.")
    sys.exit(1)

# To generate predictions, we need K_test.
# Let's see if it's computed. If not, we can only evaluate KTA or run RF directly.
HAS_K_TEST = os.path.exists(K_QAAK_TEST)

# Format the matching X_train array
AGPQK_CONFIG = os.path.join(PHYSICS_DIR, "agpqk_config.json")
if os.path.exists(AGPQK_CONFIG):
    with open(AGPQK_CONFIG, "r") as f:
        agpqk_config = json.load(f)
    selected_indices = agpqk_config.get("selected_feature_indices", [0, 2, 5, 6, 11, 12, 13, 15])
else:
    selected_indices = [0, 2, 5, 6, 11, 12, 13, 15]

X_train_rf = data["X_train"][:, selected_indices]
X_test_rf = data["X_test"][:, selected_indices]

# --- 2. Train and Evaluate Random Forest (Infinite Depth Limit) ---
print("\n--- Training Random Forest ---")
rf = RandomForestClassifier(n_estimators=500, max_depth=None, class_weight='balanced', random_state=42)
rf.fit(X_train_rf, y_train)

y_pred_rf = rf.predict(X_test_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='macro')
print(f"Random Forest Macro-F1: {f1_rf:.4f}")

# --- 3. Evaluate QAAK-SVM ---
print("\n--- Training QAAK SVM ---")
if HAS_K_TEST:
    K_train = np.load(K_QAAK_TRAIN)
    K_test = np.load(K_QAAK_TEST)
    
    # Needs balanced class weight to be a fair comparison for LCZ42
    # C=1.0 is standard for SVM unless tuned via GridSearchCV
    qaak_svm = SVC(kernel='precomputed', C=1.0, class_weight='balanced')
    qaak_svm.fit(K_train, y_train)
    
    y_pred_qaak = qaak_svm.predict(K_test)
    f1_qaak = f1_score(y_test, y_pred_qaak, average='macro')
    
    print(f"QAAK Macro-F1: {f1_qaak:.4f}")
    
    print("\nConclusion:")
    if f1_qaak > f1_rf:
        print(" [✓] Success! QAAK Autonomous Entanglement outperforms Random Forest structural partitions.")
    else:
        print(" [!] Random Forest still outperforms locally via hard splitting thresholds.")
else:
    print("QAAK test kernel not yet precomputed. Only RF baseline is available.")
