import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

print("=" * 65)
print("  EGQK vs Random Forest Benchmark Evaluator")
print("=" * 65)

# 1. Load native labels
DATA_PATH = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"[!] NPZ file missing: {DATA_PATH}")

data = np.load(DATA_PATH, allow_pickle=True)
y_tr_full = data["y_train"]
y_te_full = data["y_test"]

n_train = min(config.SUBSAMPLE_TRAIN, len(y_tr_full))
n_test = min(config.SUBSAMPLE_TEST, len(y_te_full))

y_train = y_tr_full[:n_train]
y_test = y_te_full[:n_test]

# 2. Check Random Forest Limit
try:
    with open("results/RF_baseline_F1.txt", "r") as f:
        rf_limit = float(f.read().strip())
except FileNotFoundError:
    rf_limit = 0.5105 # Historic empirical run

# 3. Load natively generated EGQK matrices
K_EGQK_TRAIN = os.path.join(config.RESULTS_DIR, "K_EGQK_train.npy")
K_EGQK_TEST = os.path.join(config.RESULTS_DIR, "K_EGQK_test.npy")

print("Loading Pure Quantum Fidelity Matrices...")
K_train = np.load(K_EGQK_TRAIN)
K_test = np.load(K_EGQK_TEST)

# 4. Tune the SVM to ensure maximum extraction of the geometric space boundaries
print("\n--- Tuning EGQK SVM Margin Boundaries ---")

param_grid = {'C': [1.0, 10.0, 50.0, 100.0]}
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid = GridSearchCV(
    SVC(kernel='precomputed', class_weight='balanced', random_state=42),
    param_grid,
    scoring="f1_macro",
    cv=inner_cv,
    n_jobs=-1,
    refit=True
)

grid.fit(K_train, y_train)
best_c = grid.best_params_['C']
print(f"Optimal Hyperplane Tuning achieved at C={best_c}")

# 5. Execute Evaluation
print("\n--- Final EGQK Test Set Benchmark ---")
y_pred_egqk = grid.predict(K_test)
egqk_f1 = f1_score(y_test, y_pred_egqk, average='macro')

print(f"EGQK Macro-F1: {egqk_f1:.4f}")
print(f"Random Forest Benchmark: {rf_limit:.4f}")

print("\nConclusion:")
if egqk_f1 > rf_limit:
    print(f" [V] EGQK mathematically outperformed Random Forest by +{(egqk_f1 - rf_limit):.4f}!")
    print("     The pure fidelity conditionally-gated architecture natively captures discontinuous tabular topologies.")
else:
    print(" [!] Random Forest retained topological dominance.")
    
# Save result explicitly
with open(os.path.join(config.RESULTS_DIR, "EGQK_Final_F1.txt"), "w") as f:
    f.write(str(egqk_f1))
