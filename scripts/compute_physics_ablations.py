"""
Physics Feature Ablation Study.
Runs RBF-SVM on the exact 8-feature subset used by AGPQK.
Isolates whether gain is from feature selection or quantum kernel.
"""

import os
import sys
import json
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score

# Ensure we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.classical_kernels import compute_rbf_kernel
from src.kernel_target_alignment import compute_centered_kta

def run_ablation():
    # --- Step 1: Load data ---
    data_path = "data/processed/physics_features_16.npz"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    data = np.load(data_path)
    X_train_all = data['X_train']
    y_train = data['y_train']
    X_test_all = data['X_test']
    y_test = data['y_test']

    # --- Step 2: Select AGPQK subset ---
    # From agpqk_config.json: "selected_feature_indices": [0, 2, 5, 6, 11, 12, 13, 15]
    selected_indices = [0, 2, 5, 6, 11, 12, 13, 15]
    X_train_sub = X_train_all[:, selected_indices]
    X_test_sub = X_test_all[:, selected_indices]

    print(f"Full physics features: {X_train_all.shape[1]}")
    print(f"AGPQK subset features: {X_train_sub.shape[1]}")

    # --- Step 3: Tuned RBF on subset ---
    print("\nTuning RBF-SVM on AGPQK subset...")
    param_grid = {'gamma': np.logspace(-3, 1, 10), 'C': [1.0, 10.0, 100.0]}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # We use a smaller subset for grid search if needed, but 2000 is small enough
    grid = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced'),
        param_grid, cv=skf, scoring='f1_macro', n_jobs=-1
    )
    grid.fit(X_train_sub, y_train)
    
    best_svc = grid.best_estimator_
    best_gamma = grid.best_params_['gamma']
    print(f"Best RBF gamma: {best_gamma:.4f}, C: {grid.best_params_['C']}")

    # --- Step 4: Evaluate ---
    y_pred = best_svc.predict(X_test_sub)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    acc = accuracy_score(y_test, y_pred)
    
    # Compute KTA for this RBF
    K_rbf = compute_rbf_kernel(X_train_sub, gamma=best_gamma)
    kta = compute_centered_kta(K_rbf, y_train, class_weighted=True)

    print(f"\nRBF (AGPQK-subset) Results:")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  KTA:      {kta:.4f}")

    # --- Step 5: Compare with Full RBF (from summary) ---
    # RBF (CV-tuned) full physics: F1=0.466, KTA=0.216
    
    # --- Step 6: Save results ---
    results = {
        "experiment": "Physics Ablation: RBF on AGPQK-selected features",
        "subset_indices": selected_indices,
        "best_params": grid.best_params_,
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "kta": float(kta),
        "comparison": {
            "rbf_full_physics_f1": 0.443, # 5-fold CV mean
            "agpqk_physics_f1": 0.487     # 5-fold CV mean
        }
    }
    
    out_dir = os.path.join("results", "physics", "ablation")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rbf_on_agpqk_features.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {os.path.join(out_dir, 'rbf_on_agpqk_features.json')}")

if __name__ == "__main__":
    run_ablation()
