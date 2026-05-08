import sys, os, json
sys.path.insert(0, '.')
import numpy as np
import config
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score

# Load subsample
data = np.load("data/processed/subsample_2000.npz")  # adjust path if needed
X_train, X_test = data["fused_X_train"], data["fused_X_test"]
y_train, y_test = data["y_train"], data["y_test"]
print(f"Loaded: X_train={X_train.shape}, X_test={X_test.shape}")

out_dir = os.path.join(config.RESULTS_DIR, "classification", "tuned_classical")
os.makedirs(out_dir, exist_ok=True)
results = {}

# 1. Tuned RBF-SVM
print("\nRunning GridSearchCV RBF-SVM...")
grid = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced', random_state=42),
    {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.01, 0.1]},
    scoring='f1_macro', cv=5, n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)
y_pred = grid.best_estimator_.predict(X_test)
results['RBF-SVM-Tuned'] = {
    'macro_f1': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'best_params': grid.best_params_,
    'cv_score': float(grid.best_score_),
}
print(f"  Best params: {grid.best_params_}")
print(f"  Test macro-F1: {results['RBF-SVM-Tuned']['macro_f1']:.4f}")

# 2. Random Forest
print("\nRunning Random Forest...")
rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
results['RandomForest'] = {
    'macro_f1': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
    'accuracy': float(accuracy_score(y_test, y_pred)),
}
print(f"  macro-F1: {results['RandomForest']['macro_f1']:.4f}")

# 3. Gradient Boosting
print("\nRunning Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=200, random_state=42)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
results['GradientBoosting'] = {
    'macro_f1': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
    'accuracy': float(accuracy_score(y_test, y_pred)),
}
print(f"  macro-F1: {results['GradientBoosting']['macro_f1']:.4f}")

# Save
out_path = os.path.join(out_dir, "tuned_classical_results.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n=== SUMMARY ===")
for name, m in results.items():
    print(f"  {name:<22} macro-F1 = {m['macro_f1']:.4f}")
print(f"\nSaved: {out_path}")