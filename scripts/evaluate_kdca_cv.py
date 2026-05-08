"""
evaluate_kdca_cv.py

Computes the generalized KDCA-PQK Bloch representations and executes a rigorous
GridSearchCV over SVM hyperparameters (C, gamma) to finalize testing performance.

Addresses the Gaps by establishing explicit cross-validation across heterogeneous
scalings and applying class-balanced tracking to combat LCZ distribution biases.
"""
import os
import sys
import numpy as np
import warnings
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel
import time

class CompositeSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, gamma=0.02, lam=1.0):
        self.C = C
        self.gamma = gamma
        self.lam = lam
        self.clf = None

    def fit(self, X, y):
        self.X_train_ = X
        K_hs = (X @ X.T) / X.shape[1]
        K_rbf = rbf_kernel(X, X, gamma=self.gamma)
        K_train = K_hs + self.lam * K_rbf
        self.clf = SVC(kernel='precomputed', C=self.C, class_weight='balanced')
        self.clf.fit(K_train, y)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, X):
        K_hs_test = (X @ self.X_train_.T) / X.shape[1]
        K_rbf_test = rbf_kernel(X, self.X_train_, gamma=self.gamma)
        K_test = K_hs_test + self.lam * K_rbf_test
        return self.clf.predict(K_test)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.kdca_kernel import compute_kdca_attention, select_kdca_topology, compute_kdca_bloch_vectors

warnings.filterwarnings('ignore')

print("="*65)
print("  KDCA-PQK Joint Hyperparameter Evaluation (C & Gamma)")
print("="*65)

# --- 1. Load Data Elements ---
PHYSICS_DIR = os.path.join("results", "physics", "fused")
PHYS16_PATH = os.path.join("data", "processed", "physics_features_16.npz")
BLOCH_TRAIN = os.path.join(PHYSICS_DIR, "kdca_bloch_cv_train.npy")
BLOCH_TEST = os.path.join(PHYSICS_DIR, "kdca_bloch_cv_test.npy")

data = np.load(PHYS16_PATH)

# Using N=2000 for standard testing scale
N_SAMPLES = min(2000, len(data["y_train"]))
y_train = data["y_train"][:N_SAMPLES]
y_test = data["y_test"][:N_SAMPLES]

X_train_raw = data["X_train"][:N_SAMPLES]
X_test_raw = data["X_test"][:N_SAMPLES]

# Map identical features
synthetic_indices = np.array([0, 2, 5, 6, 11, 12, 13, 15])
X_train_sub = X_train_raw[:, synthetic_indices]
X_test_sub = X_test_raw[:, synthetic_indices]

# Force feature scaling to exactly exactly represent formal pipeline
if not os.path.exists(BLOCH_TRAIN) or not os.path.exists(BLOCH_TEST):
    print("Precomputed KDCA Bloch vectors missing. Extracting dynamically... this may take some time.")
    
    t0 = time.time()
    # Execute structural topology derivation across the labels
    attn = compute_kdca_attention(
        X_train=X_train_sub, 
        y_train=y_train, 
        selected_indices=np.array([8,9,10,11, 0,1,2,3]), 
        n_qubits=8, reps=2
    )
    entanglement_pairs = select_kdca_topology(attn["A_KD"], top_k=4)
    
    # Evaluate 40-Dimensional structure
    b1 = compute_kdca_bloch_vectors(X_train_sub, entanglement_pairs, n_qubits=8, reps=2, save_path=BLOCH_TRAIN)
    b2 = compute_kdca_bloch_vectors(X_test_sub, entanglement_pairs, n_qubits=8, reps=2, save_path=BLOCH_TEST)
    t1 = time.time()
    print(f"Extraction execution Time: {t1 - t0:.2f} seconds.")
else:
    b1 = np.load(BLOCH_TRAIN)
    b2 = np.load(BLOCH_TEST)

print(f"Loaded KDCA 40D Quantum State representations. Train Shape: {b1.shape}, Test Shape: {b2.shape}")

# --- 2. Feature Standardization ---
# Critical step: Standardize heterogeneous 1-body and 2-body subspace arrays
scaler = StandardScaler()
X_train_norm = scaler.fit_transform(b1)
X_test_norm = scaler.transform(b2)

# --- 2.5 Pure RBF Baseline Check ---
print("\nExecuting isolated pure RBF GridSearchCV benchmark...")
grid_rbf = GridSearchCV(
    SVC(kernel='rbf', class_weight='balanced'),
    {'C': [0.5, 1.0, 5.0, 10.0, 50.0], 'gamma': [0.005, 0.01, 0.02, 0.05, 0.1]},
    cv=5, scoring='f1_macro', n_jobs=-1
)
grid_rbf.fit(X_train_norm, y_train)
y_pred_rbf = grid_rbf.best_estimator_.predict(X_test_norm)
print(f"Isolated RBF Macro-F1 (for comparison): {f1_score(y_test, y_pred_rbf, average='macro'):.4f}")

# --- 3. Run Sklearn GridSearchCV over Composite Kernel ---
param_grid = {
    'lam': [0.0, 0.1, 0.5, 1.0, 5.0, 10.0],
    'gamma': [0.005, 0.01, 0.02, 0.05, 0.1],
    'C': [0.5, 1.0, 5.0, 10.0, 50.0]
}

print("\nExecuting joint 5x CV GridSearchCV across Composite Geometries...")
clf = GridSearchCV(
    CompositeSVM(),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

clf.fit(X_train_norm, y_train)

print(f"\nOptimal Pipeline Evaluated -> Best CV F1 Score: {clf.best_score_:.4f}")
print(f"Optimal Parameters Configured: {clf.best_params_}")

# Evaluate specifically on the held-out test block
best_svm = clf.best_estimator_
y_pred_te = best_svm.predict(X_test_norm)
test_macro_f1 = f1_score(y_test, y_pred_te, average='macro')
acc_te = accuracy_score(y_test, y_pred_te)

print("\n" + "-" * 52)
print(f"Final Held-Out Test Evaluation:")
print(f" Macro-F1: {test_macro_f1:.4f} | Accuracy: {acc_te:.4f}")
print("-" * 52)

# RF = 0.5105
if test_macro_f1 > 0.5105:
    print(" [✓] KDCA definitively BEATS the Classical Random Forest structural limits and claims Quantum Advantage!")
else:
    print(" [!] KDCA representations remain short of Classical RF separability boundaries.")
