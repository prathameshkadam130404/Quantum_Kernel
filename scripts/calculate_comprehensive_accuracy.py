import os
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from experiments._bscm_split import make_or_load_split
from experiments._e_common import load_physics_16
from src.attention_kernel import select_features_by_fisher
from experiments.exp_e38_max_data_pqk import _load_so2sat_max

def evaluate_classical(X_raw, y, n_samples):
    print(f"  -> Evaluating Classical RBF & RF for N={n_samples}...")
    sss = StratifiedShuffleSplit(n_splits=5 if n_samples < 5000 else 3, test_size=0.30, random_state=42)
    rf_accs = []
    rbf_accs = []
    
    for seed in [42, 43, 44, 45, 46][:sss.n_splits]:
        sss_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        tr, te = next(sss_cv.split(X_raw, y))
        
        # RF
        rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1)
        rf.fit(X_raw[tr], y[tr])
        rf_accs.append(accuracy_score(y[te], rf.predict(X_raw[te])))
        
        # RBF
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_raw[tr])
        X_te_sc = sc.transform(X_raw[te])
        svm = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), {"C": [1, 10, 100], "gamma": ["scale", "auto"]}, cv=3, n_jobs=-1)
        svm.fit(X_tr_sc, y[tr])
        rbf_accs.append(accuracy_score(y[te], svm.predict(X_te_sc)))

    return np.mean(rf_accs), np.mean(rbf_accs)

def evaluate_cached_kernel(cache_path, y, n_samples):
    if not os.path.exists(cache_path):
        return None
    K = np.load(cache_path)
    if "K" in K:  # npz format
        K = K["K"]
    
    print(f"  -> Evaluating cached kernel: {os.path.basename(cache_path)}...")
    accs = []
    splits = 5 if n_samples < 5000 else 3
    for seed in [42, 43, 44, 45, 46][:splits]:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        tr, te = next(sss.split(K, y))
        clf = GridSearchCV(SVC(kernel="precomputed", class_weight="balanced"), {"C": [1, 10, 100]}, cv=3, n_jobs=-1)
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        accs.append(accuracy_score(y[te], clf.predict(K[np.ix_(te, tr)])))
    return np.mean(accs)

def run_e32_8qubit_evaluation():
    print(f"\n{'='*60}\nPHASE 1: 8-Qubit So2Sat Fidelity & PQK (N=1800, 8 Features)\n{'='*60}")
    X_norm, X_raw, y_all = load_physics_16()
    sel_idx, _ = select_features_by_fisher(X_raw, y_all)
    X_raw_sel = X_raw[:, sel_idx]
    
    split = make_or_load_split(y_all)
    X_eval = X_raw_sel[split.eval_pool]
    y_eval = y_all[split.eval_pool]
    
    rf_acc, rbf_acc = evaluate_classical(X_eval, y_eval, len(y_eval))
    
    caches = {
        "BSCM-uniform (Fid)": os.path.join(config.RESULTS_DIR, "bscm", "K_bscm_physics8_uniform.npy"),
        "BSCM-phi_only (Fid)": os.path.join(config.RESULTS_DIR, "bscm", "K_bscm_physics8_phi_only.npy"),
        "BSCM-psi_only (Fid)": os.path.join(config.RESULTS_DIR, "bscm", "K_bscm_physics8_psi_only.npy"),
        "SRQFM-fid (Fid)": os.path.join(config.RESULTS_DIR, "bscm", "K_srqfm_fid_physics8.npy"),
        "SRQFM-PQK": os.path.join(config.RESULTS_DIR, "bscm", "K_srqfm_pqk_physics8.npy"),
        "SG-BSCM-fid (Fid)": os.path.join(config.RESULTS_DIR, "sg_bscm", "K_SG-BSCM_so2sat_physics8.npy")
    }
    
    results = {"Random Forest (Classical)": rf_acc, "RBF-SVM (Classical)": rbf_acc}
    for name, path in caches.items():
        acc = evaluate_cached_kernel(path, y_eval, len(y_eval))
        if acc is not None:
            results[name] = acc
            
    print("\nRESULTS TABLE (N=1800):")
    print(f"{'Method':<30} | {'Accuracy':<10}")
    print("-" * 45)
    # Sort by accuracy descending
    for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:<30} | {v:.4f}")

def run_e38_16qubit_evaluation():
    print(f"\n{'='*60}\nPHASE 2: 16-Qubit So2Sat Max Capacity PQK (N=10000, 16 Features)\n{'='*60}")
    _, X_raw, y = _load_so2sat_max()
    
    rf_acc, rbf_acc = evaluate_classical(X_raw, y, len(y))
    
    caches = {
        "BSCM-PQK": os.path.join(config.RESULTS_DIR, "e38_max_data", "cache_so2sat.npz"),
        "SRQFM-PQK": os.path.join(config.RESULTS_DIR, "e38_max_data", "cache_so2sat_srqfm.npz"),
        "Standard-PQK (Havlicek)": os.path.join(config.RESULTS_DIR, "e38_max_data", "cache_so2sat_standard_pqk.npz")
    }
    
    results = {"Random Forest (Classical)": rf_acc, "RBF-SVM (Classical)": rbf_acc}
    for name, path in caches.items():
        acc = evaluate_cached_kernel(path, y, len(y))
        if acc is not None:
            results[name] = acc
            
    print("\nRESULTS TABLE (N=10000):")
    print(f"{'Method':<30} | {'Accuracy':<10}")
    print("-" * 45)
    for k, v in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{k:<30} | {v:.4f}")

if __name__ == "__main__":
    run_e32_8qubit_evaluation()
    run_e38_16qubit_evaluation()
