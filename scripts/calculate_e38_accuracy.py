import os
import sys
import numpy as np
import json
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments.exp_e38_max_data_pqk import _load_so2sat_max, _load_eurosat_10k

RESULTS_DIR = os.path.join(config.RESULTS_DIR, "e38_max_data")

def evaluate_precomputed(K, y, name):
    print(f"Evaluating {name}...")
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
    f1s, accs = [], []
    for tr, te in sss.split(K, y):
        K_tr = K[np.ix_(tr, tr)]
        K_te = K[np.ix_(te, tr)]
        clf = GridSearchCV(SVC(kernel="precomputed", class_weight="balanced"), {"C": [0.1, 1.0, 10.0]}, cv=3, n_jobs=-1)
        clf.fit(K_tr, y[tr])
        p = clf.predict(K_te)
        f1s.append(f1_score(y[te], p, average="macro"))
        accs.append(accuracy_score(y[te], p))
    return np.mean(f1s), np.mean(accs)

def evaluate_classical(X_raw, y, name):
    print(f"Evaluating {name} (Classical)...")
    sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=42)
    
    rbf_f1, rbf_acc = [], []
    rf_f1, rf_acc = [], []
    
    for tr, te in sss.split(X_raw, y):
        # SVM
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_raw[tr])
        X_te_sc = sc.transform(X_raw[te])
        svm = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"), {"C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"]}, cv=3, n_jobs=-1)
        svm.fit(X_tr_sc, y[tr])
        p_svm = svm.predict(X_te_sc)
        rbf_f1.append(f1_score(y[te], p_svm, average="macro"))
        rbf_acc.append(accuracy_score(y[te], p_svm))
        
        # RF
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_raw[tr], y[tr])
        p_rf = rf.predict(X_raw[te])
        rf_f1.append(f1_score(y[te], p_rf, average="macro"))
        rf_acc.append(accuracy_score(y[te], p_rf))
        
    return {
        "RBF-SVM": (np.mean(rbf_f1), np.mean(rbf_acc)),
        "RandomForest": (np.mean(rf_f1), np.mean(rf_acc))
    }

def process_dataset(ds_name, load_fn, cache_bscm, cache_srqfm, cache_standard):
    print(f"\n{'='*50}\nDATASET: {ds_name.upper()}\n{'='*50}")
    
    # 1. Classical
    _, X_raw, y = load_fn()
    c_res = evaluate_classical(X_raw, y, ds_name)
    
    res = {
        "RBF-SVM": c_res["RBF-SVM"],
        "RandomForest": c_res["RandomForest"]
    }
    
    # 2. BSCM-PQK
    if os.path.exists(cache_bscm):
        d = np.load(cache_bscm)
        res["BSCM-PQK"] = evaluate_precomputed(d["K"], d["y"], f"{ds_name} BSCM-PQK")
    else:
        print(f"[{ds_name}] BSCM-PQK cache not found.")
        
    # 3. SRQFM-PQK
    if os.path.exists(cache_srqfm):
        d = np.load(cache_srqfm)
        res["SRQFM-PQK"] = evaluate_precomputed(d["K"], d["y"], f"{ds_name} SRQFM-PQK")
    else:
        print(f"[{ds_name}] SRQFM-PQK cache not found.")
        
    # 4. Standard PQK
    if os.path.exists(cache_standard):
        d = np.load(cache_standard)
        res["Standard-PQK"] = evaluate_precomputed(d["K"], d["y"], f"{ds_name} Standard-PQK")
    else:
        print(f"[{ds_name}] Standard-PQK cache not found.")
        
    print("\nRESULTS TABLE:")
    print(f"{'Method':<20} | {'Macro-F1':<10} | {'Accuracy':<10}")
    print("-" * 46)
    for k, v in res.items():
        print(f"{k:<20} | {v[0]:.4f}     | {v[1]:.4f}")

if __name__ == "__main__":
    process_dataset("So2Sat", _load_so2sat_max, os.path.join(RESULTS_DIR, "cache_so2sat.npz"), os.path.join(RESULTS_DIR, "cache_so2sat_srqfm.npz"), os.path.join(RESULTS_DIR, "cache_so2sat_standard_pqk.npz"))
    process_dataset("EuroSAT", _load_eurosat_10k, os.path.join(RESULTS_DIR, "cache_eurosat.npz"), os.path.join(RESULTS_DIR, "cache_eurosat_srqfm.npz"), os.path.join(RESULTS_DIR, "cache_eurosat_standard_pqk.npz"))
