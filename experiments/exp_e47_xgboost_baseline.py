"""
E47: XGBoost classical baseline on the E38 So2Sat / EuroSAT pools.

Same data pool, same Fisher-16 features, same random_state=42, same
test_size=0.3, and same 10 stratified shuffle splits as
exp_e38_max_data_baselines.py, so the XGBoost row of Table 9 is on
the identical per-fold partition as every other row in that table.

XGBoost: 500 trees, balanced class weights via XGB sample_weight,
default learning rate.

This script does NOT require any quantum simulation.
"""
import os, sys, json, logging, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from experiments.exp_e38_max_data_pqk import _load_so2sat_max, _load_eurosat_10k

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s:   %(message)s")
log = logging.getLogger("e47_xgboost")

OUT_DIR = os.path.join(config.RESULTS_DIR, "e47_xgboost")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
N_SPLITS = 10
TEST_FRAC = 0.3


def evaluate_xgb(X_raw: np.ndarray, y: np.ndarray, name: str) -> dict:
    sss = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=TEST_FRAC, random_state=SEED)
    scores = []
    for tr, te in sss.split(X_raw, y):
        clf = XGBClassifier(
            n_estimators=500,
            scale_pos_weight=None,  # handled by class_weight below
            random_state=SEED,
            n_jobs=-1,
            verbosity=0,
        )
        clf.set_params(eval_metric="mlogloss")
        # balanced class weights: XGBoost uses scale_pos_weight for binary only,
        # so we compute sample weights inversely proportional to class frequency
        classes, counts = np.unique(y[tr], return_counts=True)
        weight_map = {c: len(y[tr]) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        sample_weight = np.array([weight_map[yi] for yi in y[tr]])
        clf.fit(X_raw[tr], y[tr], sample_weight=sample_weight)
        yp = clf.predict(X_raw[te])
        scores.append(float(f1_score(y[te], yp, average="macro")))
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores,
        "n_splits": int(N_SPLITS),
        "test_fraction": float(TEST_FRAC),
        "seed": int(SEED),
    }


def main():
    log.info("E47 -- XGBoost Baseline Evaluation")
    results = {}

    # ---- So2Sat ----
    log.info("Loading So2Sat (N=10,000) ...")
    _, X_raw_s, y_s = _load_so2sat_max()
    log.info(f"So2Sat: X_raw.shape={X_raw_s.shape}, y.shape={y_s.shape}")
    r_s = evaluate_xgb(X_raw_s, y_s, "So2Sat")
    log.info(f"==> XGBoost So2Sat macro-F1: {r_s['mean']:.4f} +/- {r_s['std']:.4f}")
    results["so2sat"] = {
        "n_samples": int(len(y_s)),
        "macro_f1_mean": r_s["mean"],
        "macro_f1_std": r_s["std"],
        "per_split_f1": r_s["scores"],
        "n_splits": r_s["n_splits"],
        "test_fraction": r_s["test_fraction"],
        "seed": r_s["seed"],
    }

    # ---- EuroSAT ----
    log.info("Loading EuroSAT (N=10,000) ...")
    _, X_raw_e, y_e = _load_eurosat_10k()
    log.info(f"EuroSAT: X_raw.shape={X_raw_e.shape}, y.shape={y_e.shape}")
    r_e = evaluate_xgb(X_raw_e, y_e, "EuroSAT")
    log.info(f"==> XGBoost EuroSAT macro-F1: {r_e['mean']:.4f} +/- {r_e['std']:.4f}")
    results["eurosat"] = {
        "n_samples": int(len(y_e)),
        "macro_f1_mean": r_e["mean"],
        "macro_f1_std": r_e["std"],
        "per_split_f1": r_e["scores"],
        "n_splits": r_e["n_splits"],
        "test_fraction": r_e["test_fraction"],
        "seed": r_e["seed"],
    }

    # Save
    out_path = os.path.join(OUT_DIR, "xgb_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
