"""
build_e38_summary.py
=====================
Consolidates the four E38-family result JSONs into a single auditable file
matching the structure of paper Table 11 (E38 maximum-capacity benchmark).

Inputs (under results/):
  * e38_max_data/baselines_summary.json         -- SRQFM, RBF-SVM, RF
  * e38_max_data/standard_pqk_summary.json      -- Standard ZZ-PQK
  * e38_max_data/max_summary_bscm_sectors.json  -- BSCM uniform/phi/psi
  * e47_xgboost/xgb_baseline.json               -- XGBoost

Output:
  * results/e38_max_data/e38_combined_summary.json

Usage:
    python experiments/build_e38_summary.py

Run this whenever any of the four input JSONs change.  Note: this script
emits f1_mean / f1_std cells only.  Holm-corrected Wilcoxon p-values for
Table 11 come from experiments/audit_e38_significance.py, which reads the
cached Gram matrices directly.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

E38_DIR = os.path.join(config.RESULTS_DIR, "e38_max_data")
XGB_PATH = os.path.join(config.RESULTS_DIR, "e47_xgboost", "xgb_baseline.json")
OUT_PATH = os.path.join(E38_DIR, "e38_combined_summary.json")


def _safe_load(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[warn] missing: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def main() -> None:
    baselines = _safe_load(os.path.join(E38_DIR, "baselines_summary.json"))
    standard = _safe_load(os.path.join(E38_DIR, "standard_pqk_summary.json"))
    bscm = _safe_load(os.path.join(E38_DIR, "max_summary_bscm_sectors.json"))
    xgb = _safe_load(XGB_PATH)

    def cell(mean, std):
        if mean is None:
            return None
        return {"f1_mean": float(mean), "f1_std": float(std)}

    out = {
        "depth": 6,
        "n_qubits": 16,
        "n_samples": {"so2sat": 10000, "eurosat": 10000},
        "tau_locked": 0.25,
        "n_splits": baselines.get("so2sat", {}).get("n_splits"),
        "so2sat": {
            "SRQFM-PQK": cell(
                baselines.get("so2sat", {}).get("srqfm_f1_mean"),
                baselines.get("so2sat", {}).get("srqfm_f1_std"),
            ),
            "RBF-SVM": cell(
                baselines.get("so2sat", {}).get("rbf_f1_mean"),
                baselines.get("so2sat", {}).get("rbf_f1_std"),
            ),
            "RandomForest": cell(
                baselines.get("so2sat", {}).get("rf_f1_mean"),
                baselines.get("so2sat", {}).get("rf_f1_std"),
            ),
            "Standard-ZZ-PQK": cell(
                standard.get("so2sat", {}).get("standard_pqk_f1_mean"),
                standard.get("so2sat", {}).get("standard_pqk_f1_std"),
            ),
            "BSCM-uniform-PQK": cell(
                bscm.get("so2sat_bscm_uniform_f1_mean"),
                bscm.get("so2sat_bscm_uniform_f1_std"),
            ),
            "BSCM-phi-PQK": cell(
                bscm.get("so2sat_bscm_phi_only_f1_mean"),
                bscm.get("so2sat_bscm_phi_only_f1_std"),
            ),
            "BSCM-psi-PQK": cell(
                bscm.get("so2sat_bscm_psi_only_f1_mean"),
                bscm.get("so2sat_bscm_psi_only_f1_std"),
            ),
            "XGBoost": cell(
                xgb.get("so2sat", {}).get("macro_f1_mean"),
                xgb.get("so2sat", {}).get("macro_f1_std"),
            ),
        },
        "eurosat": {
            "SRQFM-PQK": cell(
                baselines.get("eurosat", {}).get("srqfm_f1_mean"),
                baselines.get("eurosat", {}).get("srqfm_f1_std"),
            ),
            "RBF-SVM": cell(
                baselines.get("eurosat", {}).get("rbf_f1_mean"),
                baselines.get("eurosat", {}).get("rbf_f1_std"),
            ),
            "RandomForest": cell(
                baselines.get("eurosat", {}).get("rf_f1_mean"),
                baselines.get("eurosat", {}).get("rf_f1_std"),
            ),
            "Standard-ZZ-PQK": cell(
                standard.get("eurosat", {}).get("standard_pqk_f1_mean"),
                standard.get("eurosat", {}).get("standard_pqk_f1_std"),
            ),
            "BSCM-uniform-PQK": cell(
                bscm.get("eurosat_bscm_uniform_f1_mean"),
                bscm.get("eurosat_bscm_uniform_f1_std"),
            ),
            "BSCM-phi-PQK": cell(
                bscm.get("eurosat_bscm_phi_only_f1_mean"),
                bscm.get("eurosat_bscm_phi_only_f1_std"),
            ),
            "BSCM-psi-PQK": cell(
                bscm.get("eurosat_bscm_psi_only_f1_mean"),
                bscm.get("eurosat_bscm_psi_only_f1_std"),
            ),
            "XGBoost": cell(
                xgb.get("eurosat", {}).get("macro_f1_mean"),
                xgb.get("eurosat", {}).get("macro_f1_std"),
            ),
        },
    }

    # Architectural gap rows -- the headline claim in the abstract.
    for dset in ("so2sat", "eurosat"):
        zz = out[dset]["Standard-ZZ-PQK"]
        if not zz:
            continue
        zz_mean = zz["f1_mean"]
        out[dset]["delta_over_ZZ"] = {
            k: round(v["f1_mean"] - zz_mean, 4)
            for k, v in out[dset].items()
            if isinstance(v, dict) and "f1_mean" in v and k != "Standard-ZZ-PQK"
        }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")

    # Console summary so the user can eyeball it.
    print("\nE38 Table 11 contents (macro-F1, mean +/- std):")
    print(f"{'method':<22s} {'so2sat':>20s} {'eurosat':>20s}")
    for method in [
        "SRQFM-PQK",
        "BSCM-uniform-PQK",
        "BSCM-phi-PQK",
        "BSCM-psi-PQK",
        "Standard-ZZ-PQK",
        "RBF-SVM",
        "RandomForest",
        "XGBoost",
    ]:
        s = out["so2sat"].get(method)
        e = out["eurosat"].get(method)
        s_str = f"{s['f1_mean']:.4f} +/- {s['f1_std']:.4f}" if s else "---"
        e_str = f"{e['f1_mean']:.4f} +/- {e['f1_std']:.4f}" if e else "---"
        print(f"{method:<22s} {s_str:>20s} {e_str:>20s}")
    print("\nDelta over Standard-ZZ-PQK (positive = BSCM family beats vanilla ZZ):")
    print(f"  so2sat : {out['so2sat'].get('delta_over_ZZ', {})}")
    print(f"  eurosat: {out['eurosat'].get('delta_over_ZZ', {})}")


if __name__ == "__main__":
    main()
