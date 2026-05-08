"""
Quick in-depth evaluation of the already-computed BSCM-uniform kernel.

Loads results/bscm/K_bscm_physics8_uniform.npy (1800x1800) and runs:

  1. Spectral analysis  -- off-diag stats, effective rank (Shannon + Huang),
                          eigenvalue distribution plot
  2. KTA (kernel-target alignment) on the full 1800-sample eval pool
  3. SVM classification -- C ∈ {1, 10, 100}, 3-fold CV, 5 seeds
                          macro F1 mean ± std, per-class F1, confusion matrix
  4. Comparison table  -- vs locked_tau.json baselines (SRQFM-fid KTA) and
                          the expressibility.json KL record
  5. Per-class analysis -- identifies which LCZ classes benefit / struggle
  6. Concentration test -- histogram of off-diagonal kernel values
                          (exponential concentration check for 8-qubit kernel)

No quantum circuits are run — pure numpy/sklearn post-processing.

Output files:
  results/bscm/eval_bscm_uniform/report.txt
  results/bscm/eval_bscm_uniform/eigenvalues.npy
  results/bscm/eval_bscm_uniform/confusion_matrix.npy
  results/bscm/eval_bscm_uniform/per_class_f1.csv
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.svm import SVC

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from experiments._bscm_split import make_or_load_split
from experiments._e_common import cohens_d, load_physics_16, spectral_metrics
from src.attention_kernel import select_features_by_fisher

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SAVE_DIR = os.path.join(config.RESULTS_DIR, "bscm", "eval_bscm_uniform")
os.makedirs(SAVE_DIR, exist_ok=True)

KERNEL_PATH = os.path.join(config.RESULTS_DIR, "bscm", "K_bscm_physics8_uniform.npy")
LOCKED_TAU_PATH = os.path.join(config.RESULTS_DIR, "bscm", "locked_tau.json")
EXPRESSIBILITY_PATH = os.path.join(config.RESULTS_DIR, "bscm", "expressibility.json")

SEEDS = config.SEED_LIST          # [42, 43, 44, 45, 46]
C_GRID = [1, 10, 100]
CV_FOLDS = 3
TEST_FRAC = 0.30

LCZ_CLASS_NAMES = config.LCZ_CLASS_NAMES  # 17 names

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(SAVE_DIR, "eval.log"), mode="w"),
    ],
)
log = logging.getLogger("eval_bscm_uniform")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def tune_c(K_train: np.ndarray, y_train: np.ndarray, seed: int) -> int:
    best_c, best_f1 = C_GRID[0], -1.0
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    for C in C_GRID:
        folds = []
        for cv_tr, cv_val in skf.split(np.zeros(len(y_train)), y_train):
            K_cv = K_train[np.ix_(cv_tr, cv_tr)]
            K_cv_v = K_train[np.ix_(cv_val, cv_tr)]
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_cv, y_train[cv_tr])
            folds.append(f1_score(y_train[cv_val], clf.predict(K_cv_v),
                                   average="macro", zero_division=0))
        m = float(np.mean(folds))
        if m > best_f1:
            best_f1, best_c = m, C
    return best_c


def eval_one_seed(K_full: np.ndarray, y: np.ndarray, seed: int) -> dict:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=seed)
    (tr, te), = sss.split(np.zeros(len(y)), y)
    K_tr = K_full[np.ix_(tr, tr)]
    K_te = K_full[np.ix_(te, tr)]
    best_c = tune_c(K_tr, y[tr], seed)
    clf = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    clf.fit(K_tr, y[tr])
    yp = clf.predict(K_te)
    return {
        "seed": seed,
        "best_C": best_c,
        "macro_f1": float(f1_score(y[te], yp, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y[te], yp)),
        "y_test": y[te],
        "y_pred": yp,
        "per_class_f1": f1_score(y[te], yp, average=None, zero_division=0),
    }


def kta(K: np.ndarray, y: np.ndarray) -> float:
    n = len(y)
    Y = np.outer(y, y)
    Y = (Y > 0).astype(float) * 2.0 - 1.0   # ±1 label kernel
    Ksym = (K + K.T) / 2.0
    kta_num = float(np.sum(Ksym * Y)) / (n * n)
    kta_den = np.sqrt(float(np.sum(Ksym * Ksym)) * float(np.sum(Y * Y))) / (n * n)
    return kta_num / (kta_den + 1e-12)


def _sep(char="=", width=72):
    return char * width


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    log.info(_sep())
    log.info("  BSCM-uniform (physics8) — In-Depth Post-Hoc Evaluation")
    log.info("  Kernel: %s", KERNEL_PATH)
    log.info("  Date  : %s", datetime.now().isoformat())
    log.info(_sep())

    # ------------------------------------------------------------------ #
    # 1. Load kernel & labels
    # ------------------------------------------------------------------ #
    log.info("\n[1] Loading kernel matrix...")
    K = np.load(KERNEL_PATH)
    log.info("    Shape : %s", K.shape)
    log.info("    dtype : %s", K.dtype)
    log.info("    min/max diagonal: %.6f / %.6f", K.diagonal().min(), K.diagonal().max())
    log.info("    Expected diagonal ≈ 1.0 for a fidelity kernel")

    diag_min = K.diagonal().min()
    diag_max = K.diagonal().max()
    diag_ok = (diag_min > 0.99 and diag_max <= 1.001)
    log.info("    Diagonal health check: %s", "PASS" if diag_ok else "WARNING – diagonal ≠ 1")

    # Symmetry check
    asym = float(np.max(np.abs(K - K.T)))
    log.info("    Max |K - Kᵀ| : %.2e  (%s)", asym,
             "symmetric" if asym < 1e-6 else "WARNING: asymmetric")

    # Load labels
    log.info("\n[2] Loading labels and building eval-pool indices...")
    _, X_raw, y_full = load_physics_16()           # (2000, 16), labels
    from src.attention_kernel import select_features_by_fisher, FEATURE_NAMES_16
    sel_idx, fisher_scores = select_features_by_fisher(X_raw, y_full)
    sel_names = [FEATURE_NAMES_16[i] for i in sel_idx]
    log.info("    Fisher-selected features: %s", sel_names)

    sp = make_or_load_split(y_full)
    y = y_full[sp.eval_pool]                       # (1800,) labels
    log.info("    Eval pool size : %d  (expected 1800)", len(y))
    log.info("    Class distribution (17 LCZ classes):")
    classes, counts = np.unique(y, return_counts=True)
    for c, n in zip(classes, counts):
        cname = LCZ_CLASS_NAMES[c] if c < len(LCZ_CLASS_NAMES) else f"cls{c}"
        log.info("      [%2d] %-28s  %4d  (%.1f%%)", c, cname, n, 100*n/len(y))

    assert K.shape == (len(y), len(y)), (
        f"Kernel shape {K.shape} does not match eval pool size {len(y)}"
    )

    # ------------------------------------------------------------------ #
    # 2. Spectral / structural analysis
    # ------------------------------------------------------------------ #
    log.info("\n[3] Spectral analysis...")
    sm = spectral_metrics(K)
    Ksym = (K + K.T) / 2.0
    eig = np.linalg.eigvalsh(Ksym)

    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off = K[mask]

    log.info("    Off-diagonal mean  : %.6f", sm["off_diag_mean"])
    log.info("    Off-diagonal var   : %.6f", sm["off_diag_var"])
    log.info("    Off-diagonal std   : %.6f", np.sqrt(sm["off_diag_var"]))
    log.info("    Effective rank (Shannon): %.1f  / %d", sm["eff_rank_shannon"], n)
    log.info("    Effective rank (Huang)  : %.1f  / %d", sm["eff_rank_huang"], n)

    # Compare to locked_tau.json reference (phase 2 measurements at N=200)
    if os.path.exists(LOCKED_TAU_PATH):
        with open(LOCKED_TAU_PATH) as f:
            lt = json.load(f)
        ref_tau025 = next((r for r in lt["sweep"] if r["tau"] == 0.25), None)
        if ref_tau025:
            log.info("    Phase-2 reference (N=200, tau=0.25):")
            log.info("      off_diag_mean : %.6f  (now N=1800: %.6f)",
                     ref_tau025["off_diag_mean"], sm["off_diag_mean"])
            log.info("      off_diag_var  : %.6f  (now N=1800: %.6f)",
                     ref_tau025["off_diag_var"], sm["off_diag_var"])
            log.info("      eff_rank_shannon: %.1f  (now N=1800: %.1f)",
                     ref_tau025["eff_rank_shannon"], sm["eff_rank_shannon"])

    # Eigenvalue stats
    eig_pos = eig[eig > 0]
    log.info("    Eigenvalues — total: %d  positive: %d  near-zero/negative: %d",
             len(eig), len(eig_pos), len(eig) - len(eig_pos))
    log.info("    Eigenvalue range   : [%.4e, %.4e]", eig.min(), eig.max())
    log.info("    Top-5 eigenvalues  : %s",
             np.array2string(np.sort(eig)[::-1][:5], precision=3))

    # Concentration check: fraction of off-diag values within 1σ of mean
    mean_off = off.mean()
    std_off = off.std()
    within_1sig = float(np.mean(np.abs(off - mean_off) < std_off))
    log.info("    Off-diag concentration: %.1f%% within 1σ of mean "
             "(>95%% → exponential concentration)", 100 * within_1sig)
    if within_1sig > 0.90:
        log.info("    WARNING: High concentration — kernel near constant → "
                 "SVM sees near-degenerate Gram matrix at this N")

    np.save(os.path.join(SAVE_DIR, "eigenvalues.npy"), eig)

    # ------------------------------------------------------------------ #
    # 3. KTA
    # ------------------------------------------------------------------ #
    log.info("\n[4] Kernel-Target Alignment (KTA)...")
    kta_val = kta(K, y)
    log.info("    KTA (BSCM-uniform, N=1800) : %.6f", kta_val)
    if os.path.exists(LOCKED_TAU_PATH):
        with open(LOCKED_TAU_PATH) as f:
            lt = json.load(f)
        log.info("    Phase-2 KTA (N=200, tau=0.25): %.6f  [locked by holdout pool]",
                 lt["locked"]["kta"])
        log.info("    SRQFM-fid KTA (N=200)        : %.6f",
                 lt["srqfm_fid_baseline"]["kta"])
        log.info("    Note: KTA degrades with N (larger kernel → diluted alignment)")

    # ------------------------------------------------------------------ #
    # 4. SVM evaluation — 5 seeds, C-tuning
    # ------------------------------------------------------------------ #
    log.info("\n[5] SVM classification (5 seeds × C-tuning {1,10,100}, 3-fold CV)...")
    seed_results = []
    all_y_test, all_y_pred = [], []

    for seed in SEEDS:
        r = eval_one_seed(K, y, seed)
        seed_results.append(r)
        all_y_test.append(r["y_test"])
        all_y_pred.append(r["y_pred"])
        log.info("    Seed %d: macro_F1=%.4f  acc=%.4f  best_C=%d",
                 seed, r["macro_f1"], r["accuracy"], r["best_C"])

    f1_vals = [r["macro_f1"] for r in seed_results]
    acc_vals = [r["accuracy"] for r in seed_results]
    log.info("\n    ── Summary ──────────────────────────────────────────")
    log.info("    Macro F1 : %.4f ± %.4f  (min=%.4f  max=%.4f)",
             np.mean(f1_vals), np.std(f1_vals, ddof=1), min(f1_vals), max(f1_vals))
    log.info("    Accuracy : %.4f ± %.4f",
             np.mean(acc_vals), np.std(acc_vals, ddof=1))
    log.info("    C choices: %s", [r["best_C"] for r in seed_results])

    # ------------------------------------------------------------------ #
    # 5. Per-class F1
    # ------------------------------------------------------------------ #
    log.info("\n[6] Per-class F1 (averaged over 5 seeds)...")
    per_class_stacked = np.stack([r["per_class_f1"] for r in seed_results
                                   if r["per_class_f1"] is not None], axis=0)
    mean_pc_f1 = per_class_stacked.mean(axis=0)
    std_pc_f1 = per_class_stacked.std(axis=0, ddof=1)

    pc_rows = []
    log.info("    %4s  %-28s  %7s  %7s  %7s", "Cls", "Name", "Mean F1", "Std", "Count")
    log.info("    " + "-" * 58)
    for c in range(len(mean_pc_f1)):
        cname = LCZ_CLASS_NAMES[c] if c < len(LCZ_CLASS_NAMES) else f"cls{c}"
        cnt = int(np.sum(y == c))
        log.info("    %4d  %-28s  %.4f   %.4f   %5d",
                 c, cname, mean_pc_f1[c], std_pc_f1[c], cnt)
        pc_rows.append({"class": c, "name": cname,
                         "mean_f1": mean_pc_f1[c], "std_f1": std_pc_f1[c],
                         "count": cnt})

    pc_df = pd.DataFrame(pc_rows)
    pc_df.to_csv(os.path.join(SAVE_DIR, "per_class_f1.csv"), index=False)
    log.info("    Saved: %s/per_class_f1.csv", SAVE_DIR)

    # Highlight best/worst
    best_cls = int(mean_pc_f1.argmax())
    worst_cls = int(mean_pc_f1.argmin())
    log.info("    Best  class: [%d] %s — F1=%.4f",
             best_cls, LCZ_CLASS_NAMES[best_cls] if best_cls < 17 else f"cls{best_cls}",
             mean_pc_f1[best_cls])
    log.info("    Worst class: [%d] %s — F1=%.4f",
             worst_cls, LCZ_CLASS_NAMES[worst_cls] if worst_cls < 17 else f"cls{worst_cls}",
             mean_pc_f1[worst_cls])

    # ------------------------------------------------------------------ #
    # 6. Pooled confusion matrix (seed 42)
    # ------------------------------------------------------------------ #
    log.info("\n[7] Confusion matrix (seed=42)...")
    r0 = seed_results[0]
    cm = confusion_matrix(r0["y_test"], r0["y_pred"])
    np.save(os.path.join(SAVE_DIR, "confusion_matrix.npy"), cm)
    log.info("    Shape: %s   Saved.", cm.shape)

    # ------------------------------------------------------------------ #
    # 7. Expressibility context
    # ------------------------------------------------------------------ #
    if os.path.exists(EXPRESSIBILITY_PATH):
        with open(EXPRESSIBILITY_PATH) as f:
            ex = json.load(f)
        log.info("\n[8] Expressibility context (from bscm_expressibility.py):")
        for rec in ex["records"]:
            log.info("    %-40s  KL=%.4f  fid_mean=%.4f",
                     rec["name"], rec["kl_to_haar"], rec["fidelity_mean"])
        log.info("    Interpretation: KL=0 → Haar-random; higher KL → more structured")

    # ------------------------------------------------------------------ #
    # 8. Concentration: histogram of off-diagonal values
    # ------------------------------------------------------------------ #
    log.info("\n[9] Off-diagonal distribution (5-bin histogram)...")
    hist, edges = np.histogram(off, bins=20, range=(0, 1))
    for i in range(0, 20, 4):
        lo, hi = edges[i], edges[i+4]
        count = hist[i:i+4].sum()
        bar = "█" * (count // max(1, hist.max() // 20))
        log.info("    [%.2f-%.2f]  %7d  %s", lo, hi, count, bar)

    # ------------------------------------------------------------------ #
    # 9. Final summary table
    # ------------------------------------------------------------------ #
    log.info("\n" + _sep())
    log.info("  FINAL SUMMARY")
    log.info(_sep())
    log.info("  Kernel             : BSCM-uniform, physics8, tau=0.25, N=1800")
    log.info("  Macro F1           : %.4f ± %.4f", np.mean(f1_vals), np.std(f1_vals, ddof=1))
    log.info("  Accuracy           : %.4f ± %.4f", np.mean(acc_vals), np.std(acc_vals, ddof=1))
    log.info("  KTA (N=1800)       : %.4f", kta_val)
    log.info("  Eff rank (Shannon) : %.1f / 1800", sm["eff_rank_shannon"])
    log.info("  Off-diag mean/std  : %.4f / %.4f", sm["off_diag_mean"], np.sqrt(sm["off_diag_var"]))
    log.info("  Diag range         : %.4f – %.4f", diag_min, diag_max)
    log.info(_sep())

    # ------------------------------------------------------------------ #
    # 10. Save JSON summary
    # ------------------------------------------------------------------ #
    summary = {
        "kernel": "BSCM-uniform",
        "feature_set": "physics8",
        "tau": 0.25,
        "n_eval": int(len(y)),
        "macro_f1_mean": float(np.mean(f1_vals)),
        "macro_f1_std": float(np.std(f1_vals, ddof=1)),
        "accuracy_mean": float(np.mean(acc_vals)),
        "accuracy_std": float(np.std(acc_vals, ddof=1)),
        "kta": float(kta_val),
        "spectral": {k: float(v) for k, v in sm.items()},
        "off_diag_concentration_within_1sigma": float(within_1sig),
        "c_choices": [r["best_C"] for r in seed_results],
        "per_seed": [
            {"seed": r["seed"], "macro_f1": r["macro_f1"],
             "accuracy": r["accuracy"], "best_C": r["best_C"]}
            for r in seed_results
        ],
        "eigenvalue_stats": {
            "min": float(eig.min()),
            "max": float(eig.max()),
            "n_positive": int(len(eig_pos)),
            "n_nonpositive": int(len(eig) - len(eig_pos)),
        },
    }

    summary_path = os.path.join(SAVE_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("  Results dir : %s", SAVE_DIR)
    log.info("  summary.json: %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
