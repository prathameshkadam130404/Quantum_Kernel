"""
Experiment E2 — AGPQK 4-component ablation.

Four conditions on physics-16 features, N=2000, 5 seeds, all cross-modal:
    AGPQK-Base     : Fisher select      + uniform γ=0.5 + linear entanglement
    AGPQK-BW       : Fisher select      + per-qubit γ   + linear entanglement
    AGPQK-Attn     : Fisher select      + uniform γ=0.5 + attention pairs
    AGPQK-Full     : Fisher select      + per-qubit γ   + attention pairs

For each: Macro-F1 mean±std, ΔKTA, effective rank, off-diagonal variance.
Statistical tests: paired t-test + Cohen's d + Holm–Bonferroni across the
6 pairwise contrasts.

Output:  results/ablation_agpqk/{K_*.npy, metrics.csv, pvals.json}
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import ttest_rel

import config
from src.utils import ensure_dir, setup_logging
from src.attention_kernel import (
    compute_attention_matrix, select_features_by_fisher,
    compute_per_qubit_gamma, get_attention_entanglement_pairs,
)
from src.kernel_target_alignment import compute_centered_kta
from experiments._e_common import (
    load_physics_16, spectral_metrics, svm_eval, cohens_d, holm_bonferroni,
    build_agpqk_kernel,
)

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "ablation_agpqk"))
logger = setup_logging("exp_e2", log_file=os.path.join(RES, "e2.log"))

N_QUBITS, REPS = config.N_QUBITS, config.ZZ_REPS
N_SELECT       = 8
GAMMA_UNIFORM  = 0.5
TOP_K_PAIRS    = 4


def linear_pairs(n):
    return [(i, i + 1) for i in range(n - 1)]


def run():
    X_norm, X_raw, y = load_physics_16()
    A = compute_attention_matrix(X_raw)
    selected, fisher = select_features_by_fisher(X_raw, y, n_select=N_SELECT,
                                                 min_sar=4, min_opt=4)
    gamma_perq = compute_per_qubit_gamma(fisher, selected, gamma_base=GAMMA_UNIFORM)
    gamma_uni  = np.full(N_SELECT, GAMMA_UNIFORM)
    attn_pairs = get_attention_entanglement_pairs(
        A, selected, top_k=TOP_K_PAIRS, require_cross_modal=True,
        max_appearances=1)
    lin_pairs  = linear_pairs(N_SELECT)

    conditions = {
        "AGPQK-Base": dict(gamma=gamma_uni,  pairs=lin_pairs),
        "AGPQK-BW":   dict(gamma=gamma_perq, pairs=lin_pairs),
        "AGPQK-Attn": dict(gamma=gamma_uni,  pairs=attn_pairs),
        "AGPQK-Full": dict(gamma=gamma_perq, pairs=attn_pairs),
    }

    kernels = {}
    for name, cfg in conditions.items():
        path = os.path.join(RES, f"K_{name}.npy")
        if os.path.exists(path):
            logger.info(f"[cache] {path}")
            kernels[name] = np.load(path)
            continue
        logger.info(f"Computing {name} …")
        K, _ = build_agpqk_kernel(X_raw, y, selected, cfg["gamma"],
                                  cfg["pairs"], N_QUBITS, REPS,
                                  outer_gamma=config.PQK_GAMMA)
        np.save(path, K)
        kernels[name] = K

    records = []
    f1_by_cond = {k: [] for k in conditions}
    for name, K in kernels.items():
        spm = spectral_metrics(K)
        kta = compute_centered_kta(K, y, class_weighted=True)
        for seed in config.SEED_LIST:
            ev = svm_eval(K, y, seed)
            f1_by_cond[name].append(ev["macro_f1"])
            records.append(dict(condition=name, seed=seed,
                                macro_f1=ev["macro_f1"],
                                accuracy=ev["accuracy"],
                                kta=float(kta),
                                **spm))

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(RES, "metrics.csv"), index=False)

    conds = list(conditions.keys())
    pvals, effsz, tests = [], [], []
    for a, b in combinations(conds, 2):
        t, p = ttest_rel(f1_by_cond[a], f1_by_cond[b])
        d = cohens_d(f1_by_cond[a], f1_by_cond[b])
        pvals.append(float(p)); effsz.append(d)
        tests.append(f"{a} vs {b}")
    sig = holm_bonferroni(pvals, alpha=0.05).tolist()
    stats = [dict(contrast=t, p=p, cohen_d=d, holm_sig=s)
             for t, p, d, s in zip(tests, pvals, effsz, sig)]
    with open(os.path.join(RES, "pvals.json"), "w") as f:
        json.dump({"tests": stats,
                   "attention_pairs": [list(p) for p in attn_pairs],
                   "selected_features": selected.tolist(),
                   "gamma_per_qubit": gamma_perq.tolist()}, f, indent=2)

    logger.info("Means (Macro-F1):")
    for c in conds:
        logger.info(f"  {c:<12} {np.mean(f1_by_cond[c]):.4f} "
                    f"± {np.std(f1_by_cond[c]):.4f}")


if __name__ == "__main__":
    run()
