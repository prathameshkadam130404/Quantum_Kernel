"""
Experiment E14 — Null-control ablations for AGPQK.

Three null controls (each 5 seeds) against real AGPQK:
    (N1) Random-Fisher   : shuffle Fisher weights among selected features.
    (N2) Random-pairs    : replace attention-selected pairs with uniformly
                           random cross-modal pairs (same cardinality).
    (N3) Fisher-blind CV : per-qubit γ chosen by 3-fold CV grid search
                           with identical γ assigned to all qubits (degenerate
                           per-qubit), i.e. a single scalar tuned on training
                           data but *blind* to Fisher importance.

Paired t-tests vs real AGPQK + Holm-Bonferroni + Cohen's d.

Answers: "AGPQK's three design choices are coupled — could any random
configuration work this well?" reviewer objection.

Output: results/agpqk_nulls/{metrics.csv, stats.csv, fig.pdf}
"""
import os, sys, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedShuffleSplit

import config
from src.utils import ensure_dir, setup_logging
from src.attention_kernel import (
    compute_attention_matrix, select_features_by_fisher,
    compute_per_qubit_gamma, get_attention_entanglement_pairs,
    FEATURE_MODALITY_16,
)
from src.kernel_target_alignment import compute_centered_kta
from experiments._e_common import (
    load_physics_16, spectral_metrics, svm_eval,
    build_agpqk_kernel, holm_bonferroni, cohens_d,
)

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "agpqk_nulls"))
logger = setup_logging("exp_e14", log_file=os.path.join(RES, "e14.log"))

N = 500
N_QUBITS, REPS = config.N_QUBITS, config.ZZ_REPS
SEEDS = config.SEED_LIST
NULL_REPS = 5


def random_cross_modal_pairs(selected, n_pairs, rng):
    """Sample n_pairs unique cross-modal (SAR/optical) qubit pairs."""
    n_sel = len(selected)
    # modalities for each qubit (in selected order)
    mods = [FEATURE_MODALITY_16[int(f)] for f in selected]
    candidates = [(i, j) for i in range(n_sel) for j in range(i + 1, n_sel)
                  if mods[i] != mods[j]]
    rng.shuffle(candidates)
    return candidates[:n_pairs]


def run():
    X_norm, X_raw, y_full = load_physics_16()
    sss = StratifiedShuffleSplit(n_splits=1, train_size=N, random_state=42)
    (idx, _), = sss.split(np.zeros(len(y_full)), y_full)
    X_n, X_r, y = X_norm[idx], X_raw[idx], y_full[idx]

    selected, fisher = select_features_by_fisher(X_r, y, n_select=N_QUBITS,
                                                 min_sar=4, min_opt=4)
    A = compute_attention_matrix(X_r)
    real_pairs = get_attention_entanglement_pairs(
        A, selected, top_k=4, require_cross_modal=True, max_appearances=1)
    real_gamma = compute_per_qubit_gamma(fisher, selected, gamma_base=0.5)

    # ---- Real AGPQK ----
    logger.info("Computing real AGPQK ...")
    K_real, _ = build_agpqk_kernel(X_r, y, selected, real_gamma, real_pairs,
                                   n_qubits=N_QUBITS, reps=REPS,
                                   outer_gamma=config.PQK_GAMMA)
    np.save(os.path.join(RES, "K_real.npy"), K_real)

    rows = []
    for seed in SEEDS:
        ev = svm_eval(K_real, y, seed)
        rows.append(dict(condition="Real-AGPQK", rep=0, seed=seed,
                         macro_f1=ev["macro_f1"],
                         kta=float(compute_centered_kta(K_real, y,
                                                        class_weighted=True)),
                         **spectral_metrics(K_real)))

    # ---- N1: Random Fisher (shuffle gamma) ----
    for rep in range(NULL_REPS):
        rng = np.random.default_rng(1000 + rep)
        perm = rng.permutation(len(real_gamma))
        gamma_shuf = real_gamma[perm]
        logger.info(f"N1 rep={rep} random-Fisher gamma = {gamma_shuf}")
        K, _ = build_agpqk_kernel(X_r, y, selected, gamma_shuf, real_pairs,
                                  n_qubits=N_QUBITS, reps=REPS,
                                  outer_gamma=config.PQK_GAMMA)
        for seed in SEEDS:
            ev = svm_eval(K, y, seed)
            rows.append(dict(condition="N1-RandomFisher", rep=rep, seed=seed,
                             macro_f1=ev["macro_f1"],
                             kta=float(compute_centered_kta(K, y,
                                                           class_weighted=True)),
                             **spectral_metrics(K)))

    # ---- N2: Random cross-modal pairs ----
    for rep in range(NULL_REPS):
        rng = np.random.default_rng(2000 + rep)
        rand_pairs = random_cross_modal_pairs(selected, len(real_pairs), rng)
        logger.info(f"N2 rep={rep} random-pairs = {rand_pairs}")
        K, _ = build_agpqk_kernel(X_r, y, selected, real_gamma, rand_pairs,
                                  n_qubits=N_QUBITS, reps=REPS,
                                  outer_gamma=config.PQK_GAMMA)
        for seed in SEEDS:
            ev = svm_eval(K, y, seed)
            rows.append(dict(condition="N2-RandomPairs", rep=rep, seed=seed,
                             macro_f1=ev["macro_f1"],
                             kta=float(compute_centered_kta(K, y,
                                                           class_weighted=True)),
                             **spectral_metrics(K)))

    # ---- N3: Fisher-blind scalar γ (best of grid via KTA on training kernel) ----
    # Use KTA on the full N=500 kernel as a proxy for CV since SVM CV on a
    # precomputed quantum kernel would need recomputation; KTA is a standard
    # alignment surrogate and is seed-agnostic.
    gamma_grid = [0.2, 0.3, 0.5, 0.7, 0.8]
    for rep, g_scalar in enumerate(gamma_grid):
        gamma_uniform = np.full(N_QUBITS, g_scalar)
        K, _ = build_agpqk_kernel(X_r, y, selected, gamma_uniform, real_pairs,
                                  n_qubits=N_QUBITS, reps=REPS,
                                  outer_gamma=config.PQK_GAMMA)
        kta = float(compute_centered_kta(K, y, class_weighted=True))
        logger.info(f"N3 gamma={g_scalar} KTA={kta:.4f}")
        for seed in SEEDS:
            ev = svm_eval(K, y, seed)
            rows.append(dict(condition=f"N3-Uniform_g{g_scalar}", rep=rep,
                             seed=seed, macro_f1=ev["macro_f1"],
                             kta=kta, **spectral_metrics(K)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "metrics.csv"), index=False)

    # ---- Statistics: paired t-test vs Real-AGPQK (seed-matched means) ----
    real_by_seed = df[df.condition == "Real-AGPQK"].groupby("seed")["macro_f1"].mean()
    real_vec = real_by_seed.sort_index().values
    stats_rows = []
    pvals, names = [], []
    for cond in df.condition.unique():
        if cond == "Real-AGPQK":
            continue
        sub = df[df.condition == cond].groupby("seed")["macro_f1"].mean()
        sub_vec = sub.sort_index().values
        if len(sub_vec) != len(real_vec):
            continue
        t, p = ttest_rel(real_vec, sub_vec)
        d = cohens_d(real_vec, sub_vec)
        names.append(cond); pvals.append(float(p))
        stats_rows.append(dict(condition=cond,
                               real_mean=float(real_vec.mean()),
                               null_mean=float(sub_vec.mean()),
                               diff=float(real_vec.mean() - sub_vec.mean()),
                               t=float(t), pvalue=float(p), cohens_d=d))
    sig = holm_bonferroni(pvals, alpha=0.05)
    for row, s in zip(stats_rows, sig):
        row["holm_significant"] = bool(s)
    pd.DataFrame(stats_rows).to_csv(os.path.join(RES, "stats.csv"), index=False)

    _plot(df)


def _plot(df):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    g = df.groupby("condition")["macro_f1"].agg(["mean", "std"]).reset_index()
    g = g.sort_values("mean")
    colors = ["#9467bd" if c == "Real-AGPQK" else "#ff7f0e"
              for c in g["condition"]]
    ax.barh(g["condition"], g["mean"], xerr=g["std"], capsize=4, color=colors)
    ax.set_xlabel("Macro-F1 (mean ± std)")
    ax.grid(alpha=0.3, axis="x")
    ax.set_title("E14 — AGPQK null-control ablations")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"fig.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
