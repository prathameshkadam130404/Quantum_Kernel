"""
Experiment E9 — Geometric difference g(K_classical, K_quantum).

Huang et al., Nat Commun 12, 2631 (2021):
    g(K_q, K_c) = || K_q^{1/2} (K_c + λI)^{-1} K_q^{1/2} ||_op^{1/2}
    g ≫ 1 ⇒ a potential quantum advantage; g ≈ 1 ⇒ classically reproducible.

Quantum kernels tested: FQK, PQK, CM-FQK, AGPQK  (physics-16 regime).
Classical kernels:      RBF (scale), RBF-CV tuned γ, Linear, Poly-2.
Regularisation λ = 1/n (Huang convention); also report λ sweep.

Output: results/geometric_difference/{g_table.csv, g_matrix.png}
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

import config
from src.utils import ensure_dir, setup_logging
from src.geometric_difference import (
    compute_geometric_difference,
    compute_geometric_difference_lambda_sweep,
)
from src.classical_kernels import (
    compute_rbf_kernel, compute_linear_kernel, compute_polynomial_kernel,
)
from experiments._e_common import load_physics_16

RES = ensure_dir(os.path.join(config.RESULTS_DIR, "geometric_difference"))
logger = setup_logging("exp_e9", log_file=os.path.join(RES, "e9.log"))

PHYS_DIR = os.path.join(config.RESULTS_DIR, "physics_cv", "full_kernels")
Q_KERNELS = {
    "FQK":    os.path.join(PHYS_DIR, "K_fqk_full.npy"),
    "PQK":    os.path.join(PHYS_DIR, "K_pqk_full.npy"),
    "CM-FQK": os.path.join(PHYS_DIR, "K_cm_fqk_full.npy"),
    "AGPQK":  os.path.join(PHYS_DIR, "K_agpqk_full.npy"),
}


def build_classical_kernels(X):
    return {
        "RBF":    compute_rbf_kernel(X),
        "Linear": compute_linear_kernel(X),
        "Poly2":  compute_polynomial_kernel(X, degree=2),
    }


def run():
    X_norm, _, y = load_physics_16()
    classical = build_classical_kernels(X_norm)

    rows = []
    for qname, qpath in Q_KERNELS.items():
        if not os.path.exists(qpath):
            logger.warning(f"missing: {qpath}"); continue
        Kq = np.load(qpath)
        for cname, Kc in classical.items():
            g, diag = compute_geometric_difference(Kq, Kc,
                                                   lambda_reg=None,  # 1/n
                                                   normalize_trace=True)
            rows.append(dict(quantum=qname, classical=cname, g=float(g),
                             lambda_used=float(diag.get("lambda_used", 0))))
            logger.info(f"g({qname} || {cname}) = {g:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "g_table.csv"), index=False)

    # lambda sweep for AGPQK vs RBF (main comparison)
    if os.path.exists(Q_KERNELS["AGPQK"]):
        Kq = np.load(Q_KERNELS["AGPQK"])
        sweep = compute_geometric_difference_lambda_sweep(Kq, classical["RBF"])
        with open(os.path.join(RES, "agpqk_vs_rbf_lambda_sweep.json"), "w") as f:
            json.dump({k: (v.tolist() if hasattr(v, "tolist") else v)
                       for k, v in sweep.items()}, f, indent=2)

    _plot(df)


def _plot(df):
    import matplotlib.pyplot as plt
    pivot = df.pivot(index="quantum", columns="classical", values="g")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i,j]:.2f}",
                    ha="center", va="center", color="white", fontsize=9)
    ax.set_title("Geometric difference g(K_q || K_c)  (λ=1/n)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RES, f"g_matrix.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
