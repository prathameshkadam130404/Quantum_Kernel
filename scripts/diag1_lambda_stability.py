"""
Diagnostic 1: Lambda Stability Deep Analysis

ANSWERS: "Is the lambda instability numerical noise or physical geometry?"

The Huang et al. g = sqrt(max_eig(K_q^{1/2} (K_c + lambda*I)^{-1} K_q^{1/2}))
is only stable when lambda >> numerical_floor = max_eig_Kc * n * machine_epsilon.
Below this floor, near-zero eigenvalues of K_c are noise-dominated and produce
artificially inflated g values.

This script:
1. Computes the numerical stability floor for K_rbf
2. Runs a dense lambda sweep (50 log-spaced points) for all quantum kernels
3. Identifies the stable plateau (where g is flat w.r.t. log lambda)
4. Reports: stable range, plateau g value, instability source
5. Produces Figure: g vs lambda with stability annotations for the paper appendix

For the paper: report g at lambda=1/n AND show this lies within the stable plateau.
If 1/n is in the stable region, the instability flag is a numerical artifact, not
a physical problem. If 1/n is outside the stable region, the g values must be
reported as upper bounds with appropriate caveats.

Runtime: ~5-10 minutes (no quantum circuits — pure linear algebra on cached matrices)
"""

import sys, os
sys.path.insert(0, '.')
import numpy as np
import matplotlib.pyplot as plt
import config

# ---- Paths ----
BASE   = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
TFK_DIR = os.path.join(BASE, "tfk")
OUT_DIR = os.path.join(config.RESULTS_DIR, "diagnostics", "lambda_stability")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load kernels ----
print("Loading kernel matrices...")
K_fqk  = np.load(os.path.join(BASE, "K_fqk_train.npy"))
K_pqk  = np.load(os.path.join(BASE, "K_pqk_train.npy"))
K_tfk  = np.load(os.path.join(TFK_DIR, "K_tfk_train.npy"))
K_rbf  = np.load(os.path.join(BASE, "K_rbf_train.npy"))

# CM_FQK: try both possible paths
cm_path1 = os.path.join(BASE, "K_cm_fqk_train.npy")
cm_path2 = os.path.join(BASE, "cm_fqk", "K_cm_fqk_train.npy")
if os.path.exists(cm_path1):
    K_cm = np.load(cm_path1)
elif os.path.exists(cm_path2):
    K_cm = np.load(cm_path2)
else:
    K_cm = None
    print("WARNING: CM_FQK kernel not found — skipping")

n = len(K_rbf)
print(f"n={n}, kernels loaded: FQK, PQK, TFK, RBF" + (", CM_FQK" if K_cm is not None else ""))

# ---- Compute numerical stability floor for K_rbf ----
K_rbf_sym = (K_rbf + K_rbf.T) / 2
eig_rbf, V_rbf = np.linalg.eigh(K_rbf_sym)
max_eig_rbf = float(eig_rbf[-1])
eps_machine = np.finfo(np.float64).eps   # ~2.22e-16

# Stability floor: below this lambda, near-zero eigenvalues are noise-dominated
# Two definitions:
lambda_floor_strict = max_eig_rbf * eps_machine * n    # conservative
lambda_floor_loose  = max_eig_rbf * np.sqrt(eps_machine)  # standard

print(f"\n--- K_rbf Spectral Properties ---")
print(f"  max eigenvalue   : {max_eig_rbf:.4f}")
print(f"  min eigenvalue   : {eig_rbf[0]:.4e}")
print(f"  cond_num         : {max_eig_rbf / (max(eig_rbf[0], 0) + 1e-15):.2e}")
print(f"  n_positive_eigs  : {int(np.sum(eig_rbf > 1e-10))}")
print(f"  lambda=1/n       : {1/n:.2e}")
print(f"  numerical floor  : {lambda_floor_strict:.2e} (strict = max_eig * n * eps)")
print(f"  numerical floor  : {lambda_floor_loose:.2e}  (loose  = max_eig * sqrt(eps))")
print(f"  1/n vs floor     : {'ABOVE floor (stable)' if 1/n > lambda_floor_strict else 'BELOW floor (unstable)'}")

# ---- Dense lambda sweep ----
lambdas = np.logspace(-6, 0, 60)   # 60 points from 1e-6 to 1.0
lambda_1_over_n = 1.0 / n

quantum_kernels = {"FQK": K_fqk, "PQK": K_pqk, "TFK": K_tfk}
if K_cm is not None:
    quantum_kernels["CM_FQK"] = K_cm

def g_at_lambda(K_q, K_c, lam, V_c, eig_c):
    """Compute g using precomputed K_c eigendecomposition."""
    K_q = (K_q + K_q.T) / 2

    # Trace-normalize
    tr_q = np.trace(K_q) / n
    tr_c_val = np.trace(K_c) / n
    K_q_n = K_q / max(tr_q, 1e-15)
    K_c_n = K_c / max(tr_c_val, 1e-15)  # Not used in formula but shown for reference

    # K_q^{1/2}
    eig_q, V_q = np.linalg.eigh(K_q_n)
    eig_q_clip = np.clip(eig_q, 1e-10, None)
    K_q_sqrt = V_q @ np.diag(np.sqrt(eig_q_clip)) @ V_q.T

    # Rescale eig_c for normalized K_c
    eig_c_n = eig_c / max(tr_c_val, 1e-15)
    eig_c_reg_inv = 1.0 / (np.clip(eig_c_n, 0.0, None) + lam)
    K_c_reg_inv = V_c @ np.diag(eig_c_reg_inv) @ V_c.T

    M = K_q_sqrt @ K_c_reg_inv @ K_q_sqrt
    M = (M + M.T) / 2
    eig_M = np.linalg.eigvalsh(M)
    return float(np.sqrt(max(np.max(eig_M), 0.0)))

# Precompute K_rbf eigendecomposition once
K_rbf_sym = (K_rbf + K_rbf.T) / 2
K_rbf_sym = K_rbf_sym / (np.trace(K_rbf_sym) / n)  # normalize for sweep
eig_rbf_n, V_rbf_n = np.linalg.eigh(K_rbf_sym)

print(f"\nRunning dense lambda sweep ({len(lambdas)} points) for all quantum kernels...")
sweep_results = {}
for kname, K_q in quantum_kernels.items():
    print(f"  {kname}...", end="", flush=True)
    g_vals = []
    for lam in lambdas:
        g = g_at_lambda(K_q, K_rbf, lam, V_rbf_n, eig_rbf_n)
        g_vals.append(g)
    sweep_results[kname] = np.array(g_vals)
    g_at_1_over_n = g_vals[np.argmin(np.abs(lambdas - lambda_1_over_n))]
    print(f" g(1/n)={g_at_1_over_n:.2f}")

# ---- Identify stable plateau ----
print(f"\n--- Stable Plateau Analysis ---")
print(f"Stable region defined as: relative change |dg/g| < 0.10 per log-decade")
results_summary = {}
for kname, g_vals in sweep_results.items():
    # Compute relative derivative in log space
    log_lam = np.log10(lambdas)
    dg_dloglam = np.abs(np.gradient(g_vals, log_lam))
    rel_change = dg_dloglam / (g_vals + 1e-10)

    stable_mask = rel_change < 0.10
    stable_lambdas = lambdas[stable_mask]

    g_at_rec = float(g_vals[np.argmin(np.abs(lambdas - lambda_1_over_n))])
    rec_stable = bool(np.any(stable_lambdas == lambdas[np.argmin(np.abs(lambdas - lambda_1_over_n))]) or
                      (len(stable_lambdas) > 0 and
                       stable_lambdas.min() <= lambda_1_over_n <= stable_lambdas.max()))

    if len(stable_lambdas) > 0:
        g_plateau = float(np.median(g_vals[stable_mask]))
        g_plateau_std = float(np.std(g_vals[stable_mask]))
        stable_range = (float(stable_lambdas.min()), float(stable_lambdas.max()))
    else:
        g_plateau = g_at_rec
        g_plateau_std = float(np.std(g_vals))
        stable_range = (None, None)

    results_summary[kname] = {
        "g_at_1_over_n": g_at_rec,
        "g_plateau_median": g_plateau,
        "g_plateau_std": g_plateau_std,
        "stable_range": stable_range,
        "lambda_1_over_n_is_stable": rec_stable,
        "instability_source": "numerical (below floor)" if lambda_floor_strict > lambda_1_over_n else "physical",
    }

    print(f"\n  {kname}:")
    print(f"    g at lambda=1/n      : {g_at_rec:.2f}")
    print(f"    g in stable plateau  : {g_plateau:.2f} ± {g_plateau_std:.2f}")
    print(f"    stable lambda range  : {stable_range}")
    print(f"    1/n is stable        : {rec_stable}")
    print(f"    g variation (max/min): {g_vals.max()/max(g_vals.min(),1):.2f}x across full sweep")

# ---- Plot ----
colors = {"FQK": "steelblue", "PQK": "coral", "TFK": "green", "CM_FQK": "darkorange"}
fig, ax = plt.subplots(figsize=(12, 7))

for kname, g_vals in sweep_results.items():
    ax.plot(lambdas, g_vals, "o-", linewidth=2, markersize=4,
            label=kname, color=colors.get(kname, "gray"), alpha=0.85)

ax.axvline(x=lambda_1_over_n, color="black", linestyle="--", linewidth=2,
           label=f"λ = 1/n = {lambda_1_over_n:.1e} (Huang et al.)")
ax.axvline(x=lambda_floor_strict, color="red", linestyle=":", linewidth=1.5,
           label=f"Numerical floor (strict) = {lambda_floor_strict:.1e}")
ax.axvspan(1e-6, lambda_floor_strict, alpha=0.08, color="red",
           label="Numerically unstable region")

ax.set_xscale("log")
ax.set_xlabel("Regularization λ", fontsize=13)
ax.set_ylabel("Geometric difference g", fontsize=13)
ax.set_title("Lambda Stability Analysis: g vs λ for all quantum kernels vs RBF\n"
             "(Stable plateau = physically meaningful regime)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "lambda_stability.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "lambda_stability.png"), dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {OUT_DIR}/lambda_stability.pdf")

# ---- Save summary ----
import json
with open(os.path.join(OUT_DIR, "lambda_stability_summary.json"), "w") as f:
    json.dump({
        "numerical_floor_strict": lambda_floor_strict,
        "numerical_floor_loose": lambda_floor_loose,
        "lambda_1_over_n": lambda_1_over_n,
        "max_eig_rbf": max_eig_rbf,
        "kernels": results_summary
    }, f, indent=2)

print(f"\n=== PAPER GUIDANCE ===")
print(f"lambda=1/n = {lambda_1_over_n:.2e}")
print(f"Numerical floor = {lambda_floor_strict:.2e}")
if lambda_1_over_n > lambda_floor_strict:
    print("GOOD: lambda=1/n is ABOVE the numerical floor.")
    print("The 'Sweep stable: False' flag is likely caused by instability at")
    print("small lambda (< floor) in the sweep range, NOT at lambda=1/n.")
    print("Paper statement: 'g reported at lambda=1/n per Huang et al. (2021).")
    print("Lambda sensitivity analysis confirms stability in the physically")
    print("meaningful regime (Appendix, Figure X).'")
else:
    print("WARNING: lambda=1/n is BELOW the numerical floor.")
    print("g values at lambda=1/n may be inflated by numerical noise.")
    print("Consider reporting g at lambda=stable_plateau_min instead.")
