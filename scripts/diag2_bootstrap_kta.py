"""
Diagnostic 2: Bootstrap Confidence Intervals on KTA

ANSWERS: "Are the KTA differences between kernels statistically significant,
or could they be sampling noise from the 2000-sample subsample?"

Current KTA values are single-point estimates. Bootstrap resampling gives
error bars and p-values. This is essential for publication — a reviewer will
ask whether ΔKTA = +0.053 is statistically distinguishable from zero.

Method: stratified bootstrap (resample within each class to preserve class
distribution) over N_BOOT=500 iterations. For each bootstrap sample, compute
centered weighted KTA for each kernel. Report:
- Mean, std, 95% CI for each kernel
- p-value for ΔKTA(FQK - RBF) > 0 (one-sided bootstrap test)
- p-value for ΔKTA(CM_FQK - FQK) > 0

Stratified bootstrap is important here because of 67:1 class imbalance —
naive resampling risks dropping minority classes entirely.

Runtime: ~10-20 minutes (no quantum circuits, pure matrix operations × 500)
WARNING: Each KTA computation on 2000×2000 matrix takes ~1-2 seconds.
         500 boots × 5 kernels × ~1.5s = ~60 minutes total.
         If too slow: reduce N_BOOT to 200.
"""

import sys, os
sys.path.insert(0, '.')
import numpy as np
import json
import config

# ---- Config ----
N_BOOT = 200       # Reduce to 200 if runtime > 45 minutes
ALPHA  = 0.05      # For 95% CI
SEED   = config.RANDOM_SEED

# ---- Paths ----
BASE    = os.path.join(config.RESULTS_DIR, "geometric_difference", "fused")
TFK_DIR = os.path.join(BASE, "tfk")
OUT_DIR = os.path.join(config.RESULTS_DIR, "diagnostics", "bootstrap_kta")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load data ----
print("Loading kernel matrices and labels...")
data   = np.load("data/processed/subsample_2000.npz")
y_train = data["y_train"]
n = len(y_train)

K_fqk = np.load(os.path.join(BASE, "K_fqk_train.npy"))
K_pqk = np.load(os.path.join(BASE, "K_pqk_train.npy"))
K_tfk = np.load(os.path.join(TFK_DIR, "K_tfk_train.npy"))
K_rbf = np.load(os.path.join(BASE, "K_rbf_train.npy"))

cm_path1 = os.path.join(BASE, "K_cm_fqk_train.npy")
cm_path2 = os.path.join(BASE, "cm_fqk", "K_cm_fqk_train.npy")
if os.path.exists(cm_path1):
    K_cm = np.load(cm_path1)
elif os.path.exists(cm_path2):
    K_cm = np.load(cm_path2)
else:
    K_cm = None
    print("WARNING: CM_FQK not found")

kernels = {"FQK": K_fqk, "PQK": K_pqk, "TFK": K_tfk, "RBF": K_rbf}
if K_cm is not None:
    kernels["CM_FQK"] = K_cm

print(f"n={n}, N_BOOT={N_BOOT}, kernels: {list(kernels.keys())}")

# ---- KTA function (centered, class-weighted) ----
def centered_weighted_kta(K_sub, y_sub):
    """
    Centered class-weighted KTA on submatrix K_sub indexed by y_sub.
    Cortes, Mohri & Rostamizadeh (JMLR 2012).
    """
    n_s = len(y_sub)
    if n_s < 4:
        return np.nan

    # Build ideal kernel
    classes, counts = np.unique(y_sub, return_counts=True)
    count_dict = dict(zip(classes.tolist(), counts.tolist()))
    y_ideal = np.zeros((n_s, n_s))
    for i in range(n_s):
        for j in range(n_s):
            if y_sub[i] == y_sub[j]:
                y_ideal[i, j] = 1.0 / count_dict[y_sub[i]]

    # Centering matrix
    H = np.eye(n_s) - np.ones((n_s, n_s)) / n_s
    K_c = H @ K_sub @ H
    y_c = H @ y_ideal @ H

    num = np.sum(K_c * y_c)
    den = np.sqrt(np.sum(K_c**2) * np.sum(y_c**2) + 1e-15)
    return float(num / den)

# ---- Compute KTA on full matrix first (verify against exp1 numbers) ----
print("\n--- Full-Matrix KTA (verification vs exp1) ---")
full_kta = {}
for kname, K in kernels.items():
    kta = centered_weighted_kta(K, y_train)
    full_kta[kname] = kta
    print(f"  {kname:<10}: KTA = {kta:.4f}")

# ---- Stratified bootstrap ----
print(f"\nRunning stratified bootstrap (N_BOOT={N_BOOT})...")
print("Note: stratified = resample within each class to preserve class distribution")

classes = np.unique(y_train)
class_indices = {c: np.where(y_train == c)[0] for c in classes}

rng = np.random.RandomState(SEED)
boot_ktas = {kname: [] for kname in kernels}

from tqdm import tqdm
for b in tqdm(range(N_BOOT), desc="Bootstrap"):
    # Stratified resample: for each class, resample with replacement
    boot_idx = []
    for c in classes:
        cidx = class_indices[c]
        boot_idx.extend(rng.choice(cidx, size=len(cidx), replace=True))
    boot_idx = np.array(boot_idx)
    y_boot = y_train[boot_idx]

    for kname, K in kernels.items():
        K_sub = K[np.ix_(boot_idx, boot_idx)]
        kta = centered_weighted_kta(K_sub, y_boot)
        boot_ktas[kname].append(kta)

# ---- Compute statistics ----
print("\n--- Bootstrap KTA Statistics ---")
print(f"{'Kernel':<12} {'Full KTA':>10} {'Boot Mean':>10} {'Boot Std':>10} {'95% CI Lower':>13} {'95% CI Upper':>13}")
print("-" * 75)

stats = {}
for kname in kernels:
    vals = np.array(boot_ktas[kname])
    vals = vals[~np.isnan(vals)]
    lo = float(np.percentile(vals, 100 * ALPHA / 2))
    hi = float(np.percentile(vals, 100 * (1 - ALPHA / 2)))
    stats[kname] = {
        "full_kta": full_kta[kname],
        "boot_mean": float(np.mean(vals)),
        "boot_std": float(np.std(vals)),
        "ci_95_lower": lo,
        "ci_95_upper": hi,
    }
    print(f"  {kname:<10} {full_kta[kname]:>10.4f} {np.mean(vals):>10.4f} "
          f"{np.std(vals):>10.4f} {lo:>13.4f} {hi:>13.4f}")

# ---- ΔKTA significance tests ----
print("\n--- ΔKTA Significance Tests (one-sided bootstrap p-values) ---")
print("H0: ΔKTA <= 0   H1: ΔKTA > 0")

comparisons = [
    ("FQK", "RBF"),
    ("CM_FQK", "RBF"),
    ("TFK", "RBF"),
    ("PQK", "RBF"),
    ("CM_FQK", "FQK"),
    ("FQK", "TFK"),
]

delta_stats = {}
for (k1, k2) in comparisons:
    if k1 not in boot_ktas or k2 not in boot_ktas:
        continue
    v1 = np.array(boot_ktas[k1])
    v2 = np.array(boot_ktas[k2])
    delta_boot = v1 - v2
    delta_boot = delta_boot[~np.isnan(delta_boot)]

    # One-sided p-value: fraction of bootstrap samples where ΔKTA <= 0
    p_val = float(np.mean(delta_boot <= 0))
    delta_mean = float(np.mean(delta_boot))
    delta_std  = float(np.std(delta_boot))
    delta_lo   = float(np.percentile(delta_boot, 2.5))
    delta_hi   = float(np.percentile(delta_boot, 97.5))
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

    print(f"  ΔKTA({k1} - {k2}): "
          f"mean={delta_mean:+.4f}, 95%CI=[{delta_lo:+.4f}, {delta_hi:+.4f}], "
          f"p={p_val:.4f} {sig}")

    delta_stats[f"{k1}_vs_{k2}"] = {
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "ci_95": [delta_lo, delta_hi],
        "p_value_onesided": p_val,
        "significant_0.05": p_val < 0.05,
    }

# ---- Save ----
output = {"N_BOOT": N_BOOT, "n_samples": n, "kernel_stats": stats, "delta_stats": delta_stats}
with open(os.path.join(OUT_DIR, "bootstrap_kta_results.json"), "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved: {OUT_DIR}/bootstrap_kta_results.json")
print("\n=== PAPER GUIDANCE ===")
print("Report KTA as: value ± std (95% CI: [lo, hi])")
print("For ΔKTA claims: report p-value from bootstrap test.")
print("If p < 0.05: 'ΔKTA is statistically significant (bootstrap, N=500)'")
print("If p >= 0.05: ΔKTA is not statistically significant — cannot claim advantage")
