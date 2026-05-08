"""
diag_pqk_physics_kta.py — PQK KTA on physics features.

Loads the already-computed PQK kernel matrices and computes:
  - Centered class-weighted KTA (primary metric)
  - Full KTA table vs FQK, RBF, TFK on physics features
  - Concentration metrics (eff_rank, off-diag mean, CV)
  - ΔKTA(PQK - RBF): the key question — does PQK beat RBF?

Reads from:
  results/physics/fused/K_pqk_physics_train.npy
  results/physics/fused/K_fqk_physics_train.npy
  results/physics/fused/K_rbf_physics_train.npy
  results/physics/fused/tfk_physics/K_tfk_physics_train.npy
  data/processed/physics_features_16.npz  (for labels)

Usage:
    python scripts/diag_pqk_physics_kta.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

import config
from src.kernel_target_alignment import compute_centered_kta, compute_full_kta_analysis

# ── Paths ──────────────────────────────────────────────────────────────────
PHYSICS_DIR = os.path.join(config.RESULTS_DIR, "physics", "fused")
TFK_DIR     = os.path.join(PHYSICS_DIR, "tfk_physics")

KERNEL_PATHS = {
    "PQK":    os.path.join(PHYSICS_DIR, "K_pqk_physics_train.npy"),
    "FQK":    os.path.join(PHYSICS_DIR, "K_fqk_physics_train.npy"),
    "RBF":    os.path.join(PHYSICS_DIR, "K_rbf_physics_train.npy"),
    "TFK":    os.path.join(TFK_DIR,     "K_tfk_physics_train.npy"),
}

PHYS16_PATH = os.path.join(config.PROCESSED_DIR, "physics_features_16.npz")

# Known baselines from PCA experiment (exp1)
PCA_BASELINES = {
    "FQK":    0.1840,
    "PQK":    0.1475,
    "TFK":    0.1825,
    "CM_FQK": 0.1897,
    "RBF":    0.1309,
}

# Known physics results already computed
PHYS_KNOWN = {
    "FQK": 0.2345,
    "TFK": 0.1947,
    "RBF": 0.2956,
}


def concentration_metrics(K):
    """Compute eff_rank, off-diag mean, CV, min eigenvalue."""
    n = K.shape[0]
    off_idx = np.triu_indices(n, k=1)
    off = K[off_idx]
    cv = float(off.std() / (off.mean() + 1e-12))

    eigvals = np.linalg.eigvalsh(K)
    eigvals_pos = np.maximum(eigvals, 0)
    total = eigvals_pos.sum()
    probs = eigvals_pos / (total + 1e-12)
    eff_rank = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
    top1_frac = float(eigvals_pos[-1] / (total + 1e-12))

    return {
        "off_diag_mean": float(off.mean()),
        "off_diag_std":  float(off.std()),
        "cv":            cv,
        "eff_rank":      eff_rank,
        "top1_frac":     top1_frac,
        "min_eigval":    float(eigvals.min()),
    }


# ── Load labels ───────────────────────────────────────────────────────────
print("=" * 65)
print("  PQK Physics KTA Diagnostic")
print("=" * 65)

if not os.path.exists(PHYS16_PATH):
    print(f"ERROR: {PHYS16_PATH} not found")
    sys.exit(1)

data    = np.load(PHYS16_PATH)
y_train = data["y_train"]
n       = len(y_train)
classes, counts = np.unique(y_train, return_counts=True)
print(f"\n  Labels: {n} samples, {len(classes)} classes")
print(f"  Imbalance: min={counts.min()}, max={counts.max()}")

# ── Load and check all kernels ────────────────────────────────────────────
print("\n--- Kernel File Status ---")
kernels = {}
for name, path in KERNEL_PATHS.items():
    if os.path.exists(path):
        K = np.load(path)
        kernels[name] = K
        diag_mean = np.diag(K).mean()
        print(f"  [{name}]  shape={K.shape}  diag_mean={diag_mean:.4f}  "
              f"({'OK' if abs(diag_mean - 1.0) < 0.01 else 'WARN: diag != 1'})")
    else:
        print(f"  [{name}]  NOT FOUND at {path}")

if "PQK" not in kernels:
    print("\nERROR: PQK kernel not found. Is exp1_physics.py still running?")
    print("Check: ps aux | grep exp1_physics")
    sys.exit(1)

# ── KTA computation ───────────────────────────────────────────────────────
print("\n--- KTA Results (centered, class-weighted) ---")
print(f"\n  {'Kernel':<10} {'Physics KTA':>12} {'PCA KTA':>10} {'Δ(phys-PCA)':>12} "
      f"{'ΔKTA vs RBF':>13}")
print("  " + "-" * 62)

kta_results = {}
rbf_kta = None

# Compute RBF first as baseline
if "RBF" in kernels:
    rbf_kta = compute_centered_kta(kernels["RBF"], y_train, class_weighted=True)
    kta_results["RBF"] = rbf_kta

# Compute all others
for name in ["PQK", "FQK", "TFK"]:
    if name in kernels:
        kta = compute_centered_kta(kernels[name], y_train, class_weighted=True)
        kta_results[name] = kta

# Print table
for name in ["PQK", "FQK", "TFK", "RBF"]:
    if name not in kta_results:
        continue
    kta   = kta_results[name]
    pca   = PCA_BASELINES.get(name, float("nan"))
    delta_pca = kta - pca
    delta_rbf = kta - kta_results.get("RBF", float("nan"))
    marker = ""
    if name != "RBF":
        if delta_rbf > 0:
            marker = " ★ BEATS RBF"
        elif delta_rbf > -0.01:
            marker = " ≈ RBF"
    print(f"  {name:<10} {kta:>12.4f} {pca:>10.4f} {delta_pca:>+12.4f} "
          f"{delta_rbf:>+13.4f}{marker}")

# ── Full KTA variants for PQK ─────────────────────────────────────────────
print(f"\n--- Full KTA Variant Analysis: PQK ---")
full = compute_full_kta_analysis(kernels["PQK"], y_train)
print(f"  kta_weighted (standard):          {full['kta_weighted']:.4f}")
print(f"  kta_centered_weighted (PRIMARY):  {full['kta_centered_weighted']:.4f}")
print(f"  kta_unweighted:                   {full['kta_unweighted']:.4f}")
print(f"  kta_centered_unweighted:          {full['kta_centered_unweighted']:.4f}")
print(f"  (PRIMARY metric = kta_centered_weighted, consistent with exp1)")

# ── Concentration metrics ─────────────────────────────────────────────────
print(f"\n--- Concentration Metrics ---")
print(f"\n  {'Kernel':<10} {'off_diag_mean':>15} {'eff_rank':>10} "
      f"{'CV':>8} {'top1_frac':>10}")
print("  " + "-" * 58)

# PCA baselines for reference
PCA_CONCENTRATION = {
    "FQK": {"off_diag_mean": 0.003, "eff_rank": 33.24,  "cv": 3.17},
    "PQK": {"off_diag_mean": None,  "eff_rank": 11.99,  "cv": None},
    "TFK": {"off_diag_mean": None,  "eff_rank": 137.81, "cv": None},
    "RBF": {"off_diag_mean": None,  "eff_rank": 2.39,   "cv": None},
}

concentration = {}
for name in ["PQK", "FQK", "TFK", "RBF"]:
    if name not in kernels:
        continue
    m = concentration_metrics(kernels[name])
    concentration[name] = m
    pca_rank = PCA_CONCENTRATION.get(name, {}).get("eff_rank", float("nan"))
    print(f"  {name:<10} {m['off_diag_mean']:>15.4f} {m['eff_rank']:>10.2f} "
          f"{m['cv']:>8.4f} {m['top1_frac']:>10.4f}  "
          f"(PCA eff_rank={pca_rank:.1f})")

# ── PQK-specific analysis ─────────────────────────────────────────────────
print(f"\n--- PQK Deep Analysis ---")

if "PQK" in concentration:
    m = concentration["PQK"]
    pqk_kta = kta_results["PQK"]
    rbf_kta_val = kta_results.get("RBF", PHYS_KNOWN["RBF"])
    fqk_kta_val = kta_results.get("FQK", PHYS_KNOWN["FQK"])

    print(f"\n  PQK kernel properties:")
    print(f"    off_diag_mean = {m['off_diag_mean']:.4f}  "
          f"(FQK={concentration.get('FQK', {}).get('off_diag_mean', 0.0927):.4f}, "
          f"RBF={concentration.get('RBF', {}).get('off_diag_mean', float('nan')):.4f})")
    print(f"    eff_rank      = {m['eff_rank']:.2f}  "
          f"(PCA-PQK=11.99, Physics-FQK=85.19)")
    print(f"    min_eigval    = {m['min_eigval']:.6f}  "
          f"({'PSD OK' if m['min_eigval'] >= -1e-6 else 'NOT PSD'})")

    print(f"\n  KTA interpretation:")
    print(f"    Physics-PQK KTA  = {pqk_kta:.4f}")
    print(f"    Physics-FQK KTA  = {fqk_kta_val:.4f}")
    print(f"    Physics-RBF KTA  = {rbf_kta_val:.4f}")
    print(f"    PCA-PQK KTA      = {PCA_BASELINES['PQK']:.4f}  (for reference)")

    delta_pqk_rbf = pqk_kta - rbf_kta_val
    delta_pqk_fqk = pqk_kta - fqk_kta_val

    print(f"\n    ΔKTA(PQK - FQK_physics) = {delta_pqk_fqk:+.4f}")
    print(f"    ΔKTA(PQK - RBF_physics) = {delta_pqk_rbf:+.4f}")

    print(f"\n  --- VERDICT ---")
    if pqk_kta > rbf_kta_val:
        print(f"  ★ PQK BEATS RBF on physics features")
        print(f"    PQK KTA={pqk_kta:.4f} > RBF KTA={rbf_kta_val:.4f} (+{delta_pqk_rbf:.4f})")
        print(f"    This is the key result: PQK avoids global overlap collapse")
        print(f"    where FQK fails, confirming the local observable hypothesis.")
        print(f"    PAPER CLAIM: PQK achieves quantum advantage on physics features")
        print(f"    by measuring local quantum observables rather than global fidelity.")
    elif pqk_kta > fqk_kta_val:
        print(f"  PQK > FQK but < RBF on physics features")
        print(f"    PQK={pqk_kta:.4f} > FQK={fqk_kta_val:.4f} > ... < RBF={rbf_kta_val:.4f}")
        print(f"    Local observables helped (PQK recovered over FQK) but")
        print(f"    not enough to beat RBF. Gap to close: {-delta_pqk_rbf:.4f}")
        print(f"    Next step: cross-modal ZZ observable extension.")
    else:
        print(f"  PQK < FQK < RBF on physics features")
        print(f"    PQK={pqk_kta:.4f}, FQK={fqk_kta_val:.4f}, RBF={rbf_kta_val:.4f}")
        print(f"    Local single-qubit observables did not help over FQK.")
        print(f"    This points to the cross-modal ZZ observable extension")
        print(f"    as the necessary next step — single-qubit PQK cannot")
        print(f"    capture SAR-optical cross-modal correlations.")


# ── Summary comparison table ──────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  FULL PHYSICS KERNEL COMPARISON (all computed so far)")
print(f"{'=' * 65}")
print(f"\n  {'Kernel':<12} {'KTA':>8}  {'vs RBF':>8}  {'vs PCA-same':>12}  Note")
print(f"  {'-' * 60}")

# Merge known and computed
all_phys = {**PHYS_KNOWN, **kta_results}
notes = {
    "RBF":    "classical baseline",
    "FQK":    "global fidelity, γ=0.50",
    "PQK":    "local observables ← KEY",
    "TFK":    "trained θ, overfit anchor",
}

for name in ["RBF", "FQK", "PQK", "TFK"]:
    if name not in all_phys:
        continue
    kta   = all_phys[name]
    d_rbf = kta - all_phys.get("RBF", float("nan"))
    d_pca = kta - PCA_BASELINES.get(name, float("nan"))
    note  = notes.get(name, "")
    star  = " ★" if name != "RBF" and d_rbf > 0 else ""
    print(f"  {name:<12} {kta:>8.4f}  {d_rbf:>+8.4f}  {d_pca:>+12.4f}  {note}{star}")

print(f"\n  CM_FQK and PIQFM results pending — will appear in exp1_physics summary")
print(f"  Run this script again after exp1_physics.py completes for full table.")
