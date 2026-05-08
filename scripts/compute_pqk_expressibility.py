"""
PQK expressibility from saved Bloch vectors.
Computes fidelity distribution and KL from Haar for PQK (projected kernel).
Uses bloch_train.npy (standard PQK on PCA features) — no circuits needed.
"""

import sys, os
import numpy as np
import json
from scipy.special import rel_entr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from src.utils import ensure_dir

N_QUBITS    = 8
N_SAMPLES   = 1000   # fidelity pairs to sample
N_BINS      = 75
RANDOM_SEED = 42

def compute_haar_pmf(n_qubits=8, n_bins=75):
    """Haar random fidelity PMF for n_qubits."""
    dim = 2 ** n_qubits
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    pdf = (dim - 1) * np.power(1 - centers, dim - 2)
    pmf = pdf * width
    pmf /= pmf.sum()
    return centers, pmf

def pqk_fidelity(b1, b2, n_qubits=8):
    """
    HS fidelity between two projected quantum states.
    b1, b2: shape (3*n_qubits,) Bloch vectors [X0,Y0,Z0, X1,Y1,Z1, ...]
    Returns: scalar fidelity in [0, 1]
    """
    B1 = b1.reshape(n_qubits, 3)
    B2 = b2.reshape(n_qubits, 3)
    dots = np.sum(B1 * B2, axis=1)  # (n_qubits,)
    hs_per_qubit = (1.0 + dots) / 2.0
    return float(np.prod(hs_per_qubit))

def compute_pqk_kl(bloch_vectors, n_qubits=8, n_samples=1000, n_bins=75, seed=42):
    """Compute expressibility KL for PQK from Bloch vectors."""
    rng = np.random.default_rng(seed)
    n = len(bloch_vectors)
    
    # Sample random pairs
    idx1 = rng.integers(0, n, n_samples)
    idx2 = rng.integers(0, n, n_samples)
    mask = idx1 == idx2
    while mask.any():
        idx2[mask] = rng.integers(0, n, mask.sum())
        mask = idx1 == idx2
    
    # Compute fidelities
    fidelities = np.array([
        pqk_fidelity(bloch_vectors[idx1[k]], bloch_vectors[idx2[k]], n_qubits)
        for k in range(n_samples)
    ])
    
    # Histogram
    edges = np.linspace(0, 1, n_bins + 1)
    hist, _ = np.histogram(fidelities, bins=edges, density=False)
    hist_pmf = hist.astype(float) / hist.sum()
    
    # Haar reference
    centers, haar_pmf = compute_haar_pmf(n_qubits, n_bins)
    
    # KL
    eps = 1e-10
    h_safe = hist_pmf + eps; h_safe /= h_safe.sum()
    haar_safe = haar_pmf + eps; haar_safe /= haar_safe.sum()
    kl = float(np.sum(rel_entr(h_safe, haar_safe)))
    
    return {
        "kl_divergence": kl,
        "mean_fidelity": float(fidelities.mean()),
        "std_fidelity":  float(fidelities.std()),
        "fidelity_histogram": hist_pmf.tolist(),
        "haar_reference": haar_pmf.tolist(),
        "bin_centers": centers.tolist(),
        "n_zero_bins": int((hist_pmf == 0).sum()),
    }

def main():
    results = {}

    # 1. Standard PQK on PCA fused features
    bloch_fused_path = "results/geometric_difference/fused/bloch_train.npy"
    if os.path.exists(bloch_fused_path):
        bloch_fused = np.load(bloch_fused_path)
        print(f"PQK fused: computing from {bloch_fused.shape}")
        results['PQK_Fused'] = compute_pqk_kl(bloch_fused)

    # 2. AGPQK on physics features
    bloch_agpqk_path = "results/physics/fused/bloch_agpqk_physics_train.npy"
    if os.path.exists(bloch_agpqk_path):
        bloch_agpqk = np.load(bloch_agpqk_path)
        print(f"AGPQK physics: computing from {bloch_agpqk.shape}")
        results['AGPQK_Physics'] = compute_pqk_kl(bloch_agpqk)

    # 3. Standard PQK on physics features
    bloch_pqk_phys_path = "results/physics/fused/bloch_physics_train.npy"
    if os.path.exists(bloch_pqk_phys_path):
        bloch_pqk_phys = np.load(bloch_pqk_phys_path)
        print(f"PQK physics: computing from {bloch_pqk_phys.shape}")
        results['PQK_Physics'] = compute_pqk_kl(bloch_pqk_phys)

    # 4. Compare with FQK expressibility from EXP11
    exp11_path = "results/expressibility/exp11_results.json"
    if os.path.exists(exp11_path):
        with open(exp11_path) as f:
            fqk_exp11 = json.load(f)
        results['FQK_SAR']    = {"kl_divergence": fqk_exp11['SAR']['kl_divergence'],
                                 "mean_fidelity": fqk_exp11['SAR']['mean_fidelity']}
        results['FQK_Optical'] = {"kl_divergence": fqk_exp11['Optical']['kl_divergence'],
                                   "mean_fidelity": fqk_exp11['Optical']['mean_fidelity']}
        results['FQK_Fused']  = {"kl_divergence": fqk_exp11['Fused']['kl_divergence'],
                                  "mean_fidelity": fqk_exp11['Fused']['mean_fidelity']}

    print("\n=== COMBINED EXPRESSIBILITY TABLE ===")
    print(f"{'Kernel':<20} {'KL div':>10} {'Mean F':>10} {'Interpretation'}")
    for name, r in results.items():
        kl = r['kl_divergence']
        mf = r['mean_fidelity']
        interp = 'High KL=concentrated' if kl > 5 else ('Low KL=near-Haar' if kl < 2 else 'Moderate')
        print(f"{name:<20} {kl:>10.4f} {mf:>10.4f} {interp}")

    ensure_dir("results/reviewer_fixes")
    out_path = "results/reviewer_fixes/pqk_expressibility.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()
