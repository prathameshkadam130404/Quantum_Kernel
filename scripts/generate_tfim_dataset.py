"""
Generate a Transverse-Field Ising Model phase-classification dataset
for experiment E49.

Hamiltonian
-----------
On an L-site open-boundary chain,
    H(h/J) = -J * sum_i Z_i Z_{i+1}  -  h * sum_i X_i.
J is fixed to 1; the dimensionless control parameter is the transverse
field ratio h/J.  The model has a quantum critical point at h/J = 1
separating an ordered ferromagnetic phase (h/J < 1) and a disordered
paramagnetic phase (h/J > 1).

Dataset construction
--------------------
* L = 16 sites (Hilbert dim 2^16 = 65,536; ground state via sparse
  Lanczos, ~1 s per Hamiltonian).
* h/J values: drawn i.i.d. uniformly within [0.05, 0.9] for the ferromagnetic
  class and within [1.1, 2.5] for the paramagnetic class (seed 42).  The
  QCP slab (0.9, 1.1) is excluded so the binary labels are sharp.
* 300 samples per phase class -> 600 samples total, shuffled once before
  saving so class order is randomised.
* Features (16 per sample):
    - 15 nearest-neighbour Z-correlators  <Z_i Z_{i+1}>  for i in 0..14
    - 1 average transverse magnetisation  <X> = (1/L) sum_i <X_i>

Output
------
data/processed/tfim_phase_L16.npz with arrays:
    X_raw    : (600, 16) float64
    y        : (600,)    int (0 = ferro, 1 = para)
    h_over_J : (600,)    float64
    L        : ()        int

The script regenerates the file every time it is run (no cache check);
delete the .npz first if you want to regenerate with a different seed.

Usage
-----
    python scripts/generate_tfim_dataset.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

L = 16
N_PER_CLASS = 300
H_FERRO_RANGE = (0.05, 0.9)     # ordered phase
H_PARA_RANGE = (1.1, 2.5)       # disordered phase
SEED = 42

OUT_PATH = os.path.join(config.PROCESSED_DIR, "tfim_phase_L16.npz")


def _kron_chain(operators_on_qubits, L_total: int) -> sp.csr_matrix:
    """Build a tensor product of single-qubit operators over an L_total-site chain.

    ``operators_on_qubits`` is a dict mapping qubit index -> 2x2 numpy array.
    Sites not present in the dict get the 2x2 identity.  Convention: qubit 0
    is the most-significant tensor factor in the Kronecker product.
    """
    I2 = sp.identity(2, format="csr")
    op = None
    for q in range(L_total):
        local = sp.csr_matrix(operators_on_qubits.get(q, I2.toarray()))
        if op is None:
            op = local
        else:
            op = sp.kron(op, local, format="csr")
    return op.tocsr()


def _build_tfim_hamiltonian(L_total: int, h: float, J: float = 1.0,
                             ) -> sp.csr_matrix:
    """H = -J sum Z_i Z_{i+1}  -  h sum X_i (open BC)."""
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    dim = 2 ** L_total
    H = sp.csr_matrix((dim, dim))
    for i in range(L_total - 1):
        H = H - J * _kron_chain({i: Z, i + 1: Z}, L_total)
    for i in range(L_total):
        H = H - h * _kron_chain({i: X}, L_total)
    return H.tocsr()


def _ground_state(H: sp.csr_matrix) -> np.ndarray:
    """Lowest-eigenvalue eigenvector (column form)."""
    # sigma=None + which='SA' is the most reliable choice for real-symmetric
    # Hamiltonians; we request k=1 eigenvector only.
    vals, vecs = spla.eigsh(H, k=1, which="SA", maxiter=10_000, tol=1e-8)
    return vecs[:, 0]


def _measure_correlators(psi: np.ndarray, L_total: int) -> np.ndarray:
    """Return (15 nearest-neighbour <Z_i Z_{i+1}>, 1 average <X>) -> 16 features.
    """
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    feats = []
    for i in range(L_total - 1):
        ZZ = _kron_chain({i: Z, i + 1: Z}, L_total)
        feats.append(float(psi @ ZZ.dot(psi)))
    x_total = 0.0
    for i in range(L_total):
        Xi = _kron_chain({i: X}, L_total)
        x_total += float(psi @ Xi.dot(psi))
    feats.append(x_total / L_total)
    return np.asarray(feats, dtype=np.float64)


def main() -> None:
    rng = np.random.default_rng(SEED)
    h_ferro = rng.uniform(H_FERRO_RANGE[0], H_FERRO_RANGE[1], size=N_PER_CLASS)
    h_para = rng.uniform(H_PARA_RANGE[0], H_PARA_RANGE[1], size=N_PER_CLASS)
    h_all = np.concatenate([h_ferro, h_para])
    y_all = np.concatenate([
        np.zeros(N_PER_CLASS, dtype=np.int64),
        np.ones(N_PER_CLASS, dtype=np.int64),
    ])

    # Shuffle so ordering is not class-stratified.
    perm = rng.permutation(len(h_all))
    h_all = h_all[perm]
    y_all = y_all[perm]

    X = np.zeros((len(h_all), L), dtype=np.float64)
    t_start = time.time()
    for s in range(len(h_all)):
        if s % 25 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / max(1, s) * (len(h_all) - s) if s > 0 else None
            print(f"[{s:4d}/{len(h_all)}] h/J={h_all[s]:.3f} (y={y_all[s]})  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s" if eta else
                  f"[{s:4d}/{len(h_all)}] h/J={h_all[s]:.3f} (y={y_all[s]})")
        H = _build_tfim_hamiltonian(L, h_all[s], J=1.0)
        psi = _ground_state(H)
        X[s] = _measure_correlators(psi, L)
    print(f"Generated {len(h_all)} samples in {time.time() - t_start:.0f}s")

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    np.savez(OUT_PATH, X_raw=X, y=y_all, h_over_J=h_all, L=L)
    print(f"Saved: {OUT_PATH}")
    print(f"  X_raw shape  : {X.shape}")
    print(f"  class balance: y=0 -> {int((y_all == 0).sum())}, "
          f"y=1 -> {int((y_all == 1).sum())}")
    print(f"  feature ranges (min/max per dim):")
    for i in range(X.shape[1]):
        name = (f"<Z{i}Z{i+1}>" if i < L - 1 else "<X>_avg")
        print(f"    {name:>10s}: [{X[:, i].min():+.4f}, {X[:, i].max():+.4f}]")


if __name__ == "__main__":
    main()
