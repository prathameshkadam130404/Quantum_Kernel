"""
Layer-level ZZ-leakage analysis for BSCM.

Single-pair P2 (verified in Phase 0) states that the Pauli coefficient
of Z_i Z_j in the per-pair Hamiltonian is identically zero.  But when
multiple pairs are stacked across overlapping qubits, the *effective*
generator log(U_layer) can pick up two-body Z_i Z_j Pauli components via
nested commutators (Magnus expansion), e.g.

    [X_i X_j, Y_j Y_k]  proportional to  X_i Z_j X_k - X_i Z_j X_k  (= 0 actually)
    [X_i X_j, X_j Y_k]  proportional to  i X_i I_j Y_k - i X_i Y_j ... etc.

Whether ZZ-projections survive depends on the specific feature
dependence and topology.  This script computes log(U) numerically for
the full BSCM circuit at a small qubit count and measures the magnitude
of every two-body Z_i Z_j component, plus the integrated ZZ leakage
defined as the squared total Z-only mass:

    ZZ_leakage(x, tau)
      = sum_{i<j} | < log(U(x)) , Z_i Z_j > |^2 / || log(U(x)) ||^2

so we can quantitatively state how close to "ZZ-free" the layer-level
generator is, vs. the strict zero of the single-pair theorem.

Output: results/bscm/layer_zz_leakage.json
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy.linalg import logm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pennylane as qml

import config
from src.bscm_kernel import apply_bscm_feature_map


# Single-qubit Paulis.
I2 = np.eye(2, dtype=np.complex128)
SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULI = {"I": I2, "X": SX, "Y": SY, "Z": SZ}


def kron_at(n: int, ops: Dict[int, str]) -> np.ndarray:
    """Build an n-qubit Pauli string with `ops[q]` at qubit q (default I)."""
    mats = [PAULI[ops.get(q, "I")] for q in range(n)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def all_pauli_strings(n: int) -> List[Tuple[str, np.ndarray]]:
    """Iterate over all 4^n Pauli strings of n qubits with their matrices."""
    labels = ["I", "X", "Y", "Z"]
    out: List[Tuple[str, np.ndarray]] = []

    def rec(prefix: List[str], q: int):
        if q == n:
            mats = [PAULI[s] for s in prefix]
            m = mats[0]
            for x in mats[1:]:
                m = np.kron(m, x)
            out.append(("".join(prefix), m))
            return
        for s in labels:
            rec(prefix + [s], q + 1)

    rec([], 0)
    return out


def circuit_unitary(x: np.ndarray, n_qubits: int, reps: int, tau: float) -> np.ndarray:
    """Build the 2^n x 2^n unitary of the BSCM feature map (no adjoint)."""
    dev = qml.device("default.qubit", wires=n_qubits)
    dim = 2 ** n_qubits
    U = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(dim):
        bits = [(k >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]

        @qml.qnode(dev, diff_method=None)
        def col(bits=bits):
            qml.BasisState(np.array(bits, dtype=int), wires=range(n_qubits))
            apply_bscm_feature_map(
                x, n_qubits=n_qubits, reps=reps, tau=tau,
                coupling_threshold=1e-12, connectivity="all",
            )
            return qml.state()

        U[:, k] = np.asarray(col())
    return U


def hs_decompose(H: np.ndarray, paulis: List[Tuple[str, np.ndarray]],
                 dim: int) -> Dict[str, complex]:
    """Hilbert-Schmidt decomposition of H over Pauli basis: H = sum_P c_P P,
    c_P = <P, H>_HS / dim = Tr(P H) / dim."""
    return {label: complex(np.trace(P @ H) / dim) for label, P in paulis}


def main() -> int:
    n_qubits = 4   # 4-qubit study; keeps Pauli basis manageable (256 strings)
    reps = config.ZZ_REPS
    tau_values = [0.5, 1.0, 2.0]
    n_samples = 16

    print("Layer-level ZZ-leakage analysis")
    print("=" * 70)
    print(f"  n_qubits = {n_qubits}    reps = {reps}    tau values = {tau_values}")

    paulis = all_pauli_strings(n_qubits)  # 4^4 = 256 strings
    dim = 2 ** n_qubits
    rng = np.random.default_rng(0)

    out_records = []
    for tau in tau_values:
        zz_pair_norms_per_sample: List[float] = []
        zz_leakages: List[float] = []
        gen_norms: List[float] = []
        for s in range(n_samples):
            x = rng.uniform(0, np.pi, size=n_qubits)
            U = circuit_unitary(x, n_qubits=n_qubits, reps=reps, tau=tau)
            # Effective Hermitian generator H_eff such that U = exp(-i H_eff).
            H_eff = 1j * logm(U)
            H_eff = (H_eff + H_eff.conj().T) / 2  # symmetrize numerically
            coeffs = hs_decompose(H_eff, paulis, dim)

            # Two-body Z-only Pauli strings (exactly two Zs, rest I).
            zz_pair_labels = []
            for label, _ in paulis:
                z_positions = [k for k, ch in enumerate(label) if ch == "Z"]
                non_iz = [ch for ch in label if ch not in ("I", "Z")]
                if len(z_positions) == 2 and len(non_iz) == 0:
                    zz_pair_labels.append(label)

            zz_mass = sum(abs(coeffs[lab]) ** 2 for lab in zz_pair_labels)
            total_mass = sum(abs(c) ** 2 for c in coeffs.values()
                              if not (label == "I" * n_qubits))  # exclude II...I
            # Actually compute total non-identity mass.
            total_non_id = sum(
                abs(coeffs[lab]) ** 2 for lab in coeffs
                if lab != "I" * n_qubits
            )
            leakage = zz_mass / total_non_id if total_non_id > 0 else 0.0
            zz_pair_norms_per_sample.append(np.sqrt(zz_mass))
            zz_leakages.append(leakage)
            gen_norms.append(np.sqrt(total_non_id))

        record = {
            "tau": tau,
            "n_samples": n_samples,
            "zz_pair_l2_norm_mean": float(np.mean(zz_pair_norms_per_sample)),
            "zz_pair_l2_norm_max": float(np.max(zz_pair_norms_per_sample)),
            "zz_leakage_mean": float(np.mean(zz_leakages)),
            "zz_leakage_max": float(np.max(zz_leakages)),
            "generator_total_norm_mean": float(np.mean(gen_norms)),
        }
        out_records.append(record)
        print(
            f"  tau={tau}: "
            f"zz_pair ||c||_2 mean={record['zz_pair_l2_norm_mean']:.4f}  "
            f"leakage mean={record['zz_leakage_mean']:.4f}  "
            f"leakage max={record['zz_leakage_max']:.4f}"
        )

    # Decision rule for honest paper claim:
    #  * if leakage_mean << 1, the layer-level Hamiltonian is "ZZ-suppressed"
    #    (not ZZ-free).  Report numerically.
    #  * if leakage_mean ~ 0 at all tau, P2 is ~exact at layer level.
    out_path = os.path.join(config.RESULTS_DIR, "bscm", "layer_zz_leakage.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"records": out_records, "n_qubits": n_qubits, "reps": reps}, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
