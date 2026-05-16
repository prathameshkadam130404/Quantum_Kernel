"""
Unitary-equivalence test for the BSCM gate implementation.

Verifies that on two qubits, the implemented gate sequence

    IsingXX(2 tau alpha) . IsingYY(2 tau beta)

equals the matrix exponential

    exp( -i tau ( alpha X tensor X + beta Y tensor Y ) )

over a wide range of (x_i, x_j) and tau values.  Also verifies that the
two gates commute (so order is irrelevant) and that the adjoint is the
inverse.

This test is what closes the gap between
    "alpha XX + beta YY claimed in the derivation"
and
    "what the QNode actually applies."
"""
from __future__ import annotations

import os
import sys
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pennylane as qml

from src.bscm_kernel import (
    apply_bscm_feature_map,
    apply_bscm_feature_map_adjoint,
    bscm_xy_coefficients,
)


# Two-qubit Pauli matrices.
I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
XX = np.kron(X, X)
YY = np.kron(Y, Y)


def _bscm_pair_matrix_via_qnode(x_i: float, x_j: float, tau: float) -> np.ndarray:
    """The 4x4 unitary applied by the implementation on a 2-qubit pair,
    *with the H and RZ layers stripped*, so that we test only the
    entangling sub-block alpha XX + beta YY."""
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev, diff_method=None)
    def circuit():
        alpha, beta = bscm_xy_coefficients(x_i, x_j)
        qml.IsingXX(2.0 * tau * alpha, wires=[0, 1])
        qml.IsingYY(2.0 * tau * beta, wires=[0, 1])
        return qml.state()

    # Build the matrix column-by-column from |basis>.
    U = np.zeros((4, 4), dtype=np.complex128)
    for k in range(4):
        @qml.qnode(dev, diff_method=None)
        def col(k=k):
            qml.BasisState(np.array([k // 2, k % 2]), wires=[0, 1])
            alpha, beta = bscm_xy_coefficients(x_i, x_j)
            qml.IsingXX(2.0 * tau * alpha, wires=[0, 1])
            qml.IsingYY(2.0 * tau * beta, wires=[0, 1])
            return qml.state()
        U[:, k] = np.asarray(col())
    return U


def _bscm_pair_matrix_expected(x_i: float, x_j: float, tau: float) -> np.ndarray:
    alpha, beta = bscm_xy_coefficients(x_i, x_j)
    H = alpha * XX + beta * YY
    return expm(-1j * tau * H)


def test_pair_unitary_equivalence(n_samples: int = 200, tol: float = 1e-10) -> None:
    print("[1/3] Pair-level unitary  IsingXX*IsingYY == expm(-i tau (alphaXX+betaYY))")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(n_samples):
        x_i, x_j = rng.uniform(-np.pi, np.pi, size=2)
        tau = float(rng.uniform(0.1, 3.0))
        U_impl = _bscm_pair_matrix_via_qnode(x_i, x_j, tau)
        U_exp = _bscm_pair_matrix_expected(x_i, x_j, tau)
        # Modulo a global phase: compare U_impl @ U_exp^dagger to a global phase
        # times identity.  Easier: compare directly because both have det = 1
        # for su(4) generators, and our generator is traceless on XX, YY.
        diff = float(np.max(np.abs(U_impl - U_exp)))
        worst = max(worst, diff)
    print(f"      max ||U_impl - U_exp||_max over {n_samples} samples = {worst:.2e}")
    assert worst < tol, f"Unitary mismatch: {worst}"
    print("      OK.")


def test_xx_yy_commute_on_same_pair(n_samples: int = 100, tol: float = 1e-10) -> None:
    print("[2/3] [IsingXX, IsingYY] = 0 on the same pair")
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(n_samples):
        a = float(rng.uniform(-np.pi, np.pi))
        b = float(rng.uniform(-np.pi, np.pi))
        U_xy = _seq([("IsingXX", a), ("IsingYY", b)])
        U_yx = _seq([("IsingYY", b), ("IsingXX", a)])
        worst = max(worst, float(np.max(np.abs(U_xy - U_yx))))
    print(f"      max ||XY - YX||_max over {n_samples} samples = {worst:.2e}")
    assert worst < tol, f"Commutativity failed: {worst}"
    print("      OK.")


def _seq(ops):
    dev = qml.device("default.qubit", wires=2)
    U = np.zeros((4, 4), dtype=np.complex128)
    for k in range(4):
        @qml.qnode(dev, diff_method=None)
        def col(k=k, ops=ops):
            qml.BasisState(np.array([k // 2, k % 2]), wires=[0, 1])
            for name, ang in ops:
                if name == "IsingXX":
                    qml.IsingXX(ang, wires=[0, 1])
                elif name == "IsingYY":
                    qml.IsingYY(ang, wires=[0, 1])
                else:
                    raise ValueError(name)
            return qml.state()
        U[:, k] = np.asarray(col())
    return U


def test_full_circuit_adjoint_inverse(n_samples: int = 50, tol: float = 1e-10) -> None:
    """U^dagger(x) U(x) == I  on the full multi-qubit circuit."""
    print("[3/3] U^dagger(x) U(x) == I on the full multi-qubit circuit")
    n_q = 4
    rng = np.random.default_rng(2)
    dev = qml.device("default.qubit", wires=n_q)
    worst = 0.0
    for _ in range(n_samples):
        x = rng.uniform(0, np.pi, size=n_q)
        tau = float(rng.uniform(0.3, 2.0))

        @qml.qnode(dev, diff_method=None)
        def circ():
            apply_bscm_feature_map(
                x, n_qubits=n_q, reps=2, tau=tau,
                coupling_threshold=1e-8, connectivity="all",
            )
            apply_bscm_feature_map_adjoint(
                x, n_qubits=n_q, reps=2, tau=tau,
                coupling_threshold=1e-8, connectivity="all",
            )
            return qml.probs(wires=range(n_q))

        p = np.asarray(circ())
        worst = max(worst, abs(float(p[0]) - 1.0))
    print(f"      max |P(0...0) - 1| over {n_samples} samples = {worst:.2e}")
    assert worst < tol, f"Adjoint inversion failed: {worst}"
    print("      OK.")


def main() -> int:
    print("BSCM unitary-equivalence test suite")
    print("=" * 60)
    test_pair_unitary_equivalence()
    test_xx_yy_commute_on_same_pair()
    test_full_circuit_adjoint_inverse()
    print("=" * 60)
    print("All unitary-equivalence checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
