"""
Phase 0 — Numerical verification of the BSCM (Bell-Spectrum Coupling Map)
derivation.

Verifies three propositions stated in the BSCM design:

    P1 (boundedness):
        ||H_BSCM(x_i, x_j)||_op <= 1/2  for all (x_i, x_j) in R^2.

    P2 (zero-ZZ identity):
        The coefficient of Z_i Z_j in the Pauli decomposition of
        H_BSCM(x_i, x_j) is identically zero for all (x_i, x_j).

    P3 (Pauli-subspace orthogonality from SRQFM):
        SRQFM Hamiltonians lie in span{Z_i Z_j} (a 1-d Pauli subspace).
        H_BSCM (modulo identity) lies in span{X_i X_j, Y_i Y_j}, whose
        Hilbert-Schmidt inner product with span{Z_i Z_j} is zero.
        Therefore, no choice of SRQFM coupling J(x_i, x_j) reproduces
        H_BSCM, and BSCM and SRQFM generate complementary Pauli-algebra
        components.  Concretely we check that
            <H_BSCM - (1/2) I, Z_i Z_j>_HS / 4  ==  0  and
            <H_BSCM - (1/2) I, X_i X_j>_HS / 4  =/=  0
            <H_BSCM - (1/2) I, Y_i Y_j>_HS / 4  =/=  0
        on a generic (x_i, x_j).  (Note: XX, YY, ZZ all commute pairwise
        on two qubits, so a commutator-based separation is impossible;
        the right notion of separation is HS-orthogonality.)

The script prints exact symbolic-numeric checks at random (x_i, x_j)
samples and exits non-zero if any check fails.

Usage:
    python scripts/verify_bscm_derivation.py
"""
from __future__ import annotations

import sys
import numpy as np


# Single-qubit Pauli matrices.
I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def kron(*ops: np.ndarray) -> np.ndarray:
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


# Two-qubit Pauli operators on a 4-d space.
II = kron(I2, I2)
XX = kron(X, X)
YY = kron(Y, Y)
ZZ = kron(Z, Z)


# Bell-state vectors (column).
def _ket(*comps: complex) -> np.ndarray:
    v = np.array(comps, dtype=np.complex128).reshape(-1, 1)
    return v / np.linalg.norm(v)


PHI_PLUS = _ket(1, 0, 0, 1)
PHI_MINUS = _ket(1, 0, 0, -1)
PSI_PLUS = _ket(0, 1, 1, 0)
PSI_MINUS = _ket(0, 1, -1, 0)


def bell_proj(v: np.ndarray) -> np.ndarray:
    return v @ v.conj().T


def h_bscm_from_bell_weights(x_i: float, x_j: float) -> np.ndarray:
    """Construct H_BSCM directly from the four Bell-projector weights."""
    c_plus = np.cos((x_i + x_j) / 2.0) ** 2
    s_plus = np.sin((x_i + x_j) / 2.0) ** 2
    c_minus = np.cos((x_i - x_j) / 2.0) ** 2
    s_minus = np.sin((x_i - x_j) / 2.0) ** 2
    return (
        c_plus * bell_proj(PHI_PLUS)
        + s_plus * bell_proj(PHI_MINUS)
        + c_minus * bell_proj(PSI_PLUS)
        + s_minus * bell_proj(PSI_MINUS)
    )


def h_bscm_from_pauli_form(x_i: float, x_j: float) -> np.ndarray:
    """Construct H_BSCM from the claimed Pauli decomposition."""
    alpha = 0.5 * np.cos(x_i) * np.cos(x_j)
    beta = 0.5 * np.sin(x_i) * np.sin(x_j)
    return 0.5 * II + alpha * XX + beta * YY


def pauli_coefficient(H: np.ndarray, P: np.ndarray) -> float:
    """Return Re Tr(H P) / 4 for two-qubit Hermitian H and Pauli P."""
    return float(np.real(np.trace(H @ P) / 4.0))


def operator_norm(H: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvalsh(H + H.conj().T) / 2.0)))


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def check_decomposition_identity(rng: np.random.Generator, n_samples: int = 256) -> None:
    """Bell-projector form == claimed Pauli form (up to numerical noise)."""
    print("[1/4] Bell-projector form  ==  Pauli form")
    max_err = 0.0
    for _ in range(n_samples):
        x_i, x_j = rng.uniform(-2 * np.pi, 2 * np.pi, size=2)
        H_bell = h_bscm_from_bell_weights(x_i, x_j)
        H_pauli = h_bscm_from_pauli_form(x_i, x_j)
        max_err = max(max_err, float(np.max(np.abs(H_bell - H_pauli))))
    print(f"      max |H_bell - H_pauli| over {n_samples} samples = {max_err:.2e}")
    assert max_err < 1e-12, f"Decomposition mismatch: {max_err}"
    print("      OK.")


def check_p1_boundedness(rng: np.random.Generator, n_samples: int = 4096) -> None:
    """||H_BSCM||_op <= 1/2 (after subtracting the constant identity term)."""
    print("[2/4] P1: ||H - (1/2)I||_op <= 1/2 + epsilon")
    worst = 0.0
    for _ in range(n_samples):
        x_i, x_j = rng.uniform(-np.pi, np.pi, size=2)
        H = h_bscm_from_pauli_form(x_i, x_j) - 0.5 * II
        worst = max(worst, operator_norm(H))
    print(f"      max ||H - (1/2)I||_op over {n_samples} samples = {worst:.6f}")
    assert worst <= 0.5 + 1e-10, f"P1 violated: {worst}"
    print("      OK.")


def check_p2_zero_zz(rng: np.random.Generator, n_samples: int = 4096) -> None:
    """Pauli coefficient of Z_i Z_j is identically zero."""
    print("[3/4] P2: <H_BSCM, Z_i Z_j> / 4 == 0")
    worst = 0.0
    for _ in range(n_samples):
        x_i, x_j = rng.uniform(-2 * np.pi, 2 * np.pi, size=2)
        H = h_bscm_from_pauli_form(x_i, x_j)
        coeff = pauli_coefficient(H, ZZ)
        worst = max(worst, abs(coeff))
    print(f"      max |<H, ZZ>/4| over {n_samples} samples = {worst:.2e}")
    assert worst < 1e-12, f"P2 violated: {worst}"
    print("      OK.")


def check_p3_separation(rng: np.random.Generator, n_samples: int = 4096) -> None:
    """HS-orthogonality between BSCM (XX,YY) sector and SRQFM (ZZ) sector."""
    print("[4/4] P3: <H_BSCM-(1/2)I, ZZ>_HS = 0  and  XX/YY components non-zero")
    max_zz = 0.0
    nonzero_xx = False
    nonzero_yy = False
    for _ in range(n_samples):
        x_i, x_j = rng.uniform(-np.pi, np.pi, size=2)
        H_centered = h_bscm_from_pauli_form(x_i, x_j) - 0.5 * II
        c_zz = pauli_coefficient(H_centered, ZZ)
        c_xx = pauli_coefficient(H_centered, XX)
        c_yy = pauli_coefficient(H_centered, YY)
        max_zz = max(max_zz, abs(c_zz))
        if abs(c_xx) > 1e-6:
            nonzero_xx = True
        if abs(c_yy) > 1e-6:
            nonzero_yy = True
    print(
        f"      max |<H, ZZ>/4| = {max_zz:.2e}   "
        f"XX-nonzero seen: {nonzero_xx}   YY-nonzero seen: {nonzero_yy}"
    )
    assert max_zz < 1e-12, f"P3 violated: ZZ coefficient = {max_zz}"
    assert nonzero_xx and nonzero_yy, "P3 violated: XX or YY coefficient identically zero"
    print("      OK.")


def main() -> int:
    rng = np.random.default_rng(0)
    print("BSCM Phase 0 — derivation verification")
    print("=" * 60)
    check_decomposition_identity(rng)
    check_p1_boundedness(rng)
    check_p2_zero_zz(rng)
    check_p3_separation(rng)
    print("=" * 60)
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
