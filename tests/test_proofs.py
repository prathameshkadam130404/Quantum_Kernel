"""
Executable verification of every mathematical claim in the SRQFM/BSCM/SG-BSCM
proofs (Appendix A--C of the manuscript).

For each proposition we check both the symbolic form (where possible) and a
numerical assertion to a tight tolerance (typically 1e-10 to 1e-6 depending
on the claim).  Run as:

    pytest -v tests/test_proofs.py

or as a script:

    python -m tests.test_proofs

Author: Prathamesh Kadam
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pennylane as qml
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.bscm_kernel import (  # noqa: E402
    BELL_WEIGHT_PRESETS, bscm_pauli_coefficients, apply_bscm_feature_map,
)
from src.srqfm_kernel import fidelity_coupling  # noqa: E402

RNG = np.random.default_rng(0)
TOL_TIGHT = 1e-10
TOL_NUM = 1e-8
TOL_LOOSE = 1e-3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_qubit_state(x: float) -> np.ndarray:
    """|psi(x)> = RZ(x) H |0> = (e^{-ix/2} |0> + e^{ix/2} |1>) / sqrt(2)."""
    return np.array(
        [np.exp(-1j * x / 2.0), np.exp(1j * x / 2.0)],
        dtype=np.complex128,
    ) / np.sqrt(2.0)


def _bell_state(label: str) -> np.ndarray:
    """The four Bell states as 4-d state vectors in the |q_i q_j> basis."""
    s2 = np.sqrt(2.0)
    if label == "Phi+":
        return np.array([1, 0, 0, 1], dtype=np.complex128) / s2
    if label == "Phi-":
        return np.array([1, 0, 0, -1], dtype=np.complex128) / s2
    if label == "Psi+":
        return np.array([0, 1, 1, 0], dtype=np.complex128) / s2
    if label == "Psi-":
        return np.array([0, 1, -1, 0], dtype=np.complex128) / s2
    raise ValueError(label)


# Pauli matrices
I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _kron(*ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


# ===========================================================================
# Proposition 1: Single-qubit fidelity grounding (Appendix A)
# ===========================================================================

class TestProp1Fidelity:
    """|<psi(x_i)|psi(x_j)>|^2 = cos^2((x_i - x_j) / 2),
    so 1 - |<.>|^2 = sin^2((x_i - x_j) / 2) = J(x_i, x_j)."""

    def test_explicit_overlap(self):
        rng = np.random.default_rng(42)
        for _ in range(1000):
            xi, xj = rng.uniform(-2 * np.pi, 2 * np.pi, size=2)
            psi_i = _single_qubit_state(xi)
            psi_j = _single_qubit_state(xj)
            overlap_sq = abs(np.vdot(psi_i, psi_j)) ** 2
            expected = math.cos((xi - xj) / 2.0) ** 2
            assert abs(overlap_sq - expected) < TOL_TIGHT

    def test_J_matches_fidelity_distance(self):
        rng = np.random.default_rng(43)
        for _ in range(1000):
            xi, xj = rng.uniform(0, np.pi, size=2)
            psi_i = _single_qubit_state(xi)
            psi_j = _single_qubit_state(xj)
            d = 1.0 - abs(np.vdot(psi_i, psi_j)) ** 2
            J = fidelity_coupling(xi, xj)
            assert abs(d - J) < TOL_TIGHT

    def test_J_boundary_values(self):
        # J(x, x) = 0 exactly
        for x in [0.0, 0.7, np.pi / 2, np.pi]:
            assert abs(fidelity_coupling(x, x)) < TOL_TIGHT
        # J(0, pi) = sin^2(pi/2) = 1
        assert abs(fidelity_coupling(0.0, np.pi) - 1.0) < TOL_TIGHT


# ===========================================================================
# Proposition 2: Fubini-Study geodesic identity (Appendix A)
# ===========================================================================

class TestProp2FubiniStudy:
    """J = sin^2(D_FS(psi(x_i), psi(x_j)))."""

    def test_arccos_relationship(self):
        rng = np.random.default_rng(44)
        for _ in range(1000):
            xi, xj = rng.uniform(0, np.pi, size=2)
            psi_i = _single_qubit_state(xi)
            psi_j = _single_qubit_state(xj)
            ov = abs(np.vdot(psi_i, psi_j))
            d_fs = math.acos(min(1.0, max(-1.0, ov)))
            J = fidelity_coupling(xi, xj)
            assert abs(math.sin(d_fs) ** 2 - J) < TOL_TIGHT


# ===========================================================================
# Proposition 3: Expected coupling under uniform encoding (Appendix A)
# ===========================================================================

class TestProp3ExpectedCoupling:
    """For X, Y i.i.d. uniform on [0, gamma*pi],
    E[sin^2((X-Y)/2)] = 1/2 - 2 sin^2(gamma*pi/2) / (gamma*pi)^2."""

    @staticmethod
    def _predicted(gamma: float) -> float:
        a = gamma * np.pi
        return 0.5 - 2.0 * math.sin(a / 2.0) ** 2 / (a * a)

    def test_monte_carlo_match(self):
        rng = np.random.default_rng(45)
        N = 200_000
        for gamma in [0.25, 0.5, 0.75, 1.0]:
            X = rng.uniform(0, gamma * np.pi, size=N)
            Y = rng.uniform(0, gamma * np.pi, size=N)
            mc = float((np.sin((X - Y) / 2.0) ** 2).mean())
            pred = self._predicted(gamma)
            # Monte Carlo standard error ~ sqrt(Var)/sqrt(N) ~ 0.001 for N=2e5
            assert abs(mc - pred) < 0.005, \
                f"gamma={gamma}: MC={mc:.5f} vs pred={pred:.5f}"

    def test_predicted_at_gamma_1(self):
        # gamma=1 (full [0, pi]):  E[J] = 1/2 - 2 sin^2(pi/2) / pi^2
        #                                = 1/2 - 2 / pi^2
        expected = 0.5 - 2.0 / (math.pi ** 2)
        assert abs(self._predicted(1.0) - expected) < TOL_TIGHT
        # Approximately 0.2974
        assert abs(self._predicted(1.0) - 0.29735762) < 1e-6


# ===========================================================================
# Bell-state Pauli expansion (Appendix B Lemma)
# ===========================================================================

class TestBellPauliExpansion:
    """Verify the four Bell-projector expansions in the Pauli basis:
        |Phi+><Phi+| = (1/4)(I + XX - YY + ZZ)
        |Phi-><Phi-| = (1/4)(I - XX + YY + ZZ)
        |Psi+><Psi+| = (1/4)(I + XX + YY - ZZ)
        |Psi-><Psi-| = (1/4)(I - XX - YY - ZZ)
    """

    def test_phi_plus(self):
        proj = np.outer(_bell_state("Phi+"), _bell_state("Phi+").conj())
        rhs = 0.25 * (
            _kron(I2, I2) + _kron(X, X) - _kron(Y, Y) + _kron(Z, Z)
        )
        assert np.max(np.abs(proj - rhs)) < TOL_TIGHT

    def test_phi_minus(self):
        proj = np.outer(_bell_state("Phi-"), _bell_state("Phi-").conj())
        rhs = 0.25 * (
            _kron(I2, I2) - _kron(X, X) + _kron(Y, Y) + _kron(Z, Z)
        )
        assert np.max(np.abs(proj - rhs)) < TOL_TIGHT

    def test_psi_plus(self):
        proj = np.outer(_bell_state("Psi+"), _bell_state("Psi+").conj())
        rhs = 0.25 * (
            _kron(I2, I2) + _kron(X, X) + _kron(Y, Y) - _kron(Z, Z)
        )
        assert np.max(np.abs(proj - rhs)) < TOL_TIGHT

    def test_psi_minus(self):
        proj = np.outer(_bell_state("Psi-"), _bell_state("Psi-").conj())
        rhs = 0.25 * (
            _kron(I2, I2) - _kron(X, X) - _kron(Y, Y) - _kron(Z, Z)
        )
        assert np.max(np.abs(proj - rhs)) < TOL_TIGHT


# ===========================================================================
# BSCM Pauli-coefficient algebra (Appendix B main result)
# ===========================================================================

class TestBSCMPauliAlgebra:
    """Verify that bscm_pauli_coefficients() reproduces the analytical
    Pauli decomposition of H_ij = sum_k 2 p_k w_k |B_k><B_k| under each prior.
    """

    def _analytical_H(self, xi: float, xj: float, weights):
        """Sum up the Bell-projector expansion explicitly and return
        (alpha_XX, alpha_YY, alpha_ZZ, alpha_I) as the coefficients of
        H_ij = alpha_XX XX + alpha_YY YY + alpha_ZZ ZZ + alpha_I I.
        """
        # Bell amplitudes
        cp = math.cos((xi + xj) / 2.0) ** 2
        sp = math.sin((xi + xj) / 2.0) ** 2
        cm = math.cos((xi - xj) / 2.0) ** 2
        sm = math.sin((xi - xj) / 2.0) ** 2
        wphi_p = 0.5 * cp
        wphi_m = 0.5 * sp
        wpsi_p = 0.5 * cm
        wpsi_m = 0.5 * sm

        # H_ij = sum 2 p_k w_k |B_k><B_k|
        p1, p2, p3, p4 = weights
        H = (2 * p1 * wphi_p * np.outer(_bell_state("Phi+"), _bell_state("Phi+").conj())
             + 2 * p2 * wphi_m * np.outer(_bell_state("Phi-"), _bell_state("Phi-").conj())
             + 2 * p3 * wpsi_p * np.outer(_bell_state("Psi+"), _bell_state("Psi+").conj())
             + 2 * p4 * wpsi_m * np.outer(_bell_state("Psi-"), _bell_state("Psi-").conj()))

        # Project onto Pauli basis of (II, XX, YY, ZZ).
        def trace_inner(A, B):
            return np.trace(A.conj().T @ B).real / 4.0  # /dim

        a_I = trace_inner(_kron(I2, I2), H)
        a_XX = trace_inner(_kron(X, X), H)
        a_YY = trace_inner(_kron(Y, Y), H)
        a_ZZ = trace_inner(_kron(Z, Z), H)
        return a_XX, a_YY, a_ZZ, a_I

    def test_uniform(self):
        rng = np.random.default_rng(10)
        for _ in range(500):
            xi, xj = rng.uniform(-2 * np.pi, 2 * np.pi, size=2)
            a_xx, a_yy, a_zz = bscm_pauli_coefficients(
                xi, xj, BELL_WEIGHT_PRESETS["uniform"],
            )
            # Closed-form expectation:
            # alpha_XX = (1/2) cos(xi) cos(xj)
            # alpha_YY = (1/2) sin(xi) sin(xj)
            # alpha_ZZ = 0
            assert abs(a_xx - 0.5 * math.cos(xi) * math.cos(xj)) < TOL_NUM
            assert abs(a_yy - 0.5 * math.sin(xi) * math.sin(xj)) < TOL_NUM
            assert abs(a_zz) < TOL_NUM

    def test_phi_only(self):
        rng = np.random.default_rng(11)
        for _ in range(500):
            xi, xj = rng.uniform(-2 * np.pi, 2 * np.pi, size=2)
            a_xx, a_yy, a_zz = bscm_pauli_coefficients(
                xi, xj, BELL_WEIGHT_PRESETS["phi_only"],
            )
            # phi_only: w = (2,2,0,0)
            # alpha_XX = (1/2)(cos^2((xi+xj)/2) - sin^2((xi+xj)/2)) = (1/2) cos(xi+xj)
            # alpha_YY = -alpha_XX
            # alpha_ZZ = (1/2)(cos^2((xi+xj)/2) + sin^2((xi+xj)/2)) = 1/2
            assert abs(a_xx - 0.5 * math.cos(xi + xj)) < TOL_NUM
            assert abs(a_yy + 0.5 * math.cos(xi + xj)) < TOL_NUM
            assert abs(a_zz - 0.5) < TOL_NUM

    def test_psi_only(self):
        rng = np.random.default_rng(12)
        for _ in range(500):
            xi, xj = rng.uniform(-2 * np.pi, 2 * np.pi, size=2)
            a_xx, a_yy, a_zz = bscm_pauli_coefficients(
                xi, xj, BELL_WEIGHT_PRESETS["psi_only"],
            )
            # psi_only: w = (0,0,2,2)
            # alpha_XX = alpha_YY = (1/2)(cos^2((xi-xj)/2) - sin^2((xi-xj)/2))
            #                     = (1/2) cos(xi - xj)
            # alpha_ZZ = -1/2
            assert abs(a_xx - 0.5 * math.cos(xi - xj)) < TOL_NUM
            assert abs(a_yy - 0.5 * math.cos(xi - xj)) < TOL_NUM
            assert abs(a_zz + 0.5) < TOL_NUM

    def test_alpha_bound(self):
        """|alpha| <= 1/2 across all three priors for arbitrary inputs."""
        rng = np.random.default_rng(13)
        for prior_name, prior in BELL_WEIGHT_PRESETS.items():
            worst = 0.0
            for _ in range(10000):
                xi, xj = rng.uniform(-2 * np.pi, 2 * np.pi, size=2)
                a_xx, a_yy, a_zz = bscm_pauli_coefficients(xi, xj, prior)
                worst = max(worst, abs(a_xx), abs(a_yy), abs(a_zz))
            assert worst <= 0.5 + TOL_NUM, f"{prior_name}: max|alpha|={worst}"

    def test_function_matches_analytic_via_trace(self):
        """The closed-form Pauli coefficients used by the code exactly
        match the trace-projection of H_ij for every prior and arbitrary
        inputs."""
        rng = np.random.default_rng(14)
        for prior in [BELL_WEIGHT_PRESETS[k] for k in ("uniform", "phi_only", "psi_only")]:
            for _ in range(200):
                xi, xj = rng.uniform(-2 * np.pi, 2 * np.pi, size=2)
                code = bscm_pauli_coefficients(xi, xj, prior)
                truth = self._analytical_H(xi, xj, prior)[:3]
                for c, t in zip(code, truth):
                    assert abs(c - t) < TOL_NUM


# ===========================================================================
# Numerical alpha_ZZ identical-zero check via the actual circuit
# ===========================================================================

class TestUniformAlphaZZIsZero:
    """For BSCM-uniform on a 2-qubit register, applying one repetition of
    the circuit then measuring <Z_0 Z_1> on |00> should give 0 to numerical
    precision (independent of x and tau).  This is the strongest possible
    test that alpha_ZZ ≡ 0 in the implementation."""

    def test_two_qubit(self):
        dev = qml.device("default.qubit", wires=2)
        rng = np.random.default_rng(15)

        @qml.qnode(dev, diff_method=None)
        def circuit(x):
            apply_bscm_feature_map(
                x, n_qubits=2, reps=1, tau=1.0,
                coupling_threshold=0.0,  # no gate dropping
                connectivity="all",
                bell_weights=BELL_WEIGHT_PRESETS["uniform"],
            )
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        for _ in range(50):
            x = rng.uniform(0, np.pi, size=2)
            val = float(circuit(x))
            assert abs(val) < TOL_TIGHT, f"x={x}: <ZZ>={val}"


# ===========================================================================
# Proposition 4: SG-BSCM correlation-suppression bound (Appendix C)
# ===========================================================================

class TestProp4CorrSuppression:
    """For (X_i, X_j) ~ N(mu, Sigma) with marginal variance sigma^2 and
    correlation rho, E[w_{Psi-}] = (1/4) (1 - exp(-sigma^2 (1 - rho))).
    This is the *Gaussian* closed-form bound in Appendix C, NOT the
    second-order Taylor approximation.
    """

    @staticmethod
    def _predicted(sigma2: float, rho: float) -> float:
        return 0.25 * (1.0 - math.exp(-sigma2 * (1.0 - rho)))

    def test_monte_carlo_match(self):
        rng = np.random.default_rng(16)
        N = 200_000
        for sigma in [0.3, 0.5, 1.0]:
            for rho in [-0.9, -0.5, 0.0, 0.3, 0.7, 0.9, 0.99]:
                cov = np.array([[sigma ** 2, rho * sigma * sigma],
                                [rho * sigma * sigma, sigma ** 2]])
                Z = rng.multivariate_normal([0.0, 0.0], cov, size=N)
                Delta = Z[:, 0] - Z[:, 1]
                mc = float((np.sin(Delta / 2.0) ** 2).mean())
                # E[sin^2(Delta/2)] = 1/2 - 1/2 e^{-Var(Delta)/2}, where
                # Var(Delta) = 2 sigma^2 (1 - rho).  So
                # mc -> 0.5 - 0.5 e^{-sigma^2 (1 - rho)}.
                # E[w_{Psi-}] = (1/2) E[sin^2(Delta/2)] = predicted.
                pred = 2.0 * self._predicted(sigma ** 2, rho)
                tol = 0.005
                assert abs(mc - pred) < tol, \
                    f"sigma={sigma}, rho={rho}: MC={mc:.4f}  pred={pred:.4f}"

    def test_vanishing_at_rho_one(self):
        # rho -> 1: predicted -> 0
        for sigma2 in [0.1, 1.0, 4.0]:
            assert abs(self._predicted(sigma2, 1.0)) < TOL_TIGHT

    def test_monotonic_in_rho(self):
        # E[w_{Psi-}] is monotonically decreasing in rho on (-1, 1) at fixed sigma
        sigma2 = 0.5
        rhos = np.linspace(-0.99, 0.99, 50)
        vals = np.array([self._predicted(sigma2, r) for r in rhos])
        diffs = np.diff(vals)
        assert (diffs <= TOL_NUM).all()  # non-increasing

    def test_op_norm_bound(self):
        """||H^SG_ij|| <= 2 tau w_{Psi-} pointwise: every gate angle
        scaled by w_{Psi-} (a non-negative scalar), hence operator norm
        bounded element-wise."""
        rng = np.random.default_rng(17)
        tau = 0.25
        for _ in range(200):
            xi, xj = rng.uniform(0, np.pi, size=2)
            a_xx, a_yy, a_zz = bscm_pauli_coefficients(
                xi, xj, BELL_WEIGHT_PRESETS["uniform"],
            )
            sg = math.sin((xi - xj) / 2.0) ** 2
            scaled_max = max(abs(a_xx), abs(a_yy), abs(a_zz)) * sg
            # Per-pair generator norm <= 2 tau * scaled_max
            # by the spectral radius of IsingXX/YY/ZZ at angle 2 tau alpha.
            # Bound check: 2 tau * scaled_max <= 2 tau * (1/2) * sg = tau * sg.
            bound_pred = tau * sg
            actual = 2 * tau * scaled_max
            assert actual <= bound_pred + TOL_NUM


# ===========================================================================
# Run as script
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
