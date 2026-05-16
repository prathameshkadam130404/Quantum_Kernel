"""
Unit tests for the Quantum-Sat Classification pipeline.

Tests core modules: feature extraction, quantum kernels, classical kernels,
geometric difference, KTA, concentration, data loading.
"""

import os, sys
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


class TestFeatureExtraction(unittest.TestCase):
    def test_pca_output_shape(self):
        from src.feature_extraction import fit_and_transform_full_pipeline
        X = np.random.rand(50, 100).astype(np.float32)
        r = fit_and_transform_full_pipeline(X, None, X[:10], n_components=8)
        self.assertEqual(r["X_train"].shape, (50, 8))
        self.assertEqual(r["X_test"].shape, (10, 8))

    def test_normalization_range(self):
        from src.feature_extraction import fit_and_transform_full_pipeline
        X = np.random.rand(30, 50).astype(np.float32)
        r = fit_and_transform_full_pipeline(X, None, X[:5], n_components=4)
        self.assertGreaterEqual(r["X_train"].min(), 0.0 - 1e-6)
        self.assertLessEqual(r["X_train"].max(), np.pi + 1e-6)


class TestQuantumKernels(unittest.TestCase):
    def test_fqk_symmetry(self):
        from src.quantum_kernels import compute_fqk_kernel_matrix
        X = np.random.rand(5, 4) * np.pi
        K = compute_fqk_kernel_matrix(X, n_qubits=4, reps=1)
        np.testing.assert_allclose(K, K.T, atol=1e-6)

    def test_fqk_diagonal(self):
        from src.quantum_kernels import compute_fqk_kernel_matrix
        X = np.random.rand(5, 4) * np.pi
        K = compute_fqk_kernel_matrix(X, n_qubits=4, reps=1)
        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-4)

    def test_fqk_range(self):
        from src.quantum_kernels import compute_fqk_kernel_matrix
        X = np.random.rand(5, 4) * np.pi
        K = compute_fqk_kernel_matrix(X, n_qubits=4, reps=1)
        self.assertGreaterEqual(K.min(), -1e-4)
        self.assertLessEqual(K.max(), 1.0 + 1e-4)

    def test_pqk_output(self):
        from src.quantum_kernels import compute_pqk_kernel_matrix
        X = np.random.rand(5, 4) * np.pi
        K = compute_pqk_kernel_matrix(X, n_qubits=4, reps=1)
        self.assertEqual(K.shape, (5, 5))


class TestClassicalKernels(unittest.TestCase):
    def test_rbf_symmetry(self):
        from src.classical_kernels import compute_rbf_kernel
        X = np.random.rand(10, 8)
        K = compute_rbf_kernel(X)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_tensor_product(self):
        from src.classical_kernels import compute_tensor_product_kernel
        X = np.random.rand(10, 8)
        K = compute_tensor_product_kernel(X, n_sar_features=4, n_opt_features=4)
        self.assertEqual(K.shape, (10, 10))
        np.testing.assert_allclose(K, K.T, atol=1e-10)


class TestGeometricDifference(unittest.TestCase):
    def test_g_identity(self):
        from src.geometric_difference import compute_geometric_difference
        K = np.eye(10) * 0.5 + 0.5
        K = K @ K.T; K /= K.max(); np.fill_diagonal(K, 1.0)
        g, _ = compute_geometric_difference(K, K)
        self.assertAlmostEqual(g, 1.0, places=1)

    def test_g_lower_bound(self):
        from src.geometric_difference import compute_geometric_difference
        K1 = np.eye(10) + np.random.rand(10, 10) * 0.1
        K1 = (K1 + K1.T) / 2; np.fill_diagonal(K1, 1.0)
        K2 = np.eye(10) + np.random.rand(10, 10) * 0.3
        K2 = (K2 + K2.T) / 2; np.fill_diagonal(K2, 1.0)
        g, _ = compute_geometric_difference(K1, K2)
        self.assertGreaterEqual(g, 1.0 - 1e-6)


class TestKTA(unittest.TestCase):
    def test_ideal_kernel_kta(self):
        from src.kernel_target_alignment import compute_kta
        y = np.array([0, 0, 1, 1, 2, 2])
        K = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                K[i, j] = 1.0 if y[i] == y[j] else 0.0
        kta = compute_kta(K, y, class_weighted=False)
        self.assertAlmostEqual(kta, 1.0, places=5)

    def test_kta_range(self):
        from src.kernel_target_alignment import compute_kta
        K = np.random.rand(20, 20)
        K = (K + K.T) / 2
        y = np.random.randint(0, 3, 20)
        kta = compute_kta(K, y)
        self.assertGreaterEqual(kta, -1.0)
        self.assertLessEqual(kta, 1.0)


class TestConcentration(unittest.TestCase):
    def test_identity_cv(self):
        from src.kernel_concentration import compute_concentration_metrics
        K = np.eye(10)
        m = compute_concentration_metrics(K)
        self.assertEqual(m["cv"], 0.0)

    def test_effective_rank(self):
        from src.kernel_concentration import compute_concentration_metrics
        K = np.eye(10)
        m = compute_concentration_metrics(K)
        self.assertAlmostEqual(m["effective_rank"], 10.0, places=1)


if __name__ == "__main__":
    unittest.main()
