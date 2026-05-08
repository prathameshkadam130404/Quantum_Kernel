#!/bin/bash
source ~/quantum-sat-env/bin/activate
cd /mnt/d/Quantum_kernel/quantum-sat-classification

echo "--- 1. Syntax check ---"
python3 -c "
import ast
import sys
try:
    with open('experiments/exp2_kernel_classification.py') as f:
        ast.parse(f.read())
    print('exp2 syntax: PASS')
except Exception as e:
    print(f'exp2 syntax: FAIL ({e})')
    sys.exit(1)
"

echo "--- 2. TFK and CM-FQK appear in the file ---"
grep -n "TFK\|CM.FQK\|K_tfk\|K_cm_fqk\|load_precomputed_kernel" \
    experiments/exp2_kernel_classification.py

echo "--- 3. McNemar pool includes all four kernels ---"
grep -n "TFK-SVM.*CM-FQK-SVM\|CM-FQK-SVM.*TFK-SVM" \
    experiments/exp2_kernel_classification.py

echo "--- 4. kernel_map covers all four kernels ---"
grep -n "kernel_map" experiments/exp2_kernel_classification.py

echo "--- 5. No compute_ calls for TFK or CM_FQK (must be load-only) ---"
grep -n "compute_tfk\|compute_cm_fqk\|compute_trained\|compute_cross" \
    experiments/exp2_kernel_classification.py | grep -v "import"

echo "--- 6. Import check (patched load to avoid file dependency) ---"
python3 -c "
import sys, os
import unittest.mock as mock
sys.path.insert(0, '.')

# Mock os.path.exists and np.load to allow import without actual files
with mock.patch('os.path.exists', return_value=True):
    with mock.patch('numpy.load', return_value=None):
        import experiments.exp2_kernel_classification as m
        print('exp2 import: PASS')
"
