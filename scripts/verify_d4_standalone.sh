#!/bin/bash
source ~/quantum-sat-env/bin/activate
cd /mnt/d/Quantum_kernel/quantum-sat-classification

echo "--- 1. exp1 is clean ---"
grep -n "d4\|D4\|covariant" experiments/exp1_geometric_analysis.py
echo "--- 2. config.py is clean ---"
grep -n "D4_MODALITY\|D4_N_CHANNELS\|D4_PATCH" config.py

echo "--- 3. exp1 syntax check ---"
python3 -c "
import ast
import sys
try:
    with open('experiments/exp1_geometric_analysis.py') as f:
        ast.parse(f.read())
    print('exp1 syntax: PASS')
except Exception as e:
    print(f'exp1 syntax: FAIL ({e})')
    sys.exit(1)
"

echo "--- 4. New files exist ---"
python3 -c "
import os
files = [
    'src/d4_augmented_pca.py',
    'experiments/exp_d4_standalone.py',
]
for f in files:
    status = 'EXISTS' if os.path.exists(f) else 'MISSING'
    print(f'{status}: {f}')
"

echo "--- 5. New files syntax check ---"
python3 -c "
import ast, sys
for f in ['src/d4_augmented_pca.py', 'experiments/exp_d4_standalone.py']:
    with open(f) as fh:
        try:
            ast.parse(fh.read())
            print(f'Syntax OK: {f}')
        except Exception as e:
            print(f'Syntax ERROR in {f}: {e}')
            sys.exit(1)
"

echo "--- 6. Imports check ---"
python3 -c "
from src.d4_augmented_pca import fit_d4_augmented_pca, verify_invariance, augment_with_d4_orbits
print('src/d4_augmented_pca imports: PASS')
"

echo "--- 7. Augmentation sanity check ---"
python3 -c "
import numpy as np
from src.d4_augmented_pca import augment_with_d4_orbits
X = np.random.rand(10, 8192).astype(np.float32)
X_aug = augment_with_d4_orbits(X, height=32, width=32, n_channels=8)
assert X_aug.shape == (80, 8192), f'Wrong shape: {X_aug.shape}'
assert np.allclose(X_aug[:10], X), 'Identity element not first'
print(f'Augmentation shape: {X_aug.shape} (expected (80, 8192)): PASS')
"
