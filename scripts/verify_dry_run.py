import sys, os
import numpy as np
import config
from src.bandwidth import load_bandwidth_gamma, apply_bandwidth

phys16_path = os.path.join(config.PROCESSED_DIR, 'physics_features_16.npz')
FEATURE_INDICES = [0, 5, 6, 3, 10, 15, 9, 11]
data = np.load(phys16_path)
X16 = data['X_train']
X8_raw = X16[:, FEATURE_INDICES]
X8_scaled = apply_bandwidth(X8_raw, 'physics')
print(f'X8 raw range:    [{X8_raw.min():.4f}, {X8_raw.max():.4f}]')
print(f'X8 scaled range: [{X8_scaled.min():.4f}, {X8_scaled.max():.4f}]')
print(f'Scaling factor:  {X8_scaled.max() / X8_raw.max():.4f}  (expected 0.75)')
