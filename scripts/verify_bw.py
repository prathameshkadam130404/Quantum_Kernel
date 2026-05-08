import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.bandwidth import load_bandwidth_gamma, apply_bandwidth
import numpy as np

g = load_bandwidth_gamma('physics')
print(f'γ_physics = {g}  (expected 0.5)')
X_test = np.array([[np.pi]])
X_scaled = apply_bandwidth(X_test, 'physics')
print(f'π scaled  = {X_scaled[0,0]:.4f}  (expected {0.5*np.pi:.4f})')
