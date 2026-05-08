import sys, os
sys.path.insert(0, '.')
import config, joblib, numpy as np

print("--- 1. Confirm config changes ---")
print('D4_MODALITY:   ', config.D4_MODALITY)
print('D4_N_CHANNELS: ', config.D4_N_CHANNELS)
print('D4_PATCH_HEIGHT:', config.D4_PATCH_HEIGHT)
print('D4_PATCH_WIDTH: ', config.D4_PATCH_WIDTH)
assert config.D4_MODALITY == 'sar', 'FAIL: modality not sar'
assert config.D4_N_CHANNELS == 8,  'FAIL: n_channels not 8'
print('config.py: PASS\n')

print("--- 2. Confirm SAR PCA loads cleanly and matches D4_N_CHANNELS ---")
pca = joblib.load(os.path.join(config.PROCESSED_DIR, 'sar_pca_model.joblib'))
n_ch = pca.components_.shape[1] // 1024
print(f'SAR PCA shape: {pca.components_.shape}')
print(f'n_channels from PCA: {n_ch}')
assert n_ch == config.D4_N_CHANNELS, f'FAIL: {n_ch} != {config.D4_N_CHANNELS}'
assert pca.components_.shape[0] == config.N_QUBITS, 'FAIL: wrong n_components'
print('SAR PCA verification: PASS\n')

print("--- 3. Confirm M_e identity check with SAR PCA ---")
from src.d4_representation import compute_m_g_matrices
n_ch = pca.components_.shape[1] // 1024
M_g = compute_m_g_matrices(
    pca.components_,
    patch_height=config.D4_PATCH_HEIGHT,
    patch_width=config.D4_PATCH_WIDTH,
    n_channels=n_ch,
)
err = np.max(np.abs(M_g['e'] - np.eye(config.N_QUBITS)))
print(f'||M_e - I||_inf = {err:.8f}')
assert err < 1e-4, f'FAIL: identity error too large: {err}'
print('M_e identity check: PASS\n')

print("--- 4. Confirm exp1 no longer references fused PCA for D4 ---")
with open('experiments/exp1_geometric_analysis.py') as f:
    content = f.read()
lines = content.split('\n')
d4_lines = [f'Line {i+1}: {l}' for i, l in enumerate(lines)
            if 'd4' in l.lower() and ('fused_pca' in l or 'n_channels=14' in l
            or 'n_channels=18' in l)]
if d4_lines:
    print('FAIL — found fused PCA references in D4 code:')
    for l in d4_lines: print(' ', l)
else:
    print('exp1 D4 references: PASS — no fused PCA in D4 code\n')
