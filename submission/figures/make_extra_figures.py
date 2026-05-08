"""Additional revised-paper figures: Pauli heatmap, 14q concentration histogram,
confusion matrices for SRQFM-PQK at 16q.
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'results')
OUT = os.path.dirname(__file__)


# ============================================================
# Fig: BSCM Pauli-coefficient heatmap (uniform / phi / psi)
# ============================================================
def pauli_coeffs(xi, xj, w):
    cp = np.cos((xi + xj) / 2.0) ** 2
    sp = np.sin((xi + xj) / 2.0) ** 2
    cm = np.cos((xi - xj) / 2.0) ** 2
    sm = np.sin((xi - xj) / 2.0) ** 2
    w1, w2, w3, w4 = w
    a_xx = (w1 * cp - w2 * sp + w3 * cm - w4 * sm) / 4.0
    a_yy = (-w1 * cp + w2 * sp + w3 * cm - w4 * sm) / 4.0
    a_zz = (w1 * cp + w2 * sp - w3 * cm - w4 * sm) / 4.0
    return a_xx, a_yy, a_zz


grid = np.linspace(0, np.pi, 81)
XI, XJ = np.meshgrid(grid, grid)
priors = [
    ("uniform",  (1, 1, 1, 1)),
    ("phi_only", (2, 2, 0, 0)),
    ("psi_only", (0, 0, 2, 2)),
]

fig, axes = plt.subplots(3, 3, figsize=(9.5, 9.0), sharex=True, sharey=True)
for r, (pname, w) in enumerate(priors):
    a_xx, a_yy, a_zz = pauli_coeffs(XI, XJ, w)
    for c, (vals, label) in enumerate([(a_xx, r'$\alpha_{XX}$'),
                                        (a_yy, r'$\alpha_{YY}$'),
                                        (a_zz, r'$\alpha_{ZZ}$')]):
        ax = axes[r, c]
        im = ax.imshow(vals, origin='lower', extent=[0, np.pi, 0, np.pi],
                       vmin=-0.5, vmax=0.5, cmap='RdBu_r')
        if r == 0:
            ax.set_title(label)
        if c == 0:
            ax.set_ylabel(f'{pname}\n$x_j$')
        if r == 2:
            ax.set_xlabel(r'$x_i$')
        ax.set_xticks([0, np.pi/2, np.pi])
        ax.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
        ax.set_yticks([0, np.pi/2, np.pi])
        ax.set_yticklabels(['0', r'$\pi/2$', r'$\pi$'])
fig.subplots_adjust(right=0.90, hspace=0.15, wspace=0.15)
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cax, label=r'coefficient value (bound: $|\alpha|\leq 1/2$)')
plt.savefig(os.path.join(OUT, 'fig_pauli_heatmap.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'fig_pauli_heatmap.png'), bbox_inches='tight', dpi=160)
plt.close()
print('wrote fig_pauli_heatmap')


# ============================================================
# Fig: 8q vs 14q off-diagonal histogram (BSCM-uniform)
# ============================================================
def load_K(path):
    K = np.load(path)
    n = K.shape[0]
    iu = np.triu_indices(n, k=1)
    return K[iu]


# 8q: results/bscm/K_bscm_holdout_tau0.25.npy is 200x200 holdout. The 1800-sample
# evaluation kernel is rebuilt per-seed; use the holdout as a representative.
# For consistency with the publication-analysis numbers, we synthesise the
# off-diag distribution from the cached 14q kernel and compare to the cached
# 8q holdout kernel.
k8_path = os.path.join(RES, 'bscm', 'K_bscm_physics8_uniform.npy')
k14_path = None
for cand in ['K_bscm_14q_phys14_uniform_tau0.25_N800.npy',
             'K_bscm_14q.npy', 'K_bscm_14q_uniform.npy']:
    p = os.path.join(RES, 'bscm_14q_test', cand)
    if os.path.exists(p):
        k14_path = p
        break

if os.path.exists(k8_path):
    od8 = load_K(k8_path)
else:
    od8 = None
od14 = load_K(k14_path) if (k14_path and os.path.exists(k14_path)) else None

fig, ax = plt.subplots(figsize=(6.8, 4.0))
bins = np.linspace(0, 1, 81)
if od8 is not None:
    ax.hist(od8, bins=bins, alpha=0.55, density=True,
            label=f'8 qubits (mean={od8.mean():.3f}, var={od8.var():.4f})',
            color='#3a6db8')
if od14 is not None:
    ax.hist(od14, bins=bins, alpha=0.55, density=True,
            label=f'14 qubits (mean={od14.mean():.3f}, var={od14.var():.4f})',
            color='#c87a3a')
ax.set_xlabel('off-diagonal kernel value')
ax.set_ylabel('density')
ax.set_title('BSCM-uniform off-diagonal distribution: concentration with qubit count')
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_concentration_hist.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'fig_concentration_hist.png'), bbox_inches='tight', dpi=160)
plt.close()
print('wrote fig_concentration_hist',
      '(8q ok)' if od8 is not None else '(8q missing)',
      '(14q ok)' if od14 is not None else '(14q missing)')


# ============================================================
# Fig: simplified 4-qubit circuit diagram (text-based for now)
# ============================================================
fig, ax = plt.subplots(figsize=(8.0, 3.5))
ax.axis('off')

qubit_y = [3.0, 2.0, 1.0, 0.0]
gate_x = [0.5, 1.3, 2.6, 3.6, 4.6]
labels = ['H', '$RZ(x_i)$', 'IsingXX($\\alpha_{XX}$)',
          'IsingYY($\\alpha_{YY}$)', 'IsingZZ($\\alpha_{ZZ}$)']

# wire lines
for q in range(4):
    ax.hlines(qubit_y[q], 0, 5.0, color='gray', lw=0.7)
    ax.text(-0.2, qubit_y[q], f'$|0\\rangle_{q}$',
            fontsize=11, ha='right', va='center')

# H + RZ on each qubit
for q in range(4):
    for x, lab in zip(gate_x[:2], ['H', f'$RZ(x_{q})$']):
        ax.add_patch(mpl.patches.Rectangle(
            (x - 0.18, qubit_y[q] - 0.18), 0.36, 0.36,
            facecolor='#e8efff', edgecolor='black', lw=0.8))
        ax.text(x, qubit_y[q], lab, fontsize=8, ha='center', va='center')

# Per-pair Ising gates (qubits 0-1 shown)
for x, lab in zip(gate_x[2:], ['XX', 'YY', 'ZZ']):
    rect = mpl.patches.Rectangle(
        (x - 0.32, qubit_y[1] - 0.18), 0.64, qubit_y[0] - qubit_y[1] + 0.36,
        facecolor='#f3e7d3', edgecolor='black', lw=0.8)
    ax.add_patch(rect)
    ax.text(x, (qubit_y[0] + qubit_y[1]) / 2,
            f'Ising{lab}\n($2\\tau\\alpha_{{{lab}}}$)',
            fontsize=8, ha='center', va='center')

ax.text(2.5, -0.95,
        'Per repetition $r$: H $\\to$ RZ$(x_i)$ $\\to$ '
        '∏ IsingXX$\\cdot$IsingYY$\\cdot$IsingZZ over connectivity pairs',
        fontsize=9, ha='center')
ax.text(2.5, -1.45,
        'SRQFM: only IsingZZ($\\sin^2((x_i-x_j)/2)$). '
        'BSCM-uniform: only XX, YY ($\\alpha_{ZZ}\\equiv 0$). '
        'SG-BSCM: above $\\times \\sin^2((x_i-x_j)/2)$.',
        fontsize=9, ha='center')

ax.set_xlim(-0.6, 5.2)
ax.set_ylim(-1.7, 3.4)
ax.set_title('One repetition of the SRQFM / BSCM / SG-BSCM feature map (4-qubit example, qubit-pair (0,1) shown)',
             fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_circuit.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'fig_circuit.png'), bbox_inches='tight', dpi=160)
plt.close()
print('wrote fig_circuit')

print('all extra figures done')
