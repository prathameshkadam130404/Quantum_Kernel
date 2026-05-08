"""Generate revised-paper figures from verified JSON caches."""
import json, os
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'results')
OUT = os.path.dirname(__file__)

# ---------- Fig: depth scaling (E37) ----------
e37 = json.load(open(os.path.join(RES, 'e37_pqk_scaling', 'summary.json')))
depths = [2, 4, 6, 8, 10]
methods = ['SRQFM-PQK', 'BSCM-PQK', 'SG-BSCM-PQK']
colors = {'SRQFM-PQK': 'C0', 'BSCM-PQK': 'C1', 'SG-BSCM-PQK': 'C2'}

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)
for ax, ds, title in zip(axes, ['so2sat', 'eurosat'], ['So2Sat (17 classes)', 'EuroSAT (10 classes)']):
    for m in methods:
        means = [e37[f'{ds}_{m}_L{L}']['mean'] for L in depths]
        stds = [e37[f'{ds}_{m}_L{L}']['std'] for L in depths]
        ax.errorbar(depths, means, yerr=stds, marker='o', label=m, color=colors[m], capsize=3)
    rbf = e37[f'{ds}_RBF-SVM']['mean']
    ax.axhline(rbf, ls='--', color='k', alpha=0.6, label=f'RBF-SVM (tuned)')
    ax.set_xlabel('Circuit depth $L$ (repetitions)')
    ax.set_ylabel('Macro-F1')
    ax.set_title(title)
    ax.set_xticks(depths)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_depth_scaling.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'fig_depth_scaling.png'), bbox_inches='tight', dpi=160)
plt.close()
print('wrote fig_depth_scaling')

# ---------- Fig: E38 max capacity (N=10K, 16q) ----------
b = json.load(open(os.path.join(RES, 'e38_max_data', 'baselines_summary.json')))
m = json.load(open(os.path.join(RES, 'e38_max_data', 'max_summary.json')))
s = json.load(open(os.path.join(RES, 'e38_max_data', 'standard_pqk_summary.json')))

labels = ['Standard\nZZ-PQK', 'BSCM-PQK', 'RBF-SVM', 'SRQFM-PQK', 'Random\nForest']
so2 = [s['so2sat']['standard_pqk_f1_mean'], m['so2sat_bscm_f1_mean'],
       b['so2sat']['rbf_f1_mean'], b['so2sat']['srqfm_f1_mean'], b['so2sat']['rf_f1_mean']]
so2e = [s['so2sat']['standard_pqk_f1_std'], m['so2sat_bscm_f1_std'],
        b['so2sat']['rbf_f1_std'], b['so2sat']['srqfm_f1_std'], b['so2sat']['rf_f1_std']]
eu = [s['eurosat']['standard_pqk_f1_mean'], m['eurosat_bscm_f1_mean'],
      b['eurosat']['rbf_f1_mean'], b['eurosat']['srqfm_f1_mean'], b['eurosat']['rf_f1_mean']]
eue = [s['eurosat']['standard_pqk_f1_std'], m['eurosat_bscm_f1_std'],
       b['eurosat']['rbf_f1_std'], b['eurosat']['srqfm_f1_std'], b['eurosat']['rf_f1_std']]

x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(8.0, 4.2))
ax.bar(x - w/2, so2, w, yerr=so2e, label='So2Sat (17 cls)', capsize=3, color='#3a6db8')
ax.bar(x + w/2, eu, w, yerr=eue, label='EuroSAT (10 cls)', capsize=3, color='#c87a3a')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Macro-F1')
ax.set_title('Max-capacity benchmark (N=10{,}000, 16 qubits, depth 6)')
ax.grid(axis='y', alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_max_capacity.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'fig_max_capacity.png'), bbox_inches='tight', dpi=160)
plt.close()
print('wrote fig_max_capacity')

# ---------- Fig: SG-BSCM correlation collapse (EuroSAT physics8 vs pca8) ----------
eu_b = json.load(open(os.path.join(RES, 'eurosat_bscm_fixed', 'eurosat_bscm_fixed_results.json')))

def grab(fs, name):
    ms = eu_b['feature_sets'][fs]['method_summary']
    return ms[name]['f1_mean'], ms[name]['f1_std']

methods = ['BSCM-uniform', 'SRQFM-fid', 'RBF-SVM', 'RF']
phys = [grab('physics8', n) for n in methods]
pca = [grab('pca8', n) for n in methods]

x = np.arange(len(methods)); w = 0.38
fig, ax = plt.subplots(figsize=(7.5, 4.0))
ax.bar(x - w/2, [p[0] for p in phys], w, yerr=[p[1] for p in phys],
       label='Physics-8 (high pairwise correlation)', capsize=3, color='#b8475a')
ax.bar(x + w/2, [p[0] for p in pca], w, yerr=[p[1] for p in pca],
       label='PCA-8 (decorrelated)', capsize=3, color='#5a93b8')
ax.set_xticks(x); ax.set_xticklabels(methods)
ax.set_ylabel('Macro-F1')
ax.set_title('EuroSAT: BSCM-uniform collapses on correlated features')
# annotate the catastrophic point
ax.annotate(f'F1={phys[0][0]:.3f}\n(near random)', xy=(0 - w/2, phys[0][0]),
            xytext=(0 - w/2 + 0.05, 0.18), fontsize=9, ha='left',
            arrowprops=dict(arrowstyle='->', color='red'))
ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig_bscm_collapse.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUT, 'fig_bscm_collapse.png'), bbox_inches='tight', dpi=160)
plt.close()
print('wrote fig_bscm_collapse')

print('all done')
