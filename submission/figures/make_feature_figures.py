"""Feature-engineering diagnostic figures: correlation heatmaps for
So2Sat physics-8 and EuroSAT physics-8 (and PCA-8 for contrast)."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import config
OUT = os.path.dirname(__file__)


def _so2sat_phys8():
    """Load the So2Sat physics-8 evaluation pool used by E32/E33."""
    from experiments._e_common import load_physics_16
    from src.attention_kernel import select_features_by_fisher
    from experiments._bscm_split import make_or_load_split
    X_norm, X_raw, y = load_physics_16()
    sel_idx, _ = select_features_by_fisher(X_raw, y)
    X = X_raw[:, sel_idx]
    split = make_or_load_split(y)
    return X[split.eval_pool], y[split.eval_pool]


def _eurosat_phys8():
    """Load the EuroSAT physics-8 fixed pool (E33 indexing)."""
    from sklearn.model_selection import StratifiedShuffleSplit
    from eurosat_data import load_eurosat_allbands, compute_physics_indices
    bm, labels, _ = load_eurosat_allbands(
        cache_path=os.path.join(config.PROCESSED_DIR, "eurosat_band_means.npz"),
    )
    phys = compute_physics_indices(bm)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=1000, test_size=500, random_state=0)
    tr, te = next(sss.split(phys, labels))
    pool_idx = np.concatenate([tr, te])
    return phys[pool_idx], labels[pool_idx]


so2sat_names = ["NDVI", "NDBI", "NDWI", "BSI",
                "VV/VH", "SAR_tot", "CrossPol", "PolCoh"]
eurosat_names = ["NDVI", "NDBI", "NDWI", "BSI",
                 "SAVI", "NDRE", "MNDWI", "EVI"]


def heatmap(ax, M, names, title, vmin=-1, vmax=1):
    im = ax.imshow(M, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    n = len(names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(n):
        for j in range(n):
            t = f"{M[i, j]:+.2f}"
            color = "white" if abs(M[i, j]) > 0.6 else "black"
            ax.text(j, i, t, ha="center", va="center",
                    color=color, fontsize=6.5)
    return im


# Compute correlations
X_s, _ = _so2sat_phys8()
X_e, _ = _eurosat_phys8()
C_s = np.corrcoef(X_s.T)
C_e = np.corrcoef(X_e.T)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
im = heatmap(axes[0], C_s, so2sat_names,
             "So2Sat physics-8 (Pearson $\\rho$)")
heatmap(axes[1], C_e, eurosat_names,
        "EuroSAT physics-8 (Pearson $\\rho$)")

# colourbar
fig.subplots_adjust(right=0.90)
cax = fig.add_axes([0.92, 0.18, 0.018, 0.65])
fig.colorbar(im, cax=cax, label="Pearson correlation")

# annotate the high-correlation pairs that drive the BSCM-uniform collapse
fig.suptitle(r"Pairwise feature correlation: EuroSAT physics-8 contains "
             r"$\geq 4$ near-duplicate pairs that drive BSCM-uniform "
             r"to its data-independent fixed point", fontsize=10, y=1.02)

plt.savefig(os.path.join(OUT, "fig_feature_correlation.pdf"),
            bbox_inches="tight")
plt.savefig(os.path.join(OUT, "fig_feature_correlation.png"),
            bbox_inches="tight", dpi=160)
plt.close()
print("wrote fig_feature_correlation")

# Also print the strongest pairs (text output for paper reference)
def top_pairs(M, names, k=5):
    out = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            out.append((abs(M[i, j]), names[i], names[j], M[i, j]))
    out.sort(reverse=True)
    return out[:k]

print("\nTop-5 |rho| pairs, So2Sat physics-8:")
for r, a, b, sgn in top_pairs(C_s, so2sat_names):
    print(f"  {a:8s}-{b:8s}: {sgn:+.3f}")
print("\nTop-5 |rho| pairs, EuroSAT physics-8:")
for r, a, b, sgn in top_pairs(C_e, eurosat_names):
    print(f"  {a:8s}-{b:8s}: {sgn:+.3f}")

# Count of |rho|>0.85 pairs per dataset
def count_high_rho(M, thresh=0.85):
    n = M.shape[0]
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(M[i, j]) > thresh:
                c += 1
    return c

print(f"\nSo2Sat physics-8: {count_high_rho(C_s)} of {8*7//2} pairs have |rho|>0.85")
print(f"EuroSAT physics-8: {count_high_rho(C_e)} of {8*7//2} pairs have |rho|>0.85")
