#!/usr/bin/env python3
"""Quick test of the EuroSAT data loader."""
import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(message)s")

from eurosat_data import load_eurosat_allbands, prepare_features, compute_physics_indices
import numpy as np

bm, y, cn = load_eurosat_allbands("data/raw/eurosat/EuroSATallBands", max_per_class=10)
print(f"Band means: {bm.shape}, Labels: {np.bincount(y)}")

physics = compute_physics_indices(bm)
print(f"Physics: {physics.shape}")
names = ["NDVI", "NDBI", "NDWI", "BSI", "SAVI", "NDRE", "MNDWI", "EVI"]
for i, n in enumerate(names):
    col = physics[:, i]
    print(f"  {n}: mean={col.mean():.3f} std={col.std():.3f} range=[{col.min():.3f}, {col.max():.3f}]")

feats = prepare_features(bm, y)
print(f"PCA-8: {feats['pca8'].shape}, explained: {feats['pca_explained_var']:.3f}")
print(f"Physics-8: {feats['physics8'].shape}")
print("ALL OK")
