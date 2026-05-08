#!/usr/bin/env python3
"""
Visualize SRQFM Entanglement Coupling Angles (PCA vs Physics)
=============================================================
Generates heatmaps for both PCA-8 and Physics-8 feature sets
showing how the SRQFM kernel adapts its entanglement strength.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Add scripts to path
sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from eurosat_data import load_eurosat_allbands, compute_physics_indices

def generate_heatmaps():
    # 1. Load data
    print("Loading EuroSAT data...")
    cache_path = "data/processed/eurosat_band_means.npz"
    if not os.path.exists(cache_path):
        print("Error: Cache file not found. Run the experiment script first.")
        return

    data = np.load(cache_path)
    band_means = data["band_means"]
    labels = data["labels"]
    class_names = [
        "AnnualCrop", "Forest", "HerbaceousVeg", "Highway",
        "Industrial", "Pasture", "PermanentCrop", "Residential",
        "River", "SeaLake"
    ]

    # --- PCA HEATMAP ---
    print("Processing PCA-8...")
    scaler_pca = StandardScaler()
    bm_scaled = scaler_pca.fit_transform(band_means)
    pca = PCA(n_components=8, random_state=42)
    X_pca = pca.fit_transform(bm_scaled)
    
    mm_pca = MinMaxScaler(feature_range=(0, np.pi))
    X_pca_pi = mm_pca.fit_transform(X_pca)
    
    # Calculate Couplings for PCA
    n_samples = X_pca_pi.shape[0]
    n_pairs = 7
    c_pca = np.zeros((n_samples, n_pairs))
    for i in range(n_pairs):
        c_pca[:, i] = np.sin((X_pca_pi[:, i] - X_pca_pi[:, i+1]) / 2.0)**2
        
    pca_means = np.zeros((10, n_pairs))
    for cls_idx in range(10):
        pca_means[cls_idx] = c_pca[labels == cls_idx].mean(axis=0)

    # --- PHYSICS HEATMAP ---
    print("Processing Physics-8...")
    physics = compute_physics_indices(band_means)
    scaler_phys = StandardScaler()
    phys_scaled = scaler_phys.fit_transform(physics)
    mm_phys = MinMaxScaler(feature_range=(0, np.pi))
    X_phys_pi = mm_phys.fit_transform(phys_scaled)
    
    c_phys = np.zeros((n_samples, n_pairs))
    for i in range(n_pairs):
        c_phys[:, i] = np.sin((X_phys_pi[:, i] - X_phys_pi[:, i+1]) / 2.0)**2
        
    phys_means = np.zeros((10, n_pairs))
    for cls_idx in range(10):
        phys_means[cls_idx] = c_phys[labels == cls_idx].mean(axis=0)

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # PCA Plot
    sns.heatmap(
        pca_means, annot=True, fmt=".2f", cmap="YlGnBu",
        xticklabels=[f"PC{i}-PC{i+1}" for i in range(7)],
        yticklabels=class_names, ax=axes[0], cbar_kws={'label': 'c'}
    )
    axes[0].set_title("SRQFM Entanglement: PCA-8 Features")
    
    # Physics Plot
    pair_labels = [
        "NDVI-NDBI", "NDBI-NDWI", "NDWI-BSI", 
        "BSI-SAVI", "SAVI-NDRE", "NDRE-MNDWI", "MNDWI-EVI"
    ]
    sns.heatmap(
        phys_means, annot=True, fmt=".2f", cmap="YlGnBu",
        xticklabels=pair_labels,
        yticklabels=class_names, ax=axes[1], cbar_kws={'label': 'c'}
    )
    axes[1].set_title("SRQFM Entanglement: Physics-8 Features")
    
    plt.suptitle("SRQFM Self-Regulating Entanglement (Fidelity-Distance Coupling)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_path = "results/eurosat/entanglement_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"Heatmap comparison saved to: {output_path}")

if __name__ == "__main__":
    generate_heatmaps()
