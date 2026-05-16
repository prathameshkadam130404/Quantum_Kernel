"""
So2Sat Deep Feature Extraction via Custom Multi-Modal Autoencoder.

Trains a Convolutional Autoencoder on the exact 10,000-sample background
set to compress 18-band (SAR+Optical) So2Sat imagery into 16 features.
These features are then extracted for the 10,000-sample evaluation set
and normalized strictly using background bounds to prevent data leakage.

Output:
    data/processed/ae16_features_10k.npz
"""

import os
import sys
import numpy as np
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import local data utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from setup_data import find_local_h5_files, load_labels_only, get_stratified_indices
from test_physics_classical import normalize_to_pi

# ============================================================================
# Autoencoder Architecture
# ============================================================================

class So2SatAutoencoder(nn.Module):
    def __init__(self, in_channels=18, bottleneck_dim=16):
        super().__init__()
        
        # Encoder: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.encoder_linear = nn.Linear(256 * 4 * 4, bottleneck_dim)
        
        # Decoder
        self.decoder_linear = nn.Linear(bottleneck_dim, 256 * 4 * 4)
        self.unflatten = nn.Unflatten(1, (256, 4, 4))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        # x is [B, 18, 32, 32]
        features = self.encode(x)
        reconstructed = self.decode(features)
        return reconstructed

    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.encoder_linear(x)
        return x

    def decode(self, x):
        x = self.decoder_linear(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x


# ============================================================================
# Data Loading & Processing
# ============================================================================

def extract_raw_patches(h5_path, target_indices, desc):
    """Loads specific indices from H5 and returns (N, 18, 32, 32) tensor."""
    sar_list = []
    opt_list = []
    
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        sar_key = next(k for k in keys if k in ["sen1", "s1"])
        opt_key = next(k for k in keys if k in ["sen2", "s2"])
        
        sar_dset = f[sar_key]
        opt_dset = f[opt_key]
        
        chunk = 1000
        for start in tqdm(range(0, len(target_indices), chunk), desc=desc):
            end = min(start + chunk, len(target_indices))
            batch_idx = target_indices[start:end]
            
            sar_batch = sar_dset[batch_idx].astype(np.float32) # (B, 32, 32, 8)
            opt_batch = opt_dset[batch_idx].astype(np.float32) # (B, 32, 32, 10)
            
            sar_list.append(sar_batch)
            opt_list.append(opt_batch)
            
    sar_all = np.concatenate(sar_list, axis=0)
    opt_all = np.concatenate(opt_list, axis=0)
    
    # Stack along channel axis: (N, 32, 32, 18)
    combined = np.concatenate([sar_all, opt_all], axis=-1)
    
    # Convert to PyTorch format: (N, 18, 32, 32)
    combined = np.transpose(combined, (0, 3, 1, 2))
    return combined


def main():
    print("=" * 70)
    print("  So2Sat 18-Band Deep Feature Extraction (Zero Leakage AE-16)")
    print("=" * 70)

    # 1. Locate H5 files
    h5_files = find_local_h5_files(override_path=os.path.join(config.PROJECT_ROOT, "data", "raw"))
    if not h5_files["train"]:
        print("ERROR: Could not locate training.h5.")
        sys.exit(1)
        
    train_h5 = h5_files["train"]
    print(f"Found H5: {train_h5}")

    # 2. Get 20,000 stratified indices (Identical to extract_so2sat_10k.py)
    print("Loading labels to generate stratified indices...")
    y_full = load_labels_only(train_h5)
    
    N_EVAL = 10000
    N_BG = 10000
    TOTAL_SAMPLES = N_EVAL + N_BG
    
    indices = get_stratified_indices(y_full, TOTAL_SAMPLES, seed=42)
    eval_indices = indices[:N_EVAL]
    bg_indices = indices[N_EVAL:]
    y_eval = y_full[eval_indices]

    # 3. Load Raw Data
    print("\nLoading Background Data (For Unsupervised Training)...")
    bg_data = extract_raw_patches(train_h5, bg_indices, "Reading Background")
    
    print("\nLoading Evaluation Data (For Inference Only)...")
    eval_data = extract_raw_patches(train_h5, eval_indices, "Reading Eval")
    
    # PER-CHANNEL standardisation, fit strictly on the background set.
    # The 18 channels span very different magnitudes -- Sentinel-1 SAR
    # backscatter values are O(1e-3 to 1e0) while Sentinel-2 reflectance
    # digital numbers are O(1e2 to 1e4).  A single GLOBAL scalar mean/std
    # (the previous version of this script) lets the optical channels
    # dominate the AE's MSE reconstruction loss to the point that the
    # 16-D bottleneck is effectively an optical-only encoder; the SAR
    # contribution gets squashed into rounding noise.  Per-channel
    # statistics put all 18 channels on equal footing in the reconstruction
    # objective so the bottleneck encodes both modalities meaningfully.
    # Background-only fit; evaluation set uses the same per-channel stats.
    ch_mean = bg_data.mean(axis=(0, 2, 3), keepdims=True)  # (1, 18, 1, 1)
    ch_std = bg_data.std(axis=(0, 2, 3), keepdims=True) + 1e-8
    print(f"  Per-channel mean range: [{ch_mean.min():.4g}, {ch_mean.max():.4g}]")
    print(f"  Per-channel std range : [{ch_std.min():.4g}, {ch_std.max():.4g}]")
    bg_data = (bg_data - ch_mean) / ch_std
    eval_data = (eval_data - ch_mean) / ch_std

    # 4. Train Autoencoder on Background Data
    print("\nTraining Autoencoder on Background Data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = So2SatAutoencoder(in_channels=18, bottleneck_dim=16).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    bg_tensor = torch.tensor(bg_data).float()
    dataset = TensorDataset(bg_tensor, bg_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    EPOCHS = 10
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch [{epoch+1}/{EPOCHS}], MSE Loss: {total_loss/len(loader):.4f}")

    # 5. Extract 16-D Features
    print("\nExtracting Deep Features...")
    model.eval()
    with torch.no_grad():
        bg_features_raw = []
        # Extract for background to fit scaler
        for i in range(0, len(bg_data), 500):
            batch = torch.tensor(bg_data[i:i+500]).float().to(device)
            feats = model.encode(batch).cpu().numpy()
            bg_features_raw.append(feats)
        bg_features_raw = np.concatenate(bg_features_raw, axis=0)
        
        # Extract for evaluation
        eval_features_raw = []
        for i in range(0, len(eval_data), 500):
            batch = torch.tensor(eval_data[i:i+500]).float().to(device)
            feats = model.encode(batch).cpu().numpy()
            eval_features_raw.append(feats)
        eval_features_raw = np.concatenate(eval_features_raw, axis=0)

    # 6. Apply strictly disjoint normalization (StandardScaler -> MinMaxScaler(0, pi))
    print("\nApplying Zero-Leakage Quantum Normalization...")
    from sklearn.preprocessing import StandardScaler
    
    # 6a. StandardScaler fit on BG, applied to Eval
    scaler = StandardScaler()
    bg_features_std = scaler.fit_transform(bg_features_raw)
    eval_features_std = scaler.transform(eval_features_raw)
    
    # 6b. Map to [0, pi] using bounds from BG
    _, lo, hi = normalize_to_pi(bg_features_std)
    features_norm, _, _ = normalize_to_pi(eval_features_std, lo=lo, hi=hi)

    # 7. Save output
    out_path = os.path.join(config.PROCESSED_DIR, "ae16_features_10k.npz")
    feature_names = [f"ae_feat_{i}" for i in range(16)]
    
    np.savez_compressed(
        out_path,
        X_train=features_norm,
        X_train_raw=eval_features_std, # The pre-angle-encoded features for SVM
        y_train=y_eval,
        feature_names=np.array(feature_names)
    )
    
    print(f"\nSuccessfully generated 10,000 AE features with zero data leakage.")
    print(f"Saved to: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
