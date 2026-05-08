import os
import sys
import numpy as np
from tqdm import tqdm
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from setup_data import find_local_h5_files, load_labels_only, get_stratified_indices
from scripts.test_physics_classical import extract_16_features, normalize_to_pi

# Goal: Extract exactly 10,000 samples for the maximum-capacity So2Sat experiment.
N_SAMPLES = 10000

def main():
    print("=" * 60)
    print(f"  Extracting {N_SAMPLES} So2Sat Physics Features")
    print("=" * 60)

    # 1. Locate H5 files
    h5_files = find_local_h5_files(override_path=os.path.join(config.PROJECT_ROOT, "data", "raw"))
    if not h5_files["train"]:
        print("ERROR: Could not locate training.h5. Make sure your hard drive is connected.")
        sys.exit(1)
        
    train_h5 = h5_files["train"]
    print(f"Found H5: {train_h5}")

    # 2. Get 10,000 stratified indices
    print("Loading labels to generate stratified indices...")
    y_full = load_labels_only(train_h5)
    indices = get_stratified_indices(y_full, N_SAMPLES, seed=42)
    y_10k = y_full[indices]

    # 3. Load patches in chunks to preserve RAM
    print(f"Loading Patches for {N_SAMPLES} samples in batches...")
    
    with h5py.File(train_h5, "r") as f:
        keys = list(f.keys())
        sar_key = next(k for k in keys if k in ["sen1", "s1"])
        opt_key = next(k for k in keys if k in ["sen2", "s2"])
        
        sar_dset = f[sar_key]
        opt_dset = f[opt_key]
        
        features_raw_list = []
        
        chunk = 1000
        for start in tqdm(range(0, len(indices), chunk), desc="Processing Batches"):
            end = min(start + chunk, len(indices))
            batch_idx = indices[start:end]
            
            # Read batch into memory
            sar_batch = sar_dset[batch_idx].astype(np.float32)
            opt_batch = opt_dset[batch_idx].astype(np.float32)
            
            # Compute physical parameters for all 16 features
            f_raw, feature_names = extract_16_features(sar_batch, opt_batch)
            features_raw_list.append(f_raw)
            
    # 4. Combine and scale to [0, pi]
    print("Combining features and applying normalization bounds...")
    features_raw = np.concatenate(features_raw_list, axis=0)
    
    # Calculate global bounds specifically for this 10k dataset and scale to [0, pi]
    features_norm, lo, hi = normalize_to_pi(features_raw)
    
    # 5. Save output
    out_path = os.path.join(config.PROCESSED_DIR, "physics_features_10k.npz")
    np.savez_compressed(
        out_path,
        X_train=features_norm,
        X_train_raw=features_raw,
        y_train=y_10k,
        feature_names=np.array(feature_names)
    )
    
    print(f"Successfully generated {N_SAMPLES} samples.")
    print(f"Saved to: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
