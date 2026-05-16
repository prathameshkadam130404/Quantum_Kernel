"""
Extract the 10,000-sample So2Sat physics-Fisher-16 evaluation pool for E38.

Selects 20,000 stratified indices from the So2Sat training H5; the first
10,000 form the evaluation pool, the second 10,000 form a disjoint
background pool used to fit the per-feature [0, pi] normalisation bounds.
This split guarantees zero test-set leakage in the encoded representation.

Output: data/processed/physics_features_10k.npz with X_train (encoded in
[0, pi]), X_train_raw (pre-normalisation), y_train, and feature_names.
"""
import os
import sys
import numpy as np
from tqdm import tqdm
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from setup_data import find_local_h5_files, load_labels_only, get_stratified_indices
from test_physics_classical import extract_16_features, normalize_to_pi

N_EVAL = 10000        # evaluation pool size
N_BG = 10000          # disjoint background pool for normalisation fit
TOTAL_SAMPLES = N_EVAL + N_BG

def main():
    print("=" * 60)
    print(f"  Extracting {N_EVAL} So2Sat Physics Features (Zero Leakage)")
    print("=" * 60)

    # 1. Locate H5 files
    h5_files = find_local_h5_files(override_path=os.path.join(config.PROJECT_ROOT, "data", "raw"))
    if not h5_files["train"]:
        print("ERROR: Could not locate training.h5. Make sure your hard drive is connected.")
        sys.exit(1)
        
    train_h5 = h5_files["train"]
    print(f"Found H5: {train_h5}")

    # 2. Get 20,000 stratified indices
    print("Loading labels to generate stratified indices...")
    y_full = load_labels_only(train_h5)
    indices = get_stratified_indices(y_full, TOTAL_SAMPLES, seed=42)
    
    eval_indices = indices[:N_EVAL]
    bg_indices = indices[N_EVAL:]
    y_eval = y_full[eval_indices]

    def extract_features(target_indices, desc):
        features_raw_list = []
        with h5py.File(train_h5, "r") as f:
            keys = list(f.keys())
            sar_key = next(k for k in keys if k in ["sen1", "s1"])
            opt_key = next(k for k in keys if k in ["sen2", "s2"])
            
            sar_dset = f[sar_key]
            opt_dset = f[opt_key]
            
            chunk = 1000
            for start in tqdm(range(0, len(target_indices), chunk), desc=desc):
                end = min(start + chunk, len(target_indices))
                batch_idx = target_indices[start:end]
                
                sar_batch = sar_dset[batch_idx].astype(np.float32)
                opt_batch = opt_dset[batch_idx].astype(np.float32)
                
                f_raw, feature_names = extract_16_features(sar_batch, opt_batch)
                features_raw_list.append(f_raw)
        return np.concatenate(features_raw_list, axis=0), feature_names

    # 3. Extract Background features and fit scaler
    bg_features_raw, _ = extract_features(bg_indices, "Processing Background Batches")
    
    # 4. Extract Evaluation features
    eval_features_raw, feature_names = extract_features(eval_indices, "Processing Evaluation Batches")

    # 5. Apply strictly disjoint normalization
    print("Computing normalization bounds on background set...")
    _, lo, hi = normalize_to_pi(bg_features_raw)
    
    print("Applying background bounds to evaluation set...")
    features_norm, _, _ = normalize_to_pi(eval_features_raw, lo=lo, hi=hi)
    
    # 6. Save output
    out_path = os.path.join(config.PROCESSED_DIR, "physics_features_10k.npz")
    np.savez_compressed(
        out_path,
        X_train=features_norm,
        X_train_raw=eval_features_raw,
        y_train=y_eval,
        feature_names=np.array(feature_names)
    )
    
    print(f"Successfully generated {N_EVAL} samples with zero data leakage.")
    print(f"Saved to: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
