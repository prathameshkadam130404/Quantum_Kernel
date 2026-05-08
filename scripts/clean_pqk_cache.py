"""
Clean corrupted PQK cache files.

Iterates through RESULTS_DIR and deletes all K_pqk_test.npy files that have
the transposed (n_train, n_test) shape instead of the required (n_test, n_train).
This prevents ValueErrors in SVM prediction after the PQK bugfix.
"""
import os
import sys
import argparse
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def clean_cache(dry_run=True):
    results_dir = config.RESULTS_DIR
    print(f"Searching for corrupted PQK cache in: {results_dir}")
    print(f"Dry run: {dry_run}")
    
    count = 0
    deleted = 0
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == "K_pqk_test.npy" or file.endswith("_pqk_te.npy") or "_pqk_test" in file:
                path = os.path.join(root, file)
                try:
                    K = np.load(path)
                    n_rows, n_cols = K.shape
                    
                    # Heuristic: in most of our experiments, n_train (e.g. 500 or 100)
                    # is larger than n_test (e.g. 100 or 20).
                    # If n_rows > n_cols, it's likely transposed.
                    # More reliably, we can check if it's square vs rectangular.
                    if n_rows == n_cols:
                        continue # Square kernel is likely a training kernel, ignore
                        
                    # If it's K_pqk_test, it should be (n_test, n_train)
                    # In config.py: N_SUBSAMPLE_TRAIN=500, N_SUBSAMPLE_TEST=100
                    # So shape should be (100, 500). If it is (500, 100), it's definitely transposed.
                    if n_rows > n_cols:
                        print(f"Found corrupted kernel: {path} (shape {K.shape})")
                        count += 1
                        if not dry_run:
                            os.remove(path)
                            print(f"  Deleted: {path}")
                            deleted += 1
                except Exception as e:
                    print(f"Error checking {path}: {e}")

    print(f"\nScan complete.")
    print(f"Found: {count} corrupted files.")
    if not dry_run:
        print(f"Deleted: {deleted} files.")
    else:
        print(f"Run with --force to actually delete files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean transposed PQK test kernels.")
    parser.add_argument("--force", action="store_true", help="Actually delete files.")
    args = parser.parse_args()
    
    clean_cache(dry_run=not args.force)
