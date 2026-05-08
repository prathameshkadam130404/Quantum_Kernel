import os
import sys
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config

def run_test():
    print("=" * 60)
    print("TEST 02: D4 n_channels Verification (SAR Only)")
    print("=" * 60)

    try:
        print(f"[CHECK] config.D4_MODALITY == 'sar' (Expected: sar, Actual: {getattr(config, 'D4_MODALITY', None)})")
        if getattr(config, 'D4_MODALITY', None) != 'sar':
            print("FAIL: config.D4_MODALITY is not 'sar'.")
            return

        print(f"[CHECK] config.D4_N_CHANNELS == 8 (Expected: 8, Actual: {getattr(config, 'D4_N_CHANNELS', None)})")
        if getattr(config, 'D4_N_CHANNELS', None) != 8:
            print("FAIL: config.D4_N_CHANNELS is not 8.")
            return

        print(f"[CHECK] config.D4_PATCH_HEIGHT == 32 and config.D4_PATCH_WIDTH == 32")
        if config.D4_PATCH_HEIGHT != 32 or config.D4_PATCH_WIDTH != 32:
            print(f"FAIL: height={config.D4_PATCH_HEIGHT}, width={config.D4_PATCH_WIDTH}")
            return

        sar_pca_path = os.path.join(config.PROCESSED_DIR, "sar_pca_model.joblib")
        print(f"[INFO] Loading SAR PCA from: {sar_pca_path}")
        if not os.path.exists(sar_pca_path):
            print("FAIL: sar_pca_model.joblib not found.")
            return
        
        pca_sar = joblib.load(sar_pca_path)
        print(f"[CHECK] pca.components_.shape — Expected: (8, 8192), Actual: {pca_sar.components_.shape}")
        
        n_channels = pca_sar.components_.shape[1] / 1024
        print(f"[CHECK] n_channels computation (shape[1]/1024) — Expected: 8.0, Actual: {n_channels}")
        if n_channels != 8.0:
            print(f"FAIL: Computed n_channels is {n_channels}, expected exactly 8.0")
            return
            
        print(f"[CHECK] pca.components_.shape[0] == config.N_QUBITS — Expected: {config.N_QUBITS}, Actual: {pca_sar.components_.shape[0]}")
        if pca_sar.components_.shape[0] != config.N_QUBITS:
            print("FAIL: n_components wrong.")
            return
            
        print(f"[CHECK] pca.components_.shape[1] % 1024 == 0 — Expected: True, Actual: {pca_sar.components_.shape[1] % 1024 == 0}")
        if pca_sar.components_.shape[1] % 1024 != 0:
            print("FAIL: raw_dim not divisible by 1024.")
            return

        print("\n[INFO] Checking fused PCA for reference...")
        fused_pca_path = os.path.join(config.PROCESSED_DIR, "fused_pca_model.joblib")
        if os.path.exists(fused_pca_path):
            pca_fused = joblib.load(fused_pca_path)
            fused_ch = pca_fused.components_.shape[1] / 1024
            print(f"  Fused PCA shape: {pca_fused.components_.shape}")
            print(f"  Fused n_channels: {fused_ch}")
            print("  Note: Fused PCA represents a flat concatenation without true spatial structure. D4 NOT applicable.")
        else:
            print("  Fused PCA not found.")

        print("\nPASS: D4 n_channels verified correctly for SAR modality.")

    except Exception as e:
        print(f"\nFAIL: An exception occurred: {e}")

if __name__ == "__main__":
    run_test()
