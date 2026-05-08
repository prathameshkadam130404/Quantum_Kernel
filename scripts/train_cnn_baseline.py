"""
CNN Baseline for So2Sat LCZ42 Classification.
Trains a lightweight 3-layer CNN on raw SAR and Optical patches (32x32).
Uses the exact same 2000-sample subsample as the quantum kernel experiments.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# Ensure we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config
from setup_data import find_local_h5_files, load_labels_only, get_stratified_indices, load_h5_by_indices
from sklearn.model_selection import train_test_split

# ---- CNN Architecture ----
class So2SatCNN(nn.Module):
    def __init__(self, in_channels=18, n_classes=17):
        super(So2SatCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 4x4
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

def train_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Step 1: Find same indices ---
    h5_files = find_local_h5_files()
    y_train_full = load_labels_only(h5_files["train"])
    y_test_full = load_labels_only(h5_files["test"])

    # Replicate setup_data's logic (Rule I1)
    train_indices_full = get_stratified_indices(y_train_full, config.PCA_FIT_SAMPLES)
    test_indices_full = get_stratified_indices(y_test_full, config.SUBSAMPLE_TEST * 4)

    y_train_pca = y_train_full[train_indices_full]
    y_test_pca = y_test_full[test_indices_full]

    # Replicate create_subsample_from_processed's logic
    _, _, _, y_train_sub = train_test_split(
        train_indices_full, y_train_pca,
        test_size=config.SUBSAMPLE_TRAIN, stratify=y_train_pca, random_state=config.RANDOM_SEED
    )
    # We need the indices themselves
    _, train_indices, _, _ = train_test_split(
        train_indices_full, y_train_pca,
        test_size=config.SUBSAMPLE_TRAIN, stratify=y_train_pca, random_state=config.RANDOM_SEED
    )
    _, test_indices, _, _ = train_test_split(
        test_indices_full, y_test_pca,
        test_size=config.SUBSAMPLE_TEST, stratify=y_test_pca, random_state=config.RANDOM_SEED
    )

    # --- Step 2: Load raw 4D patches ---
    sar_tr, opt_tr, _ = load_h5_by_indices(h5_files["train"], train_indices)
    sar_te, opt_te, _ = load_h5_by_indices(h5_files["test"], test_indices)
    
    # (N, 32, 32, C) -> (N, C, 32, 32)
    X_tr = np.concatenate([sar_tr, opt_tr], axis=-1).transpose(0, 3, 1, 2)
    X_te = np.concatenate([sar_te, opt_te], axis=-1).transpose(0, 3, 1, 2)
    
    y_tr = y_train_full[train_indices]
    y_te = y_test_full[test_indices]

    # Simple normalization: scale to roughly [0, 1] range based on observed values
    # SAR range is approx [-150, 2500], Optical is [0, 0.5]
    X_tr[:, :8] = (X_tr[:, :8] + 150) / 2650.0  # SAR approx [0, 1]
    X_tr[:, 8:] = X_tr[:, 8:] * 2.0             # Optical approx [0, 1]
    X_te[:, :8] = (X_te[:, :8] + 150) / 2650.0
    X_te[:, 8:] = X_te[:, 8:] * 2.0

    # --- Step 3: PyTorch setup ---
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.long)

    train_ds = TensorDataset(X_tr_t, y_tr_t)
    test_ds = TensorDataset(X_te_t, y_te_t)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = So2SatCNN().to(device)
    
    # Calculate class weights for imbalance
    counts = np.bincount(y_tr, minlength=17)
    weights = 1.0 / (counts + 1e-6)
    weights /= weights.sum()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- Step 4: Training ---
    print("\nStarting CNN training...")
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_dl)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # --- Step 5: Evaluation ---
    model.eval()
    y_preds = []
    with torch.no_grad():
        for xb, _ in test_dl:
            xb = xb.to(device)
            out = model(xb)
            preds = torch.argmax(out, dim=1)
            y_preds.extend(preds.cpu().numpy())

    macro_f1 = f1_score(y_te, y_preds, average='macro')
    acc = accuracy_score(y_te, y_preds)

    print("\nCNN Baseline Results:")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Accuracy: {acc:.4f}")

    # --- Step 6: Save results ---
    results = {
        "model": "Lightweight-CNN",
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "params": {
            "epochs": epochs,
            "batch_size": 32,
            "lr": 0.001,
            "device": str(device),
            "input_shape": list(X_tr.shape[1:])
        }
    }
    
    out_dir = os.path.join("results", "classification")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "cnn_results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved results to {os.path.join(out_dir, 'cnn_results.json')}")

if __name__ == "__main__":
    train_cnn()
