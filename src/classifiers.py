"""
Classifiers for the Quantum-Sat Classification pipeline.

Implements:
    - Precomputed kernel SVM (for quantum kernels): SVC(kernel='precomputed')
    - Classical SVMs: RBF, Polynomial, Linear
    - CNN baseline: ResNet-18 frozen feature extractor → linear head
    - ALL SVMs use class_weight='balanced' (Rule I2)
    - ALL metrics include macro-F1 as PRIMARY (Rule I3)
    - McNemar's test for statistical comparison

Anti-patterns enforced:
    - Rule I2: class_weight='balanced' everywhere
    - Rule I3: macro-F1 primary, accuracy secondary
    - Rule I5: class-weighted cross-entropy for CNN/QNN
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import compute_full_metrics

logger = logging.getLogger(__name__)


# ============ PRECOMPUTED KERNEL SVM ============

def train_precomputed_svm(
    K_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 100.0,
    seed: int = config.RANDOM_SEED,
) -> SVC:
    """
    Train SVM with precomputed kernel matrix.

    ALWAYS uses class_weight='balanced' per Rule I2.

    Args:
        K_train: Precomputed training kernel matrix, shape (n_train, n_train).
        y_train: Training labels.
        C: Regularization parameter.
        seed: Random seed.

    Returns:
        SVC: Trained SVM classifier.
    """
    clf = SVC(
        kernel="precomputed",
        class_weight="balanced",  # Rule I2: ALWAYS balanced
        C=C,
        random_state=seed,
    )
    clf.fit(K_train, y_train)

    logger.info(
        f"Trained precomputed SVM: {len(y_train)} samples, "
        f"{len(np.unique(y_train))} classes, C={C}"
    )

    return clf


def evaluate_classifier(
    clf: SVC,
    K_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict:
    """
    Evaluate a precomputed kernel SVM.

    Returns full metrics per Rule I3: macro-F1 (PRIMARY), accuracy,
    per-class F1, Cohen's κ, confusion matrix data.

    Args:
        clf: Trained SVM.
        K_test: Test kernel matrix, shape (n_test, n_train).
        y_test: Test labels.
        model_name: Name for logging.

    Returns:
        dict: Full metrics (see src.utils.compute_full_metrics).
    """
    y_pred = clf.predict(K_test)
    metrics = compute_full_metrics(y_test, y_pred)
    metrics["model_name"] = model_name
    metrics["predictions"] = y_pred

    logger.info(
        f"[{model_name}] macro-F1={metrics['macro_f1']:.4f} (PRIMARY), "
        f"accuracy={metrics['accuracy']:.4f}, κ={metrics['cohen_kappa']:.4f}"
    )

    return metrics


# ============ CLASSICAL SVMs ============

def train_classical_svms(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = config.RANDOM_SEED,
) -> Dict[str, Dict]:
    """
    Train and evaluate all classical SVM baselines.

    ALL use class_weight='balanced' per Rule I2.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        seed: Random seed.

    Returns:
        dict: {model_name: metrics_dict}.
    """
    results = {}

    classifiers = {
        "RBF-SVM": SVC(
            kernel="rbf",
            class_weight="balanced",  # Rule I2
            random_state=seed,
        ),
        "Linear-SVM": SVC(
            kernel="linear",
            class_weight="balanced",
            random_state=seed,
        ),
        "Poly2-SVM": SVC(
            kernel="poly",
            degree=2,
            class_weight="balanced",
            random_state=seed,
        ),
        "Poly3-SVM": SVC(
            kernel="poly",
            degree=3,
            class_weight="balanced",
            random_state=seed,
        ),
    }

    for name, clf in classifiers.items():
        logger.info(f"Training {name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        metrics = compute_full_metrics(y_test, y_pred)
        metrics["model_name"] = name
        metrics["predictions"] = y_pred
        results[name] = metrics

        logger.info(
            f"  [{name}] macro-F1={metrics['macro_f1']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}"
        )

    return results


# ============ CNN BASELINE — ResNet-18 ============

def train_cnn_baseline(
    X_train_images: np.ndarray,
    y_train: np.ndarray,
    X_test_images: np.ndarray,
    y_test: np.ndarray,
    n_classes: int = config.LCZ_N_CLASSES,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.RANDOM_SEED,
) -> Dict:
    """
    Train ResNet-18 CNN baseline (MANDATORY per Rule 14).

    Architecture: Pretrained ResNet-18 (ImageNet, frozen) → feature extraction
    (512-d) → linear head (512→17), Adam lr=1e-3, 20 epochs.
    Uses class-weighted cross-entropy (Rule I5).

    Args:
        X_train_images: Training images, shape (N, C, H, W) or (N, H, W, C).
        y_train: Training labels.
        X_test_images: Test images.
        y_test: Test labels.
        n_classes: Number of output classes.
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        seed: Random seed.

    Returns:
        dict: Full metrics.
    """
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)

    # Prepare images for ResNet (expects 3 channels)
    # If input has >3 channels, take first 3
    if X_train_images.ndim == 4:
        if X_train_images.shape[-1] in [3, 8, 10]:  # (N, H, W, C) format
            X_train_images = np.transpose(X_train_images, (0, 3, 1, 2))
            X_test_images = np.transpose(X_test_images, (0, 3, 1, 2))

    # Take first 3 channels if more
    if X_train_images.shape[1] > 3:
        X_train_images = X_train_images[:, :3]
        X_test_images = X_test_images[:, :3]

    # Resize to 224×224 if needed (ResNet expects this)
    if X_train_images.shape[2] != 224:
        import torch.nn.functional as F
        X_train_t = torch.FloatTensor(X_train_images)
        X_test_t = torch.FloatTensor(X_test_images)
        X_train_t = F.interpolate(X_train_t, size=(224, 224), mode="bilinear", align_corners=False)
        X_test_t = F.interpolate(X_test_t, size=(224, 224), mode="bilinear", align_corners=False)
    else:
        X_train_t = torch.FloatTensor(X_train_images)
        X_test_t = torch.FloatTensor(X_test_images)

    # Load pretrained ResNet-18
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze feature extractor
    for param in resnet.parameters():
        param.requires_grad = False

    # Replace final layer
    n_features = resnet.fc.in_features  # 512
    resnet.fc = nn.Linear(n_features, n_classes)

    # Class-weighted loss (Rule I5)
    class_counts = np.bincount(y_train, minlength=n_classes)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = len(y_train) / (n_classes * class_counts.astype(float))
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))

    optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=lr)

    # Training
    y_train_t = torch.LongTensor(y_train)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    resnet.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = resnet(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)

        if (epoch + 1) % 5 == 0:
            logger.info(f"  CNN epoch {epoch+1}/{epochs}: loss={epoch_loss/len(y_train):.4f}")

    # Evaluation
    resnet.eval()
    with torch.no_grad():
        test_outputs = []
        test_dataset = TensorDataset(X_test_t)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        for (batch,) in test_loader:
            out = resnet(batch)
            test_outputs.append(out)
        test_outputs = torch.cat(test_outputs)
        y_pred = test_outputs.argmax(dim=1).numpy()

    metrics = compute_full_metrics(y_test, y_pred)
    metrics["model_name"] = "ResNet-18-CNN"

    logger.info(
        f"  [CNN] macro-F1={metrics['macro_f1']:.4f}, "
        f"accuracy={metrics['accuracy']:.4f}"
    )

    return metrics


# ============ McNemar's TEST ============

def mcnemars_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
) -> Dict:
    """
    McNemar's test for statistical comparison of two classifiers.

    Tests H0: both models have the same error rate.

    Args:
        y_true: Ground truth labels.
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.
        model_a_name: Name of model A.
        model_b_name: Name of model B.

    Returns:
        dict: {
            'statistic': float,
            'p_value': float,
            'n_a_correct_b_wrong': int,
            'n_a_wrong_b_correct': int,
            'significant': bool,  # at α = 0.05
        }
    """
    from scipy.stats import chi2

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency table
    n_01 = np.sum(correct_a & ~correct_b)  # A correct, B wrong
    n_10 = np.sum(~correct_a & correct_b)  # A wrong, B correct

    # McNemar statistic (with continuity correction)
    if n_01 + n_10 == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (abs(n_01 - n_10) - 1) ** 2 / (n_01 + n_10)
        p_value = 1 - chi2.cdf(statistic, df=1)

    result = {
        "statistic": float(statistic),
        "p_value": float(p_value),
        f"n_{model_a_name}_correct_{model_b_name}_wrong": int(n_01),
        f"n_{model_a_name}_wrong_{model_b_name}_correct": int(n_10),
        "significant": p_value < 0.05,
    }

    logger.info(
        f"McNemar's test ({model_a_name} vs {model_b_name}): "
        f"χ²={statistic:.4f}, p={p_value:.4f} "
        f"{'*** SIGNIFICANT ***' if p_value < 0.05 else '(not significant)'}"
    )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  src/classifiers.py — Self-test")
    print("=" * 60)

    np.random.seed(config.RANDOM_SEED)

    n_train, n_test = 100, 30
    d = 8

    X_train = np.random.rand(n_train, d) * np.pi
    y_train = np.random.randint(0, 5, n_train)
    X_test = np.random.rand(n_test, d) * np.pi
    y_test = np.random.randint(0, 5, n_test)

    # Test classical SVMs
    print("\nClassical SVMs:")
    results = train_classical_svms(X_train, y_train, X_test, y_test)
    for name, metrics in results.items():
        print(f"  {name}: macro-F1={metrics['macro_f1']:.4f}")

    # Test precomputed kernel SVM
    print("\nPrecomputed kernel SVM:")
    from sklearn.metrics.pairwise import rbf_kernel
    K_train = rbf_kernel(X_train)
    K_test = rbf_kernel(X_test, X_train)  # Rectangular!

    clf = train_precomputed_svm(K_train, y_train)
    metrics = evaluate_classifier(clf, K_test, y_test, "RBF-Precomputed")
    print(f"  macro-F1={metrics['macro_f1']:.4f}")

    # Test McNemar's
    print("\nMcNemar's test:")
    y_pred_a = np.random.randint(0, 5, n_test)
    y_pred_b = np.random.randint(0, 5, n_test)
    mcnemar = mcnemars_test(y_test, y_pred_a, y_pred_b, "ModelA", "ModelB")
    print(f"  p-value={mcnemar['p_value']:.4f}")

    print("\n  All self-tests passed.")
