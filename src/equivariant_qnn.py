"""
Equivariant Quantum Neural Network for Quantum-Sat Classification.

Implements:
    - 2×4 qubit grid with Klein V₄ symmetry group.
    - Equivariant QNN: Parameters shared under group orbits.
    - Non-equivariant QNN baseline: All parameters independent.
    - Classical equivariant MLP and standard MLP baselines.
    - Training with class-weighted cross-entropy loss (Rule I5).

Klein V₄ symmetry on 2×4 grid:
    Grid layout:  q0 q1 q2 q3
                  q4 q5 q6 q7

    H-reflection: q0↔q3, q1↔q2, q4↔q7, q5↔q6
    V-reflection: q0↔q4, q1↔q5, q2↔q6, q3↔q7
    Composition:  q0↔q7, q1↔q6, q2↔q5, q3↔q4

Reference: Equivariant quantum circuits for QML.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

logger = logging.getLogger(__name__)

# ============ KLEIN V₄ SYMMETRY ============

# Grid layout (row-major):
# q0 q1 q2 q3
# q4 q5 q6 q7

# H-reflection (horizontal flip): left↔right
H_REFLECT = {0: 3, 1: 2, 2: 1, 3: 0, 4: 7, 5: 6, 6: 5, 7: 4}

# V-reflection (vertical flip): top↔bottom
V_REFLECT = {0: 4, 1: 5, 2: 6, 3: 7, 4: 0, 5: 1, 6: 2, 7: 3}

# HV composition
HV_REFLECT = {
    i: V_REFLECT[H_REFLECT[i]] for i in range(8)
}

# Parameter orbits under V₄
# Each orbit is a set of qubit indices that share parameters
V4_ORBITS = [
    {0, 3, 4, 7},  # Corners
    {1, 2, 5, 6},  # Middle
]

# Grid-neighbor connectivity (2×4 grid)
GRID_EDGES = [
    (0, 1), (1, 2), (2, 3),  # Top row
    (4, 5), (5, 6), (6, 7),  # Bottom row
    (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical connections
]


def get_orbit_index(qubit: int) -> int:
    """
    Get the parameter orbit index for a qubit under V₄ symmetry.

    Args:
        qubit: Qubit index (0-7).

    Returns:
        int: Orbit index (0 or 1).
    """
    for idx, orbit in enumerate(V4_ORBITS):
        if qubit in orbit:
            return idx
    raise ValueError(f"Qubit {qubit} not in any orbit")


# ============ EQUIVARIANT QNN ============

class EquivariantQNN(nn.Module):
    """
    Equivariant Quantum Neural Network with V₄ symmetry on 2×4 grid.

    Circuit: RY(x_i) → N_LAYERS equivariant → ⟨Z⟩ per qubit → linear(8→17) → softmax.
    Parameters are shared within group orbits (2 unique params per layer vs 8).

    Args:
        n_qubits: Number of qubits (default: 8).
        n_layers: Number of variational layers.
        n_classes: Number of output classes.
        equivariant: If True, share parameters (2 per layer).
                     If False, all independent (8 per layer).
    """

    def __init__(
        self,
        n_qubits: int = config.N_QUBITS,
        n_layers: int = config.N_EQUIVARIANT_LAYERS,
        n_classes: int = config.LCZ_N_CLASSES,
        equivariant: bool = True,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.equivariant = equivariant

        # Number of unique rotation parameters per layer
        if equivariant:
            n_params_per_layer = len(V4_ORBITS)  # 2 orbits
        else:
            n_params_per_layer = n_qubits  # 8 independent

        self.params = nn.ParameterList([
            nn.Parameter(torch.randn(n_params_per_layer) * 0.1)
            for _ in range(n_layers)
        ])

        # Classical post-processing: qubit measurements → classes
        self.classifier = nn.Linear(n_qubits, n_classes)

        self._build_circuit()

    def _build_circuit(self):
        """Build the PennyLane QNode."""
        import pennylane as qml

        dev = config.get_device(self.n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def circuit(inputs, params_list, equivariant, n_qubits, n_layers):
            # Data encoding: RY(x_i) on each qubit
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            for layer in range(n_layers):
                layer_params = params_list[layer]

                # RY rotations with parameter sharing
                for q in range(n_qubits):
                    if equivariant:
                        orbit_idx = get_orbit_index(q)
                        qml.RY(layer_params[orbit_idx], wires=q)
                    else:
                        qml.RY(layer_params[q], wires=q)

                # Entangling layer: CNOT on grid edges
                for q1, q2 in GRID_EDGES:
                    qml.CNOT(wires=[q1, q2])

            # Measurements: ⟨Z⟩ per qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x → quantum circuit → measurements → classifier → logits.

        Args:
            x: Input features, shape (batch, n_qubits).

        Returns:
            torch.Tensor: Class logits, shape (batch, n_classes).
        """
        batch_size = x.shape[0]
        measurements = []

        for i in range(batch_size):
            meas = self._circuit(
                x[i], list(self.params),
                self.equivariant, self.n_qubits, self.n_layers,
            )
            measurements.append(torch.stack(meas))

        measurements = torch.stack(measurements)  # (batch, n_qubits)
        logits = self.classifier(measurements)  # (batch, n_classes)

        return logits

    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable parameters.

        Returns:
            dict: {'quantum': int, 'classical': int, 'total': int}
        """
        quantum_params = sum(p.numel() for p in self.params)
        classical_params = sum(
            p.numel() for p in self.classifier.parameters()
        )
        return {
            "quantum": quantum_params,
            "classical": classical_params,
            "total": quantum_params + classical_params,
        }


# ============ CLASSICAL BASELINES ============

class ClassicalMLP(nn.Module):
    """
    Standard classical MLP baseline.

    Architecture: input(8) → hidden(64) → ReLU → hidden(32) → ReLU → output(17).

    Args:
        n_input: Input dimension.
        n_classes: Output classes.
        hidden_sizes: Hidden layer sizes.
    """

    def __init__(
        self,
        n_input: int = config.N_QUBITS,
        n_classes: int = config.LCZ_N_CLASSES,
        hidden_sizes: List[int] = None,
    ):
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        layers = []
        prev = n_input
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)])
            prev = h
        layers.append(nn.Linear(prev, n_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EquivariantMLP(nn.Module):
    """
    Classical equivariant MLP baseline.

    Shares weights according to V₄ orbits (similar parameter reduction
    as equivariant QNN).

    Args:
        n_input: Input dimension.
        n_classes: Output classes.
    """

    def __init__(
        self,
        n_input: int = config.N_QUBITS,
        n_classes: int = config.LCZ_N_CLASSES,
    ):
        super().__init__()

        # Create orbit-aware input: average features within each orbit
        n_orbits = len(V4_ORBITS)

        self.network = nn.Sequential(
            nn.Linear(n_orbits, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Average features within orbits
        orbit_features = []
        for orbit in V4_ORBITS:
            orbit_idx = list(orbit)
            orbit_mean = x[:, orbit_idx].mean(dim=1, keepdim=True)
            orbit_features.append(orbit_mean)

        x_orbit = torch.cat(orbit_features, dim=1)  # (batch, n_orbits)
        return self.network(x_orbit)


# ============ TRAINING ============

def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs: int = config.EQUIVARIANT_EPOCHS,
    lr: float = config.EQUIVARIANT_LR,
    batch_size: int = config.BATCH_SIZE,
    seed: int = config.RANDOM_SEED,
) -> Dict:
    """
    Train a model with class-weighted cross-entropy (Rule I5).

    Args:
        model: PyTorch model.
        X_train: Training features.
        y_train: Training labels (integers).
        X_val: Validation features (optional).
        y_val: Validation labels (optional).
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        seed: Random seed.

    Returns:
        dict: Training history with losses and accuracies.
    """
    torch.manual_seed(seed)

    # Class-weighted loss (Rule I5)
    class_counts = np.bincount(y_train, minlength=config.LCZ_N_CLASSES)
    # Avoid division by zero for missing classes
    class_counts = np.where(class_counts == 0, 1, class_counts)
    class_weights = len(y_train) / (config.LCZ_N_CLASSES * class_counts.astype(float))
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create DataLoader
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

        history["train_loss"].append(epoch_loss / total)
        history["train_acc"].append(correct / total)

        # Validation
        if X_val is not None and y_val is not None:
            model.eval()
            with torch.no_grad():
                X_val_t = torch.FloatTensor(X_val)
                y_val_t = torch.LongTensor(y_val)
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                val_preds = val_logits.argmax(dim=1)
                val_acc = (val_preds == y_val_t).float().mean().item()
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = (
                f"  Epoch {epoch+1}/{epochs}: "
                f"loss={history['train_loss'][-1]:.4f}, "
                f"acc={history['train_acc'][-1]:.4f}"
            )
            if X_val is not None:
                msg += f", val_loss={history['val_loss'][-1]:.4f}, val_acc={history['val_acc'][-1]:.4f}"
            logger.info(msg)

    return history


def predict(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int = config.BATCH_SIZE,
) -> np.ndarray:
    """
    Generate predictions from a trained model.

    Args:
        model: Trained PyTorch model.
        X: Features, shape (n, d).
        batch_size: Batch size for inference.

    Returns:
        np.ndarray: Predicted labels, shape (n,).
    """
    model.eval()
    all_preds = []

    X_tensor = torch.FloatTensor(X)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (batch,) in dataloader:
            logits = model(batch)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.numpy())

    return np.concatenate(all_preds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  src/equivariant_qnn.py — Self-test")
    print("=" * 60)

    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    # Test orbits
    print(f"\n  V₄ orbits: {V4_ORBITS}")
    for q in range(8):
        print(f"    q{q} → orbit {get_orbit_index(q)}")

    # Test Classical MLP
    mlp = ClassicalMLP()
    X_dummy = torch.randn(5, 8)
    out = mlp(X_dummy)
    print(f"\n  MLP output shape: {out.shape} (expected (5, 17))")
    assert out.shape == (5, 17)

    # Test Equivariant MLP
    eq_mlp = EquivariantMLP()
    out_eq = eq_mlp(X_dummy)
    print(f"  Equivariant MLP output: {out_eq.shape}")

    # Test parameter counts
    mlp_params = sum(p.numel() for p in mlp.parameters())
    eq_mlp_params = sum(p.numel() for p in eq_mlp.parameters())
    print(f"  MLP params: {mlp_params}, Equivariant MLP params: {eq_mlp_params}")
    assert eq_mlp_params < mlp_params, "Equivariant should have fewer params"

    # Quick training test with MLP (skip QNN for speed)
    X_train = np.random.rand(50, 8).astype(np.float32) * np.pi
    y_train = np.random.randint(0, 17, 50)

    history = train_model(mlp, X_train, y_train, epochs=5, batch_size=16)
    print(f"\n  Training loss (5 epochs): {history['train_loss'][-1]:.4f}")

    preds = predict(mlp, X_train)
    print(f"  Predictions shape: {preds.shape}")

    print("\n  All self-tests passed.")
