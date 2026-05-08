"""
Central configuration for Geometric Quantum Kernel Analysis pipeline.

Contains all paths, hyperparameters, random seeds, device factory,
dataset configuration, and class imbalance handling constants.
All scripts import from this module — no hardcoded values elsewhere.

Target: So2Sat LCZ42 — 17-Class Local Climate Zone Classification
Hardware: RTX 4050 (6GB VRAM), Intel i5-13500HX, 16GB RAM, WSL2, CUDA 12
Constraint: Maximum 8 qubits for simulation; IBM Quantum hardware validation
"""

import os
import pennylane as qml

# ============ PATHS ============
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TOPO_DIR = os.path.join(DATA_DIR, "topological")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ============ RANDOM SEED ============
RANDOM_SEED = 42
SEED_LIST = [42, 43, 44, 45, 46]

# ============ QUANTUM ============
N_QUBITS = 8
ZZ_REPS = 2
# PQK bandwidth: set to inverse median pairwise Frobenius distance in Bloch space.
# Diagnostic (diagnose_pqk.py) showed mean Frobenius distance ≈ 2.22 on SAR data.
# gamma=1.0 (generic default) caused PQK to appear more concentrated than FQK,
# contradicting Thanasilp et al. (2024). Median heuristic gives gamma ≈ 0.67,
# but empirical CV analysis shows gamma=2.0 fully recovers the theoretical prediction.
# We use gamma=0.67 (principled median heuristic) to avoid tuning bias.
# Full sensitivity table reported in paper supplementary (diagnose_pqk.py output).
PQK_GAMMA = 0.67
GEOMETRIC_DIFF_LAMBDA = 1e-6

# ============ TFK (TRAINED FIDELITY KERNEL) ============
TFK_N_EPOCHS = 75  # Reduced from 100: higher-quality gradients at subset_size=68
# converge faster. 75 epochs × ~109s ≈ 136 min total.
TFK_SUBSET_SIZE = 51  # 3 samples per class × 17 classes.
# Binding constraint: class 6 has only 22 samples
# in the 2000-sample subsample. 3/class gives
# 3 intra-class pairs/class — minimum for
# meaningful ideal kernel block structure.
TFK_N_PAIR_SAMPLES = 1275  # ALL pairs — no stochastic sampling.
# n*(n-1)/2 = 51*50/2 = 1275.
# Full gradient matches Hubregtsen et al. (2022).
TFK_LR = 0.02  # Chosen manually from LR sweep: provides strong baseline KTA gain
# (+0.065 over 75 epochs) while safely projecting to ||theta-1|| ≈ 3.05,
# staying comfortably away from the ±2π divergence cliff.
TFK_N_PARAMS = 8  # One scaling parameter per qubit


# ============ HYPERPARAMETER SWEEP ============
PQK_GAMMA_LIST = [0.1, 0.5, 1.0, 2.0, 5.0]
ZZ_REPS_LIST = [1, 2, 3]

# ============ DATA ============
N_PCA_COMPONENTS = 8
PCA_FIT_SAMPLES = 15000  # Samples for PCA fitting (15k is plenty for 8 components)
SUBSAMPLE_TRAIN = 2000  # Quantum kernel train size (O(n²) constraint)
SUBSAMPLE_TEST = 2000  # Test size (≥100 avg per class for 17 classes)
FEWSHOT_SIZES = [50, 100, 200, 500, 1000, 2000]
BATCH_SIZE = 32

# Quantum kernels computed only for these modalities (Optical PCA=0.84 → classical sufficient)
QUANTUM_MODALITIES = {"sar", "fused"}

# ============ EQUIVARIANT ============
N_EQUIVARIANT_LAYERS = 3
EQUIVARIANT_LR = 0.01
EQUIVARIANT_EPOCHS = 50

# ============ TOPOLOGICAL ============
TOPOLOGICAL_MAX_DIM = 1
TOPOLOGICAL_MAX_EDGE = 2.0
PERSISTENCE_IMAGE_RESOLUTION = [20, 20]
PERSISTENCE_IMAGE_SIGMA = 0.1

# ============ DEQUANTIZATION ============
RFF_N_FEATURES_LIST = [64, 128, 256, 512]

# ============ CROSS-VALIDATION ============
CV_FOLDS = 5
CV_SEED = 42  # Seed for StratifiedKFold split

# ============ CONTROLLED MI ============
MIXING_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CONTROLLED_MI_NOISE_SEEDS = [42, 43, 44]

# ============ EXP10 V2 (REDESIGNED) ============
# Controlled MI experiment: N=300 with 5 noise seeds per alpha.
# N=300 matches Bowles et al. (2024, N=250) and exceeds Hubregtsen et al.
# (2022, N=30-60). 5 seeds provide ~0.95 statistical power for Spearman
# trend test. Total: 55 evaluation points (11 alphas × 5 seeds).
# Fits within a single 9-hour Kaggle session (~6.9 hrs per kernel).
EXP10_V2_N_TRAIN = 300
EXP10_V2_N_TEST = 300
EXP10_V2_NOISE_SEEDS = list(range(42, 47))  # 5 seeds: 42-46
EXP10_V2_ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EXP10_V2_USE_CV = True  # 5-fold CV at each alpha

# ============ EXP10 SPECIFIC (ORIGINAL) ============
# Use smaller N for Exp10: at N=2000, FQK concentration destroys the signal.
# At N=200, kernel matrices have genuine discriminative structure.
# Reference: Bowles et al. (2024) use N=250 for 8-qubit benchmark.
N_EXP10_TRAIN = 200
N_EXP10_TEST = 100

# ============ TFK ANCHOR SENSITIVITY ============
TFK_ANCHOR_SIZES = [50, 102, 200, 300, 500]

# ============ EXPRESSIBILITY ============
EXPRESSIBILITY_N_SAMPLES = 1000

# CM-FQK Settings
CM_FQK_TOP_K_PAIRS: int = 3  # Top MI pair cross-modal interactions for CM-FQK

# ============ HARDWARE ============
IBM_QUANTUM_CHANNEL = "ibm_quantum"
IBM_SHOTS = 4096
IBM_OPTIMIZATION_LEVEL = 3
IBM_PILOT_N_TRAIN = 10
IBM_FULL_N_TRAIN = 15
IBM_FULL_N_TEST = 10

# ============ DEVICE FACTORY ============
_DEVICE_CACHE = {}
_DEVICE_INITIALIZED = False


def get_device(wires, shots=None):
    """
    Get a PennyLane quantum device.

    Tries devices in order: lightning.gpu → lightning.kokkos → lightning.qubit.
    Caches per (wires, shots) pair. On Kaggle P100 (sm_60), only lightning.qubit
    works — GPU backends require sm_70+.

    Args:
        wires: Number of qubits.
        shots: Number of measurement shots (None for analytic).

    Returns:
        qml.Device: PennyLane device instance.
    """
    global _DEVICE_INITIALIZED
    cache_key = (wires, shots)
    if cache_key in _DEVICE_CACHE:
        return _DEVICE_CACHE[cache_key]

    if not _DEVICE_INITIALIZED:
        # Test all devices once at startup
        _DEVICE_INITIALIZED = True
        for dev_name in ["lightning.gpu", "lightning.kokkos", "lightning.qubit"]:
            try:
                dev = qml.device(dev_name, wires=wires, shots=shots)

                @qml.qnode(dev)
                def _test(x):
                    qml.RY(x, wires=0)
                    return qml.expval(qml.PauliZ(0))

                _test(0.1)
                print(f"[DEVICE] Using {dev_name} with {wires} wires")
                _DEVICE_CACHE[cache_key] = dev
                return dev
            except Exception:
                continue
        # Final fallback
        dev = qml.device("default.qubit", wires=wires, shots=shots)
        print(f"[DEVICE] Using default.qubit (CPU) with {wires} wires")
        _DEVICE_CACHE[cache_key] = dev
        return dev
    else:
        # Already tested — use cached device type
        # Just create a new device with the same backend
        try:
            dev = qml.device("lightning.qubit", wires=wires, shots=shots)
        except Exception:
            dev = qml.device("default.qubit", wires=wires, shots=shots)
        _DEVICE_CACHE[cache_key] = dev
        return dev


def get_ibm_device(wires):
    """
    Get an IBM Quantum hardware device via qiskit.remote.

    Uses QiskitRuntimeService to find the least busy real backend
    with at least `wires` qubits.

    Args:
        wires: Number of qubits required.

    Returns:
        qml.Device: PennyLane device connected to IBM hardware.

    Raises:
        ImportError: If qiskit-ibm-runtime is not installed.
        RuntimeError: If no suitable backend is found.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(channel=IBM_QUANTUM_CHANNEL)
    backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=wires
    )
    dev = qml.device("qiskit.remote", wires=wires, backend=backend, shots=IBM_SHOTS)
    print(
        f"[DEVICE] Using IBM Quantum: {backend.name} with {wires} wires, {IBM_SHOTS} shots"
    )
    return dev


# ============ LOCAL DATASET ============
LOCAL_SO2SAT_FOLDER = "So2sat Full"
LOCAL_DATASET_SEARCH_PATHS = [
    os.path.join(PROJECT_ROOT, "So2sat Full"),
    os.path.join(RAW_DIR, "So2sat Full"),
    os.path.expanduser("~/So2sat Full"),
    os.path.expanduser("~/Downloads/So2sat Full"),
    os.path.join(PROJECT_ROOT, "So2sat_Full"),
    os.path.join(RAW_DIR, "So2sat_Full"),
    # Also check parent directory (common layout)
    os.path.join(os.path.dirname(PROJECT_ROOT), "So2sat Full"),
    os.path.join(os.path.dirname(PROJECT_ROOT), "So2Sat Full"),
]

# ============ CLASS NAMES (ALL 17 LCZ CLASSES) ============
LCZ_N_CLASSES = 17
LCZ_CLASS_NAMES = [
    "Compact High Rise",  # 0
    "Compact Mid Rise",  # 1
    "Compact Low Rise",  # 2
    "Open High Rise",  # 3
    "Open Mid Rise",  # 4
    "Open Low Rise",  # 5
    "Lightweight Low Rise",  # 6
    "Large Low Rise",  # 7
    "Sparsely Built",  # 8
    "Heavy Industry",  # 9
    "Dense Trees",  # 10
    "Scattered Trees",  # 11
    "Bush/Scrub",  # 12
    "Low Plants",  # 13
    "Bare Rock/Paved",  # 14
    "Bare Soil/Sand",  # 15
    "Water",  # 16
]


if __name__ == "__main__":
    print("=" * 60)
    print("Quantum-Sat Classification — Configuration Summary")
    print("=" * 60)
    print(f"  PROJECT_ROOT:    {PROJECT_ROOT}")
    print(f"  DATA_DIR:        {DATA_DIR}")
    print(f"  RESULTS_DIR:     {RESULTS_DIR}")
    print(f"  N_QUBITS:        {N_QUBITS}")
    print(f"  ZZ_REPS:         {ZZ_REPS}")
    print(f"  PQK_GAMMA:       {PQK_GAMMA}")
    print(f"  RANDOM_SEED:     {RANDOM_SEED}")
    print(f"  SEED_LIST:       {SEED_LIST}")
    print(f"  N_PCA_COMPONENTS: {N_PCA_COMPONENTS}")
    print(f"  SUBSAMPLE_TRAIN: {SUBSAMPLE_TRAIN}")
    print(f"  SUBSAMPLE_TEST:  {SUBSAMPLE_TEST}")
    print(f"  LCZ_N_CLASSES:   {LCZ_N_CLASSES}")
    print(f"  LCZ_CLASS_NAMES: {LCZ_CLASS_NAMES}")
    print()
    print("Local dataset search paths:")
    for p in LOCAL_DATASET_SEARCH_PATHS:
        exists = os.path.isdir(p)
        print(f"  {'✓' if exists else '✗'} {p}")
    print()
    print("Testing device factory...")
    dev = get_device(N_QUBITS)
    print(f"  Device: {dev.name}")
    print()
    print("Configuration OK.")
