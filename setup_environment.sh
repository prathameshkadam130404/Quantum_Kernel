#!/bin/bash
# ============================================================================
# Quantum-Sat Classification — WSL2 Environment Setup
# ============================================================================
# Run this ONCE inside WSL2:
#   bash setup_environment.sh
#
# Prerequisites:
#   - Windows with WSL2 installed
#   - NVIDIA GPU with Windows driver supporting WSL (RTX 4050)
#   - This script installs Python 3.11, CUDA toolkit, and all dependencies
# ============================================================================

set -e  # Exit on any error

echo "============================================"
echo "  Quantum-Sat Classification — Setup"
echo "============================================"
echo ""

# ============ STEP 1: Verify GPU access ============
echo "=== Step 1/6: Checking GPU access ==="
nvidia-smi
if [ $? -ne 0 ]; then
    echo "ERROR: nvidia-smi failed. Install NVIDIA Windows driver with WSL support."
    echo "Download from: https://developer.nvidia.com/cuda/wsl"
    exit 1
fi
echo "GPU detected successfully."
echo ""

# ============ STEP 2: Install system dependencies ============
echo "=== Step 2/6: Installing system dependencies ==="
sudo apt update && sudo apt install -y software-properties-common \
    build-essential cmake git wget curl libopenblas-dev

# Add deadsnakes PPA for Python 3.11
echo "Adding deadsnakes PPA for Python 3.11..."
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update

# Try to install Python 3.11; if it fails, detect best available
if sudo apt install -y python3.11 python3.11-venv python3.11-dev; then
    PYTHON_BIN="python3.11"
    echo "Python 3.11 installed successfully."
else
    echo "WARNING: Python 3.11 not available. Detecting best available version..."
    # Find the best available python3.x (prefer 3.11 > 3.10 > 3.9 > 3.12)
    for v in python3.11 python3.10 python3.12 python3.9 python3; do
        if command -v "$v" &>/dev/null; then
            PYTHON_BIN="$v"
            # Ensure venv is available for this version
            PY_MINOR=$("$v" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            sudo apt install -y "python${PY_MINOR}-venv" "python${PY_MINOR}-dev" 2>/dev/null || true
            break
        fi
    done

    if [ -z "$PYTHON_BIN" ]; then
        echo "ERROR: No suitable Python 3.x found. Please install Python 3.10+ manually."
        exit 1
    fi
    echo "Using: $PYTHON_BIN ($($PYTHON_BIN --version))"
fi

echo "System dependencies installed."
echo ""

# ============ STEP 3: Create Python virtual environment ============
echo "=== Step 3/6: Creating Python virtual environment ==="
if [ -d "$HOME/quantum-sat-env" ]; then
    echo "Virtual environment already exists at ~/quantum-sat-env"
else
    $PYTHON_BIN -m venv ~/quantum-sat-env
    echo "Virtual environment created at ~/quantum-sat-env"
fi
source ~/quantum-sat-env/bin/activate
echo "Activated: $(which python) ($(python --version))"
echo ""

# ============ STEP 4: Install CUDA toolkit inside WSL ============
echo "=== Step 4/6: Checking CUDA toolkit ==="
if ! command -v nvcc &>/dev/null; then
    echo "Installing CUDA toolkit..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-6
    rm -f cuda-keyring_1.1-1_all.deb
    echo "CUDA toolkit installed."
else
    echo "CUDA toolkit already installed."
fi
echo ""

# ============ STEP 5: Install Python packages ============
echo "=== Step 5/6: Installing Python packages ==="
echo "This may take 10-15 minutes..."
echo ""

pip install --upgrade pip setuptools wheel

echo "--- Installing core scientific packages ---"
pip install numpy scipy pandas matplotlib seaborn tqdm joblib

echo "--- Installing scikit-learn ---"
pip install scikit-learn

echo "--- Installing PyTorch (CUDA 12.1) ---"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "--- Installing PennyLane ---"
pip install pennylane pennylane-lightning

echo "--- Installing PennyLane GPU backend ---"
# CRITICAL: Use pennylane-lightning-gpu (separate package, latest 0.44.0)
# Do NOT use pennylane-lightning[gpu] — that is deprecated
pip install pennylane-lightning-gpu

echo "--- Installing Qiskit ---"
# pennylane-qiskit 0.44.1 uses qiskit.remote device with Qiskit 2.x
pip install pennylane-qiskit qiskit qiskit-ibm-runtime

echo "--- Installing Giotto-TDA ---"
pip install giotto-tda

echo "--- Installing H5py ---"
pip install h5py

echo "--- Installing TensorFlow Datasets (fallback) ---"
pip install tensorflow-datasets

echo "--- Installing pytest ---"
pip install pytest

echo ""
echo "All Python packages installed."
echo ""

# ============ STEP 6: Create project directories ============
echo "=== Step 6/6: Creating project directories ==="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$SCRIPT_DIR/data/raw"
mkdir -p "$SCRIPT_DIR/data/processed"
mkdir -p "$SCRIPT_DIR/data/topological"
mkdir -p "$SCRIPT_DIR/results/geometric_difference"
mkdir -p "$SCRIPT_DIR/results/classification"
mkdir -p "$SCRIPT_DIR/results/fewshot"
mkdir -p "$SCRIPT_DIR/results/topological"
mkdir -p "$SCRIPT_DIR/results/equivariant"
mkdir -p "$SCRIPT_DIR/results/dequantization"
mkdir -p "$SCRIPT_DIR/results/hardware"
mkdir -p "$SCRIPT_DIR/results/multidataset"
mkdir -p "$SCRIPT_DIR/results/ablation"
mkdir -p "$SCRIPT_DIR/results/controlled_mi"
mkdir -p "$SCRIPT_DIR/results/expressibility"
mkdir -p "$SCRIPT_DIR/paper/figures"
mkdir -p "$SCRIPT_DIR/paper/tables"
mkdir -p "$SCRIPT_DIR/theory"
mkdir -p "$SCRIPT_DIR/notebooks"
echo "Project directories created."
echo ""

# ============ DONE ============
echo "============================================"
echo "  Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Activate the environment:"
echo "       source ~/quantum-sat-env/bin/activate"
echo ""
echo "  2. Verify installation:"
echo "       python verify_install.py"
echo ""
echo "  3. Prepare data:"
echo "       python setup_data.py"
echo ""
echo "  4. Run tests:"
echo "       python -m pytest tests/ -v"
echo ""
