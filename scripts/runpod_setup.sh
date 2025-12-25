#!/bin/bash
# RunPod Pod Setup Script
# Run this after launching a pod with network volume attached at /workspace

set -e

echo "=== RunPod TCT Experiment Setup ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU detected')"
echo

WORKSPACE="/workspace"
DATA_DIR="$WORKSPACE/data"
CODE_DIR="$WORKSPACE/nanochat-tct"

# Check if network volume is mounted
if [ ! -d "$WORKSPACE" ]; then
    echo "ERROR: /workspace not found. Make sure network volume is attached!"
    exit 1
fi

cd "$WORKSPACE"

# Install system dependencies
echo "[1/5] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git wget curl htop tmux python3.12 python3.12-venv python3.12-dev
echo "      Done."

# Clone or update code from GitHub (needed for pyproject.toml)
echo "[2/5] Setting up code..."
if [ ! -d "$CODE_DIR" ]; then
    echo "      Cloning nanochat-tct from GitHub..."
    git clone https://github.com/jonas-schmitt/nanochat-tct.git "$CODE_DIR"
else
    echo "      Updating nanochat-tct from GitHub..."
    cd "$CODE_DIR"
    git pull
fi
echo "      Done."

# Install uv if not present
echo "[3/6] Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "      uv version: $(uv --version)"

# Create/activate venv with Python 3.12
echo "[4/6] Setting up Python 3.12 virtual environment..."
VENV_DIR="$WORKSPACE/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "      Creating venv..."
    uv venv --python 3.12 "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "      Using Python: $(python --version)"

# Install Python dependencies from pyproject.toml
echo "[5/6] Installing Python dependencies from pyproject.toml..."
cd "$CODE_DIR"
uv pip install -e ".[gpu,eval]"
# Install TCT wheels
if [ -d "$WORKSPACE/tct-wheels" ]; then
    uv pip install "$WORKSPACE/tct-wheels"/*.whl
else
    echo "      WARNING: tct-wheels not found at $WORKSPACE/tct-wheels"
fi
echo "      Done."

# Extract data if needed
echo "[6/6] Setting up data..."
if [ -f "$WORKSPACE/tct-experiment-data.tar.gz" ] && [ ! -d "$DATA_DIR/kubernetes-tct-bpe" ]; then
    echo "      Extracting experiment data..."
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"
    tar -xvzf "$WORKSPACE/tct-experiment-data.tar.gz"
    cd "$CODE_DIR"
    echo "      Data extracted."
else
    echo "      Data already extracted."
fi

# Copy merge tables if needed
if [ -d "$WORKSPACE/bpe-merges" ] && [ ! -d "$CODE_DIR/bpe-merges" ]; then
    echo "      Copying BPE merge tables..."
    cp -r "$WORKSPACE/bpe-merges" "$CODE_DIR/"
fi

# Create checkpoints directory
mkdir -p "$CODE_DIR/checkpoints"
echo "      Done."

# Verify setup
echo
echo "=== Setup Verification ==="
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo "VRAM: $(python -c 'import torch; print(f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if torch.cuda.is_available() else "None")')"
echo

# Check data
echo "=== Data Check ==="
for dir in kubernetes-tct-bpe kubernetes-utf8-bpe eslintrc-tct-bpe-10k eslintrc-utf8-bpe-10k tsconfig-tct-bpe-10k tsconfig-utf8-bpe-10k; do
    if [ -d "$DATA_DIR/$dir" ]; then
        echo "  [OK] $dir"
    else
        echo "  [MISSING] $dir"
    fi
done
echo

# Check TCT wheels
echo "=== TCT Wheels Check ==="
python -c "import tct_kubernetes_20k; print('  [OK] tct_kubernetes_20k')" 2>/dev/null || echo "  [MISSING] tct_kubernetes_20k"
python -c "import tct_eslintrc_10k; print('  [OK] tct_eslintrc_10k')" 2>/dev/null || echo "  [MISSING] tct_eslintrc_10k"
python -c "import tct_tsconfig_10k; print('  [OK] tct_tsconfig_10k')" 2>/dev/null || echo "  [MISSING] tct_tsconfig_10k"
echo

echo "=== Setup Complete ==="
echo "Code directory: $CODE_DIR"
echo "Data directory: $DATA_DIR"
echo "Venv directory: $VENV_DIR"
echo
echo "Next steps:"
echo "  source /workspace/venv/bin/activate  # Activate venv (if new shell)"
echo "  cd $CODE_DIR"
echo "  bash scripts/test_setup.sh           # Verify setup"
echo "  bash scripts/run_tsconfig.sh         # Run tsconfig experiments"
echo "  bash scripts/run_eslintrc.sh         # Run eslintrc experiments"
echo "  bash scripts/run_kubernetes.sh       # Run kubernetes experiments"
echo
