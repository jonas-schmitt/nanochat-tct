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
echo "Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git wget curl htop tmux

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --quiet xgrammar

# Install TCT wheels if present
if [ -d "$WORKSPACE/tct-wheels" ]; then
    echo "Installing TCT wheels..."
    pip install --quiet "$WORKSPACE/tct-wheels"/*.whl
else
    echo "WARNING: tct-wheels not found at $WORKSPACE/tct-wheels"
fi

# Extract data if needed
if [ -f "$WORKSPACE/tct-experiment-data.tar.gz" ] && [ ! -d "$DATA_DIR/kubernetes-tct-bpe" ]; then
    echo "Extracting experiment data..."
    mkdir -p "$DATA_DIR"
    cd "$DATA_DIR"
    tar -xzf "$WORKSPACE/tct-experiment-data.tar.gz"
    cd "$WORKSPACE"
fi

# Clone or update code from GitHub
if [ ! -d "$CODE_DIR" ]; then
    echo "Cloning nanochat-tct from GitHub..."
    git clone https://github.com/jonas-schmitt/nanochat-tct.git "$CODE_DIR"
else
    echo "Updating nanochat-tct from GitHub..."
    cd "$CODE_DIR"
    git pull
fi

cd "$CODE_DIR"

# Copy merge tables if needed
if [ -d "$WORKSPACE/bpe-merges" ] && [ ! -d "$CODE_DIR/bpe-merges" ]; then
    echo "Copying BPE merge tables..."
    cp -r "$WORKSPACE/bpe-merges" "$CODE_DIR/"
fi

# Create checkpoints directory
mkdir -p "$CODE_DIR/checkpoints"

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
echo
echo "To run experiments:"
echo "  cd $CODE_DIR"
echo "  bash scripts/run_experiments.sh"
echo
