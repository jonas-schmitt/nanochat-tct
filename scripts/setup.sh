#!/bin/bash -l
# Unified Setup Script for TCT Experiments
# Works on: RunPod, NHR (Alex/TinyGPU), local machines
#
# Usage (interactive):
#   bash scripts/setup.sh
#
# On NHR, run in an interactive job:
#   srun --partition=a100 --gres=gpu:1 --time=01:00:00 --pty bash -l
#   bash scripts/setup.sh

set -e

echo "============================================================"
echo "TCT Experiment Setup"
echo "============================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo

# =============================================================================
# Detect platform and set paths
# =============================================================================

detect_platform() {
    if [ -d "/workspace" ] && [ -w "/workspace" ]; then
        echo "runpod"
    elif [ -n "$WORK" ] && [ -d "$WORK" ]; then
        echo "nhr"
    elif [ -n "$SCRATCH" ] && [ -d "$SCRATCH" ]; then
        echo "hpc"
    else
        echo "local"
    fi
}

PLATFORM=$(detect_platform)
echo "Detected platform: $PLATFORM"

# CODE_DIR is always computed from script location - works regardless of where repo is cloned
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# DATA_DIR is always sibling of CODE_DIR - consistent across all platforms
DATA_DIR="$(dirname "$CODE_DIR")/data"

case $PLATFORM in
    runpod)
        VENV_DIR="/workspace/venv"
        ;;
    nhr)
        VENV_DIR="${WORK:-$(dirname "$CODE_DIR")}/venv-tct"
        ;;
    hpc)
        VENV_DIR="${SCRATCH:-$(dirname "$CODE_DIR")}/venv-tct"
        ;;
    local)
        VENV_DIR="$CODE_DIR/.venv"
        ;;
esac

echo "Code dir:  $CODE_DIR"
echo "Data dir:  $DATA_DIR"
echo "Venv dir:  $VENV_DIR"
echo

# =============================================================================
# Platform-specific setup
# =============================================================================

echo "[1/6] Platform-specific setup..."

case $PLATFORM in
    runpod)
        # Install system dependencies
        apt-get update -qq && apt-get install -y -qq git wget curl htop tmux python3.12 python3.12-venv python3.12-dev 2>/dev/null || true
        ;;
    nhr)
        # Load modules (per NHR docs: https://doc.nhr.fau.de/environment/python-env/)
        module purge 2>/dev/null || true
        module load cuda 2>/dev/null || true

        # Try multiple Python module options (in order of preference)
        PYTHON_LOADED=""
        for pymod in "python/3.12" "python/3.11" "python/3.10" "python"; do
            if module load "$pymod" 2>/dev/null; then
                echo "Loaded module: $pymod"
                PYTHON_LOADED="$pymod"
                break
            fi
        done

        if [ -z "$PYTHON_LOADED" ]; then
            echo "WARNING: No Python module loaded, trying system Python"
        fi

        echo "Loaded modules:"
        module list 2>&1 | head -10

        # Verify Python is available
        if ! command -v python3 &>/dev/null; then
            echo "ERROR: python3 not found after loading modules"
            echo "Available Python modules:"
            module avail python 2>&1 | head -20
            exit 1
        fi
        echo "Python: $(python3 --version)"

        # Configure paths to use $WORK (not $HOME - quota issues)
        export PYTHONUSERBASE="$WORK/software/private"
        mkdir -p "$PYTHONUSERBASE"

        # Set proxy for pip/uv (required on compute nodes)
        export http_proxy=http://proxy.nhr.fau.de:80
        export https_proxy=http://proxy.nhr.fau.de:80
        export HTTP_PROXY=http://proxy.nhr.fau.de:80
        export HTTPS_PROXY=http://proxy.nhr.fau.de:80
        echo "Proxy configured for NHR"
        ;;
    hpc)
        # Generic HPC - try common module names
        module purge 2>/dev/null || true
        module load python 2>/dev/null || true
        module load cuda 2>/dev/null || true
        ;;
    local)
        echo "Local setup - assuming dependencies are installed"
        ;;
esac
echo "Done."

# =============================================================================
# Clone or update code
# =============================================================================

echo "[2/6] Setting up code..."
if [ "$PLATFORM" != "local" ]; then
    if [ ! -d "$CODE_DIR" ]; then
        echo "Cloning nanochat-tct..."
        git clone https://github.com/jonas-schmitt/nanochat-tct.git "$CODE_DIR"
    else
        echo "Updating nanochat-tct..."
        cd "$CODE_DIR"
        git pull
    fi
fi
cd "$CODE_DIR"
echo "Done."

# =============================================================================
# Setup Python environment
# =============================================================================

echo "[3/6] Setting up Python environment..."

# Try to use uv, fall back to pip
USE_UV=""
export PATH="$HOME/.local/bin:$PATH"

if command -v uv &> /dev/null; then
    echo "Found uv: $(uv --version)"
    USE_UV="1"
else
    echo "uv not found, attempting to install..."
    if curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null; then
        if command -v uv &> /dev/null; then
            echo "Installed uv: $(uv --version)"
            USE_UV="1"
        fi
    fi
    if [ -z "$USE_UV" ]; then
        echo "uv installation failed, will use pip instead"
    fi
fi

# Create virtual environment (per NHR docs: use venv, not conda)
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    if [ -n "$USE_UV" ]; then
        case $PLATFORM in
            runpod)
                uv venv --python 3.12 "$VENV_DIR" || python3.12 -m venv "$VENV_DIR"
                ;;
            nhr|hpc)
                # Use the Python from loaded module
                uv venv --python python3 "$VENV_DIR" || python3 -m venv "$VENV_DIR"
                ;;
            local)
                uv venv "$VENV_DIR" || python3 -m venv "$VENV_DIR"
                ;;
        esac
    else
        python3 -m venv "$VENV_DIR"
    fi
else
    echo "Virtual environment already exists at $VENV_DIR"
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "ERROR: Failed to create virtual environment at $VENV_DIR"
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Python: $(python --version)"
echo "Done."

# =============================================================================
# Install Python dependencies
# =============================================================================

echo "[4/6] Installing Python dependencies..."
cd "$CODE_DIR"

# Helper function for package installation
pip_install() {
    if [ -n "$USE_UV" ]; then
        uv pip install "$@" || pip install "$@"
    else
        pip install "$@"
    fi
}

pip_install -e ".[gpu,eval]"
echo "Done."

# =============================================================================
# Install TCT wheels
# =============================================================================

echo "[5/6] Installing TCT wheels..."
TCT_WHEELS=""
if [ -d "$WORKSPACE/tct-wheels" ]; then
    TCT_WHEELS="$WORKSPACE/tct-wheels"
elif [ -d "$DATA_DIR/../tct-wheels" ]; then
    TCT_WHEELS="$DATA_DIR/../tct-wheels"
elif [ -d "$CODE_DIR/tct-wheels" ]; then
    TCT_WHEELS="$CODE_DIR/tct-wheels"
fi

if [ -n "$TCT_WHEELS" ] && [ -d "$TCT_WHEELS" ]; then
    echo "Installing from $TCT_WHEELS..."
    pip_install "$TCT_WHEELS"/*.whl
else
    echo "WARNING: TCT wheels not found. You may need to install them manually."
fi
echo "Done."

# =============================================================================
# Setup data directories and extract archives if needed
# =============================================================================

echo "[6/6] Setting up data..."
mkdir -p "$DATA_DIR"
mkdir -p "$CODE_DIR/checkpoints"
mkdir -p "$CODE_DIR/logs"

# Required datasets (must match schema_configs.py)
DATASETS="tsconfig-tct-base tsconfig-utf8-base-matched eslintrc-tct-bpe-500 eslintrc-utf8-bpe-500 kubernetes-tct-bpe-1k kubernetes-utf8-bpe-1k"

# Check and extract missing datasets
extract_if_missing() {
    local name="$1"
    local archive="$CODE_DIR/data/${name}.tar.xz"
    local target="$DATA_DIR/$name"

    if [ -d "$target" ] && [ -f "$target/all.jsonl" ]; then
        echo "  [OK] $name (exists)"
        return 0
    fi

    if [ ! -f "$archive" ]; then
        echo "  [!!] $name (archive not found: $archive)"
        return 1
    fi

    echo "  [>>] $name (extracting...)"
    mkdir -p "$target"
    tar --no-same-owner -xJf "$archive" -C "$DATA_DIR"

    if [ -f "$target/all.jsonl" ]; then
        echo "       Done: $(du -sh "$target" | cut -f1)"
    else
        echo "       ERROR: extraction failed"
        return 1
    fi
}

echo "Checking/extracting datasets to $DATA_DIR..."
for dataset in $DATASETS; do
    extract_if_missing "$dataset"
done
echo "Done."

# =============================================================================
# Verification
# =============================================================================

echo
echo "============================================================"
echo "Setup Verification"
echo "============================================================"
echo "Python:   $(python --version)"
echo "PyTorch:  $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "CUDA:     $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "GPU:      $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")' 2>/dev/null || echo 'N/A')"
echo

# Check datasets
echo "Datasets in $DATA_DIR:"
for dir in $DATASETS; do
    if [ -d "$DATA_DIR/$dir" ] && [ -f "$DATA_DIR/$dir/all.jsonl" ]; then
        echo "  [OK] $dir"
    else
        echo "  [--] $dir"
    fi
done
echo

# =============================================================================
# Save environment config
# =============================================================================

cat > "$CODE_DIR/.env" << EOF
# TCT Environment Configuration
# Generated: $(date)
# Platform: $PLATFORM

export TCT_PLATFORM="$PLATFORM"
export TCT_WORKSPACE="$WORKSPACE"
export TCT_DATA_DIR="$DATA_DIR"
export TCT_CODE_DIR="$CODE_DIR"
export TCT_VENV_DIR="$VENV_DIR"
EOF

echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo
echo "Environment saved to: $CODE_DIR/.env"
echo
echo "To activate environment in new shells:"
echo "  source $VENV_DIR/bin/activate"
echo "  source $CODE_DIR/.env"
echo
echo "To run experiments:"
echo "  bash scripts/run.sh kubernetes tsconfig eslintrc  # All schemas"
echo "  bash scripts/run.sh kubernetes small medium       # Specific sizes"
echo "  bash scripts/run.sh kubernetes resume             # Resume from checkpoint"
echo
