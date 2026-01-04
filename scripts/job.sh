#!/bin/bash -l
#SBATCH --job-name=tct
#SBATCH --output=logs/slurm-%j.log
#SBATCH --error=logs/slurm-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#
# Unified Job Script for TCT Experiments
# Works on: RunPod (direct), NHR/HPC (via sbatch)
#
# Usage:
#   RunPod:  bash scripts/job.sh [args]
#   NHR:     sbatch scripts/job.sh [args]
#
# Arguments (passed to run scripts):
#   small medium large   - Model size filter
#   tct utf8             - Tokenizer filter
#   tsconfig eslintrc kubernetes - Schema filter
#
# Examples:
#   bash scripts/job.sh small              # All small models
#   sbatch scripts/job.sh tsconfig small   # tsconfig small only
#   bash scripts/job.sh                    # All 18 models

set -e
unset SLURM_EXPORT_ENV 2>/dev/null || true

# =============================================================================
# Environment Detection & Setup
# =============================================================================

detect_platform() {
    if [ -n "$SLURM_JOB_ID" ]; then
        echo "slurm"
    elif [ -d "/workspace" ] && [ -w "/workspace" ]; then
        echo "runpod"
    elif [ -n "$WORK" ]; then
        echo "nhr"
    else
        echo "local"
    fi
}

PLATFORM=$(detect_platform)

# Set paths based on platform (consistent with setup.sh/submit.sh)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$(dirname "$CODE_DIR")/data"

case $PLATFORM in
    slurm|nhr)
        VENV_DIR="${WORK:-$(dirname "$CODE_DIR")}/venv-tct"
        CONDA_ENV_DIR="${WORK:-$(dirname "$CODE_DIR")}/software/conda/envs/tct-py312"

        # Load modules (per NHR docs)
        echo "Loading modules..."
        module purge 2>/dev/null || true
        module load cuda 2>/dev/null || echo "  cuda module not available"

        # Load Python 3.12 module
        PYTHON_LOADED=""
        for pymod in "python/3.12-conda" "python/3.12"; do
            if module load "$pymod" 2>&1; then
                PYTHON_LOADED="$pymod"
                echo "  Loaded: $pymod"
                break
            fi
        done

        if [ -z "$PYTHON_LOADED" ]; then
            echo "ERROR: Python 3.12 module not found"
            module avail python 2>&1
            exit 1
        fi
        ;;
    runpod)
        VENV_DIR="/workspace/venv"
        ;;
    local)
        VENV_DIR="$CODE_DIR/.venv"
        ;;
esac

# Activate Python environment (try venv first, fall back to conda)
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Activating venv: $VENV_DIR"
    source "$VENV_DIR/bin/activate"
elif [ -n "$CONDA_ENV_DIR" ] && [ -d "$CONDA_ENV_DIR" ]; then
    echo "Activating conda env: $CONDA_ENV_DIR"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate tct-py312
else
    echo "ERROR: No Python environment found"
    echo "  Checked venv: $VENV_DIR"
    [ -n "$CONDA_ENV_DIR" ] && echo "  Checked conda: $CONDA_ENV_DIR"
    exit 1
fi

cd "$CODE_DIR"
mkdir -p logs

# =============================================================================
# GPU Detection & Batch Size Optimization
# =============================================================================

get_gpu_memory_gb() {
    python3 -c "
import torch
if torch.cuda.is_available():
    print(int(torch.cuda.get_device_properties(0).total_memory / 1e9))
else:
    print(0)
" 2>/dev/null || echo "0"
}

GPU_MEM=$(get_gpu_memory_gb)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")

echo "============================================================"
echo "TCT Job Started"
echo "============================================================"
echo "Date:     $(date)"
echo "Host:     $(hostname)"
echo "Platform: $PLATFORM"
echo "GPU:      $GPU_NAME ($GPU_MEM GB)"
echo "Args:     $@"
echo "============================================================"
echo

# Set batch size multiplier based on GPU memory
# Base: RTX 4090/3090 (24GB) = 1x
# A100 40GB = ~1.5x, A100 80GB = ~3x, H100 = ~4x
if [ "$GPU_MEM" -ge 90 ]; then
    export TCT_BATCH_MULTIPLIER=4
elif [ "$GPU_MEM" -ge 70 ]; then
    export TCT_BATCH_MULTIPLIER=3
elif [ "$GPU_MEM" -ge 35 ]; then
    export TCT_BATCH_MULTIPLIER=2
else
    export TCT_BATCH_MULTIPLIER=1
fi

echo "Batch multiplier: ${TCT_BATCH_MULTIPLIER}x (based on ${GPU_MEM}GB VRAM)"
echo

# =============================================================================
# Parse Arguments & Run
# =============================================================================

SCHEMAS=""
SIZES=""
TOKENIZERS=""

for arg in "$@"; do
    case $arg in
        tsconfig|eslintrc|kubernetes) SCHEMAS="$SCHEMAS $arg" ;;
        small|medium|large) SIZES="$SIZES $arg" ;;
        tct|utf8) TOKENIZERS="$TOKENIZERS $arg" ;;
    esac
done

# Default to all if not specified
SCHEMAS="${SCHEMAS:-tsconfig eslintrc kubernetes}"
SIZES="${SIZES:-small medium large}"

export DATA_DIR
export CODE_DIR

# Run experiments
for size in $SIZES; do
    for schema in $SCHEMAS; do
        echo "[$(date +%H:%M:%S)] Running $schema $size..."

        case $schema in
            tsconfig)  bash scripts/run_tsconfig.sh $size $TOKENIZERS ;;
            eslintrc)  bash scripts/run_eslintrc.sh $size $TOKENIZERS ;;
            kubernetes) bash scripts/run_kubernetes.sh $size $TOKENIZERS ;;
        esac
    done
done

echo
echo "============================================================"
echo "TCT Job Completed at $(date)"
echo "============================================================"
