#!/bin/bash
# Slurm Job Submission Script for NHR FAU (Alex/TinyGPU)
#
# Usage:
#   bash scripts/submit.sh kubernetes small medium           # Submit job for kubernetes
#   bash scripts/submit.sh kubernetes small --gpu=a100       # Use A100 GPU
#   bash scripts/submit.sh kubernetes resume --gpu=a40       # Resume with A40
#   bash scripts/submit.sh --setup                           # Run setup only
#
# GPU options:
#   --gpu=a40      Alex A40 (40GB) - default
#   --gpu=a100     Alex/TinyGPU A100 (40GB)
#   --gpu=a100_80  Alex A100 (80GB)
#   --gpu=v100     TinyGPU V100 (32GB)
#   --gpu=rtx3080  TinyGPU RTX 3080 (10GB)
#
# Other options:
#   --time=HH:MM:SS   Wall time (default: 24:00:00)
#   --setup           Run setup before training
#   --dry-run         Print sbatch script without submitting

set -e

# =============================================================================
# Parse arguments
# =============================================================================

GPU_TYPE="a40"
TIME_LIMIT="24:00:00"
RUN_SETUP=""
DRY_RUN=""
RUN_ARGS=""

for arg in "$@"; do
    case $arg in
        --gpu=*) GPU_TYPE="${arg#*=}" ;;
        --time=*) TIME_LIMIT="${arg#*=}" ;;
        --setup) RUN_SETUP="1" ;;
        --dry-run) DRY_RUN="1" ;;
        *) RUN_ARGS="$RUN_ARGS $arg" ;;
    esac
done

RUN_ARGS="${RUN_ARGS# }"  # Trim leading space

# Validate we have something to run
if [ -z "$RUN_ARGS" ] && [ -z "$RUN_SETUP" ]; then
    echo "Usage: bash scripts/submit.sh <schema>... [size]... [tokenizer] [resume] [options]"
    echo ""
    echo "Schemas: kubernetes, tsconfig, eslintrc"
    echo "Sizes: small, medium, large, small-deep"
    echo "Tokenizers: tct, utf8"
    echo ""
    echo "Options:"
    echo "  --gpu=TYPE    GPU type: a40 (default), a100, a100_80, v100, rtx3080"
    echo "  --time=TIME   Wall time limit (default: 24:00:00)"
    echo "  --setup       Run setup before training"
    echo "  --dry-run     Print job script without submitting"
    echo ""
    echo "Examples:"
    echo "  bash scripts/submit.sh kubernetes small medium --gpu=a100"
    echo "  bash scripts/submit.sh kubernetes resume --gpu=a40 --time=12:00:00"
    echo "  bash scripts/submit.sh --setup  # Setup only"
    exit 1
fi

# =============================================================================
# GPU configuration
# =============================================================================

case $GPU_TYPE in
    a40)
        CLUSTER="alex"
        PARTITION="a40"
        GRES="gpu:a40:1"
        CONSTRAINT=""
        GPU_MEM=40
        ;;
    a100)
        CLUSTER="alex"
        PARTITION="a100"
        GRES="gpu:a100:1"
        CONSTRAINT=""
        GPU_MEM=40
        ;;
    a100_80)
        CLUSTER="alex"
        PARTITION="a100"
        GRES="gpu:a100:1"
        CONSTRAINT="-C a100_80"
        GPU_MEM=80
        ;;
    v100)
        CLUSTER="tinygpu"
        PARTITION="v100"
        GRES="gpu:v100:1"
        CONSTRAINT=""
        GPU_MEM=32
        ;;
    rtx3080)
        CLUSTER="tinygpu"
        PARTITION="rtx3080"
        GRES="gpu:rtx3080:1"
        CONSTRAINT=""
        GPU_MEM=10
        ;;
    *)
        echo "Unknown GPU type: $GPU_TYPE"
        echo "Available: a40, a100, a100_80, v100, rtx3080"
        exit 1
        ;;
esac

# Calculate batch multiplier based on GPU memory
if [ "$GPU_MEM" -ge 70 ]; then
    BATCH_MULTIPLIER=3    # A100 80GB
elif [ "$GPU_MEM" -ge 35 ]; then
    BATCH_MULTIPLIER=2    # A40, A100 40GB
elif [ "$GPU_MEM" -ge 30 ]; then
    BATCH_MULTIPLIER=1    # V100 32GB
    BATCH_BOOST=4
else
    BATCH_MULTIPLIER=1    # RTX 3080 10GB (may need smaller batch)
    BATCH_BOOST=0
fi

# =============================================================================
# Generate job name
# =============================================================================

# Extract schemas and sizes for job name
JOB_NAME="tct"
for arg in $RUN_ARGS; do
    case $arg in
        kubernetes|tsconfig|eslintrc|small|medium|large|small-deep|tct|utf8)
            JOB_NAME="${JOB_NAME}_${arg}"
            ;;
    esac
done

# =============================================================================
# Generate sbatch script
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
JOB_SCRIPT=$(mktemp /tmp/tct_job_XXXXXX.sh)

cat > "$JOB_SCRIPT" << EOF
#!/bin/bash -l
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --gres=$GRES
$( [ -n "$CONSTRAINT" ] && echo "#SBATCH $CONSTRAINT" )
#SBATCH --time=$TIME_LIMIT
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --export=NONE

# Prevent inheriting environment
unset SLURM_EXPORT_ENV

# =============================================================================
# Environment setup
# =============================================================================

echo "============================================================"
echo "TCT Training Job"
echo "============================================================"
echo "Job ID:     \$SLURM_JOB_ID"
echo "Node:       \$SLURMD_NODENAME"
echo "GPU:        $GPU_TYPE ($PARTITION)"
echo "Start:      \$(date)"
echo "============================================================"
echo

# Load modules
module purge
module load cuda 2>/dev/null || true

# Try Python 3.12 module first, fall back to conda
USE_CONDA=""
if module load python/3.12 2>/dev/null; then
    echo "Using Python 3.12 module"
elif module load python/3.12-anaconda 2>/dev/null; then
    echo "Using Python 3.12-anaconda module"
else
    echo "Python 3.12 module not available, using conda"
    module load python 2>/dev/null || true
    USE_CONDA="1"
fi

# Set paths
export WORK="\${WORK:-\$HOME}"
export CODE_DIR="\$WORK/nanochat-tct"
export DATA_DIR="\$WORK/data/tct"
export VENV_DIR="\$WORK/venv-tct"

# Batch size scaling for GPU
export TCT_BATCH_MULTIPLIER=$BATCH_MULTIPLIER
$( [ -n "$BATCH_BOOST" ] && echo "export TCT_BATCH_SIZE_BOOST=$BATCH_BOOST" )

cd "\$CODE_DIR"

# Activate environment
if [ -n "\$USE_CONDA" ]; then
    # Conda environment
    CONDA_ENV_NAME="tct-py312"
    source "\$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | grep -q "^\${CONDA_ENV_NAME} "; then
        conda activate "\$CONDA_ENV_NAME"
    else
        echo "ERROR: Conda environment \$CONDA_ENV_NAME not found"
        echo "Run setup first: bash scripts/submit.sh --setup --gpu=$GPU_TYPE"
        exit 1
    fi
elif [ -f "\$VENV_DIR/bin/activate" ]; then
    # Venv with Python 3.12 module
    source "\$VENV_DIR/bin/activate"
else
    echo "ERROR: No Python environment found"
    echo "Run setup first: bash scripts/submit.sh --setup --gpu=$GPU_TYPE"
    exit 1
fi

echo "Python:  \$(python --version)"
echo "PyTorch: \$(python -c 'import torch; print(torch.__version__)')"
echo "CUDA:    \$(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU:     \$(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo

EOF

# Add setup step if requested
if [ -n "$RUN_SETUP" ]; then
    cat >> "$JOB_SCRIPT" << 'EOF'
# =============================================================================
# Run setup
# =============================================================================

echo "Running setup..."
bash scripts/setup.sh
echo

EOF
fi

# Add training command if we have run args
if [ -n "$RUN_ARGS" ]; then
    cat >> "$JOB_SCRIPT" << EOF
# =============================================================================
# Run training
# =============================================================================

echo "Starting training: $RUN_ARGS"
echo

bash scripts/run.sh $RUN_ARGS

EOF
fi

# Add job completion
cat >> "$JOB_SCRIPT" << 'EOF'
# =============================================================================
# Job complete
# =============================================================================

echo
echo "============================================================"
echo "Job completed at $(date)"
echo "============================================================"
EOF

# =============================================================================
# Submit or display
# =============================================================================

echo "============================================================"
echo "Job Configuration"
echo "============================================================"
echo "Cluster:    $CLUSTER"
echo "Partition:  $PARTITION"
echo "GPU:        $GPU_TYPE (${GPU_MEM}GB)"
echo "Time:       $TIME_LIMIT"
echo "Batch mult: ${BATCH_MULTIPLIER}x"
echo "Job name:   $JOB_NAME"
echo "Run args:   ${RUN_ARGS:-<setup only>}"
echo "============================================================"
echo

if [ -n "$DRY_RUN" ]; then
    echo "=== Job Script (dry run) ==="
    cat "$JOB_SCRIPT"
    rm "$JOB_SCRIPT"
else
    # Ensure logs directory exists
    mkdir -p "$CODE_DIR/logs"

    # Submit to appropriate cluster
    if [ "$CLUSTER" = "alex" ]; then
        echo "Submitting to Alex cluster..."
        sbatch "$JOB_SCRIPT"
    else
        echo "Submitting to TinyGPU cluster..."
        sbatch.tinygpu "$JOB_SCRIPT"
    fi

    echo
    echo "Monitor with: squeue -u \$USER"
    echo "Cancel with:  scancel <jobid>"

    rm "$JOB_SCRIPT"
fi
