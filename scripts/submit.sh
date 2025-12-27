#!/bin/bash
# Slurm Job Submission Script for NHR FAU (Alex/TinyGPU)
#
# Usage:
#   bash scripts/submit.sh kubernetes small medium           # Submit job for kubernetes
#   bash scripts/submit.sh kubernetes small --gpu=a100       # Use A100 GPU
#   bash scripts/submit.sh kubernetes resume --gpu=a40       # Resume with A40
#   bash scripts/submit.sh --setup                           # Run setup only
#   bash scripts/submit.sh kubernetes --verbose              # Verbose output
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
#   --verbose         Verbose output for debugging

set -e

# =============================================================================
# Helper functions
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_verbose() { [ -n "$VERBOSE" ] && echo -e "${BLUE}[DEBUG]${NC} $1" || true; }

die() {
    log_error "$1"
    [ -n "$2" ] && echo "  Hint: $2"
    exit 1
}

show_help() {
    echo "Usage: bash scripts/submit.sh <schema>... [size]... [tokenizer] [resume] [options]"
    echo ""
    echo "Schemas: kubernetes, tsconfig, eslintrc"
    echo "Sizes: small, medium, large, small-deep"
    echo "Tokenizers: tct, utf8"
    echo ""
    echo "Options:"
    echo "  --gpu=TYPE      GPU type: a40 (default), a100, a100_80, v100, rtx3080"
    echo "  --time=TIME     Wall time limit (default: 24:00:00)"
    echo "  --setup         Run setup before training"
    echo "  --dry-run       Print job script without submitting"
    echo "  --verbose, -v   Verbose output for debugging"
    echo "  --help, -h      Show this help message"
    echo ""
    echo "GPU Options:"
    echo "  a40       Alex cluster, 40GB VRAM, batch multiplier 2x"
    echo "  a100      Alex/TinyGPU, 40GB VRAM, batch multiplier 2x"
    echo "  a100_80   Alex cluster, 80GB VRAM, batch multiplier 3x"
    echo "  v100      TinyGPU, 32GB VRAM, batch multiplier 1x"
    echo "  rtx3080   TinyGPU, 10GB VRAM, batch multiplier 1x"
    echo ""
    echo "Examples:"
    echo "  bash scripts/submit.sh kubernetes small medium --gpu=a100"
    echo "  bash scripts/submit.sh kubernetes resume --gpu=a40 --time=12:00:00"
    echo "  bash scripts/submit.sh --setup --gpu=a100  # Setup only"
    echo "  bash scripts/submit.sh kubernetes --dry-run --verbose"
    exit 0
}

# =============================================================================
# Parse arguments
# =============================================================================

GPU_TYPE="a40"
TIME_LIMIT="24:00:00"
RUN_SETUP=""
DRY_RUN=""
VERBOSE=""
RUN_ARGS=""

for arg in "$@"; do
    case $arg in
        --gpu=*) GPU_TYPE="${arg#*=}" ;;
        --time=*) TIME_LIMIT="${arg#*=}" ;;
        --setup) RUN_SETUP="1" ;;
        --dry-run) DRY_RUN="1" ;;
        --verbose|-v) VERBOSE="1" ;;
        --help|-h) show_help ;;
        *) RUN_ARGS="$RUN_ARGS $arg" ;;
    esac
done

RUN_ARGS="${RUN_ARGS# }"  # Trim leading space

# Validate we have something to run
if [ -z "$RUN_ARGS" ] && [ -z "$RUN_SETUP" ]; then
    die "No schema or --setup specified" "Use --help to see usage"
fi

log_verbose "GPU type: $GPU_TYPE"
log_verbose "Time limit: $TIME_LIMIT"
log_verbose "Run args: $RUN_ARGS"
log_verbose "Setup mode: ${RUN_SETUP:-no}"
log_verbose "Dry run: ${DRY_RUN:-no}"

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
        die "Unknown GPU type: $GPU_TYPE" "Available: a40, a100, a100_80, v100, rtx3080"
        ;;
esac

log_verbose "Cluster: $CLUSTER"
log_verbose "Partition: $PARTITION"
log_verbose "GRES: $GRES"
log_verbose "GPU memory: ${GPU_MEM}GB"

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
    log_warn "RTX 3080 has limited VRAM (10GB) - large models may OOM"
fi

log_verbose "Batch multiplier: ${BATCH_MULTIPLIER}x"
[ -n "$BATCH_BOOST" ] && log_verbose "Batch boost: +$BATCH_BOOST"

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
export DATA_DIR="\$CODE_DIR/../data"
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
[ -n "$VERBOSE" ] && echo "Verbose:    enabled"
echo "============================================================"
echo

log_verbose "Job script location: $JOB_SCRIPT"

if [ -n "$DRY_RUN" ]; then
    log_info "Dry run mode - showing job script without submitting"
    echo ""
    echo "=== Job Script ==="
    cat "$JOB_SCRIPT"
    echo "=== End Job Script ==="
    rm "$JOB_SCRIPT"
    log_info "To submit, remove --dry-run flag"
else
    # Ensure logs directory exists
    mkdir -p "$CODE_DIR/logs"
    log_verbose "Log directory: $CODE_DIR/logs"

    # Submit to appropriate cluster
    if [ "$CLUSTER" = "alex" ]; then
        log_info "Submitting to Alex cluster..."
        log_verbose "Running: sbatch $JOB_SCRIPT"
        if sbatch "$JOB_SCRIPT"; then
            log_success "Job submitted successfully"
        else
            die "Failed to submit job" "Check that you're on the login node and sbatch is available"
        fi
    else
        log_info "Submitting to TinyGPU cluster..."
        log_verbose "Running: sbatch.tinygpu $JOB_SCRIPT"
        if sbatch.tinygpu "$JOB_SCRIPT"; then
            log_success "Job submitted successfully"
        else
            die "Failed to submit job" "Check that you're on the login node and sbatch.tinygpu is available"
        fi
    fi

    echo
    echo "Monitor with: squeue -u \$USER"
    echo "View output:  tail -f logs/slurm_<jobid>.out"
    echo "Cancel with:  scancel <jobid>"

    rm "$JOB_SCRIPT"
fi
