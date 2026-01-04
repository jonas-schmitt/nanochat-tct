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
#   --gpu=a40      Alex A40 (40GB)
#   --gpu=a100     Alex/TinyGPU A100 (40GB) - default
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
    echo "  --gpu=TYPE      GPU type: a40, a100 (default), a100_80, v100, rtx3080"
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

GPU_TYPE="a100"
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
# Paths - use CODE_DIR as anchor, data is sibling directory
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
# Data is always sibling of nanochat-tct, works on all platforms
DATA_DIR="$(dirname "$CODE_DIR")/data"

# Platform detection (for info only)
detect_platform() {
    if [ -d "/workspace" ] && [ -w "/workspace" ]; then
        echo "runpod"
    elif [ -n "$WORK" ] && [ -d "$WORK" ]; then
        echo "nhr"
    else
        echo "local"
    fi
}
PLATFORM=$(detect_platform)

log_verbose "Platform: $PLATFORM"
log_verbose "Code dir: $CODE_DIR"
log_verbose "Data dir: $DATA_DIR"

# =============================================================================
# Generate sbatch script
# =============================================================================

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

# Load modules (per NHR docs: https://doc.nhr.fau.de/environment/python-env/)
echo "Loading modules..."
module purge
module load cuda 2>/dev/null || echo "  cuda module not available"

# Load Python 3.12 module (try variants in order)
PYTHON_LOADED=""
for pymod in "python/3.12-conda" "python/3.12"; do
    if module load "\$pymod" 2>&1; then
        PYTHON_LOADED="\$pymod"
        echo "  Loaded: \$pymod"
        break
    else
        echo "  \$pymod not available, trying next..."
    fi
done

if [ -z "\$PYTHON_LOADED" ]; then
    echo ""
    echo "ERROR: Python 3.12 module not found!"
    echo ""
    echo "Tried: python/3.12-conda, python/3.12"
    echo ""
    echo "Available Python modules on this system:"
    module avail python 2>&1
    echo ""
    echo "Currently loaded modules:"
    module list 2>&1
    exit 1
fi

# Verify module is loaded and Python works
echo ""
echo "Module verification:"
echo "  Loaded modules:"
module list 2>&1 | grep -E "python|cuda" || echo "    (none matching python/cuda)"
echo "  Python path: \$(which python3)"
echo "  Python version: \$(python3 --version)"
echo ""

# Set paths - DATA_DIR is sibling of CODE_DIR (set at submission time)
export CODE_DIR="$CODE_DIR"
export DATA_DIR="$DATA_DIR"
export VENV_DIR="\${WORK:-\$(dirname "$CODE_DIR")}/venv-tct"

# CHECKPOINT_DIR: use HPCVAULT (500GB) or WORK (1TB) on NHR to avoid home quota (50GB)
if [ -n "\$HPCVAULT" ] && [ -d "\$HPCVAULT" ]; then
    export CHECKPOINT_DIR="\$HPCVAULT/checkpoints"
elif [ -n "\$WORK" ] && [ -d "\$WORK" ]; then
    export CHECKPOINT_DIR="\$WORK/checkpoints"
else
    export CHECKPOINT_DIR="\$CODE_DIR/checkpoints"
fi
mkdir -p "\$CHECKPOINT_DIR"

echo "CODE_DIR:       \$CODE_DIR"
echo "DATA_DIR:       \$DATA_DIR"
echo "VENV_DIR:       \$VENV_DIR"
echo "CHECKPOINT_DIR: \$CHECKPOINT_DIR"

# Verify data exists (should have been extracted by submit_all.sh)
DATASETS="tsconfig-tct-base tsconfig-utf8-base-matched eslintrc-tct-bpe-500 eslintrc-utf8-bpe-500 kubernetes-tct-bpe-1k kubernetes-utf8-bpe-1k"
echo "Verifying datasets..."
MISSING=0
for dataset in \$DATASETS; do
    if [ -d "\$DATA_DIR/\$dataset" ] && [ -f "\$DATA_DIR/\$dataset/all.jsonl" ]; then
        echo "  [OK] \$dataset"
    else
        echo "  [!!] \$dataset MISSING"
        MISSING=1
    fi
done
if [ "\$MISSING" = "1" ]; then
    echo "ERROR: Some datasets are missing from \$DATA_DIR"
    echo "Run 'bash scripts/submit_all.sh' to extract data first"
    exit 1
fi

# Stage to node-local TMPDIR on NHR for fast I/O
if [ -n "\$TMPDIR" ] && [ -d "\$TMPDIR" ]; then
    STAGED_DATA_DIR="\$TMPDIR/data"
    echo "Staging data from \$DATA_DIR to \$STAGED_DATA_DIR..."
    mkdir -p "\$STAGED_DATA_DIR"
    for dataset in \$DATASETS; do
        if [ -d "\$DATA_DIR/\$dataset" ] && [ ! -d "\$STAGED_DATA_DIR/\$dataset" ]; then
            echo "  Copying \$dataset..."
            cp -r "\$DATA_DIR/\$dataset" "\$STAGED_DATA_DIR/"
        fi
    done
    echo "Data staging complete: \$(du -sh "\$STAGED_DATA_DIR" | cut -f1)"
    export DATA_DIR="\$STAGED_DATA_DIR"
fi

# Batch size scaling for GPU
export TCT_BATCH_MULTIPLIER=$BATCH_MULTIPLIER
$( [ -n "$BATCH_BOOST" ] && echo "export TCT_BATCH_SIZE_BOOST=$BATCH_BOOST" )

cd "\$CODE_DIR"

# Activate Python environment (try venv first, fall back to conda)
CONDA_ENV_NAME="tct-py312"
CONDA_ENV_DIR="\${WORK}/software/conda/envs/\$CONDA_ENV_NAME"

if [ -f "\$VENV_DIR/bin/activate" ]; then
    echo "Activating venv: \$VENV_DIR"
    source "\$VENV_DIR/bin/activate"
elif [ -d "\$CONDA_ENV_DIR" ]; then
    echo "Activating conda env: \$CONDA_ENV_NAME"
    source "\$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "\$CONDA_ENV_NAME"
elif command -v conda &>/dev/null && conda env list 2>/dev/null | grep -q "^\${CONDA_ENV_NAME} "; then
    echo "Activating conda env: \$CONDA_ENV_NAME"
    source "\$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "\$CONDA_ENV_NAME"
else
    echo "ERROR: No Python environment found"
    echo "  Checked venv: \$VENV_DIR"
    echo "  Checked conda: \$CONDA_ENV_DIR"
    echo ""
    echo "Run setup first:"
    echo "  srun --partition=a100 --gres=gpu:1 --time=01:00:00 --pty bash -l"
    echo "  bash scripts/setup.sh"
    echo ""
    echo "Or use fresh_start.sh which runs setup automatically:"
    echo "  bash scripts/fresh_start.sh"
    exit 1
fi

# Verify environment is working
echo ""
echo "Environment verification:"
echo "  Python path: \$(which python)"
echo "  Python version: \$(python --version)"
echo "  PyTorch: \$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "  CUDA available: \$(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "  CUDA version: \$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'N/A')"
echo "  GPU: \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Verify module is still loaded (important for venv to work correctly)
echo "Loaded modules after activation:"
module list 2>&1 | head -5
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
echo "Data dir:   $DATA_DIR"
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
