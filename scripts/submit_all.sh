#!/bin/bash
# Submit all training jobs to Slurm
#
# This script handles everything automatically:
# 1. Checks if setup is complete (venv exists)
# 2. Extracts training data if needed
# 3. Submits all training jobs
#
# Submits separate jobs for each schema/size/tokenizer combination:
# - small: tct + utf8 together (faster, they fit in one job)
# - medium/large: tct and utf8 separately (longer runs)
#
# Usage:
#   bash scripts/submit_all.sh              # Submit all jobs (with resume)
#   bash scripts/submit_all.sh --no-resume  # Submit without resume (fresh start)
#   bash scripts/submit_all.sh --dry-run    # Show what would be submitted
#   bash scripts/submit_all.sh --setup      # Run setup first if needed

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
DRY_RUN=""
RESUME="resume"
RUN_SETUP=""
NO_PULL=""

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN="--dry-run" ;;
        --no-resume) RESUME="" ;;
        --setup) RUN_SETUP="1" ;;
        --no-pull) NO_PULL="1" ;;
    esac
done

# =============================================================================
# Platform detection and paths
# =============================================================================

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

case $PLATFORM in
    runpod)
        WORKSPACE="/workspace"
        VENV_DIR="$WORKSPACE/venv"
        DATA_DIR="$WORKSPACE/data"
        ;;
    nhr)
        WORKSPACE="${WORK:-$HOME}"
        VENV_DIR="$WORKSPACE/venv-tct"
        DATA_DIR="$WORKSPACE/data"
        # Check for conda environment as fallback
        CONDA_ENV_DIR="$WORKSPACE/software/conda/envs/tct-py312"
        ;;
    local)
        WORKSPACE="$HOME"
        VENV_DIR="$CODE_DIR/.venv"
        DATA_DIR="$HOME/Desktop/data"
        ;;
esac

echo "============================================================"
echo "Submitting All Training Jobs"
echo "============================================================"
echo "Date: $(date)"
echo "Platform: $PLATFORM"
echo "Workspace: $WORKSPACE"
echo "Data dir: $DATA_DIR"
echo "Epochs: tsconfig=50, eslintrc=75, kubernetes=100"
echo "Checkpoint: every 5% of training"
echo "GPU: small/medium=A100, large=A100_80"
[ -n "$DRY_RUN" ] && echo "Mode: DRY RUN"
echo "============================================================"
echo

# =============================================================================
# Update code (non-local platforms)
# =============================================================================

if [ "$PLATFORM" != "local" ] && [ -z "$NO_PULL" ]; then
    echo ">>> Updating code"
    echo
    if [ -n "$DRY_RUN" ]; then
        echo "[DRY RUN] Would run: git pull"
    else
        git pull || echo "  [WARN] git pull failed (continuing anyway)"
    fi
    echo
fi

# =============================================================================
# Check setup status
# =============================================================================

echo ">>> Checking setup status"
echo

SETUP_OK=1

# Check for venv OR conda environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "  [OK] Virtual environment: $VENV_DIR"
elif [ -n "$CONDA_ENV_DIR" ] && [ -d "$CONDA_ENV_DIR" ]; then
    echo "  [OK] Conda environment: $CONDA_ENV_DIR"
else
    echo "  [!!] No Python environment found"
    echo "       Expected venv at: $VENV_DIR"
    [ -n "$CONDA_ENV_DIR" ] && echo "       Or conda env at: $CONDA_ENV_DIR"
    SETUP_OK=0
fi

# Check data archives exist
ARCHIVE_COUNT=$(ls "$CODE_DIR/data/"*.tar.xz 2>/dev/null | wc -l)
if [ "$ARCHIVE_COUNT" -gt 0 ]; then
    echo "  [OK] Found $ARCHIVE_COUNT dataset archives"
else
    echo "  [!!] No dataset archives in $CODE_DIR/data/"
    SETUP_OK=0
fi

echo

# Handle missing setup
if [ "$SETUP_OK" = "0" ]; then
    if [ -n "$RUN_SETUP" ]; then
        echo ">>> Running setup (--setup flag provided)..."
        echo
        if [ -n "$DRY_RUN" ]; then
            echo "[DRY RUN] Would run: bash scripts/setup.sh"
        else
            # Source setup.sh to inherit any module/environment changes
            source "$SCRIPT_DIR/setup.sh"
        fi
        echo
    else
        echo "ERROR: Setup not complete."
        echo
        echo "Options:"
        echo "  1. Run setup manually (recommended for first time):"
        if [ "$PLATFORM" = "nhr" ]; then
            echo "     srun --partition=a100 --gres=gpu:1 --time=01:00:00 --pty bash -l"
            echo "     bash scripts/setup.sh"
        else
            echo "     bash scripts/setup.sh"
        fi
        echo
        echo "  2. Run with --setup flag (automatic setup):"
        echo "     bash scripts/submit_all.sh --setup $*"
        echo
        exit 1
    fi
fi

# =============================================================================
# Extract training data if missing
# =============================================================================

echo ">>> Checking training data"
echo

DATASETS="tsconfig-tct-base tsconfig-utf8-base-matched eslintrc-tct-bpe-500 eslintrc-utf8-bpe-500 kubernetes-tct-bpe-1k kubernetes-utf8-bpe-1k"

mkdir -p "$DATA_DIR"

MISSING=0
for dataset in $DATASETS; do
    archive="$CODE_DIR/data/${dataset}.tar.xz"
    target="$DATA_DIR/$dataset"

    if [ -d "$target" ] && [ -f "$target/all.jsonl" ]; then
        echo "  [OK] $dataset"
    elif [ -f "$archive" ]; then
        if [ -n "$DRY_RUN" ]; then
            echo "  [DRY RUN] Would extract $dataset"
        else
            echo "  [>>] Extracting $dataset..."
            tar -xJf "$archive" -C "$DATA_DIR"
            echo "       Done: $(du -sh "$target" | cut -f1)"
        fi
    else
        echo "  [!!] MISSING: $dataset (no archive at $archive)"
        MISSING=1
    fi
done

if [ "$MISSING" = "1" ]; then
    echo
    echo "ERROR: Some dataset archives are missing from $CODE_DIR/data/"
    echo "       Ensure the repository is complete (git pull, git lfs pull)"
    exit 1
fi
echo

# =============================================================================
# Submit jobs
# =============================================================================

SCHEMAS="kubernetes tsconfig eslintrc"
SIZES_COMBINED="small"           # tct + utf8 together

submit_job() {
    local args="$1"
    echo "[SUBMIT] $args"
    bash "$SCRIPT_DIR/submit.sh" $args $RESUME $DRY_RUN
    echo
}

# Submit jobs for each schema
for schema in $SCHEMAS; do
    echo ">>> Schema: $schema"
    echo

    # Small: tct + utf8 together
    for size in $SIZES_COMBINED; do
        submit_job "$schema $size"
    done

    # Medium: tct and utf8 separately
    submit_job "$schema medium tct"
    submit_job "$schema medium utf8"

    # Large: tct and utf8 separately (A100_80 for headroom)
    submit_job "$schema large tct --gpu=a100_80"
    submit_job "$schema large utf8 --gpu=a100_80"

    echo
done

echo "============================================================"
echo "All jobs submitted"
echo "============================================================"
echo
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo
echo "View logs:"
echo "  tail -f logs/slurm_*.out"
echo
