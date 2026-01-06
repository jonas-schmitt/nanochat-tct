#!/bin/bash
# Submit all training jobs to Slurm
#
# This script handles everything automatically:
# 1. Checks if setup is complete (venv exists)
# 2. Extracts training data if needed
# 3. Submits all training jobs
#
# Submits ALL model sizes (small/medium/large) for ALL schemas and tokenizers
# Total: 3 schemas × 2 tokenizers × 3 sizes = 18 experiments with dropout=0.2
#
# Usage:
#   bash scripts/submit_all.sh              # Submit all jobs (resume from checkpoints)
#   bash scripts/submit_all.sh --no-resume  # Start fresh (no resume)
#   bash scripts/submit_all.sh --dry-run    # Show what would be submitted
#   bash scripts/submit_all.sh --setup      # Run setup first if needed

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
# Data is always sibling of nanochat-tct - works on all platforms
DATA_DIR="$(dirname "$CODE_DIR")/data"

DRY_RUN=""
RESUME="resume"  # Default: resume from checkpoints
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

CURRENT_COMMIT=$(cd "$CODE_DIR" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")

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

# Venv location depends on platform
case $PLATFORM in
    runpod)
        VENV_DIR="/workspace/venv"
        ;;
    nhr)
        VENV_DIR="${WORK:-$(dirname "$CODE_DIR")}/venv-tct"
        CONDA_ENV_DIR="${WORK:-$(dirname "$CODE_DIR")}/software/conda/envs/tct-py312"
        ;;
    local)
        VENV_DIR="$CODE_DIR/.venv"
        ;;
esac

echo "============================================================"
echo "Submitting All Training Jobs"
echo "============================================================"
echo "Date: $(date)"
echo "Platform: $PLATFORM"
echo "Code dir: $CODE_DIR"
echo "Data dir: $DATA_DIR"
echo "Commit: $CURRENT_COMMIT"
echo "Resume: $([ -n "$RESUME" ] && echo "YES (from checkpoints)" || echo "NO (fresh start)")"
echo "Epochs: tsconfig=100, eslintrc=125, kubernetes=150"
echo "Checkpoint: every 10% of training"
echo "Models: small, medium, large (all schemas, TCT + UTF8)"
echo "GPU: small/medium=A100, large=A100_80"
echo "Total jobs: 18 (3 schemas × 2 tokenizers × 3 sizes)"
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
        cd "$CODE_DIR"
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

# Check for venv or conda
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "  [OK] Virtual environment: $VENV_DIR"
elif [ -n "$CONDA_ENV_DIR" ] && [ -d "$CONDA_ENV_DIR" ]; then
    echo "  [OK] Conda environment: $CONDA_ENV_DIR"
else
    echo "  [!!] No Python environment found"
    echo "       Expected venv at: $VENV_DIR"
    [ -n "$CONDA_ENV_DIR" ] && echo "       Or conda at: $CONDA_ENV_DIR"
    SETUP_OK=0
fi

# Check data archives exist
ARCHIVE_COUNT=$(ls "$CODE_DIR/data/"*.tar.xz 2>/dev/null | wc -l)
if [ "$ARCHIVE_COUNT" -gt 0 ]; then
    echo "  [OK] Found $ARCHIVE_COUNT dataset archives in $CODE_DIR/data/"
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
            cd "$CODE_DIR"
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

echo ">>> Checking/extracting training data to $DATA_DIR"
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
            tar --no-same-owner -xJf "$archive" -C "$DATA_DIR"
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
DROPOUT="0.2"  # Default dropout for medium/large models

submit_job() {
    local args="$1"
    echo "[SUBMIT] $args"
    bash "$SCRIPT_DIR/submit.sh" $args $RESUME $DRY_RUN
    echo
}

# Submit jobs for each schema (all model sizes, all tokenizers)
# Each submit_job creates a SEPARATE Slurm job
# Total: 3 schemas × 2 tokenizers × 3 sizes = 18 separate jobs
for schema in $SCHEMAS; do
    echo ">>> Schema: $schema"
    echo

    # Small TCT (separate job)
    submit_job "$schema small tct --dropout=$DROPOUT"
    # Small UTF8 (separate job)
    submit_job "$schema small utf8 --dropout=$DROPOUT"

    # Medium TCT (separate job)
    submit_job "$schema medium tct --dropout=$DROPOUT"
    # Medium UTF8 (separate job)
    submit_job "$schema medium utf8 --dropout=$DROPOUT"

    # Large TCT (separate job, A100_80 for memory)
    submit_job "$schema large tct --gpu=a100_80 --dropout=$DROPOUT"
    # Large UTF8 (separate job, A100_80 for memory)
    submit_job "$schema large utf8 --gpu=a100_80 --dropout=$DROPOUT"

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
