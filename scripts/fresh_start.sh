#!/bin/bash
# Fresh Start: Setup + Clean + Submit All Training Jobs
#
# This is a fully automatic script that handles everything:
# 1. Runs setup if needed (venv missing)
# 2. Extracts training data if needed
# 3. Deletes all checkpoints and logs
# 4. Submits all training jobs (no resume)
#
# Usage:
#   bash scripts/fresh_start.sh              # Interactive confirmation for cleanup
#   bash scripts/fresh_start.sh --force      # No confirmation
#   bash scripts/fresh_start.sh --dry-run    # Show what would happen
#   bash scripts/fresh_start.sh --setup      # Force re-run setup even if venv exists

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

FORCE=""
DRY_RUN=""
FORCE_SETUP=""

for arg in "$@"; do
    case $arg in
        --force|-f) FORCE="--force" ;;
        --dry-run) DRY_RUN="--dry-run" ;;
        --setup) FORCE_SETUP="1" ;;
    esac
done

echo "============================================================"
echo "Fresh Start: Setup + Clean + Submit All"
echo "============================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================================"
echo

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

# CODE_DIR and DATA_DIR are always computed from script location
# This ensures consistency regardless of where repo is cloned
CODE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$(dirname "$CODE_DIR")/data"

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

echo "Platform:  $PLATFORM"
echo "Code dir:  $CODE_DIR"
echo "Data dir:  $DATA_DIR"
echo "Venv:      $VENV_DIR"
echo

# =============================================================================
# Step 0: Update code (non-local platforms)
# =============================================================================

if [ "$PLATFORM" != "local" ]; then
    echo ">>> Step 0: Updating code"
    echo
    if [ -n "$DRY_RUN" ]; then
        echo "[DRY RUN] Would run: git pull"
    else
        git pull || echo "  [WARN] git pull failed (continuing anyway)"
    fi
    echo
fi

# =============================================================================
# Step 1: Check and run setup if needed
# =============================================================================

echo ">>> Step 1: Check setup status"
echo

NEEDS_SETUP=0

# Check for venv OR conda environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "  [OK] Virtual environment exists"
elif [ -n "$CONDA_ENV_DIR" ] && [ -d "$CONDA_ENV_DIR" ]; then
    echo "  [OK] Conda environment exists"
else
    echo "  [!!] No Python environment found"
    NEEDS_SETUP=1
fi

# Check data archives exist (required)
ARCHIVE_COUNT=$(ls "$CODE_DIR/data/"*.tar.xz 2>/dev/null | wc -l)
if [ "$ARCHIVE_COUNT" -gt 0 ]; then
    echo "  [OK] Found $ARCHIVE_COUNT dataset archives"
else
    echo "  [!!] No dataset archives in $CODE_DIR/data/"
    echo "       This is fatal - ensure repository is complete"
    exit 1
fi

# Check if any data is extracted
EXTRACTED=0
for dataset in tsconfig-tct-base eslintrc-tct-bpe-500 kubernetes-tct-bpe-1k; do
    if [ -d "$DATA_DIR/$dataset" ] && [ -f "$DATA_DIR/$dataset/all.jsonl" ]; then
        EXTRACTED=1
        break
    fi
done
if [ "$EXTRACTED" = "1" ]; then
    echo "  [OK] Some datasets already extracted"
else
    echo "  [--] No datasets extracted yet (will extract later)"
fi

echo

# Run setup if needed or forced
if [ "$NEEDS_SETUP" = "1" ] || [ -n "$FORCE_SETUP" ]; then
    if [ -n "$FORCE_SETUP" ] && [ "$NEEDS_SETUP" = "0" ]; then
        echo ">>> Running setup (--setup flag, re-installing)..."
    else
        echo ">>> Running setup (Python environment not found)..."
    fi
    echo

    if [ -n "$DRY_RUN" ]; then
        echo "[DRY RUN] Would run: source scripts/setup.sh"
    else
        # Use source to inherit module/environment changes
        source "$SCRIPT_DIR/setup.sh"
    fi
    echo

    # Verify setup succeeded
    if [ -z "$DRY_RUN" ]; then
        if [ ! -f "$VENV_DIR/bin/activate" ]; then
            # Check conda fallback
            if [ -n "$CONDA_ENV_DIR" ] && [ -d "$CONDA_ENV_DIR" ]; then
                echo "  [OK] Setup complete (using conda environment)"
            else
                echo "ERROR: Setup failed - no Python environment found"
                exit 1
            fi
        else
            echo "  [OK] Setup complete"
        fi
    fi
    echo
fi

# =============================================================================
# Step 2: Cleanup
# =============================================================================

echo ">>> Step 2: Cleanup checkpoints and logs"
echo

if [ -n "$DRY_RUN" ]; then
    echo "[DRY RUN] Would run: bash scripts/cleanup.sh --force"
else
    bash "$SCRIPT_DIR/cleanup.sh" $FORCE
fi
echo

# =============================================================================
# Step 3: Submit all jobs (will extract data if needed)
# =============================================================================

echo ">>> Step 3: Submit All Jobs (no resume)"
echo

# Pass --setup if we just ran setup (or would in dry-run), so submit_all doesn't fail
SETUP_FLAG=""
if [ "$NEEDS_SETUP" = "1" ]; then
    SETUP_FLAG="--setup"
fi

# Pass through dry-run flag, always use --no-resume for fresh start
bash "$SCRIPT_DIR/submit_all.sh" --no-resume --no-pull $SETUP_FLAG $DRY_RUN
