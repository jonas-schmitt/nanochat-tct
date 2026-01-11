#!/bin/bash -l
# Fresh Start: Setup + Clean + Submit All Training Jobs
#
# IMPORTANT: Uses bash -l (login shell) to ensure module system is available on HPC
#
# This is a fully automatic script that handles everything:
# 1. Always runs setup (ensures clean environment)
# 2. Extracts training data if needed
# 3. Deletes all checkpoints and logs
# 4. Submits all training jobs (no resume)
#
# Usage:
#   bash scripts/fresh_start.sh              # Interactive confirmation for cleanup
#   bash scripts/fresh_start.sh --force      # No confirmation
#   bash scripts/fresh_start.sh --dry-run    # Show what would happen

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

FORCE=""
DRY_RUN=""

for arg in "$@"; do
    case $arg in
        --force|-f) FORCE="--force" ;;
        --dry-run) DRY_RUN="--dry-run" ;;
    esac
done

echo "============================================================"
echo "Fresh Start: Setup + Clean + Submit All"
echo "============================================================"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "Shell: $SHELL"
echo "============================================================"
echo

# =============================================================================
# Debug: Environment diagnostics
# =============================================================================

echo ">>> Environment Diagnostics"
echo "  PWD: $(pwd)"
echo "  SCRIPT_DIR: $SCRIPT_DIR"
echo "  HOME: $HOME"
echo "  WORK: ${WORK:-<not set>}"
echo "  HPCVAULT: ${HPCVAULT:-<not set>}"
echo "  PATH (first 200 chars): ${PATH:0:200}..."
echo

# Check critical commands
echo "  Command availability:"
echo "    bash: $(command -v bash || echo 'NOT FOUND')"
echo "    python3: $(command -v python3 || echo 'NOT FOUND')"
echo "    module: $(command -v module || echo 'NOT FOUND (will try to init)')"
echo "    git: $(command -v git || echo 'NOT FOUND')"
echo "    tar: $(command -v tar || echo 'NOT FOUND')"
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
echo ">>> Platform Detection"
echo "  Detected platform: $PLATFORM"
echo "  Detection logic:"
echo "    /workspace exists and writable: $([ -d '/workspace' ] && [ -w '/workspace' ] && echo 'YES' || echo 'NO')"
echo "    \$WORK set and exists: $([ -n "$WORK" ] && [ -d "$WORK" ] && echo 'YES ('$WORK')' || echo 'NO')"
echo

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

echo ">>> Path Configuration"
echo "  Platform:  $PLATFORM"
echo "  Code dir:  $CODE_DIR"
echo "  Data dir:  $DATA_DIR"
echo "  Venv:      $VENV_DIR"
[ -n "$CONDA_ENV_DIR" ] && echo "  Conda:     $CONDA_ENV_DIR"
echo

# Verify paths exist
echo ">>> Path Verification"
echo "  Code dir exists: $([ -d "$CODE_DIR" ] && echo 'YES' || echo 'NO')"
echo "  Data dir exists: $([ -d "$DATA_DIR" ] && echo 'YES' || echo 'NO')"
echo "  Venv exists: $([ -f "$VENV_DIR/bin/activate" ] && echo 'YES' || echo 'NO')"
[ -n "$CONDA_ENV_DIR" ] && echo "  Conda exists: $([ -d "$CONDA_ENV_DIR" ] && echo 'YES' || echo 'NO')"
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
# Step 0.5: Validate checkpoint directory (fail early if not writable)
# =============================================================================

echo ">>> Validating checkpoint directory"

# Same logic as run.sh: HPCVAULT -> WORK -> CODE_DIR/checkpoints
if [ -z "$CHECKPOINT_DIR" ]; then
    if [ -n "$HPCVAULT" ] && [ -d "$HPCVAULT" ]; then
        CHECKPOINT_DIR="$HPCVAULT/checkpoints"
    elif [ -n "$WORK" ] && [ -d "$WORK" ]; then
        CHECKPOINT_DIR="$WORK/checkpoints"
    else
        CHECKPOINT_DIR="$CODE_DIR/checkpoints"
    fi
fi

mkdir -p "$CHECKPOINT_DIR" || { echo "ERROR: Failed to create checkpoint directory: $CHECKPOINT_DIR"; exit 1; }
if [ ! -w "$CHECKPOINT_DIR" ]; then
    echo "ERROR: Checkpoint directory not writable: $CHECKPOINT_DIR"
    exit 1
fi
echo "  [OK] Checkpoint dir: $CHECKPOINT_DIR"
export CHECKPOINT_DIR
echo

# =============================================================================
# Step 1: Run setup (always, ensures clean environment)
# =============================================================================

echo ">>> Step 1: Running setup..."

# Check data archives exist (required)
ARCHIVE_COUNT=$(ls "$CODE_DIR/data/"*.tar.xz 2>/dev/null | wc -l)
if [ "$ARCHIVE_COUNT" -eq 0 ]; then
    echo "  [!!] No dataset archives in $CODE_DIR/data/"
    echo "       This is fatal - ensure repository is complete"
    exit 1
fi
echo "  [OK] Found $ARCHIVE_COUNT dataset archives"
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
    if [ -f "$VENV_DIR/bin/activate" ]; then
        echo "  [OK] Setup complete (venv at $VENV_DIR)"
    elif [ -n "$CONDA_ENV_DIR" ] && [ -d "$CONDA_ENV_DIR" ]; then
        echo "  [OK] Setup complete (conda at $CONDA_ENV_DIR)"
    else
        echo "ERROR: Setup failed - no Python environment found"
        echo "  Checked venv: $VENV_DIR"
        [ -n "$CONDA_ENV_DIR" ] && echo "  Checked conda: $CONDA_ENV_DIR"
        exit 1
    fi
fi
echo

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

echo ">>> Step 3: Submit All Jobs (fresh start, no resume)"
echo

# Pass through dry-run flag, use --no-resume for fresh start
bash "$SCRIPT_DIR/submit_all.sh" --no-resume --no-pull $DRY_RUN
