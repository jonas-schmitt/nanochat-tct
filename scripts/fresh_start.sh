#!/bin/bash
# Clean up all checkpoints/logs and submit all training jobs
#
# This is a "fresh start" script that:
# 1. Detects platform and sets paths
# 2. Checks/extracts training data if missing
# 3. Deletes all checkpoints and logs
# 4. Submits all training jobs (without resume)
#
# Usage:
#   bash scripts/fresh_start.sh              # Interactive confirmation
#   bash scripts/fresh_start.sh --force      # No confirmation
#   bash scripts/fresh_start.sh --dry-run    # Show what would happen

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"

FORCE=""
DRY_RUN=""

for arg in "$@"; do
    case $arg in
        --force|-f) FORCE="--force" ;;
        --dry-run) DRY_RUN="--dry-run" ;;
    esac
done

echo "============================================================"
echo "Fresh Start: Clean + Submit All"
echo "============================================================"
echo "Date: $(date)"
echo "============================================================"
echo

# =============================================================================
# Detect platform and set DATA_DIR
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
echo "Platform: $PLATFORM"

case $PLATFORM in
    runpod)
        DATA_DIR="/workspace/data"
        ;;
    nhr)
        DATA_DIR="${WORK:-$HOME}/data/tct"
        ;;
    local)
        DATA_DIR="$HOME/Desktop/data"
        ;;
esac

echo "Data dir: $DATA_DIR"
echo

# =============================================================================
# Step 1: Check and extract training data if missing
# =============================================================================

echo ">>> Step 1: Check/extract training data"
echo

# Required datasets (must match schema_configs.py)
DATASETS="tsconfig-tct-base tsconfig-utf8-base-matched eslintrc-tct-bpe-500 eslintrc-utf8-bpe-500 kubernetes-tct-bpe-1k kubernetes-utf8-bpe-1k"

mkdir -p "$DATA_DIR"

extract_if_missing() {
    local name="$1"
    local archive="$CODE_DIR/data/${name}.tar.xz"
    local target="$DATA_DIR/$name"

    if [ -d "$target" ] && [ -f "$target/all.jsonl" ]; then
        echo "  [OK] $name"
        return 0
    fi

    if [ ! -f "$archive" ]; then
        echo "  [!!] $name (archive not found)"
        return 1
    fi

    if [ -n "$DRY_RUN" ]; then
        echo "  [DRY RUN] Would extract $name"
        return 0
    fi

    echo "  [>>] $name (extracting...)"
    tar -xJf "$archive" -C "$DATA_DIR"

    if [ -f "$target/all.jsonl" ]; then
        echo "       Done: $(du -sh "$target" | cut -f1)"
    else
        echo "       ERROR: extraction failed"
        return 1
    fi
}

for dataset in $DATASETS; do
    extract_if_missing "$dataset"
done
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
# Step 3: Submit all (without resume since we just cleaned up)
# =============================================================================

echo ">>> Step 3: Submit All Jobs (no resume)"
echo

bash "$SCRIPT_DIR/submit_all.sh" --no-resume $DRY_RUN
