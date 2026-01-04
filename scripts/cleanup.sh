#!/bin/bash
# Clean up all checkpoints and logs
#
# Cleans up checkpoints from all locations:
# - Local: $CODE_DIR/checkpoints
# - NHR: $HPCVAULT/checkpoints
#
# Usage:
#   bash scripts/cleanup.sh          # Interactive confirmation
#   bash scripts/cleanup.sh --force  # No confirmation

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Checkpoint directories to clean (local + NHR vault/work)
CHECKPOINT_DIRS=("$REPO_DIR/checkpoints")
[ -n "$HPCVAULT" ] && [ -d "$HPCVAULT" ] && CHECKPOINT_DIRS+=("$HPCVAULT/checkpoints")
[ -n "$WORK" ] && [ -d "$WORK" ] && CHECKPOINT_DIRS+=("$WORK/checkpoints")

LOG_DIR="$REPO_DIR/logs"

FORCE=""
for arg in "$@"; do
    case $arg in
        --force|-f) FORCE="1" ;;
    esac
done

echo "============================================================"
echo "Cleanup: Checkpoints and Logs"
echo "============================================================"

# Show what will be deleted
TOTAL_CKPT_SIZE=0
TOTAL_CKPT_COUNT=0
for ckpt_dir in "${CHECKPOINT_DIRS[@]}"; do
    if [ -d "$ckpt_dir" ]; then
        CKPT_SIZE=$(du -sh "$ckpt_dir" 2>/dev/null | cut -f1)
        CKPT_COUNT=$(find "$ckpt_dir" -name "*.pt" 2>/dev/null | wc -l)
        echo "Checkpoints: $ckpt_dir"
        echo "  Size: $CKPT_SIZE"
        echo "  Files: $CKPT_COUNT .pt files"
        TOTAL_CKPT_COUNT=$((TOTAL_CKPT_COUNT + CKPT_COUNT))
    fi
done
[ "$TOTAL_CKPT_COUNT" -eq 0 ] && echo "Checkpoints: (none)"

if [ -d "$LOG_DIR" ]; then
    LOG_SIZE=$(du -sh "$LOG_DIR" 2>/dev/null | cut -f1)
    LOG_COUNT=$(find "$LOG_DIR" -name "*.log" 2>/dev/null | wc -l)
    echo "Logs: $LOG_DIR"
    echo "  Size: $LOG_SIZE"
    echo "  Files: $LOG_COUNT .log files"
else
    echo "Logs: (none)"
fi

echo "============================================================"

# Confirm unless --force
if [ -z "$FORCE" ]; then
    echo
    read -p "Delete all checkpoints and logs? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Delete checkpoints from all locations
for ckpt_dir in "${CHECKPOINT_DIRS[@]}"; do
    if [ -d "$ckpt_dir" ]; then
        rm -rf "$ckpt_dir"
        echo "Deleted: $ckpt_dir"
    fi
done

if [ -d "$LOG_DIR" ]; then
    rm -rf "$LOG_DIR"
    echo "Deleted: $LOG_DIR"
fi

echo
echo "Cleanup complete."
