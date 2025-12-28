#!/bin/bash
# Clean up all checkpoints and logs
#
# Usage:
#   bash scripts/cleanup.sh          # Interactive confirmation
#   bash scripts/cleanup.sh --force  # No confirmation

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CHECKPOINT_DIR="$REPO_DIR/checkpoints"
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
if [ -d "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_SIZE=$(du -sh "$CHECKPOINT_DIR" 2>/dev/null | cut -f1)
    CHECKPOINT_COUNT=$(find "$CHECKPOINT_DIR" -name "*.pt" 2>/dev/null | wc -l)
    echo "Checkpoints: $CHECKPOINT_DIR"
    echo "  Size: $CHECKPOINT_SIZE"
    echo "  Files: $CHECKPOINT_COUNT .pt files"
else
    echo "Checkpoints: (none)"
fi

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

# Delete
if [ -d "$CHECKPOINT_DIR" ]; then
    rm -rf "$CHECKPOINT_DIR"
    echo "Deleted: $CHECKPOINT_DIR"
fi

if [ -d "$LOG_DIR" ]; then
    rm -rf "$LOG_DIR"
    echo "Deleted: $LOG_DIR"
fi

echo
echo "Cleanup complete."
