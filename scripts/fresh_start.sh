#!/bin/bash
# Clean up all checkpoints/logs and submit all training jobs
#
# This is a "fresh start" script that:
# 1. Deletes all checkpoints and logs
# 2. Submits all training jobs (without resume)
#
# Usage:
#   bash scripts/fresh_start.sh              # Interactive confirmation
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
echo "Fresh Start: Clean + Submit All"
echo "============================================================"
echo "Date: $(date)"
echo "============================================================"
echo

# Step 1: Cleanup
echo ">>> Step 1: Cleanup"
echo

if [ -n "$DRY_RUN" ]; then
    echo "[DRY RUN] Would run: bash scripts/cleanup.sh --force"
else
    bash "$SCRIPT_DIR/cleanup.sh" $FORCE
fi
echo

# Step 2: Submit all (without resume since we just cleaned up)
echo ">>> Step 2: Submit All Jobs (no resume)"
echo

bash "$SCRIPT_DIR/submit_all.sh" --no-resume $DRY_RUN
