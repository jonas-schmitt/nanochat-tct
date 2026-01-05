#!/bin/bash
# Remove all checkpoints except the last two and the best from each run
#
# Keeps: last 2 epoch_*.pt, best checkpoint (from config.json epoch), best.pt, config.json
# Removes: all older epoch_*.pt files (unless they are the best)
#
# Usage:
#   bash scripts/cleanup_old_checkpoints.sh          # Interactive confirmation
#   bash scripts/cleanup_old_checkpoints.sh --force  # No confirmation
#   bash scripts/cleanup_old_checkpoints.sh --dry-run # Show what would be deleted

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CHECKPOINT_DIR="$REPO_DIR/checkpoints"

FORCE=""
DRY_RUN=""

for arg in "$@"; do
    case $arg in
        --force|-f) FORCE="1" ;;
        --dry-run|-n) DRY_RUN="1" ;;
    esac
done

echo "============================================================"
echo "Cleanup: Old Checkpoints"
echo "============================================================"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "No checkpoints directory found: $CHECKPOINT_DIR"
    exit 0
fi

# Collect files to delete
TO_DELETE=()
TOTAL_SIZE=0

for run_dir in "$CHECKPOINT_DIR"/*/; do
    [ -d "$run_dir" ] || continue

    run_name=$(basename "$run_dir")

    # Find all epoch checkpoints, sorted by epoch number
    mapfile -t epochs < <(ls "$run_dir"epoch_*.pt 2>/dev/null | sort -V)

    if [ ${#epochs[@]} -le 2 ]; then
        # 0, 1, or 2 checkpoints, nothing to clean
        continue
    fi

    # Get best epoch from config.json (if available)
    best_epoch=""
    best_checkpoint=""
    if [ -f "$run_dir/config.json" ]; then
        best_epoch=$(python3 -c "import json; d=json.load(open('$run_dir/config.json')); print(d.get('epoch', ''))" 2>/dev/null || echo "")
        if [ -n "$best_epoch" ]; then
            best_checkpoint=$(printf "epoch_%03d.pt" "$best_epoch")
        fi
    fi

    # Keep the last two, mark others for deletion
    latest="${epochs[-1]}"
    second_latest="${epochs[-2]}"
    latest_name=$(basename "$latest")
    second_latest_name=$(basename "$second_latest")

    echo ">>> $run_name"
    echo "    Keeping: $latest_name (latest)"
    echo "    Keeping: $second_latest_name (2nd latest)"

    # Check if best checkpoint is different from the last two
    keep_best_separately=""
    if [ -n "$best_checkpoint" ] && [ "$best_checkpoint" != "$latest_name" ] && [ "$best_checkpoint" != "$second_latest_name" ]; then
        if [ -f "$run_dir/$best_checkpoint" ]; then
            echo "    Keeping: $best_checkpoint (best val_loss at epoch $best_epoch)"
            keep_best_separately="$best_checkpoint"
        fi
    fi

    [ -f "$run_dir/best.pt" ] && echo "    Keeping: best.pt"

    for ((i=0; i<${#epochs[@]}-2; i++)); do
        file="${epochs[i]}"
        file_name=$(basename "$file")

        # Skip if this is the best checkpoint
        if [ "$file_name" = "$keep_best_separately" ]; then
            continue
        fi

        file_size=$(stat -c%s "$file" 2>/dev/null || echo 0)
        TOTAL_SIZE=$((TOTAL_SIZE + file_size))
        TO_DELETE+=("$file")
        echo "    Delete:  $file_name"
    done
    echo
done

if [ ${#TO_DELETE[@]} -eq 0 ]; then
    echo "Nothing to clean up."
    exit 0
fi

# Convert bytes to human readable (pure bash, no bc)
if [ $TOTAL_SIZE -ge 1073741824 ]; then
    SIZE_HR="$((TOTAL_SIZE / 1073741824))G"
elif [ $TOTAL_SIZE -ge 1048576 ]; then
    SIZE_HR="$((TOTAL_SIZE / 1048576))M"
elif [ $TOTAL_SIZE -ge 1024 ]; then
    SIZE_HR="$((TOTAL_SIZE / 1024))K"
else
    SIZE_HR="${TOTAL_SIZE}B"
fi

echo "============================================================"
echo "Files to delete: ${#TO_DELETE[@]}"
echo "Space to free: $SIZE_HR"
echo "============================================================"

if [ -n "$DRY_RUN" ]; then
    echo
    echo "[DRY RUN] No files deleted."
    exit 0
fi

# Confirm unless --force
if [ -z "$FORCE" ]; then
    echo
    read -p "Delete old checkpoints? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Delete files
for file in "${TO_DELETE[@]}"; do
    rm -f "$file"
done

echo
echo "Deleted ${#TO_DELETE[@]} old checkpoints (kept last 2), freed $SIZE_HR"
