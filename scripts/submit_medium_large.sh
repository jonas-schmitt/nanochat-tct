#!/bin/bash
# Submit medium and large training jobs to Slurm with dropout=0.2
#
# This script submits only medium and large models with higher dropout (0.2)
# Results are saved to separate checkpoint folders (e.g., kubernetes_tct_medium_drop0.2)
#
# Usage:
#   bash scripts/submit_medium_large.sh              # Submit all jobs (resume)
#   bash scripts/submit_medium_large.sh --no-resume  # Start fresh
#   bash scripts/submit_medium_large.sh --dry-run    # Show what would be submitted

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$(dirname "$CODE_DIR")/data"

DRY_RUN=""
RESUME="resume"
NO_PULL=""

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN="--dry-run" ;;
        --no-resume) RESUME="" ;;
        --no-pull) NO_PULL="1" ;;
    esac
done

# Platform detection
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

echo "============================================================"
echo "Submit Medium/Large Models (dropout=0.2)"
echo "============================================================"
echo "Date: $(date)"
echo "Platform: $PLATFORM"
echo "Code dir: $CODE_DIR"
echo "Data dir: $DATA_DIR"
[ -n "$DRY_RUN" ] && echo "Mode: DRY RUN"
[ -n "$RESUME" ] && echo "Resume: enabled"
echo "============================================================"
echo

# Update code (non-local platforms)
if [ "$PLATFORM" != "local" ] && [ -z "$NO_PULL" ]; then
    echo ">>> Updating code"
    cd "$CODE_DIR"
    git pull || echo "  [WARN] git pull failed (continuing anyway)"
    echo
fi

# Check data exists
echo ">>> Checking data"
DATASETS="tsconfig-tct-base tsconfig-utf8-base-matched eslintrc-tct-bpe-500 eslintrc-utf8-bpe-500 kubernetes-tct-bpe-1k kubernetes-utf8-bpe-1k"
mkdir -p "$DATA_DIR"

for dataset in $DATASETS; do
    archive="$CODE_DIR/data/${dataset}.tar.xz"
    target="$DATA_DIR/$dataset"
    if [ -d "$target" ] && [ -f "$target/all.jsonl" ]; then
        echo "  [OK] $dataset"
    elif [ -f "$archive" ]; then
        if [ -z "$DRY_RUN" ]; then
            echo "  [>>] Extracting $dataset..."
            tar --no-same-owner -xJf "$archive" -C "$DATA_DIR"
        else
            echo "  [DRY RUN] Would extract $dataset"
        fi
    else
        echo "  [!!] MISSING: $dataset"
    fi
done
echo

# Submit jobs
SCHEMAS="kubernetes tsconfig eslintrc"
DROPOUT="0.2"

submit_job() {
    local args="$1"
    echo "[SUBMIT] $args"
    bash "$SCRIPT_DIR/submit.sh" $args $RESUME $DRY_RUN
    echo
}

echo ">>> Submitting jobs (dropout=$DROPOUT)"
echo

for schema in $SCHEMAS; do
    echo "--- $schema ---"

    # Medium: tct and utf8 separately
    submit_job "$schema medium tct --dropout=$DROPOUT"
    submit_job "$schema medium utf8 --dropout=$DROPOUT"

    # Large: tct and utf8 separately (A100_80 for headroom)
    submit_job "$schema large tct --gpu=a100_80 --dropout=$DROPOUT"
    submit_job "$schema large utf8 --gpu=a100_80 --dropout=$DROPOUT"
done

echo "============================================================"
echo "All jobs submitted"
echo "============================================================"
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f logs/slurm_*.out"
