#!/bin/bash
# Submit all training jobs to Slurm
#
# Submits separate jobs for each schema/size/tokenizer combination:
# - small: tct + utf8 together (faster, they fit in one job)
# - medium/large: tct and utf8 separately (longer runs)
#
# All jobs use A100 (default) and resume from checkpoints if available.
#
# Usage:
#   bash scripts/submit_all.sh              # Submit all jobs (with resume)
#   bash scripts/submit_all.sh --no-resume  # Submit without resume (fresh start)
#   bash scripts/submit_all.sh --dry-run    # Show what would be submitted

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
DRY_RUN=""
RESUME="resume"

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN="--dry-run" ;;
        --no-resume) RESUME="" ;;
    esac
done

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

case $PLATFORM in
    runpod)
        DATA_DIR="/workspace/data"
        ;;
    nhr)
        DATA_DIR="${WORK:-$HOME}/data"
        ;;
    local)
        DATA_DIR="$HOME/Desktop/data"
        ;;
esac

echo "============================================================"
echo "Submitting All Training Jobs"
echo "============================================================"
echo "Date: $(date)"
echo "Platform: $PLATFORM"
echo "Data dir: $DATA_DIR"
echo "Epochs: tsconfig=50, eslintrc=75, kubernetes=100"
echo "Checkpoint: every 5% of training"
echo "GPU: small/medium=A100, large=A100_80"
[ -n "$DRY_RUN" ] && echo "Mode: DRY RUN"
echo "============================================================"
echo

# =============================================================================
# Check and extract training data if missing
# =============================================================================

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
        echo "  [!!] MISSING: $dataset (no archive)"
        MISSING=1
    fi
done

if [ "$MISSING" = "1" ]; then
    echo
    echo "ERROR: Some datasets are missing. Run setup.sh first or check data archives."
    exit 1
fi
echo

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
echo "  ls -la logs/"
echo
