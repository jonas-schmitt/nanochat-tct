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
#   bash scripts/submit_all.sh              # Submit all jobs
#   bash scripts/submit_all.sh --dry-run    # Show what would be submitted

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRY_RUN=""

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN="--dry-run" ;;
    esac
done

echo "============================================================"
echo "Submitting All Training Jobs"
echo "============================================================"
echo "Date: $(date)"
echo "Epochs: tsconfig=50, eslintrc=100, kubernetes=200"
echo "Checkpoint: every 5% of training"
echo "GPU: A100 (default)"
[ -n "$DRY_RUN" ] && echo "Mode: DRY RUN"
echo "============================================================"
echo

SCHEMAS="kubernetes tsconfig eslintrc"
SIZES_COMBINED="small"           # tct + utf8 together
SIZES_SEPARATE="medium large"    # tct and utf8 separately

submit_job() {
    local args="$1"
    echo "[SUBMIT] $args"
    bash "$SCRIPT_DIR/submit.sh" $args resume $DRY_RUN
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

    # Medium/Large: tct and utf8 separately
    for size in $SIZES_SEPARATE; do
        submit_job "$schema $size tct"
        submit_job "$schema $size utf8"
    done

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
