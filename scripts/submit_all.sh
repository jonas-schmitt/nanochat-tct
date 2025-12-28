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
DRY_RUN=""
RESUME="resume"

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN="--dry-run" ;;
        --no-resume) RESUME="" ;;
    esac
done

echo "============================================================"
echo "Submitting All Training Jobs"
echo "============================================================"
echo "Date: $(date)"
echo "Epochs: tsconfig=50, eslintrc=100, kubernetes=200"
echo "Checkpoint: every 5% of training"
echo "GPU: small/medium=A100, large=A100_80"
[ -n "$DRY_RUN" ] && echo "Mode: DRY RUN"
echo "============================================================"
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
