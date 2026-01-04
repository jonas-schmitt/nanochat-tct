#!/bin/bash
# Submit all TCT experiments to SLURM
#
# This submits 12 jobs: 3 schemas × 2 sizes × 2 tokenizers (medium/large only)
# Uses dropout=0.2 for all experiments
#
# Usage:
#   ./submit_all.sh           # Submit all experiments
#   ./submit_all.sh --dry-run # Show what would be submitted

set -e

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "DRY RUN - showing what would be submitted"
    echo ""
fi

SCRIPT_DIR="$(dirname "$0")"
cd "${SCRIPT_DIR}/../.."

# Schemas and their expected training times
SCHEMAS=("tsconfig" "eslintrc" "kubernetes")
TOKENIZERS=("tct" "utf8")
MODEL_SIZES=("medium" "large")  # Only medium and large
DROPOUT="0.2"

# Time limits by schema × model size
declare -A TIME_LIMITS
TIME_LIMITS["tsconfig-medium"]="8:00:00"
TIME_LIMITS["tsconfig-large"]="24:00:00"
TIME_LIMITS["eslintrc-medium"]="12:00:00"
TIME_LIMITS["eslintrc-large"]="36:00:00"
TIME_LIMITS["kubernetes-medium"]="48:00:00"
TIME_LIMITS["kubernetes-large"]="72:00:00"

echo "=============================================="
echo "TCT Experiment Batch Submission"
echo "=============================================="
echo "Models: medium, large (dropout=$DROPOUT)"
echo ""

JOB_COUNT=0
for SCHEMA in "${SCHEMAS[@]}"; do
    for TOKENIZER in "${TOKENIZERS[@]}"; do
        for MODEL_SIZE in "${MODEL_SIZES[@]}"; do
            EXPERIMENT="${SCHEMA}_${TOKENIZER}_${MODEL_SIZE}"
            TIME_KEY="${SCHEMA}-${MODEL_SIZE}"
            TIME_LIMIT="${TIME_LIMITS[$TIME_KEY]}"

            echo "Submitting: ${EXPERIMENT} (time: ${TIME_LIMIT}, dropout: ${DROPOUT})"

            if [[ "$DRY_RUN" == "false" ]]; then
                sbatch \
                    --job-name="tct-${EXPERIMENT}" \
                    --time="${TIME_LIMIT}" \
                    scripts/slurm/train_nhr.sh \
                    "${SCHEMA}" "${TOKENIZER}" "${MODEL_SIZE}" "--dropout=${DROPOUT}"
            fi

            JOB_COUNT=$((JOB_COUNT + 1))
        done
    done
done

echo ""
echo "=============================================="
echo "Submitted ${JOB_COUNT} jobs"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all: scancel -u \$USER -n tct-*"
