#!/bin/bash -l
# Run TSConfig Experiments
# Trains models with base encoding (no BPE) - most data available (117M tokens)
#
# Usage:
#   bash scripts/run_tsconfig.sh           # Run all
#   bash scripts/run_tsconfig.sh small     # Run only small models
#   bash scripts/run_tsconfig.sh tct       # Run only TCT models
#   bash scripts/run_tsconfig.sh resume    # Resume from latest checkpoint

set -e

# Auto-detect paths if not set by job.sh
if [ -z "$CODE_DIR" ]; then
    if [ -d "/workspace/nanochat-tct" ]; then
        CODE_DIR="/workspace/nanochat-tct"
        DATA_DIR="/workspace/data"
    elif [ -n "$WORK" ]; then
        CODE_DIR="$WORK/nanochat-tct"
        DATA_DIR="$WORK/data/tct"
    else
        CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
        DATA_DIR="$HOME/Desktop/data"
    fi
fi

LOG_DIR="${LOG_DIR:-$CODE_DIR/logs}"

SCHEMA="tsconfig"
TOKENIZERS="${TOKENIZERS:-tct utf8}"
SIZES="${SIZES:-small medium large}"

# Parse filter arguments
FILTER_TOKENIZER=""
FILTER_SIZE=""
RESUME_MODE=""

for arg in "$@"; do
    case $arg in
        tct|utf8) FILTER_TOKENIZER="$arg" ;;
        small|medium|large) FILTER_SIZE="$arg" ;;
        resume) RESUME_MODE="1" ;;
    esac
done

# Function to find latest checkpoint epoch
find_latest_epoch() {
    local exp_name=$1
    local checkpoint_dir="checkpoints/${exp_name}"
    if [ -d "$checkpoint_dir" ]; then
        ls "$checkpoint_dir"/epoch_*.pt 2>/dev/null | sort -V | tail -1 | grep -oP 'epoch_\K\d+' | sed 's/^0*//'
    fi
}

mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

echo "============================================================"
echo "TSConfig Experiments (50 epochs, context=2048, base encoding)"
echo "============================================================"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "Data: $DATA_DIR"
echo "============================================================"
echo

for tokenizer in $TOKENIZERS; do
    [ -n "$FILTER_TOKENIZER" ] && [ "$tokenizer" != "$FILTER_TOKENIZER" ] && continue

    for size in $SIZES; do
        [ -n "$FILTER_SIZE" ] && [ "$size" != "$FILTER_SIZE" ] && continue

        exp_name="${SCHEMA}_${tokenizer}_${size}"
        log_file="$LOG_DIR/${exp_name}.log"

        # Skip if already completed
        if [ -f "checkpoints/${exp_name}/best.pt" ]; then
            echo "[SKIP] $exp_name (already completed)"
            continue
        fi

        # Check for resume
        RESUME_ARG=""
        if [ -n "$RESUME_MODE" ]; then
            latest_epoch=$(find_latest_epoch "$exp_name")
            if [ -n "$latest_epoch" ]; then
                echo "[RESUME] $exp_name from epoch $latest_epoch"
                RESUME_ARG="--resume_from_epoch=$latest_epoch"
            fi
        fi

        echo "[START] $exp_name at $(date)"

        python -m scripts.train_unified \
            --schema="$SCHEMA" \
            --tokenizer="$tokenizer" \
            --model_size="$size" \
            --data_root="$DATA_DIR" \
            $RESUME_ARG \
            2>&1 | tee "$log_file"

        echo "[DONE] $exp_name at $(date)"
        echo
    done
done

echo "============================================================"
echo "TSConfig experiments completed at $(date)"
echo "============================================================"
