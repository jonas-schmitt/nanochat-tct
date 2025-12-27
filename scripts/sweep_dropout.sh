#!/bin/bash
# Dropout Sweep for Hyperparameter Tuning
#
# Tests dropout values: 0.0, 0.1, 0.2
# Uses kubernetes schema with small model, tct tokenizer, eff_batch=64
#
# Usage:
#   bash scripts/sweep_dropout.sh          # Fresh run
#   bash scripts/sweep_dropout.sh resume   # Resume from checkpoints

set -e

RESUME_MODE=""
EPOCHS=30

for arg in "$@"; do
    case $arg in
        resume) RESUME_MODE="1" ;;
        [0-9]*) EPOCHS="$arg" ;;
    esac
done

SCHEMA="kubernetes"
TOKENIZER="tct"
MODEL_SIZE="small"
EFF_BATCH=64
BATCH=16
GRAD_ACCUM=4

# Auto-detect paths
if [ -d "/workspace/nanochat-tct" ]; then
    CODE_DIR="/workspace/nanochat-tct"
    DATA_DIR="/workspace/data"
else
    CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
    DATA_DIR="$HOME/Desktop/data"
fi

LOG_DIR="${LOG_DIR:-$CODE_DIR/logs}"
mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

echo "============================================================"
echo "Dropout Sweep"
echo "============================================================"
echo "Date: $(date)"
echo "Schema: $SCHEMA"
echo "Model: $MODEL_SIZE"
echo "Tokenizer: $TOKENIZER"
echo "Epochs: $EPOCHS"
echo "Effective batch: $EFF_BATCH"
echo "Dropout values: 0.0, 0.1, 0.2"
[ -n "$RESUME_MODE" ] && echo "Mode: RESUME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "============================================================"
echo

find_latest_epoch() {
    local exp_name=$1
    local checkpoint_dir="checkpoints/${exp_name}"
    if [ -d "$checkpoint_dir" ]; then
        ls "$checkpoint_dir"/epoch_*.pt 2>/dev/null | sort -V | tail -n 1 | grep -oP 'epoch_\K\d+' | sed 's/^0*//'
    fi
}

for dropout in 0.0 0.1 0.2; do
    # Format dropout for filename (replace . with p)
    dropout_str=$(echo "$dropout" | tr '.' 'p')
    exp_name="sweep_${SCHEMA}_${TOKENIZER}_${MODEL_SIZE}_dropout${dropout_str}"
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

    echo "[START] Dropout=$dropout"
    echo "Log: $log_file"

    # Use constant LR to isolate dropout effect
    python -m scripts.train_unified \
        --schema="$SCHEMA" \
        --tokenizer="$TOKENIZER" \
        --model_size="$MODEL_SIZE" \
        --data_root="$DATA_DIR" \
        --device_batch_size="$BATCH" \
        --gradient_accumulation_override="$GRAD_ACCUM" \
        --epochs="$EPOCHS" \
        --model_tag="$exp_name" \
        --dropout="$dropout" \
        --lr_schedule="constant" \
        $RESUME_ARG \
        2>&1 | tee -a "$log_file"

    echo "[DONE] Dropout=$dropout"
    echo
done

echo "============================================================"
echo "Sweep Complete"
echo "============================================================"
echo
echo "Compare results:"
echo "  grep 'Val loss' $LOG_DIR/sweep_*_dropout*.log"
echo
