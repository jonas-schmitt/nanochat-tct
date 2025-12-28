#!/bin/bash
# Learning Rate Sweep for eff_batch=64
#
# Tests 11 learning rates: 5e-5 to 2e-3
# Uses kubernetes schema with small model, tct tokenizer, eff_batch=64, dropout=0.0
# Estimated runtime: ~11 hours overnight
#
# Usage:
#   bash scripts/sweep_lr_eff64.sh          # Fresh run
#   bash scripts/sweep_lr_eff64.sh resume   # Resume from checkpoints
#   bash scripts/sweep_lr_eff64.sh 50       # Run for 50 epochs

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
DROPOUT=0.0

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
echo "Learning Rate Sweep (eff_batch=64)"
echo "============================================================"
echo "Date: $(date)"
echo "Schema: $SCHEMA"
echo "Model: $MODEL_SIZE"
echo "Tokenizer: $TOKENIZER"
echo "Epochs: $EPOCHS"
echo "Effective batch: $EFF_BATCH (batch=$BATCH, grad_accum=$GRAD_ACCUM)"
echo "Dropout: $DROPOUT"
echo "Learning rates: 4e-4, 5e-4, 6e-4, 2e-4, 1e-4, 8e-4, 1e-3, 1.5e-4, 5e-5, 1.5e-3, 2e-3"
echo "Estimated time: ~11 hours (11 LRs × ~1h each)"
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

# Test LRs across full range (skip 3e-4 - that's the baseline)
# Order: most promising first (slightly higher than baseline), then lower, then very high
for lr in 4e-4 5e-4 6e-4 2e-4 1e-4 8e-4 1e-3 1.5e-4 5e-5 1.5e-3 2e-3; do
    # Format LR for filename (1e-4 -> 1e4)
    lr_str=$(echo "$lr" | tr -d '-')
    exp_name="sweep_${SCHEMA}_${TOKENIZER}_${MODEL_SIZE}_eff64_lr${lr_str}"
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

    echo "[START] LR=$lr"
    echo "Log: $log_file"

    # Use constant LR schedule to isolate learning rate effect
    python -m scripts.train_unified \
        --schema="$SCHEMA" \
        --tokenizer="$TOKENIZER" \
        --model_size="$MODEL_SIZE" \
        --data_root="$DATA_DIR" \
        --device_batch_size="$BATCH" \
        --gradient_accumulation_override="$GRAD_ACCUM" \
        --epochs="$EPOCHS" \
        --model_tag="$exp_name" \
        --dropout="$DROPOUT" \
        --learning_rate_override="$lr" \
        --lr_schedule="constant" \
        $RESUME_ARG \
        2>&1 | tee -a "$log_file"

    echo "[DONE] LR=$lr"
    echo
done

echo "============================================================"
echo "Sweep Complete"
echo "============================================================"
echo
echo "Compare results:"
echo "  grep 'Min val loss' $LOG_DIR/sweep_*_eff64_lr*.log"
echo
