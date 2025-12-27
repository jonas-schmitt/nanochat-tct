#!/bin/bash
# Batch Size Sweep for Hyperparameter Tuning
#
# Tests effective batch sizes: 16, 32, 64
# Uses kubernetes schema with small model and tct tokenizer
#
# Usage:
#   bash scripts/sweep_batch_size.sh
#   bash scripts/sweep_batch_size.sh 30   # Run for 30 epochs instead of default

set -e

EPOCHS=${1:-30}  # Default 30 epochs (enough to see trends)
SCHEMA="kubernetes"
TOKENIZER="tct"
MODEL_SIZE="small"

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
echo "Batch Size Sweep"
echo "============================================================"
echo "Date: $(date)"
echo "Schema: $SCHEMA"
echo "Model: $MODEL_SIZE"
echo "Tokenizer: $TOKENIZER"
echo "Epochs: $EPOCHS"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "============================================================"
echo

# Sweep configurations:
# Maximize micro batch size for GPU efficiency (batch=16 fits on RTX 4090 for small model)
# - eff=16: batch=16, grad_accum=1
# - eff=32: batch=16, grad_accum=2
# - eff=64: batch=16, grad_accum=4

for eff_batch in 16 32 64; do
    case $eff_batch in
        16) BATCH=16; GRAD_ACCUM=1 ;;
        32) BATCH=16; GRAD_ACCUM=2 ;;
        64) BATCH=16; GRAD_ACCUM=4 ;;
    esac

    exp_name="sweep_${SCHEMA}_${TOKENIZER}_${MODEL_SIZE}_eff${eff_batch}"
    log_file="$LOG_DIR/${exp_name}.log"

    echo "[START] Effective batch=$eff_batch (batch=$BATCH, grad_accum=$GRAD_ACCUM)"
    echo "Log: $log_file"

    python -m scripts.train_unified \
        --schema="$SCHEMA" \
        --tokenizer="$TOKENIZER" \
        --model_size="$MODEL_SIZE" \
        --data_root="$DATA_DIR" \
        --device_batch_size="$BATCH" \
        --gradient_accumulation_override="$GRAD_ACCUM" \
        --epochs="$EPOCHS" \
        --model_tag="$exp_name" \
        2>&1 | tee "$log_file"

    echo "[DONE] Effective batch=$eff_batch"
    echo
done

echo "============================================================"
echo "Sweep Complete"
echo "============================================================"
echo
echo "Compare results:"
echo "  grep 'Val loss' $LOG_DIR/sweep_*.log"
echo
