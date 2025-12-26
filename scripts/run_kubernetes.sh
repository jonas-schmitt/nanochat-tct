#!/bin/bash -l
# Run Kubernetes Experiments
# Trains models with BPE-20k compression - complex manifests (42M tokens)
#
# Usage:
#   bash scripts/run_kubernetes.sh           # Run all
#   bash scripts/run_kubernetes.sh small     # Run only small models
#   bash scripts/run_kubernetes.sh tct       # Run only TCT models

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

# Auto-detect GPU VRAM and set batch multiplier
# Base config is for 24GB (RTX 4090). Scale up for larger GPUs.
if [ -z "$TCT_BATCH_MULTIPLIER" ]; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    GPU_MEM=${GPU_MEM:-24000}
    GPU_MEM=$((GPU_MEM / 1000))  # Convert to GB

    if [ "$GPU_MEM" -ge 90 ]; then
        export TCT_BATCH_MULTIPLIER=4    # H100 NVL (94GB), H200 (141GB)
    elif [ "$GPU_MEM" -ge 70 ]; then
        export TCT_BATCH_MULTIPLIER=3    # A100 80GB, H100 80GB
    elif [ "$GPU_MEM" -ge 44 ]; then
        export TCT_BATCH_MULTIPLIER=2    # L40S/L40/A40/RTX 6000 Ada (48GB)
    elif [ "$GPU_MEM" -ge 30 ]; then
        export TCT_BATCH_MULTIPLIER=1    # RTX 5090 (32GB) - 1.33x headroom
        export TCT_BATCH_SIZE_BOOST=4    # Add 4 to batch size for 32GB
    else
        export TCT_BATCH_MULTIPLIER=1    # RTX 4090/3090 (24GB)
    fi
    echo "GPU VRAM: ${GPU_MEM}GB -> Batch multiplier: ${TCT_BATCH_MULTIPLIER}x"
fi

SCHEMA="kubernetes"
TOKENIZERS="${TOKENIZERS:-tct utf8}"
SIZES="${SIZES:-small medium large}"

# Parse filter arguments
FILTER_TOKENIZER=""
FILTER_SIZE=""

for arg in "$@"; do
    case $arg in
        tct|utf8) FILTER_TOKENIZER="$arg" ;;
        small|medium|large) FILTER_SIZE="$arg" ;;
    esac
done

mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

echo "============================================================"
echo "Kubernetes Experiments (150 epochs, context=2048, BPE-20k)"
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

        echo "[START] $exp_name at $(date)"

        python -m scripts.train_unified \
            --schema="$SCHEMA" \
            --tokenizer="$tokenizer" \
            --model_size="$size" \
            --data_root="$DATA_DIR" \
            2>&1 | tee "$log_file"

        echo "[DONE] $exp_name at $(date)"
        echo
    done
done

echo "============================================================"
echo "Kubernetes experiments completed at $(date)"
echo "============================================================"
