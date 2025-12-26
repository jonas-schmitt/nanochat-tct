#!/bin/bash -l
# Unified Run Script for TCT Experiments
# Works with all schemas: kubernetes, tsconfig, eslintrc
#
# Usage:
#   bash scripts/run.sh kubernetes              # Run all kubernetes experiments
#   bash scripts/run.sh kubernetes small        # Run only small models
#   bash scripts/run.sh kubernetes tct          # Run only TCT tokenizer
#   bash scripts/run.sh kubernetes resume       # Resume from latest checkpoint
#   bash scripts/run.sh kubernetes resume small tct  # Combine filters

set -e

# =============================================================================
# Parse arguments
# =============================================================================

SCHEMA=""
FILTER_TOKENIZER=""
FILTER_SIZE=""
RESUME_MODE=""

for arg in "$@"; do
    case $arg in
        kubernetes|tsconfig|eslintrc) SCHEMA="$arg" ;;
        tct|utf8) FILTER_TOKENIZER="$arg" ;;
        small|small-deep|medium|large) FILTER_SIZE="$arg" ;;
        resume) RESUME_MODE="1" ;;
    esac
done

if [ -z "$SCHEMA" ]; then
    echo "Usage: bash scripts/run.sh <schema> [size] [tokenizer] [resume]"
    echo ""
    echo "Schemas: kubernetes, tsconfig, eslintrc"
    echo "Sizes: small, small-deep, medium, large"
    echo "Tokenizers: tct, utf8"
    echo ""
    echo "Examples:"
    echo "  bash scripts/run.sh kubernetes"
    echo "  bash scripts/run.sh kubernetes small tct"
    echo "  bash scripts/run.sh kubernetes resume small"
    exit 1
fi

# =============================================================================
# Auto-detect paths
# =============================================================================

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

# =============================================================================
# Auto-detect GPU VRAM and set batch multiplier
# =============================================================================

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

# =============================================================================
# Schema-specific settings
# =============================================================================

case $SCHEMA in
    kubernetes)
        TOKENIZERS="${TOKENIZERS:-tct utf8}"
        SIZES="${SIZES:-small small-deep medium large}"
        EPOCHS=150
        ;;
    tsconfig)
        TOKENIZERS="${TOKENIZERS:-tct utf8}"
        SIZES="${SIZES:-small medium large}"
        EPOCHS=50
        ;;
    eslintrc)
        TOKENIZERS="${TOKENIZERS:-tct utf8}"
        SIZES="${SIZES:-small medium large}"
        EPOCHS=100
        ;;
esac

# =============================================================================
# Helper functions
# =============================================================================

find_latest_epoch() {
    local exp_name=$1
    local checkpoint_dir="checkpoints/${exp_name}"
    if [ -d "$checkpoint_dir" ]; then
        ls "$checkpoint_dir"/epoch_*.pt 2>/dev/null | sort -V | tail -1 | grep -oP 'epoch_\K\d+' | sed 's/^0*//'
    fi
}

# =============================================================================
# Run experiments
# =============================================================================

mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

echo "============================================================"
echo "$SCHEMA Experiments ($EPOCHS epochs, context=2048)"
echo "============================================================"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "Data: $DATA_DIR"
echo "Sizes: $SIZES"
echo "Tokenizers: $TOKENIZERS"
[ -n "$RESUME_MODE" ] && echo "Mode: RESUME"
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
                # Add separator to log file
                echo "" >> "$log_file"
                echo "============================================================" >> "$log_file"
                echo "RESUMING from epoch $latest_epoch at $(date)" >> "$log_file"
                echo "============================================================" >> "$log_file"
            fi
        fi

        echo "[START] $exp_name at $(date)"

        python -m scripts.train_unified \
            --schema="$SCHEMA" \
            --tokenizer="$tokenizer" \
            --model_size="$size" \
            --data_root="$DATA_DIR" \
            $RESUME_ARG \
            2>&1 | tee -a "$log_file"

        echo "[DONE] $exp_name at $(date)"
        echo
    done
done

echo "============================================================"
echo "$SCHEMA experiments completed at $(date)"
echo "============================================================"
