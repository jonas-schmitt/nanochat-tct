#!/bin/bash
# Run Kubernetes Experiments
# Trains 6 models: 2 tokenizers × 3 sizes
# Estimated time: ~40-50 hours on RTX 4090
#
# Usage:
#   bash scripts/run_kubernetes.sh           # Run all
#   bash scripts/run_kubernetes.sh small     # Run only small models
#   bash scripts/run_kubernetes.sh tct       # Run only TCT models

set -e

WORKSPACE="${WORKSPACE:-/workspace}"
CODE_DIR="${CODE_DIR:-$WORKSPACE/nanochat-tct}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"
LOG_DIR="${LOG_DIR:-$CODE_DIR/logs}"

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
echo "Kubernetes Experiments (100 epochs, context=2048)"
echo "============================================================"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
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
