#!/bin/bash -l
# Run All Experiments
# Trains 18 models: 3 schemas × 2 tokenizers × 3 sizes
# Order: all small, then all medium, then all large
#
# Usage:
#   bash scripts/run_all.sh

set -e

# Auto-detect paths if not set
if [ -z "$CODE_DIR" ]; then
    if [ -d "/workspace/nanochat-tct" ]; then
        CODE_DIR="/workspace/nanochat-tct"
    elif [ -n "$WORK" ]; then
        CODE_DIR="$WORK/nanochat-tct"
    else
        CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
    fi
fi

cd "$CODE_DIR"

echo "============================================================"
echo "Running All Experiments"
echo "============================================================"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "============================================================"
echo

for size in small medium large; do
    echo "============================================================"
    echo "Starting $size models at $(date)"
    echo "============================================================"

    bash scripts/run_tsconfig.sh $size
    bash scripts/run_eslintrc.sh $size
    bash scripts/run_kubernetes.sh $size

    echo "============================================================"
    echo "Completed $size models at $(date)"
    echo "============================================================"
    echo
done

echo "============================================================"
echo "All experiments completed at $(date)"
echo "============================================================"
