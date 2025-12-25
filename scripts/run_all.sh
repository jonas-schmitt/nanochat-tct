#!/bin/bash
# Run All Experiments
# Trains 18 models: 3 schemas × 2 tokenizers × 3 sizes
# Order: all small, then all medium, then all large
#
# Usage:
#   bash scripts/run_all.sh

set -e

WORKSPACE="${WORKSPACE:-/workspace}"
CODE_DIR="${CODE_DIR:-$WORKSPACE/nanochat-tct}"

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
