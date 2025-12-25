#!/bin/bash
# Run TCT Small Base Experiments
# Trains 2 models: base encoding (no BPE) for tsconfig and eslintrc
#
# Usage:
#   bash scripts/train_tct_small.sh

set -e

WORKSPACE="${WORKSPACE:-/workspace}"
CODE_DIR="${CODE_DIR:-$WORKSPACE/nanochat-tct}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"
LOG_DIR="${LOG_DIR:-$CODE_DIR/logs}"

mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

echo "============================================================"
echo "TCT Small Base Experiments (no BPE)"
echo "============================================================"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "============================================================"
echo

# 1. tsconfig-base (small vocab, 2048 context, 150 epochs)
exp_name="tsconfig-base_tct_small"
log_file="$LOG_DIR/${exp_name}.log"

if [ -f "checkpoints/${exp_name}/best.pt" ]; then
    echo "[SKIP] $exp_name (already completed)"
else
    echo "[START] $exp_name at $(date)"

    python -m scripts.train_unified \
        --schema=tsconfig-base \
        --tokenizer=tct \
        --model_size=small \
        --data_root="$DATA_DIR" \
        2>&1 | tee "$log_file"

    echo "[DONE] $exp_name at $(date)"
fi
echo

# 2. eslintrc-base (small vocab, 2048 context, 150 epochs)
exp_name="eslintrc-base_tct_small"
log_file="$LOG_DIR/${exp_name}.log"

if [ -f "checkpoints/${exp_name}/best.pt" ]; then
    echo "[SKIP] $exp_name (already completed)"
else
    echo "[START] $exp_name at $(date)"

    python -m scripts.train_unified \
        --schema=eslintrc-base \
        --tokenizer=tct \
        --model_size=small \
        --data_root="$DATA_DIR" \
        2>&1 | tee "$log_file"

    echo "[DONE] $exp_name at $(date)"
fi
echo

echo "============================================================"
echo "TCT Small experiments completed at $(date)"
echo "============================================================"
