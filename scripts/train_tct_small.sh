#!/bin/bash
# Train all TCT models: base encoding for eslintrc/tsconfig, BPE for kubernetes
# Usage: bash scripts/train_tct_small.sh

set -e

WORKSPACE="${WORKSPACE:-/workspace}"
CODE_DIR="${CODE_DIR:-$WORKSPACE/nanochat-tct}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"
LOG_DIR="${LOG_DIR:-$CODE_DIR/logs}"

mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

echo "============================================================"
echo "TCT Training Pipeline"
echo "============================================================"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "============================================================"
echo

# 1. tsconfig-base (small vocab, 2048 context)
exp_name="tsconfig-base_tct_small"
log_file="$LOG_DIR/${exp_name}.log"

if [ -f "checkpoints/${exp_name}/best.pt" ]; then
    echo "[SKIP] $exp_name (already completed)"
else
    echo "[1/3] Training tsconfig-base..."
    echo "      Context: 2048, Vocab: 257, Epochs: 150"
    echo "      Log: $log_file"
    echo "------------------------------------------------------------"
    python -m scripts.train_unified \
        --schema=tsconfig-base \
        --tokenizer=tct \
        --model_size=small \
        --data_root="$DATA_DIR" \
        2>&1 | tee "$log_file"
    echo "[1/3] tsconfig-base complete!"
fi
echo

# 2. eslintrc-base (small vocab, 2048 context)
exp_name="eslintrc-base_tct_small"
log_file="$LOG_DIR/${exp_name}.log"

if [ -f "checkpoints/${exp_name}/best.pt" ]; then
    echo "[SKIP] $exp_name (already completed)"
else
    echo "[2/3] Training eslintrc-base..."
    echo "      Context: 2048, Vocab: 257, Epochs: 150"
    echo "      Log: $log_file"
    echo "------------------------------------------------------------"
    python -m scripts.train_unified \
        --schema=eslintrc-base \
        --tokenizer=tct \
        --model_size=small \
        --data_root="$DATA_DIR" \
        2>&1 | tee "$log_file"
    echo "[2/3] eslintrc-base complete!"
fi
echo

# 3. kubernetes with BPE (large vocab, 2048 context)
exp_name="kubernetes_tct_small"
log_file="$LOG_DIR/${exp_name}.log"

if [ -f "checkpoints/${exp_name}/best.pt" ]; then
    echo "[SKIP] $exp_name (already completed)"
else
    echo "[3/3] Training kubernetes (BPE)..."
    echo "      Context: 2048, Vocab: 20000, Epochs: 200"
    echo "      Log: $log_file"
    echo "------------------------------------------------------------"
    python -m scripts.train_unified \
        --schema=kubernetes \
        --tokenizer=tct \
        --model_size=small \
        --data_root="$DATA_DIR" \
        2>&1 | tee "$log_file"
    echo "[3/3] kubernetes complete!"
fi
echo

echo "============================================================"
echo "All TCT training complete at $(date)"
echo "Logs: $LOG_DIR"
echo "============================================================"
