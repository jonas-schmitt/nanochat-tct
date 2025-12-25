#!/bin/bash
# Train all TCT models: base encoding for eslintrc/tsconfig, BPE for kubernetes
# Usage: bash scripts/train_all_tct.sh

set -e

WORKSPACE="${WORKSPACE:-/workspace}"
CODE_DIR="${CODE_DIR:-$WORKSPACE/nanochat-tct}"

cd "$CODE_DIR"

echo "============================================================"
echo "TCT Training Pipeline"
echo "============================================================"
echo

# 1. tsconfig-base (small vocab, 2048 context)
echo "[1/3] Training tsconfig-base..."
echo "      Context: 2048, Vocab: 257, Epochs: 150"
echo "------------------------------------------------------------"
python -m scripts.train_unified --schema=tsconfig-base --tokenizer=tct --model_size=small
echo
echo "[1/3] tsconfig-base complete!"
echo

# 2. eslintrc-base (small vocab, 2048 context)
echo "[2/3] Training eslintrc-base..."
echo "      Context: 2048, Vocab: 257, Epochs: 150"
echo "------------------------------------------------------------"
python -m scripts.train_unified --schema=eslintrc-base --tokenizer=tct --model_size=small
echo
echo "[2/3] eslintrc-base complete!"
echo

# 3. kubernetes with BPE (large vocab, 2048 context)
echo "[3/3] Training kubernetes (BPE)..."
echo "      Context: 2048, Vocab: 20000, Epochs: 200"
echo "------------------------------------------------------------"
python -m scripts.train_unified --schema=kubernetes --tokenizer=tct --model_size=small
echo
echo "[3/3] kubernetes complete!"
echo

echo "============================================================"
echo "All TCT training complete!"
echo "============================================================"
