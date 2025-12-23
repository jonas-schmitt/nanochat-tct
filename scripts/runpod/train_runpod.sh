#!/bin/bash
# TCT Experiment Training Script for RunPod
#
# Usage:
#   ./train_runpod.sh kubernetes tct small
#
# This script:
# 1. Downloads training data from cloud storage (if not present)
# 2. Runs training
# 3. Uploads checkpoints to cloud storage

set -e

# Parse arguments
SCHEMA=${1:-kubernetes}
TOKENIZER=${2:-tct}
MODEL_SIZE=${3:-small}

EXPERIMENT_TAG="${SCHEMA}_${TOKENIZER}_${MODEL_SIZE}"

echo "=============================================="
echo "TCT Experiment: ${EXPERIMENT_TAG}"
echo "=============================================="
echo "Schema: ${SCHEMA}"
echo "Tokenizer: ${TOKENIZER}"
echo "Model size: ${MODEL_SIZE}"
echo "=============================================="

# Setup paths
WORKSPACE="/workspace"
PROJECT_DIR="${WORKSPACE}/nanochat-tct"
DATA_DIR="${WORKSPACE}/data"
CHECKPOINT_DIR="${WORKSPACE}/checkpoints"

# Data directories by schema and tokenizer
declare -A DATA_DIRS
DATA_DIRS["tsconfig-tct"]="tsconfig-tct-bpe-10k"
DATA_DIRS["tsconfig-utf8"]="tsconfig-utf8-bpe-10k"
DATA_DIRS["eslintrc-tct"]="eslintrc-tct-bpe-10k"
DATA_DIRS["eslintrc-utf8"]="eslintrc-utf8-bpe-10k"
DATA_DIRS["kubernetes-tct"]="kubernetes-tct-bpe"
DATA_DIRS["kubernetes-utf8"]="kubernetes-utf8-bpe"

DATA_SUBDIR="${DATA_DIRS[${SCHEMA}-${TOKENIZER}]}"

# Check if project exists
if [[ ! -d "${PROJECT_DIR}" ]]; then
    echo "Cloning project..."
    cd "${WORKSPACE}"
    git clone https://github.com/YOUR_USERNAME/nanochat-tct.git
fi

cd "${PROJECT_DIR}"
git pull

# Install dependencies
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -q -e .

# Check if data exists
DATA_PATH="${DATA_DIR}/${DATA_SUBDIR}"
if [[ ! -d "${DATA_PATH}" ]]; then
    echo "=============================================="
    echo "DATA NOT FOUND: ${DATA_PATH}"
    echo "=============================================="
    echo ""
    echo "Please upload training data to ${DATA_PATH}"
    echo "Expected structure:"
    echo "  ${DATA_PATH}/"
    echo "    train.jsonl"
    echo "    validate.jsonl"
    echo "    metadata.json"
    echo "    stats.json"
    echo ""
    echo "You can upload data using:"
    echo "  rsync -avz --progress ${DATA_SUBDIR}/ root@YOUR_POD_IP:${DATA_PATH}/"
    echo ""
    exit 1
fi

echo "Data found at: ${DATA_PATH}"
ls -la "${DATA_PATH}"
echo ""

# Create checkpoint directory
mkdir -p "${CHECKPOINT_DIR}"

# Run training
echo "=============================================="
echo "Starting training..."
echo "=============================================="

python -m scripts.train_unified \
    --schema="${SCHEMA}" \
    --tokenizer="${TOKENIZER}" \
    --model_size="${MODEL_SIZE}" \
    --data_root="${DATA_DIR}" \
    --model_tag="${EXPERIMENT_TAG}"

echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Checkpoints saved to: ${PROJECT_DIR}/checkpoints/${EXPERIMENT_TAG}/"
echo ""
echo "To download checkpoints:"
echo "  rsync -avz root@YOUR_POD_IP:${PROJECT_DIR}/checkpoints/${EXPERIMENT_TAG}/ ./${EXPERIMENT_TAG}/"
