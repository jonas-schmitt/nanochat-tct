#!/bin/bash
#SBATCH --job-name=tct-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_%j_%x.log
#SBATCH --error=logs/slurm_%j_%x.err

# TCT Experiment Training Script for NHR Cluster
#
# Usage:
#   sbatch train_nhr.sh --schema kubernetes --tokenizer tct --model_size small
#
# Or with environment variables:
#   SCHEMA=kubernetes TOKENIZER=tct MODEL_SIZE=small sbatch train_nhr.sh

set -e

# Load required modules (adjust for your cluster)
module load cuda/12.1
module load python/3.12

# Activate virtual environment
source ~/venvs/tct/bin/activate

# Change to project directory
cd ~/git/nanochat-tct

# Create logs directory
mkdir -p logs

# Parse command line arguments or use environment variables
SCHEMA=${SCHEMA:-${1:-kubernetes}}
TOKENIZER=${TOKENIZER:-${2:-tct}}
MODEL_SIZE=${MODEL_SIZE:-${3:-small}}
EPOCHS=${EPOCHS:-}
DATA_ROOT=${DATA_ROOT:-~/Desktop/data}

# Build experiment tag
EXPERIMENT_TAG="${SCHEMA}_${TOKENIZER}_${MODEL_SIZE}"

echo "=============================================="
echo "TCT Experiment: ${EXPERIMENT_TAG}"
echo "=============================================="
echo "Schema: ${SCHEMA}"
echo "Tokenizer: ${TOKENIZER}"
echo "Model size: ${MODEL_SIZE}"
echo "Data root: ${DATA_ROOT}"
echo "SLURM Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

# Run training
python -m scripts.train_unified \
    --schema="${SCHEMA}" \
    --tokenizer="${TOKENIZER}" \
    --model_size="${MODEL_SIZE}" \
    --data_root="${DATA_ROOT}" \
    --model_tag="${EXPERIMENT_TAG}" \
    ${EPOCHS:+--epochs="${EPOCHS}"}

echo "=============================================="
echo "Training complete!"
echo "=============================================="
