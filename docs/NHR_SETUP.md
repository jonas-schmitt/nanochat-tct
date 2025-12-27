# NHR FAU Cluster Setup Guide

Setup instructions for running TCT experiments on NHR FAU clusters (Alex, TinyGPU).

## Prerequisites

1. **NHR account**: You need an NHR FAU username (e.g., `ab12cdef`)
2. **SSH key**: Set up SSH key authentication for the cluster
3. **Local data**: Pre-tokenized datasets in `~/Desktop/data/`

## Step 1: Upload Data

Use the upload script to transfer datasets to the cluster:

```bash
# Upload all datasets
bash scripts/upload_nhr.sh <username>

# Upload specific schema only
bash scripts/upload_nhr.sh <username> kubernetes

# Custom data path
bash scripts/upload_nhr.sh <username> --data=/path/to/data

# Use TinyGPU instead of Alex
bash scripts/upload_nhr.sh <username> --cluster=tinygpu

# Dry run (see what would be uploaded)
bash scripts/upload_nhr.sh <username> --dry-run --verbose
```

### Datasets Uploaded

| Schema | TCT Directory | UTF8 Directory |
|--------|---------------|----------------|
| tsconfig | `tsconfig-tct-base` | `tsconfig-utf8-base-matched` |
| eslintrc | `eslintrc-tct-bpe-500` | `eslintrc-utf8-bpe-500` |
| kubernetes | `kubernetes-tct-bpe` | `kubernetes-utf8-bpe` |

Data is uploaded to `$WORK/data/tct/` on the cluster.

## Step 2: SSH to Cluster

```bash
ssh <username>@alex.nhr.fau.de
# or
ssh <username>@tinygpu.nhr.fau.de
```

## Step 3: Start Interactive Job

Request a GPU node for setup:

```bash
# Alex cluster (A100)
srun --partition=a100 --gres=gpu:a100:1 --time=01:00:00 --pty bash -l

# TinyGPU cluster
srun --partition=gpu --gres=gpu:1 --time=01:00:00 --pty bash -l
```

## Step 4: Clone Repository and Setup

```bash
cd $WORK
git clone https://github.com/jonas-schmitt/nanochat-tct.git
cd nanochat-tct
bash scripts/setup.sh
```

The setup script will:
- Create a Python virtual environment
- Install PyTorch and dependencies
- Install TCT tokenizers from wheels (if available in `$WORK/tct-wheels/`)

## Step 5: Exit Interactive Job and Submit Training

```bash
# Exit the interactive session
exit

# Navigate to repo
cd $WORK/nanochat-tct

# Submit training jobs
bash scripts/submit.sh kubernetes small medium --gpu=a100
```

## Cluster-Specific Notes

### Alex Cluster

- **Partitions**: `a100` (NVIDIA A100 80GB)
- **GPU memory**: 80GB VRAM
- **Batch sizes**: Can use larger micro batches (see `model_configs.py`)

### TinyGPU Cluster

- **Partitions**: `gpu` (various GPUs)
- **GPU memory**: Varies by node
- **Batch sizes**: May need to reduce for smaller GPUs

## File Locations on Cluster

| Item | Path |
|------|------|
| Data | `$WORK/data/tct/` |
| Code | `$WORK/nanochat-tct/` |
| TCT wheels | `$WORK/tct-wheels/` |
| Checkpoints | `$WORK/nanochat-tct/checkpoints/` |
| Logs | `$WORK/nanochat-tct/logs/` |

## Troubleshooting

### SSH Connection Fails

```bash
# Set up SSH key
ssh-copy-id <username>@alex.nhr.fau.de
```

### TCT Tokenizer Not Found

```bash
# Manually install from wheels
pip install $WORK/tct-wheels/*.whl
```

### Out of Memory

Reduce batch size in training command:
```bash
python -m scripts.train_unified \
    --schema=kubernetes \
    --tokenizer=tct \
    --model_size=small \
    --device_batch_size=8  # Reduce from default 16
```

### Job Queue Full

Check queue status:
```bash
squeue -u $USER
```

Cancel a job:
```bash
scancel <job_id>
```
