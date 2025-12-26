# NHR FAU Training Guide

Step-by-step guide for running TCT experiments on NHR FAU clusters (Alex/TinyGPU).

## Prerequisites

- NHR FAU account with access to Alex or TinyGPU
- SSH key configured for cluster access
- Local data prepared in `~/Desktop/data/`

## Step 1: Upload Data

From your local machine:

```bash
cd ~/git/nanochat-tct

# Upload all schemas
bash scripts/upload_nhr.sh <username>

# Or upload specific schemas
bash scripts/upload_nhr.sh <username> kubernetes

# Dry run first to see what will be uploaded
bash scripts/upload_nhr.sh <username> --dry-run
```

This uploads:
- `tsconfig-tct-base`, `tsconfig-utf8-base-matched`
- `eslintrc-tct-bpe-500`, `eslintrc-utf8-bpe-500`
- `kubernetes-tct-bpe`, `kubernetes-utf8-bpe`
- `tct-wheels/` (if present)

## Step 2: SSH to Cluster

```bash
ssh <username>@alex.nhr.fau.de
# or
ssh <username>@tinygpu.nhr.fau.de
```

## Step 3: Setup Environment (One-time)

**Important**: Run setup in an interactive GPU job to ensure proper CUDA detection.

```bash
# Request interactive job with GPU
srun --partition=a100 --gres=gpu:a100:1 --time=01:00:00 --pty bash -l

# Clone repository
cd $WORK
git clone https://github.com/jonas-schmitt/nanochat-tct.git
cd nanochat-tct

# Run setup
bash scripts/setup.sh

# Verify installation
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import tct_kubernetes_20k; print('TCT OK')"

# Exit interactive job
exit
```

## Step 4: Submit Training Jobs

From the login node:

```bash
cd $WORK/nanochat-tct

# Submit kubernetes training (small, small-deep, medium)
bash scripts/submit.sh kubernetes small small-deep medium --gpu=a100

# Submit with specific GPU
bash scripts/submit.sh kubernetes small --gpu=a40

# Resume from checkpoint
bash scripts/submit.sh kubernetes resume --gpu=a100

# Multiple schemas
bash scripts/submit.sh kubernetes tsconfig small medium --gpu=a100
```

### GPU Options

| GPU | Cluster | VRAM | Batch Multiplier |
|-----|---------|------|------------------|
| `a40` | Alex | 40GB | 2x |
| `a100` | Alex/TinyGPU | 40GB | 2x |
| `a100_80` | Alex | 80GB | 3x |
| `v100` | TinyGPU | 32GB | 1x |
| `rtx3080` | TinyGPU | 10GB | 1x |

### Other Options

```bash
--time=HH:MM:SS   # Wall time (default: 24:00:00)
--setup           # Run setup before training
--dry-run         # Show job script without submitting
```

## Step 5: Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Watch job output
tail -f logs/slurm_<jobid>.out

# Cancel job
scancel <jobid>

# Check completed jobs
sacct -u $USER --format=JobID,JobName,State,Elapsed,MaxRSS
```

## Step 6: Download Results

From your local machine:

```bash
# Download checkpoints
rsync -avz <username>@alex.nhr.fau.de:\$WORK/nanochat-tct/checkpoints/ ./checkpoints/

# Download logs
rsync -avz <username>@alex.nhr.fau.de:\$WORK/nanochat-tct/logs/ ./logs/
```

## Directory Structure on NHR

```
$WORK/
├── nanochat-tct/           # Code repository
│   ├── checkpoints/        # Trained models
│   ├── logs/               # Training logs + Slurm output
│   └── scripts/            # Run/submit scripts
├── data/tct/               # Training data
│   ├── kubernetes-tct-bpe/
│   ├── kubernetes-utf8-bpe/
│   ├── tsconfig-tct-base/
│   ├── tsconfig-utf8-base-matched/
│   ├── eslintrc-tct-bpe-500/
│   └── eslintrc-utf8-bpe-500/
├── tct-wheels/             # TCT tokenizer wheels
└── venv-tct/               # Python virtual environment
```

## Troubleshooting

### Job fails immediately
- Check `logs/slurm_<jobid>.err` for errors
- Verify data exists: `ls $WORK/data/tct/`
- Verify venv exists: `ls $WORK/venv-tct/`

### CUDA not available
- Re-run setup in an interactive GPU job
- Check module is loaded: `module list`

### Out of memory
- Use smaller batch size (happens automatically based on GPU)
- Try a different GPU with more VRAM

### Package installation fails
- Proxy should be auto-configured in setup.sh
- Try running setup in interactive job

## Quick Reference

```bash
# Upload data
bash scripts/upload_nhr.sh <username>

# Setup (in interactive job)
srun --partition=a100 --gres=gpu:a100:1 --time=01:00:00 --pty bash -l
bash scripts/setup.sh

# Submit job
bash scripts/submit.sh kubernetes small medium --gpu=a100

# Monitor
squeue -u $USER
tail -f logs/slurm_*.out
```
