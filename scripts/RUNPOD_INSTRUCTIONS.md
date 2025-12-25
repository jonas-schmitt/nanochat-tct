# RunPod Experiment Instructions

Step-by-step guide to run TCT experiments on RunPod.

## Prerequisites

Before starting, ensure you have:
1. A RunPod account with credits
2. `runpodctl` installed locally (optional, for fast uploads)
3. The experiment data prepared locally

## Step 1: Prepare Upload Package (Local Machine)

Run this on your local machine to create the upload package:

```bash
cd ~/git/nanochat-tct
bash scripts/prepare_runpod_upload.sh
```

This creates `runpod-upload/` containing:
- `tct-experiment-data.tar.gz` - Tokenized training data
- `tct-wheels/` - TCT tokenizer Python wheels
- `bpe-merges/` - BPE merge tables for UTF8 decoding
- `nanochat-tct-code.tar.gz` - Code archive

## Step 2: Create Network Volume (RunPod Console)

1. Go to https://console.runpod.io/storage
2. Click "New Network Volume"
3. Settings:
   - **Name**: `tct-experiments`
   - **Region**: Choose based on GPU availability (US preferred)
   - **Size**: 50 GB (sufficient for data + checkpoints)
4. Click "Create"

## Step 3: Upload Data to Network Volume

### Option A: Using RunPod File Browser

1. Create a temporary pod attached to the network volume (use cheapest GPU)
2. Use the web terminal's file upload feature
3. Upload all files from `runpod-upload/` to `/workspace/`

### Option B: Using runpodctl (Faster)

```bash
# Install runpodctl if not already installed
# See: https://github.com/runpod/runpodctl

# Set your API key
runpodctl config --apikey YOUR_API_KEY

# Send files (creates a receive code)
runpodctl send runpod-upload/

# On the RunPod pod:
# runpodctl receive <CODE>
```

### Option C: Using SSH/SCP

1. Start a pod with SSH enabled
2. Use SCP to upload:
```bash
scp -r runpod-upload/* root@POD_IP:/workspace/
```

## Step 4: Launch GPU Pod

1. Go to https://console.runpod.io/deploy
2. Select GPU:
   - **Recommended**: RTX 4090 (~$0.44/hr) - fastest, best cost-efficiency
   - **Alternative**: RTX 3090 (~$0.22/hr) or RTX A5000 (~$0.20/hr) - slower but lower hourly rate
3. Settings:
   - **Template**: RunPod PyTorch (latest)
   - **Network Volume**: Select `tct-experiments`
   - **Container Disk**: 20 GB (for temp files)
4. Click "Deploy"

## Step 5: Setup Pod Environment

Connect to the pod via web terminal or SSH, then run:

```bash
# Run the setup script
bash /workspace/nanochat-tct-code/scripts/runpod_setup.sh

# Or if using tarball:
cd /workspace
tar -xzf nanochat-tct-code.tar.gz
bash scripts/runpod_setup.sh
```

The setup script:
- Installs system dependencies (git, tmux, htop)
- Installs Python packages (torch, xgrammar)
- Installs TCT wheels
- Extracts data if needed
- Verifies the setup

## Step 6: Verify Setup

First, run a quick test to verify all configurations work:

```bash
cd /workspace/nanochat-tct
bash scripts/test_setup.sh
```

This runs 10 iterations for each of the 18 model configurations to verify:
- Data loading works
- Model initialization works
- Forward/backward pass works
- No OOM errors

## Step 7: Run Experiments

Experiments are split by schema. Run in separate tmux sessions or sequentially:

```bash
tmux new -s training
```

### TSConfig (fastest, ~10-15h on RTX 4090)

```bash
bash scripts/run_tsconfig.sh           # All 6 models
bash scripts/run_tsconfig.sh small     # Only small models
bash scripts/run_tsconfig.sh tct       # Only TCT tokenizer
```

### ESLintrc (~20-25h on RTX 4090)

```bash
bash scripts/run_eslintrc.sh           # All 6 models
bash scripts/run_eslintrc.sh small     # Only small models
```

### Kubernetes (longest, ~40-50h on RTX 4090)

```bash
bash scripts/run_kubernetes.sh         # All 6 models
bash scripts/run_kubernetes.sh small   # Only small models
```

### Resume After Interruption

All scripts automatically skip completed experiments (those with `best.pt`):

```bash
# Just run again - it will resume from where it left off
bash scripts/run_tsconfig.sh
```

## Step 8: Monitor Training

```bash
# View logs
tail -f logs/kubernetes_tct_small.log

# Check GPU usage
nvidia-smi -l 1

# Check disk usage
df -h /workspace
```

## Step 9: Download Results

When training completes, download the checkpoints:

```bash
# On the pod: compress checkpoints
cd /workspace/nanochat-tct
tar -czvf checkpoints.tar.gz checkpoints/

# Using runpodctl:
runpodctl send checkpoints.tar.gz

# On local machine:
runpodctl receive <CODE>
```

## Cost Estimates

| GPU | Price | Speed | 18 Models Est. | Wall Time |
|-----|-------|-------|----------------|-----------|
| RTX 4090 | $0.44/hr | 1x | ~$30-35 | ~70-80h |
| RTX 3090 | $0.22/hr | 0.45x | ~$35-45 | ~180h |
| RTX A5000 | $0.20/hr | 0.35x | ~$40-50 | ~200h |

**RTX 4090 is recommended** - faster training AND lower total cost due to 2.5-3x speed advantage.

Tips to minimize cost:
- Use spot instances when available (50-70% cheaper)
- Stop pod when not training
- Delete pod after downloading results (keep network volume)
- Run smaller models first to verify setup

## Troubleshooting

### Out of Memory (OOM)

If training crashes with OOM:
1. Reduce batch size: `--device_batch_size=N`
2. The batch sizes are optimized for 24GB VRAM; if using less, reduce proportionally

### Pod Disconnected

If your connection drops:
1. Reconnect via web terminal or SSH
2. Reattach to tmux: `tmux attach -t training`
3. If the training crashed, just run `bash scripts/run_experiments.sh` again

### Missing Data

If data verification fails:
1. Check if tarball was fully uploaded: `ls -la /workspace/*.tar.gz`
2. Re-extract: `cd /workspace/data && tar -xzf ../tct-experiment-data.tar.gz`

### Slow Training

If training is slower than expected:
1. Verify GPU is being used: `nvidia-smi`
2. Check for CPU bottleneck: `htop`
3. Data loading should use GPU; check that device is `cuda`

## Experiment Order

Recommended order (fastest to slowest):

1. `tsconfig small` (~1h) - Quick validation
2. `eslintrc small` (~2h)
3. `kubernetes small` (~4h)
4. `tsconfig medium` (~2h)
5. `eslintrc medium` (~4h)
6. `kubernetes medium` (~12h)
7. `tsconfig large` (~6h)
8. `eslintrc large` (~10h)
9. `kubernetes large` (~30h)

Run TCT and UTF8 in parallel if you have multiple GPUs.
