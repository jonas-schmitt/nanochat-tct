# Environment Setup for Nanochat-TCT

Complete setup guide for training a decoder-only transformer on GitHub Actions workflows using TCT tokenization.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Verification](#verification)
5. [Training Setup](#training-setup)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware

**Minimum (for testing)**:
- CPU: 4+ cores
- RAM: 16GB
- Storage: 50GB SSD
- GPU: Not required (CPU-only mode works for small tests)

**Recommended (for full training)**:
- CPU: 16+ cores
- RAM: 64GB
- Storage: 200GB NVMe SSD
- GPU: NVIDIA A100 (40GB VRAM) or H100 (80GB VRAM)

### Software

**Operating System**:
- **Required**: Linux x86_64 with glibc 2.34+
- **Tested**: Ubuntu 22.04 LTS, Debian 12
- **Not supported**: macOS, Windows, older Linux distributions

**Python**:
- **Version**: 3.12+ (CPython only, PyPy not supported)
- **Package manager**: pip 24.0+

**CUDA** (for GPU training):
- **Version**: 12.1+ (for PyTorch 2.5+)
- **Drivers**: NVIDIA 535+

---

## Installation

### Step 1: Create Virtual Environment

```bash
# Create project directory
mkdir -p nanochat-tct
cd nanochat-tct

# Create virtual environment
python3.12 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 2: Install TCT Tokenizer

```bash
# Install TCT wheel from CI (Linux x86_64, Python 3.12+)
# Download the latest wheel from TCT releases or CI artifacts
pip install tct_github_workflow-1.0.5-cp312-cp312-manylinux_2_34_x86_64.whl

# Verify installation
python3 -c "from tct_github_workflow import encode, decode, vocab_size; print(f'TCT vocab: {vocab_size()}')"
# Expected output: TCT vocab: 8192
```

### Step 3: Install Training Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Data processing
pip install numpy pandas

# Training utilities
pip install tqdm wandb

# Development tools
pip install pytest black mypy ipython

# Verify PyTorch installation
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Expected output: PyTorch: 2.5.0, CUDA: True (or False if CPU-only)
```

### Step 4: Clone Data Scripts

```bash
# Extract bundle scripts
tar -xzf nanochat-tct-bundle.tar.gz

# Verify structure
tree nanochat-tct-bundle/
```

---

## Data Preparation

### Step 1: Verify Workflow Data

**For nanochat-tct**: Workflows are already available at `~/Desktop/data/workflows-100k/json/`

```bash
# Verify workflows exist
ls ~/Desktop/data/workflows-100k/json/ | wc -l
# Expected: 100000

# Check a sample workflow
head -n 20 ~/Desktop/data/workflows-100k/json/$(ls ~/Desktop/data/workflows-100k/json/ | head -1)
```

### Step 2: Prepare Training Windows

```bash
# Create training windows with decoder-only position tokens
cd ~/git/nanochat-tct/tct-bundle/scripts

python3 prepare_training_data.py \
  --input ~/Desktop/data/workflows-100k/json/ \
  --output ~/Desktop/data/prepared/ \
  --context-size 512 \
  --train-split 0.8

# Expected output:
# Found 100000 JSON workflow files
# Context size: 512
#
# Processing workflows: 100000/100000
# Extracted ~2,500,000 training windows
#
# Tensor shape: (2500000, 513)
#   (num_windows, window_size) = (2500000, 513)
#
# Train/val split:
#   Train: 2,000,000 windows (80%)
#   Val:   500,000 windows (20%)
#
# ✅ Saved training data:
#   Train: ~/Desktop/data/prepared/train.pt
#   Val:   ~/Desktop/data/prepared/val.pt
#
# ✅ Saved metadata: ~/Desktop/data/prepared/metadata.json
```

**Output structure**:
```
~/Desktop/data/prepared/
├── train.pt          # Training windows (PyTorch tensor)
├── val.pt            # Validation windows (PyTorch tensor)
└── metadata.json     # Dataset statistics
```

---

## Verification

### Test 1: Tokenization Round-Trip

```bash
cd examples

# Run quick start example
python3 example_quickstart.py

# Expected output:
# ✅ Encoded: 42 tokens
# ✅ Decoded: {"name": "CI", "on": "push", ...}
# ✅ Round-trip: SUCCESS (input == output)
```

### Test 2: Windowed Generation

```bash
# Test decoder-only windowed generation
python3 example_windowed.py

# Expected output:
# Full sequence: 180 tokens
# Window 0: [0, tok_0, tok_1, ..., tok_255]  (position=0)
# Window 50: [50, tok_50, tok_51, ..., tok_305]  (position=50)
# ✅ All windows have correct structure (1 position token + content)
```

### Test 3: TCT vs Tiktoken Comparison

```bash
# Install tiktoken
pip install tiktoken

# Run comparison
python3 example_vs_tiktoken.py

# Expected output:
# TCT tokens: 252
# Tiktoken tokens: 493
# Compression improvement: 48.9% fewer tokens
```

### Test 4: Data Loading

```bash
# Verify prepared data can be loaded
python3 -c "
import torch
train_data = torch.load('data/prepared/train.pt')
print(f'Train data shape: {train_data.shape}')
# Expected: Train data shape: torch.Size([2000000, 257])
#                                            ↑        ↑
#                                        windows  context+position
"
```

---

## Training Setup

### Configuration File

Create `config.py`:

```python
# Model architecture
vocab_size = 8192          # TCT vocabulary
context_size = 256         # Window size (excluding position token)
n_layer = 6                # Transformer layers
n_head = 6                 # Attention heads
n_embd = 384               # Embedding dimension
dropout = 0.1              # Dropout rate

# Training
batch_size = 32            # Per-device batch size
gradient_accumulation = 4  # Effective batch = 128
learning_rate = 3e-4       # Peak learning rate
weight_decay = 0.1         # AdamW weight decay
beta1 = 0.9                # Adam beta1
beta2 = 0.95               # Adam beta2
grad_clip = 1.0            # Gradient clipping max norm

# Learning rate schedule
warmup_iters = 2000        # Warmup iterations
lr_decay_iters = 100000    # Total iterations
min_lr = 3e-5              # Minimum learning rate

# Logging
eval_interval = 500        # Evaluate every N iterations
save_interval = 5000       # Save checkpoint every N iterations
log_interval = 10          # Log metrics every N iterations
eval_iters = 200           # Validation batches per evaluation

# Paths
data_dir = "data/prepared/"
checkpoint_dir = "checkpoints/"
log_dir = "logs/"

# Wandb (optional)
wandb_project = "nanochat-tct"
wandb_run_name = "exp_baseline_6layer_384embd"
```

### Launch Training

```bash
# Create checkpoint directory
mkdir -p checkpoints logs

# Start training
python3 train.py --config config.py

# Or with command-line overrides
python3 train.py \
  --config config.py \
  --batch-size 64 \
  --learning-rate 5e-4 \
  --max-iters 50000
```

### Monitor Training

**Option 1: Terminal logs**

```bash
# Training will print metrics every log_interval
# Output:
# iter 100 | loss 5.234 | lr 0.000030 | grad_norm 0.87 | tokens/sec 15000 | cost $0.42
# iter 200 | loss 4.891 | lr 0.000060 | grad_norm 0.92 | tokens/sec 15200 | cost $0.83
```

**Option 2: Weights & Biases (wandb)**

```bash
# Login to wandb
wandb login

# Metrics will be logged automatically
# View at: https://wandb.ai/<your-username>/nanochat-tct
```

**Option 3: TensorBoard**

```bash
# Start TensorBoard
tensorboard --logdir logs/

# Open browser: http://localhost:6006
```

---

## Verification Checklist

Before starting full training, verify:

- [ ] **TCT tokenizer installed**: `python3 -c "from tct_github_workflow import vocab_size; print(vocab_size())"`
- [ ] **PyTorch with CUDA**: `python3 -c "import torch; print(torch.cuda.is_available())"`
- [ ] **Round-trip verification**: `cd examples && python3 example_quickstart.py`
- [ ] **Windowed generation**: `cd examples && python3 example_windowed.py`
- [ ] **Data prepared**: `ls data/prepared/train.pt data/prepared/val.pt`
- [ ] **Training smoke test**: `python3 train.py --max-iters 10`

---

## Troubleshooting

### Issue: TCT wheel installation fails

**Error**:
```
ERROR: tct_github_workflow-1.0.5-cp312-cp312-manylinux_2_34_x86_64.whl is not a supported wheel on this platform.
```

**Solution**:
- Verify Python version: `python3 --version` (must be 3.12+)
- Verify platform: `uname -m` (must be x86_64)
- Verify glibc: `ldd --version` (must be 2.34+)
- Use Ubuntu 22.04+ or Debian 12+

### Issue: CUDA not available

**Error**:
```python
import torch
print(torch.cuda.is_available())  # False
```

**Solution**:
```bash
# Check NVIDIA drivers
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python3 -c "import torch; print(torch.version.cuda)"
```

### Issue: OOM (Out of Memory) during training

**Error**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:
1. **Reduce batch size**: `--batch-size 16` (instead of 32)
2. **Enable gradient checkpointing**: Add `use_checkpoint=True` in model config
3. **Use mixed precision**: Add `torch.cuda.amp.autocast()`
4. **Reduce model size**: `--n-layer 4 --n-embd 256`

### Issue: Training loss not decreasing

**Symptoms**:
- Loss stays constant or increases
- Perplexity doesn't improve

**Debugging**:
1. **Overfitting test**: Train on 10 samples → should reach near-zero loss
2. **Check learning rate**: Plot loss vs LR → find optimal range
3. **Inspect gradients**: Print grad norms → ensure not vanishing/exploding
4. **Verify data**: Print decoded samples → check data pipeline

```bash
# Overfitting test
python3 train.py --train-samples 10 --max-iters 1000

# Expected: Loss should decrease to <0.1
```

### Issue: Slow training speed

**Symptoms**:
- <5000 tokens/second on A100

**Optimizations**:
1. **Enable flash attention**: Install `pip install flash-attn`
2. **Compile model**: Use `torch.compile()` (PyTorch 2.0+)
3. **Increase num_workers**: `DataLoader(num_workers=8)`
4. **Mixed precision**: Use `torch.cuda.amp` for 2x speedup

```python
# Enable compilation (PyTorch 2.0+)
model = torch.compile(model)
```

### Issue: Round-trip verification fails

**Error**:
```python
AssertionError: Round-trip mismatch! decoded != original
```

**Solution**:
- This should NEVER happen with TCT tokenizer
- Report bug to TCT maintainers with failing workflow
- Check if workflow JSON is valid schema

```bash
# Verify JSON schema
python3 -c "
import json
import jsonschema

with open('workflow.json') as f:
    workflow = json.load(f)

# Should have 'on' and 'jobs' fields
assert 'on' in workflow
assert 'jobs' in workflow
"
```

---

## Performance Benchmarks

### Expected Training Speed (A100 40GB)

| Batch Size | Tokens/sec | Iterations/sec | GPU Memory | Hours to 100k iters |
|------------|------------|----------------|------------|---------------------|
| 16         | 8,000      | 31             | 18GB       | 89 hours            |
| 32         | 15,000     | 58             | 28GB       | 48 hours            |
| 64         | 22,000     | 85             | 38GB       | 33 hours            |

**Note**: Actual speed depends on GPU model, CUDA version, and PyTorch optimizations.

### Cost Estimates (AWS p4d.24xlarge with 8x A100)

| Configuration | Tokens/sec | Hours to 100k | Cost ($2.50/hour) |
|---------------|------------|---------------|-------------------|
| Batch=32, Single GPU | 15,000 | 48h | $120 |
| Batch=32, 2x GPUs | 30,000 | 24h | $60 |
| Batch=32, 4x GPUs | 60,000 | 12h | $30 |

**Budget target**: $100 → Use single A100 for ~40 hours or 2x A100 for ~20 hours.

---

## Next Steps

After setup is complete:

1. **Verify all tests pass**: Run through Verification section
2. **Run training smoke test**: `python3 train.py --max-iters 100`
3. **Start full training**: `python3 train.py --config config.py`
4. **Monitor progress**: Use wandb or TensorBoard
5. **Evaluate checkpoints**: Run validation and sample generation
6. **Tune hyperparameters**: Experiment with different configs

---

**For detailed API reference, see [TCT.md](TCT.md)**
**For development guidelines, see [CLAUDE.md](CLAUDE.md)**

---

*Last updated: 2025-11-02*
