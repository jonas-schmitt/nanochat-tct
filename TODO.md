# TCT-Nanochat Integration TODO

**Goal**: Train a high-quality language model for GitHub Actions workflow generation using TCT's schema-aware tokenization and nanochat's training infrastructure.

**Status**: ✅ Proof-of-Concept Complete | Ready for Phase 3 (Full Training)

---

## Phase 1: Setup & Verification ✅ (Reversible)

### 1.1 Environment Setup ✅
- [x] Python 3.12 virtual environment created
- [x] Nanochat dependencies installed (PyTorch, wandb, datasets, etc.)
- [x] TCT tokenizer wheel installed (`tct_github_workflow`)
- [x] Vocabulary size verified (8192 tokens)
- [x] Round-trip encoding/decoding tested

### 1.2 TCT Verification ✅
- [x] Run `example_quickstart.py` - Basic encode/decode round-trip
  **Result**: ✅ Round-trip SUCCESS, 13 tokens, 9.31 chars/token compression

- [x] Run `example_windowed.py` - Decoder-only windowed generation
  **Result**: ✅ Windows have correct structure (1 position token + content)

- [x] Verify data location exists
  **Result**: ✅ 100,000 workflow files available

### 1.3 Sample Data Preparation ✅
- [x] Prepare sample dataset (100 workflows for PoC)
  **Result**: ✅ Created `~/Desktop/data/prepared-100/`
  - Train: 40,096 windows (~20M tokens)
  - Val: 10,025 windows

- [x] Verify data shape
  **Result**: ✅ `torch.Size([40096, 513])` (513 = 1 position + 512 context)

---

## Phase 2: Integration (First Training Run) ✅

### 2.1 Create Training Script ✅
- [x] Created `scripts/tct_train.py` with TCT integration
  - ✅ TCT tokenizer adapter imported
  - ✅ TCT dataloader for pre-prepared windows
  - ✅ Optimized "small" config (20M params, 384d×8L×6H)
  - ✅ Nanochat training loop reused unchanged

- [ ] Key integration points:
  ```python
  # 1. Tokenizer (replace BPE)
  from tct_tokenizer_adapter import TCTTokenizer
  tokenizer = TCTTokenizer()

  # 2. Dataloader (use pre-prepared data)
  from tct_dataloader import tokenizing_distributed_data_loader
  train_loader = tokenizing_distributed_data_loader(
      device_batch_size=32,
      context_size=512,
      split="train",
      data_dir="~/Desktop/data/prepared-sample/"
  )

  # 3. Model config (TCT-optimized)
  from model_config import get_config
  config = get_config("small")  # 50M params, 8192 vocab, 512 context
  ```

### 2.2 Smoke Test ✅
- [x] Ran 10 training iterations on sample data
  **Result**: ✅ All checks passed

- [x] Verified smoke test checklist:
  - [x] Script runs without import errors ✅
  - [x] Loss computed (not NaN) ✅ (9.01 → 0.023)
  - [x] Loss value reasonable ✅
  - [x] Gradients flow ✅ (healthy grad norms)
  - [x] No CUDA OOM errors ✅
  - [x] Checkpoint saving works ✅

### 2.3 Proof-of-Concept Training (100 Workflows) ✅
- [x] Trained on 100 workflows for 5,000 iterations (~7 minutes)
  **Result**: ✅ Perfect overfitting (train loss 0.023, val loss 5.57)

- [x] Created generation script `scripts/generate_workflow.py`
  **Result**: ✅ Generates valid JSON workflows with TCT decoding

- [x] Tested model generation with TCT encoding/decoding
  **Result**: ✅ 100% valid JSON, schema-compliant workflows
  - See `POC_RESULTS.md` for detailed analysis

**Decision point**: ✅ PoC successful! Ready for Phase 3 (full dataset)

---

## Phase 3: Full Training (Scale Up) - SMALL MODEL

**Configuration**: context_size=512, stride=32, vocab_size=8192
**Model**: Small (20M params) - Fast experimentation baseline
**Budget**: ~$15, ~2 hours on 8×A100 (or ~16 hours on single GPU)

### 3.1 Full Data Preparation with Striding ⚠️ NEW

**IMPORTANT**: We now use stride=32 to keep vocab_size=8192

- [ ] **Step 1**: Verify `prepare_training_data.py` supports `--stride` parameter
  ```bash
  cd ~/git/nanochat-tct/tct-bundle/scripts
  python prepare_training_data.py --help | grep -i stride
  ```

  **If stride is not supported**, we need to modify the script first (see below)

- [ ] **Step 2**: Prepare complete dataset with stride
  ```bash
  cd ~/git/nanochat-tct/tct-bundle/scripts
  python prepare_training_data.py \
    --input ~/Desktop/data/workflows-100k/json/ \
    --output ~/Desktop/data/prepared-100k-512-s32/ \
    --context-size 512 \
    --stride 32 \
    --train-split 0.8
  ```

  **Expected time**: ~10-15 minutes (100k workflows, strided windowing)
  **Expected output**:
  - `train.pt` (shape: [N, 513] where 513 = 1 position + 512 content)
  - `val.pt` (shape: [M, 513])
  - `metadata.json` (includes stride info)

  **Expected sizes**:
  - Fewer windows than stride=1 (better: focuses on important parts)
  - Total windows: ~500k-1M (estimate, depends on workflow lengths)
  - Disk space: ~2-4GB total

- [ ] **Step 3**: Verify prepared dataset
  ```bash
  cd ~/git/nanochat-tct
  python3 -c "
  import torch, json
  from pathlib import Path

  data_dir = Path.home() / 'Desktop/data/prepared-100k-512-s32'
  train = torch.load(data_dir / 'train.pt')
  val = torch.load(data_dir / 'val.pt')

  with open(data_dir / 'metadata.json') as f:
      meta = json.load(f)

  print('=== Dataset Verification ===')
  print(f'Train shape: {train.shape}')
  print(f'Val shape: {val.shape}')
  print(f'Context size: {meta[\"context_size\"]}')
  print(f'Stride: {meta.get(\"stride\", \"NOT SET - CHECK!\")}')
  print(f'Total workflows: {meta[\"total_workflows\"]}')
  print()
  print('Sample window (first 20 tokens):')
  print(f'Position token: {train[0, 0].item()}')
  print(f'Content tokens: {train[0, 1:21].tolist()}')
  print()
  print('Max position token:', train[:, 0].max().item())
  print('Expected: < 8192 for vocab compatibility')
  "
  ```

  **Validation checklist**:
  - [ ] Train shape is [N, 513] ✓
  - [ ] Val shape is [M, 513] ✓
  - [ ] `metadata.json` has `"stride": 32` ✓
  - [ ] Max position token < 8192 ✓ (CRITICAL!)
  - [ ] Context size = 512 ✓

### 3.2 Choose Model Size & Budget
- [ ] **Decision**: Select model configuration based on budget

  | Model | Params | Architecture | Cost | Time (8×A100) | Use Case |
  |-------|--------|--------------|------|---------------|----------|
  | Small | **20M** | **384d×8L×6H** | $15 | 2h | **Optimized for workflows** ⭐ |
  | Medium | 100M | 768d×8L×12H | $50 | 7h | Production quality 🎯 |
  | Large | 200M | 1024d×12L×16H | $100 | 15h | Maximum quality 🏆 |

- [ ] Update `scripts/tct_train.py` with chosen config:
  ```python
  config = get_config("medium")  # or "small" / "large"
  ```

### 3.3 Launch Full Training with Checkpointing & Resumption

**Goal**: Train Small (20M) model on 100k workflows with proper checkpointing for continuation if interrupted

- [ ] **Step 1**: Update model config to use correct parameters
  ```bash
  cd ~/git/nanochat-tct

  # Verify model config has been updated to context_size=512
  python3 -c "
  import sys
  sys.path.insert(0, 'tct-bundle/adapters')
  from model_config import get_config

  config = get_config('small')
  print('=== Small Model Configuration ===')
  print(f'Vocab size: {config[\"vocab_size\"]}')
  print(f'Context size: {config[\"context_size\"]}')
  print(f'd_model: {config[\"d_model\"]}')
  print(f'n_layers: {config[\"n_layers\"]}')
  print(f'n_heads: {config[\"n_heads\"]}')

  assert config['vocab_size'] == 8192, 'Vocab must be 8192!'
  assert config['context_size'] == 512, 'Context must be 512!'
  print('✅ Config validated')
  "
  ```

  **If context_size ≠ 512**, update `tct-bundle/adapters/model_config.py` first!

- [ ] **Step 2**: Create experiment directory with proper structure
  ```bash
  # Create organized experiment directory
  EXP_NAME="exp001-small-100k-ctx512-s32"
  EXP_DIR="experiments/${EXP_NAME}"

  mkdir -p ${EXP_DIR}/{checkpoints,logs,generated_samples}

  cd ~/git/nanochat-tct
  echo "# Experiment 001 - Small Model Baseline (100k workflows)

**Date**: $(date)
**Configuration**:
- Model: Small (20M params, 384d×8L×6H)
- Data: 100k workflows
- Context size: 512 tokens
- Stride: 32 (position mapping for vocab_size=8192)
- Vocab size: 8,192 (TCT base, no expansion)
- Training: 50,000 iterations
- Batch size: 32 (effective batch with gradient accumulation)
- Learning rate: 3e-4 with warmup
- Checkpoints: Every 5,000 iterations
- Evaluation: Every 500 iterations

**Goal**: Establish baseline performance for context_size=512 approach

**Status**: Ready to launch
  " > ${EXP_DIR}/README.md

  # Save config snapshot
  cp tct-bundle/adapters/model_config.py ${EXP_DIR}/model_config_snapshot.py
  cp scripts/tct_train.py ${EXP_DIR}/train_script_snapshot.py

  echo "Experiment directory created: ${EXP_DIR}"
  ```

- [ ] **Step 3**: Launch training with proper checkpointing
  ```bash
  cd ~/git/nanochat-tct

  # Set experiment parameters
  EXP_NAME="exp001-small-100k-ctx512-s32"
  DATA_DIR="${HOME}/Desktop/data/prepared-100k-512-s32"
  CHECKPOINT_DIR="${HOME}/Desktop/checkpoints/${EXP_NAME}"
  LOG_FILE="experiments/${EXP_NAME}/logs/training_$(date +%Y%m%d_%H%M%S).log"

  mkdir -p ${CHECKPOINT_DIR}
  mkdir -p $(dirname ${LOG_FILE})

  # Launch training (single GPU)
  python -m scripts.tct_train \
    --model_size=small \
    --data_dir=${DATA_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --num_iterations=50000 \
    --device_batch_size=32 \
    --eval_every=500 \
    --learning_rate=3e-4 \
    --warmup_iters=2000 \
    --grad_clip=1.0 \
    2>&1 | tee ${LOG_FILE}

  # For multi-GPU (8×A100), use:
  # torchrun --standalone --nproc_per_node=8 -m scripts.tct_train \
  #   --model_size=small \
  #   --data_dir=${DATA_DIR} \
  #   --checkpoint_dir=${CHECKPOINT_DIR} \
  #   --num_iterations=50000 \
  #   --device_batch_size=32 \
  #   --eval_every=500 \
  #   2>&1 | tee ${LOG_FILE}
  ```

  **Training will create**:
  - Checkpoints every 5,000 iterations:
    - `model_005000.pt`, `model_010000.pt`, ..., `model_050000.pt`
    - `optim_005000.pt` (optimizer state for resumption)
    - `meta_005000.json` (training metadata)
  - Log file with full output in `experiments/${EXP_NAME}/logs/`

- [ ] **Step 4**: Monitor training (check every 30 min for first 2 hours)

  **Health Checks**:
  - [ ] Loss decreases steadily (target: <2.0 by iteration 50k)
  - [ ] Gradient norms healthy (0.5 - 2.0 range, logged in output)
  - [ ] Validation loss decreases with training loss
  - [ ] GPU utilization >80% (check with `nvidia-smi`)
  - [ ] No OOM errors in logs
  - [ ] Checkpoints save successfully (check ${CHECKPOINT_DIR})

  **Quick monitoring script**:
  ```bash
  # Monitor latest log file
  tail -f experiments/exp001-small-100k-ctx512-s32/logs/*.log

  # Check GPU usage
  watch -n 5 nvidia-smi

  # Check latest checkpoint
  ls -lth ${HOME}/Desktop/checkpoints/exp001-small-100k-ctx512-s32/ | head -5
  ```

### 3.4 Training Continuation / Resumption (if interrupted)

**IMPORTANT**: If training is interrupted (crash, OOM, manual stop, etc.), you can resume from the last checkpoint!

- [ ] **Find last checkpoint**:
  ```bash
  cd ~/git/nanochat-tct
  EXP_NAME="exp001-small-100k-ctx512-s32"
  CHECKPOINT_DIR="${HOME}/Desktop/checkpoints/${EXP_NAME}"

  # List all checkpoints
  ls -lt ${CHECKPOINT_DIR}/model_*.pt

  # Find the latest iteration number
  LATEST_CKPT=$(ls ${CHECKPOINT_DIR}/model_*.pt | sed 's/.*model_0*\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
  echo "Latest checkpoint: iteration ${LATEST_CKPT}"
  ```

- [ ] **Resume training from checkpoint**:
  ```bash
  cd ~/git/nanochat-tct

  EXP_NAME="exp001-small-100k-ctx512-s32"
  DATA_DIR="${HOME}/Desktop/data/prepared-100k-512-s32"
  CHECKPOINT_DIR="${HOME}/Desktop/checkpoints/${EXP_NAME}"

  # Get latest checkpoint iteration
  LATEST_CKPT=$(ls ${CHECKPOINT_DIR}/model_*.pt | sed 's/.*model_0*\([0-9]*\)\.pt/\1/' | sort -n | tail -1)

  # Resume training (will load model_XXXXXX.pt and optim_XXXXXX.pt)
  python -m scripts.tct_train \
    --model_size=small \
    --data_dir=${DATA_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --num_iterations=50000 \
    --device_batch_size=32 \
    --eval_every=500 \
    --resume_from_iteration=${LATEST_CKPT} \
    2>&1 | tee experiments/${EXP_NAME}/logs/training_resumed_$(date +%Y%m%d_%H%M%S).log
  ```

  **What happens on resume**:
  1. Script loads `model_${LATEST_CKPT}.pt` (model weights)
  2. Script loads `optim_${LATEST_CKPT}.pt` (optimizer state - Adam momentum, etc.)
  3. Training continues from iteration `${LATEST_CKPT} + 1`
  4. Learning rate schedule continues from correct step
  5. All state is preserved - exact continuation!

- [ ] **Verify resumption worked**:
  ```bash
  # Check logs for resumption message
  tail -50 experiments/${EXP_NAME}/logs/training_resumed_*.log | grep -i "resumed\|checkpoint"

  # Should see something like:
  # "Resuming from checkpoint: iteration 15000"
  # "Loaded model weights from: .../model_015000.pt"
  # "Loaded optimizer state from: .../optim_015000.pt"
  ```

### 3.5 Training Timeline & Expectations

**Small Model (20M params) on 100k workflows**:
- **Single GPU (e.g., RTX 4090)**: ~16-20 hours for 50k iterations
- **8×A100**: ~2-3 hours for 50k iterations
- **Checkpoints**: Every 5k iterations = ~2-3 hours apart (single GPU)

**Expected Results** (after 50k iterations):
- Training loss: <2.0 (ideally 1.5-1.8)
- Validation loss: <2.5 (some gap is normal)
- Gradient norm: stable in 0.5-2.0 range
- Generated workflows: Valid JSON structure with recognizable GitHub Actions syntax

**If training doesn't finish in one session**:
- Checkpoints saved every 5k iterations provide resumption points
- Can stop/resume as many times as needed
- Total progress = last checkpoint iteration → target iteration

---

## Phase 4: Evaluation & Iteration

### 4.1 During Training (Every 5k iterations)
- [ ] Check loss curves (should decrease to <2.0 for Small, <1.5 for Medium)
- [ ] Generate sample workflows from latest checkpoint
  ```python
  # TODO: Create generation script
  python scripts/generate_workflow.py \
    --checkpoint experiments/exp001/.../ckpt_50000.pt \
    --num_samples 10
  ```

- [ ] Evaluate generated workflows:
  - [ ] Valid JSON parsing rate (target: >90%)
  - [ ] Schema adherence (has 'on', 'jobs' fields)
  - [ ] Semantic coherence (jobs/steps make sense)

### 4.2 Post-Training Evaluation
- [ ] Calculate validation perplexity (target: <10)
- [ ] Generate 100 workflows and analyze:
  - [ ] JSON validity rate
  - [ ] Schema compliance rate
  - [ ] Average workflow length (tokens/chars)
  - [ ] Diversity (unique workflows vs duplicates)
  - [ ] Manual inspection (10 random samples)

- [ ] Compare to baseline:
  - [ ] TCT vs Tiktoken compression ratio
  - [ ] Training loss convergence speed
  - [ ] Generation quality (qualitative)

### 4.3 Hyperparameter Tuning (If baseline works)
- [ ] **Experiment 002**: Increase model size (Medium → Large)
- [ ] **Experiment 003**: Longer training (100k → 150k iterations)
- [ ] **Experiment 004**: Different learning rate schedule
- [ ] **Experiment 005**: Larger batch size (32 → 64)

---

## Quality Checkpoints

### Before Training
- [ ] TCT tokenizer verified (examples pass)
- [ ] Sample data prepared and validated
- [ ] Smoke test passes (10 iterations, no errors)

### During Training (Every 10k iterations)
- [ ] Loss decreases steadily
- [ ] Gradients are healthy (not vanishing/exploding)
- [ ] Checkpoints save successfully
- [ ] Validation loss computed

### After Training
- [ ] Model generates workflows (decode test samples)
- [ ] Workflows are valid JSON (>90% parse rate)
- [ ] Workflows follow schema (have required fields)
- [ ] Perplexity is reasonable (<10)

---

## Troubleshooting Guide

### Issue: Smoke test fails with NaN loss
**Cause**: Learning rate too high or gradient explosion
**Solution**:
```python
# Try lower learning rate
--learning_rate 1e-4  # instead of 3e-4

# Add gradient clipping
--grad_clip 1.0
```

### Issue: Loss doesn't decrease
**Causes**:
1. Data corruption (wrong window format)
2. Vocab size mismatch (model vs tokenizer)
3. Learning rate too low

**Debug**:
```python
# 1. Check data format
import torch
data = torch.load('~/Desktop/data/prepared-sample/train.pt')
print(data[0])  # Should be [position_token, content_tokens...]

# 2. Check vocab size
from tct_github_workflow import vocab_size
from model_config import get_config
print(f"Tokenizer vocab: {vocab_size()}")
print(f"Model vocab: {get_config('small')['vocab_size']}")
# Must match: 8192

# 3. Try overfitting on 10 samples (should reach loss <0.1)
python -m scripts.tct_train --max_files 10 --max_iters 1000
```

### Issue: OOM (Out of Memory)
**Solutions**:
```bash
# Reduce batch size
--device_batch_size 16  # instead of 32

# Use smaller model
config = get_config("small")  # instead of "medium"

# Enable gradient checkpointing (if implemented)
--gradient_checkpointing
```

### Issue: Generated workflows are gibberish
**Causes**:
1. Model undertrained (need more iterations)
2. Bad hyperparameters
3. Data issue

**Debug**:
- Check if loss actually decreased (<2.0)
- Try larger model (Small → Medium)
- Verify data preparation (re-run prepare_training_data.py)

---

## Success Criteria

### Minimum Viable Model (Baseline)
- ✅ Training completes without errors
- ✅ Loss decreases to <2.0 (Small) or <1.5 (Medium)
- ✅ Generated workflows parse as valid JSON (>90%)
- ✅ Workflows have required fields ('on', 'jobs')

### Production Quality Model (Goal)
- ✅ Validation perplexity <10
- ✅ Generated workflows are semantically coherent
- ✅ Compression better than Tiktoken (fewer tokens)
- ✅ Reproducible (clear documentation, saved configs)

---

## Next Immediate Actions (Priority Order)

1. **Run TCT examples** (5 min) - Verify tokenizer works independently
2. **Prepare sample data** (2 min) - 10 workflows for quick testing
3. **Create `tct_train.py`** (30 min) - Adapt base_train.py for TCT
4. **Smoke test** (5 min) - 10 iterations to catch bugs early
5. **Prepare full data** (5 min) - 100k workflows for real training
6. **Launch training** (7 hours) - Medium model, 100k iterations
7. **Evaluate** (1 hour) - Generate samples, check quality

**Total estimated time to first trained model**: ~8-9 hours (mostly GPU time)

---

---

## Configuration Decision: Context Size & Stride Analysis

**Date**: 2025-11-02
**Analysis**: Analyzed top 1000 largest workflows from 100k dataset

### Key Findings
- **Longest workflow**: `OpenDDS_OpenDDS_build_and_test.json` (170,655 tokens)
- **Average workflow**: ~180 tokens
- **Dataset**: 100,000 workflows

### Configuration Options Evaluated

| Context | Stride | Vocab | Coverage | Memory | Speed | Training Data |
|---------|--------|-------|----------|--------|-------|---------------|
| 512     | 32     | 8,192 | 284%     | 1.0x   | 1.0x  | More windows  |
| 1024    | 32     | 8,192 | 569%     | 2.0x   | 0.25x | Fewer windows |
| 2048    | 32     | 8,192 | 1138%    | 4.0x   | 0.06x | Even fewer    |

### **DECISION: context_size=512, stride=32** ✅

**Rationale**:
1. **Same vocab requirement**: All options require stride=32 to fit in 8,192 vocab
2. **4x faster training**: Attention is O(n²), so 512 is 4x faster than 1024
3. **Sufficient coverage**: 512 tokens = 284% of average workflow (covers most entirely)
4. **More training data**: Long workflows generate more windows = better learning
5. **Better budget efficiency**: Same $15-100 budget → 4x more training iterations

**Configuration Parameters**:
```python
vocab_size = 8192        # TCT base vocabulary (no expansion needed)
context_size = 512       # Window size for training
stride = 32              # Power-of-2 stride for position mapping
max_position = 5,332     # Maximum mapped position (< 8192 ✓)
safety_margin = 2,860    # Unused positions (35% buffer)
```

**Trade-offs accepted**:
- Less context for hierarchical structure (workflows: triggers → jobs → steps)
- May need to scale to 1024 later if quality suffers
- Worth the risk: 4x speedup enables rapid experimentation

---

**Last updated**: 2025-11-02
**Phase**: Configuration finalized, ready for Phase 3 (full training)
