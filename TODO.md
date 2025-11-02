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

## Phase 3: Full Training (Scale Up) - SMALL MODEL ✅ IN PROGRESS

**Configuration**: context_size=1024, stride=32, vocab_size=8192
**Model**: Small (20M params) - Fast experimentation baseline
**Budget**: ~$15, ~2.5 hours on single RTX 4070

### 3.1 Full Data Preparation with Striding ✅ COMPLETED

**IMPORTANT**: Using stride=32 to keep vocab_size=8192, context_size=1024 for better workflow coverage

- [x] **Step 1**: Verified `prepare_training_data.py` supports `--stride` parameter ✅

- [x] **Step 2**: Prepared complete dataset with stride ✅
  ```bash
  cd ~/git/nanochat-tct/tct-bundle/scripts
  python prepare_training_data.py \
    --input ~/Desktop/data/workflows-100k/json/ \
    --output ~/Desktop/data/prepared-100k-1024-s32/ \
    --context-size 1024 \
    --stride 32 \
    --train-split 0.8
  ```

  **Actual results**:
  - `train.pt` (shape: [772186, 1024])
  - `val.pt` (shape: [193047, 1024])
  - `metadata.json` ✅
  - Total windows: 965,233
  - Processing time: ~3 minutes
  - Skipped 81,539 short sequences (<1023 tokens)

- [x] **Step 3**: Verified prepared dataset ✅

  **Validation results**:
  - [x] Train shape: [772186, 1024] ✅
  - [x] Val shape: [193047, 1024] ✅
  - [x] `metadata.json` has `"stride": 32` ✅
  - [x] Max position token: 5,301 < 8192 ✅
  - [x] Context size: 1024 ✅
  - [x] All token IDs within vocab_size=8192 ✅

### 3.2 Choose Model Size & Budget ✅ COMPLETED
- [x] **Decision**: Small model selected

  | Model | Params | Architecture | Cost | Time (RTX 4070) | Use Case |
  |-------|--------|--------------|------|-----------------|----------|
  | **Small** | **20M** | **384d×8L×6H** | $0 | **2.5h** | **Baseline experiment** ⭐ |
  | Medium | 100M | 768d×8L×12H | TBD | ~8h | Production quality 🎯 |
  | Large | 200M | 1024d×12L×16H | TBD | ~15h | Maximum quality 🏆 |

- [x] Updated `scripts/tct_train.py` with Small config
- [x] Updated `tct-bundle/adapters/model_config.py` to context_size=1024

### 3.3 Launch Full Training with Checkpointing ✅ IN PROGRESS

**Goal**: Train Small (20M) model on 100k workflows (50,000 iterations)

**Current Status**: Training running (started 2025-11-02 18:19)
- **Progress**: ~7% complete (step 3,500/50,000)
- **Loss**: 9.01 → 1.44 (84% reduction, excellent progress ✅)
- **Speed**: ~167-199ms/step, ~195k tokens/sec
- **GPU**: RTX 4070 Ti (8GB VRAM, ~70% utilization)
- **ETA**: ~2 hours remaining

- [x] **Step 1**: Verified model config ✅
  ```
  Vocab size: 8,192 ✅
  Context size: 1024 ✅
  d_model: 384 ✅
  n_layers: 8 ✅
  n_heads: 6 ✅
  Total params: 20,447,232
  ```

- [x] **Step 2**: Training script configured with parameters:
  ```python
  model_size = "small"
  data_dir = "~/Desktop/data/prepared-100k-1024-s32"
  num_iterations = 50000
  device_batch_size = 32
  eval_every = 5000  # Validation evaluation
  save_every = 5000  # Checkpoint saving
  ```

- [x] **Step 3**: Launched training ✅
  ```bash
  cd ~/git/nanochat-tct
  python -m scripts.tct_train
  ```

  **Training creates**:
  - Checkpoints every 5,000 iterations → `~/Desktop/checkpoints/tct_small/`
    - `model_005000.pt`, `model_010000.pt`, ..., `model_050000.pt`
    - Includes optimizer state for potential resumption
  - Validation evaluation every 5,000 steps

- [ ] **Step 4**: Monitor training progress

  **Health Checks** (monitored during training):
  - [x] Loss decreases steadily ✅ (9.01 → 1.44 at step 3,500)
  - [x] Gradient norms healthy ✅
  - [ ] Validation loss evaluated at step 5,000
  - [x] GPU utilization ~70% ✅
  - [x] No OOM errors ✅
  - [ ] First checkpoint at step 5,000

  **Monitoring**:
  - Check background process: `BashOutput tool with bash_id: b646b4`
  - Monitor GPU: `nvidia-smi`
  - Check checkpoints: `ls -lth ~/Desktop/checkpoints/tct_small/`

### 3.4 Training Timeline & Progress

**Small Model (20M params) on 100k workflows** (RTX 4070):
- **Total iterations**: 50,000
- **Estimated time**: ~2.5 hours
- **Checkpoint frequency**: Every 5,000 steps (~30 minutes)
- **Validation frequency**: Every 5,000 steps

**Current Progress** (as of last check):
- Step: ~3,500/50,000 (7%)
- Loss: 9.01 → 1.44 (strong learning ✅)
- Speed: ~167-199ms/step, ~195k tokens/sec
- Time remaining: ~2 hours

**Expected Milestones**:
- Step 5,000 (30 min): First checkpoint + validation
- Step 10,000 (1h): Second checkpoint
- Step 25,000 (2h): Mid-training checkpoint
- Step 50,000 (2.5h): Final model

**Expected Results** (after 50k iterations):
- Training loss: <2.0 (ideally 1.0-1.5)
- Validation loss: <2.5
- Perplexity: <10
- Generated workflows: Valid JSON with GitHub Actions schema

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

### **DECISION: context_size=1024, stride=32** ✅

**Rationale**:
1. **Better workflow coverage**: 1024 tokens covers 568% of average workflow length (180 tokens)
2. **Handles complex workflows**: 100% of workflows at 98th percentile fit in 1024 context
3. **Same vocab requirement**: stride=32 keeps position tokens < 8192
4. **Sufficient generation headroom**: 1023 content tokens allows long workflow generation
5. **Acceptable speed**: While slower than 512, we prioritize coverage for large workflows

**Configuration Parameters**:
```python
vocab_size = 8192        # TCT base vocabulary (no expansion needed)
context_size = 1024      # TOTAL window size (1 position + 1023 content tokens)
stride = 32              # Power-of-2 stride for position mapping
max_position = 5,301     # Maximum mapped position (< 8192 ✓)
safety_margin = 2,891    # Unused positions (35% buffer)
```

**Window Format**:
- Each window: `[position_token, content_tok_0, ..., content_tok_1022]`
- Total length: exactly 1024 tokens
- Model's context_size parameter must be 1024

**Important**: context_size includes the position token. During training and generation, the model processes exactly 1024 tokens total.

---

**Last updated**: 2025-11-02
**Phase**: Configuration finalized, ready for Phase 3 (full training)
