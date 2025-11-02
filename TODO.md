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

## Phase 3: Full Training (Scale Up)

### 3.1 Full Data Preparation
- [ ] Prepare complete dataset (100k workflows)
  ```bash
  cd tct-bundle/scripts
  python prepare_training_data.py \
    --input ~/Desktop/data/workflows-100k/json/ \
    --output ~/Desktop/data/prepared/ \
    --context-size 512 \
    --train-split 0.8
  ```
  **Expected time**: ~5 minutes
  **Expected output**:
  - `train.pt` (~2M windows, ~4GB)
  - `val.pt` (~500k windows, ~1GB)
  - `metadata.json`

- [ ] Verify full dataset
  ```bash
  python -c "
  import torch, json
  train = torch.load('~/Desktop/data/prepared/train.pt')
  val = torch.load('~/Desktop/data/prepared/val.pt')
  meta = json.load(open('~/Desktop/data/prepared/metadata.json'))
  print(f'Train: {train.shape}')
  print(f'Val: {val.shape}')
  print(f'Total workflows: {meta[\"total_workflows\"]}')
  "
  ```

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

### 3.3 Launch Full Training
- [ ] Create experiment directory
  ```bash
  mkdir -p experiments/exp001-medium-baseline
  cd experiments/exp001-medium-baseline
  ```

- [ ] Start training with monitoring
  ```bash
  # Single GPU
  python -m scripts.tct_train \
    --max_iters 100000 \
    --device_batch_size 32 \
    --eval_interval 500 \
    --save_interval 5000 \
    --wandb_project nanochat-tct \
    --wandb_run exp001-medium-baseline

  # Or multi-GPU (8×A100)
  torchrun --standalone --nproc_per_node=8 -m scripts.tct_train -- \
    --max_iters 100000 \
    --device_batch_size 32 \
    --eval_interval 500 \
    --save_interval 5000
  ```

- [ ] Monitor training health (check every 30 min for first 2 hours):
  - [ ] Loss decreases steadily (not flat, not NaN)
  - [ ] Gradient norms healthy (0.5 - 2.0 range)
  - [ ] Validation loss tracks training loss
  - [ ] GPU utilization >80%
  - [ ] No OOM errors

- [ ] Save experiment config and notes:
  ```bash
  # Copy config to experiment dir
  cp scripts/tct_train.py experiments/exp001-medium-baseline/config.py

  # Document experiment
  echo "# Experiment 001 - Medium Baseline

  **Model**: Medium (100M params)
  **Data**: 100k workflows, 512 context
  **Training**: 100k iterations, batch=32, lr=3e-4
  **Start**: $(date)
  **Goal**: Establish baseline performance
  " > experiments/exp001-medium-baseline/README.md
  ```

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

**Last updated**: 2025-11-02
**Phase**: Ready to start Phase 1.2 (TCT verification)
