# Claude AI Assistant Guide for Nanochat-TCT

Development guide for integrating TCT tokenization into nanochat to train workflow generation models. **Goal**: Achieve excellent ML results through clean integration and disciplined experimentation.

## 🎯 Project Mission

**Train high-quality language models for GitHub Actions workflow generation using:**
- Nanochat's proven training infrastructure (minimal changes)
- TCT's schema-aware tokenization (8,192 vocab, optimized for workflows)
- 100k real-world workflows (~/Desktop/data/workflows-100k/json/)

**Success criteria**: Generate valid, useful GitHub Actions workflows

## 📂 Repository Structure

```
nanochat-tct/
├── CLAUDE.md            # This file - integration & experiment guide
├── tct-bundle/          # TCT integration bundle (tokenizer, adapters, docs)
├── nanochat/            # Core training library (REUSE, minimal changes)
├── scripts/             # Training scripts (adapt base_train.py for TCT)
└── experiments/         # Experiment tracking (configs, results, logs)
```

**Data locations**:
- Workflows: `~/Desktop/data/workflows-100k/json/` (100k JSON files)
- Prepared: `~/Desktop/data/prepared/` (windowed training data)

## 🎯 Core Principles

### 1. Integration Excellence
- **Minimal changes** to nanochat core (tokenizer, dataloader, model config only)
- **Reuse everything**: training loop, optimizer, learning rate schedule, checkpointing
- **Clean boundaries**: Adapt at interfaces, not internals
- **Documentation**: Every change explained, integration steps reproducible

### 2. ML Quality Standards
- **Data correctness**: Verify tokenization, window format, no corruption
- **Training stability**: No NaN losses, loss decreases, gradients healthy
- **Reproducibility**: Clear steps, anyone can replicate results
- **Model evaluation**: Generated workflows are valid and useful

### 3. Experiment Discipline
- **Track experiments**: Save configs, logs, checkpoints with clear naming
- **Document changes**: Hyperparameter choices, model sizes, why they were chosen
- **Measure results**: Loss curves, perplexity, workflow validity, human evaluation
- **Learn systematically**: One change at a time, compare to baseline

## 🔧 Integration Steps

### Phase 1: Setup & Verification (Reversible)

**Step 1: Install TCT tokenizer**
```bash
pip install tct_github_workflow-1.0.5-cp312-cp312-manylinux_2_34_x86_64.whl
python3 -c "from tct_github_workflow import vocab_size; print(vocab_size())"
# Expected: 8192
```

**Step 2: Verify TCT works independently**
```bash
cd ~/git/nanochat-tct/tct-bundle/examples
python3 example_quickstart.py  # Round-trip test
python3 example_windowed.py    # Window format test
# Both should pass ✅
```

**Step 3: Prepare sample data (10 workflows for quick testing)**
```bash
cd ~/git/nanochat-tct/tct-bundle/scripts
python3 prepare_training_data.py \
  --input ~/Desktop/data/workflows-100k/json/ \
  --output ~/Desktop/data/prepared-sample/ \
  --context-size 512 \
  --train-split 0.8 \
  --max-files 10

# Verify output
python3 -c "import torch; print(torch.load('~/Desktop/data/prepared-sample/train.pt').shape)"
# Expected: torch.Size([N, 513]) where N = number of windows
```

### Phase 2: Integration (First Training Run)

**Step 4: Adapt base_train.py for TCT**

Create a new training script `scripts/tct_train.py` (or modify `base_train.py`):

```python
# scripts/tct_train.py
import sys
import os
sys.path.insert(0, "tct-bundle/adapters")

from tct_tokenizer_adapter import TCTTokenizer
from tct_dataloader import tokenizing_distributed_data_loader
from model_config import get_config

# Use TCT tokenizer (drop-in replacement)
tokenizer = TCTTokenizer()

# Use pre-prepared TCT data
train_loader = tokenizing_distributed_data_loader(
    device_batch_size=32,
    context_size=512,
    split="train",
    data_dir=os.path.expanduser("~/Desktop/data/prepared-sample/")
)

# Model config optimized for workflows
config = get_config("small")  # Start with small (50M) for quick testing
model = GPT(
    vocab_size=config["vocab_size"],      # 8192
    context_size=config["context_size"],  # 512
    d_model=config["d_model"],
    n_layers=config["n_layers"],
    n_heads=config["n_heads"],
    dropout=config["dropout"]
)

# Rest of training loop unchanged (reuse nanochat's implementation)
```

**Step 5: Smoke test training (10 iterations)**
```bash
python -m scripts.tct_train --max_iters 10 --device_batch_size 32

# Check:
# - Script runs without errors ✅
# - Loss is computed (not NaN) ✅
# - Gradients flow (check gradient norms) ✅
```

If smoke test passes → integration successful!

### Phase 3: Full Training (Scale Up)

**Step 6: Prepare full dataset (100k workflows)**
```bash
cd ~/git/nanochat-tct/tct-bundle/scripts
python3 prepare_training_data.py \
  --input ~/Desktop/data/workflows-100k/json/ \
  --output ~/Desktop/data/prepared/ \
  --context-size 512 \
  --train-split 0.8

# Takes ~5 minutes, creates ~5GB of data
# Output: train.pt (~2M windows), val.pt (~500k windows)
```

**Step 7: Choose model size & train**

Start with **Medium (100M params)** for $50 budget:
```bash
# Update config in tct_train.py
config = get_config("medium")  # 100M params, 512 context

# Train on 8×A100 (~7 hours)
torchrun --standalone --nproc_per_node=8 -m scripts.tct_train -- \
  --max_iters 100000 \
  --device_batch_size 32 \
  --eval_interval 500 \
  --save_interval 5000
```

**Model size guide**:
- **Small (50M)**: $20, 3h - Quick experiments, proof-of-concept
- **Medium (100M)**: $50, 7h - **Recommended** for production
- **Large (200M)**: $100, 15h - Maximum quality

See `tct-bundle/adapters/model_config.py` for complete configs.

## 📊 Model Configurations

All configs optimized for workflows (not chat):
- **Vocab**: 8,192 (not 50k+ BPE)
- **Context**: 512 (not 2048 - workflows average 180 tokens)
- **Layers**: 6-12 (not 20+ - structured data is easier than natural language)

```python
# Small (50M params)
vocab=8192, context=512, d_model=512, layers=6, heads=8

# Medium (100M params) ⭐ Recommended
vocab=8192, context=512, d_model=768, layers=8, heads=12

# Large (200M params)
vocab=8192, context=512, d_model=1024, layers=12, heads=16
```

## ✅ Quality Checkpoints

### Before Training
- [ ] TCT tokenizer installed and verified (example_quickstart.py passes)
- [ ] Sample data prepared correctly (shape: [N, 513])
- [ ] Smoke test passes (10 iterations, no errors, loss computed)

### During Training
- [ ] **Loss decreases** steadily (not flat, not NaN)
- [ ] **Gradients healthy** (not vanishing, not exploding - check grad norms)
- [ ] **Checkpoints saved** every 5k iterations
- [ ] **Validation loss** computed regularly (eval_interval)

### After Training
- [ ] **Model generates workflows** (decode test samples)
- [ ] **Workflows are valid JSON** (parse without errors)
- [ ] **Workflows follow schema** (have 'on', 'jobs', etc.)
- [ ] **Perplexity reasonable** (lower than baseline/random)

## 🔍 Debugging Common Issues

### Training crashes immediately
**Check**: Vocab size mismatch
```python
# Ensure model uses TCT vocab (8192)
model = GPT(vocab_size=8192, ...)  # NOT 50k+
```

### Loss is NaN
**Check**: Learning rate too high, or gradient explosion
```bash
# Try lower LR or gradient clipping
--learning_rate 1e-4  # instead of 3e-4
--grad_clip 1.0
```

### OOM (out of memory)
**Check**: Batch size too large
```bash
# Reduce batch size
--device_batch_size 16  # instead of 32
```

### Loss doesn't decrease
**Check**: Data corruption or wrong window format
```python
# Verify window format
import torch
windows = torch.load('~/Desktop/data/prepared/train.pt')
print(windows[0])  # Should be [position_token, content_tokens...]
```

### Generated workflows are gibberish
**Check**: Model undertrained or bad hyperparameters
- Train longer (more iterations)
- Try larger model (Medium instead of Small)
- Check if loss actually decreased

## 📝 Experiment Tracking

Create `experiments/` directory to track runs:

```bash
experiments/
├── exp001-small-baseline/
│   ├── config.json          # Model config, hyperparameters
│   ├── training.log         # Full training log
│   ├── checkpoints/         # Model checkpoints
│   └── results.md           # Loss curves, eval results, notes
├── exp002-medium-extended/
└── ...
```

**Document each experiment**:
- What changed from previous run?
- Why this hyperparameter choice?
- Training loss curve
- Validation perplexity
- Generated workflow samples
- What worked, what didn't?

## 🎯 Success Metrics

### Quantitative
- **Training loss**: Should decrease to <2.0 for Small, <1.5 for Medium
- **Validation perplexity**: <10 indicates reasonable model
- **Workflow validity**: >90% of generated workflows parse as valid JSON

### Qualitative
- **Schema adherence**: Workflows have required fields (on, jobs, etc.)
- **Semantic coherence**: Jobs/steps make sense together
- **Usefulness**: Workflows could plausibly be used in real repos

## 🔗 Essential References

- **TCT tokenizer API**: `tct-bundle/docs/TCT.md`
- **Adapter code**: `tct-bundle/adapters/` (tokenizer, dataloader, configs)
- **Setup guide**: `tct-bundle/docs/SETUP.md`
- **Nanochat docs**: `README.md` (understand the base system)

## 🚨 When Things Go Wrong

If integration fails or training doesn't work:

1. **Check the checklist**: Did you skip a verification step?
2. **Start smaller**: Use sample data (10 files), not full dataset
3. **Isolate the issue**: Test TCT independently, test nanochat independently
4. **Compare to baseline**: Does vanilla nanochat work? Then issue is in integration
5. **Ask for help**: Describe what you tried, what failed, logs/errors

### Rollback Points
- After Step 3: Delete sample data, uninstall TCT wheel
- After Step 5: Revert changes to training script
- After Step 6: Delete full prepared dataset

## 💬 Working with Claude Assistants

**Announce intentions clearly**:
- "I'm going to prepare sample data with 10 workflows for testing"
- "I'm going to modify base_train.py to use TCT adapters"
- "I'm going to run a smoke test with 10 training iterations"

**Wait for confirmation at major steps**:
- Before modifying nanochat code
- Before preparing full dataset (takes time + disk space)
- Before starting long training runs

**Provide context**:
- "Smoke test failed with NaN loss - gradient explosion likely"
- "Training works but loss isn't decreasing - might need longer warmup"

## 🎓 Development Philosophy

**This is ML engineering, not software engineering**:
- Experiments may fail - that's normal and valuable
- Iterative refinement over perfect-first-time
- Empirical validation (does it train?) over theoretical guarantees
- Clean integration over extensive testing

**But maintain excellence**:
- Reproducible (clear instructions, tracked experiments)
- Debuggable (logs, checkpoints, documented changes)
- Measurable (metrics, evaluations, comparisons)
- Reversible (can rollback, try alternatives)

---

**Version**: 1.0.0 | **TCT**: 1.0.5 | **Updated**: 2025-11-02

*Integrate cleanly, experiment systematically, achieve excellent workflow generation.*
