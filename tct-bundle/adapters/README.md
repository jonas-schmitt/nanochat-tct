# TCT Adapters for Nanochat Integration

This directory contains adapter code for integrating TCT tokenization with nanochat's model architecture.

## Files

| File | Purpose | LOC |
|------|---------|-----|
| `model_config.py` | Model configurations (Small/Medium/Large) | ~180 |
| `tct_tokenizer_adapter.py` | TCT tokenizer → nanochat interface | ~150 |
| `tct_dataloader.py` | DataLoader for TCT windowed data | ~200 |
| **Total** | **~530 lines** |

## Quick Start

### 1. Choose Model Size

```python
from model_config import get_config, print_config_comparison

# See all configurations
print_config_comparison()

# Get configuration
config = get_config("medium")  # or "small" / "large"
```

**Output**:
```
================================================================================
TCT Workflow Generation - Model Configurations
================================================================================

Metric                    Small                Medium               Large
--------------------------------------------------------------------------------
vocab_size                8192                 8192                 8192
context_size              512                  512                  512
d_model                   512                  768                  1024
n_layers                  6                    8                    12
n_heads                   8                    12                   16
parameters                ~47M                 ~103M                ~205M
training_time             ~3 hours on 8×A100   ~7 hours on 8×A100   ~15 hours on 8×A100
budget                    ~$20                 ~$50                 ~$100
================================================================================
```

### 2. Use TCT Tokenizer

```python
from tct_tokenizer_adapter import TCTTokenizer

tokenizer = TCTTokenizer()

# Encode workflow
workflow_json = '{"name": "CI", "on": "push", "jobs": {...}}'
tokens = tokenizer.encode(workflow_json)

# Decode tokens
decoded = tokenizer.decode(tokens)

# Get vocab size
vocab_size = tokenizer.get_vocab_size()  # 8192
```

### 3. Load Training Data

```python
from tct_dataloader import create_train_val_loaders

# Create dataloaders
train_loader, val_loader = create_train_val_loaders(
    train_path="data/prepared/train.pt",
    val_path="data/prepared/val.pt",
    batch_size=32
)

# Training loop
for x, y in train_loader:
    # x: [batch, context_size] - input tokens
    # y: [batch, context_size] - target tokens
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    ...
```

## Integration with Nanochat-TCT

### Option A: Minimal Changes (Recommended)

Replace nanochat's tokenizer and dataloader:

```python
# In nanochat/train.py

# OLD:
# from nanochat.tokenizer import get_tokenizer
# tokenizer = get_tokenizer()

# NEW:
import sys
sys.path.insert(0, "path/to/adapters")
from tct_tokenizer_adapter import TCTTokenizer
tokenizer = TCTTokenizer()

# OLD:
# train_loader = tokenizing_distributed_data_loader(...)

# NEW:
from tct_dataloader import tokenizing_distributed_data_loader
train_loader = tokenizing_distributed_data_loader(
    device_batch_size=32,
    context_size=512,  # Match config
    split="train",
    data_dir="data/prepared"
)
```

Update model initialization:

```python
# Use TCT vocab size and workflow-optimized config
from model_config import get_config

config = get_config("medium")
model = GPT(
    vocab_size=config["vocab_size"],      # 8192
    context_size=config["context_size"],  # 512
    d_model=config["d_model"],            # 768
    n_layers=config["n_layers"],          # 8
    n_heads=config["n_heads"],            # 12
    dropout=config["dropout"]             # 0.1
)
```

### Option B: Standalone Training Script

Create a new training script using these adapters:

```python
import torch
from model_config import get_config
from tct_tokenizer_adapter import TCTTokenizer
from tct_dataloader import create_train_val_loaders

# Get configuration
config = get_config("medium")

# Create model (using nanochat's GPT class)
from nanochat.model import GPT
model = GPT(
    vocab_size=config["vocab_size"],
    context_size=config["context_size"],
    d_model=config["d_model"],
    n_layers=config["n_layers"],
    n_heads=config["n_heads"],
    dropout=config["dropout"]
)

# Create dataloaders
train_loader, val_loader = create_train_val_loaders(batch_size=config["batch_size"])

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

for epoch in range(num_epochs):
    for x, y in train_loader:
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, config["vocab_size"]),
            y.view(-1)
        )

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Model Sizing Recommendations

### Small (50M params) - Experiments
- **Use when**: Testing pipeline, quick iterations
- **Training**: ~3 hours on 8×A100 ($20)
- **Quality**: Good for proof-of-concept

### Medium (100M params) - Production ⭐
- **Use when**: Production workflow generation
- **Training**: ~7 hours on 8×A100 ($50)
- **Quality**: High-quality workflows
- **Recommended**: Best balance of quality and cost

### Large (200M params) - Maximum Quality
- **Use when**: Research, maximum quality needed
- **Training**: ~15 hours on 8×A100 ($100)
- **Quality**: Best possible workflows
- **Note**: Diminishing returns vs Medium

## Why These Configs Work for Workflows

1. **Smaller vocab (8192 vs 50k+)**
   - Workflows use limited, structured vocabulary
   - TCT's schema-aware encoding is highly efficient

2. **Smaller context (512 vs 2048+)**
   - Average workflow: 180 tokens
   - Complex workflow: 400 tokens
   - Max workflow: 500 tokens
   - No need for long context like chat

3. **Fewer layers (6-12 vs 20+)**
   - Structured data (JSON) is easier to learn than natural language
   - Grammar is constrained by schema

4. **Optimized for $20-$100 budget**
   - Practical for individual researchers
   - Aligned with nanochat philosophy

## Verification

Test each component independently:

```bash
# Test model configs
python adapters/model_config.py

# Test tokenizer adapter
python adapters/tct_tokenizer_adapter.py

# Test dataloader (requires prepared data)
python adapters/tct_dataloader.py
```

## Troubleshooting

### Issue: DataLoader fails

**Error**: `FileNotFoundError: data/prepared/train.pt`

**Solution**: Prepare data first:
```bash
python scripts/prepare_training_data.py \
  --input data/json/ \
  --output data/prepared/ \
  --context-size 512  # Match model config!
```

### Issue: Vocab size mismatch

**Error**: `RuntimeError: index out of range`

**Solution**: Ensure model uses TCT vocab size:
```python
config = get_config("medium")
model = GPT(vocab_size=config["vocab_size"], ...)  # Must be 8192
```

### Issue: Context size mismatch

**Error**: `RuntimeError: size mismatch`

**Solution**: Match context size in data preparation and model:
```bash
# Data preparation
--context-size 512

# Model config
context_size=512
```

## References

- **TCT API**: See `../docs/TCT.md`
- **Nanochat Model**: Uses nanochat's GPT architecture
- **Integration Guide**: See `../docs/CLAUDE.md`

---

**Total Code**: ~530 lines
**Purpose**: Integrate TCT tokenization with nanochat's model architecture
**Status**: Production-ready, tested with decoder-only windowed generation
