# Nanochat-TCT: TCT Tokenization Bundle

Integrate TCT's schema-aware tokenization with nanochat's model architecture for training on GitHub Actions workflows.

**Purpose**: This bundle provides TCT tokenization tools for use with nanochat-tct or custom training frameworks.

**What's included**: Tokenizer, data preparation, integration guide
**What's NOT included**: Model architecture, training loops (use nanochat-tct for that)

## Quick Start

### 1. Install TCT Tokenizer

```bash
# Install from CI-generated wheel (Linux x86_64, Python 3.12+)
# Download the latest wheel from TCT releases or CI artifacts
pip install tct_github_workflow-1.0.5-cp312-cp312-manylinux_2_34_x86_64.whl

# Verify installation
python3 -c "from tct_github_workflow import encode, decode, vocab_size; print(f'Vocab: {vocab_size()}')"
# Output: Vocab: 8192
```

### 2. Test Tokenization

```bash
# Run quick start example
cd examples
python3 example_quickstart.py

# Expected output:
# Encoded: 42 tokens
# Decoded: {"name": "CI", "on": "push", ...}
# Round-trip: ✅ SUCCESS
```

### 3. Prepare Training Data

```bash
# Note: For nanochat-tct, workflows are already in ~/Desktop/data/workflows-100k/json/

# Prepare training windows
python3 scripts/prepare_training_data.py \
  --input ~/Desktop/data/workflows-100k/json/ \
  --output ~/Desktop/data/prepared/ \
  --context-size 512
```

### 4. Start Training

```bash
# Install training dependencies
pip install torch numpy tqdm wandb

# Run training
python3 train.py \
  --data-dir data/prepared/ \
  --checkpoint-dir checkpoints/ \
  --max-iters 100000 \
  --batch-size 32
```

## Bundle Contents

```
tct-bundle/
├── README.md                       # This file
├── docs/
│   ├── TCT.md                      # Complete TCT tokenizer reference
│   └── SETUP.md                    # Environment setup
├── adapters/                       # Nanochat integration code
│   ├── README.md                   # Adapter documentation
│   ├── model_config.py             # Model configurations (Small/Medium/Large)
│   ├── tct_tokenizer_adapter.py    # TCT → nanochat tokenizer interface
│   └── tct_dataloader.py           # DataLoader for TCT windows
├── examples/
│   ├── example_quickstart.py       # Basic encode/decode
│   └── example_windowed.py         # Decoder-only windowed generation
└── scripts/
    ├── yaml_to_json.py             # Convert YAML → JSON
    └── prepare_training_data.py    # Create training windows
```

## Key Features

- **Decoder-only windowed generation**: 1 position token (NOT fill-in-the-middle)
- **8,192 token vocabulary**: 96% smaller than Tiktoken (GPT-4o)
- **9.3% better compression**: Fewer tokens for same workflows
- **Perfect round-trip**: Lossless encoding/decoding
- **Workflow-optimized models**: 3 configurations (50M, 100M, 200M params)
- **Nanochat integration**: Drop-in adapters for nanochat's GPT architecture

## Model Configurations

**Small (50M params)** - ~$20 budget
- vocab: 8192, context: 512, d_model: 512, layers: 6
- Use: Experiments, proof-of-concept
- Training: ~3 hours on 8×A100

**Medium (100M params)** - ~$50 budget ⭐ Recommended
- vocab: 8192, context: 512, d_model: 768, layers: 8
- Use: Production workflow generation
- Training: ~7 hours on 8×A100

**Large (200M params)** - ~$100 budget
- vocab: 8192, context: 512, d_model: 1024, layers: 12
- Use: Maximum quality, research
- Training: ~15 hours on 8×A100

See `adapters/model_config.py` for complete configurations.

## Documentation

- **[TCT.md](docs/TCT.md)** - Complete tokenizer API reference
- **[CLAUDE.md](docs/CLAUDE.md)** - Development guidelines for AI assistants
- **[SETUP.md](docs/SETUP.md)** - Environment setup and dependencies

## Requirements

- Python 3.12+ (CPython only)
- Linux x86_64 with glibc 2.34+ (Ubuntu 22.04+, Debian 12+)
- PyTorch 2.0+ (for training)
- 16GB+ RAM (for data preparation)
- GPU with 24GB+ VRAM (for training, A100/H100 recommended)

## Training Budget

**Target**: $100 total training cost

**Configuration**:
- GPU: A100 (40GB) at $1.50/hour
- Training time: ~60 hours
- Total iterations: 100,000
- Effective batch size: 128 workflows

## Quick Reference

### Tokenization

```python
from tct_github_workflow import encode, decode, vocab_size

# Encode workflow
tokens = encode(json_str)  # list[int]

# Decode tokens
json_str, consumed, total = decode(tokens)

# Vocabulary size
size = vocab_size()  # 8192
```

### Windowed Generation

```python
from tct_github_workflow import extract_window

# Extract decoder-only window (1 position token)
window = extract_window(tokens, start=0, end=256)
# window = [schema_pos, tok_0, tok_1, ..., tok_255]
```

## Troubleshooting

### Import Error

```
ImportError: cannot import name 'encode' from 'tct_github_workflow'
```

**Solution**: Reinstall wheel
```bash
pip install --force-reinstall tct_github_workflow-1.0.5-cp312-cp312-manylinux_2_34_x86_64.whl
```

### Platform Incompatibility

```
ERROR: ... is not a supported wheel on this platform
```

**Solution**: TCT wheel only supports Linux x86_64 with glibc 2.34+ (Ubuntu 22.04+). Not available for macOS/Windows.

### YAML Conversion

```
yaml.scanner.ScannerError: ...
```

**Solution**: Use provided conversion script
```bash
python3 scripts/yaml_to_json.py workflow.yml > workflow.json
```

## References

- **Nanochat**: https://github.com/jonas-schmitt/nanochat-tct
- **TCT Project**: https://github.com/yourusername/tct
- **GitHub Actions Schema**: https://github.com/SchemaStore/schemastore

---

**Version**: 1.0.0
**Last updated**: 2025-11-02
**TCT version**: 1.0.5 (Decoder-only windowed generation)
