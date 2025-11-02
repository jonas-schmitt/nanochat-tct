# Nanochat-TCT Bundle Manifest

## Bundle Information

- **Version**: 1.0.0
- **TCT Version**: 1.0.5 (Decoder-only windowed generation)
- **Created**: 2025-11-02
- **Purpose**: Integration bundle for nanochat-tct with TCT tokenization, model configs, and adapters
- **Note**: TCT wheel is not bundled - install from CI artifacts or TCT releases

## Contents

### Documentation (`docs/`)

| File | Description | Size |
|------|-------------|------|
| `CLAUDE.md` | AI assistant development guide for nanochat-tct | ~22 KB |
| `TCT.md` | Complete TCT tokenizer API reference | ~25 KB |
| `SETUP.md` | Environment setup and installation guide | ~15 KB |

### Examples (`examples/`)

| File | Description |
|------|-------------|
| `example_quickstart.py` | Basic encode/decode with round-trip verification |
| `example_windowed.py` | Decoder-only windowed generation demonstration |

### Scripts (`scripts/`)

| File | Description |
|------|-------------|
| `yaml_to_json.py` | Convert YAML workflows to JSON format |
| `prepare_training_data.py` | Create training windows from JSON workflows |

### Adapters (`adapters/`)

| File | Description | Size |
|------|-------------|------|
| `model_config.py` | Model configurations (Small/Medium/Large) | ~180 LOC |
| `tct_tokenizer_adapter.py` | TCT tokenizer → nanochat interface | ~150 LOC |
| `tct_dataloader.py` | DataLoader for TCT windowed data | ~200 LOC |
| `README.md` | Adapter documentation and integration guide | ~10 KB |

**Total adapter code**: ~530 lines

### Root Files

| File | Description |
|------|-------------|
| `README.md` | Quick start guide and bundle overview |
| `MANIFEST.md` | This file - bundle contents listing |

## Key Features

### Decoder-Only Windowed Generation

**CRITICAL CHANGE**: TCT now uses **1 position token** (not 2) for left-to-right autoregressive training.

- **Window format**: `[schema_pos, content_tokens...]`
- **Position token**: Indicates absolute position in original sequence (0-based)
- **NO gap token**: Pure decoder-only mode (not fill-in-the-middle)

### Tokenizer Specifications

- **Vocabulary size**: 8,192 tokens (Mode 1 complete encoding)
- **Base vocab**: 1,025 tokens (schema-aware + dictionary)
- **BPE merges**: 7,167 tokens (sequence compression)
- **Position tokens**: 512 (for windowed generation, separate range)
- **Compression**: 9.3% better than Tiktoken (GPT-4o) on average

### Model Configurations

Three pre-configured model sizes optimized for workflow generation:

**Small (50M params)** - $20 budget, 3 hours on 8×A100
- vocab: 8192, context: 512, d_model: 512, layers: 6, heads: 8

**Medium (100M params)** - $50 budget, 7 hours on 8×A100 ⭐ Recommended
- vocab: 8192, context: 512, d_model: 768, layers: 8, heads: 12

**Large (200M params)** - $100 budget, 15 hours on 8×A100
- vocab: 8192, context: 512, d_model: 1024, layers: 12, heads: 16

See `adapters/model_config.py` for complete configurations.

## Usage

### 1. Install TCT Tokenizer

```bash
# Download TCT wheel from CI artifacts or releases
# Linux x86_64, glibc 2.34+, Python 3.12+
pip install tct_github_workflow-1.0.5-cp312-cp312-manylinux_2_34_x86_64.whl
```

### 2. Verify Installation

```bash
python3 -c "from tct_github_workflow import vocab_size; print(f'Vocab: {vocab_size()}')"
# Expected: Vocab: 8192
```

### 3. Run Examples

```bash
cd examples
python3 example_quickstart.py    # Basic tokenization
python3 example_windowed.py      # Windowed generation
```

### 4. Prepare Training Data

```bash
# Convert YAML → JSON (if needed)
python3 scripts/yaml_to_json.py data/workflows/*.yml --output data/json/

# Create training windows
python3 scripts/prepare_training_data.py \
  --input data/json/ \
  --output data/prepared/ \
  --context-size 256
```

## System Requirements

### Minimum (for testing)

- **OS**: Linux x86_64 (Ubuntu 22.04+, Debian 12+)
- **Python**: 3.12+ (CPython only)
- **RAM**: 16GB
- **Storage**: 50GB SSD

### Recommended (for training)

- **OS**: Linux x86_64 (Ubuntu 22.04+, Debian 12+)
- **Python**: 3.12+
- **RAM**: 64GB
- **Storage**: 200GB NVMe SSD
- **GPU**: NVIDIA A100 (40GB) or H100 (80GB)
- **CUDA**: 12.1+ (for PyTorch 2.5+)

## Dependencies

### Core (included in wheel)

- Rust libraries (compiled into wheel)
- PyO3 bindings

### Training (install separately)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas tqdm wandb
```

### Data preparation (install separately)

```bash
pip install pyyaml  # For YAML → JSON conversion
```

## Verification Checklist

Before starting training:

- [ ] TCT tokenizer installed and working
- [ ] PyTorch with CUDA available
- [ ] Round-trip verification passing (see `example_quickstart.py`)
- [ ] Windowed generation working (see `example_windowed.py`)
- [ ] Training data prepared (`prepare_training_data.py`)

## Known Limitations

1. **Platform**: Only Linux x86_64 with glibc 2.34+ (no macOS/Windows support)
2. **Python version**: Requires exactly Python 3.12 (CPython only)
3. **Sequence length**: Max 512 tokens for windowed generation (position token limit)
4. **Schema**: Workflows must have `on` and `jobs` fields (GitHub Actions schema)

## Troubleshooting

### Wheel installation fails

**Error**: `not a supported wheel on this platform`

**Solution**: Verify Python 3.12 on Linux x86_64 with glibc 2.34+

### Round-trip verification fails

**Error**: Decoded workflow doesn't match original

**Solution**: This should NEVER happen - report bug to TCT maintainers

### OOM during training

**Error**: `CUDA out of memory`

**Solutions**:
- Reduce batch size (`--batch-size 16`)
- Enable gradient checkpointing
- Use smaller model (`--n-layer 4 --n-embd 256`)

## References

- **TCT Documentation**: See `docs/TCT.md`
- **Development Guide**: See `docs/CLAUDE.md`
- **Setup Instructions**: See `docs/SETUP.md`
- **Nanochat**: https://github.com/jonas-schmitt/nanochat-tct

---

**Checksum** (SHA-256):
- Bundle: (regenerate after tarball creation)
- Wheel: (included in wheel metadata)

**Contact**: (your contact info)
**License**: (your license, e.g., MIT)

---

*This bundle provides integration code, model configurations, and documentation for training a nanochat-style decoder-only transformer on GitHub Actions workflows using TCT's schema-aware tokenization. Install the TCT tokenizer wheel separately from CI artifacts.*
