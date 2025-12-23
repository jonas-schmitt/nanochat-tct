# TCT Training Data Report

Generated: 2025-12-23

## Overview

This report documents the training data prepared for TCT (Type-Constrained Tokenization) experiments comparing schema-aware TCT-BPE tokenization against standard UTF8-BPE tokenization.

## Recommended Context Lengths

| Schema | Context Length | Coverage | Rationale |
|--------|---------------|----------|-----------|
| **tsconfig** | 256 | 99.3%+ | P99=148-159, most configs are simple |
| **eslintrc** | 512 | 99.6%+ | P99=341-345, more complex configs |
| **kubernetes** | 2048 | 99%+ | P99=2622-2653, complex manifests |

## Dataset Statistics

### eslintrc (ESLint Configuration)

| Metric | TCT-BPE-10k | UTF8-BPE-10k |
|--------|-------------|--------------|
| **Vocabulary Size** | 9,999 | 18,336 |
| **Total Tokens** | 4,477,343 | 4,476,205 |
| **Avg Tokens/Sequence** | 35.19 | 35.18 |
| **P50 (Median)** | 15 | 15 |
| **P90** | 74 | 74 |
| **P95** | 137 | 127 |
| **P99** | 345 | 341 |
| **Max Length** | 4,084 | 4,260 |
| **Train Files** | 114,500 | 114,500 |
| **Validation Files** | 12,722 | 12,722 |
| **Total Files** | 127,222 | 127,222 |

### tsconfig (TypeScript Configuration)

| Metric | TCT-BPE-10k | UTF8-BPE-10k |
|--------|-------------|--------------|
| **Vocabulary Size** | 9,999 | 16,196 |
| **Total Tokens** | 8,806,180 | 8,804,774 |
| **Avg Tokens/Sequence** | 24.75 | 24.75 |
| **P50 (Median)** | 11 | 12 |
| **P90** | 31 | 33 |
| **P95** | 47 | 47 |
| **P99** | 159 | 148 |
| **Max Length** | 222,334 | 226,903 |
| **Train Files** | 320,202 | 320,202 |
| **Validation Files** | 35,578 | 35,578 |
| **Total Files** | 355,780 | 355,780 |

### kubernetes (Kubernetes Manifests)

| Metric | TCT-BPE-20k | UTF8-BPE-matched |
|--------|-------------|------------------|
| **Vocabulary Size** | 19,999 | 23,886 |
| **Total Tokens** | 46,593,101 | 46,590,132 |
| **Avg Tokens/Sequence** | 189.07 | 189.05 |
| **P50 (Median)** | 37 | 37 |
| **P90** | 205 | 201 |
| **P95** | 418 | 411 |
| **P99** | 2,622 | 2,653 |
| **Max Length** | 338,837 | 332,663 |
| **Train Files** | 221,795 | 221,795 |
| **Validation Files** | 24,643 | 24,643 |
| **Total Files** | 246,438 | 246,438 |

## Data Locations

### Encoded Training Data

All encoded sequences are stored in JSONL format with 90/10 train/validation split.

```
/home/josch/Desktop/data/
├── eslintrc-tct-bpe-10k/
│   ├── train.jsonl       # 114,500 sequences
│   ├── validate.jsonl    # 12,722 sequences
│   ├── metadata.json
│   └── stats.json
├── eslintrc-utf8-bpe-10k/
│   ├── train.jsonl       # 114,500 sequences
│   ├── validate.jsonl    # 12,722 sequences
│   ├── metadata.json
│   └── stats.json
├── tsconfig-tct-bpe-10k/
│   ├── train.jsonl       # 320,202 sequences
│   ├── validate.jsonl    # 35,578 sequences
│   ├── metadata.json
│   └── stats.json
├── tsconfig-utf8-bpe-10k/
│   ├── train.jsonl       # 320,202 sequences
│   ├── validate.jsonl    # 35,578 sequences
│   ├── metadata.json
│   └── stats.json
├── kubernetes-tct-bpe/
│   ├── train.jsonl       # 221,795 sequences
│   ├── validate.jsonl    # 24,643 sequences
│   ├── metadata.json
│   └── stats.json
└── kubernetes-utf8-bpe/
    ├── train.jsonl       # 221,795 sequences
    ├── validate.jsonl    # 24,643 sequences
    ├── metadata.json
    └── stats.json
```

### BPE Merge Tables

```
/home/josch/git/tct/schemas/bpe/
├── eslintrc-tct-bpe-10k.json      # 9,742 merges, vocab 10,000
├── eslintrc-utf8-bpe-10k.json     # 18,080 merges, vocab 18,336
├── tsconfig-tct-bpe-10k.json      # 9,742 merges, vocab 10,000
├── tsconfig-utf8-bpe-10k.json     # 15,940 merges, vocab 16,196
├── kubernetes-tct-bpe-20k.json    # 19,742 merges, vocab 20,000
└── kubernetes-utf8-bpe-matched.json # 23,630 merges, vocab 23,886
```

## Key Findings

### Vocabulary Efficiency

TCT-BPE achieves the same average sequence length with significantly smaller vocabularies:

| Schema | TCT-BPE Vocab | UTF8-BPE Vocab | Vocab Reduction |
|--------|---------------|----------------|-----------------|
| eslintrc | 10,000 | 18,336 | **45% smaller** |
| tsconfig | 10,000 | 16,196 | **38% smaller** |
| kubernetes | 20,000 | 23,886 | **16% smaller** |

### Compression Parity

Both tokenizers achieve nearly identical compression rates (avg tokens/sequence):
- eslintrc: 35.19 vs 35.18 (0.03% difference)
- tsconfig: 24.75 vs 24.75 (0.00% difference)
- kubernetes: 189.07 vs 189.05 (0.01% difference)

This validates the "matched average" training approach where UTF8-BPE is trained until its average sequence length matches TCT-BPE.

## JSONL Format

Each line in the JSONL files contains a JSON array of token IDs:

```json
[1, 45, 123, 456, 789, 234, 567, 890, 2]
```

Token ID 0 is reserved as the padding token for batch processing.

## Python Tokenizer Packages

### Available Packages

Located in `tct-wheels/`:

| Package | Type | Vocab Size | Features |
|---------|------|------------|----------|
| `tct_eslintrc_10k` | TCT-BPE (schema-aware) | 10,000 | encode, decode, decode_prefix |
| `tct_tsconfig_10k` | TCT-BPE (schema-aware) | 10,000 | encode, decode, decode_prefix |
| `tct_kubernetes_20k` | TCT-BPE (schema-aware) | 20,000 | encode, decode, decode_prefix |
| `utf8_bpe` | UTF8-BPE (generic) | runtime | Utf8BpeTokenizer class |

### UTF8-BPE Merge Tables

Located in `bpe-merges/`:

| File | Vocab Size | For Schema |
|------|------------|------------|
| `eslintrc-utf8-bpe-10k.json` | 18,336 | eslintrc |
| `tsconfig-utf8-bpe-10k.json` | 16,196 | tsconfig |
| `kubernetes-utf8-bpe-matched.json` | 23,886 | kubernetes |

### TCT-BPE Package Usage (Schema-Aware)

```python
import tct_kubernetes_20k

# Encode JSON to tokens
json_text = '{"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "nginx"}}'
tokens = tct_kubernetes_20k.encode(json_text)
print(f"Encoded: {len(tokens)} tokens")

# Decode tokens back to JSON (returns tuple: json_str, consumed, total)
json_str, consumed, total = tct_kubernetes_20k.decode(tokens)
print(f"Decoded: {json_str}")

# Streaming decode with partial tokens (for LLM autocompletion)
partial_tokens = tokens[:len(tokens)//2]
partial_json, fields_decoded, complete = tct_kubernetes_20k.decode_prefix(partial_tokens)
print(f"Partial ({len(partial_tokens)} tokens): {partial_json}")
print(f"Fields decoded: {fields_decoded}, Complete: {complete}")
```

### UTF8-BPE Package Usage (Generic)

```python
from utf8_bpe import Utf8BpeTokenizer

# Load tokenizer with merge table
tokenizer = Utf8BpeTokenizer("bpe-merges/kubernetes-utf8-bpe-matched.json")
print(f"Vocab size: {tokenizer.vocab_size()}")

# Encode string to tokens
text = '{"apiVersion": "v1", "kind": "Pod"}'
tokens = tokenizer.encode(text)
print(f"Encoded: {len(tokens)} tokens")

# Decode tokens back to string
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

# Streaming decode (handles partial UTF-8 sequences)
partial_tokens = tokens[:len(tokens)//2]
partial_text, bytes_buffered = tokenizer.decode_streaming(partial_tokens)
print(f"Partial: {partial_text}")
remaining = tokenizer.flush()  # Get any buffered bytes
```

## Installation

```bash
# Install TCT-BPE packages
pip install tct-wheels/tct_eslintrc_10k-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl
pip install tct-wheels/tct_tsconfig_10k-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl
pip install tct-wheels/tct_kubernetes_20k-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl

# Install UTF8-BPE package
pip install tct-wheels/utf8_bpe-0.1.0-cp312-cp312-manylinux_2_34_x86_64.whl
```
