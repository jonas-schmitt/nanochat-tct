# TCT Training Data Report

Generated: 2025-12-25

## Overview

This report documents the training data prepared for TCT (Type-Constrained Tokenization) experiments comparing schema-aware TCT tokenization against standard UTF8-BPE tokenization.

**Key Design Principle**: Each TCT/UTF8 pair is matched by **average sequence length** to ensure fair comparison of model quality at equal computational cost.

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Context Length | 2048 (all schemas) |
| Train/Validate Split | 90% / 10% |
| Model Size | Small (~50M params) |

## Recommended Datasets

### ESLINTRC (BPE-500)

Lower BPE compression to maximize training tokens given limited data (127k instances).

| Dataset | Vocab | Avg Len | P99 | Train Tokens | Train Seqs |
|---------|-------|---------|-----|--------------|------------|
| **eslintrc-tct-bpe-500** | 499 | 188 | 1,858 | 21.5M | 114,499 |
| **eslintrc-utf8-bpe-500** | 726 | 188 | 1,749 | 21.6M | 114,499 |

- Context coverage: P99 (1,858) fits within 2048 context
- Data-limited: ~47 epochs needed for Chinchilla-optimal 50M model
- Vocab efficiency: TCT uses **31% smaller** vocabulary for same compression

### TSCONFIG (Base Encoding)

Minimal/no BPE - uses base structural tokens only. Most data available (356k instances).

| Dataset | Vocab | Avg Len | P99 | Train Tokens | Train Seqs |
|---------|-------|---------|-----|--------------|------------|
| **tsconfig-tct-base** | 257 | 327 | 1,108 | 117M | 320,202 |
| **tsconfig-utf8-base-matched** | 276 | 325 | 1,115 | 103M | 320,202 |

- Context coverage: P99 (1,115) easily fits within 2048 context
- Data-sufficient: ~9 epochs for Chinchilla-optimal 50M model
- Best schema for main experiment due to data abundance

### KUBERNETES (BPE-20k)

High BPE compression required due to complex manifests with long sequences.

| Dataset | Vocab | Avg Len | P99 | Train Tokens | Train Seqs |
|---------|-------|---------|-----|--------------|------------|
| **kubernetes-tct-bpe** | 19,999 | 189 | 2,622 | 42M | 221,794 |
| **kubernetes-utf8-bpe** | 23,886 | 189 | 2,653 | 42M | 221,794 |

- Context coverage: P99 (~2,650) exceeds 2048; ~1% of sequences truncated
- Data-moderate: ~24 epochs for Chinchilla-optimal 50M model
- Vocab efficiency: TCT uses **16% smaller** vocabulary for same compression

## Data Locations

### Encoded Training Data

All encoded sequences stored in JSONL format:

```
~/Desktop/data/
├── eslintrc-tct-bpe-500/
│   ├── train.jsonl       # 114,499 sequences
│   ├── validate.jsonl    # 12,722 sequences
│   ├── metadata.json
│   └── stats.json
├── eslintrc-utf8-bpe-500/
│   ├── train.jsonl       # 114,499 sequences
│   ├── validate.jsonl    # 12,722 sequences
│   ├── metadata.json
│   └── stats.json
├── tsconfig-tct-base/
│   ├── train.jsonl       # 320,202 sequences
│   ├── validate.jsonl    # 35,578 sequences
│   ├── metadata.json
│   └── stats.json
├── tsconfig-utf8-base-matched/
│   ├── train.jsonl       # 320,202 sequences
│   ├── validate.jsonl    # 35,578 sequences
│   ├── metadata.json
│   └── stats.json
├── kubernetes-tct-bpe/
│   ├── train.jsonl       # 221,794 sequences
│   ├── validate.jsonl    # 24,644 sequences
│   ├── metadata.json
│   └── stats.json
└── kubernetes-utf8-bpe/
    ├── train.jsonl       # 221,794 sequences
    ├── validate.jsonl    # 24,644 sequences
    ├── metadata.json
    └── stats.json
```

### BPE Merge Tables

```
~/git/tct/schemas/bpe/
├── eslintrc-tct-bpe-500.json       # 242 merges, vocab 499
├── eslintrc-utf8-bpe-500.json      # 470 merges, vocab 726
├── tsconfig-utf8-base-matched.json # 20 merges, vocab 276
├── kubernetes-tct-bpe-20k.json     # 19,742 merges, vocab 19,999
└── kubernetes-utf8-bpe.json         # 23,630 merges, vocab 23,886
```

Note: tsconfig-tct-base uses no BPE (base vocab 257 only).

## Training Priority

| Priority | Schema | Rationale |
|----------|--------|-----------|
| 1 | **tsconfig** | Most data (117M tokens), sufficient for proper training |
| 2 | kubernetes | Moderate data (42M tokens), tests complex schema handling |
| 3 | eslintrc | Limited data (21M tokens), risk of overfitting |

## Chinchilla Scaling Reference

For optimal training, tokens ≈ 20× parameters:

| Model Size | Optimal Tokens | eslintrc epochs | tsconfig epochs | k8s epochs |
|------------|----------------|-----------------|-----------------|------------|
| 50M | 1.0B | 47 | 9 | 24 |
| 125M | 2.5B | 116 | 24 | 60 |

## JSONL Format

Each line contains a JSON array of token IDs:

```json
[1, 45, 123, 456, 789, 234, 567, 890, 2]
```

Token ID 0 is reserved as padding token.

## Key Findings

### Vocabulary Efficiency

TCT achieves same average sequence length with smaller vocabularies:

| Schema | TCT Vocab | UTF8 Vocab | Reduction |
|--------|-----------|------------|-----------|
| eslintrc-500 | 499 | 726 | **31% smaller** |
| tsconfig-base | 257 | 276 | **7% smaller** |
| kubernetes-20k | 19,999 | 23,886 | **16% smaller** |

### Compression Parity

Both tokenizers achieve nearly identical average sequence lengths (by design):
- eslintrc: 188 vs 188 tokens
- tsconfig: 327 vs 325 tokens
- kubernetes: 189 vs 189 tokens

This enables fair comparison of model quality at equal sequence length.
