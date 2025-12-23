# Experiment Plan: Type-Constrained Tokenization

## Goal

Provide sufficient empirical evidence to support the claim that **Type-Constrained Tokenization (TCT) enables syntax-free learning** — i.e., by encoding syntax in the tokenization, models can focus entirely on learning semantics, resulting in faster convergence and better performance.

---

## Core Hypothesis

> When syntax is guaranteed by the tokenization scheme, models learn faster and better because their capacity is dedicated to learning semantics rather than syntactic structure.

---

## Research Questions

We evaluate TCT on three JSON Schema domains to answer:

1. **RQ1**: Does TCT achieve better perplexity and faster convergence than BPE at equal compression?
2. **RQ2**: Does TCT's advantage persist when BPE is augmented with constrained decoding?
3. **RQ3**: How does the advantage scale with model size and schema complexity?

---

## Experimental Design

### Schemas (3 domains)

| Schema | Domain | Complexity | Data Source | Dataset Size |
|--------|--------|------------|-------------|--------------|
| **Kubernetes manifests** | DevOps/Infrastructure | High (nested, discriminated unions) | [substratusai/the-stack-yaml-k8s](https://huggingface.co/datasets/substratusai/the-stack-yaml-k8s) | 246K samples |
| **ESLint configs** | JavaScript tooling | Medium | The Stack (JSON + YAML + JS subsets) | 127K samples |
| **TSConfig** | TypeScript tooling | Medium | The Stack (JSON subset) | 356K samples |

**Rationale**: Three distinct domains demonstrate generality. All use JSON Schema, enabling apples-to-apples comparison.

**Data pipeline**:
1. Extract from The Stack using `scripts/extract_configs.py` (or `download_k8s_manifests.py` for K8s)
2. Deduplicate by content hash
3. Filter to only TCT-encodable instances (schema-valid)
4. Both TCT and BPE models train on identical, valid data

---

### Schema Creation

| Schema | Source | Processing |
|--------|--------|------------|
| **ESLint** | [SchemaStore](https://schemastore.org) | Direct download (`eslintrc.json`) |
| **TSConfig** | [SchemaStore](https://schemastore.org) | Direct download (`tsconfig.json`) |
| **Kubernetes** | Kubernetes OpenAPI spec | Multi-step processing (see below) |

**Kubernetes schema creation** (complex due to discriminated unions):

1. **Extract definitions** from Kubernetes OpenAPI spec (`kubernetes-definitions.json`)
2. **Filter stable APIs** (`generate_k8s_schema.py`):
   - Remove alpha/beta/deprecated APIs
   - Keep only stable v1 resources
   - Add `apiVersion` const constraints for discriminator detection
   - Create `oneOf` union schema (~50 resource types discriminated by `kind`)
3. **Enhance enums** (`enhance_k8s_schema.py`):
   - Add enum constraints for string fields with documented values
   - Examples: `imagePullPolicy` → `["Always", "Never", "IfNotPresent"]`
   - Enables TCT's EnumIndexOptimization for better compression

**Schema locations**: `~/git/tct/schemas/popular/`

---

### Data Directory Structure

All data is stored in `~/Desktop/data/` with consistent naming.

**Tokenization Approaches**:

| Approach | Description | Vocab Size |
|----------|-------------|------------|
| **TCT-BPE** | TCT structural tokens + BPE compression | 10k-20k |
| **UTF8-BPE** | Standard BPE on raw UTF-8 text | 16k-24k |

**Key insight**: Both approaches use BPE, but on different representations:
- TCT-BPE: BPE on structural TCT tokens (257 base + ~10-20k merges)
- UTF8-BPE: BPE on raw UTF-8 bytes (256 base + ~16-24k merges)
- Both achieve **same average sequence length** (compression-matched)
- TCT-BPE achieves this with **smaller vocabulary** (38-45% reduction)

**PAD token**: Token 0 is reserved for padding in both approaches.

**Encoded Datasets** (JSONL format, pre-tokenized):

| Schema | TCT-BPE Directory | UTF8-BPE Directory | Files |
|--------|-------------------|-------------------|-------|
| **tsconfig** | `tsconfig-tct-bpe-10k/` | `tsconfig-utf8-bpe-10k/` | 356K |
| **eslintrc** | `eslintrc-tct-bpe-10k/` | `eslintrc-utf8-bpe-10k/` | 127K |
| **kubernetes** | `kubernetes-tct-bpe/` | `kubernetes-utf8-bpe/` | 246K |

Each directory contains:
- `train.jsonl` - 90% of sequences
- `validate.jsonl` - 10% of sequences
- `metadata.json` - Encoding parameters
- `stats.json` - Token statistics

---

### Model Sizes (3 scales)

**Approach**: Match transformer architecture (layers, heads, dims), not total parameters. TCT-BPE and UTF8-BPE models will have different total params due to vocabulary size, but **identical transformer capacity**.

| Size | d_model | Layers | Heads | Head Dim | Transformer Params |
|------|---------|--------|-------|----------|-------------------|
| **Small** | 512 | 10 | 8 | 64 | ~31M |
| **Medium** | 768 | 13 | 12 | 64 | ~92M |
| **Large** | 1024 | 24 | 16 | 64 | ~302M |

**Total parameters by schema** (kubernetes is reference, targets ~50M/~125M/~350M):

| Schema | Size | TCT-BPE Params | UTF8-BPE Params | Target |
|--------|------|----------------|-----------------|--------|
| **kubernetes** | Small | 51.9M | 55.9M | ~50M |
| **kubernetes** | Medium | 122.7M | 128.7M | ~125M |
| **kubernetes** | Large | 342.9M | 350.9M | ~350M |
| eslintrc | Small | 41.7M | 50.2M | - |
| eslintrc | Medium | 107.4M | 120.2M | - |
| eslintrc | Large | 322.5M | 339.5M | - |
| tsconfig | Small | 41.7M | 48.0M | - |
| tsconfig | Medium | 107.4M | 116.9M | - |
| tsconfig | Large | 322.5M | 335.2M | - |

**Vocab sizes by schema and tokenizer**:

| Schema | TCT-BPE | UTF8-BPE | TCT Advantage |
|--------|---------|----------|---------------|
| tsconfig | 10,000 | 16,197 | 38% smaller |
| eslintrc | 10,000 | 18,337 | 45% smaller |
| kubernetes | 20,000 | 23,887 | 16% smaller |

**Key insight**: Same transformer architecture, but UTF8-BPE has more embedding parameters due to larger vocab. TCT-BPE achieves same compression with smaller vocab = fewer total params.

---

### Context Lengths (Schema-Specific)

| Schema | Context | P99 | Coverage | Rationale |
|--------|---------|-----|----------|-----------|
| **tsconfig** | 256 | 159 | 99.3%+ | Simple configs |
| **eslintrc** | 512 | 345 | 99.6%+ | Medium complexity |
| **kubernetes** | 2048 | 2653 | 99%+ | Complex manifests |

---

### Training Configuration (Epoch-Based)

**Epochs by schema complexity**:

| Schema | Epochs | Train Files | Train Tokens (TCT) | Avg Len |
|--------|--------|-------------|-------------------|---------|
| tsconfig | 50 | 320,202 | ~8M | 24.75 |
| eslintrc | 75 | 114,500 | ~4M | 35.19 |
| kubernetes | 100 | 221,795 | ~46M | 189.07 |

**Batch sizes** (scale with context, target RTX 4090):

| Context | Small | Medium | Large |
|---------|-------|--------|-------|
| 256 | 64×2 | 32×4 | 16×8 |
| 512 | 32×4 | 16×8 | 8×16 |
| 2048 | 16×8 | 4×32 | 2×64 |

**Training hyperparameters**:

| Size | Learning Rate |
|------|---------------|
| Small | 3.0e-4 |
| Medium | 2.5e-4 |
| Large | 2.0e-4 |

**Constant across all models**:
- Optimizer: AdamW (β1=0.9, β2=0.95)
- Weight decay: 0.1
- Warmup: 5% of first epoch
- Gradient clipping: 1.0
- Architecture: RoPE, RMSNorm, no bias

**Checkpointing** (for ICML paper):
- Evaluate every epoch (clean learning curves)
- Save checkpoint every 10% of training
- Save best model (lowest validation loss)

---

### Baselines

#### 1. UTF8-BPE (Compression-Matched Primary Baseline)
- Standard byte-pair encoding trained on raw UTF-8 JSON text
- **Critical**: Train until **same average sequence length** as TCT-BPE
- This controls for compression efficiency, isolating the effect of tokenization approach
- Same model architecture, same data, same compute, same sequence lengths

**Compression matching achieved** (see TRAINING_DATA_REPORT.md):

| Schema | TCT-BPE Avg | UTF8-BPE Avg | Difference |
|--------|-------------|--------------|------------|
| tsconfig | 24.75 | 24.75 | 0.00% |
| eslintrc | 35.19 | 35.18 | 0.03% |
| kubernetes | 189.07 | 189.05 | 0.01% |

Both tokenizers achieve nearly identical compression, so any performance difference is purely from the tokenization approach.

#### 2. UTF8-BPE + Constrained Decoding (Secondary Baseline)
- Same compression-matched UTF8-BPE model as baseline 1
- At inference: apply grammar-constrained sampling (XGrammar library)
- Shows: TCT advantage is from **learning**, not just validity guarantees

**Comparison matrix:**

| Method | Training | Inference | Validity | What Model Learns |
|--------|----------|-----------|----------|-------------------|
| **TCT-BPE** | TCT tokenization | Normal sampling | Guaranteed | Semantics only |
| **UTF8-BPE** | UTF8 tokenization | Normal sampling | Not guaranteed | Syntax + semantics |
| **UTF8-BPE + XGrammar** | UTF8 tokenization | Constrained sampling | Guaranteed* | Syntax + semantics |

*XGrammar guarantees validity only when generation terminates naturally. TCT's `decode_prefix` always produces valid JSON even from truncated sequences.

**Key insight**: By matching compression, we isolate the key question: *Does structural tokenization help learning?* If TCT wins at equal sequence length, it's not just about vocab size.

**TCT Tokenizer Wheels:**
- `tct_kubernetes_20k` (20,000 vocab)
- `tct_eslintrc_10k` (10,000 vocab)
- `tct_tsconfig_10k` (10,000 vocab)

All wheels support `decode_prefix()` for truncation-tolerant streaming decode.

---

### Training Approach

**Train all models from scratch** with identical:
- Architecture (GPT-2 style decoder-only transformer)
- Training data (per schema)
- Compute budget (steps × batch size)
- Hyperparameters (learning rate, optimizer, etc.)

**Only variable: tokenization scheme (TCT vs BPE)**

This isolates the effect of tokenization, which is the core scientific claim. Using pretrained models would introduce confounds (different pretraining data, optimization history, etc.).

---

## Experiments

### Experiment 1: Training Efficiency

**Question**: Does TCT-BPE lead to faster convergence than UTF8-BPE?

**Metrics**:
- Epochs to reach threshold loss (e.g., loss < 1.0)
- Training curves (loss vs. epochs)
- Final loss after fixed epochs

**Expected Results**:
| Metric | TCT-BPE | UTF8-BPE |
|--------|---------|----------|
| Epochs to threshold | Lower | Higher |
| Final loss | Lower | Higher |

---

### Experiment 2: Model Quality

**Question**: Does TCT-BPE produce better models than UTF8-BPE?

**Metrics**:
- Validation loss / perplexity
- Bits-per-byte (for fair comparison across tokenizers)
- Cross-entropy on held-out test set

**Expected Results**:
- TCT-BPE achieves 2-3x lower perplexity than UTF8-BPE

---

### Experiment 3: Syntactic Validity

**Question**: Does TCT-BPE guarantee valid outputs?

**Metrics**:
- Validity rate: % of generated outputs that parse as valid JSON
- Validity rate: % that conform to schema

**Expected Results**:
| | TCT-BPE | UTF8-BPE | UTF8-BPE + Constrained |
|---|---------|----------|------------------------|
| Parse validity | 100% | <100% | 100% |
| Schema validity | 100% | <<100% | ~100% |

---

### Experiment 4: Constrained Decoding Comparison

**Question**: Is TCT-BPE's advantage from better learning or just validity guarantees?

**Setup**:
1. TCT-BPE model: trained with TCT tokenization
2. UTF8-BPE model: trained with standard BPE
3. UTF8-BPE + XGrammar: UTF8-BPE model with constrained decoding at inference

**Metrics**:
- Output quality (semantic correctness, not just validity)
- Inference speed (tokens/second)
- Training efficiency (UTF8-BPE vs TCT-BPE)

**Expected Results**:
- TCT-BPE beats UTF8-BPE+XGrammar on quality (proves learning advantage)
- TCT-BPE faster at inference (no grammar checking overhead)

---

### Experiment 5: Inference Speed

**Question**: Does TCT-BPE have lower inference overhead than constrained decoding?

**Metrics**:
- Tokens/second: TCT-BPE vs. UTF8-BPE vs. UTF8-BPE + XGrammar
- Latency per sample

**Expected Results**:
| Method | Tokens/sec | Overhead |
|--------|------------|----------|
| TCT-BPE | High | None (just decoding) |
| UTF8-BPE | High | None |
| UTF8-BPE + XGrammar | Lower | FSM constraint checking |

**Rationale**: Validates that TCT has low inference cost while constrained decoding has high cost.

---

### Experiment 6: Scaling Analysis

**Question**: How does TCT-BPE's advantage change with model size and schema complexity?

**Setup**: For each schema, train Small/Medium/Large models with TCT-BPE and UTF8-BPE

**Metrics**:
- Relative improvement (TCT-BPE loss / UTF8-BPE loss) at each scale
- Convergence speed ratio at each scale
- Compare across schema complexity (K8s complex vs. TSConfig simple)

**Expected Results**:
- TCT-BPE advantage persists or grows with scale
- Smaller TCT-BPE models match larger UTF8-BPE models (efficiency claim)
- Advantage may be larger on complex schemas (more syntax to offload)

---

### Experiment 7: Vocabulary Analysis

**Question**: How does vocabulary size affect efficiency?

**Metrics**:
- Vocabulary size: TCT-BPE (10-20K) vs UTF8-BPE (16-24K)
- Tokens per sample (compression) - matched by design
- Embedding table size (memory)

**Expected Results** (from TRAINING_DATA_REPORT.md):
| Schema | TCT-BPE Vocab | UTF8-BPE Vocab | Reduction |
|--------|---------------|----------------|-----------|
| tsconfig | 10,000 | 16,197 | 38% |
| eslintrc | 10,000 | 18,337 | 45% |
| kubernetes | 20,000 | 23,887 | 16% |

**Key insight**: TCT-BPE achieves the same compression with 16-45% smaller vocabulary.

---

## Ablations

### Ablation 1: Schema Complexity

**Question**: Does TCT's advantage grow with schema complexity?

**Setup**: Order schemas by complexity, measure relative improvement

**Hypothesis**: TCT advantage increases with complexity (more syntax = more for BPE to learn)

**Note**: This is now folded into Experiment 6 (Scaling Analysis).

---

## Downstream Evaluation (Stretch Goal)

### Semantic Correctness

Beyond syntactic validity, measure whether outputs are **semantically meaningful**:

1. **K8s manifests**:
   - Do generated Deployments reference valid container images?
   - Are resource limits reasonable?
   - Would `kubectl apply` succeed?

2. **ESLint configs**:
   - Are rule configurations valid?
   - Do referenced plugins exist?

3. **TSConfig**:
   - Are compiler options compatible?
   - Do path mappings make sense?

---

## Evaluation Script

### Usage

The `scripts/eval_generation.py` script provides comprehensive evaluation for the three-way comparison:

```bash
# Validation set evaluation (loss, perplexity, bits-per-byte)
python -m scripts.eval_generation \
    --schema kubernetes \
    --tct_checkpoint checkpoints/k8s_tct_small/ \
    --utf8_checkpoint checkpoints/k8s_utf8_small/ \
    --eval_validation

# Generation evaluation with XGrammar-constrained decoding
python -m scripts.eval_generation \
    --schema kubernetes \
    --tct_checkpoint checkpoints/k8s_tct_small/ \
    --utf8_checkpoint checkpoints/k8s_utf8_small/ \
    --eval_generation \
    --xgrammar \
    --num_samples 100

# Both evaluations with output file
python -m scripts.eval_generation \
    --schema kubernetes \
    --tct_checkpoint checkpoints/k8s_tct_small/ \
    --utf8_checkpoint checkpoints/k8s_utf8_small/ \
    --eval_validation \
    --eval_generation \
    --xgrammar \
    --output results.json \
    --save_samples samples/
```

### Validation Set Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Validation Loss** | Cross-entropy on validate.jsonl | Primary quality measure |
| **Perplexity** | exp(validation_loss) | Interpretable quality |
| **Bits-per-byte** | (loss × tokens) / (bytes × ln(2)) | Fair cross-tokenizer comparison |
| **Token Accuracy** | % correct next-token predictions | Fine-grained quality |
| **Top-5 Accuracy** | % correct in top-5 predictions | Model confidence measure |

### Generation Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| **JSON Validity %** | % outputs that parse as valid JSON | Syntactic correctness |
| **Schema Validity %** | % outputs that match JSON schema | Schema conformance |
| **Field Coverage %** | % with required fields (apiVersion, kind, metadata) | Structural completeness |
| **Tokens/sec** | Generation throughput | Inference speed |
| **Unique %** | % distinct outputs (SHA256 hash) | Output diversity |
| **Mean/Min/Max Tokens** | Output length statistics | Generation behavior |

### Statistical Rigor

- **Bootstrap CIs**: 95% confidence intervals for all rates
- **Reproducibility**: Fixed seeds (default: 42) for fair comparison
- **Data consistency**: TCT and UTF8 models evaluate on same underlying samples

### Tokenizer Consistency

For UTF8-BPE evaluation, both constrained and unconstrained generation use `UTF8BPEDecoder` from the merge table (`bpe-merges/kubernetes-utf8-bpe-matched.json`). This ensures:
1. Consistent tokenization between constrained and unconstrained modes
2. No dependency on external tokenizer modules for evaluation
3. Fair comparison using the same decoding logic

---

## Summary: Minimum Viable Experiments for ICML

### Must-Have (Core Claims)

| Experiment | Supports Claim | RQ |
|------------|----------------|-----|
| Training efficiency (3 schemas × 3 sizes) | TCT-BPE learns faster | RQ1 |
| Model quality comparison | TCT-BPE learns better | RQ1 |
| Syntactic validity rates | Syntax is in tokenization | RQ1 |
| Constrained decoding comparison | Advantage is from learning, not filtering | RQ2 |
| Inference speed comparison | TCT-BPE has low inference cost | RQ2 |

### Should-Have (Strengthens Paper)

| Experiment | Supports Claim | RQ |
|------------|----------------|-----|
| Scaling analysis (model size + schema complexity) | TCT-BPE works at scale | RQ3 |
| Vocabulary analysis | TCT-BPE is more efficient | RQ1 |

### Nice-to-Have (If Time Permits)

| Experiment | Supports Claim |
|------------|----------------|
| Semantic correctness eval | TCT-BPE outputs are meaningful |
| Non-JSON domain | Generality beyond JSON |

---

## Resource Estimates

Assuming RTX 4090 or A100 GPUs:

| Experiment | GPU Hours (Est.) |
|------------|------------------|
| 3 schemas × 3 sizes (~33M, ~60M, ~130M) × 2 methods (TCT-BPE, UTF8-BPE) | ~150-200 hours |
| Constrained decoding inference (UTF8-BPE + XGrammar) | ~10 hours |
| Ablations (data efficiency, etc.) | ~30 hours |
| **Total** | **~200-250 GPU hours** |

---

## Success Criteria

The paper is ready for ICML submission when we can show:

1. **TCT-BPE trains 2x+ faster** than UTF8-BPE across all schemas and model sizes
2. **TCT-BPE achieves 2x+ lower perplexity** than UTF8-BPE
3. **TCT-BPE outputs are 100% valid** by construction
4. **TCT-BPE beats constrained decoding** on output quality (not just validity)
5. **Results are consistent** across 3 schemas and 3 model sizes

If we achieve these, the title claim "Type-Constrained Tokenization **Enables** Syntax-Free Learning" is well-supported.
