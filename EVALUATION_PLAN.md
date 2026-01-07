# TCT vs BPE+XGrammar Evaluation Plan

**Goal**: Provide rigorous, fair comparisons for ICML 2026 submission.

**Core insight**: Loss comparisons across different tokenizations are apples-to-oranges. We need metrics that are unambiguously the same for both methods.

---

## 1. Generation Quality Evaluation (Primary)

### Objective
Show that TCT generates configs that better match real data distributions.

### Method

#### 1.1 Sample Generation
```
For each schema (TSConfig, ESLint, Kubernetes):
  - Generate N=10,000 samples from TCT model (native)
  - Generate N=10,000 samples from BPE model + XGrammar
  - Use same sampling parameters (temperature, top-p) for both
```

#### 1.2 Field Value Distribution Comparison
```
For each field in schema:
  1. Extract field values from generated samples
  2. Extract field values from validation set (ground truth distribution)
  3. Compute distributional distance:
     - KL divergence: KL(real || generated)
     - Total variation distance
     - Chi-squared statistic (for categorical fields)
```

#### 1.3 Fields to Evaluate

**TSConfig** (boolean/enum heavy):
- `compilerOptions.strict` (boolean)
- `compilerOptions.target` (enum: ES5, ES6, ES2017, ES2020, ESNext, ...)
- `compilerOptions.module` (enum: CommonJS, ESNext, AMD, ...)
- `compilerOptions.moduleResolution` (enum: node, bundler, ...)
- `compilerOptions.jsx` (enum: react, react-jsx, preserve, ...)
- `compilerOptions.esModuleInterop` (boolean)
- `compilerOptions.skipLibCheck` (boolean)
- `compilerOptions.forceConsistentCasingInFileNames` (boolean)

**ESLint** (mixed):
- `env.browser` (boolean)
- `env.node` (boolean)
- `env.es6` (boolean)
- `parserOptions.ecmaVersion` (enum/number)
- `parserOptions.sourceType` (enum: module, script)
- Rule severity distribution (off/warn/error)

**Kubernetes** (complex enums):
- `kind` (enum: Pod, Deployment, Service, ...)
- `apiVersion` (enum)
- `spec.restartPolicy` (enum: Always, OnFailure, Never)
- `spec.containers[].imagePullPolicy` (enum)

#### 1.4 Metrics to Report

| Metric | Description |
|--------|-------------|
| Mean KL divergence | Average KL(real \|\| gen) across fields |
| Field accuracy | % of fields where mode matches real data mode |
| Distributional accuracy | % of fields with KL < threshold |
| Coverage | % of real field values that appear in generated samples |

---

## 2. Semantic Field Accuracy (Secondary)

### Objective
Show that TCT assigns higher probability to correct semantic values during validation.

### Method

#### 2.1 For Boolean Fields
```python
# TCT
p_correct_tct = model_tct.get_prob(
    token=ground_truth_bool,  # 0 or 1
    position=bool_decision_position,
    context=tct_prefix
)

# BPE
# "true" or "false" is typically a single token
p_correct_bpe = model_bpe.get_prob(
    token=tokenizer.encode(ground_truth_str)[0],  # "true" or "false"
    context=bpe_prefix
)
```

#### 2.2 For Enum Fields
```python
# TCT
p_correct_tct = model_tct.get_prob(
    token=ground_truth_enum_index,
    position=enum_decision_position,
    context=tct_prefix
)

# BPE - may be multiple tokens
enum_tokens = tokenizer.encode(ground_truth_enum_str)
p_correct_bpe = product([
    model_bpe.get_prob(token=t, context=bpe_prefix + enum_tokens[:i])
    for i, t in enumerate(enum_tokens)
])
```

#### 2.3 Metrics to Report

| Metric | Description |
|--------|-------------|
| Mean P(correct) | Average probability assigned to correct value |
| Top-1 accuracy | % where argmax prediction is correct |
| Top-3 accuracy | % where correct value in top 3 predictions |
| Calibration | Reliability diagram (predicted prob vs actual accuracy) |

---

## 3. Constrained BPB (Tertiary)

### Objective
Information-theoretic comparison with XGrammar constraints applied.

### Method

```python
def compute_constrained_bpb(model, tokenizer, xgrammar, validation_data):
    total_loss = 0
    total_bytes = 0

    for json_doc in validation_data:
        tokens = tokenizer.encode(json_doc)
        json_bytes = len(json_doc.encode('utf-8'))

        xg_state = xgrammar.init_state()

        for i, target_token in enumerate(tokens):
            logits = model(tokens[:i])

            # Get valid token mask
            valid_mask = xgrammar.get_valid_tokens(xg_state)

            # Renormalize over valid tokens
            valid_logits = logits[valid_mask]
            log_probs = log_softmax(valid_logits)

            # Map target token to valid-set index
            valid_idx = valid_mask_to_index(target_token, valid_mask)
            total_loss += -log_probs[valid_idx]

            # Advance grammar state
            xg_state = xgrammar.advance(xg_state, target_token)

        total_bytes += json_bytes

    return total_loss / math.log(2) / total_bytes
```

### Metrics to Report

| Metric | Description |
|--------|-------------|
| BPB (TCT) | Bits per byte for TCT model |
| BPB (BPE raw) | Bits per byte for BPE without constraints |
| BPB (BPE constrained) | Bits per byte for BPE with XGrammar |

---

## 4. Implementation Checklist

### Prerequisites
- [ ] XGrammar integration with BPE models
- [ ] Generation loop for BPE+XGrammar
- [ ] Field extraction utilities for each schema
- [ ] Distribution comparison utilities

### Generation Quality
- [ ] Generate 10K samples from TCT (TSConfig)
- [ ] Generate 10K samples from BPE+XGrammar (TSConfig)
- [ ] Extract field values from generated samples
- [ ] Extract field values from validation set
- [ ] Compute KL divergence per field
- [ ] Repeat for ESLint
- [ ] Repeat for Kubernetes (if schema ready)

### Semantic Accuracy
- [ ] Identify boolean/enum field positions in TCT sequences
- [ ] Identify boolean/enum field positions in BPE sequences
- [ ] Extract probabilities at decision points
- [ ] Compute accuracy metrics

### Constrained BPB
- [ ] Implement constrained loss computation
- [ ] Run on validation set
- [ ] Compare TCT vs BPE raw vs BPE constrained

---

## 5. Expected Results & Narrative

### Hypothesis
TCT will outperform BPE+XGrammar on:
1. **Distributional match**: TCT-generated configs closer to real data
2. **Semantic accuracy**: TCT assigns higher P(correct) to field values
3. **Constrained BPB**: TCT needs fewer bits per byte

### Why We Expect This
- TCT trains on semantic decisions only; BPE trains on syntax + semantics
- TCT's loss directly reflects semantic uncertainty; BPE's loss includes syntactic noise
- Even with XGrammar masking, BPE model learned to predict syntax tokens

### Paper Narrative
> "While constrained decoding ensures validity, it cannot recover capacity wasted learning syntax during training. TCT models, trained only on semantic decisions, achieve higher probability on correct field values (Table X) and generate configs that better match real data distributions (Table Y)."

---

## 6. Sampling Parameters

For fair comparison, use identical sampling for both models:

```python
sampling_config = {
    "temperature": 1.0,      # or 0.8 for slightly sharper
    "top_p": 0.95,           # nucleus sampling
    "top_k": None,           # disabled
    "max_tokens": 2048,      # sufficient for all schemas
    "num_samples": 10000,
}
```

Consider also reporting results at multiple temperatures (0.7, 1.0, 1.2) to show robustness.

---

## 7. Statistical Significance

For all comparisons:
- Report mean ± std across multiple runs (if stochastic)
- Use bootstrap confidence intervals for distributional metrics
- Report p-values for field-level comparisons (paired t-test or Wilcoxon)

---

## 8. Potential Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| BPE+XGrammar generation is slow | Use batched generation, budget sufficient time |
| Some field values never generated | Report coverage metric, increase sample count |
| XGrammar state tracking bugs | Validate on small examples first |
| Field extraction errors | Manual validation on sample of outputs |
| Multi-token enum values in BPE | Carefully compute joint probability |

---

## 9. Timeline

| Task | Estimated Time |
|------|----------------|
| XGrammar integration | 1 day |
| Generation infrastructure | 1 day |
| Field extraction utilities | 1 day |
| Run generation (all schemas) | 0.5 days |
| Distribution comparison | 0.5 days |
| Semantic accuracy evaluation | 1 day |
| Constrained BPB | 0.5 days |
| Analysis & visualization | 1 day |
| **Total** | **~6-7 days** |
