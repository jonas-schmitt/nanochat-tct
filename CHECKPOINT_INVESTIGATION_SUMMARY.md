# Medium Model Checkpoint Investigation Summary

**Date**: 2025-11-07
**Checkpoints Investigated**: `model_035000.pt` (best val 1.654) and `model_040000.pt` (final)
**Model**: Medium 83M params (768d×10L×12H), decoder-only, prefix_mode=all

## Executive Summary

The medium model training **completed successfully** with excellent validation loss (1.654 @ step 35k, beating small's 1.797 by 8%). However, the model **cannot generate valid workflows** due to a fundamental mismatch between the windowed training approach and TCT's schema-based decoding requirements.

## What Works ✅

1. **Training Success**
   - Converged to val loss 1.654 (best @ step 35k)
   - 8% better than small model's 1.797
   - Learning rate adjustments successfully resolved early overfitting (steps 6k-8k)
   - No NaN losses, healthy gradients throughout

2. **Model Functionality**
   - Loads checkpoints correctly
   - Generates tokens within vocab range [0, 8191]
   - Can predict next tokens given context windows

3. **TCT Mechanics**
   - Position token mechanism works correctly for training
   - Windows decode successfully when position token is stripped: `decode(window[1:])`
   - Round-trip encoding/decoding works for complete valid workflows

## What Doesn't Work ❌

1. **From-Scratch Generation**
   - **Result**: Fails immediately with schema violations
   - **Example**: Generated `[1, 0, 1, 5, ...]` instead of valid start `[1, 1024, 50, ...]`
   - **Error**: "Decode failed: Unexpected end of token stream..."

2. **Workflow Extension**
   - **Test**: Start with valid 11-token prefix `{"name": "CI", "on": "push", "jobs": {}}`
   - **Result**: Model generates 200 tokens, but decoder only consumes original 11
   - **Outcome**: 0/200 new tokens are valid (100% rejection rate)

3. **Schema Compliance**
   - Generates tokens that violate TCT structural constraints
   - Examples of invalid tokens in context: 317, 3328, 5953
   - Errors like "Invalid token '317' for this type"
   - "Invalid data: Key '' does not match pattern"

4. **Mode Collapse**
   - Low temperature (0.1) produces repetitive loops: `[1,1,1,1024,20,0,1,1,1,1024,20,0,...]`
   - High temperature (2.0) produces "Arithmetic overflow in multi-token codec"

## Root Cause Analysis

### The Problem

**TCT uses schema-based deterministic encoding** where:
- Each token must satisfy structural constraints based on the current schema state
- Multi-token sequences form "modules" that must be complete to decode
- Token validity depends on context (e.g., token 1024 valid after token 1, invalid after token 0)

**The model learned statistical patterns** from:
- Windowed segments (1024 tokens, stride=32) of valid workflows
- Position-aware contexts `[position, content...]`
- Only valid token sequences (never saw violations during training)

### Why Generation Fails

1. **No Schema Knowledge**: Model predicts statistically likely tokens, not schema-valid tokens
2. **Window Training Mismatch**: Trained on mid-sequence windows, not workflow starts/ends
3. **No Hard Constraints**: Nothing prevents generating invalid token combinations
4. **Drift Accumulation**: One invalid token breaks entire downstream sequence

### Training Data Discovery

Training data shape: `[870,351 × 1024]` windows

**Example window** (position 0):
```
[0, 1, 34, 66, 2326, 4149, ...]  # Position 0 prepended
```

**Critical findings**:
- Each window has format `[position_token, content_tokens...]`
- Windows are NOT individually decodable (incomplete modules)
- Position token must be stripped before decoding: `decode(window[1:])`
- Prepending position token 0 to valid workflow breaks it: "Invalid token '1024' for this type"

## Key Discoveries

### 1. Position Token Mechanism

```python
# Training window extraction
window = extract_window(tokens, start=0, end=10)
# Returns: [0, 1, 1024, 50, 0, 25, 0, 0, 0, 1, 1024]
#           ^-- position token

# Decoding
decode(window)      # ❌ FAILS: "Invalid token '1024' for this type"
decode(window[1:])  # ✅ WORKS: Successfully decodes content
```

### 2. Valid Workflow Structure

```python
# Minimal valid workflow
{"name": "CI", "on": "push", "jobs": {}}

# Encodes to:
[1, 1024, 50, 0, 25, 0, 0, 0, 0, 0, 0]  # 11 tokens
#  ^-- Starts with token 1, then 1024

# NOT token 0 (position marker)
```

### 3. Generation Behavior

**Greedy (temp=0.1)**:
- Mode collapse: `[0,1,1,1,1,1,1,1,1,1,1,1024,20,0,1,1,1,1,1024,20,0,...]`
- Error: "Key '' does not match pattern"

**Medium (temp=1.2)**:
- More diverse (59 unique tokens)
- Error: "Invalid token '317' for this type"

**High (temp=2.0)**:
- Most diverse (104 unique tokens)
- Error: "Arithmetic overflow in multi-token codec"

**All fail** regardless of temperature or seed.

## Attempted Solutions

### Attempted #1: Generate from Scratch
**Approach**: Let model generate starting from empty context
**Result**: Fails immediately - first tokens violate schema

### Attempted #2: Start with Valid Prefix
**Approach**: Provide `{"name": "CI", "on": "push", "jobs": {}}` as prompt
**Result**: Prefix decodes, but all 200 generated tokens invalid (11/211 consumed)

### Attempted #3: Force Position Token 0
**Approach**: Start generation with position token 0
**Result**: Mode collapse or schema violations

### Attempted #4: Vary Temperature
**Approach**: Test temp ∈ {0.1, 0.8, 1.2, 1.5, 2.0}
**Result**: Different errors, same fundamental failure

## Debugging Artifacts

Created comprehensive debugging files:

### `tct_decoding_issues.json`
- 8 categories of issues with examples
- Token sequences from different generation attempts
- Comparison with valid workflow tokens
- Decoding error messages with context
- Training data characteristics

### `tct_generated_samples.json`
- 5 greedy samples with different seeds
- All fail with "Invalid token '1024' for this type"

### `scripts/debug_generation.py`
- Incremental decoding analysis tool
- Token-by-token decode attempts
- Position-finding for problematic tokens

## Implications for TCT

These findings suggest potential issues with the windowed decoding approach:

1. **Position Token Behavior**: Token 0 as position marker breaks workflows when included in decode
2. **Schema State Tracking**: May need explicit state machine for valid next tokens
3. **Module Boundaries**: Unclear how to handle incomplete modules during generation

## Next Steps & Recommendations

### Immediate Actions

1. **Verify TCT Windowed Decoding**
   - Review how `extract_window()` is intended to work with generation
   - Check if there's a corresponding `decode_window()` or integration mechanism
   - Investigate if position tokens have special handling during decode

2. **Test Window Integration**
   - Implement window-based generation that maintains full workflow state
   - Try: Start with valid workflow → extract window → generate next token → integrate → repeat

3. **Constrained Decoding**
   - Investigate if TCT exposes "valid next tokens" API
   - Implement beam search with schema validation
   - Filter logits to only allow schema-valid tokens

### Long-term Considerations

1. **Alternative Training**
   - Train on complete workflows (not windows) if memory allows
   - Add schema validation loss during training
   - Use teacher forcing with hard schema constraints

2. **Model Architecture**
   - Consider pointer networks or tree-structured models
   - Explore grammar-constrained generation
   - Investigate hierarchical models that respect workflow structure

3. **Hybrid Approach**
   - Use model for high-level decisions (workflow structure)
   - Use TCT rules for low-level token generation
   - Combine statistical and rule-based methods

## Training Metrics (For Reference)

| Step | Val Loss | Perplexity | Notes |
|------|----------|------------|-------|
| 2k   | 3.202    | 24.57      | Initial |
| 6k   | 2.867    | 17.58      | Overfitting starts |
| 10k  | 2.443    | 11.51      | Recovery via LR decay |
| 20k  | 2.214    | 9.15       | 50% complete |
| 30k  | 1.923    | 6.84       | Approaching small's 1.797 |
| **35k** | **1.654** | **5.23** | **BEST - beats small by 8%** |
| 40k  | 1.785    | 5.96       | Final checkpoint |

**Best checkpoint**: `model_035000.pt` (val loss 1.654)

## Conclusion

The medium model training was **technically successful** - excellent convergence, beats small model, no training issues. However, it **cannot generate usable workflows** due to the fundamental incompatibility between statistical token prediction and TCT's deterministic schema constraints.

**The core issue is not the model, but the generation approach.** The model learned exactly what it was trained to do (predict next tokens in windows), but this doesn't translate to generating schema-valid sequences.

**Recommendation**: Focus on understanding TCT's intended window-based generation mechanism before attempting further model improvements. The solution likely involves properly integrating generated windows into workflow sequences, not training a better model.

---

**Files Created**:
- `tct_decoding_issues.json` - Comprehensive debugging information
- `tct_generated_samples.json` - Sample sequences from different seeds
- `scripts/debug_generation.py` - Incremental decoding analysis tool
- `CHECKPOINT_INVESTIGATION_SUMMARY.md` - This document
