# Training Plan: TCT Workflow Generation with Epoch-Based Offsets

**Date:** 2025-11-03
**Model:** 384×12 (27.5M params)
**Training Strategy:** Epoch-based offset windowing with dedicated PAD token

---

## Problem Summary

Training on GitHub Actions workflows with TCT tokenization faced two key challenges:

1. **Overfitting with high overlap (stride=32)**
   - Validation loss degraded from 3.78 → 4.01 at step 30k
   - 97% overlap between consecutive windows
   - Only 0.55 epochs completed in 30k steps

2. **Padding ambiguity with position tokens**
   - Short workflows (78% are < 1024 tokens) need padding
   - Token `0` could be: position token, padding, or real content
   - Position semantics inconsistent with padding

---

## Solution: Epoch-Based Offsets + Dedicated PAD Token

### 1. Epoch-Based Offset Windowing

**Key Innovation:** Different window alignment per epoch, 0% overlap within epochs

```
Epoch 0 (offset=0):   [0-1023], [1024-2047], [2048-3071], ...
Epoch 1 (offset=32):  [32-1055], [1056-2079], [2080-3103], ...
Epoch 2 (offset=64):  [64-1087], [1088-2111], [2112-3135], ...
...
Epoch 31 (offset=992): [992-2015], [2016-3039], ...
```

**Benefits:**
- ✅ No overlap within each epoch (clean training signal)
- ✅ 32 different views across epochs (data augmentation)
- ✅ Always includes workflow beginning (critical context)
- ✅ Efficient: ~40k windows/epoch vs 967k with stride=32
- ✅ Expected: ~40+ epochs over 50k iterations

**Implementation:** `tct-bundle/adapters/tct_epoch_dataloader.py`

### 2. Dedicated PAD Token

**Problem:** Token ambiguity breaks position semantics
```python
# Ambiguous with token 0:
[pos=0, 0, 0, 0, tok_0, ...]  # Which 0 is which?
```

**Solution:** Expand vocab to 8193, use token 8192 for padding
```python
# Unambiguous:
[pos=0, tok_0, ..., tok_499, 8192, 8192, ...]  # PAD token 8192
         ↑ Real content always starts after position
```

**Padding Strategy:**
- **Location:** END of sequence
- **Input (x):** Token 8192 (valid for embedding, never from TCT)
- **Targets (y):** -1 at padding positions (ignored by `cross_entropy`)
- **Why END:** Position token semantics stay consistent (real content always starts immediately after)

**Changes:**
- `vocab_size: 8192 → 8193` in `model_config.py`
- TCT tokens: 0-8191 (unchanged, no BPE sacrificed)
- PAD token: 8192 (dedicated)

---

## Model Configuration

**Architecture:**
```python
vocab_size = 8193          # TCT (8192) + PAD token (8192)
context_size = 1024        # Total window (1 position + 1023 content)
d_model = 384              # Narrow but efficient
n_layers = 12              # Deep for workflow hierarchy
n_heads = 6                # head_dim = 64 (optimal)
parameters = 27,525,888    # ~27.5M params
```

**Training:**
```python
max_iters = 50000
batch_size = 32
learning_rate = 3e-4
warmup_iters = 1500
grad_clip = 1.0
```

**Data:**
- 100k GitHub Actions workflows
- Train: 90k workflows (~40k windows/epoch)
- Val: 10k workflows (~4.4k windows)

---

## Key Differences from Previous Run

| Aspect | Previous (stride=32) | Current (epoch-offset) |
|--------|---------------------|------------------------|
| Overlap per epoch | 97% | 0% |
| Windows per epoch | 967k | ~40k |
| Data augmentation | Minimal | 32 views/workflow |
| Epochs in 50k steps | 0.55 | ~40+ |
| Padding token | Token 0 (ambiguous) | Token 8192 (dedicated) |
| Vocab size | 8192 | 8193 |

---

## Expected Outcomes

**Generalization:**
- No overfitting (0% overlap prevents memorization)
- Better validation loss trajectory
- Model sees more diverse workflow contexts

**Success Metrics:**
- Validation loss continues improving (no plateau at 0.55 epochs)
- Final perplexity < 10
- Generated workflows: >95% valid JSON, >90% schema-correct

---

## Files Modified

1. **`tct-bundle/adapters/tct_epoch_dataloader.py`** (NEW)
   - Implements epoch-based offset windowing
   - Dynamic window creation per epoch
   - Dedicated PAD token (8192) with END padding
   - Loss masking (-1 in targets)

2. **`tct-bundle/adapters/model_config.py`**
   - Updated `MEDIUM_SMALL_CONFIG["vocab_size"]` from 8192 → 8193

3. **`scripts/tct_train.py`**
   - Switched to `create_epoch_offset_dataloader()`
   - Reads raw JSON workflows (not pre-prepared data)
   - Added epoch update in training iterator

---

## Training Command

```bash
# Start training (unbuffered output for real-time monitoring)
nohup python -u -m scripts.tct_train > ~/tct-training-epoch-offset.log 2>&1 &

# Monitor progress
tail -f ~/tct-training-epoch-offset.log

# Check process
ps aux | grep tct_train
```

**Expected timeline:**
- Initialization: 15-30 minutes (tokenize 90k workflows)
- Training: ~8-12 hours on RTX 4090 (50k iterations)

---

## Next Steps

1. ✅ Implement epoch-based offset dataloader
2. ✅ Add dedicated PAD token (vocab → 8193)
3. ✅ Update training script
4. ⏳ **Run full training (50k iterations)**
5. ⏳ Monitor validation loss (should improve vs stride=32)
6. ⏳ Evaluate generated workflows
7. ⏳ Compare to baseline: loss curves, perplexity, success rate

---

## Open Questions

- Will 0% overlap slow learning vs high overlap? (Trade-off: slower convergence but better generalization)
- Is 32 offset positions optimal or should we try more? (More = more augmentation but thinner data per view)
- Should we use dynamic context sizes in future? (Mix of 512/768/1024 for variety)

---

**Status:** Ready to train. Waiting for user confirmation to start 50k iteration run.
