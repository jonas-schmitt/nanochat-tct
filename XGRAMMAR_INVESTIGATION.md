# XGrammar Investigation - RESOLVED

## Problem (Original)

XGrammar-constrained generation had a ~50% failure rate across ALL schemas.

## Root Cause - IDENTIFIED

The `GrammarMatcher` was created with default settings (`terminate_without_stop_token=False`), which requires an EOS token to terminate. However, the model was trained WITHOUT EOS tokens - training data uses PAD tokens for padding, not as end markers.

### Training Setup (from `configs/jsonl_dataloader.py`)
```
Input:  [BOS, token1, token2, ..., tokenN, PAD, PAD, ...]
Target: [token1, token2, ..., tokenN, -1, -1, ...]
```

The model never learns to produce EOS because training targets mask PAD positions with `-1`.

## Fix Applied

**Commit**: `c04e36e` - Fix XGrammar ~50% failure rate with terminate_without_stop_token=True

```python
# scripts/eval_icml.py line 917-920
matchers = [xgrammar.GrammarMatcher(compiled_grammar, terminate_without_stop_token=True)
            for _ in range(current_batch_size)]
```

With `terminate_without_stop_token=True`, the grammar terminates when JSON structure is complete, without needing an EOS token.

## Results After Fix

| Method | Completion Rate | Notes |
|--------|-----------------|-------|
| UTF8+XGrammar | ~99% | Was ~50% before fix |
| TCT | ~95% | Unchanged |

## JSON Repair Value

With the fix, JSON repair is now a **minor optimization** rather than critical:
- Only ~1-2% of samples fail (hit max_tokens without completing)
- These are typically very long/verbose JSON structures
- Repair can recover some of these, but the impact is small

JSON repair remains valuable for:
1. Recovering the ~1-2% that hit max_tokens
2. Handling edge cases in production
3. Providing fallback for unexpected failures

## Key Learnings

1. **Match training behavior**: XGrammar settings must align with how the model was trained
2. **No EOS in training = use `terminate_without_stop_token=True`**
3. **Bytecode caching**: When testing fixes, ensure fresh Python imports (delete `__pycache__`)
