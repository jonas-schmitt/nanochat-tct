# TCT UTF-8 String Encoding Fix Plan

**Status**: Ready for Implementation
**Priority**: Critical (blocks 28% of workflow generation)
**Estimated Time**: 4-5 hours hands-on + 2.5 hours dict/BPE training + 2.5 hours model training = ~9 hours total
**Expected Impact**: 72% → 95%+ success rate

---

## Executive Summary

The nanochat-tct model currently has a 28% failure rate due to **invalid UTF-8 sequences** in generated strings. Root cause analysis reveals that TCT's byte-level string encoding allows the model to generate invalid UTF-8 byte sequences that fail during decoding.

**Current TCT Architecture**:
1. Base encoding: Byte-level for strings (PROBLEM HERE)
2. Dictionary compression on top of base encoding
3. BPE compression on top of dictionary

**Recommended Solution**: Modify TCT to use Unicode codepoint encoding instead of byte-level encoding at the base layer. This guarantees valid UTF-8 by construction through Rust's `char::from_u32()` validation.

**⚠️ IMPORTANT**: After changing base encoding, both dictionary and BPE training must be redone on the new character-level tokens.

---

## 1. Problem Statement

### Current Behavior
- **Baseline success rate**: 72% (36/50 workflows generated successfully)
- **Failure rate**: 28% (14/50 workflows fail with UTF-8 errors)
- **Error message**: `Utf8Error` during `String::from_utf8()` decoding

### Example Failure
```
Original workflow name: "update schema"
Model generates: "updaté schéma"

Encoding (byte-level):
  'é' → bytes [195, 169] (0xC3 0xA9 in UTF-8)

Model prediction (INCORRECT):
  Token sequence: [..., 195, 50, ...]

Problem:
  - 195 (0xC3) starts a 2-byte UTF-8 sequence
  - 50 (0x32 = ASCII '2') is INVALID continuation byte
  - Valid range: 0x80-0xBF
  - Result: String::from_utf8() fails with Utf8Error
```

### Root Cause Location
- **File**: `/home/josch/git/tct/src/tokenizer/primitives.rs`
- **Lines**: 153-183
- **Function**: `impl JsonTokenEncoder for String`

### Technical Root Cause
TCT currently encodes strings as raw bytes (1 token = 1 byte):
```rust
fn encode(&self, tokens: &mut Vec<u32>) {
    tokens.push(self.len() as u32);  // BYTE length
    tokens.extend(self.as_bytes().iter().map(|&b| b as u32));  // RAW BYTES
}
```

The model predicts each byte **independently** without UTF-8 constraints, leading to invalid sequences.

---

## 2. Proposed Solution: Unicode Codepoint Encoding

### Approach
Replace byte-level encoding with Unicode codepoint encoding:
- **Current**: 1 token = 1 byte (0-255)
- **Proposed**: 1 token = 1 Unicode character (codepoint 0-0x10FFFF)

### Key Insight
Rust's `char::from_u32()` automatically validates Unicode codepoints, making invalid UTF-8 **impossible by construction**.

### Implementation

**File to modify**: `/home/josch/git/tct/src/tokenizer/primitives.rs` (lines 153-183)

**Current implementation**:
```rust
impl JsonTokenEncoder for String {
    fn encode(&self, tokens: &mut Vec<u32>) {
        tokens.push(self.len() as u32);
        tokens.extend(self.as_bytes().iter().map(|&b| b as u32));
    }

    fn decode(tokens: &mut impl Iterator<Item = u32>) -> Result<Self, DecodeError> {
        let len_token = tokens.next().ok_or(DecodeError::UnexpectedEndOfStream {
            context: EOFContext::new("String", Some("length"), "string length"),
        })?;
        let len = usize::try_from(len_token).map_err(|_| DecodeError::InvalidToken(len_token))?;
        let mut bytes = Vec::with_capacity(len);
        for _ in 0..len {
            let byte_val = tokens.next().ok_or(DecodeError::UnexpectedEndOfStream {
                context: EOFContext::new("String", Some("byte"), "UTF-8 byte"),
            })?;
            if byte_val > 255 {
                return Err(DecodeError::InvalidToken(byte_val));
            }
            let byte = byte_val as u8;
            bytes.push(byte);
        }
        String::from_utf8(bytes).map_err(|_| DecodeError::Utf8Error)  // FAILS HERE
    }

    fn token_count(&self) -> usize {
        1 + self.len()  // 1 for length + 1 per byte
    }
}
```

**Proposed implementation**:
```rust
impl JsonTokenEncoder for String {
    fn encode(&self, tokens: &mut Vec<u32>) {
        // Encode as Unicode codepoints, not bytes
        let chars: Vec<char> = self.chars().collect();
        tokens.push(chars.len() as u32);  // CHARACTER count
        tokens.extend(chars.iter().map(|&c| c as u32));  // Unicode codepoints
    }

    fn decode(tokens: &mut impl Iterator<Item = u32>) -> Result<Self, DecodeError> {
        let len_token = tokens.next().ok_or(DecodeError::UnexpectedEndOfStream {
            context: EOFContext::new("String", Some("length"), "string character count"),
        })?;
        let len = usize::try_from(len_token).map_err(|_| DecodeError::InvalidToken(len_token))?;

        let mut chars = Vec::with_capacity(len);
        for _ in 0..len {
            let codepoint = tokens.next().ok_or(DecodeError::UnexpectedEndOfStream {
                context: EOFContext::new("String", Some("character"), "Unicode codepoint"),
            })?;

            // char::from_u32() validates Unicode codepoints
            // Returns None for invalid codepoints (surrogates, out of range)
            let ch = char::from_u32(codepoint)
                .ok_or(DecodeError::InvalidToken(codepoint))?;

            chars.push(ch);
        }

        // Convert validated chars to String - cannot fail
        Ok(chars.iter().collect())
    }

    fn token_count(&self) -> usize {
        1 + self.chars().count()  // 1 for length + 1 per character
    }
}
```

### Why This Works
1. `char::from_u32()` validates codepoints:
   - Rejects surrogates (0xD800-0xDFFF)
   - Rejects out-of-range values (> 0x10FFFF)
   - Only accepts valid Unicode scalar values

2. Valid Unicode codepoints → valid UTF-8:
   - `chars.iter().collect()` creates String from validated chars
   - Rust guarantees UTF-8 validity for String constructed from chars
   - **Impossible to generate invalid UTF-8**

3. Dictionary and BPE work on top:
   - Base layer provides validated character tokens
   - Dictionary compresses common character sequences
   - BPE learns subword patterns from character-level tokens
   - All layers preserve UTF-8 validity guarantee

### Trade-offs

#### Advantages
✅ **Guarantees valid UTF-8** - decoding cannot fail
✅ **Simple implementation** - single file change
✅ **No vocabulary changes** - still 8,192 base tokens
✅ **More efficient for Unicode** - 'é' = 1 token instead of 2 bytes
✅ **Model learns semantic units** - characters instead of bytes

#### Disadvantages
❌ **Slightly more tokens for non-ASCII** - ~30% increase for heavy Unicode usage
❌ **Requires retraining** - existing checkpoints incompatible
❌ **Breaks backward compatibility** - old data needs regeneration

### Performance Impact

**ASCII strings** (majority of workflow content):
- No change: `"update schema"` = 13 tokens (before and after)

**Unicode strings** (less common):
```
Before (byte-level):
  "updaté schéma 🚀" → 1 (length) + 22 (bytes) = 23 tokens

After (codepoint-level):
  "updaté schéma 🚀" → 1 (length) + 15 (chars) = 16 tokens
```

**Actual impact on workflow generation**:
- Most workflow strings are ASCII (job IDs, step IDs, actions)
- Unicode primarily in user-facing strings (workflow names, descriptions)
- Expected overall token increase: **<5% on average**

---

## 3. Implementation Plan

### Phase 1: Modify TCT Tokenizer

#### Step 1.1: Backup Current TCT
```bash
cd ~/git/tct
git status  # Ensure clean working directory
git checkout -b fix/utf8-codepoint-encoding
```

#### Step 1.2: Edit primitives.rs
```bash
# Open in editor
vim src/tokenizer/primitives.rs

# Navigate to line 153
# Replace the String implementation with proposed version (see above)
```

#### Step 1.3: Build and Test TCT
```bash
# Build Rust library
cargo build --release

# Run TCT's unit tests
cargo test

# Run integration tests
cargo test --test '*'
```

Expected: All existing tests should pass (encoding/decoding still works, just different format)

### Phase 2: Retrain Dictionary and BPE

**⚠️ CRITICAL**: The base encoding change invalidates existing dictionary and BPE compression. Both must be retrained on workflows with the new character-level encoding.

#### Step 2.1: Prepare Training Corpus
```bash
cd ~/git/tct

# Use the workflow dataset to train dictionary and BPE
# This should be a representative sample of GitHub Actions workflows

# Option A: Use existing workflow collection
WORKFLOW_DIR=~/Desktop/data/workflows-100k/json/

# Option B: Use augmented workflows for better coverage
# WORKFLOW_DIR=~/Desktop/data/workflows-augmented-500k-v2/
```

#### Step 2.2: Train Dictionary
```bash
# Run dictionary training on workflow corpus
# This creates the frequency-based dictionary for common token sequences

cargo run --release --bin train-dict -- \
  --input $WORKFLOW_DIR \
  --output data/dict.bin \
  --dict-size 4096  # Adjust based on TCT config

# Verify dictionary created
ls -lh data/dict.bin
```

#### Step 2.3: Train BPE
```bash
# Run BPE training on workflow corpus with new base encoding
# This learns merge rules for the character-level tokens

cargo run --release --bin train-bpe -- \
  --input $WORKFLOW_DIR \
  --dict data/dict.bin \
  --output data/bpe.bin \
  --vocab-size 8192 \  # Target vocabulary size
  --num-merges 4096    # Number of BPE merge operations

# Verify BPE model created
ls -lh data/bpe.bin
```

#### Step 2.4: Update Tokenizer Config
```bash
# Update the tokenizer to use new dictionary and BPE models
# This may involve updating paths in config files or rebuild scripts

# Check current config location
grep -r "dict.bin\|bpe.bin" scripts/
```

**Note**: Dictionary and BPE training times:
- Dictionary: ~10-30 minutes on 100k workflows
- BPE: ~2 hours on 100k workflows
- Total: ~2-2.5 hours

**Expected output**:
- `data/dict.bin`: Frequency dictionary (character-level tokens → common sequences)
- `data/bpe.bin`: BPE merge rules (subword vocabulary)
- New vocabulary size: Still 8,192 (adjusted during BPE training)

### Phase 3: Build and Install Wheel

#### Step 3.1: Create Python Wheel
```bash
cd ~/git/tct
python3 scripts/build_wheel.py

# Output: tct_github_workflow-X.Y.Z-*.whl
# Verify file created:
ls -lh target/wheels/
```

#### Step 2.2: Install in nanochat-tct
```bash
cd ~/git/nanochat-tct

# Uninstall old version
pip uninstall -y tct_github_workflow

# Install new version
pip install ~/git/tct/target/wheels/tct_github_workflow-*.whl

# Verify installation
python3 -c "from tct_github_workflow import vocab_size; print(vocab_size())"
# Expected: 8192
```

### Phase 3: Regenerate Training Data

#### Step 3.1: Clean Old Data
```bash
# Remove old prepared data (incompatible format)
rm -rf ~/Desktop/data/prepared-*

# Keep original workflows (still valid)
# ~/Desktop/data/workflows-100k/json/ - KEEP
# ~/Desktop/data/workflows-augmented-500k-v2/ - KEEP
```

#### Step 3.2: Test Tokenization
```bash
cd ~/git/nanochat-tct/tct-bundle/examples

# Test basic round-trip
python3 example_quickstart.py
# Expected: ✅ Workflow encodes and decodes correctly

# Test Unicode handling
python3 -c "
from tct_github_workflow import encode_workflow, decode_workflow
import json

wf = {
    'name': 'Tëst Wörkflöw 🚀',
    'on': {'push': {}},
    'jobs': {'test': {'runs-on': 'ubuntu-latest', 'steps': []}}
}

tokens = encode_workflow(json.dumps(wf))
print(f'Tokens: {len(tokens)}')

decoded = decode_workflow(tokens)
print(f'Decoded name: {json.loads(decoded)[\"name\"]}')
"
# Expected: Name with Unicode characters preserved
```

#### Step 3.3: Prepare Training Data

**Decision point**: Use original 100k OR augmented 500k workflows?

**Option A: Original 100k workflows** (recommended first)
```bash
cd ~/git/nanochat-tct/tct-bundle/scripts

python3 prepare_training_data.py \
  --input ~/Desktop/data/workflows-100k/json/ \
  --output ~/Desktop/data/prepared-100k-utf8-fix/ \
  --context-size 1024 \
  --stride 32 \
  --train-split 0.9 \
  --timeout 3600000

# Expected: ~965k windows (same as before)
# Verify output:
python3 -c "import torch; data = torch.load('~/Desktop/data/prepared-100k-utf8-fix/train.pt'); print(f'Train: {data.shape}')"
```

**Option B: Augmented 500k workflows** (after verifying Option A works)
```bash
python3 prepare_training_data.py \
  --input ~/Desktop/data/workflows-augmented-500k-v2/ \
  --output ~/Desktop/data/prepared-500k-utf8-fix/ \
  --context-size 1024 \
  --stride 32 \
  --train-split 0.9 \
  --timeout 21600000  # 6 hours

# Expected: ~4.8M windows (5x more)
```

### Phase 4: Train Model

#### Step 4.1: Quick Smoke Test (10 iterations)
```bash
cd ~/git/nanochat-tct

# Test that training works with new encoding
python -m scripts.tct_train \
  --max_iters 10 \
  --device_batch_size 32 \
  --data_dir ~/Desktop/data/prepared-100k-utf8-fix/

# Check:
# - No errors ✅
# - Loss computed (not NaN) ✅
# - Model saves checkpoint ✅
```

#### Step 4.2: Full Training (Small Model)
```bash
torchrun --standalone --nproc_per_node=8 -m scripts.tct_train -- \
  --max_iters 50000 \
  --device_batch_size 32 \
  --data_dir ~/Desktop/data/prepared-100k-utf8-fix/ \
  --checkpoint_dir ~/Desktop/checkpoints/tct_small_utf8_fix/ \
  --eval_interval 500 \
  --save_interval 5000

# Expected duration: ~2.5 hours on 8×A100
# Expected best val loss: <1.3 (similar to baseline)
```

### Phase 5: Evaluation

#### Step 5.1: Test String Generation
```bash
cd ~/git/nanochat-tct

# Generate 50 workflows with best checkpoint
python -m scripts.generate_workflow \
  --checkpoint ~/Desktop/checkpoints/tct_small_utf8_fix/model_020000.pt \
  --num_samples 50 \
  --output_dir /tmp/test_utf8_fix/

# Check success rate
SUCCESS=$(ls /tmp/test_utf8_fix/*.json 2>/dev/null | wc -l)
echo "Success rate: $SUCCESS/50 ($(($SUCCESS * 100 / 50))%)"

# Expected: 47-50/50 (94-100% success rate)
```

#### Step 5.2: Verify UTF-8 Validity
```bash
# Check that all generated workflows have valid UTF-8
for f in /tmp/test_utf8_fix/*.json; do
  if ! python3 -c "import json; json.load(open('$f'))" 2>/dev/null; then
    echo "FAILED: $f"
  fi
done

# Expected: No failures (all valid JSON with valid UTF-8)
```

#### Step 5.3: Test Unicode Handling
```bash
# Interactive testing with Unicode strings
python -m scripts.generate_workflow \
  --checkpoint ~/Desktop/checkpoints/tct_small_utf8_fix/model_020000.pt \
  --interactive

# Try completing workflows with Unicode names:
# - "déployment workflow"
# - "tëst rünner 🚀"
# - Mixed ASCII and Unicode

# Verify: Model generates valid UTF-8 strings
```

#### Step 5.4: Compare to Baseline
```bash
# Baseline (byte-level encoding): 72% success (36/50)
# New (codepoint encoding): Expected 94-100% (47-50/50)

# Improvement: +22-28 percentage points
# Root cause eliminated: UTF-8 errors should be zero
```

---

## 4. Testing Strategy

### Unit Tests (TCT Level)

**Test 1: ASCII Strings**
```rust
#[test]
fn test_ascii_string_roundtrip() {
    let original = "hello world";
    let mut tokens = Vec::new();
    original.to_string().encode(&mut tokens);
    let decoded = String::decode(&mut tokens.into_iter()).unwrap();
    assert_eq!(original, decoded);
}
```

**Test 2: Unicode Strings**
```rust
#[test]
fn test_unicode_string_roundtrip() {
    let original = "Hëllö Wörld 🚀";
    let mut tokens = Vec::new();
    original.to_string().encode(&mut tokens);
    let decoded = String::decode(&mut tokens.into_iter()).unwrap();
    assert_eq!(original, decoded);
}
```

**Test 3: Invalid Codepoints**
```rust
#[test]
fn test_invalid_codepoint_rejected() {
    let invalid_codepoint = 0xD800; // UTF-16 surrogate
    let tokens = vec![1, invalid_codepoint];
    let result = String::decode(&mut tokens.into_iter());
    assert!(result.is_err());
}
```

### Integration Tests (Python Level)

**Test 1: Workflow Round-Trip**
```python
def test_workflow_with_unicode():
    from tct_github_workflow import encode_workflow, decode_workflow
    import json

    workflow = {
        "name": "Tëst Wörkflöw 🚀",
        "on": {"push": {}},
        "jobs": {
            "test": {
                "runs-on": "ubuntu-latest",
                "steps": [
                    {"name": "Chëckout cödë", "uses": "actions/checkout@v3"}
                ]
            }
        }
    }

    encoded = encode_workflow(json.dumps(workflow))
    decoded = decode_workflow(encoded)
    result = json.loads(decoded)

    assert result["name"] == "Tëst Wörkflöw 🚀"
    assert result["jobs"]["test"]["steps"][0]["name"] == "Chëckout cödë"
```

**Test 2: Invalid Codepoint Handling**
```python
def test_invalid_codepoint_in_token_stream():
    from tct_github_workflow import decode_workflow

    # Token stream with invalid codepoint (UTF-16 surrogate)
    invalid_tokens = [0, 3, 1, 0xD800, 100]  # Invalid codepoint in middle

    with pytest.raises(Exception):  # Should raise DecodeError
        decode_workflow(invalid_tokens)
```

### Model-Level Tests

**Test 1: Generate 100 Workflows**
```bash
python -m scripts.generate_workflow \
  --checkpoint ~/Desktop/checkpoints/tct_small_utf8_fix/model_020000.pt \
  --num_samples 100 \
  --output_dir /tmp/validation/

# Metrics:
# - Success rate (should be >95%)
# - JSON validity (all parseable)
# - UTF-8 validity (zero errors)
# - Schema compliance (all valid GitHub Actions)
```

**Test 2: Stress Test Unicode**
```python
# Generate workflows with high-Unicode prompts
prompts = [
    '{"name":"Dëplöy Äpplicatiön 🚀"',
    '{"name":"Tëst Süitë 🧪"',
    '{"name":"Bûild Pïpëlïnë 🏗️"',
]

for prompt in prompts:
    # Complete workflow
    # Verify: Valid UTF-8, no corruption
```

---

## 5. Rollback Plan

If the UTF-8 fix causes issues:

### Rollback Step 1: Reinstall Old TCT
```bash
cd ~/git/nanochat-tct
pip uninstall -y tct_github_workflow
pip install tct-bundle/wheels/tct_github_workflow-1.0.5-cp312-cp312-manylinux_2_34_x86_64.whl
```

### Rollback Step 2: Restore Old Data
```bash
# Old prepared data still exists (don't delete until validation passes)
# Use original: ~/Desktop/data/prepared-100k-1024-s32/
```

### Rollback Step 3: Use Baseline Checkpoint
```bash
# Baseline checkpoint still exists
# ~/Desktop/checkpoints/tct_small/model_020000.pt
```

---

## 6. Expected Outcomes

### Success Criteria

✅ **TCT builds and passes all tests**
✅ **Workflow round-trip works with Unicode**
✅ **Training data regenerates successfully**
✅ **Model trains without errors**
✅ **Generated workflows have >95% success rate**
✅ **Zero UTF-8 decoding errors**

### Performance Metrics

| Metric | Baseline (Byte-Level) | Target (Codepoint) |
|--------|----------------------|-------------------|
| Success Rate | 72% (36/50) | **95-100%** (47-50/50) |
| UTF-8 Errors | 28% (14/50) | **0%** (0/50) |
| Avg Tokens/Workflow | 609 | ~610-640 (+1-5%) |
| Training Time | 2h32m | ~2h30m (similar) |
| Validation Loss | 1.276 | <1.3 (similar) |

### Key Improvements

1. **Guaranteed UTF-8 validity**: Impossible to generate invalid sequences
2. **Better semantic learning**: Model learns characters, not bytes
3. **More efficient Unicode**: 'é' = 1 token vs 2 bytes
4. **Simpler error handling**: No UTF-8 decode failures

---

## 7. Alternative Solutions (Not Recommended)

### Alternative 1: Lenient Decoding

**Approach**: Replace `String::from_utf8()` with `String::from_utf8_lossy()`

```rust
// In primitives.rs decode function:
Ok(String::from_utf8_lossy(&bytes).into_owned())
```

**Pros**:
- Quick fix (1 line change)
- No retraining needed

**Cons**:
- Generates corrupted strings (� replacement characters)
- Doesn't fix root cause
- Poor user experience

**Verdict**: Not recommended - addresses symptom, not cause

### Alternative 2: BPE String Encoding

**Approach**: Apply Byte-Pair Encoding to string bytes (GPT-2/3 style)

**Pros**:
- Industry standard approach
- Most token-efficient for all text types

**Cons**:
- Requires BPE training pipeline
- Needs large corpus analysis
- Complex implementation
- Still possible to generate invalid UTF-8 (though less likely)

**Verdict**: Not recommended for now - complex, doesn't guarantee validity

---

## 8. Risk Assessment

### Low Risk
✅ Code changes localized to single file
✅ Existing tests validate correctness
✅ Rollback plan available
✅ No production dependencies

### Medium Risk
⚠️ Requires retraining (time investment)
⚠️ Slight token count increase (~5%)
⚠️ Breaking change for existing checkpoints

### Mitigation
- Keep baseline checkpoint for comparison
- Test thoroughly on small dataset first
- Monitor token counts during training
- Validate on Unicode-heavy workflows

---

## 9. Timeline

### Day 1: Implementation (2-3 hours)
- [ ] Modify primitives.rs (30 min)
- [ ] Build and test TCT (30 min)
- [ ] Train dictionary on workflows (30 min)
- [ ] Train BPE on workflows (2 hours)
- [ ] Create and install wheel (30 min)
- [ ] Test tokenization (30 min)
- [ ] Prepare training data (30-60 min)

### Day 2: Training (2.5 hours)
- [ ] Run model training (2.5 hours automated)

### Day 3: Evaluation (1 hour)
- [ ] Generate test workflows (15 min)
- [ ] Measure success rate (15 min)
- [ ] Compare to baseline (15 min)
- [ ] Document results (15 min)

**Total**: ~4-5 hours hands-on + 2.5 hours dict/BPE training + 2.5 hours model training = ~9 hours total

---

## 10. Decision Points

### Decision 1: Augmented Data?
**Question**: Use original 100k or augmented 500k workflows?

**Recommendation**: Start with 100k to isolate UTF-8 fix impact, then try 500k if results are good.

### Decision 2: Model Size?
**Question**: Small (20M) or Medium (100M) model?

**Recommendation**: Small first (faster iteration), upgrade to Medium if UTF-8 fix succeeds.

### Decision 3: Comprehensive Augmentation (v3)?
**Question**: Use v3 script with full string augmentation?

**Recommendation**: Not immediately necessary - UTF-8 fix is the priority. Augmentation can be added later if needed.

---

## 11. Success Validation

After implementation, verify success with:

```bash
# 1. Generate 50 workflows
python -m scripts.generate_workflow \
  --checkpoint ~/Desktop/checkpoints/tct_small_utf8_fix/model_020000.pt \
  --num_samples 50 \
  --output_dir /tmp/final_test/

# 2. Count successes
SUCCESS=$(ls /tmp/final_test/*.json 2>/dev/null | wc -l)
echo "Success rate: $SUCCESS/50"

# 3. Check for UTF-8 errors
grep -r "Utf8Error" /tmp/final_test/ && echo "FAILED: UTF-8 errors found" || echo "PASSED: No UTF-8 errors"

# 4. Validate all JSON
for f in /tmp/final_test/*.json; do
  python3 -m json.tool "$f" > /dev/null || echo "Invalid JSON: $f"
done

# Expected results:
# - Success rate: 47-50/50 (94-100%)
# - UTF-8 errors: 0
# - All JSON valid
```

If all checks pass → **UTF-8 fix is successful** ✅

---

## 12. Next Steps After Fix

Once UTF-8 fix is validated:

1. **Commit changes to TCT**:
   ```bash
   cd ~/git/tct
   git add src/tokenizer/primitives.rs
   git commit -m "Fix: Use Unicode codepoint encoding for strings

   - Changes string encoding from byte-level to codepoint-level
   - Guarantees valid UTF-8 via char::from_u32() validation
   - Eliminates 28% failure rate from invalid UTF-8 sequences
   - Slight token increase (~5%) for better semantic learning"
   ```

2. **Train larger model** (if desired):
   ```bash
   # Medium model (100M params) for production quality
   torchrun --standalone --nproc_per_node=8 -m scripts.tct_train -- \
     --model_size medium \
     --max_iters 100000
   ```

3. **Consider augmented data** (if >95% success rate achieved):
   ```bash
   # Retrain with 500k augmented workflows for even more diversity
   python3 tct-bundle/scripts/prepare_training_data.py \
     --input ~/Desktop/data/workflows-augmented-500k-v2/
   ```

4. **Update documentation**:
   - Update CLAUDE.md with UTF-8 fix details
   - Document new success rates
   - Add Unicode handling guidelines

---

## Conclusion

The UTF-8 fix addresses the root cause of 28% of workflow generation failures by guaranteeing valid UTF-8 through Unicode codepoint encoding. Implementation is straightforward (single file change), with clear testing and validation steps.

**Recommendation**: Proceed with implementation immediately - this is the highest-impact fix available.

**Expected outcome**: 72% → 95%+ success rate with zero UTF-8 errors.
