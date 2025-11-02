# TCT-Nanochat Training Progress

**Status**: Phase 3 - Full Training In Progress

---

## Phase 1: Setup & Verification ✅

### 1.1 Environment Setup ✅
- [x] Python 3.12 virtual environment created
- [x] Nanochat dependencies installed (PyTorch, wandb, datasets, etc.)
- [x] TCT tokenizer wheel installed (`tct_github_workflow`)
- [x] Vocabulary size verified (8192 tokens)
- [x] Round-trip encoding/decoding tested

### 1.2 TCT Verification ✅
- [x] Verified TCT examples (`example_quickstart.py`, `example_windowed.py`)
- [x] Confirmed 100k workflow files available

### 1.3 Sample Data Preparation ✅
- [x] Prepared sample dataset (100 workflows for PoC)
- [x] Verified data shape and format

---

## Phase 2: Integration & PoC ✅
- [x] Created `scripts/tct_train.py` with TCT integration
- [x] Smoke test (10 iterations) - all checks passed
- [x] PoC training (100 workflows, 5k iterations) - 100% valid workflow generation

---

## Phase 3: Full Training - Small Model ⏳

**Configuration**:
- Model: Small (20M params, 384d×8L×6H)
- Data: 100k workflows → 965k windows (train: 772k, val: 193k)
- Context: 1024 tokens, stride=32, vocab=8192
- Training: 50k iterations, batch_size=32

### 3.1 Data Preparation ✅
- [x] Prepared dataset with stride=32 (`~/Desktop/data/prepared-100k-1024-s32/`)
- [x] Verified constraints (position tokens < 8192, shapes correct)

### 3.2 Model Configuration ✅
- [x] Selected Small model for baseline experiment
- [x] Updated training script with context_size=1024

### 3.3 Training ⏳

**Status**: Running (started 2025-11-02 18:19, bash_id: b646b4)
- Progress: Step 5,130/50,000 (~10%)
- Loss: 9.01 → 1.31 (85% reduction)
- Speed: ~167ms/step, ~195k tokens/sec
- Checkpoints: Every 5k steps → `~/Desktop/checkpoints/tct_small/`

**Milestones**:
- [x] Step 5,000: First checkpoint saved ✅
- [ ] Step 10,000: Second checkpoint
- [ ] Step 25,000: Mid-training
- [ ] Step 50,000: Final model (~2.5h total)

---

## Phase 4: Evaluation

### Post-Training Tasks
- [ ] Generate 100 workflows from final checkpoint
- [ ] Evaluate JSON validity rate (target: >90%)
- [ ] Check schema compliance (required fields present)
- [ ] Calculate validation perplexity (target: <10)
- [ ] Manual inspection of sample outputs

### Future Experiments
- [ ] Medium model (100M params, 8h training)
- [ ] Longer training (100k iterations)
- [ ] Hyperparameter tuning

---

## Notes

### Configuration Rationale
- **context_size=1024**: Covers 98th percentile of workflow lengths (avg ~180 tokens)
- **stride=32**: Keeps position tokens within vocab (max position 5,301 < 8,192)
- **vocab_size=8192**: TCT base vocabulary, no expansion needed

See `CLAUDE.md` for troubleshooting and detailed integration guide.
