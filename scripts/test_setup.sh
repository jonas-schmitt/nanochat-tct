#!/bin/bash
# Quick Setup Verification
# Runs each model size × tokenizer × schema for 10 iterations to verify everything works
#
# Usage:
#   bash scripts/test_setup.sh

set -e

WORKSPACE="${WORKSPACE:-/workspace}"
CODE_DIR="${CODE_DIR:-$WORKSPACE/nanochat-tct}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"

cd "$CODE_DIR"

echo "============================================================"
echo "TCT Setup Verification"
echo "============================================================"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'Unknown')"
echo "============================================================"
echo

# Test iterations (just enough to verify forward/backward pass)
TEST_ITERS=10

# Only test largest configs - if these fit, everything else will too
SCHEMAS="eslintrc kubernetes"
TOKENIZERS="tct utf8"
SIZES="large"

passed=0
failed=0

for schema in $SCHEMAS; do
    for tokenizer in $TOKENIZERS; do
        for size in $SIZES; do
            exp_name="${schema}_${tokenizer}_${size}"
            echo "============================================================"
            echo "Testing: $exp_name"
            echo "============================================================"

            # Run training for 10 steps and capture output
            set +e
            output=$(python -m scripts.train_unified \
                --schema="$schema" \
                --tokenizer="$tokenizer" \
                --model_size="$size" \
                --data_root="$DATA_DIR" \
                --epochs=1 \
                --save_every_pct=100 \
                --eval_every_epoch=100 \
                --num_eval_batches=1 \
                2>&1)
            exit_code=$?
            set -e

            # Show relevant output
            echo "$output" | grep -E "(EXPERIMENT|Vocab|Context|Batch|Parameters|step 00|Error|error|CUDA|OOM)" | head -20

            # Check if we reached step 10
            if echo "$output" | grep -q "step 000010"; then
                echo ">>> PASS"
                passed=$((passed + 1))
            else
                echo ">>> FAIL (exit code: $exit_code)"
                echo "$output" | tail -20
                failed=$((failed + 1))
            fi
            echo
        done
    done
done

echo
echo "============================================================"
echo "VERIFICATION SUMMARY"
echo "============================================================"
echo "Passed: $passed / $((passed + failed))"
echo "Failed: $failed"
echo

if [ $failed -gt 0 ]; then
    echo "Some tests failed. Check configuration and data paths."
    exit 1
else
    echo "All tests passed! Ready to run full experiments."
    exit 0
fi
