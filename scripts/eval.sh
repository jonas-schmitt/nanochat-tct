#!/bin/bash -l
# Unified Evaluation Script for TCT vs UTF8+XGrammar
# Works with all schemas: kubernetes, tsconfig, eslintrc
#
# Usage:
#   bash scripts/eval.sh kubernetes              # Evaluate kubernetes (auto-detect size)
#   bash scripts/eval.sh kubernetes mini         # Evaluate specific size
#   bash scripts/eval.sh kubernetes --samples=1000  # Custom sample count
#   bash scripts/eval.sh kubernetes mini o200k   # Use o200k-matched baseline

set -e

# =============================================================================
# Parse arguments
# =============================================================================

SCHEMA=""
SIZE=""
NUM_SAMPLES=""
NUM_GEN_SAMPLES=""
BASELINE=""
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        kubernetes|tsconfig|eslintrc) SCHEMA="$arg" ;;
        tiny|mini|base|small|small-wide|medium|large) SIZE="$arg" ;;
        --samples=*) NUM_SAMPLES="${arg#--samples=}" ;;
        --gen_samples=*) NUM_GEN_SAMPLES="${arg#--gen_samples=}" ;;
        o200k|o200k-matched) BASELINE="o200k-matched" ;;
        --baseline=*) BASELINE="${arg#--baseline=}" ;;
        --bpb_only|--generation_only) EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

if [ -z "$SCHEMA" ]; then
    echo "Usage: bash scripts/eval.sh <schema> [size] [options]"
    echo ""
    echo "Schemas: kubernetes, tsconfig, eslintrc"
    echo "Sizes: tiny, mini, base, small, small-wide, medium, large (auto-detect if omitted)"
    echo "Options:"
    echo "  --samples=N       BPB samples (default: full validation set)"
    echo "  --gen_samples=N   Generation samples (default: 10000)"
    echo "  o200k             Use o200k-matched baseline"
    echo ""
    echo "Examples:"
    echo "  bash scripts/eval.sh kubernetes              # Auto-detect size, full validation"
    echo "  bash scripts/eval.sh kubernetes mini         # Specific size"
    echo "  bash scripts/eval.sh kubernetes --samples=100 --gen_samples=100  # Quick test"
    echo "  bash scripts/eval.sh kubernetes mini o200k   # o200k baseline"
    exit 1
fi

# =============================================================================
# Auto-detect paths (same as run.sh)
# =============================================================================

# CODE_DIR: use environment if set, otherwise compute from script location
if [ -z "$CODE_DIR" ]; then
    CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

# DATA_DIR: use environment if set, otherwise sibling of CODE_DIR
if [ -z "$DATA_DIR" ]; then
    DATA_DIR="$(dirname "$CODE_DIR")/data"
fi

# CHECKPOINT_DIR: use environment if set, otherwise platform-specific
# NHR uses $HPCVAULT (500GB) or $WORK (1TB) to avoid home quota (50GB)
if [ -z "$CHECKPOINT_DIR" ]; then
    if [ -n "$HPCVAULT" ] && [ -d "$HPCVAULT" ]; then
        CHECKPOINT_DIR="$HPCVAULT/checkpoints"
    elif [ -n "$WORK" ] && [ -d "$WORK" ]; then
        CHECKPOINT_DIR="$WORK/checkpoints"
    else
        CHECKPOINT_DIR="$CODE_DIR/checkpoints"
    fi
fi

RESULTS_DIR="${RESULTS_DIR:-$CODE_DIR/results}"
mkdir -p "$RESULTS_DIR"

# =============================================================================
# Find checkpoints
# =============================================================================

# Auto-detect size if not specified
if [ -z "$SIZE" ]; then
    # Try sizes in order of preference
    for try_size in mini small base medium large tiny; do
        if [ -n "$BASELINE" ] && [ "$BASELINE" = "o200k-matched" ]; then
            tct_try="${CHECKPOINT_DIR}/${SCHEMA}_tct_o200k_${try_size}"
            utf8_try="${CHECKPOINT_DIR}/${SCHEMA}_utf8_o200k_${try_size}"
        else
            tct_try="${CHECKPOINT_DIR}/${SCHEMA}_tct_${try_size}"
            utf8_try="${CHECKPOINT_DIR}/${SCHEMA}_utf8_${try_size}"
        fi
        if [ -d "$tct_try" ] && [ -d "$utf8_try" ]; then
            SIZE="$try_size"
            break
        fi
    done
    if [ -z "$SIZE" ]; then
        echo "ERROR: No matching checkpoints found for schema '$SCHEMA'"
        echo "Looking in: $CHECKPOINT_DIR"
        echo ""
        echo "Available checkpoints:"
        ls -d "$CHECKPOINT_DIR"/${SCHEMA}_* 2>/dev/null || echo "  (none)"
        exit 1
    fi
    echo "Auto-detected size: $SIZE"
fi

# Build checkpoint paths
if [ -n "$BASELINE" ] && [ "$BASELINE" = "o200k-matched" ]; then
    TCT_CHECKPOINT="${CHECKPOINT_DIR}/${SCHEMA}_tct_o200k_${SIZE}"
    UTF8_CHECKPOINT="${CHECKPOINT_DIR}/${SCHEMA}_utf8_o200k_${SIZE}"
else
    TCT_CHECKPOINT="${CHECKPOINT_DIR}/${SCHEMA}_tct_${SIZE}"
    UTF8_CHECKPOINT="${CHECKPOINT_DIR}/${SCHEMA}_utf8_${SIZE}"
fi

# Validate checkpoints exist
if [ ! -d "$TCT_CHECKPOINT" ]; then
    echo "ERROR: TCT checkpoint not found: $TCT_CHECKPOINT"
    exit 1
fi
if [ ! -d "$UTF8_CHECKPOINT" ]; then
    echo "ERROR: UTF8 checkpoint not found: $UTF8_CHECKPOINT"
    exit 1
fi

# =============================================================================
# Run evaluation
# =============================================================================

# Build output filename
if [ -n "$BASELINE" ] && [ "$BASELINE" = "o200k-matched" ]; then
    OUTPUT_FILE="$RESULTS_DIR/${SCHEMA}_o200k_${SIZE}_eval.json"
else
    OUTPUT_FILE="$RESULTS_DIR/${SCHEMA}_${SIZE}_eval.json"
fi

echo "============================================================"
echo "TCT vs UTF8+XGrammar Evaluation"
echo "============================================================"
echo "Date:       $(date)"
echo "Schema:     $SCHEMA"
echo "Size:       $SIZE"
echo "Baseline:   ${BASELINE:-bpe-matched}"
echo "TCT:        $TCT_CHECKPOINT"
echo "UTF8:       $UTF8_CHECKPOINT"
echo "Data:       $DATA_DIR"
echo "BPB:        ${NUM_SAMPLES:-all} samples"
echo "Generation: ${NUM_GEN_SAMPLES:-10000} samples"
echo "Output:     $OUTPUT_FILE"
echo "============================================================"
echo

cd "$CODE_DIR"

# Build command with optional arguments
CMD="python -m scripts.eval_icml --schema $SCHEMA --tct_checkpoint $TCT_CHECKPOINT --utf8_checkpoint $UTF8_CHECKPOINT"
[ -n "$NUM_SAMPLES" ] && CMD="$CMD --num_samples $NUM_SAMPLES"
[ -n "$NUM_GEN_SAMPLES" ] && CMD="$CMD --num_gen_samples $NUM_GEN_SAMPLES"
[ -n "$EXTRA_ARGS" ] && CMD="$CMD $EXTRA_ARGS"
CMD="$CMD --output $OUTPUT_FILE --latex"

eval $CMD

echo ""
echo "============================================================"
echo "Results saved to: $OUTPUT_FILE"
echo "============================================================"
