#!/bin/bash -l
# Evaluate all finished schema/size combinations
#
# Usage:
#   bash scripts/eval_all.sh                    # Evaluate all schemas, skip existing results
#   bash scripts/eval_all.sh --schema=kubernetes  # Evaluate only kubernetes
#   bash scripts/eval_all.sh --force            # Re-run even if results exist
#   bash scripts/eval_all.sh --samples=1000     # Custom sample count
#   bash scripts/eval_all.sh --dry-run          # Show what would be evaluated

set -e

# =============================================================================
# Parse arguments
# =============================================================================

NUM_SAMPLES=""
NUM_GEN_SAMPLES=""
DRY_RUN=false
FORCE=false
SCHEMA_FILTER=""
EXTRA_ARGS=""

for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN=true ;;
        --force) FORCE=true ;;
        --schema=*) SCHEMA_FILTER="${arg#--schema=}" ;;
        --samples=*) NUM_SAMPLES="$arg" ;;
        --gen_samples=*) NUM_GEN_SAMPLES="$arg" ;;
        *) EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

# =============================================================================
# Auto-detect paths (same as eval.sh)
# =============================================================================

if [ -z "$CODE_DIR" ]; then
    CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

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

echo "============================================================"
echo "Evaluate All Finished Checkpoints"
echo "============================================================"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Results dir:    $RESULTS_DIR"
echo "============================================================"
echo

# =============================================================================
# Find all valid schema/size combinations
# =============================================================================

# Use filtered schema if provided, otherwise all schemas
if [ -n "$SCHEMA_FILTER" ]; then
    SCHEMAS="$SCHEMA_FILTER"
    echo "Schema filter:  $SCHEMA_FILTER"
else
    SCHEMAS="kubernetes tsconfig eslintrc"
fi
SIZES="tiny mini base small small-wide medium large"
BASELINES="default o200k"

found_any=false

for schema in $SCHEMAS; do
    for size in $SIZES; do
        for baseline in $BASELINES; do
            # Build checkpoint paths based on baseline
            if [ "$baseline" = "o200k" ]; then
                tct_dir="${CHECKPOINT_DIR}/${schema}_tct_o200k_${size}"
                utf8_dir="${CHECKPOINT_DIR}/${schema}_utf8_o200k_${size}"
                baseline_arg="o200k"
                output_file="$RESULTS_DIR/${schema}_o200k_${size}_eval.json"
            else
                tct_dir="${CHECKPOINT_DIR}/${schema}_tct_${size}"
                utf8_dir="${CHECKPOINT_DIR}/${schema}_utf8_${size}"
                baseline_arg=""
                output_file="$RESULTS_DIR/${schema}_${size}_eval.json"
            fi

            # Check if both checkpoints exist
            if [ -d "$tct_dir" ] && [ -d "$utf8_dir" ]; then
                # Check if checkpoint files exist (epoch_*.pt or best.pt)
                tct_has_checkpoint=false
                utf8_has_checkpoint=false

                if ls "$tct_dir"/epoch_*.pt >/dev/null 2>&1 || [ -f "$tct_dir/best.pt" ]; then
                    tct_has_checkpoint=true
                fi
                if ls "$utf8_dir"/epoch_*.pt >/dev/null 2>&1 || [ -f "$utf8_dir/best.pt" ]; then
                    utf8_has_checkpoint=true
                fi

                if [ "$tct_has_checkpoint" = true ] && [ "$utf8_has_checkpoint" = true ]; then
                    # Check if training completed (config.json exists and has epoch info)
                    tct_epoch=$(python3 -c "import json; print(json.load(open('$tct_dir/config.json')).get('epoch', 0))" 2>/dev/null || echo 0)
                    utf8_epoch=$(python3 -c "import json; print(json.load(open('$utf8_dir/config.json')).get('epoch', 0))" 2>/dev/null || echo 0)

                    # Require at least some epochs completed (allow 105+ for overfitting cases)
                    if [ "$tct_epoch" -lt 100 ] || [ "$utf8_epoch" -lt 100 ]; then
                        echo "[WAIT] $schema $size ${baseline_arg:-default} - incomplete (TCT: $tct_epoch, UTF8: $utf8_epoch)"
                        continue
                    fi

                    found_any=true

                    # Check if already evaluated (skip unless --force)
                    if [ -f "$output_file" ] && [ "$FORCE" = false ]; then
                        echo "[SKIP] $schema $size ${baseline_arg:-default} - already evaluated (use --force to re-run)"
                        continue
                    fi

                    echo "[EVAL] $schema $size ${baseline_arg:-default}"

                    if [ "$DRY_RUN" = true ]; then
                        echo "       Would run: bash scripts/eval.sh $schema $size $baseline_arg $NUM_SAMPLES $NUM_GEN_SAMPLES $EXTRA_ARGS"
                    else
                        # Build command
                        cmd="bash $CODE_DIR/scripts/eval.sh $schema $size"
                        [ -n "$baseline_arg" ] && cmd="$cmd $baseline_arg"
                        [ -n "$NUM_SAMPLES" ] && cmd="$cmd $NUM_SAMPLES"
                        [ -n "$NUM_GEN_SAMPLES" ] && cmd="$cmd $NUM_GEN_SAMPLES"
                        [ -n "$EXTRA_ARGS" ] && cmd="$cmd $EXTRA_ARGS"

                        echo "       Running: $cmd"
                        echo
                        eval $cmd
                        echo
                        echo "------------------------------------------------------------"
                        echo
                    fi
                fi
            fi
        done
    done
done

if [ "$found_any" = false ]; then
    echo "No finished checkpoint pairs found."
    echo ""
    echo "Looking for patterns like:"
    echo "  ${CHECKPOINT_DIR}/{schema}_tct_{size}/best.pt"
    echo "  ${CHECKPOINT_DIR}/{schema}_utf8_{size}/best.pt"
    echo ""
    echo "Available checkpoints:"
    ls -d "$CHECKPOINT_DIR"/*/ 2>/dev/null | head -20 || echo "  (none)"
    exit 1
fi

echo "============================================================"
echo "All evaluations complete!"
echo "============================================================"
