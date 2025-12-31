#!/bin/bash -l
# Unified Run Script for TCT Experiments
# Works with all schemas: kubernetes, tsconfig, eslintrc
#
# Usage:
#   bash scripts/run.sh kubernetes              # Run all kubernetes experiments (small, medium, large)
#   bash scripts/run.sh kubernetes tsconfig     # Run both schemas
#   bash scripts/run.sh kubernetes small        # Run only small models
#   bash scripts/run.sh kubernetes small-deep   # Run small-deep (must be explicit)
#   bash scripts/run.sh kubernetes tct          # Run only TCT tokenizer
#   bash scripts/run.sh kubernetes resume       # Resume from latest checkpoint
#   bash scripts/run.sh kubernetes tsconfig small medium resume  # Combine options

set -e

# =============================================================================
# Parse arguments
# =============================================================================

SCHEMAS=""
FILTER_TOKENIZER=""
FILTER_SIZES=""
RESUME_MODE=""
DROPOUT=""
LR_SCHEDULE=""
EFF_BATCH=""
GRAD_CKPT=""

for arg in "$@"; do
    case $arg in
        kubernetes|tsconfig|eslintrc) SCHEMAS="$SCHEMAS $arg" ;;
        tct|utf8) FILTER_TOKENIZER="$arg" ;;
        small|medium|large) FILTER_SIZES="$FILTER_SIZES $arg" ;;
        resume) RESUME_MODE="1" ;;
        --dropout=*) DROPOUT="${arg#--dropout=}" ;;
        dropout=*) DROPOUT="${arg#dropout=}" ;;
        --lr_schedule=*) LR_SCHEDULE="${arg#--lr_schedule=}" ;;
        constant) LR_SCHEDULE="constant" ;;
        --eff_batch=*) EFF_BATCH="${arg#--eff_batch=}" ;;
        --batch=*) EFF_BATCH="${arg#--batch=}" ;;
        --gradient_checkpointing|--grad_ckpt) GRAD_CKPT="True" ;;
    esac
done

# Trim leading spaces
SCHEMAS="${SCHEMAS# }"
FILTER_SIZES="${FILTER_SIZES# }"

if [ -z "$SCHEMAS" ]; then
    echo "Usage: bash scripts/run.sh <schema>... [size]... [tokenizer] [resume] [options]"
    echo ""
    echo "Schemas: kubernetes, tsconfig, eslintrc"
    echo "Sizes: small, medium, large"
    echo "Tokenizers: tct, utf8"
    echo "Options:"
    echo "  resume              Resume from latest checkpoint"
    echo "  --dropout=0.1       Set dropout (default: 0.1)"
    echo "  --lr_schedule=X     LR schedule: cosine (default) or constant"
    echo "  constant            Shorthand for --lr_schedule=constant"
    echo "  --eff_batch=N       Effective batch size (default: 64)"
    echo "  --grad_ckpt         Enable gradient checkpointing (saves memory, ~4% slower)"
    echo ""
    echo "Examples:"
    echo "  bash scripts/run.sh kubernetes"
    echo "  bash scripts/run.sh kubernetes tsconfig eslintrc    # Run all schemas"
    echo "  bash scripts/run.sh kubernetes small tct"
    echo "  bash scripts/run.sh kubernetes tsconfig resume"
    echo "  bash scripts/run.sh kubernetes --dropout=0.2        # Higher dropout"
    echo "  bash scripts/run.sh kubernetes constant             # No LR decay"
    exit 1
fi

# =============================================================================
# Auto-detect paths
# =============================================================================

if [ -z "$CODE_DIR" ]; then
    if [ -d "/workspace/nanochat-tct" ]; then
        CODE_DIR="/workspace/nanochat-tct"
        DATA_DIR="/workspace/data"
    elif [ -n "$WORK" ]; then
        CODE_DIR="$WORK/nanochat-tct"
        DATA_DIR="$WORK/data"  # Sibling of nanochat-tct (not /tct subdir)
    else
        CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
        DATA_DIR="$HOME/Desktop/data"
    fi
fi

LOG_DIR="${LOG_DIR:-$CODE_DIR/logs}"

# GPU info (batch sizes are computed dynamically in Python based on detected VRAM)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "No GPU")
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
GPU_MEM=${GPU_MEM:-24000}
GPU_MEM=$((GPU_MEM / 1000))  # Convert to GB
echo "GPU: $GPU_NAME (${GPU_MEM}GB) - batch sizes auto-scaled"

# =============================================================================
# Schema-specific settings
# =============================================================================

get_epochs() {
    local schema=$1
    case $schema in
        tsconfig)    echo 50 ;;    # Converges fast
        eslintrc)    echo 75 ;;    # Medium dataset
        kubernetes)  echo 100 ;;   # Larger dataset
        *)           echo 100 ;;   # Default
    esac
}

DEFAULT_SIZES="small medium large"
TOKENIZERS="${TOKENIZERS:-tct utf8}"

# =============================================================================
# Helper functions
# =============================================================================

find_latest_epoch() {
    local exp_name=$1
    local checkpoint_dir="checkpoints/${exp_name}"
    if [ -d "$checkpoint_dir" ]; then
        ls "$checkpoint_dir"/epoch_*.pt 2>/dev/null | sort -V | tail -1 | grep -oP 'epoch_\K\d+' | sed 's/^0*//'
    fi
}

# =============================================================================
# Run experiments
# =============================================================================

mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

# Use filter sizes if provided, otherwise defaults
SIZES="${FILTER_SIZES:-$DEFAULT_SIZES}"

echo "============================================================"
echo "TCT Experiments"
echo "============================================================"
echo "Date: $(date)"
echo "GPU: $GPU_NAME (${GPU_MEM}GB)"
echo "Data: $DATA_DIR"
echo "Schemas: $SCHEMAS"
echo "Sizes: $SIZES"
echo "Tokenizers: ${FILTER_TOKENIZER:-$TOKENIZERS}"
[ -n "$DROPOUT" ] && echo "Dropout: $DROPOUT"
[ -n "$LR_SCHEDULE" ] && echo "LR Schedule: $LR_SCHEDULE"
[ -n "$EFF_BATCH" ] && echo "Effective batch: $EFF_BATCH"
[ -n "$RESUME_MODE" ] && echo "Mode: RESUME"
echo "============================================================"
echo

for SCHEMA in $SCHEMAS; do
    EPOCHS=$(get_epochs "$SCHEMA")

    echo "============================================================"
    echo "$SCHEMA ($EPOCHS epochs)"
    echo "============================================================"

    for size in $SIZES; do
        for tokenizer in $TOKENIZERS; do
            [ -n "$FILTER_TOKENIZER" ] && [ "$tokenizer" != "$FILTER_TOKENIZER" ] && continue

            exp_name="${SCHEMA}_${tokenizer}_${size}"
            [ -n "$DROPOUT" ] && exp_name="${exp_name}_drop${DROPOUT}"
            [ "$LR_SCHEDULE" = "constant" ] && exp_name="${exp_name}_constlr"
            [ -n "$EFF_BATCH" ] && exp_name="${exp_name}_batch${EFF_BATCH}"
            log_file="$LOG_DIR/${exp_name}.log"

            # Skip if already completed
            if [ -f "checkpoints/${exp_name}/best.pt" ]; then
                echo "[SKIP] $exp_name (already completed)"
                continue
            fi

            # Check for resume
            RESUME_ARG=""
            if [ -n "$RESUME_MODE" ]; then
                latest_epoch=$(find_latest_epoch "$exp_name")
                if [ -n "$latest_epoch" ]; then
                    echo "[RESUME] $exp_name from epoch $latest_epoch"
                    RESUME_ARG="--resume_from_epoch=$latest_epoch"
                    # Add separator to log file
                    echo "" >> "$log_file"
                    echo "============================================================" >> "$log_file"
                    echo "RESUMING from epoch $latest_epoch at $(date)" >> "$log_file"
                    echo "============================================================" >> "$log_file"
                fi
            fi

            echo "[START] $exp_name at $(date)"

            EXTRA_ARGS=""
            [ -n "$DROPOUT" ] && EXTRA_ARGS="$EXTRA_ARGS --dropout=$DROPOUT"
            [ -n "$LR_SCHEDULE" ] && EXTRA_ARGS="$EXTRA_ARGS --lr_schedule=$LR_SCHEDULE"
            [ -n "$EFF_BATCH" ] && EXTRA_ARGS="$EXTRA_ARGS --eff_batch=$EFF_BATCH"
            [ -n "$GRAD_CKPT" ] && EXTRA_ARGS="$EXTRA_ARGS --gradient_checkpointing=$GRAD_CKPT"
            # Use custom model_tag if we have custom settings
            [ -n "$DROPOUT" ] || [ -n "$LR_SCHEDULE" ] || [ -n "$EFF_BATCH" ] && EXTRA_ARGS="$EXTRA_ARGS --model_tag=$exp_name"

            python -m scripts.train_unified \
                --schema="$SCHEMA" \
                --tokenizer="$tokenizer" \
                --model_size="$size" \
                --data_root="$DATA_DIR" \
                $RESUME_ARG \
                $EXTRA_ARGS \
                2>&1 | tee -a "$log_file"

            echo "[DONE] $exp_name at $(date)"
            echo
        done
    done
done

echo "============================================================"
echo "All experiments completed at $(date)"
echo "============================================================"
