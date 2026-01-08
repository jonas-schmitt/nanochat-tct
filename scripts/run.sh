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
EPOCHS_OVERRIDE=""
COMPILE_MODE=""

for arg in "$@"; do
    case $arg in
        kubernetes|tsconfig|eslintrc) SCHEMAS="$SCHEMAS $arg" ;;
        tct|utf8) FILTER_TOKENIZER="$arg" ;;
        tiny|mini|base|small|small-wide|medium|large) FILTER_SIZES="$FILTER_SIZES $arg" ;;
        resume) RESUME_MODE="1" ;;
        --dropout=*) DROPOUT="${arg#--dropout=}" ;;
        dropout=*) DROPOUT="${arg#dropout=}" ;;
        --lr_schedule=*) LR_SCHEDULE="${arg#--lr_schedule=}" ;;
        constant) LR_SCHEDULE="constant" ;;
        --eff_batch=*) EFF_BATCH="${arg#--eff_batch=}" ;;
        --batch=*) EFF_BATCH="${arg#--batch=}" ;;
        --gradient_checkpointing|--grad_ckpt) GRAD_CKPT="True" ;;
        --epochs=*) EPOCHS_OVERRIDE="${arg#--epochs=}" ;;
        epochs=*) EPOCHS_OVERRIDE="${arg#epochs=}" ;;
        --compile_mode=*) COMPILE_MODE="${arg#--compile_mode=}" ;;
        --max-autotune) COMPILE_MODE="max-autotune" ;;
    esac
done

# Trim leading spaces
SCHEMAS="${SCHEMAS# }"
FILTER_SIZES="${FILTER_SIZES# }"

if [ -z "$SCHEMAS" ]; then
    echo "Usage: bash scripts/run.sh <schema>... [size]... [tokenizer] [resume] [options]"
    echo ""
    echo "Schemas: kubernetes, tsconfig, eslintrc"
    echo "Sizes: tiny, mini, base, small, small-wide, medium, large"
    echo "Tokenizers: tct, utf8"
    echo "Options:"
    echo "  resume              Resume from latest checkpoint"
    echo "  --epochs=N          Override max epochs (default: schema-specific)"
    echo "  --dropout=X         Override dropout (default: model-dependent)"
    echo "  --lr_schedule=X     LR schedule: cosine (default) or constant"
    echo "  constant            Shorthand for --lr_schedule=constant"
    echo "  --eff_batch=N       Effective batch size (default: 64)"
    echo "  --grad_ckpt         Enable gradient checkpointing (saves memory, ~4% slower)"
    echo "  --max-autotune      Use torch.compile max-autotune (slower compile, faster run)"
    echo "  --use_muon=False    Disable Muon optimizer (default: True, uses Muon+AdamW)"
    echo "  --scale_lr_by_batch Scale AdamW LRs by sqrt(batch/524K) (default: False)"
    echo ""
    echo "Examples:"
    echo "  bash scripts/run.sh kubernetes"
    echo "  bash scripts/run.sh kubernetes tsconfig eslintrc    # Run all schemas"
    echo "  bash scripts/run.sh kubernetes small tct"
    echo "  bash scripts/run.sh kubernetes tsconfig resume"
    echo "  bash scripts/run.sh kubernetes --epochs=50          # Shorter training"
    echo "  bash scripts/run.sh kubernetes tiny mini base       # Run smaller models"
    echo "  bash scripts/run.sh kubernetes constant             # No LR decay"
    exit 1
fi

# =============================================================================
# Auto-detect paths - CODE_DIR and DATA_DIR can be set by environment
# =============================================================================

# CODE_DIR: use environment if set, otherwise compute from script location
if [ -z "$CODE_DIR" ]; then
    CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

# DATA_DIR: use environment if set, otherwise sibling of CODE_DIR
if [ -z "$DATA_DIR" ]; then
    DATA_DIR="$(dirname "$CODE_DIR")/data"
fi

LOG_DIR="${LOG_DIR:-$CODE_DIR/logs}"

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
export CHECKPOINT_DIR
mkdir -p "$CHECKPOINT_DIR"
echo "Checkpoint dir: $CHECKPOINT_DIR"

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
    # Uniform 100 epochs for all schemas (simpler, allows fair comparison)
    # best.pt captures peak performance via early stopping
    echo 100
}

# Default sizes for full experiment suite (can override with explicit size args)
# tiny/mini/base recommended for limited data, small/medium/large for larger datasets
DEFAULT_SIZES="small medium large"
TOKENIZERS="${TOKENIZERS:-tct utf8}"

# =============================================================================
# Helper functions
# =============================================================================

find_latest_epoch() {
    local exp_name=$1
    local ckpt_dir="${CHECKPOINT_DIR}/${exp_name}"
    if [ -d "$ckpt_dir" ]; then
        ls "$ckpt_dir"/epoch_*.pt 2>/dev/null | sort -V | tail -1 | grep -oP 'epoch_\K\d+' | sed 's/^0*//'
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
[ -n "$EPOCHS_OVERRIDE" ] && echo "Epochs: $EPOCHS_OVERRIDE (override)"
[ -n "$DROPOUT" ] && echo "Dropout: $DROPOUT (override)"
[ -n "$LR_SCHEDULE" ] && echo "LR Schedule: $LR_SCHEDULE"
[ -n "$EFF_BATCH" ] && echo "Effective batch: $EFF_BATCH"
[ -n "$RESUME_MODE" ] && echo "Mode: RESUME"
echo "============================================================"
echo

for SCHEMA in $SCHEMAS; do
    # Use override if provided, otherwise schema default
    if [ -n "$EPOCHS_OVERRIDE" ]; then
        EPOCHS="$EPOCHS_OVERRIDE"
    else
        EPOCHS=$(get_epochs "$SCHEMA")
    fi

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
            if [ -f "${CHECKPOINT_DIR}/${exp_name}/best.pt" ]; then
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
            [ -n "$COMPILE_MODE" ] && EXTRA_ARGS="$EXTRA_ARGS --compile_mode=$COMPILE_MODE"
            # Use custom model_tag if we have custom settings
            [ -n "$DROPOUT" ] || [ -n "$LR_SCHEDULE" ] || [ -n "$EFF_BATCH" ] && EXTRA_ARGS="$EXTRA_ARGS --model_tag=$exp_name"

            python -m scripts.train_unified \
                --schema="$SCHEMA" \
                --tokenizer="$tokenizer" \
                --model_size="$size" \
                --epochs="$EPOCHS" \
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
