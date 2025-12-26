#!/bin/bash
# Upload data to NHR FAU clusters
#
# Usage:
#   bash scripts/upload_nhr.sh <username>                    # Upload all datasets
#   bash scripts/upload_nhr.sh <username> kubernetes         # Upload only kubernetes
#   bash scripts/upload_nhr.sh <username> --cluster=tinygpu  # Use TinyGPU instead of Alex
#   bash scripts/upload_nhr.sh <username> --dry-run          # Show what would be uploaded
#
# Datasets uploaded:
#   tsconfig:   tsconfig-tct-base, tsconfig-utf8-base-matched
#   eslintrc:   eslintrc-tct-bpe-500, eslintrc-utf8-bpe-500
#   kubernetes: kubernetes-tct-bpe, kubernetes-utf8-bpe

set -e

# =============================================================================
# Parse arguments
# =============================================================================

USERNAME=""
CLUSTER="alex"
DRY_RUN=""
SCHEMAS=""

for arg in "$@"; do
    case $arg in
        --cluster=*) CLUSTER="${arg#*=}" ;;
        --dry-run) DRY_RUN="1" ;;
        kubernetes|tsconfig|eslintrc) SCHEMAS="$SCHEMAS $arg" ;;
        -*) echo "Unknown option: $arg"; exit 1 ;;
        *) USERNAME="$arg" ;;
    esac
done

if [ -z "$USERNAME" ]; then
    echo "Usage: bash scripts/upload_nhr.sh <username> [schemas...] [options]"
    echo ""
    echo "Arguments:"
    echo "  username              NHR FAU username (e.g., ab12cdef)"
    echo "  schemas               Optional: kubernetes, tsconfig, eslintrc (default: all)"
    echo ""
    echo "Options:"
    echo "  --cluster=NAME        Cluster: alex (default), tinygpu"
    echo "  --dry-run             Show what would be uploaded without uploading"
    echo ""
    echo "Examples:"
    echo "  bash scripts/upload_nhr.sh ab12cdef                    # Upload all"
    echo "  bash scripts/upload_nhr.sh ab12cdef kubernetes         # Only kubernetes"
    echo "  bash scripts/upload_nhr.sh ab12cdef --dry-run          # Dry run"
    exit 1
fi

# Default to all schemas
if [ -z "$SCHEMAS" ]; then
    SCHEMAS="tsconfig eslintrc kubernetes"
fi
SCHEMAS="${SCHEMAS# }"  # Trim leading space

# =============================================================================
# Configuration
# =============================================================================

LOCAL_DATA="$HOME/Desktop/data"
REMOTE_HOST="${USERNAME}@${CLUSTER}.nhr.fau.de"
REMOTE_DATA="\$WORK/data/tct"

# Dataset directories per schema (must match schema_configs.py)
declare -A TCT_DIRS
declare -A UTF8_DIRS

TCT_DIRS[tsconfig]="tsconfig-tct-base"
UTF8_DIRS[tsconfig]="tsconfig-utf8-base-matched"

TCT_DIRS[eslintrc]="eslintrc-tct-bpe-500"
UTF8_DIRS[eslintrc]="eslintrc-utf8-bpe-500"

TCT_DIRS[kubernetes]="kubernetes-tct-bpe"
UTF8_DIRS[kubernetes]="kubernetes-utf8-bpe"

# =============================================================================
# Verify local data exists
# =============================================================================

echo "============================================================"
echo "NHR Data Upload"
echo "============================================================"
echo "Local data:  $LOCAL_DATA"
echo "Remote:      $REMOTE_HOST"
echo "Schemas:     $SCHEMAS"
echo "============================================================"
echo

DIRS_TO_UPLOAD=""
MISSING=""

for schema in $SCHEMAS; do
    tct_dir="${TCT_DIRS[$schema]}"
    utf8_dir="${UTF8_DIRS[$schema]}"

    if [ -d "$LOCAL_DATA/$tct_dir" ]; then
        DIRS_TO_UPLOAD="$DIRS_TO_UPLOAD $tct_dir"
        echo "[OK] $tct_dir"
    else
        MISSING="$MISSING $tct_dir"
        echo "[MISSING] $tct_dir"
    fi

    if [ -d "$LOCAL_DATA/$utf8_dir" ]; then
        DIRS_TO_UPLOAD="$DIRS_TO_UPLOAD $utf8_dir"
        echo "[OK] $utf8_dir"
    else
        MISSING="$MISSING $utf8_dir"
        echo "[MISSING] $utf8_dir"
    fi
done

echo

if [ -n "$MISSING" ]; then
    echo "WARNING: Some directories are missing:$MISSING"
    echo "Continue with available directories? [y/N]"
    read -r response
    if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
        echo "Aborted."
        exit 1
    fi
fi

if [ -z "$DIRS_TO_UPLOAD" ]; then
    echo "ERROR: No directories to upload"
    exit 1
fi

DIRS_TO_UPLOAD="${DIRS_TO_UPLOAD# }"  # Trim leading space

# =============================================================================
# Create remote directory
# =============================================================================

echo "Creating remote directory..."
if [ -z "$DRY_RUN" ]; then
    ssh "$REMOTE_HOST" "mkdir -p \$WORK/data/tct"
else
    echo "[DRY-RUN] ssh $REMOTE_HOST mkdir -p \$WORK/data/tct"
fi

# =============================================================================
# Upload datasets
# =============================================================================

echo
echo "Uploading datasets..."
echo

for dir in $DIRS_TO_UPLOAD; do
    echo ">>> $dir"
    if [ -z "$DRY_RUN" ]; then
        rsync -avz --progress "$LOCAL_DATA/$dir/" "$REMOTE_HOST:\$WORK/data/tct/$dir/"
    else
        echo "[DRY-RUN] rsync -avz --progress $LOCAL_DATA/$dir/ $REMOTE_HOST:\$WORK/data/tct/$dir/"
    fi
    echo
done

# =============================================================================
# Upload TCT wheels
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
TCT_WHEELS="$CODE_DIR/tct-wheels"

if [ -d "$TCT_WHEELS" ]; then
    echo ">>> tct-wheels"
    if [ -z "$DRY_RUN" ]; then
        rsync -avz --progress "$TCT_WHEELS/" "$REMOTE_HOST:\$WORK/tct-wheels/"
    else
        echo "[DRY-RUN] rsync -avz --progress $TCT_WHEELS/ $REMOTE_HOST:\$WORK/tct-wheels/"
    fi
    echo
else
    echo "NOTE: tct-wheels not found at $TCT_WHEELS (skipping)"
fi

# =============================================================================
# Summary
# =============================================================================

echo "============================================================"
echo "Upload Complete"
echo "============================================================"
echo
echo "Uploaded to: $REMOTE_HOST:\$WORK/data/tct/"
echo
echo "Next steps:"
echo "  1. SSH to cluster:  ssh $REMOTE_HOST"
echo "  2. Start interactive job:"
echo "     srun --partition=a100 --gres=gpu:a100:1 --time=01:00:00 --pty bash -l"
echo "  3. Run setup:"
echo "     cd \$WORK && git clone https://github.com/jonas-schmitt/nanochat-tct.git"
echo "     cd nanochat-tct && bash scripts/setup.sh"
echo "  4. Submit training:"
echo "     bash scripts/submit.sh kubernetes small medium --gpu=a100"
echo
