#!/bin/bash
# Upload data to NHR FAU clusters
#
# Usage:
#   bash scripts/upload_nhr.sh <username>                        # Upload all datasets
#   bash scripts/upload_nhr.sh <username> kubernetes             # Upload only kubernetes
#   bash scripts/upload_nhr.sh <username> --data=/path/to/data   # Custom data path
#   bash scripts/upload_nhr.sh <username> --cluster=tinygpu      # Use TinyGPU instead of Alex
#   bash scripts/upload_nhr.sh <username> --dry-run              # Show what would be uploaded
#   bash scripts/upload_nhr.sh <username> --verbose              # Verbose output
#
# Datasets uploaded:
#   tsconfig:   tsconfig-tct-base, tsconfig-utf8-base-matched
#   eslintrc:   eslintrc-tct-bpe-500, eslintrc-utf8-bpe-500
#   kubernetes: kubernetes-tct-bpe, kubernetes-utf8-bpe

set -e

# =============================================================================
# Helper functions
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_verbose() { [ -n "$VERBOSE" ] && echo -e "${BLUE}[DEBUG]${NC} $1" || true; }

die() {
    log_error "$1"
    [ -n "$2" ] && echo "  Hint: $2"
    exit 1
}

# =============================================================================
# Parse arguments
# =============================================================================

USERNAME=""
CLUSTER="alex"
DRY_RUN=""
VERBOSE=""
SCHEMAS=""
LOCAL_DATA=""

for arg in "$@"; do
    case $arg in
        --cluster=*) CLUSTER="${arg#*=}" ;;
        --data=*) LOCAL_DATA="${arg#*=}" ;;
        --dry-run) DRY_RUN="1" ;;
        --verbose|-v) VERBOSE="1" ;;
        kubernetes|tsconfig|eslintrc) SCHEMAS="$SCHEMAS $arg" ;;
        --help|-h)
            echo "Usage: bash scripts/upload_nhr.sh <username> [schemas...] [options]"
            echo ""
            echo "Arguments:"
            echo "  username              NHR FAU username (e.g., ab12cdef)"
            echo "  schemas               Optional: kubernetes, tsconfig, eslintrc (default: all)"
            echo ""
            echo "Options:"
            echo "  --data=PATH           Path to data directory (default: ~/Desktop/data)"
            echo "  --cluster=NAME        Cluster: alex (default), tinygpu"
            echo "  --dry-run             Show what would be uploaded without uploading"
            echo "  --verbose, -v         Verbose output for debugging"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash scripts/upload_nhr.sh ab12cdef                        # Upload all"
            echo "  bash scripts/upload_nhr.sh ab12cdef kubernetes             # Only kubernetes"
            echo "  bash scripts/upload_nhr.sh ab12cdef --data=/mnt/data       # Custom path"
            echo "  bash scripts/upload_nhr.sh ab12cdef --dry-run --verbose    # Dry run with debug"
            exit 0
            ;;
        -*) die "Unknown option: $arg" "Use --help to see available options" ;;
        *) USERNAME="$arg" ;;
    esac
done

if [ -z "$USERNAME" ]; then
    die "Missing username" "Usage: bash scripts/upload_nhr.sh <username> [options]"
fi

# Default to all schemas
if [ -z "$SCHEMAS" ]; then
    SCHEMAS="tsconfig eslintrc kubernetes"
fi
SCHEMAS="${SCHEMAS# }"  # Trim leading space

# Default data path
if [ -z "$LOCAL_DATA" ]; then
    LOCAL_DATA="$HOME/Desktop/data"
fi

log_verbose "Local data path: $LOCAL_DATA"
log_verbose "Username: $USERNAME"
log_verbose "Cluster: $CLUSTER"
log_verbose "Schemas: $SCHEMAS"

# Verify data path exists
if [ ! -d "$LOCAL_DATA" ]; then
    die "Data directory not found: $LOCAL_DATA" "Use --data=/path/to/data to specify location"
fi

# =============================================================================
# Configuration
# =============================================================================

REMOTE_HOST="${USERNAME}@${CLUSTER}.nhr.fau.de"
log_verbose "Remote host: $REMOTE_HOST"

# Validate cluster name
case $CLUSTER in
    alex|tinygpu) ;;
    *) die "Invalid cluster: $CLUSTER" "Valid options: alex, tinygpu" ;;
esac
REMOTE_DATA="\$WORK/data"

# Dataset directories per schema (must match schema_configs.py)
declare -A TCT_DIRS
declare -A UTF8_DIRS

TCT_DIRS[tsconfig]="tsconfig-tct-base"
UTF8_DIRS[tsconfig]="tsconfig-utf8-base-matched"

TCT_DIRS[eslintrc]="eslintrc-tct-bpe-500"
UTF8_DIRS[eslintrc]="eslintrc-utf8-bpe-500"

TCT_DIRS[kubernetes]="kubernetes-tct-bpe-1k"
UTF8_DIRS[kubernetes]="kubernetes-utf8-bpe-1k"

# =============================================================================
# Verify local data exists
# =============================================================================

echo "============================================================"
echo "NHR Data Upload"
echo "============================================================"
echo "Local data:  $LOCAL_DATA"
echo "Remote:      $REMOTE_HOST"
echo "Schemas:     $SCHEMAS"
[ -n "$DRY_RUN" ] && echo "Mode:        DRY RUN"
[ -n "$VERBOSE" ] && echo "Verbose:     enabled"
echo "============================================================"
echo

log_info "Checking local directories..."

DIRS_TO_UPLOAD=""
MISSING=""

for schema in $SCHEMAS; do
    tct_dir="${TCT_DIRS[$schema]}"
    utf8_dir="${UTF8_DIRS[$schema]}"

    log_verbose "Checking $schema: TCT=$tct_dir, UTF8=$utf8_dir"

    if [ -d "$LOCAL_DATA/$tct_dir" ]; then
        DIRS_TO_UPLOAD="$DIRS_TO_UPLOAD $tct_dir"
        log_success "$tct_dir"
        log_verbose "  Path: $LOCAL_DATA/$tct_dir"
        log_verbose "  Size: $(du -sh "$LOCAL_DATA/$tct_dir" 2>/dev/null | cut -f1)"
    else
        MISSING="$MISSING $tct_dir"
        log_warn "$tct_dir (not found)"
    fi

    if [ -d "$LOCAL_DATA/$utf8_dir" ]; then
        DIRS_TO_UPLOAD="$DIRS_TO_UPLOAD $utf8_dir"
        log_success "$utf8_dir"
        log_verbose "  Path: $LOCAL_DATA/$utf8_dir"
        log_verbose "  Size: $(du -sh "$LOCAL_DATA/$utf8_dir" 2>/dev/null | cut -f1)"
    else
        MISSING="$MISSING $utf8_dir"
        log_warn "$utf8_dir (not found)"
    fi
done

echo

if [ -n "$MISSING" ]; then
    log_warn "Some directories are missing:$MISSING"
    echo ""
    echo "Continue with available directories? [y/N]"
    read -r response
    if [ "$response" != "y" ] && [ "$response" != "Y" ]; then
        log_info "Aborted by user."
        exit 1
    fi
fi

if [ -z "$DIRS_TO_UPLOAD" ]; then
    die "No directories to upload" "Check that data exists in $LOCAL_DATA"
fi

DIRS_TO_UPLOAD="${DIRS_TO_UPLOAD# }"  # Trim leading space

# =============================================================================
# Test SSH connection
# =============================================================================

log_info "Testing SSH connection to $REMOTE_HOST..."
log_verbose "Running: ssh -o ConnectTimeout=10 $REMOTE_HOST echo OK"

if [ -z "$DRY_RUN" ]; then
    if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$REMOTE_HOST" "echo OK" >/dev/null 2>&1; then
        die "Cannot connect to $REMOTE_HOST" "Check SSH key: ssh-copy-id $REMOTE_HOST"
    fi
    log_success "SSH connection OK"
else
    log_info "[DRY-RUN] Would test SSH connection"
fi

# =============================================================================
# Create remote directory
# =============================================================================

log_info "Creating remote directory..."
log_verbose "Running: ssh $REMOTE_HOST mkdir -p \$WORK/data/tct"

if [ -z "$DRY_RUN" ]; then
    if ! ssh "$REMOTE_HOST" "mkdir -p \$WORK/data/tct"; then
        die "Failed to create remote directory" "Check that \$WORK is accessible on $CLUSTER"
    fi
    log_success "Remote directory ready"
else
    log_info "[DRY-RUN] ssh $REMOTE_HOST mkdir -p \$WORK/data/tct"
fi

# =============================================================================
# Upload datasets
# =============================================================================

echo
log_info "Uploading datasets..."
echo

UPLOAD_FAILED=""
for dir in $DIRS_TO_UPLOAD; do
    echo ">>> $dir"
    log_verbose "Source: $LOCAL_DATA/$dir/"
    log_verbose "Dest:   $REMOTE_HOST:\$WORK/data/tct/$dir/"

    if [ -z "$DRY_RUN" ]; then
        if rsync -avz --progress "$LOCAL_DATA/$dir/" "$REMOTE_HOST:\$WORK/data/tct/$dir/"; then
            log_success "$dir uploaded"
        else
            log_error "$dir upload failed"
            UPLOAD_FAILED="$UPLOAD_FAILED $dir"
        fi
    else
        log_info "[DRY-RUN] rsync -avz --progress $LOCAL_DATA/$dir/ $REMOTE_HOST:\$WORK/data/tct/$dir/"
    fi
    echo
done

if [ -n "$UPLOAD_FAILED" ]; then
    log_warn "Some uploads failed:$UPLOAD_FAILED"
fi

# =============================================================================
# Upload TCT wheels
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
TCT_WHEELS="$CODE_DIR/tct-wheels"

log_verbose "Looking for TCT wheels at: $TCT_WHEELS"

if [ -d "$TCT_WHEELS" ]; then
    echo ">>> tct-wheels"
    log_verbose "Found wheels: $(ls "$TCT_WHEELS"/*.whl 2>/dev/null | wc -l) files"

    if [ -z "$DRY_RUN" ]; then
        if rsync -avz --progress "$TCT_WHEELS/" "$REMOTE_HOST:\$WORK/tct-wheels/"; then
            log_success "tct-wheels uploaded"
        else
            log_error "tct-wheels upload failed"
        fi
    else
        log_info "[DRY-RUN] rsync -avz --progress $TCT_WHEELS/ $REMOTE_HOST:\$WORK/tct-wheels/"
    fi
    echo
else
    log_warn "tct-wheels not found at $TCT_WHEELS (skipping)"
    log_verbose "You may need to copy wheels manually or build on cluster"
fi

# =============================================================================
# Summary
# =============================================================================

echo "============================================================"
if [ -n "$UPLOAD_FAILED" ]; then
    log_warn "Upload completed with errors"
else
    log_success "Upload Complete"
fi
echo "============================================================"
echo
echo "Uploaded to: $REMOTE_HOST:\$WORK/data/tct/"
echo
echo "Next steps:"
echo "  1. SSH to cluster:"
echo "     ssh $REMOTE_HOST"
echo ""
echo "  2. Start interactive job:"
echo "     srun --partition=a100 --gres=gpu:a100:1 --time=01:00:00 --pty bash -l"
echo ""
echo "  3. Run setup:"
echo "     cd \$WORK"
echo "     git clone https://github.com/jonas-schmitt/nanochat-tct.git"
echo "     cd nanochat-tct && bash scripts/setup.sh"
echo ""
echo "  4. Exit interactive job and submit training:"
echo "     exit"
echo "     cd \$WORK/nanochat-tct"
echo "     bash scripts/submit.sh kubernetes small medium --gpu=a100"
echo
