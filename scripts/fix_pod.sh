#!/bin/bash
# Fix pod datasets and restart training
# Run: bash scripts/fix_pod.sh

set -e

DATA_DIR="${DATA_DIR:-/workspace/data}"
CODE_DIR="${CODE_DIR:-/workspace/nanochat-tct}"

cd "$DATA_DIR"

echo "============================================================"
echo "Fixing Pod Datasets"
echo "============================================================"
echo

# Remove old datasets
echo "[1/4] Removing old datasets..."
rm -rf eslintrc-tct-bpe-10k eslintrc-utf8-bpe-10k
rm -rf tsconfig-tct-bpe-10k tsconfig-utf8-bpe-10k
rm -rf eslintrc-tct-base
echo "Done."

# Check datasets exist or extract from tarballs
echo "[2/4] Setting up new datasets..."

for dataset in eslintrc-tct-bpe-500 eslintrc-utf8-bpe-500 tsconfig-utf8-base-matched; do
    if [ -d "$dataset" ]; then
        echo "  - $dataset (already exists)"
    elif [ -d "/workspace/$dataset" ]; then
        echo "  - $dataset (copying from /workspace)..."
        cp -r "/workspace/$dataset" "$dataset"
    elif [ -f "$dataset.tar.gz" ]; then
        echo "  - $dataset (extracting from local tarball)..."
        tar -xzf "$dataset.tar.gz" && rm "$dataset.tar.gz"
    elif [ -f "/workspace/$dataset.tar.gz" ]; then
        echo "  - $dataset (extracting from /workspace tarball)..."
        tar -xzf "/workspace/$dataset.tar.gz"
    else
        echo "  - $dataset (MISSING!)"
        exit 1
    fi
done

echo "Done."

# Remove incomplete checkpoints
echo "[3/4] Removing incomplete checkpoints..."
cd "$CODE_DIR"
rm -rf checkpoints/eslintrc_tct_small
rm -rf checkpoints/eslintrc_utf8_small
rm -rf checkpoints/tsconfig_utf8_small
echo "Done."

# Update code
echo "[4/4] Updating code..."
git pull
echo "Done."

# Verify
echo
echo "============================================================"
echo "Verification"
echo "============================================================"
echo "Datasets in $DATA_DIR:"
ls -la "$DATA_DIR"
echo
echo "Ready to run:"
echo "  bash scripts/run_all.sh"
echo
