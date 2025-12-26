#!/bin/bash
# Fix pod datasets and restart training
# Run: bash scripts/fix_pod.sh

set -e

DATA_DIR="${DATA_DIR:-/workspace/data}"
CODE_DIR="${CODE_DIR:-/workspace/nanochat-tct}"
S3_BUCKET="s3://m4rt30bdh2"
S3_OPTS="--region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io"

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

# Download new datasets
echo "[2/4] Downloading new datasets..."

echo "  - eslintrc-tct-bpe-500..."
aws s3 cp $S3_BUCKET/eslintrc-tct-bpe-500.tar.gz . $S3_OPTS
tar -xzf eslintrc-tct-bpe-500.tar.gz && rm eslintrc-tct-bpe-500.tar.gz

echo "  - eslintrc-utf8-bpe-500..."
aws s3 cp $S3_BUCKET/eslintrc-utf8-bpe-500.tar.gz . $S3_OPTS
tar -xzf eslintrc-utf8-bpe-500.tar.gz && rm eslintrc-utf8-bpe-500.tar.gz

echo "  - tsconfig-utf8-base-matched..."
aws s3 cp $S3_BUCKET/tsconfig-utf8-base-matched.tar.gz . $S3_OPTS
tar -xzf tsconfig-utf8-base-matched.tar.gz && rm tsconfig-utf8-base-matched.tar.gz

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
