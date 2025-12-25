#!/bin/bash
# Prepare data package for RunPod upload
# Creates archives with training data, wheels, and merge tables
# Code will be cloned from GitHub on the pod

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${HOME}/Desktop/data"
OUTPUT_DIR="${PROJECT_DIR}/runpod-upload"

echo "=== Preparing RunPod Upload Package ==="
echo "Project dir: $PROJECT_DIR"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define data directories to include
DATA_DIRS=(
    "kubernetes-tct-bpe"
    "kubernetes-utf8-bpe"
    "eslintrc-tct-bpe-10k"
    "eslintrc-utf8-bpe-10k"
    "tsconfig-tct-bpe-10k"
    "tsconfig-utf8-bpe-10k"
)

echo "Creating data archive..."
cd "$DATA_DIR"

# Create tarball of data directories
tar -cvzf "$OUTPUT_DIR/tct-experiment-data.tar.gz" "${DATA_DIRS[@]}"
echo "Created: $OUTPUT_DIR/tct-experiment-data.tar.gz"
echo "Size: $(du -sh "$OUTPUT_DIR/tct-experiment-data.tar.gz" | cut -f1)"
echo

echo "Copying TCT wheels..."
cp -r "$PROJECT_DIR/tct-wheels" "$OUTPUT_DIR/"
echo "Size: $(du -sh "$OUTPUT_DIR/tct-wheels" | cut -f1)"
echo

echo "Copying BPE merge tables..."
cp -r "$PROJECT_DIR/bpe-merges" "$OUTPUT_DIR/"
echo "Size: $(du -sh "$OUTPUT_DIR/bpe-merges" | cut -f1)"
echo

echo "=== Upload Package Summary ==="
echo
du -sh "$OUTPUT_DIR"/*
echo
echo "Total size:"
du -sh "$OUTPUT_DIR"
echo
echo "=== Next Steps ==="
echo "1. Upload to RunPod network volume"
echo "2. Code will be cloned from: https://github.com/jonas-schmitt/nanochat-tct"
echo
