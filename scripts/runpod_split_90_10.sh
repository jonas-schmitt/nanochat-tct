#!/bin/bash
# Apply 90/10 split to all datasets on RunPod
# Run this after git pull on the pod

set -e

WORKSPACE="${WORKSPACE:-/workspace}"
DATA_DIR="${DATA_DIR:-$WORKSPACE/data}"
CODE_DIR="${CODE_DIR:-$WORKSPACE/nanochat-tct}"

cd "$CODE_DIR"

echo "============================================================"
echo "Applying 90/10 split to all datasets"
echo "============================================================"
echo

# All BPE datasets
for dir in "$DATA_DIR"/*-tct-bpe "$DATA_DIR"/*-utf8-bpe; do
    if [ -d "$dir" ] && [ -f "$dir/all.jsonl" ]; then
        echo "Processing: $(basename $dir)"
        python scripts/split_preserving_order.py "$dir" --split 0.90
        echo
    fi
done

# Base datasets (no BPE)
for dir in "$DATA_DIR"/*-tct-base; do
    if [ -d "$dir" ] && [ -f "$dir/all.jsonl" ]; then
        echo "Processing: $(basename $dir)"
        python scripts/split_preserving_order.py "$dir" --split 0.90
        echo
    fi
done

echo "============================================================"
echo "Done! All datasets now have 90/10 split"
echo "============================================================"
