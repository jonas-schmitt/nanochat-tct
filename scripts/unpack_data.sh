#!/bin/bash
# Unpack data archives to the data directory
#
# Usage:
#   bash scripts/unpack_data.sh              # Unpack all archives to $DATA_DIR
#   bash scripts/unpack_data.sh kubernetes   # Unpack only kubernetes datasets
#
# Data archives are in data/*.tar.xz within the repo.
# Extracts to:
#   - RunPod: /workspace/data/
#   - NHR: $WORK/data/
#   - Local: ~/Desktop/data/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
ARCHIVE_DIR="$REPO_DIR/data"

# Auto-detect platform and set DATA_DIR
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    DATA_DIR="/workspace/data"
elif [ -n "$WORK" ] && [ -d "$WORK" ]; then
    DATA_DIR="$WORK/data"
else
    DATA_DIR="$HOME/Desktop/data"
fi

# Allow override via environment
DATA_DIR="${DATA_DIR:-$DATA_DIR}"

echo "============================================================"
echo "Unpack TCT Data Archives"
echo "============================================================"
echo "Archive source: $ARCHIVE_DIR"
echo "Target dir:     $DATA_DIR"
echo "============================================================"
echo

mkdir -p "$DATA_DIR"

# Parse arguments for schema filter
FILTER=""
for arg in "$@"; do
    case $arg in
        kubernetes|tsconfig|eslintrc) FILTER="$FILTER $arg" ;;
    esac
done
FILTER="${FILTER# }"

# Find and unpack archives
unpacked=0
for archive in "$ARCHIVE_DIR"/*.tar.xz; do
    [ -f "$archive" ] || continue

    name=$(basename "$archive" .tar.xz)

    # Apply filter if specified
    if [ -n "$FILTER" ]; then
        match=""
        for schema in $FILTER; do
            case $name in
                ${schema}-*) match="1" ;;
            esac
        done
        [ -z "$match" ] && continue
    fi

    # Check if already extracted
    if [ -d "$DATA_DIR/$name" ]; then
        echo "[SKIP] $name (already exists)"
        continue
    fi

    echo "[UNPACK] $name..."
    tar -xJf "$archive" -C "$DATA_DIR"
    unpacked=$((unpacked + 1))
    echo "         -> $DATA_DIR/$name/"
done

echo
if [ $unpacked -eq 0 ]; then
    echo "No new archives unpacked (all already extracted or filtered out)"
else
    echo "Unpacked $unpacked archive(s)"
fi

echo
echo "Contents of $DATA_DIR:"
ls -la "$DATA_DIR"
