#!/bin/bash
# Unpack data archive(s) and move to ../data/ directory
#
# Usage: bash scripts/unpack_data.sh                    # All .tar.gz in ../
#        bash scripts/unpack_data.sh ../archive.tar.gz  # Single archive
#        bash scripts/unpack_data.sh /path/to/dir       # All .tar.gz in dir

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Target data directory (in parent of repo)
DATA_DIR="$(dirname "$REPO_DIR")/data"

# Input: file or directory (default: parent of repo)
INPUT="${1:-$(dirname "$REPO_DIR")}"

echo "Target data dir: $DATA_DIR"
mkdir -p "$DATA_DIR"

if [ -f "$INPUT" ]; then
    # Single archive file
    echo ">>> Unpacking $(basename "$INPUT")..."
    tar -xzf "$INPUT" -C "$DATA_DIR"
    echo "    Done"
elif [ -d "$INPUT" ]; then
    # Directory with multiple archives
    echo "Archive source: $INPUT"
    echo
    for archive in "$INPUT"/*.tar.gz; do
        [ -f "$archive" ] || continue
        name=$(basename "$archive" .tar.gz)
        echo ">>> Unpacking $name..."
        tar -xzf "$archive" -C "$DATA_DIR"
        echo "    Done: $DATA_DIR/$name/"
    done
else
    echo "Error: $INPUT is not a file or directory"
    exit 1
fi

echo
echo "Contents of $DATA_DIR:"
ls -la "$DATA_DIR"
