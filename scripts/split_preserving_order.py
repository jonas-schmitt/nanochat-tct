#!/usr/bin/env python3
"""
Split all.jsonl into train/validate while preserving line order.

Unlike reshuffle_split.py, this does NOT shuffle - it takes the first N%
for training and the rest for validation. This preserves the original
line-to-source-file mapping.

Usage:
    python scripts/split_preserving_order.py ~/Desktop/data/tsconfig-tct-bpe --split 0.95
"""

import argparse
import json
from pathlib import Path


def split_preserving_order(data_dir: Path, train_ratio: float = 0.95):
    all_file = data_dir / "all.jsonl"
    train_file = data_dir / "train.jsonl"
    validate_file = data_dir / "validate.jsonl"
    metadata_file = data_dir / "metadata.json"

    if not all_file.exists():
        print(f"Error: {all_file} not found")
        return False

    # Read all lines
    print(f"Reading {all_file}...")
    with open(all_file, 'r') as f:
        lines = f.readlines()

    total = len(lines)
    print(f"Total sequences: {total:,}")

    # Split sequentially (NO shuffle - preserves order)
    split_idx = int(total * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    print(f"Train: {len(train_lines):,} ({100*len(train_lines)/total:.1f}%) [lines 0-{split_idx-1}]")
    print(f"Validate: {len(val_lines):,} ({100*len(val_lines)/total:.1f}%) [lines {split_idx}-{total-1}]")

    # Backup old files
    if train_file.exists():
        backup = train_file.with_suffix('.jsonl.bak')
        if not backup.exists():  # Don't overwrite existing backup
            train_file.rename(backup)
            print(f"Backed up {train_file.name} -> {backup.name}")
        else:
            print(f"Backup already exists, overwriting {train_file.name}")

    if validate_file.exists():
        backup = validate_file.with_suffix('.jsonl.bak')
        if not backup.exists():
            validate_file.rename(backup)
            print(f"Backed up {validate_file.name} -> {backup.name}")
        else:
            print(f"Backup already exists, overwriting {validate_file.name}")

    # Write new files
    print(f"Writing {train_file}...")
    with open(train_file, 'w') as f:
        f.writelines(train_lines)

    print(f"Writing {validate_file}...")
    with open(validate_file, 'w') as f:
        f.writelines(val_lines)

    # Update metadata
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        metadata["train_count"] = len(train_lines)
        metadata["validate_count"] = len(val_lines)
        metadata["train_ratio"] = train_ratio
        metadata["validate_ratio"] = 1.0 - train_ratio
        metadata["split_method"] = "sequential"  # Mark that order is preserved

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Updated {metadata_file.name}")

    print("Done! Line order preserved - source file mapping still valid.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Split all.jsonl into train/validate preserving order"
    )
    parser.add_argument("data_dir", type=Path, help="Directory containing all.jsonl")
    parser.add_argument("--split", type=float, default=0.95, help="Train ratio (default: 0.95)")
    args = parser.parse_args()

    split_preserving_order(args.data_dir, args.split)


if __name__ == "__main__":
    main()
