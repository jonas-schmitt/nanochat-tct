#!/usr/bin/env python3
"""
Reshuffle all.jsonl into train/validate split.

Usage:
    python scripts/reshuffle_split.py ~/Desktop/data/tsconfig-tct-bpe --split 0.95
    python scripts/reshuffle_split.py ~/Desktop/data/tsconfig-utf8-bpe --split 0.95
"""

import argparse
import random
from pathlib import Path


def reshuffle_split(data_dir: Path, train_ratio: float = 0.95, seed: int = 42):
    all_file = data_dir / "all.jsonl"
    train_file = data_dir / "train.jsonl"
    validate_file = data_dir / "validate.jsonl"

    if not all_file.exists():
        print(f"Error: {all_file} not found")
        return False

    # Read all lines
    print(f"Reading {all_file}...")
    with open(all_file, 'r') as f:
        lines = f.readlines()

    total = len(lines)
    print(f"Total sequences: {total:,}")

    # Shuffle
    random.seed(seed)
    random.shuffle(lines)

    # Split
    split_idx = int(total * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    print(f"Train: {len(train_lines):,} ({100*len(train_lines)/total:.1f}%)")
    print(f"Validate: {len(val_lines):,} ({100*len(val_lines)/total:.1f}%)")

    # Backup old files
    if train_file.exists():
        backup = train_file.with_suffix('.jsonl.bak')
        train_file.rename(backup)
        print(f"Backed up {train_file} -> {backup}")

    if validate_file.exists():
        backup = validate_file.with_suffix('.jsonl.bak')
        validate_file.rename(backup)
        print(f"Backed up {validate_file} -> {backup}")

    # Write new files
    print(f"Writing {train_file}...")
    with open(train_file, 'w') as f:
        f.writelines(train_lines)

    print(f"Writing {validate_file}...")
    with open(validate_file, 'w') as f:
        f.writelines(val_lines)

    print("Done!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Reshuffle all.jsonl into train/validate split")
    parser.add_argument("data_dir", type=Path, help="Directory containing all.jsonl")
    parser.add_argument("--split", type=float, default=0.95, help="Train ratio (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    reshuffle_split(args.data_dir, args.split, args.seed)


if __name__ == "__main__":
    main()
