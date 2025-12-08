#!/usr/bin/env python3
"""
Train BPE tokenizer until average sequence length matches TCT.

Uses rustbpe's built-in stopping criterion feature.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import rustbpe


def yaml_iterator(directory: str, limit: int | None = None):
    """Iterate over YAML files, yielding file contents."""
    files = sorted(Path(directory).glob("*.yaml"))
    if limit:
        files = files[:limit]

    total = len(files)
    for i, path in enumerate(files):
        if (i + 1) % 50000 == 0:
            print(f"  Loading {i + 1}/{total} files...", file=sys.stderr)
        yield path.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Train BPE until avg sequence length matches target"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/Desktop/data/k8s-yaml"),
        help="Directory with YAML manifests",
    )
    parser.add_argument(
        "--target-avg-length",
        type=float,
        default=523.0,
        help="Target average sequence length (default: 523, TCT's average)",
    )
    parser.add_argument(
        "--max-vocab-size",
        type=int,
        default=200000,
        help="Maximum vocabulary size (default: 200000)",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=5000,
        help="Check stopping criterion every N merges (default: 5000)",
    )
    parser.add_argument(
        "--output-dir",
        default="tokenizers/bpe-k8s-matched",
        help="Output directory for trained tokenizer",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files (for testing)",
    )
    args = parser.parse_args()

    # Enable logging from rustbpe
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"Training BPE to match target avg length: {args.target_avg_length}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Max vocab size: {args.max_vocab_size}")
    print(f"  Check interval: {args.check_interval}")
    print()

    # Count files
    yaml_files = list(Path(args.data_dir).glob("*.yaml"))
    if args.limit:
        yaml_files = yaml_files[:args.limit]
    print(f"Found {len(yaml_files):,} YAML files")
    print()

    # Train with stopping criterion
    print(f"Training BPE with target avg length {args.target_avg_length}...")
    tokenizer = rustbpe.Tokenizer()
    tokenizer.train_from_iterator(
        yaml_iterator(args.data_dir, args.limit),
        vocab_size=args.max_vocab_size,
        pattern="",  # Pure byte-level BPE
        target_avg_length=args.target_avg_length,
        check_interval=args.check_interval,
    )

    # Get final vocab size
    final_vocab_size = tokenizer.vocab_size()
    print(f"\nFinal vocabulary size: {final_vocab_size:,}")

    # Save tokenizer
    os.makedirs(args.output_dir, exist_ok=True)

    ranks = tokenizer.get_mergeable_ranks()
    ranks_serializable = [[list(token), rank] for token, rank in ranks]

    with open(os.path.join(args.output_dir, "mergeable_ranks.json"), "w") as f:
        json.dump(ranks_serializable, f)

    # Save metadata
    metadata = {
        "vocab_size": final_vocab_size,
        "target_avg_length": args.target_avg_length,
        "max_vocab_size": args.max_vocab_size,
        "num_files": len(yaml_files),
        "type": "rustbpe_pure_byte",
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved to {args.output_dir}")

    # Test encoding on single sample
    sample = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: test\n"
    tokens = tokenizer.encode(sample)
    print(f"\nTest encoding:")
    print(f"  Input bytes: {len(sample.encode())}")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Compression: {len(sample.encode())/len(tokens):.2f}x")

    # VERIFY: Encode actual YAML files and check avg length
    print(f"\n=== VERIFICATION: Actual encoding on {min(1000, len(yaml_files))} files ===")
    sample_files = yaml_files[:1000]  # Sample for verification
    actual_lengths = []
    byte_lengths = []

    for yf in sample_files:
        text = yf.read_text(encoding="utf-8")
        tokens = tokenizer.encode(text)
        actual_lengths.append(len(tokens))
        byte_lengths.append(len(text.encode("utf-8")))

    actual_avg = sum(actual_lengths) / len(actual_lengths)
    byte_avg = sum(byte_lengths) / len(byte_lengths)
    compression = byte_avg / actual_avg

    print(f"  Sample size: {len(sample_files)} files")
    print(f"  Avg bytes per file: {byte_avg:.1f}")
    print(f"  Avg tokens per file (ACTUAL): {actual_avg:.1f}")
    print(f"  Compression ratio: {compression:.2f}x")
    print(f"  Target was: {args.target_avg_length}")

    # Check if actual matches claimed
    claimed_avg = 323.0  # from the training log
    if abs(actual_avg - claimed_avg) > claimed_avg * 0.1:
        print(f"\n  WARNING: Actual avg ({actual_avg:.1f}) differs from claimed ({claimed_avg:.1f}) by >10%!")
    else:
        print(f"\n  OK: Actual encoding matches training statistics")

    return 0


if __name__ == "__main__":
    sys.exit(main())
