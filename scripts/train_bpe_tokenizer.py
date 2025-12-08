#!/usr/bin/env python3
"""
Train a pure byte-level BPE tokenizer on Kubernetes YAML manifests.

This trains a UTF-8 BPE tokenizer WITHOUT regex splitting for fair
comparison with TCT's schema-aware tokenization.
"""

import argparse
import json
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
        if (i + 1) % 10000 == 0:
            print(f"  Loaded {i + 1}/{total} files", file=sys.stderr)
        yield path.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Train pure byte-level BPE on Kubernetes YAML"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/Desktop/data/k8s-yaml"),
        help="Directory with YAML manifests",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=20000,
        help="Target vocabulary size (default: 20000 to match TCT)",
    )
    parser.add_argument(
        "--output-dir",
        default="tokenizers/bpe-k8s-20k",
        help="Output directory for trained tokenizer",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to train on (for testing)",
    )
    args = parser.parse_args()

    # Count files
    yaml_files = list(Path(args.data_dir).glob("*.yaml"))
    if args.limit:
        yaml_files = yaml_files[:args.limit]

    print(f"Training pure byte-level BPE (no regex splitting)")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Files: {len(yaml_files)}")
    print(f"  Vocab size: {args.vocab_size}")
    print(f"  Output: {args.output_dir}")

    if len(yaml_files) == 0:
        print("\nError: No YAML files found.")
        print("Run prepare_bpe_data.py first to convert JSON to YAML.")
        return 1

    # Train with rustbpe using empty pattern (no regex splitting)
    print("\nTraining...")
    tokenizer = rustbpe.Tokenizer()
    tokenizer.train_from_iterator(
        yaml_iterator(args.data_dir, args.limit),
        vocab_size=args.vocab_size,
        pattern="",  # Empty pattern = no regex splitting = pure byte BPE
    )

    # Save tokenizer - save mergeable_ranks as JSON
    os.makedirs(args.output_dir, exist_ok=True)

    ranks = tokenizer.get_mergeable_ranks()
    ranks_serializable = [[list(token), rank] for token, rank in ranks]

    with open(os.path.join(args.output_dir, "mergeable_ranks.json"), "w") as f:
        json.dump(ranks_serializable, f)

    print(f"\nSaved tokenizer to {args.output_dir}")
    print(f"  Vocab size: {args.vocab_size}")

    # Save metadata
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump({
            "vocab_size": args.vocab_size,
            "data_dir": args.data_dir,
            "num_files": len(yaml_files),
            "type": "rustbpe_pure_byte",
        }, f, indent=2)

    # Test encoding
    sample = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: test\n"
    tokens = tokenizer.encode(sample)
    print(f"\nTest encoding:")
    print(f"  Input bytes: {len(sample.encode())}")
    print(f"  Tokens: {len(tokens)}")
    print(f"  Compression: {len(sample.encode())/len(tokens):.2f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
