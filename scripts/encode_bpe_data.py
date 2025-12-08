#!/usr/bin/env python3
"""
Encode Kubernetes YAML manifests with BPE tokenizer and save as JSONL.

Creates train.jsonl and validate.jsonl in the same format as TCT encoding,
allowing reuse of the same dataloader infrastructure.
"""

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Encode YAML manifests with BPE tokenizer"
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/Desktop/data/k8s-yaml"),
        help="Directory with YAML manifests",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizers/bpe-k8s-20k/tokenizer.pkl",
        help="Path to trained BPE tokenizer",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.expanduser("~/Desktop/data/k8s-bpe-encoded"),
        help="Output directory for encoded JSONL files",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio of data to use for training (default: 0.9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    with open(args.tokenizer, "rb") as f:
        enc = pickle.load(f)
    print(f"  Vocab size: {enc.n_vocab}")

    # Get YAML files
    yaml_files = sorted(Path(args.data_dir).glob("*.yaml"))
    if args.limit:
        yaml_files = yaml_files[:args.limit]
    print(f"\nFound {len(yaml_files)} YAML files in {args.data_dir}")

    # Shuffle and split
    random.seed(args.seed)
    indices = list(range(len(yaml_files)))
    random.shuffle(indices)

    train_count = int(len(indices) * args.train_ratio)
    train_indices = set(indices[:train_count])
    val_indices = set(indices[train_count:])

    print(f"Split: {len(train_indices)} train, {len(val_indices)} validate")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Encode and write
    train_file = output_dir / "train.jsonl"
    val_file = output_dir / "validate.jsonl"

    train_tokens = 0
    val_tokens = 0
    train_seqs = 0
    val_seqs = 0

    print(f"\nEncoding manifests...")
    with open(train_file, "w") as f_train, open(val_file, "w") as f_val:
        for i, yaml_path in enumerate(yaml_files):
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1}/{len(yaml_files)} files")

            # Read and encode
            text = yaml_path.read_text(encoding="utf-8")
            tokens = enc.encode(text)

            # Write to appropriate file
            if i in train_indices:
                f_train.write(json.dumps(tokens) + "\n")
                train_tokens += len(tokens)
                train_seqs += 1
            else:
                f_val.write(json.dumps(tokens) + "\n")
                val_tokens += len(tokens)
                val_seqs += 1

    # Write metadata
    metadata = {
        "tokenizer": args.tokenizer,
        "vocab_size": enc.n_vocab,
        "source_dir": args.data_dir,
        "total_files": len(yaml_files),
        "train_count": train_seqs,
        "validate_count": val_seqs,
        "train_tokens": train_tokens,
        "validate_tokens": val_tokens,
        "avg_tokens_per_manifest": (train_tokens + val_tokens) / len(yaml_files),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone!")
    print(f"  Train: {train_seqs:,} sequences, {train_tokens:,} tokens")
    print(f"  Val:   {val_seqs:,} sequences, {val_tokens:,} tokens")
    print(f"  Avg tokens/manifest: {metadata['avg_tokens_per_manifest']:.1f}")
    print(f"\nSaved to: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
