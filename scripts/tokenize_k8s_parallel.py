#!/usr/bin/env python3
"""
Parallel Kubernetes Manifest Tokenization using Rust TCT

Tokenizes manifests using fast parallel Rust implementation and saves to PyTorch cache.

Usage:
    python scripts/tokenize_k8s_parallel.py --input ~/Desktop/data-test --output ~/Desktop/data-test/.cache
    python scripts/tokenize_k8s_parallel.py --input ~/Desktop/data --output ~/Desktop/data/.cache
"""

import sys
import argparse
import subprocess
import json
from pathlib import Path
import time
import os

import torch


def tokenize_parallel_rust(manifest_files, output_jsonl, train_split=0.9):
    """
    Tokenize manifests using parallel Rust implementation.

    Args:
        manifest_files: List of manifest file paths
        output_jsonl: Where to save JSONL output from Rust
        train_split: Train/val split ratio
    """
    print(f"\n{'='*60}")
    print(f"PARALLEL RUST TOKENIZATION")
    print(f"{'='*60}")
    print(f"Files to tokenize: {len(manifest_files):,}")
    print(f"Output (temp): {output_jsonl}")
    print()

    # Prepare glob pattern for Rust encoder
    # Since we have a list of files, we need to pass them via a glob pattern
    # The Rust code uses expand_glob_pattern, so we need to create a pattern

    # Get the parent directory of first file and create pattern
    if not manifest_files:
        raise ValueError("No manifest files provided")

    parent_dir = Path(manifest_files[0]).parent
    glob_pattern = str(parent_dir / "*.json")

    print(f"Using glob pattern: {glob_pattern}")
    print()

    # Paths to TCT resources
    tct_binary = Path.home() / "git/tct/target/release/tct-encode-batch"
    schema_path = Path.home() / "git/tct/schemas/popular/kubernetes.json"
    bpe_path = Path.home() / "git/tct/schemas/bpe/kubernetes-bpe20000.json"

    if not tct_binary.exists():
        raise FileNotFoundError(f"TCT encoder not found: {tct_binary}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    if not bpe_path.exists():
        raise FileNotFoundError(f"BPE merges not found: {bpe_path}")

    # Run parallel Rust encoder
    cmd = [
        str(tct_binary),
        "--schema", str(schema_path),
        "--bpe-merges", str(bpe_path),
        "--instances", glob_pattern,
        "--output", str(output_jsonl),
        "-O", "3",  # Maximum optimization
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        raise RuntimeError(f"Rust encoder failed with exit code {result.returncode}")

    print(f"\n✅ Rust encoding completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"   Rate: {len(manifest_files)/elapsed:.0f} files/sec")
    print()

    return output_jsonl


def jsonl_to_pytorch_cache(jsonl_file, output_cache, train_split=0.9):
    """
    Convert JSONL token sequences to PyTorch cache.

    Args:
        jsonl_file: JSONL file with token sequences
        output_cache: Output .pt file
        train_split: Not used (kept for compatibility)
    """
    print(f"\n{'='*60}")
    print(f"CONVERTING TO PYTORCH CACHE")
    print(f"{'='*60}")
    print(f"Input: {jsonl_file}")
    print(f"Output: {output_cache}")
    print()

    start_time = time.time()
    tokenized = []

    print("Reading JSONL...")
    with open(jsonl_file, 'r') as f:
        for i, line in enumerate(f, 1):
            if i % 10000 == 0:
                print(f"  Progress: {i:,} sequences")

            tokens = json.loads(line)
            tokenized.append(torch.tensor(tokens, dtype=torch.long))

    elapsed = time.time() - start_time

    print(f"\n✅ Loaded {len(tokenized):,} sequences in {elapsed:.1f}s")
    print()

    # Save cache
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokenized, output_cache)

    cache_size_mb = output_cache.stat().st_size / 1024 / 1024

    print(f"✅ Cache saved: {output_cache}")
    print(f"   Size: {cache_size_mb:.1f} MB")
    print()

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Fast parallel tokenization using Rust")
    parser.add_argument("--input", required=True, help="Input directory (with k8s-split-XX subdirs)")
    parser.add_argument("--output", required=True, help="Output cache directory")
    parser.add_argument("--train-split", type=float, default=0.9, help="Train/val split")
    parser.add_argument("--keep-jsonl", action="store_true", help="Keep intermediate JSONL file")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()

    # Find all manifests
    print(f"Scanning {input_dir}...")
    manifest_files = []
    split_dirs = sorted(input_dir.glob("k8s-split-*"))

    if not split_dirs:
        print(f"❌ No k8s-split-* directories found in {input_dir}")
        sys.exit(1)

    for split_dir in split_dirs:
        files = sorted(split_dir.glob("*.json"))
        manifest_files.extend(files)
        print(f"  {split_dir.name}: {len(files):,} files")

    print(f"Total: {len(manifest_files):,} files")

    # Tokenize (process each split directory separately for better parallelization)
    all_tokenized = []
    temp_dir = Path("/tmp/k8s_tokenization")
    temp_dir.mkdir(exist_ok=True)

    for split_dir in split_dirs:
        split_name = split_dir.name
        split_files = sorted(split_dir.glob("*.json"))

        if not split_files:
            continue

        print(f"\n{'='*60}")
        print(f"Processing {split_name}")
        print(f"{'='*60}")

        # Tokenize this split
        jsonl_file = temp_dir / f"{split_name}_tokens.jsonl"
        tokenize_parallel_rust(split_files, jsonl_file, args.train_split)

        # Convert to tensors
        print(f"Converting {split_name} to tensors...")
        with open(jsonl_file, 'r') as f:
            for line in f:
                tokens = json.loads(line)
                all_tokenized.append(torch.tensor(tokens, dtype=torch.long))

        # Clean up JSONL unless asked to keep
        if not args.keep_jsonl:
            jsonl_file.unlink()

    # Save final cache
    cache_file = output_dir / f"tokenized_k8s_split{int(args.train_split*100)}_{len(manifest_files)}files.pt"

    print(f"\n{'='*60}")
    print(f"SAVING FINAL CACHE")
    print(f"{'='*60}")
    print(f"Total sequences: {len(all_tokenized):,}")
    print(f"Cache file: {cache_file}")
    print()

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_tokenized, cache_file)

    cache_size_mb = cache_file.stat().st_size / 1024 / 1024
    print(f"✅ Cache saved!")
    print(f"   Path: {cache_file}")
    print(f"   Size: {cache_size_mb:.1f} MB")
    print(f"   Sequences: {len(all_tokenized):,}")
    print()

    print(f"\n✅ Done! Cache ready for training.")
    print(f"   Training will load instantly from: {cache_file}")


if __name__ == "__main__":
    main()
