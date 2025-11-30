#!/usr/bin/env python3
"""
Fast Kubernetes Manifest Tokenization using Rust-backed TCT

Tokenizes manifests using the efficient Rust implementation from ~/git/tct
and saves to PyTorch cache format for instant loading during training.

Usage:
    python scripts/tokenize_k8s_fast.py --input ~/Desktop/data-test --output ~/Desktop/data-test/.cache
    python scripts/tokenize_k8s_fast.py --input ~/Desktop/data --output ~/Desktop/data/.cache
"""

import sys
import argparse
from pathlib import Path
import time

import torch

# Use the fast Rust tokenizer from ~/git/tct
sys.path.insert(0, str(Path.home() / "git/tct/.venv/lib/python3.12/site-packages"))
import tct_kubernetes as tct

def tokenize_manifests_fast(manifest_files, output_file, train_split=0.9):
    """
    Tokenize manifests using Rust implementation (fast!).

    Args:
        manifest_files: List of manifest file paths
        output_file: Where to save cache
        train_split: Train/val split ratio
    """
    print(f"\n{'='*60}")
    print(f"FAST TOKENIZATION (Rust-backed)")
    print(f"{'='*60}")
    print(f"Files to tokenize: {len(manifest_files):,}")
    print(f"Output: {output_file}")
    print(f"Train split: {train_split:.0%}")
    print()

    start_time = time.time()
    tokenized = []
    failed = 0

    # Tokenize (single-threaded, but Rust is FAST)
    for i, manifest_file in enumerate(manifest_files, 1):
        if i % 1000 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = (len(manifest_files) - i) / rate
            print(f"  Progress: {i:,}/{len(manifest_files):,} ({100*i/len(manifest_files):.1f}%) | "
                  f"Rate: {rate:.0f} files/sec | ETA: {remaining:.0f}s")

        try:
            with open(manifest_file, 'r') as f:
                manifest_json = f.read()

            # Rust encode (fast!)
            tokens = tct.encode(manifest_json)
            tokenized.append(torch.tensor(tokens, dtype=torch.long))

        except Exception as e:
            failed += 1
            if failed <= 10:
                print(f"  ⚠️  Failed: {manifest_file}: {e}")

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"TOKENIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Rate: {len(manifest_files)/elapsed:.0f} files/sec")
    print(f"  Success: {len(tokenized):,}")
    print(f"  Failed: {failed:,}")
    print()

    # Save cache
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokenized, output_file)
    print(f"✅ Cache saved: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    return tokenized


def main():
    parser = argparse.ArgumentParser(description="Fast tokenization using Rust")
    parser.add_argument("--input", required=True, help="Input directory (with k8s-split-XX subdirs)")
    parser.add_argument("--output", required=True, help="Output cache directory")
    parser.add_argument("--train-split", type=float, default=0.9, help="Train/val split")
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

    # Tokenize
    cache_file = output_dir / f"tokenized_k8s_split{int(args.train_split*100)}_{len(manifest_files)}files.pt"
    tokenized = tokenize_manifests_fast(manifest_files, cache_file, args.train_split)

    print(f"\n✅ Done! Cache ready for training.")
    print(f"   Training will load instantly from: {cache_file}")


if __name__ == "__main__":
    main()
