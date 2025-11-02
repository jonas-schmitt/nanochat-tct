#!/usr/bin/env python3
"""
Prepare Training Data for Nanochat-TCT

Converts JSON workflows to windowed training data using TCT tokenization.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed", file=sys.stderr)
    print("Install with: pip install torch", file=sys.stderr)
    sys.exit(1)

try:
    from tct_github_workflow import encode, extract_window
except ImportError:
    print("Error: TCT tokenizer not installed", file=sys.stderr)
    print("Install wheel: pip install tct_github_workflow-1.0.4-*.whl", file=sys.stderr)
    sys.exit(1)

from tqdm import tqdm


def prepare_windows(json_files, context_size=256, stride=1, skip_short=True):
    """
    Convert JSON workflows to training windows.

    Args:
        json_files: List of paths to JSON workflow files
        context_size: Window size (number of content tokens, excluding position token)
        stride: Stride for windowing (default: 1 for all positions, 32 for vocab_size=8192)
        skip_short: Skip sequences shorter than context_size

    Returns:
        List of windows: [[position, tok_0, ..., tok_N], ...]
    """
    windows = []
    skipped = 0

    for json_file in tqdm(json_files, desc="Processing workflows"):
        try:
            # Load workflow
            with open(json_file) as f:
                json_str = f.read()

            # Encode with TCT
            tokens = encode(json_str)

            # Skip if too short
            if len(tokens) <= context_size:
                if skip_short:
                    skipped += 1
                    continue
                # For short sequences, just create one window with padding
                # (alternative: skip entirely if skip_short=True)

            # Extract strided windows (stride controls position sampling)
            for start in range(0, len(tokens) - context_size, stride):
                end = start + context_size
                # Map position: divide by stride to keep position tokens < 8192
                mapped_start = start // stride
                # Extract window with mapped position
                window_tokens = tokens[start:end]
                # Create window: [mapped_position, content_tok_0, ..., content_tok_N]
                window = [mapped_start] + window_tokens
                windows.append(window)

        except Exception as e:
            print(f"Error processing {json_file}: {e}", file=sys.stderr)
            continue

    if skipped > 0:
        print(f"⚠️  Skipped {skipped} short sequences (<{context_size} tokens)")

    return windows


def main():
    parser = argparse.ArgumentParser(
        description="Prepare windowed training data from JSON workflows"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input directory containing JSON workflow files",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for prepared data",
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=256,
        help="Window size (default: 256)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8 for 80%% train)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for windowing (default: 1 for all positions, use 32 for vocab_size=8192)",
    )
    parser.add_argument(
        "--no-skip-short",
        action="store_true",
        help="Don't skip sequences shorter than context_size",
    )

    args = parser.parse_args()

    # Find JSON files
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    json_files = list(input_dir.glob("*.json"))
    if len(json_files) == 0:
        print(f"Error: No JSON files found in {input_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(json_files)} JSON workflow files")
    print(f"Context size: {args.context_size}")
    print(f"Stride: {args.stride}" + (" (strided windowing for vocab_size=8192)" if args.stride > 1 else ""))
    print()

    # Prepare windows
    windows = prepare_windows(
        json_files,
        context_size=args.context_size,
        stride=args.stride,
        skip_short=not args.no_skip_short,
    )

    if len(windows) == 0:
        print("Error: No windows extracted", file=sys.stderr)
        return 1

    print(f"Extracted {len(windows)} training windows")
    print()

    # Convert to tensor
    windows_tensor = torch.tensor(windows, dtype=torch.long)
    print(f"Tensor shape: {windows_tensor.shape}")
    print(f"  (num_windows, window_size) = ({windows_tensor.shape[0]}, {windows_tensor.shape[1]})")
    print()

    # Split train/val
    num_train = int(len(windows) * args.train_split)
    train_windows = windows_tensor[:num_train]
    val_windows = windows_tensor[num_train:]

    print(f"Train/val split:")
    print(f"  Train: {len(train_windows)} windows ({args.train_split * 100:.0f}%)")
    print(f"  Val:   {len(val_windows)} windows ({(1 - args.train_split) * 100:.0f}%)")
    print()

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.pt"
    val_path = output_dir / "val.pt"

    torch.save(train_windows, train_path)
    torch.save(val_windows, val_path)

    print(f"✅ Saved training data:")
    print(f"  Train: {train_path}")
    print(f"  Val:   {val_path}")
    print()

    # Save metadata
    metadata = {
        "context_size": args.context_size,
        "stride": args.stride,
        "train_split": args.train_split,
        "num_workflows": len(json_files),
        "num_windows": len(windows),
        "num_train": len(train_windows),
        "num_val": len(val_windows),
        "window_shape": list(windows_tensor.shape),
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved metadata: {metadata_path}")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
