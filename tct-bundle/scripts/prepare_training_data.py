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


def prepare_windows(json_files, context_size=256, stride=1, position_stride=None, skip_short=True):
    """
    Convert JSON workflows to training windows.

    Args:
        json_files: List of paths to JSON workflow files
        context_size: TOTAL window size (including position token)
                     E.g., context_size=1024 means 1 position + 1023 content tokens
        stride: Stride for windowing (how much to slide between windows)
        position_stride: Stride for position mapping (defaults to stride if not specified)
                        E.g., stride=512, position_stride=32 means:
                        - Windows slide by 512 tokens (50% overlap for 1024 context)
                        - But positions mapped with granularity of 32 tokens (for vocab)
        skip_short: Skip sequences shorter than context_size

    Returns:
        List of windows: [[position, tok_0, ..., tok_N], ...] each of length context_size
    """
    if position_stride is None:
        position_stride = stride
    windows = []
    skipped = 0

    # Content size = context_size - 1 (reserve 1 token for position)
    content_size = context_size - 1

    for json_file in tqdm(json_files, desc="Processing workflows"):
        try:
            # Load workflow
            with open(json_file) as f:
                json_str = f.read()

            # Encode with TCT
            tokens = encode(json_str)

            # Skip if too short (need at least content_size tokens)
            if len(tokens) < content_size:
                if skip_short:
                    skipped += 1
                    continue
                # For short sequences, just create one window with padding
                # (alternative: skip entirely if skip_short=True)

            # Extract strided windows (stride controls window sliding)
            for start in range(0, len(tokens) - content_size + 1, stride):
                end = start + content_size
                # Map position: divide by position_stride to keep position tokens < 8192
                # This allows fine-grained position info even with large windowing stride
                mapped_start = start // position_stride
                # Extract window with mapped position
                window_tokens = tokens[start:end]
                # Create window: [mapped_position, content_tok_0, ..., content_tok_(N-1)]
                # Total length = 1 + content_size = context_size
                window = [mapped_start] + window_tokens
                windows.append(window)

        except Exception as e:
            print(f"Error processing {json_file}: {e}", file=sys.stderr)
            continue

    if skipped > 0:
        print(f"⚠️  Skipped {skipped} short sequences (<{content_size} tokens)")

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
        help="TOTAL window size including position token (default: 256)",
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
        help="Stride for windowing (default: 1 for all positions, use 512 for 50%% overlap)",
    )
    parser.add_argument(
        "--position-stride",
        type=int,
        default=None,
        help="Stride for position mapping (default: same as --stride, use 32 for vocab_size=8192)",
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
    print(f"Windowing stride: {args.stride}")
    position_stride = args.position_stride if args.position_stride is not None else args.stride
    print(f"Position stride: {position_stride} (for vocab mapping)")
    if args.stride != position_stride:
        overlap_pct = ((args.context_size - args.stride) / args.context_size) * 100
        print(f"  → {overlap_pct:.1f}% overlap between windows")
        print(f"  → Position granularity: every {position_stride} tokens")
    print()

    # Prepare windows
    windows = prepare_windows(
        json_files,
        context_size=args.context_size,
        stride=args.stride,
        position_stride=args.position_stride,
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
        "position_stride": position_stride,
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
