#!/usr/bin/env python3
"""
Test script for constrained BPB evaluation.

Tests the compute_constrained_bpb function on TSConfig validation data.

Usage:
    python -m scripts.test_constrained_bpb \
        --checkpoint checkpoints/tsconfig_utf8_small/ \
        --schema tsconfig \
        --num_samples 100
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.xgrammar_tokenizer import (
    UTF8BPEDecoder,
    build_xgrammar_tokenizer_info,
    compile_json_schema_grammar,
    compute_constrained_bpb,
    load_schema,
)


def load_model(checkpoint_dir: Path, device: str = "cuda"):
    """Load model from checkpoint directory."""
    config_path = checkpoint_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    # Create model config
    model_config = GPTConfig(
        vocab_size=config_dict.get("vocab_size", 20000),
        sequence_len=config_dict.get("sequence_len", config_dict.get("context_size", 2048)),
        n_layer=config_dict.get("n_layer", 10),
        n_head=config_dict.get("n_head", 6),
        n_embd=config_dict.get("n_embd", 384),
    )

    model = GPT(model_config)

    # Find checkpoint file
    checkpoint_files = list(checkpoint_dir.glob("*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .pt files in {checkpoint_dir}")

    # Prefer best.pt or latest
    checkpoint_path = checkpoint_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = sorted(checkpoint_files)[-1]

    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Handle torch.compile wrapper prefix
    cleaned = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)

    model.to(device)
    model.eval()

    return model, config_dict


def load_validation_tokens(data_dir: Path, max_samples: int = 100) -> list:
    """Load validation tokens from JSONL file."""
    # Try validate.jsonl first
    val_path = data_dir / "validate.jsonl"

    if not val_path.exists():
        # Try alternative paths
        for alt in ["val.jsonl", "test.jsonl"]:
            alt_path = data_dir / alt
            if alt_path.exists():
                val_path = alt_path
                break

    if not val_path.exists():
        # Fall back to all.jsonl and use last 10% as validation
        all_path = data_dir / "all.jsonl"
        if all_path.exists():
            print(f"Using last portion of all.jsonl as validation...")
            # Count lines first
            with open(all_path) as f:
                total_lines = sum(1 for _ in f)

            # Use last 10% as validation
            val_start = int(total_lines * 0.9)
            tokens = []
            with open(all_path) as f:
                for i, line in enumerate(f):
                    if i >= val_start:
                        tokens.append(json.loads(line))
                        if len(tokens) >= max_samples:
                            break
            print(f"Loaded {len(tokens)} validation sequences from {all_path} (lines {val_start}+)")
            return tokens
        else:
            raise FileNotFoundError(f"No validation file found in {data_dir}")

    tokens = []
    with open(val_path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            tokens.append(json.loads(line))

    print(f"Loaded {len(tokens)} validation sequences from {val_path}")
    return tokens


def create_random_model(vocab_size: int, device: str = "cuda"):
    """Create a random model for testing."""
    model_config = GPTConfig(
        vocab_size=vocab_size,
        sequence_len=512,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
    )
    model = GPT(model_config)
    model.to(device)
    model.eval()
    return model, {
        "vocab_size": vocab_size,
        "sequence_len": 512,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
    }


def main():
    parser = argparse.ArgumentParser(description="Test constrained BPB evaluation")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint directory")
    parser.add_argument("--random_model", action="store_true", help="Use random model for testing")
    parser.add_argument("--schema", type=str, default="tsconfig", choices=["tsconfig", "eslintrc", "kubernetes"])
    parser.add_argument("--data_dir", type=str, help="Data directory (default: auto-detect)")
    parser.add_argument("--merge_table", type=str, help="BPE merge table path (default: auto-detect)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.checkpoint and not args.random_model:
        print("ERROR: Must specify --checkpoint or --random_model")
        sys.exit(1)

    # Find merge table first (needed for vocab size)
    if args.merge_table:
        merge_table = Path(args.merge_table)
    else:
        # Auto-detect from bpe-merges directory
        bpe_dir = Path(__file__).parent.parent / "bpe-merges"
        merge_candidates = list(bpe_dir.glob(f"{args.schema}*.json"))
        if not merge_candidates:
            raise FileNotFoundError(f"No merge table found for {args.schema} in {bpe_dir}")
        merge_table = merge_candidates[0]

    print(f"\nUsing merge table: {merge_table}")

    # Create decoder to get vocab size
    utf8_decoder = UTF8BPEDecoder(merge_table)
    vocab_size = utf8_decoder.vocab_size()
    print(f"Decoder vocab size: {vocab_size}")

    # Load or create model
    print(f"\n{'='*60}")
    if args.random_model:
        print(f"Creating random model (vocab_size={vocab_size})")
        print(f"{'='*60}")
        model, config = create_random_model(vocab_size, args.device)
        print(f"Model config: vocab_size={config.get('vocab_size')}, n_layer={config.get('n_layer')}")
    else:
        checkpoint_dir = Path(args.checkpoint)
        print(f"Loading model from {checkpoint_dir}")
        print(f"{'='*60}")
        model, config = load_model(checkpoint_dir, args.device)
        print(f"Model config: vocab_size={config.get('vocab_size')}, n_layer={config.get('n_layer')}")

    # Build XGrammar tokenizer info (merge_table and utf8_decoder already created above)
    print("\nBuilding XGrammar tokenizer info...")
    tokenizer_info = build_xgrammar_tokenizer_info(merge_table)

    # Load and compile schema
    print(f"Loading schema: {args.schema}")
    schema = load_schema(args.schema)
    print(f"Compiling grammar...")
    compiled_grammar = compile_json_schema_grammar(tokenizer_info, schema)

    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Try common locations
        candidates = [
            Path(__file__).parent.parent.parent / "data" / f"{args.schema}-utf8-bpe-1k",
            Path(__file__).parent.parent.parent / "data" / f"{args.schema}-utf8-base-matched",
            Path("/tmp") / f"{args.schema}-utf8-bpe-1k",
            Path("/tmp") / f"{args.schema}-utf8-base-matched",
        ]
        data_dir = None
        for c in candidates:
            if c.exists():
                data_dir = c
                break

        if data_dir is None:
            print(f"\nCould not find data directory. Tried: {candidates}")
            print("Please specify --data_dir")
            sys.exit(1)

    print(f"\nUsing data directory: {data_dir}")

    # Load validation tokens
    validation_tokens = load_validation_tokens(data_dir, args.num_samples)

    # Compute constrained BPB
    print(f"\n{'='*60}")
    print(f"Computing constrained BPB on {len(validation_tokens)} sequences")
    print(f"{'='*60}")

    result = compute_constrained_bpb(
        model=model,
        tokenizer_info=tokenizer_info,
        compiled_grammar=compiled_grammar,
        utf8_decoder=utf8_decoder,
        validation_tokens=validation_tokens,
        device=args.device,
        max_seq_len=args.max_seq_len,
        show_progress=True,
    )

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Sequences evaluated: {result.num_sequences}")
    print(f"Total tokens:        {result.total_tokens}")
    print(f"Total bytes:         {result.total_bytes}")
    print(f"")
    print(f"Raw BPB:             {result.raw_bpb:.4f}")
    print(f"Constrained BPB:     {result.constrained_bpb:.4f}")
    print(f"BPB reduction:       {(result.raw_bpb - result.constrained_bpb) / result.raw_bpb * 100:.1f}%")
    print(f"")
    print(f"Raw loss (nats):     {result.raw_loss:.2f}")
    print(f"Constrained loss:    {result.constrained_loss:.2f}")


if __name__ == "__main__":
    main()
