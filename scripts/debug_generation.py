#!/usr/bin/env python3
"""
Debug workflow generation issues by examining token-by-token decoding.

This script loads a checkpoint and generates tokens, then attempts to decode
them incrementally to find where decoding fails.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

# Add tct-bundle adapters to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tct-bundle" / "adapters"))

from nanochat.gpt import GPT, GPTConfig
from model_config import get_config
from tct_github_workflow import encode, decode, vocab_size

def load_model(checkpoint_path, device="cuda"):
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint metadata
    meta_path = str(checkpoint_path).replace("model_", "meta_").replace(".pt", ".json")
    if Path(meta_path).exists():
        with open(meta_path) as f:
            meta = json.load(f)
        config = meta["config"]
        model_config_kwargs = dict(
            sequence_len=config["context_size"],
            vocab_size=config["vocab_size"],
            n_layer=config["n_layers"],
            n_head=config["n_heads"],
            n_kv_head=config["n_heads"],
            n_embd=config["d_model"],
        )
    else:
        raise ValueError(f"No metadata found at {meta_path}")

    # Create model
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
    model.to(device)

    # Load weights
    model.load_state_dict(ckpt)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Vocab size: {vocab_size():,}")
    print()

    return model, model_config

@torch.no_grad()
def generate(model, prompt_tokens, max_new_tokens=256, temperature=0.8, top_k=40, device="cuda"):
    """Generate tokens autoregressively."""
    tokens = prompt_tokens.copy() if prompt_tokens else []
    context_size = model.config.sequence_len

    for _ in range(max_new_tokens):
        # Get context window
        if len(tokens) > 0:
            context = tokens[-context_size:] if len(tokens) > context_size else tokens
        else:
            context = [0]

        # Pad if needed
        if len(context) < context_size:
            context = [0] * (context_size - len(context)) + context

        # Convert to tensor and add batch dimension
        x = torch.tensor([context], dtype=torch.long, device=device)

        # Forward pass
        logits = model(x)  # (batch, seq_len, vocab_size)

        # Get logits for last position
        logits = logits[0, -1, :]  # (vocab_size,)

        # Apply temperature
        if temperature > 0:
            logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float('Inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            # Greedy sampling
            next_token = torch.argmax(logits, dim=-1).item()

        # Append generated token
        tokens.append(next_token)

    return tokens

def analyze_token_sequence(tokens, max_display=50):
    """Analyze a token sequence by attempting incremental decoding."""
    print("\n" + "="*80)
    print("INCREMENTAL DECODING ANALYSIS")
    print("="*80)

    # Show token statistics
    print(f"\nTotal tokens: {len(tokens)}")
    print(f"Unique tokens: {len(set(tokens))}")
    print(f"Token range: [{min(tokens)}, {max(tokens)}]")
    print(f"Vocab size: {vocab_size()}")

    # Check for out-of-vocab tokens
    out_of_vocab = [t for t in tokens if t >= vocab_size() or t < 0]
    if out_of_vocab:
        print(f"\n⚠️  WARNING: {len(out_of_vocab)} out-of-vocab tokens found!")
        print(f"   Out-of-vocab tokens: {set(out_of_vocab)}")
    else:
        print(f"✅ All tokens are within vocab range [0, {vocab_size()-1}]")

    # Show first N tokens
    print(f"\nFirst {min(max_display, len(tokens))} tokens:")
    print(f"   {tokens[:max_display]}")

    # Try incremental decoding to find where it fails
    print("\n" + "-"*80)
    print("Attempting incremental decoding...")
    print("-"*80)

    last_successful_length = 0
    last_successful_json = ""
    error_position = None
    error_message = None

    # Try decoding at different lengths
    test_lengths = [1, 2, 5, 10, 20, 50, 100, 150, 200, 250, 300]
    test_lengths = [l for l in test_lengths if l <= len(tokens)]
    test_lengths.append(len(tokens))  # Always test full length

    for length in test_lengths:
        subset = tokens[:length]
        try:
            json_str, consumed, total = decode(subset)
            last_successful_length = length
            last_successful_json = json_str

            # Show summary
            if length <= 20 or length % 50 == 0 or length == len(tokens):
                print(f"  Length {length:3d}: ✅ Success (consumed {consumed}/{total})")
                if json_str:
                    snippet = json_str[:80].replace('\n', ' ')
                    if len(json_str) > 80:
                        snippet += "..."
                    print(f"             Output: {snippet}")
        except Exception as e:
            error_position = length
            error_message = str(e)
            print(f"  Length {length:3d}: ❌ FAILED")
            print(f"             Error: {error_message}")
            break

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if error_position is None:
        print(f"✅ Successfully decoded all {len(tokens)} tokens")
        print(f"\nFinal output length: {len(last_successful_json)} characters")

        # Try to parse as JSON
        try:
            workflow = json.loads(last_successful_json)
            print(f"✅ Valid JSON workflow")
            print(f"\nWorkflow keys: {list(workflow.keys())}")
            print(f"\nFull workflow:")
            print(json.dumps(workflow, indent=2))
        except json.JSONDecodeError as e:
            print(f"⚠️  Not valid JSON: {e}")
            print(f"\nRaw output:")
            print(last_successful_json)
    else:
        print(f"❌ Decoding failed at position {error_position}")
        print(f"   Error: {error_message}")
        print(f"\n✅ Last successful decode: {last_successful_length} tokens")

        # Show problematic token(s)
        if error_position <= len(tokens):
            problematic_token = tokens[error_position - 1]
            print(f"\n🔍 Problematic token at position {error_position - 1}:")
            print(f"   Token ID: {problematic_token}")

            # Show context around the problematic token
            context_start = max(0, error_position - 10)
            context_end = min(len(tokens), error_position + 5)
            context_tokens = tokens[context_start:context_end]

            print(f"\n   Context (tokens {context_start} to {context_end-1}):")
            for i, tok in enumerate(context_tokens):
                idx = context_start + i
                marker = " <<<" if idx == error_position - 1 else ""
                print(f"      [{idx:3d}] Token {tok}{marker}")

        if last_successful_json:
            print(f"\n📄 Last successful output ({last_successful_length} tokens):")
            print("-" * 80)
            print(last_successful_json)
            print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Debug workflow generation issues")
    parser.add_argument("--checkpoint", type=str,
                      default="checkpoints/tct_prefix_fim_medium_p100/model_040000.pt",
                      help="Path to model checkpoint")
    parser.add_argument("--max_tokens", type=int, default=300,
                      help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                      help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=100,
                      help="Top-k sampling parameter")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    # Expand paths
    checkpoint_path = Path(args.checkpoint).expanduser()

    # Load model
    model, model_config = load_model(checkpoint_path, device=args.device)

    # Generate from scratch
    print("🎲 Generating workflow from scratch...")
    print(f"   max_tokens={args.max_tokens}, temperature={args.temperature}, top_k={args.top_k}")
    print()

    prompt_tokens = []
    generated_tokens = generate(
        model,
        prompt_tokens,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device
    )

    print(f"✅ Generated {len(generated_tokens)} tokens")

    # Analyze the token sequence
    analyze_token_sequence(generated_tokens)

if __name__ == "__main__":
    main()
