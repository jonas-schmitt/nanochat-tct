#!/usr/bin/env python3
"""
Generate GitHub Actions workflows using trained TCT model.

Usage:
    # Generate from a partial workflow prompt
    python scripts/generate_workflow.py --checkpoint ~/Desktop/checkpoints/tct_small/model_005000.pt

    # Generate multiple samples
    python scripts/generate_workflow.py --checkpoint ~/Desktop/checkpoints/tct_small/model_005000.pt --num_samples 5

    # Use custom prompt workflow
    python scripts/generate_workflow.py --checkpoint ~/Desktop/checkpoints/tct_small/model_005000.pt --prompt workflow.json
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
        model_config_kwargs = meta["model_config"]
    else:
        # Fallback to default small config
        print("Warning: No metadata found, using default small config")
        config = get_config("small")
        model_config_kwargs = dict(
            sequence_len=config["context_size"],
            vocab_size=config["vocab_size"],
            n_layer=config["n_layers"],
            n_head=config["n_heads"],
            n_kv_head=config["n_heads"],
            n_embd=config["d_model"],
        )

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
    """
    Generate tokens autoregressively from a prompt.

    Args:
        model: Trained GPT model
        prompt_tokens: List of token IDs to start from
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Only sample from top k tokens (0 = no filtering)
        device: Device to run on

    Returns:
        List of generated token IDs (including prompt)
    """
    tokens = prompt_tokens.copy()
    context_size = model.config.sequence_len

    for _ in range(max_new_tokens):
        # Get context window (last context_size tokens)
        context = tokens[-context_size:] if len(tokens) > context_size else tokens

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

        # Try to decode periodically to check if we have a complete workflow
        if len(tokens) > 10 and len(tokens) % 20 == 0:
            try:
                _, consumed, total = decode(tokens)
                if consumed == total:
                    # Successfully decoded complete workflow
                    print(f"  Generated {len(tokens)} tokens (complete workflow)")
                    break
            except Exception:
                pass  # Continue generating

    return tokens

def main():
    parser = argparse.ArgumentParser(description="Generate workflows with trained TCT model")
    parser.add_argument("--checkpoint", type=str, default="~/Desktop/checkpoints/tct_small/model_005000.pt",
                      help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                      help="Path to JSON workflow prompt (optional)")
    parser.add_argument("--num_samples", type=int, default=1,
                      help="Number of samples to generate")
    parser.add_argument("--max_tokens", type=int, default=256,
                      help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                      help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40,
                      help="Top-k sampling parameter")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    # Expand paths
    checkpoint_path = Path(args.checkpoint).expanduser()

    # Load model
    model, model_config = load_model(checkpoint_path, device=args.device)

    # Prepare prompt
    if args.prompt:
        # Use custom prompt from file
        with open(args.prompt) as f:
            prompt_workflow = json.load(f)
        prompt_json = json.dumps(prompt_workflow)
        print(f"Using custom prompt from {args.prompt}")
    else:
        # Use minimal default prompt
        prompt_workflow = {
            "name": "CI",
            "on": "push",
            "jobs": {}
        }
        prompt_json = json.dumps(prompt_workflow)
        print("Using default minimal prompt")

    print(f"Prompt: {prompt_json}")
    print()

    # Encode prompt
    prompt_tokens = encode(prompt_json)
    print(f"Prompt encoded to {len(prompt_tokens)} tokens")
    print()

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    print("=" * 80)

    valid_count = 0

    for i in range(args.num_samples):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{args.num_samples}")
        print('='*80)

        # Generate tokens
        generated_tokens = generate(
            model,
            prompt_tokens,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device
        )

        print(f"Generated {len(generated_tokens)} tokens total ({len(generated_tokens) - len(prompt_tokens)} new)")

        # Decode back to JSON
        try:
            decoded_json, consumed, total = decode(generated_tokens)

            # Try to parse JSON
            try:
                workflow = json.loads(decoded_json)
                valid_json = True
                valid_count += 1
            except json.JSONDecodeError as e:
                valid_json = False
                print(f"⚠️  JSON parse error: {e}")

            print()
            print("Generated workflow:")
            print("-" * 80)
            if valid_json:
                print(json.dumps(workflow, indent=2))
                print("-" * 80)
                print(f"✅ Valid JSON workflow")
                print(f"   Tokens consumed: {consumed}/{total}")

                # Check schema compliance
                has_name = "name" in workflow
                has_on = "on" in workflow
                has_jobs = "jobs" in workflow
                schema_valid = has_name and has_on and has_jobs

                if schema_valid:
                    print(f"✅ Schema compliant (has 'name', 'on', 'jobs')")
                else:
                    print(f"⚠️  Schema incomplete (name={has_name}, on={has_on}, jobs={has_jobs})")
            else:
                print(decoded_json)
                print("-" * 80)
                print(f"❌ Invalid JSON")

        except Exception as e:
            print(f"❌ Decode error: {e}")

    # Summary
    print()
    print("=" * 80)
    print("Generation Summary")
    print("=" * 80)
    print(f"Valid workflows: {valid_count}/{args.num_samples} ({100*valid_count/args.num_samples:.1f}%)")
    print()

if __name__ == "__main__":
    main()
