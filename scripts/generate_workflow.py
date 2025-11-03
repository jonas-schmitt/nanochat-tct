#!/usr/bin/env python3
"""
Generate GitHub Actions workflows using trained TCT model.

Usage:
    # Interactive mode (complete partial workflows)
    python scripts/generate_workflow.py --checkpoint ~/Desktop/checkpoints/tct_small/model_20000.pt --interactive

    # Step-by-step token prediction (see model predict token-by-token)
    python scripts/generate_workflow.py --checkpoint ~/Desktop/checkpoints/tct_small/model_20000.pt --step_by_step

    # Generate from scratch (no prompt)
    python scripts/generate_workflow.py --checkpoint ~/Desktop/checkpoints/tct_small/model_20000.pt --from_scratch --num_samples 5

    # Generate from a partial workflow prompt
    python scripts/generate_workflow.py --checkpoint ~/Desktop/checkpoints/tct_small/model_20000.pt --prompt workflow.json

    # Batch generation with custom parameters
    python scripts/generate_workflow.py --checkpoint ~/Desktop/checkpoints/tct_small/model_20000.pt --num_samples 10 --temperature 0.9
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
        prompt_tokens: List of token IDs to start from (empty list for from-scratch generation)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Only sample from top k tokens (0 = no filtering)
        device: Device to run on

    Returns:
        List of generated token IDs (including prompt)
    """
    # Handle empty prompt (from-scratch generation)
    tokens = prompt_tokens.copy() if prompt_tokens else []
    context_size = model.config.sequence_len

    for _ in range(max_new_tokens):
        # Get context window (last context_size tokens)
        if len(tokens) > 0:
            context = tokens[-context_size:] if len(tokens) > context_size else tokens
        else:
            # For very first token, use a minimal context
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

@torch.no_grad()
def predict_next_token(model, tokens, temperature=0.8, top_k=40, device="cuda"):
    """
    Predict the next single token given a context.

    Args:
        model: Trained GPT model
        tokens: List of context token IDs
        temperature: Sampling temperature
        top_k: Top-k filtering
        device: Device to run on

    Returns:
        tuple: (next_token_id, top_k_tokens_with_probs)
    """
    context_size = model.config.sequence_len

    # Get context window
    if len(tokens) > 0:
        context = tokens[-context_size:] if len(tokens) > context_size else tokens
    else:
        context = [0]

    # Pad if needed
    if len(context) < context_size:
        context = [0] * (context_size - len(context)) + context

    # Forward pass
    x = torch.tensor([context], dtype=torch.long, device=device)
    logits = model(x)[0, -1, :]  # Get last position logits

    # Apply temperature
    if temperature > 0:
        logits = logits / temperature

    # Get top-k predictions with probabilities
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, min(top_k, logits.size(-1)))

    # Apply top-k filtering for sampling
    if top_k > 0:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[-1]] = -float('Inf')

    # Sample
    filtered_probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(filtered_probs, num_samples=1).item()

    # Return next token and top-k predictions for display
    top_predictions = [
        (idx.item(), prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

    return next_token, top_predictions


def step_by_step_mode(model, model_config, device, temperature, top_k):
    """Step-by-step token prediction mode."""
    print("\n" + "="*80)
    print("🔍 STEP-BY-STEP TOKEN PREDICTION")
    print("="*80)
    print("\nThis mode shows you how the model predicts token-by-token.")
    print("\nCommands during prediction:")
    print("  - Press ENTER to predict next token")
    print("  - Type 'show' to see current workflow")
    print("  - Type 'reset' to start over")
    print("  - Type 'exit' to quit\n")

    tokens = []

    while True:
        print("-" * 80)
        print("Choose starting point:")
        print("  1. Press ENTER to start from scratch (empty)")
        print("  2. Paste partial JSON and press Ctrl+D")
        print("  3. Type 'exit' to quit")
        print()

        lines = []
        try:
            while True:
                try:
                    line = input()
                    if line.strip().lower() in ['exit', 'quit', 'q']:
                        print("\n👋 Goodbye!")
                        return
                    if line == "" and len(lines) == 0:
                        break
                    lines.append(line)
                except EOFError:
                    break
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return

        # Initialize tokens
        if lines:
            context_json = "\n".join(lines).strip()
            try:
                json.loads(context_json)
                tokens = list(encode(context_json))
                print(f"\n✅ Encoded {len(tokens)} tokens from context")
                print(f"📋 Starting context:\n{context_json}\n")
            except json.JSONDecodeError as e:
                print(f"\n❌ Invalid JSON: {e}")
                continue
        else:
            tokens = []
            print(f"\n🎲 Starting from scratch (empty context)")
            print(f"The model will generate the very first token of a workflow!")
            print(f"Press ENTER to see the first prediction...\n")

        # Prediction loop
        step = 0
        while True:
            print("-" * 80)
            command = input(f"[Step {step}] Press ENTER for next token ('show'/'reset'/'exit'): ").strip().lower()

            if command in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                return
            elif command == 'reset':
                print("\n🔄 Resetting...\n")
                break
            elif command == 'show':
                try:
                    current_json, consumed, total = decode(tokens)
                    workflow = json.loads(current_json)
                    print("\n" + "="*80)
                    print("📋 CURRENT WORKFLOW")
                    print("="*80)
                    print(json.dumps(workflow, indent=2))
                    print("="*80)
                    print(f"Tokens: {len(tokens)} (consumed: {consumed}/{total})")
                    print("="*80 + "\n")
                except json.JSONDecodeError:
                    print(f"\n⚠️  Not yet valid JSON ({len(tokens)} tokens)\n")
                except Exception as e:
                    print(f"\n⚠️  Decode error: {e}\n")
                continue
            elif command == '':
                # Predict next token
                next_token, top_predictions = predict_next_token(
                    model, tokens, temperature, top_k, device
                )

                tokens.append(next_token)
                step += 1

                print(f"\n✨ Predicted token: {next_token}")
                print(f"📊 Top 5 predictions:")
                for i, (token_id, prob) in enumerate(top_predictions[:5]):
                    print(f"   {i+1}. Token {token_id}: {prob*100:.2f}%")

                # Try to decode and show progress
                try:
                    current_json, consumed, total = decode(tokens)

                    # Show a snippet
                    snippet_len = 100
                    if len(current_json) > snippet_len:
                        snippet = current_json[:snippet_len] + "..."
                    else:
                        snippet = current_json

                    print(f"\n📄 Current output:\n{snippet}")

                    # Check if valid JSON
                    try:
                        workflow = json.loads(current_json)
                        print(f"✅ Valid JSON! (tokens: {consumed}/{total})")

                        # Check schema
                        has_on = "on" in workflow
                        has_jobs = "jobs" in workflow
                        if has_on and has_jobs:
                            print(f"✅ Schema complete!")
                    except json.JSONDecodeError:
                        print(f"⏳ Not yet valid JSON (tokens: {consumed}/{total})")

                except Exception as e:
                    print(f"⚠️  Decode incomplete: {e}")

                print()
            else:
                print(f"Unknown command: {command}")


def interactive_mode(model, model_config, device, temperature, top_k, max_tokens):
    """Interactive mode for testing workflow generation and completion."""
    print("\n" + "="*80)
    print("🚀 INTERACTIVE WORKFLOW GENERATOR")
    print("="*80)
    print("\nOptions:")
    print("  1. Generate from scratch: Press ENTER on empty line")
    print("  2. Complete partial workflow: Paste JSON, then press Ctrl+D (Linux/Mac) or Ctrl+Z+ENTER (Windows)")
    print("  3. Exit: Type 'exit' or 'quit'\n")

    while True:
        print("-" * 80)
        print("📝 Enter partial workflow JSON (or ENTER for from-scratch, 'exit' to quit):")
        print()

        # Read multiline input
        lines = []
        try:
            while True:
                try:
                    line = input()
                    if line.strip().lower() in ['exit', 'quit', 'q']:
                        print("\n👋 Goodbye!")
                        return
                    if line == "" and len(lines) == 0:
                        # Empty input = generate from scratch
                        break
                    lines.append(line)
                except EOFError:
                    # Ctrl+D pressed
                    break
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            return

        prompt_json = "\n".join(lines).strip() if lines else None

        # Encode prompt
        if prompt_json:
            try:
                # Validate JSON
                json.loads(prompt_json)
                prompt_tokens = encode(prompt_json)
                print(f"\n✅ Valid JSON prompt ({len(prompt_tokens)} tokens)")
                print(f"📋 Prompt:\n{prompt_json}\n")
            except json.JSONDecodeError as e:
                print(f"\n❌ Invalid JSON: {e}")
                print("Please try again with valid JSON.\n")
                continue
        else:
            # Generate from scratch
            prompt_tokens = []
            print(f"\n🎲 Generating workflow from scratch...")

        print("⏳ Generating...\n")

        # Generate
        generated_tokens = generate(
            model,
            prompt_tokens,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device
        )

        # Decode
        try:
            decoded_json, consumed, total = decode(generated_tokens)
            workflow = json.loads(decoded_json)

            print("=" * 80)
            print("✅ GENERATED WORKFLOW")
            print("=" * 80)
            print(json.dumps(workflow, indent=2))
            print("=" * 80)
            print(f"📊 Stats:")
            print(f"  - Total tokens: {len(generated_tokens)}")
            print(f"  - New tokens: {len(generated_tokens) - len(prompt_tokens)}")
            print(f"  - Tokens consumed by decoder: {consumed}/{total}")

            # Check schema
            has_name = "name" in workflow
            has_on = "on" in workflow
            has_jobs = "jobs" in workflow
            if has_name and has_on and has_jobs:
                print(f"  - Schema: ✅ Valid (has name, on, jobs)")
            else:
                print(f"  - Schema: ⚠️  Incomplete (name={has_name}, on={has_on}, jobs={has_jobs})")

            print("=" * 80)
            print()

        except json.JSONDecodeError as e:
            print("=" * 80)
            print("❌ INVALID JSON OUTPUT")
            print("=" * 80)
            print(f"Decode error: {e}")
            print(f"\nRaw output:\n{decoded_json}")
            print("=" * 80)
            print()
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate workflows with trained TCT model")
    parser.add_argument("--checkpoint", type=str, default="~/Desktop/checkpoints/tct_small/model_020000.pt",
                      help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default=None,
                      help="Path to JSON workflow prompt (optional)")
    parser.add_argument("--interactive", action="store_true",
                      help="Interactive mode for testing")
    parser.add_argument("--step_by_step", action="store_true",
                      help="Step-by-step token prediction mode")
    parser.add_argument("--from_scratch", action="store_true",
                      help="Generate from scratch (no prompt)")
    parser.add_argument("--num_samples", type=int, default=1,
                      help="Number of samples to generate")
    parser.add_argument("--max_tokens", type=int, default=512,
                      help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.9,
                      help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=100,
                      help="Top-k sampling parameter")
    parser.add_argument("--device", type=str, default="",
                      help="Device to use (cuda/cpu/mps, empty=autodetect)")

    args = parser.parse_args()

    # Expand paths
    checkpoint_path = Path(args.checkpoint).expanduser()

    # Auto-detect device if not specified
    if args.device == "":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Auto-detected device: {device}")
    else:
        device = args.device

    # Load model
    model, model_config = load_model(checkpoint_path, device=device)

    # Interactive mode
    if args.interactive:
        interactive_mode(model, model_config, device, args.temperature, args.top_k, args.max_tokens)
        return

    # Step-by-step mode
    if args.step_by_step:
        step_by_step_mode(model, model_config, device, args.temperature, args.top_k)
        return

    # Prepare prompt
    if args.from_scratch:
        # Generate from scratch (no prompt)
        prompt_tokens = []
        print("🎲 Generating workflows from scratch (no prompt)")
        print()
    elif args.prompt:
        # Use custom prompt from file
        with open(args.prompt) as f:
            prompt_workflow = json.load(f)
        prompt_json = json.dumps(prompt_workflow)
        prompt_tokens = encode(prompt_json)
        print(f"Using custom prompt from {args.prompt}")
        print(f"Prompt: {prompt_json}")
        print(f"Encoded to {len(prompt_tokens)} tokens")
        print()
    else:
        # Use minimal default prompt
        prompt_workflow = {
            "name": "CI",
            "on": "push",
            "jobs": {}
        }
        prompt_json = json.dumps(prompt_workflow)
        prompt_tokens = encode(prompt_json)
        print("Using default minimal prompt")
        print(f"Prompt: {prompt_json}")
        print(f"Encoded to {len(prompt_tokens)} tokens")
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
            device=device
        )

        new_tokens = len(generated_tokens) - len(prompt_tokens)
        print(f"Generated {len(generated_tokens)} tokens total ({new_tokens} new)")

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
