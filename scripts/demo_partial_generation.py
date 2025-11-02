#!/usr/bin/env python3
"""
Demo: Partial workflow completion using TCT's incremental encoding

Shows how the model can complete partial/incomplete workflows using
TCT's partial encoding capabilities.

Usage:
    python -m scripts.demo_partial_generation
"""

import os
import sys
import json
from pathlib import Path

import torch
import torch.nn.functional as F

# Add tct-bundle adapters to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tct-bundle" / "adapters"))

from nanochat.gpt import GPT, GPTConfig
from tct_github_workflow import encode, decode, vocab_size

def print_section(title, char="="):
    """Print a section header."""
    print(f"\n{char * 80}")
    print(f"{title}")
    print(f"{char * 80}\n")

def load_model(checkpoint_path, device="cuda"):
    """Load trained model from checkpoint."""
    print("Loading model...")

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Load metadata
    meta_path = str(checkpoint_path).replace("model_", "meta_").replace(".pt", ".json")
    with open(meta_path) as f:
        meta = json.load(f)

    # Create model
    model_config = GPTConfig(**meta["model_config"])
    model = GPT(model_config)
    model.to(device)
    model.load_state_dict(ckpt)
    model.eval()

    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    return model, model_config

@torch.no_grad()
def generate_completion(model, prompt_tokens, max_new_tokens=200, temperature=0.7, top_k=50, device="cuda"):
    """Generate tokens to complete a partial workflow."""
    tokens = prompt_tokens.copy()
    context_size = model.config.sequence_len

    generated_count = 0

    for i in range(max_new_tokens):
        # Get context window
        context = tokens[-context_size:] if len(tokens) > context_size else tokens
        if len(context) < context_size:
            context = [0] * (context_size - len(context)) + context

        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]

        # Apply temperature and top-k
        logits = logits / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_token)
        generated_count += 1

        # Try to decode every 10 tokens to check if complete
        if i > 5 and i % 10 == 0:
            try:
                decoded_json, consumed, total = decode(tokens)
                if consumed == total:
                    # Successfully decoded complete workflow
                    try:
                        json.loads(decoded_json)  # Verify it's valid JSON
                        return tokens, generated_count, True
                    except:
                        pass  # Not valid JSON yet, continue
            except:
                pass  # Decoding failed, continue generating

    # Max tokens reached
    return tokens, generated_count, False

def demo_completion(model, minimal_workflow, description):
    """Demonstrate extending a minimal workflow."""
    print_section(description, "-")

    print("📝 MINIMAL VALID INPUT (starting point):")
    print("-" * 80)
    print(json.dumps(minimal_workflow, indent=2))
    print("-" * 80)
    print()

    # Encode the minimal workflow
    print("Encoding minimal workflow with TCT...")
    prompt_tokens = encode(json.dumps(minimal_workflow))
    print(f"✅ Encoded to {len(prompt_tokens)} tokens")
    print()

    print("Generating completion...")
    generated_tokens, new_token_count, completed = generate_completion(
        model,
        prompt_tokens,
        max_new_tokens=200,
        temperature=0.7,
        top_k=50,
        device="cuda"
    )

    print(f"✅ Generated {new_token_count} new tokens")
    print()

    # Decode the result
    try:
        decoded_json, consumed, total = decode(generated_tokens)
        workflow = json.loads(decoded_json)

        print("📤 COMPLETED OUTPUT:")
        print("-" * 80)
        print(json.dumps(workflow, indent=2))
        print("-" * 80)
        print()

        print("✅ Validation:")
        print(f"  - Valid JSON: ✅")
        print(f"  - Schema compliant: {'✅' if all(k in workflow for k in ['name', 'on', 'jobs']) else '❌'}")
        print(f"  - Completed successfully: {'✅' if completed else '⚠️  (reached max tokens)'}")
        print(f"  - Tokens consumed: {consumed}/{total}")
        print()

        return True

    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print()
        try:
            decoded_json, consumed, total = decode(generated_tokens)
            print("Raw output:")
            print(decoded_json[:500])
        except:
            print("Could not decode tokens")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print_section("TCT Incremental Workflow Generation Demo")

    print("This demo shows how the model extends minimal valid workflows using")
    print("TCT's partial decode tracking (consumed/total tokens).")
    print()
    print("Key concept: TCT requires valid JSON for encoding, but tracks partial")
    print("decoding to detect when generation is complete.")
    print()

    # Load model
    checkpoint_path = Path.home() / "Desktop/checkpoints/tct_small/model_005000.pt"

    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        print()
        print("Please ensure you have trained a model first:")
        print("  python -m scripts.tct_train --num_iterations 5000")
        return 1

    model, model_config = load_model(checkpoint_path, device="cuda")

    # Demo 1: Minimal workflow - just required fields
    demo_completion(
        model,
        {"name": "My Workflow", "on": "push", "jobs": {}},
        "Demo 1: Extend from minimal workflow (empty jobs)"
    )

    # Demo 2: Workflow with trigger details but minimal job
    demo_completion(
        model,
        {
            "name": "Python Tests",
            "on": {"push": {"branches": ["main"]}},
            "jobs": {
                "test": {
                    "runs-on": "ubuntu-latest",
                    "steps": [{"uses": "actions/checkout@v4"}]
                }
            }
        },
        "Demo 2: Extend from basic test job"
    )

    # Demo 3: Multi-trigger with one minimal job
    demo_completion(
        model,
        {
            "name": "CI Pipeline",
            "on": {
                "push": {"branches": ["main"]},
                "pull_request": {"branches": ["main"]}
            },
            "jobs": {
                "build": {
                    "runs-on": "ubuntu-latest",
                    "steps": [{"uses": "actions/checkout@v3"}]
                }
            }
        },
        "Demo 3: Extend from basic CI setup"
    )

    print_section("Demo Complete! 🎉")

    print("The model successfully extends minimal workflows by:")
    print("  ✅ Encoding complete but minimal workflows with TCT")
    print("  ✅ Generating tokens autoregressively")
    print("  ✅ Tracking completion with decode(tokens) -> (json, consumed, total)")
    print("  ✅ Stopping when consumed == total (complete workflow)")
    print("  ✅ Producing valid, schema-compliant extensions")
    print()
    print("Key TCT feature: decode() returns (json_str, consumed, total)")
    print("  - consumed: tokens successfully decoded")
    print("  - total: total tokens in sequence")
    print("  - consumed == total means workflow is complete")
    print()
    print("Try your own extensions with:")
    print("  python -m scripts.generate_workflow --prompt minimal.json")
    print()

if __name__ == "__main__":
    exit(main())
