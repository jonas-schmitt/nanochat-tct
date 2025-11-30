#!/usr/bin/env python3
"""Test Kubernetes manifest generation with trained model."""

import torch
import json
import yaml
import tct_kubernetes
from pathlib import Path
from nanochat.gpt import GPT, GPTConfig

# Load the best checkpoint
checkpoint_path = "checkpoints/k8s_baseline_v1/model_200000.pt"
print(f"Loading checkpoint: {checkpoint_path}")

# Load model config (small-2048)
vocab_size = 20000
context_size = 2048
d_model = 384
n_layers = 8
n_heads = 6

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config_kwargs = dict(
    sequence_len=context_size,
    vocab_size=vocab_size,
    n_layer=n_layers,
    n_head=n_heads,
    n_kv_head=n_heads,
    n_embd=d_model,
)

model_config = GPTConfig(**model_config_kwargs)
model = GPT(model_config)
model.to(device)

# Load checkpoint (state dict is saved directly, not wrapped)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print(f"✅ Model loaded successfully")
print(f"   Device: {device}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# Generate manifests
print("=" * 80)
print("GENERATING KUBERNETES MANIFESTS")
print("=" * 80)
print()

num_samples = 20
max_tokens = 512
output_dir = Path("generated_manifests")
temperature = 0.8
top_k = 50

# Load some real sequences from the training cache to use as prompts
print("Loading real sequences from training data...")
cache_path = "/home/josch/Desktop/data/.cache/tokenized_k8s_split90_254908files.pt"
all_sequences = torch.load(cache_path)
print(f"Loaded {len(all_sequences)} sequences")

# Pick sequences for generation
import random
random.seed(42)
PAD_TOKEN = 19999  # Kubernetes vocab PAD token
max_retries = 5

def generate_one(seq_idx):
    """Generate a manifest from a prompt sequence. Returns (success, manifest_or_error)."""
    full_sequence = all_sequences[seq_idx].tolist()
    prompt_len = min(20, len(full_sequence) // 4)
    start_tokens = full_sequence[:prompt_len]

    # Convert to tensor
    x = torch.tensor([start_tokens], dtype=torch.long, device=device)
    generated_count = 0

    with torch.no_grad():
        for step in range(max_tokens):
            logits = model(x)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == PAD_TOKEN:
                break

            x = torch.cat([x, next_token], dim=1)
            generated_count += 1

            if x.size(1) >= context_size:
                break

    generated_tokens = x[0].tolist()

    # Try to decode
    try:
        decoded = tct_kubernetes.decode(generated_tokens)
        manifest_str = decoded[0] if isinstance(decoded, tuple) else decoded
        manifest = json.loads(manifest_str)
        return True, manifest, len(full_sequence), prompt_len, generated_count
    except Exception as e:
        return False, str(e), len(full_sequence), prompt_len, generated_count

# Create output directory
output_dir.mkdir(exist_ok=True)

# Generate samples with retry on failure
successful = 0
generated_manifests = []

for i in range(num_samples):
    print(f"Sample {i+1}/{num_samples}:")
    print("-" * 80)

    for attempt in range(max_retries):
        seq_idx = random.randint(0, len(all_sequences) - 1)
        success, result, full_len, prompt_len, gen_count = generate_one(seq_idx)

        if success:
            manifest = result
            print(f"Sequence #{seq_idx} (attempt {attempt + 1})")
            print(f"Full: {full_len} tokens, Prompt: {prompt_len} tokens, Generated: {gen_count} tokens")
            print("✅ Valid JSON!")
            print(f"   Kind: {manifest.get('kind', 'N/A')}")
            print(f"   API Version: {manifest.get('apiVersion', 'N/A')}")
            print(f"   Has metadata: {manifest.get('metadata') is not None}")
            print(f"   Has spec: {manifest.get('spec') is not None}")
            print()
            print("Generated manifest:")
            print(json.dumps(manifest, indent=2))
            generated_manifests.append(manifest)

            # Save to individual file
            kind = manifest.get('kind', 'unknown').lower()
            name = manifest.get('metadata', {}).get('name', f'manifest-{i+1}')
            filename = f"{kind}-{name}.yaml"
            filepath = output_dir / filename
            with open(filepath, 'w') as f:
                yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
            print(f"📁 Saved to {filepath}")

            successful += 1
            break
        else:
            print(f"  Attempt {attempt + 1}: ❌ {result[:80]}... retrying")
    else:
        print(f"  Failed after {max_retries} attempts")

    print()
    print()

print("=" * 80)
print(f"Generation complete! {successful}/{num_samples} successful")
print("=" * 80)

print(f"\n✅ Saved {len(generated_manifests)} manifests to {output_dir}/")
