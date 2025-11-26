#!/usr/bin/env python3
"""
Kubernetes Manifest Autocompletion with Streaming Decode

Demonstrates autocompletion of Kubernetes manifests using the trained 200k model
with the streaming decode_prefix() for truncation-tolerant decoding.

Features:
- Uses decode_prefix() for graceful handling of incomplete token sequences
- Shows real-time progress as tokens are generated
- Early stopping when manifest is complete
"""

import torch
import json
import sys
import random
from pathlib import Path

import tct_kubernetes_streaming as tct
from nanochat.gpt import GPT, GPTConfig


# Constants
VOCAB_SIZE = 20000
CONTEXT_SIZE = 2048
PAD_TOKEN = tct.pad_token()


def load_model(checkpoint_path: str, device: torch.device) -> GPT:
    """Load the trained model from checkpoint."""
    model_config = GPTConfig(
        sequence_len=CONTEXT_SIZE,
        vocab_size=VOCAB_SIZE,
        n_layer=8,
        n_head=6,
        n_kv_head=6,
        n_embd=384,
    )

    model = GPT(model_config)
    model.to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def generate_with_streaming(
    model: GPT,
    prompt_tokens: list[int],
    device: torch.device,
    max_tokens: int = 400,
    temperature: float = 0.7,
    top_k: int = 50,
    decode_interval: int = 10,
    show_progress: bool = True,
) -> tuple[list[int], dict | None]:
    """
    Generate tokens with streaming decode_prefix to show progress.

    Returns:
        Tuple of (generated_tokens, final_manifest_or_None)
    """
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated_tokens = list(prompt_tokens)

    last_kind = ""
    last_name = ""

    if show_progress:
        print(f"Starting with {len(prompt_tokens)} prompt tokens")
        print("-" * 60)

    with torch.no_grad():
        for step in range(max_tokens):
            # Generate next token
            logits = model(x)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            # Check for pad token (end of manifest)
            if token_id == PAD_TOKEN:
                if show_progress:
                    print(f"[{len(generated_tokens):3d} tokens] PAD - generation complete")
                break

            generated_tokens.append(token_id)
            x = torch.cat([x, next_token], dim=1)

            # Streaming decode with decode_prefix
            if show_progress and len(generated_tokens) % decode_interval == 0:
                json_str, fields_decoded, is_complete = tct.decode_prefix(generated_tokens)
                try:
                    result = json.loads(json_str)
                    kind = result.get('kind', '')
                    name = result.get('metadata', {}).get('name', '')

                    # Show progress when we get new info
                    if kind and (kind != last_kind or name != last_name):
                        status = "COMPLETE" if is_complete else "partial"
                        print(f"[{len(generated_tokens):3d} tokens] {status:8} | kind={kind:20} | name={name or '-'}")
                        last_kind = kind
                        last_name = name

                    # Early stopping if complete
                    if is_complete:
                        print(f"[{len(generated_tokens):3d} tokens] Manifest complete - stopping early")
                        break
                except json.JSONDecodeError:
                    pass

            # Safety limit
            if len(generated_tokens) >= CONTEXT_SIZE:
                if show_progress:
                    print(f"[{len(generated_tokens):3d} tokens] Context limit reached")
                break

    if show_progress:
        print("-" * 60)

    # Final decode with decode_prefix
    json_str, fields_decoded, is_complete = tct.decode_prefix(generated_tokens)
    try:
        manifest = json.loads(json_str)
        if show_progress:
            print(f"Final: {len(generated_tokens)} tokens, fields={fields_decoded}, complete={is_complete}")
        return generated_tokens, manifest
    except json.JSONDecodeError as e:
        if show_progress:
            print(f"Final decode failed: {e}")
        return generated_tokens, None


def demo_autocompletion():
    """Demonstrate autocompletion on example prompts from training data."""

    print("=" * 70)
    print("KUBERNETES MANIFEST AUTOCOMPLETION WITH STREAMING decode_prefix()")
    print("=" * 70)
    print()

    # Load model
    checkpoint_path = "checkpoints/k8s_baseline_v1/model_200000.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model: {checkpoint_path}")
    print(f"Device: {device}")
    model = load_model(checkpoint_path, device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} parameters)")
    print()

    # Load training data for prompts
    cache_path = Path("/home/josch/Desktop/data/.cache/tokenized_k8s_split90_254908files.pt")
    print(f"Loading training sequences from {cache_path}")
    sequences = torch.load(cache_path)
    print(f"Loaded {len(sequences)} sequences")
    print()

    # Select diverse examples
    random.seed(42)

    examples = []

    # Short sequences (simple resources)
    short_seqs = [i for i, s in enumerate(sequences)
                  if 20 < len([t for t in s.tolist() if t != PAD_TOKEN]) < 60]
    if short_seqs:
        examples.extend(random.sample(short_seqs, min(2, len(short_seqs))))

    # Medium sequences (Services, Deployments)
    medium_seqs = [i for i, s in enumerate(sequences)
                   if 60 <= len([t for t in s.tolist() if t != PAD_TOKEN]) < 200]
    if medium_seqs:
        examples.extend(random.sample(medium_seqs, min(2, len(medium_seqs))))

    # Longer sequences (complex resources)
    long_seqs = [i for i, s in enumerate(sequences)
                 if 200 <= len([t for t in s.tolist() if t != PAD_TOKEN]) < 500]
    if long_seqs:
        examples.extend(random.sample(long_seqs, min(1, len(long_seqs))))

    results = []

    for i, seq_idx in enumerate(examples):
        print()
        print("=" * 70)
        print(f"EXAMPLE {i+1}/{len(examples)} (sequence #{seq_idx})")
        print("=" * 70)
        print()

        # Get sequence and remove padding
        full_seq = sequences[seq_idx].tolist()
        full_seq = [t for t in full_seq if t != PAD_TOKEN]

        # Decode original to show what we're completing
        orig_json, orig_fields, orig_complete = tct.decode_prefix(full_seq)
        try:
            orig_manifest = json.loads(orig_json)
            orig_kind = orig_manifest.get('kind', 'unknown')
            orig_name = orig_manifest.get('metadata', {}).get('name', 'unknown')
            print(f"Original: {orig_kind}/{orig_name} ({len(full_seq)} tokens)")
        except:
            print(f"Original: (decode failed) ({len(full_seq)} tokens)")
            continue

        # Use first N tokens as prompt (about 25% of the sequence)
        prompt_len = max(10, len(full_seq) // 4)
        prompt_tokens = full_seq[:prompt_len]

        print(f"Using first {prompt_len} tokens as prompt...")
        print()

        # Generate completion with streaming progress
        generated, manifest = generate_with_streaming(
            model=model,
            prompt_tokens=prompt_tokens,
            device=device,
            max_tokens=500,
            temperature=0.7,
            top_k=50,
            decode_interval=10,
            show_progress=True,
        )

        if manifest:
            gen_kind = manifest.get('kind', 'unknown')
            gen_name = manifest.get('metadata', {}).get('name', 'unknown')
            print()
            print(f"Generated: {gen_kind}/{gen_name}")
            print()
            print("Manifest preview:")
            print(json.dumps(manifest, indent=2)[:600])
            if len(json.dumps(manifest)) > 600:
                print("... (truncated)")

            results.append({
                'seq_idx': seq_idx,
                'original_kind': orig_kind,
                'generated_kind': gen_kind,
                'prompt_tokens': prompt_len,
                'generated_tokens': len(generated),
                'success': True
            })
        else:
            print()
            print("Generation failed to produce valid manifest")
            results.append({
                'seq_idx': seq_idx,
                'original_kind': orig_kind,
                'success': False
            })

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    successful = sum(1 for r in results if r['success'])
    print(f"Successful completions: {successful}/{len(results)}")
    print()

    for i, r in enumerate(results):
        if r['success']:
            print(f"  {i+1}. ✅ {r['original_kind']:15} → {r['generated_kind']:15} ({r['prompt_tokens']} → {r['generated_tokens']} tokens)")
        else:
            print(f"  {i+1}. ❌ {r.get('original_kind', 'unknown')}")

    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(demo_autocompletion())
