#!/usr/bin/env python3
"""
Test TCT model generation capabilities.

Tests:
1. Continuation from partial sequences (25%, 50%, 75% prefix)
2. Generation from scratch with fixed token counts
3. Token-level prediction accuracy

Usage:
    python scripts/test_tct_generation.py --checkpoint checkpoints/tsconfig_tct_small_runpod_new
    python scripts/test_tct_generation.py --checkpoint checkpoints/tsconfig_tct_small_runpod_new --temperature 0.5
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F


def load_model(checkpoint_dir: Path, device: str = "cuda"):
    """Load model from checkpoint directory."""
    from nanochat.gpt import GPT, GPTConfig

    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    model_cfg = config_dict.get("model_config", config_dict)

    model_config = GPTConfig(
        vocab_size=model_cfg.get("vocab_size", 258),
        sequence_len=model_cfg.get("sequence_len", 2048),
        n_layer=model_cfg.get("n_layer", 16),
        n_head=model_cfg.get("n_head", 8),
        n_kv_head=model_cfg.get("n_kv_head", 8),
        n_embd=model_cfg.get("n_embd", 512),
        use_swiglu=model_cfg.get("use_swiglu", True),
        ffn_mult=model_cfg.get("ffn_mult", 2.5),
    )

    model = GPT(model_config)

    # Find latest checkpoint
    checkpoint_files = list(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    latest = max(checkpoint_files, key=lambda p: int(p.stem.split("_")[1]))
    print(f"Loading checkpoint: {latest}")

    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)

    return model, config_dict


def load_tct_module(schema: str):
    """Load the appropriate TCT tokenizer module."""
    module_map = {
        "tsconfig": "tct_tsconfig_base",
        "eslintrc": "tct_eslintrc_bpe_500",
        "kubernetes": "tct_kubernetes_bpe_1k",
    }

    module_name = module_map.get(schema)
    if not module_name:
        raise ValueError(f"Unknown schema: {schema}")

    import importlib
    return importlib.import_module(module_name.replace("-", "_"))


def generate_tokens(model, prefix_tokens: list, num_tokens: int, temperature: float = 0.8, top_k: int = 50, device: str = "cuda"):
    """Generate exactly num_tokens starting from prefix_tokens."""
    tokens = list(prefix_tokens)

    for _ in range(num_tokens):
        input_ids = torch.tensor([tokens], device=device)
        with torch.no_grad():
            logits = model(input_ids)[0, -1, :]

        if temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token = torch.argmax(logits).item()

        tokens.append(next_token)

    return tokens


def test_continuation(model, tct_module, val_samples: list, temperature: float, device: str, num_samples: int = 10):
    """Test model's ability to continue partial sequences."""
    print("\n" + "=" * 70)
    print("CONTINUATION TEST")
    print(f"Temperature: {temperature}, Samples: {num_samples}")
    print("=" * 70)

    results = {25: [], 50: [], 75: []}

    for sample_tokens in val_samples[:num_samples]:
        full_len = len(sample_tokens)

        for pct in [25, 50, 75]:
            prefix_len = int(full_len * pct / 100)
            remaining = full_len - prefix_len

            prefix = sample_tokens[:prefix_len]
            completed = generate_tokens(model, prefix, remaining, temperature=temperature, device=device)

            # Token-level accuracy
            matches = sum(1 for a, b in zip(completed, sample_tokens) if a == b)
            accuracy = matches / len(sample_tokens) * 100

            # Check if valid decode
            try:
                decoded, _, _ = tct_module.decode(completed)
                valid = True
            except:
                valid = False

            results[pct].append({"accuracy": accuracy, "valid": valid})

    # Print summary
    for pct in [25, 50, 75]:
        accs = [r["accuracy"] for r in results[pct]]
        valids = [r["valid"] for r in results[pct]]
        avg_acc = sum(accs) / len(accs)
        valid_rate = sum(valids) / len(valids) * 100
        print(f"\n{pct}% prefix:")
        print(f"  Avg token accuracy: {avg_acc:.1f}%")
        print(f"  Valid JSON rate:    {valid_rate:.0f}%")

    return results


def test_generation_from_scratch(model, tct_module, temperature: float, device: str, num_samples: int = 10):
    """Test model's ability to generate from scratch."""
    print("\n" + "=" * 70)
    print("GENERATION FROM SCRATCH")
    print(f"Temperature: {temperature}, Samples: {num_samples}")
    print("=" * 70)

    results = {}

    for num_tokens in [100, 200, 300]:
        valid_count = 0
        samples = []

        for _ in range(num_samples):
            tokens = generate_tokens(model, [0], num_tokens - 1, temperature=temperature, device=device)

            try:
                decoded, consumed, surplus = tct_module.decode(tokens)
                valid = True
                samples.append(decoded)
            except:
                try:
                    partial, consumed, complete = tct_module.decode_prefix(tokens)
                    valid = False
                    samples.append(f"[partial: {partial[:50]}...]")
                except:
                    valid = False
                    samples.append("[decode failed]")

            if valid:
                valid_count += 1

        results[num_tokens] = {"valid_rate": valid_count / num_samples * 100, "samples": samples}

        print(f"\n{num_tokens} tokens:")
        print(f"  Valid JSON rate: {valid_count}/{num_samples} ({valid_count/num_samples*100:.0f}%)")
        print(f"  Sample outputs:")
        for i, s in enumerate(samples[:3]):
            print(f"    {i+1}. {s[:80]}...")

    return results


def test_token_prediction(model, tct_module, val_samples: list, device: str, num_samples: int = 10):
    """Test model's next-token prediction accuracy at various positions."""
    print("\n" + "=" * 70)
    print("TOKEN PREDICTION ACCURACY")
    print("=" * 70)

    positions = [1, 5, 10, 20, 50, 100]
    accuracy_by_pos = {p: [] for p in positions}

    for sample_tokens in val_samples[:num_samples]:
        for pos in positions:
            if pos >= len(sample_tokens):
                continue

            prefix = sample_tokens[:pos]
            actual_next = sample_tokens[pos]

            input_ids = torch.tensor([prefix], device=device)
            with torch.no_grad():
                logits = model(input_ids)[0, -1, :]

            predicted = torch.argmax(logits).item()
            accuracy_by_pos[pos].append(1.0 if predicted == actual_next else 0.0)

    print("\nGreedy prediction accuracy by position:")
    for pos in positions:
        if accuracy_by_pos[pos]:
            acc = sum(accuracy_by_pos[pos]) / len(accuracy_by_pos[pos]) * 100
            print(f"  Position {pos:3d}: {acc:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Test TCT model generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--schema", type=str, default="tsconfig", help="Schema name")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    # Load model
    checkpoint_dir = Path(args.checkpoint)
    model, config = load_model(checkpoint_dir, args.device)

    # Get schema from config if not specified
    schema = config.get("schema", args.schema)
    print(f"Schema: {schema}")

    # Load TCT module
    tct_module = load_tct_module(schema)
    print(f"TCT vocab size: {tct_module.vocab_size()}")

    # Find data directory
    data_dir_map = {
        "tsconfig": "tsconfig-tct-base",
        "eslintrc": "eslintrc-tct-bpe-500",
        "kubernetes": "kubernetes-tct-bpe-1k",
    }

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Try config path first
        schema_config = config.get("schema_config", {})
        data_path = schema_config.get("data_path_tct")
        if data_path and Path(data_path).exists():
            data_dir = Path(data_path)
        else:
            # Fall back to local paths
            for data_root in [Path.home() / "Desktop" / "data", Path.home() / "git" / "data"]:
                candidate = data_root / data_dir_map.get(schema, f"{schema}-tct")
                if candidate.exists():
                    data_dir = candidate
                    break
            else:
                raise FileNotFoundError(f"Data directory not found for schema {schema}")

    print(f"Data dir: {data_dir}")

    # Load validation samples
    val_path = data_dir / "validate.jsonl"
    if not val_path.exists():
        print(f"Warning: {val_path} not found, using all.jsonl")
        val_path = data_dir / "all.jsonl"

    with open(val_path) as f:
        val_samples = [json.loads(line.strip()) for line in f.readlines()[:args.num_samples * 2]]

    print(f"Loaded {len(val_samples)} validation samples")

    # Run tests
    test_token_prediction(model, tct_module, val_samples, args.device, args.num_samples)
    test_continuation(model, tct_module, val_samples, args.temperature, args.device, args.num_samples)
    test_generation_from_scratch(model, tct_module, args.temperature, args.device, args.num_samples)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
