#!/usr/bin/env python3
"""
Analyze generation quality across training checkpoints.

Generates samples at each checkpoint and measures:
- JSON validity rate
- Schema validity rate (has apiVersion, kind, metadata)
- Completion rate (decode_prefix returns is_complete=True)

Usage:
    python -m scripts.analyze_generation_quality \
        --checkpoint_dir checkpoints_analysis/kubernetes_tct_small \
        --output_dir checkpoints_analysis/results
"""

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt import GPT, GPTConfig


@dataclass
class GenerationResult:
    """Result of a single generation."""
    tokens: List[int]
    json_output: Optional[str]
    is_complete: bool
    is_valid_json: bool
    has_api_version: bool
    has_kind: bool
    has_metadata: bool
    kind_value: Optional[str] = None


@dataclass
class CheckpointMetrics:
    """Aggregated metrics for a checkpoint."""
    epoch: int
    val_loss: float
    num_samples: int
    valid_json_rate: float
    schema_valid_rate: float
    completion_rate: float
    kind_distribution: Dict[str, int]


def load_first_token_distribution(data_dir: Path) -> Dict[int, float]:
    """Compute empirical distribution of first tokens from training data."""
    all_path = data_dir / "all.jsonl"
    if not all_path.exists():
        raise FileNotFoundError(f"Training data not found: {all_path}")

    first_tokens = Counter()
    with open(all_path) as f:
        for line in f:
            tokens = json.loads(line)
            if tokens:
                first_tokens[tokens[0]] += 1

    total = sum(first_tokens.values())
    return {token: count / total for token, count in first_tokens.items()}


def load_checkpoint(checkpoint_path: Path, device: str = "cpu") -> Tuple[GPT, dict]:
    """Load model from checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Model config (infer from weights if not stored)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Infer config from weights
    vocab_size, d_model = state_dict["transformer.wte.weight"].shape
    layer_nums = set()
    for k in state_dict.keys():
        if k.startswith("transformer.h."):
            layer_nums.add(int(k.split(".")[2]))
    n_layers = max(layer_nums) + 1

    # Check for SwiGLU (has w_gate)
    use_swiglu = any("w_gate" in k for k in state_dict.keys())

    # Get FFN mult from gate weight shape
    ffn_mult = 2.5  # default
    if use_swiglu:
        gate_key = "transformer.h.0.mlp.w_gate.weight"
        if gate_key in state_dict:
            ffn_dim = state_dict[gate_key].shape[0]
            ffn_mult = ffn_dim / d_model

    config = GPTConfig(
        vocab_size=vocab_size,
        n_embd=d_model,
        n_layer=n_layers,
        n_head=d_model // 64,  # assume head_dim=64
        n_kv_head=d_model // 64,
        sequence_len=2048,
        use_swiglu=use_swiglu,
        ffn_mult=ffn_mult,
    )

    model = GPT(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    metadata = {
        "epoch": ckpt.get("epoch", 0),
        "val_loss": ckpt.get("val_loss", 0.0),
        "step": ckpt.get("step", 0),
    }

    return model, metadata


def generate_sample(
    model: GPT,
    tct_module,
    first_token_distribution: Dict[int, float],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    seed: Optional[int] = None,
    use_bos: bool = False,
) -> GenerationResult:
    """Generate a single sample from the model.

    Args:
        use_bos: If True, start with BOS token (pad_token) and let model predict first token.
                 If False (default), sample first token from training distribution.
                 Use True for models trained with BOS prepending, False for older models.
    """
    device = next(model.parameters()).device

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if use_bos:
        # For BOS-trained models: start with BOS token, model predicts first real token
        bos_token = tct_module.pad_token()
        generated_tokens = [bos_token]
    else:
        # For older models: sample first token from empirical distribution
        first_tokens = list(first_token_distribution.keys())
        first_probs = list(first_token_distribution.values())
        first_token = random.choices(first_tokens, weights=first_probs, k=1)[0]
        generated_tokens = [first_token]

    # Generate remaining tokens
    for _ in range(max_tokens - 1):
        input_ids = torch.tensor([generated_tokens], device=device)

        with torch.no_grad():
            logits = model(input_ids)
            logits = logits[0, -1, :]

        if temperature > 0:
            logits = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated_tokens.append(next_token)

    # Decode using decode_prefix
    json_output = None
    is_complete = False
    is_valid_json = False
    has_api_version = False
    has_kind = False
    has_metadata = False
    kind_value = None

    try:
        json_output, consumed, is_complete = tct_module.decode_prefix(generated_tokens)

        if json_output:
            try:
                parsed = json.loads(json_output)
                is_valid_json = True
                has_api_version = "apiVersion" in parsed
                has_kind = "kind" in parsed
                has_metadata = "metadata" in parsed
                kind_value = parsed.get("kind")
            except json.JSONDecodeError:
                pass
    except Exception:
        pass

    return GenerationResult(
        tokens=generated_tokens,
        json_output=json_output,
        is_complete=is_complete,
        is_valid_json=is_valid_json,
        has_api_version=has_api_version,
        has_kind=has_kind,
        has_metadata=has_metadata,
        kind_value=kind_value,
    )


def analyze_checkpoint(
    checkpoint_path: Path,
    tct_module,
    first_token_distribution: Dict[int, float],
    num_samples: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    device: str = "cpu",
    use_bos: bool = False,
) -> Tuple[CheckpointMetrics, List[GenerationResult]]:
    """Analyze generation quality for a single checkpoint.

    Args:
        use_bos: If True, use BOS token for generation (for BOS-trained models).
                 If False, sample first token from distribution (for older models).
    """
    print(f"  Loading {checkpoint_path.name}...")
    model, metadata = load_checkpoint(checkpoint_path, device)

    results = []
    kind_counts = Counter()

    print(f"  Generating {num_samples} samples...")
    for i in tqdm(range(num_samples), desc="  Generating", leave=False):
        result = generate_sample(
            model=model,
            tct_module=tct_module,
            first_token_distribution=first_token_distribution,
            max_tokens=512,
            temperature=temperature,
            top_k=top_k,
            seed=i,  # Deterministic per sample
            use_bos=use_bos,
        )
        results.append(result)
        if result.kind_value:
            kind_counts[result.kind_value] += 1

    # Compute metrics
    valid_json = sum(1 for r in results if r.is_valid_json)
    schema_valid = sum(1 for r in results if r.has_api_version and r.has_kind and r.has_metadata)
    complete = sum(1 for r in results if r.is_complete)

    metrics = CheckpointMetrics(
        epoch=metadata["epoch"],
        val_loss=metadata["val_loss"],
        num_samples=num_samples,
        valid_json_rate=valid_json / num_samples,
        schema_valid_rate=schema_valid / num_samples,
        completion_rate=complete / num_samples,
        kind_distribution=dict(kind_counts),
    )

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, results


def plot_results(all_metrics: List[CheckpointMetrics], output_dir: Path):
    """Generate plots from the analysis results."""
    import matplotlib.pyplot as plt

    epochs = [m.epoch for m in all_metrics]
    val_losses = [m.val_loss for m in all_metrics]
    valid_json_rates = [m.valid_json_rate * 100 for m in all_metrics]
    schema_valid_rates = [m.schema_valid_rate * 100 for m in all_metrics]
    completion_rates = [m.completion_rate * 100 for m in all_metrics]

    # Plot 1: Quality metrics vs epoch
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, valid_json_rates, "b-o", label="Valid JSON")
    ax1.plot(epochs, schema_valid_rates, "g-s", label="Schema Valid")
    ax1.plot(epochs, completion_rates, "r-^", label="Complete")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Rate (%)")
    ax1.set_title("Generation Quality vs Training Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 105)

    # Plot 2: Val loss vs quality metrics
    ax2.scatter(val_losses, valid_json_rates, c="blue", label="Valid JSON", alpha=0.7)
    ax2.scatter(val_losses, schema_valid_rates, c="green", label="Schema Valid", alpha=0.7)
    ax2.set_xlabel("Validation Loss")
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("Quality vs Validation Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()  # Lower loss is better, so put it on the right

    plt.tight_layout()
    plt.savefig(output_dir / "generation_quality.png", dpi=150)
    plt.close()

    print(f"  Saved generation_quality.png")

    # Plot 3: Quality vs loss scatter with correlation
    from scipy import stats

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(val_losses, valid_json_rates, c="blue", s=100, alpha=0.7)

    # Add epoch labels
    for i, m in enumerate(all_metrics):
        ax.annotate(f"E{m.epoch}", (val_losses[i], valid_json_rates[i]),
                   textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Compute correlation
    corr, p_value = stats.spearmanr(val_losses, valid_json_rates)
    ax.set_title(f"Valid JSON Rate vs Val Loss\nSpearman r={corr:.3f}, p={p_value:.3f}")
    ax.set_xlabel("Validation Loss")
    ax.set_ylabel("Valid JSON Rate (%)")
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    plt.tight_layout()
    plt.savefig(output_dir / "quality_vs_loss.png", dpi=150)
    plt.close()

    print(f"  Saved quality_vs_loss.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze generation quality across checkpoints")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing checkpoint files")
    parser.add_argument("--data_dir", type=str,
                       default=str(Path.home() / "Desktop" / "data" / "kubernetes-tct-bpe-1k"),
                       help="Directory containing training data (for first token distribution)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory for output files")
    parser.add_argument("--num_samples", type=int, default=100,
                       help="Number of samples per checkpoint")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on")
    parser.add_argument("--no_bos", action="store_true",
                       help="Don't use BOS token (for older models trained without BOS prepending)")

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Generation Quality Analysis")
    print("=" * 60)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Data dir: {data_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Samples per checkpoint: {args.num_samples}")
    print(f"Device: {args.device}")
    use_bos = not args.no_bos
    print(f"Use BOS: {use_bos} {'(default, for BOS-trained models)' if use_bos else '(for older models)'}")

    # Load TCT module
    print("\nLoading TCT tokenizer...")
    import tct_kubernetes_bpe_1k as tct
    print(f"  Vocab size: {tct.vocab_size()}, Pad token: {tct.pad_token()}")

    # Load first token distribution
    print("\nComputing first token distribution...")
    first_token_dist = load_first_token_distribution(data_dir)
    print(f"  Unique first tokens: {len(first_token_dist)}")
    top_tokens = sorted(first_token_dist.items(), key=lambda x: -x[1])[:5]
    print(f"  Top 5: {[(t, f'{p:.1%}') for t, p in top_tokens]}")

    # Find checkpoints
    checkpoint_files = sorted(checkpoint_dir.glob("epoch_*.pt"))
    print(f"\nFound {len(checkpoint_files)} checkpoints")

    # Analyze each checkpoint
    all_metrics = []
    all_results = {}

    for ckpt_path in checkpoint_files:
        print(f"\n{'-' * 40}")
        print(f"Checkpoint: {ckpt_path.name}")

        metrics, results = analyze_checkpoint(
            checkpoint_path=ckpt_path,
            tct_module=tct,
            first_token_distribution=first_token_dist,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
            use_bos=use_bos,
        )

        all_metrics.append(metrics)
        all_results[metrics.epoch] = results

        print(f"  Epoch {metrics.epoch}: val_loss={metrics.val_loss:.4f}")
        print(f"    Valid JSON: {metrics.valid_json_rate:.1%}")
        print(f"    Schema Valid: {metrics.schema_valid_rate:.1%}")
        print(f"    Complete: {metrics.completion_rate:.1%}")
        print(f"    Kinds: {dict(sorted(metrics.kind_distribution.items(), key=lambda x: -x[1])[:3])}")

    # Save metrics
    print(f"\n{'=' * 60}")
    print("Saving results...")

    metrics_data = [
        {
            "epoch": m.epoch,
            "val_loss": m.val_loss,
            "valid_json_rate": m.valid_json_rate,
            "schema_valid_rate": m.schema_valid_rate,
            "completion_rate": m.completion_rate,
            "kind_distribution": m.kind_distribution,
        }
        for m in all_metrics
    ]

    with open(output_dir / "generation_metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  Saved generation_metrics.json")

    # Save sample generations (first 5 from key epochs)
    sample_epochs = [15, 75, 150]
    samples_data = {}
    for epoch in sample_epochs:
        if epoch in all_results:
            samples_data[epoch] = [
                {
                    "json_output": r.json_output,
                    "is_complete": r.is_complete,
                    "is_valid_json": r.is_valid_json,
                    "kind": r.kind_value,
                }
                for r in all_results[epoch][:10]  # First 10 samples
            ]

    with open(output_dir / "sample_generations.json", "w") as f:
        json.dump(samples_data, f, indent=2)
    print(f"  Saved sample_generations.json")

    # Generate plots
    print("\nGenerating plots...")
    plot_results(all_metrics, output_dir)

    # Summary table
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    print(f"{'Epoch':>6} {'Val Loss':>10} {'Valid JSON':>12} {'Schema':>10} {'Complete':>10}")
    print("-" * 50)
    for m in all_metrics:
        print(f"{m.epoch:>6} {m.val_loss:>10.4f} {m.valid_json_rate:>11.1%} {m.schema_valid_rate:>9.1%} {m.completion_rate:>9.1%}")

    print("\nDone!")


if __name__ == "__main__":
    main()
