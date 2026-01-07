#!/usr/bin/env python3
"""
Analyze weight distributions across training checkpoints.

Tracks how weight statistics evolve during training:
- Per-layer statistics (mean, std, min, max, norm)
- Attention weight analysis (Q, K, V, O projections)
- Weight histograms at key epochs

Usage:
    python -m scripts.analyze_weights \
        --checkpoint_dir checkpoints_analysis/kubernetes_tct_small \
        --output_dir checkpoints_analysis/results
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np


@dataclass
class WeightStats:
    """Statistics for a weight tensor."""
    name: str
    shape: Tuple[int, ...]
    mean: float
    std: float
    min_val: float
    max_val: float
    norm: float  # Frobenius norm
    num_params: int


def compute_weight_stats(name: str, tensor: torch.Tensor) -> WeightStats:
    """Compute statistics for a weight tensor."""
    t = tensor.float()
    return WeightStats(
        name=name,
        shape=tuple(tensor.shape),
        mean=t.mean().item(),
        std=t.std().item(),
        min_val=t.min().item(),
        max_val=t.max().item(),
        norm=t.norm().item(),
        num_params=tensor.numel(),
    )


def categorize_weight(name: str) -> str:
    """Categorize a weight by its type."""
    if "wte" in name or "wpe" in name:
        return "embedding"
    elif "lm_head" in name:
        return "lm_head"
    elif "c_q" in name:
        return "attn_q"
    elif "c_k" in name:
        return "attn_k"
    elif "c_v" in name:
        return "attn_v"
    elif "c_proj" in name:
        return "attn_o"
    elif "w_gate" in name:
        return "ffn_gate"
    elif "w_up" in name:
        return "ffn_up"
    elif "w_down" in name:
        return "ffn_down"
    elif "c_fc" in name:
        return "ffn_fc"
    else:
        return "other"


def analyze_checkpoint(checkpoint_path: Path) -> Dict[str, List[WeightStats]]:
    """Analyze weights in a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    # Group stats by category
    stats_by_category: Dict[str, List[WeightStats]] = {}

    for name, tensor in state_dict.items():
        if tensor.dim() == 0:  # Skip scalars
            continue

        stats = compute_weight_stats(name, tensor)
        category = categorize_weight(name)

        if category not in stats_by_category:
            stats_by_category[category] = []
        stats_by_category[category].append(stats)

    return stats_by_category


def aggregate_category_stats(stats_list: List[WeightStats]) -> Dict[str, float]:
    """Aggregate statistics across weights in a category."""
    if not stats_list:
        return {}

    total_params = sum(s.num_params for s in stats_list)
    avg_norm = np.mean([s.norm for s in stats_list])
    avg_std = np.mean([s.std for s in stats_list])

    return {
        "num_weights": len(stats_list),
        "total_params": total_params,
        "avg_norm": avg_norm,
        "avg_std": avg_std,
        "min_norm": min(s.norm for s in stats_list),
        "max_norm": max(s.norm for s in stats_list),
    }


def plot_results(all_stats: Dict[int, Dict[str, Dict]], output_dir: Path):
    """Generate plots from weight statistics."""
    import matplotlib.pyplot as plt

    epochs = sorted(all_stats.keys())

    # Categories to track
    categories = ["attn_q", "attn_k", "attn_v", "attn_o", "ffn_gate", "ffn_up", "ffn_down", "lm_head", "embedding"]

    # Plot 1: Norm evolution by category
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Attention weights
    ax1 = axes[0, 0]
    for cat in ["attn_q", "attn_k", "attn_v", "attn_o"]:
        norms = [all_stats[e].get(cat, {}).get("avg_norm", 0) for e in epochs]
        if any(n > 0 for n in norms):
            ax1.plot(epochs, norms, "-o", label=cat, markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Frobenius Norm")
    ax1.set_title("Attention Weight Norms")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # FFN weights
    ax2 = axes[0, 1]
    for cat in ["ffn_gate", "ffn_up", "ffn_down"]:
        norms = [all_stats[e].get(cat, {}).get("avg_norm", 0) for e in epochs]
        if any(n > 0 for n in norms):
            ax2.plot(epochs, norms, "-o", label=cat, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average Frobenius Norm")
    ax2.set_title("FFN Weight Norms")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Embedding and lm_head
    ax3 = axes[1, 0]
    for cat in ["embedding", "lm_head"]:
        norms = [all_stats[e].get(cat, {}).get("avg_norm", 0) for e in epochs]
        if any(n > 0 for n in norms):
            ax3.plot(epochs, norms, "-o", label=cat, markersize=4)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Average Frobenius Norm")
    ax3.set_title("Embedding & Output Norms")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Q*K scale proxy (product of Q and K norms)
    ax4 = axes[1, 1]
    q_norms = [all_stats[e].get("attn_q", {}).get("avg_norm", 1) for e in epochs]
    k_norms = [all_stats[e].get("attn_k", {}).get("avg_norm", 1) for e in epochs]
    qk_scale = [q * k for q, k in zip(q_norms, k_norms)]
    ax4.plot(epochs, qk_scale, "-o", color="purple", markersize=4)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Q_norm × K_norm")
    ax4.set_title("Attention Scale Proxy (Q×K)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "weight_norms.png", dpi=150)
    plt.close()
    print(f"  Saved weight_norms.png")

    # Plot 2: Weight std evolution (indicates learning dynamics)
    fig, ax = plt.subplots(figsize=(12, 6))
    for cat in categories:
        stds = [all_stats[e].get(cat, {}).get("avg_std", 0) for e in epochs]
        if any(s > 0 for s in stds):
            ax.plot(epochs, stds, "-o", label=cat, markersize=4, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Weight Std")
    ax.set_title("Weight Standard Deviation Over Training")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "weight_stds.png", dpi=150)
    plt.close()
    print(f"  Saved weight_stds.png")


def plot_histograms(checkpoint_paths: List[Path], output_dir: Path):
    """Plot weight histograms for selected checkpoints."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(checkpoint_paths), 3, figsize=(15, 4 * len(checkpoint_paths)))
    if len(checkpoint_paths) == 1:
        axes = [axes]

    for idx, ckpt_path in enumerate(checkpoint_paths):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        epoch = ckpt.get("epoch", 0)

        # Collect weights by category
        attn_weights = []
        ffn_weights = []
        other_weights = []

        for name, tensor in state_dict.items():
            if tensor.dim() < 2:
                continue
            flat = tensor.flatten().float().numpy()

            cat = categorize_weight(name)
            if cat.startswith("attn"):
                attn_weights.extend(flat[::100].tolist())  # Subsample for speed
            elif cat.startswith("ffn"):
                ffn_weights.extend(flat[::100].tolist())
            else:
                other_weights.extend(flat[::100].tolist())

        # Plot histograms
        ax1, ax2, ax3 = axes[idx]

        if attn_weights:
            ax1.hist(attn_weights, bins=100, density=True, alpha=0.7)
            ax1.set_title(f"Epoch {epoch}: Attention Weights")
            ax1.set_xlabel("Value")
            ax1.set_ylabel("Density")

        if ffn_weights:
            ax2.hist(ffn_weights, bins=100, density=True, alpha=0.7, color="orange")
            ax2.set_title(f"Epoch {epoch}: FFN Weights")
            ax2.set_xlabel("Value")

        if other_weights:
            ax3.hist(other_weights, bins=100, density=True, alpha=0.7, color="green")
            ax3.set_title(f"Epoch {epoch}: Embedding/Output Weights")
            ax3.set_xlabel("Value")

    plt.tight_layout()
    plt.savefig(output_dir / "weight_histograms.png", dpi=150)
    plt.close()
    print(f"  Saved weight_histograms.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze weight distributions")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing checkpoint files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory for output files")

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Weight Distribution Analysis")
    print("=" * 60)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Output dir: {output_dir}")

    # Find checkpoints
    checkpoint_files = sorted(checkpoint_dir.glob("epoch_*.pt"))
    print(f"\nFound {len(checkpoint_files)} checkpoints")

    # Analyze each checkpoint
    all_stats: Dict[int, Dict[str, Dict]] = {}

    for ckpt_path in checkpoint_files:
        print(f"\nAnalyzing {ckpt_path.name}...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", 0)

        stats_by_category = analyze_checkpoint(ckpt_path)

        # Aggregate stats
        epoch_stats = {}
        for category, stats_list in stats_by_category.items():
            epoch_stats[category] = aggregate_category_stats(stats_list)

        all_stats[epoch] = epoch_stats

        # Print summary for this epoch
        print(f"  Epoch {epoch}:")
        for cat in ["attn_q", "attn_k", "ffn_gate", "lm_head"]:
            if cat in epoch_stats:
                print(f"    {cat}: norm={epoch_stats[cat]['avg_norm']:.2f}, std={epoch_stats[cat]['avg_std']:.4f}")

    # Save statistics
    stats_output = {
        str(epoch): {
            cat: {k: float(v) if isinstance(v, (int, float, np.floating)) else v
                  for k, v in stats.items()}
            for cat, stats in epoch_stats.items()
        }
        for epoch, epoch_stats in all_stats.items()
    }

    with open(output_dir / "weight_stats.json", "w") as f:
        json.dump(stats_output, f, indent=2)
    print(f"\nSaved weight_stats.json")

    # Generate plots
    print("\nGenerating plots...")
    plot_results(all_stats, output_dir)

    # Weight histograms for key epochs
    key_epochs = [15, 75, 150]
    key_checkpoints = [
        ckpt_path for ckpt_path in checkpoint_files
        if any(f"epoch_{e:03d}" in ckpt_path.name for e in key_epochs)
    ]
    if key_checkpoints:
        print("\nGenerating weight histograms...")
        plot_histograms(key_checkpoints, output_dir)

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary: Weight Norm Evolution")
    print("=" * 60)

    epochs = sorted(all_stats.keys())
    if len(epochs) >= 2:
        first, last = epochs[0], epochs[-1]

        print(f"\n{'Category':<15} {'First':>10} {'Last':>10} {'Change':>10}")
        print("-" * 50)

        for cat in ["attn_q", "attn_k", "attn_v", "attn_o", "ffn_gate", "lm_head"]:
            first_norm = all_stats[first].get(cat, {}).get("avg_norm", 0)
            last_norm = all_stats[last].get(cat, {}).get("avg_norm", 0)
            if first_norm > 0:
                change = (last_norm - first_norm) / first_norm * 100
                print(f"{cat:<15} {first_norm:>10.2f} {last_norm:>10.2f} {change:>+9.1f}%")

        # Q*K scale
        q_first = all_stats[first].get("attn_q", {}).get("avg_norm", 1)
        k_first = all_stats[first].get("attn_k", {}).get("avg_norm", 1)
        q_last = all_stats[last].get("attn_q", {}).get("avg_norm", 1)
        k_last = all_stats[last].get("attn_k", {}).get("avg_norm", 1)

        qk_first = q_first * k_first
        qk_last = q_last * k_last
        qk_change = (qk_last - qk_first) / qk_first * 100

        print("-" * 50)
        print(f"{'Q×K scale':<15} {qk_first:>10.2f} {qk_last:>10.2f} {qk_change:>+9.1f}%")

    print("\nDone!")


if __name__ == "__main__":
    main()
