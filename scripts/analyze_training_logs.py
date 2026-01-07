#!/usr/bin/env python3
"""
Analyze training logs to extract loss curves and identify patterns.

Parses training logs to extract:
- Train loss per step/epoch
- Validation loss per epoch
- Learning rate schedule
- Train/val gap (overfitting indicator)

Usage:
    python -m scripts.analyze_training_logs \
        --log_file runpod_logs/kubernetes_tct_small.log \
        --output_dir checkpoints_analysis/results
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


@dataclass
class TrainStep:
    """Data from a training step log line."""
    epoch: int
    step: int
    total_steps: int
    loss: float
    ppl: float
    lr: float
    dt_ms: float
    tok_per_sec: float


@dataclass
class ValEpoch:
    """Data from a validation log line."""
    epoch: int
    step: int
    val_loss: float
    val_ppl: float
    is_best: bool


def parse_train_line(line: str) -> Optional[TrainStep]:
    """Parse a training step log line.

    Format: E00 step 000010/130350 (0.0%) | loss: 6.8699 | ppl: 962.9 | lr: 1.02e-04 | dt: 502ms | tok/s: 261,244
    """
    pattern = r'E(\d+) step (\d+)/(\d+).*loss: ([\d.]+).*ppl: ([\d.]+).*lr: ([\d.e+-]+).*dt: (\d+)ms.*tok/s: ([\d,]+)'
    match = re.search(pattern, line)
    if match:
        return TrainStep(
            epoch=int(match.group(1)),
            step=int(match.group(2)),
            total_steps=int(match.group(3)),
            loss=float(match.group(4)),
            ppl=float(match.group(5)),
            lr=float(match.group(6)),
            dt_ms=float(match.group(7)),
            tok_per_sec=float(match.group(8).replace(',', '')),
        )
    return None


def parse_val_line(line: str) -> Optional[ValEpoch]:
    """Parse a validation log line.

    Format: Epoch   1 | Step    869 | Val loss: 1.4943 | Val ppl: 4.46 (best)
    """
    pattern = r'Epoch\s+(\d+)\s*\|\s*Step\s+(\d+)\s*\|\s*Val loss:\s*([\d.]+)\s*\|\s*Val ppl:\s*([\d.]+)(?:\s*\(best\))?'
    match = re.search(pattern, line)
    if match:
        return ValEpoch(
            epoch=int(match.group(1)),
            step=int(match.group(2)),
            val_loss=float(match.group(3)),
            val_ppl=float(match.group(4)),
            is_best='(best)' in line,
        )
    return None


def parse_log_file(log_path: Path) -> Tuple[List[TrainStep], List[ValEpoch]]:
    """Parse entire log file and extract training/validation data."""
    train_steps = []
    val_epochs = []

    with open(log_path) as f:
        for line in f:
            # Try parsing as train step
            train = parse_train_line(line)
            if train:
                train_steps.append(train)
                continue

            # Try parsing as validation
            val = parse_val_line(line)
            if val:
                val_epochs.append(val)

    return train_steps, val_epochs


def compute_epoch_train_loss(train_steps: List[TrainStep]) -> dict:
    """Compute average training loss per epoch."""
    epoch_losses = {}
    for step in train_steps:
        if step.epoch not in epoch_losses:
            epoch_losses[step.epoch] = []
        epoch_losses[step.epoch].append(step.loss)

    return {epoch: sum(losses) / len(losses) for epoch, losses in epoch_losses.items()}


def plot_results(train_steps: List[TrainStep], val_epochs: List[ValEpoch], output_dir: Path):
    """Generate plots from the parsed data."""
    import matplotlib.pyplot as plt

    # Convert to DataFrames for easier manipulation
    train_df = pd.DataFrame([vars(s) for s in train_steps])
    val_df = pd.DataFrame([vars(v) for v in val_epochs])

    # Compute epoch-level train loss
    epoch_train_loss = compute_epoch_train_loss(train_steps)

    # Plot 1: Loss curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Train loss vs step (smoothed)
    ax1 = axes[0, 0]
    window = min(100, len(train_df) // 10)
    if window > 0:
        smoothed = train_df['loss'].rolling(window=window, min_periods=1).mean()
        ax1.plot(train_df['step'], smoothed, 'b-', alpha=0.8, label='Train (smoothed)')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss vs Step')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Val loss vs epoch
    ax2 = axes[0, 1]
    ax2.plot(val_df['epoch'], val_df['val_loss'], 'r-o', label='Val Loss')
    # Mark best epochs
    best_epochs = val_df[val_df['is_best']]
    ax2.scatter(best_epochs['epoch'], best_epochs['val_loss'], c='green', s=100,
                zorder=5, label='Best', marker='*')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss vs Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Train vs Val loss comparison (per epoch)
    ax3 = axes[1, 0]
    epochs = sorted(set(epoch_train_loss.keys()) & set(val_df['epoch']))
    train_losses = [epoch_train_loss[e] for e in epochs]
    val_losses = [val_df[val_df['epoch'] == e]['val_loss'].values[0] for e in epochs]

    ax3.plot(epochs, train_losses, 'b-o', label='Train')
    ax3.plot(epochs, val_losses, 'r-o', label='Val')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Train vs Val Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Train/Val gap (overfitting indicator)
    ax4 = axes[1, 1]
    gaps = [v - t for t, v in zip(train_losses, val_losses)]
    ax4.plot(epochs, gaps, 'g-o')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Val Loss - Train Loss')
    ax4.set_title('Generalization Gap (positive = overfitting)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curves.png', dpi=150)
    plt.close()
    print(f"  Saved loss_curves.png")

    # Plot 2: Learning rate schedule
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_df['step'], train_df['lr'], 'b-')
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'lr_schedule.png', dpi=150)
    plt.close()
    print(f"  Saved lr_schedule.png")

    # Plot 3: Training throughput
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_df['step'], train_df['tok_per_sec'] / 1000, 'b-', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Throughput (k tok/s)')
    ax.set_title('Training Throughput')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput.png', dpi=150)
    plt.close()
    print(f"  Saved throughput.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze training logs")
    parser.add_argument("--log_file", type=str, required=True,
                       help="Path to training log file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory for output files")

    args = parser.parse_args()

    log_path = Path(args.log_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training Log Analysis")
    print("=" * 60)
    print(f"Log file: {log_path}")
    print(f"Output dir: {output_dir}")

    # Parse log file
    print("\nParsing log file...")
    train_steps, val_epochs = parse_log_file(log_path)
    print(f"  Found {len(train_steps)} training steps")
    print(f"  Found {len(val_epochs)} validation epochs")

    if not train_steps:
        print("ERROR: No training steps found in log file")
        sys.exit(1)

    # Save parsed data
    train_df = pd.DataFrame([vars(s) for s in train_steps])
    val_df = pd.DataFrame([vars(v) for v in val_epochs])

    train_df.to_csv(output_dir / 'train_steps.csv', index=False)
    val_df.to_csv(output_dir / 'val_epochs.csv', index=False)
    print(f"\n  Saved train_steps.csv ({len(train_df)} rows)")
    print(f"  Saved val_epochs.csv ({len(val_df)} rows)")

    # Generate plots
    print("\nGenerating plots...")
    plot_results(train_steps, val_epochs, output_dir)

    # Summary statistics
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)

    if val_epochs:
        best_val = min(val_epochs, key=lambda v: v.val_loss)
        final_val = val_epochs[-1]

        print(f"\nValidation Loss:")
        print(f"  First epoch: {val_epochs[0].val_loss:.4f}")
        print(f"  Best epoch:  {best_val.val_loss:.4f} (epoch {best_val.epoch})")
        print(f"  Final epoch: {final_val.val_loss:.4f} (epoch {final_val.epoch})")

        # Check for plateau
        last_10 = val_epochs[-10:] if len(val_epochs) >= 10 else val_epochs
        improvement = last_10[0].val_loss - last_10[-1].val_loss
        print(f"\n  Last 10 epochs improvement: {improvement:.4f} ({improvement/last_10[0].val_loss*100:.1f}%)")

        if improvement < 0.01:
            print("  ⚠️  Training appears to have plateaued")

    # Compute epoch-level stats
    epoch_train_loss = compute_epoch_train_loss(train_steps)
    if epoch_train_loss and val_epochs:
        epochs = sorted(set(epoch_train_loss.keys()) & set(v.epoch for v in val_epochs))
        if epochs:
            # Check for overfitting
            final_train = epoch_train_loss[epochs[-1]]
            final_val = next(v.val_loss for v in val_epochs if v.epoch == epochs[-1])
            gap = final_val - final_train

            print(f"\nGeneralization Gap (final epoch):")
            print(f"  Train loss: {final_train:.4f}")
            print(f"  Val loss:   {final_val:.4f}")
            print(f"  Gap:        {gap:.4f}")

            if gap > 0.1:
                print("  ⚠️  Significant overfitting detected")

    print("\nDone!")


if __name__ == "__main__":
    main()
