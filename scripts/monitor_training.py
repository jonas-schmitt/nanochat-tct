#!/usr/bin/env python3
"""
Training Monitor - Real-time training progress tracker

Usage:
    # One-time check
    python scripts/monitor_training.py /tmp/tct_train_decoder_only_v2.log

    # Continuous monitoring (updates every 5 seconds)
    watch -n 5 python scripts/monitor_training.py /tmp/tct_train_decoder_only_v2.log
"""

import sys
import re
from pathlib import Path
from datetime import datetime, timedelta

def parse_log(log_path):
    """Parse training log and extract key metrics."""
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    # Extract configuration
    config = {}
    model_size = None
    vocab_size = None
    context_size = None
    d_model = None
    n_layers = None
    n_heads = None
    max_iters = None
    learning_rate = None
    geometric_p = None

    # Extract training progress
    current_step = None
    current_loss = None
    current_lr = None
    val_loss = None
    tokens_per_sec = None

    # Data preparation progress
    tokenizing_progress = None
    index_building = False
    training_started = False

    for line in lines:
        # Config extraction
        if "Model configuration:" in line or "Loading" in line and "model configuration" in line:
            idx = lines.index(line)
            # Parse config block
            for i in range(idx, min(idx + 20, len(lines))):
                config_line = lines[i]
                if "Vocab size:" in config_line:
                    match = re.search(r'(\d+,?\d*)', config_line)
                    if match:
                        vocab_size = match.group(1).replace(',', '')
                elif "Context size:" in config_line:
                    match = re.search(r'(\d+)', config_line)
                    if match:
                        context_size = match.group(1)
                elif "Model dim:" in config_line or "d_model:" in config_line:
                    match = re.search(r'(\d+)', config_line)
                    if match:
                        d_model = match.group(1)
                elif "Layers:" in config_line:
                    match = re.search(r'(\d+)', config_line)
                    if match:
                        n_layers = match.group(1)
                elif "Heads:" in config_line:
                    match = re.search(r'(\d+)', config_line)
                    if match:
                        n_heads = match.group(1)
                elif "Max iterations:" in config_line:
                    match = re.search(r'(\d+,?\d*)', config_line)
                    if match:
                        max_iters = match.group(1).replace(',', '')
                elif "Learning rate:" in config_line:
                    match = re.search(r'([\d.e-]+)', config_line)
                    if match:
                        learning_rate = match.group(1)
                elif "geometric_p" in config_line:
                    match = re.search(r'geometric_p[=:]?\s*([\d.]+)', config_line)
                    if match:
                        geometric_p = match.group(1)

        # Model size detection
        if "model_size" in line.lower() or "Loading" in line and "model configuration" in line:
            # Check for -1024 suffix first (more specific)
            if "small-1024" in line:
                model_size = "small-1024"
            elif "medium-1024" in line:
                model_size = "medium-1024"
            elif "large-1024" in line:
                model_size = "large-1024"
            # Then check base names
            elif "small" in line:
                model_size = "small"
            elif "medium" in line:
                model_size = "medium"
            elif "large" in line:
                model_size = "large"

        # Progress: Tokenizing
        if "Progress:" in line and "workflows" in line:
            match = re.search(r'(\d+,?\d*)/(\d+,?\d*) workflows', line)
            if match:
                tokenizing_progress = (match.group(1).replace(',', ''), match.group(2).replace(',', ''))

        # Progress: Index building
        if "Building prefix index" in line:
            index_building = True

        # Training started
        if "step" in line.lower() and ("loss" in line.lower() or "iter" in line.lower()):
            training_started = True
            # Parse training step line
            # Format: step 00100 | loss 3.456 | lr 1.23e-04 | 12345 tok/s
            match = re.search(r'step\s+(\d+)', line, re.IGNORECASE)
            if match:
                current_step = int(match.group(1))

            match = re.search(r'loss[:\s]+([\d.]+)', line, re.IGNORECASE)
            if match:
                current_loss = float(match.group(1))

            match = re.search(r'lr[:\s]+([\d.e-]+)', line, re.IGNORECASE)
            if match:
                current_lr = float(match.group(1))

            match = re.search(r'([\d,]+)\s*tok/s', line)
            if match:
                tokens_per_sec = int(match.group(1).replace(',', ''))

        # Validation loss
        if "val" in line.lower() and "loss" in line.lower():
            match = re.search(r'val.*loss[:\s]+([\d.]+)', line, re.IGNORECASE)
            if match:
                val_loss = float(match.group(1))

    return {
        'model_size': model_size,
        'vocab_size': vocab_size,
        'context_size': context_size,
        'd_model': d_model,
        'n_layers': n_layers,
        'n_heads': n_heads,
        'max_iters': max_iters,
        'learning_rate': learning_rate,
        'geometric_p': geometric_p,
        'tokenizing_progress': tokenizing_progress,
        'index_building': index_building,
        'training_started': training_started,
        'current_step': current_step,
        'current_loss': current_loss,
        'current_lr': current_lr,
        'val_loss': val_loss,
        'tokens_per_sec': tokens_per_sec,
    }

def print_status(metrics, log_path):
    """Print formatted training status."""
    print("=" * 80)
    print(f"TCT TRAINING MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print(f"Log file: {log_path}")
    print()

    if metrics is None:
        print("❌ Log file not found!")
        return

    # Configuration
    print("📝 CONFIGURATION")
    print("-" * 80)
    if metrics['model_size']:
        print(f"  Model: {metrics['model_size']}")
    if metrics['vocab_size']:
        print(f"  Vocab: {metrics['vocab_size']}")
    if metrics['context_size']:
        print(f"  Context: {metrics['context_size']}")
    if metrics['d_model'] and metrics['n_layers'] and metrics['n_heads']:
        print(f"  Architecture: {metrics['d_model']}d × {metrics['n_layers']}L × {metrics['n_heads']}H")
    if metrics['max_iters']:
        print(f"  Max iterations: {metrics['max_iters']}")
    if metrics['learning_rate']:
        print(f"  Learning rate: {metrics['learning_rate']}")
    if metrics['geometric_p']:
        decoder_pct = float(metrics['geometric_p']) * 100
        fim_pct = 100 - decoder_pct
        print(f"  Mode: {decoder_pct:.0f}% decoder-only, {fim_pct:.0f}% FIM (p={metrics['geometric_p']})")
    print()

    # Progress
    print("🚀 PROGRESS")
    print("-" * 80)

    if metrics['tokenizing_progress']:
        current, total = metrics['tokenizing_progress']
        pct = (int(current) / int(total)) * 100
        print(f"  Tokenizing: {current}/{total} workflows ({pct:.1f}%)")
        print("  Status: ⏳ Data preparation in progress...")

    elif metrics['index_building']:
        print("  Building prefix index...")
        print("  Status: ⏳ Data preparation in progress...")

    elif metrics['training_started']:
        if metrics['current_step'] and metrics['max_iters']:
            pct = (metrics['current_step'] / int(metrics['max_iters'])) * 100
            print(f"  Step: {metrics['current_step']:,}/{metrics['max_iters']} ({pct:.1f}%)")
        elif metrics['current_step']:
            print(f"  Step: {metrics['current_step']:,}")

        if metrics['current_loss']:
            print(f"  Training loss: {metrics['current_loss']:.4f}")

        if metrics['val_loss']:
            print(f"  Validation loss: {metrics['val_loss']:.4f}")

        if metrics['current_lr']:
            print(f"  Learning rate: {metrics['current_lr']:.2e}")

        if metrics['tokens_per_sec']:
            print(f"  Speed: {metrics['tokens_per_sec']:,} tokens/sec")

        print("  Status: ✅ Training in progress...")

        # ETA estimation
        if metrics['current_step'] and metrics['max_iters'] and metrics['tokens_per_sec']:
            remaining_steps = int(metrics['max_iters']) - metrics['current_step']
            if remaining_steps > 0 and metrics['tokens_per_sec'] > 0:
                # Estimate tokens per step (batch_size * context_size)
                # Assume batch_size=32, context_size from config
                if metrics['context_size']:
                    batch_size = 32  # Default assumption
                    tokens_per_step = batch_size * int(metrics['context_size'])
                    seconds_per_step = tokens_per_step / metrics['tokens_per_sec']
                    eta_seconds = remaining_steps * seconds_per_step
                    eta = timedelta(seconds=int(eta_seconds))
                    print(f"  ETA: ~{eta} remaining")

    else:
        print("  Status: 🔄 Initializing...")

    print()
    print("=" * 80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/monitor_training.py <log_file>")
        print()
        print("Example:")
        print("  python scripts/monitor_training.py /tmp/tct_train_decoder_only_v2.log")
        print()
        print("For continuous monitoring:")
        print("  watch -n 5 python scripts/monitor_training.py /tmp/tct_train_decoder_only_v2.log")
        sys.exit(1)

    log_path = sys.argv[1]
    metrics = parse_log(log_path)
    print_status(metrics, log_path)

if __name__ == "__main__":
    main()
