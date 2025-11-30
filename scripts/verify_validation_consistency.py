#!/usr/bin/env python3
"""
Verify that validation dataset returns consistent examples across multiple iterations.

This checks if the validation set is truly fixed or if it changes between evaluations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "tct-bundle" / "adapters"))

import torch
from torch.utils.data import DataLoader
from tct_dataloader import tokenizing_distributed_data_loader

def main():
    print("="*80)
    print("VALIDATION CONSISTENCY CHECK")
    print("="*80)

    # Create validation dataset
    print("\n1. Creating validation dataset...")
    val_dataset = tokenizing_distributed_data_loader(
        device_batch_size=4,
        context_size=1024,
        split="val",
        data_dir=str(Path.home() / "Desktop/data/workflows/json"),
        train_split=0.9,
        geometric_p=1.0,
        prefix_mode="all",
        cache_file=None,
    )

    print(f"   Val examples: {len(val_dataset):,}")

    # Create dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # Sample first 10 batches, 3 times
    print("\n2. Sampling first 10 batches, 3 times...")

    samples = []
    for iteration in range(3):
        print(f"\n   Iteration {iteration + 1}:")
        batch_samples = []

        for i, (x, y) in enumerate(val_loader):
            if i >= 10:
                break

            # Store first example from each batch
            first_x = x[0].clone()
            first_y = y[0].clone()
            batch_samples.append((first_x, first_y))

            # Print some stats
            if i < 3:
                non_pad = (first_x != 8191).sum().item()
                print(f"      Batch {i}: first example has {non_pad} non-pad tokens")

        samples.append(batch_samples)

    # Compare iterations
    print("\n3. Comparing iterations...")

    all_match = True
    for batch_idx in range(10):
        iter1_x, iter1_y = samples[0][batch_idx]
        iter2_x, iter2_y = samples[1][batch_idx]
        iter3_x, iter3_y = samples[2][batch_idx]

        match_12 = torch.equal(iter1_x, iter2_x) and torch.equal(iter1_y, iter2_y)
        match_23 = torch.equal(iter2_x, iter3_x) and torch.equal(iter2_y, iter3_y)
        match_13 = torch.equal(iter1_x, iter3_x) and torch.equal(iter1_y, iter3_y)

        if not (match_12 and match_23 and match_13):
            print(f"   ❌ Batch {batch_idx}: MISMATCH between iterations!")
            all_match = False
        else:
            if batch_idx < 3:
                print(f"   ✅ Batch {batch_idx}: Consistent across all iterations")

    # Final verdict
    print("\n" + "="*80)
    if all_match:
        print("✅ VALIDATION SET IS CONSISTENT")
        print("   All batches match across iterations - validation sampling is correct")
    else:
        print("❌ VALIDATION SET IS INCONSISTENT")
        print("   Different examples returned on each iteration - THIS IS A BUG!")
    print("="*80)

    return all_match

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
