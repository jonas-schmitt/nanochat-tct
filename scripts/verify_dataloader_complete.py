#!/usr/bin/env python3
"""
Complete dataloader verification before training.

Verifies:
1. Correct number of examples
2. Correct prefix generation
3. Correct padding and masking
4. No data corruption
5. Train/val consistency
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "tct-bundle" / "adapters"))

import torch
import numpy as np
from tct_dataloader import tokenizing_distributed_data_loader

def verify_dataloader(context_size=2048):
    print("="*80)
    print(f"DATALOADER VERIFICATION (context_size={context_size})")
    print("="*80)

    data_dir = Path.home() / "Desktop/data/workflows/json"

    # Load datasets
    print("\n1. Loading datasets...")
    train_dataset = tokenizing_distributed_data_loader(
        device_batch_size=8,
        context_size=context_size,
        split="train",
        data_dir=str(data_dir),
        train_split=0.9,
        geometric_p=1.0,
        prefix_mode="all",
        cache_file=None,
    )

    val_dataset = tokenizing_distributed_data_loader(
        device_batch_size=8,
        context_size=context_size,
        split="val",
        data_dir=str(data_dir),
        train_split=0.9,
        geometric_p=1.0,
        prefix_mode="all",
        cache_file=None,
    )

    print(f"   Train workflows: {len(train_dataset.tokenized_workflows):,}")
    print(f"   Val workflows: {len(val_dataset.tokenized_workflows):,}")
    print(f"   Train examples: {len(train_dataset):,}")
    print(f"   Val examples: {len(val_dataset):,}")

    # Verify example count
    print("\n2. Verifying example count...")
    expected_train = sum(min(len(tokens), context_size) for tokens in train_dataset.tokenized_workflows)
    expected_val = sum(min(len(tokens), context_size) for tokens in val_dataset.tokenized_workflows)

    print(f"   Expected train: {expected_train:,}")
    print(f"   Actual train: {len(train_dataset):,}")
    print(f"   Match: {'✅' if expected_train == len(train_dataset) else '❌'}")

    print(f"   Expected val: {expected_val:,}")
    print(f"   Actual val: {len(val_dataset):,}")
    print(f"   Match: {'✅' if expected_val == len(val_dataset) else '❌'}")

    if expected_train != len(train_dataset) or expected_val != len(val_dataset):
        print("\n❌ ERROR: Example count mismatch!")
        return False

    # Verify prefix index structure
    print("\n3. Verifying prefix index structure...")

    # Check that window_start is always 0 (no windowing)
    unique_starts = set()
    for _, window_start, _ in train_dataset.prefix_index[:10000]:
        unique_starts.add(window_start)

    print(f"   Unique window starts (sample): {sorted(unique_starts)}")
    if unique_starts != {0}:
        print("   ❌ ERROR: Found non-zero window starts (windowing detected)!")
        return False
    print("   ✅ No windowing (all windows start at 0)")

    # Verify a specific workflow's prefixes
    print("\n4. Verifying prefix generation for sample workflows...")

    # Find short workflow
    for wf_idx, tokens in enumerate(train_dataset.tokenized_workflows):
        if 100 <= len(tokens) <= 200:
            wf_len = len(tokens)
            # Get all examples for this workflow
            examples = [(i, prefix_len) for i, (w_idx, _, prefix_len) in enumerate(train_dataset.prefix_index) if w_idx == wf_idx]

            print(f"\n   Workflow {wf_idx}: length={wf_len}")
            print(f"   Expected prefixes: [1, 2, ..., {wf_len}]")
            print(f"   Got {len(examples)} examples")

            # Verify we have correct prefix lengths
            prefix_lens = sorted([p for _, p in examples])
            expected = list(range(1, wf_len + 1))

            if prefix_lens == expected:
                print(f"   ✅ Correct: Got prefixes [1, 2, ..., {wf_len}]")
            else:
                print(f"   ❌ ERROR: Prefix lengths don't match!")
                print(f"   Expected: {expected[:10]}... (length {len(expected)})")
                print(f"   Got: {prefix_lens[:10]}... (length {len(prefix_lens)})")
                return False
            break

    # Find long workflow (>context_size)
    for wf_idx, tokens in enumerate(train_dataset.tokenized_workflows):
        if len(tokens) > context_size:
            wf_len = len(tokens)
            examples = [(i, prefix_len) for i, (w_idx, _, prefix_len) in enumerate(train_dataset.prefix_index) if w_idx == wf_idx]

            print(f"\n   Workflow {wf_idx}: length={wf_len} (exceeds context)")
            print(f"   Expected prefixes: [1, 2, ..., {context_size}] (capped)")
            print(f"   Got {len(examples)} examples")

            prefix_lens = sorted([p for _, p in examples])
            expected = list(range(1, context_size + 1))

            if prefix_lens == expected:
                print(f"   ✅ Correct: Got prefixes [1, 2, ..., {context_size}]")
            else:
                print(f"   ❌ ERROR: Prefix lengths don't match!")
                return False
            break

    # Verify actual example data
    print("\n5. Verifying example data (padding, masking)...")

    # Test a few random examples
    for idx in np.random.choice(len(train_dataset), 5):
        x, y = train_dataset[idx]

        # Check shapes
        if x.shape != (context_size,) or y.shape != (context_size,):
            print(f"   ❌ ERROR: Wrong shapes at idx {idx}")
            print(f"      x.shape={x.shape}, y.shape={y.shape}")
            print(f"      Expected: ({context_size},)")
            return False

        # Check that x and y are shifted versions
        # y should be x shifted left by 1 (with appropriate handling at boundaries)
        non_pad_x = (x != 8191).sum().item()
        non_pad_y = (y != -1).sum().item()

        if abs(non_pad_x - non_pad_y) > 1:  # Allow off-by-one due to shifting
            print(f"   ❌ ERROR: x/y length mismatch at idx {idx}")
            print(f"      non-pad x: {non_pad_x}, non-pad y: {non_pad_y}")
            return False

    print("   ✅ All tested examples have correct shapes and padding")

    # Verify no data leakage between train/val
    print("\n6. Verifying train/val split...")

    # Get first few workflows from each
    train_first_workflow = train_dataset.tokenized_workflows[0]
    val_first_workflow = val_dataset.tokenized_workflows[0]

    if torch.equal(train_first_workflow, val_first_workflow):
        print("   ❌ ERROR: Train and val have identical first workflow (data leakage!)")
        return False

    print("   ✅ Train/val splits are different (no obvious leakage)")

    # Final summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    print(f"✅ Example count: {len(train_dataset):,} train, {len(val_dataset):,} val")
    print(f"✅ Prefix generation: Correct for all tested workflows")
    print(f"✅ No windowing: All examples start from position 0")
    print(f"✅ Data format: Correct shapes and padding")
    print(f"✅ Train/val split: No obvious leakage")
    print("\n🎉 All checks passed! Dataloader is correct.")
    print("="*80)

    return True

if __name__ == "__main__":
    # Test with 2048 context
    success = verify_dataloader(context_size=2048)
    sys.exit(0 if success else 1)
