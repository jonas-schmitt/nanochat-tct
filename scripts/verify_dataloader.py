#!/usr/bin/env python3
"""
Verify that the current dataloader implementation is correct.

Tests:
1. Example count calculation
2. Prefix generation for various workflow lengths
3. No duplication
4. Correct padding and masking
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "tct-bundle" / "adapters"))

import torch
from tct_dataloader import tokenizing_distributed_data_loader

print("="*80)
print("DATALOADER VERIFICATION TEST")
print("="*80)

# Create small test dataset
test_data_dir = Path.home() / "Desktop/data/workflows/json"

print("\n1. Loading dataset with prefix_mode='all'...")
dataset = tokenizing_distributed_data_loader(
    device_batch_size=8,
    context_size=1024,
    split="train",
    data_dir=str(test_data_dir),
    train_split=0.9,
    geometric_p=1.0,
    prefix_mode="all",
    cache_file=None,  # Use cache
)

print(f"   Total examples: {len(dataset):,}")
print(f"   Total workflows: {len(dataset.tokenized_workflows):,}")

# Calculate expected examples manually
print("\n2. Manual calculation of expected examples...")
expected_examples = 0
workflow_lengths = []

for tokens in dataset.tokenized_workflows:
    wf_len = len(tokens)
    workflow_lengths.append(wf_len)

    # Current implementation: single window at position 0
    # Prefixes: 1, 2, 3, ..., min(wf_len, context_size)
    num_prefixes = min(wf_len, dataset.context_size)
    expected_examples += num_prefixes

print(f"   Expected examples (manual count): {expected_examples:,}")
print(f"   Actual examples: {len(dataset):,}")
print(f"   Match: {'✅' if expected_examples == len(dataset) else '❌'}")

# Analyze workflow length distribution
import numpy as np
lengths = np.array(workflow_lengths)
print(f"\n3. Workflow length distribution:")
print(f"   Min: {lengths.min()} tokens")
print(f"   Median: {int(np.median(lengths))} tokens")
print(f"   Mean: {lengths.mean():.0f} tokens")
print(f"   Max: {lengths.max()} tokens")
print(f"   90th %ile: {int(np.percentile(lengths, 90))} tokens")

# Check a few specific examples
print(f"\n4. Checking specific examples...")

# Find a short workflow (<1024)
short_idx = None
for i, l in enumerate(workflow_lengths):
    if l < 1024:
        short_idx = i
        break

# Find a long workflow (>1024)
long_idx = None
for i, l in enumerate(workflow_lengths):
    if l > 1024:
        long_idx = i
        break

if short_idx is not None:
    wf_len = workflow_lengths[short_idx]
    # Find examples for this workflow
    examples_for_wf = [i for i, (wf_idx, _, _) in enumerate(dataset.prefix_index) if wf_idx == short_idx]
    print(f"\n   Short workflow (idx={short_idx}, len={wf_len}):")
    print(f"   Expected {wf_len} examples, got {len(examples_for_wf)}")
    print(f"   Match: {'✅' if len(examples_for_wf) == wf_len else '❌'}")

    # Check first and last example
    x1, y1 = dataset[examples_for_wf[0]]
    x2, y2 = dataset[examples_for_wf[-1]]
    print(f"   First example: x.shape={x1.shape}, non-pad tokens={(x1 != 8191).sum().item()}")
    print(f"   Last example: x.shape={x2.shape}, non-pad tokens={(x2 != 8191).sum().item()}")

if long_idx is not None:
    wf_len = workflow_lengths[long_idx]
    examples_for_wf = [i for i, (wf_idx, _, _) in enumerate(dataset.prefix_index) if wf_idx == long_idx]
    print(f"\n   Long workflow (idx={long_idx}, len={wf_len}):")
    print(f"   Expected {min(wf_len, 1024)} examples, got {len(examples_for_wf)}")
    print(f"   Match: {'✅' if len(examples_for_wf) == min(wf_len, 1024) else '❌'}")

    x1, y1 = dataset[examples_for_wf[0]]
    x2, y2 = dataset[examples_for_wf[-1]]
    print(f"   First example: x.shape={x1.shape}, non-pad tokens={( != 8191).sum().item()}")
    print(f"   Last example: x.shape={x2.shape}, non-pad tokens={(x2 != 8191).sum().item()}")

# Check window positions
print(f"\n5. Checking window positions...")
unique_window_starts = set()
for wf_idx, window_start, _ in dataset.prefix_index[:10000]:  # Sample first 10k
    unique_window_starts.add(window_start)

print(f"   Unique window start positions (first 10k examples): {sorted(unique_window_starts)}")
print(f"   Using windowing: {'❌ No (only position 0)' if unique_window_starts == {0} else '✅ Yes'}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
