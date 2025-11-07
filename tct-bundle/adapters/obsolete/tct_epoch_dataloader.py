"""
Epoch-Based Offset DataLoader for TCT Workflows

Implements dynamic windowing with epoch-based offsets for data augmentation:
- Epoch 0 (offset=0): Windows at [0-1023], [1024-2047], [2048-3071], ...
- Epoch 1 (offset=32): Windows at [32-1055], [1056-2079], [2080-3103], ...
- Epoch 2 (offset=64): Windows at [64-1087], [1088-2111], ...
- ...
- Epoch 31 (offset=992): Windows at [992-2015], [2016-3039], ...

Key benefits:
- No overlap within each epoch (clean training signal)
- 32 different views across epochs (data augmentation)
- Always includes beginning of workflows (critical context)
- Efficient: ~40k windows per epoch vs 967k with stride=32
"""

import sys
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Add TCT to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tct-github-workflow/target/wheels"))
from tct_github_workflow import encode


class EpochOffsetWorkflowDataset(Dataset):
    """
    Dataset that creates non-overlapping windows with epoch-based offset.

    Each epoch uses a different offset (0, 32, 64, ..., 992) to window the workflows.
    This provides natural data augmentation while maintaining no overlap within epochs.
    """

    def __init__(self, workflow_files, context_size=1024, offset_stride=32, split="train",
                 workflow_dir=None, train_split=0.9):
        """
        Initialize dataset.

        Args:
            workflow_files: List of paths to JSON workflow files
            context_size: Size of context window (default: 1024)
            offset_stride: Stride for epoch offset (default: 32, gives 32 different views)
            split: "train" or "val" (affects shuffling)
            workflow_dir: Path to workflow directory (for caching, optional)
            train_split: Train/val split ratio (for cache naming, default: 0.9)
        """
        self.workflow_files = list(workflow_files)
        self.context_size = context_size
        self.offset_stride = offset_stride
        self.split = split

        # Current epoch offset (set by set_epoch())
        self.current_offset = 0

        # Try to load from cache if workflow_dir provided
        cache_file = None
        if workflow_dir is not None:
            workflow_dir_path = Path(workflow_dir)
            cache_dir = workflow_dir_path / ".cache"
            cache_dir.mkdir(exist_ok=True)
            # Cache key: split + train_split + number of files
            cache_file = cache_dir / f"tokenized_{split}_split{int(train_split*100)}_{len(self.workflow_files)}files.pt"

            if cache_file.exists():
                print(f"Loading tokenized workflows from cache: {cache_file}")
                try:
                    self.tokenized_workflows = torch.load(cache_file)
                    print(f"Loaded {len(self.tokenized_workflows)} tokenized workflows from cache ✅")
                except Exception as e:
                    print(f"Warning: Failed to load cache ({e}), will tokenize from scratch")
                    cache_file = None  # Force re-tokenization
                    self.tokenized_workflows = None

        # Tokenize if no cache or cache load failed
        if cache_file is None or not hasattr(self, 'tokenized_workflows') or self.tokenized_workflows is None:
            print(f"Tokenizing {len(self.workflow_files)} workflows...")
            self.tokenized_workflows = []
            for wf_file in self.workflow_files:
                with open(wf_file) as f:
                    json_str = f.read()
                tokens = encode(json_str)
                self.tokenized_workflows.append(torch.tensor(tokens, dtype=torch.long))

            # Save to cache if workflow_dir provided
            if workflow_dir is not None and cache_file is not None:
                print(f"Saving tokenized workflows to cache: {cache_file}")
                try:
                    torch.save(self.tokenized_workflows, cache_file)
                    print(f"Cache saved successfully ✅")
                except Exception as e:
                    print(f"Warning: Failed to save cache ({e})")

        # Build initial window index
        self._rebuild_window_index()

        print(f"Dataset initialized: {len(self)} windows (offset={self.current_offset})")

    def set_epoch(self, epoch):
        """
        Set the current epoch, which determines the windowing offset.

        Args:
            epoch: Current training epoch
        """
        # Cycle through offsets: 0, 32, 64, ..., 992, 0, 32, ...
        self.current_offset = (epoch * self.offset_stride) % self.context_size
        self._rebuild_window_index()
        print(f"Epoch {epoch}: offset={self.current_offset}, {len(self)} windows")

    def _rebuild_window_index(self):
        """
        Build index of all windows for current offset.

        For offset=0: [0-1023], [1024-2047], ...
        For offset=32: [32-1055], [1056-2079], ... (skip first 32 tokens)

        For workflows smaller than context_size:
        - If offset=0: Include as single padded window
        - If offset>0 and workflow too small: Skip (already seen in epoch 0)
        """
        self.window_index = []  # List of (workflow_idx, window_start)

        for wf_idx, tokens in enumerate(self.tokenized_workflows):
            # Handle workflows smaller than context_size
            if len(tokens) < self.context_size:
                # Only include in epoch 0 (offset=0), pad to context_size
                if self.current_offset == 0:
                    self.window_index.append((wf_idx, 0))
                # Skip in other epochs (already saw full workflow in epoch 0)
                continue

            # Start from offset (skip first `offset` tokens)
            pos = self.current_offset

            # Create non-overlapping windows
            while pos + self.context_size <= len(tokens):
                self.window_index.append((wf_idx, pos))
                pos += self.context_size

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        """
        Get a training window.

        Returns:
            (x, y) tuple:
            - x: Input tokens [context_size] (position + content[:-1])
            - y: Target tokens [context_size] (content)
        """
        wf_idx, start = self.window_index[idx]
        tokens = self.tokenized_workflows[wf_idx]

        # Extract window
        window = tokens[start:start + self.context_size]
        actual_len = len(window)

        # Pad if needed (for workflows smaller than context_size)
        # For INPUT (x): use token 8191 (dedicated PAD token, never produced by TCT)
        # For TARGETS (y): use -1 at padding positions (ignored by loss via ignore_index=-1)
        # Pad at END: [real_tokens, PAD, PAD, ...]
        #   - Position token semantics consistent: real content always starts after position
        #   - Natural completion signal: real_tokens → PAD (workflow ends)
        PAD_TOKEN = 8191  # Dedicated padding token (vocab_size = 8192)
        if actual_len < self.context_size:
            # Pad at end with 8191 (dedicated PAD token)
            padding = torch.full((self.context_size - actual_len,), PAD_TOKEN, dtype=torch.long)
            window = torch.cat([window, padding])

        # Create position token (map to vocab space)
        # Use stride 32 for position mapping to keep positions < 8192
        position = start // 32

        # Prepend position token
        window_with_pos = torch.cat([
            torch.tensor([position], dtype=torch.long),
            window
        ])

        # x: [position, tok_0, ..., tok_N-1, PAD, PAD]  (use 8191 for padding - dedicated token)
        # y: [tok_0, tok_1, ..., tok_N, PAD, PAD]      (use -1 for padding - ignored in loss)
        x = window_with_pos[:-1].clone()
        y = window_with_pos[1:].clone()

        # Mask padding positions in targets: set to -1 so cross_entropy ignores them
        # Padding is at END (last N positions where N = context_size - actual_len)
        if actual_len < self.context_size:
            # Padding starts at position actual_len (after real tokens)
            y[actual_len:] = -1  # Mask padding at end

        return x, y


def create_epoch_offset_dataloader(
    workflow_dir,
    context_size=1024,
    offset_stride=32,
    batch_size=32,
    split="train",
    train_split=0.9,
    num_workers=0,
    pin_memory=True,
):
    """
    Create dataloader with epoch-based offset windowing.

    Args:
        workflow_dir: Directory containing JSON workflow files
        context_size: Window size (default: 1024)
        offset_stride: Stride for epoch offset (default: 32)
        batch_size: Batch size
        split: "train" or "val"
        train_split: Fraction of data for training (default: 0.9)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        DataLoader instance
    """
    workflow_dir = Path(workflow_dir)
    workflow_files = sorted(list(workflow_dir.glob("*.json")))

    if len(workflow_files) == 0:
        raise ValueError(f"No workflow files found in {workflow_dir}")

    # Split train/val
    num_train = int(len(workflow_files) * train_split)
    if split == "train":
        workflow_files = workflow_files[:num_train]
    else:
        workflow_files = workflow_files[num_train:]

    # Create dataset
    dataset = EpochOffsetWorkflowDataset(
        workflow_files,
        context_size=context_size,
        offset_stride=offset_stride,
        split=split,
        workflow_dir=workflow_dir,
        train_split=train_split,
    )

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches to prevent OOM from recompilation
    )

    return loader


# Compatibility function for existing training scripts
def create_dataloader(data_path, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
    """
    Compatibility function - not used with epoch-based approach.
    Use create_epoch_offset_dataloader instead.
    """
    raise NotImplementedError(
        "This dataloader uses epoch-based offset windowing. "
        "Use create_epoch_offset_dataloader() instead of create_dataloader()."
    )


if __name__ == "__main__":
    import os

    # Test the dataloader
    workflow_dir = Path.home() / "Desktop/data/workflows-100k/json"

    if not workflow_dir.exists():
        print(f"Error: Workflow directory not found: {workflow_dir}")
        exit(1)

    print("Creating epoch-offset dataloader...")
    loader = create_epoch_offset_dataloader(
        workflow_dir,
        context_size=1024,
        offset_stride=32,
        batch_size=4,
        split="train",
        num_workers=0,
    )

    dataset = loader.dataset

    # Test multiple epochs
    for epoch in range(3):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}")
        print(f"{'='*60}")

        dataset.set_epoch(epoch)

        # Test first batch
        for i, (x, y) in enumerate(loader):
            print(f"\nBatch {i}:")
            print(f"  x shape: {x.shape}")
            print(f"  y shape: {y.shape}")
            print(f"  First window:")
            print(f"    Position token: {x[0, 0].item()}")
            print(f"    First 10 content tokens: {x[0, 1:11].tolist()}")
            print(f"    Target first 10 tokens: {y[0, :10].tolist()}")

            # Verify target is shifted input
            assert torch.equal(x[0, 1:], y[0, :-1]), "Target should be shifted input!"

            if i >= 2:  # Test only first 3 batches per epoch
                break

    print("\n✅ Epoch-based offset dataloader test successful!")
