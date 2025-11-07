"""
Fill-in-the-Middle (FIM) Epoch-Based Offset DataLoader for TCT Workflows

Implements FIM training with epoch-based offsets for enhanced data augmentation:
- O(N²) training examples: Each window generates context_size different FIM examples
- Bidirectional context: Model learns from both left and right context
- Same caching as regular dataloader (tokenized workflows are model-independent)

FIM Training:
- Epoch 0, offset=0, gap=0: Predict token 0 from [PAD] + context[1:]
- Epoch 0, offset=0, gap=128: Predict token 128 from context[0:128] + context[129:]
- Epoch 0, offset=0, gap=511: Predict token 511 from context[0:511] + [PAD]
- ...and so on for each window and each gap position

Key benefits:
- No overlap within each epoch (clean training signal)
- 32 different window offsets × context_size gap positions = massive augmentation
- Bidirectional learning (like BERT, CodeLlama, GitHub Copilot)
- Efficient caching (reuses tokenized workflows)
"""

import sys
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Add TCT to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tct-github-workflow/target/wheels"))
from tct_github_workflow import encode


class FIMEpochOffsetWorkflowDataset(Dataset):
    """
    Dataset that creates FIM training examples with epoch-based offset.

    Each epoch uses a different offset (0, 32, 64, ..., 992) to window the workflows.
    Within each window, we randomly sample gap positions for FIM training.
    This provides massive data augmentation while maintaining no overlap within epochs.
    """

    def __init__(self, workflow_files, context_size=1024, offset_stride=32, split="train",
                 workflow_dir=None, train_split=0.9):
        """
        Initialize FIM dataset.

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
            # Cache key: split + train_split + number of files (model-independent!)
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

        print(f"FIM Dataset initialized: {len(self)} training examples (offset={self.current_offset})")

    def set_epoch(self, epoch):
        """
        Set the current epoch, which determines the windowing offset.

        Args:
            epoch: Current training epoch
        """
        # Cycle through offsets: 0, 32, 64, ..., 992, 0, 32, ...
        self.current_offset = (epoch * self.offset_stride) % self.context_size
        self._rebuild_window_index()
        print(f"Epoch {epoch}: offset={self.current_offset}, {len(self)} training examples")

    def _rebuild_window_index(self):
        """
        Build index of all (window, gap_position) pairs for current offset.

        For each window, enumerate ALL possible gap positions (0 to window_size).
        This creates window_size+1 training examples per window.

        For offset=0: [0-1023], [1024-2047], ...
        For offset=32: [32-1055], [1056-2079], ... (skip first 32 tokens)

        For workflows smaller than context_size:
        - If offset=0: Include with all gap positions (0 to actual_len)
        - If offset>0 and workflow too small: Skip (already seen in epoch 0)

        Example: context_size=512, window has 512 tokens
        → Gap positions: 0, 1, 2, ..., 512 (513 total)
        → Creates 513 training examples from this window
        """
        self.window_index = []  # List of (workflow_idx, window_start, gap_position)

        for wf_idx, tokens in enumerate(self.tokenized_workflows):
            # Handle workflows smaller than context_size
            if len(tokens) < self.context_size:
                # Only include in epoch 0 (offset=0), pad to context_size
                if self.current_offset == 0:
                    actual_len = len(tokens)
                    # Enumerate all gap positions: 0 to actual_len (inclusive)
                    for gap_pos in range(actual_len + 1):
                        self.window_index.append((wf_idx, 0, gap_pos))
                # Skip in other epochs (already saw full workflow in epoch 0)
                continue

            # Start from offset (skip first `offset` tokens)
            pos = self.current_offset

            # Create non-overlapping windows
            while pos + self.context_size <= len(tokens):
                # Enumerate all gap positions: 0 to context_size (inclusive)
                for gap_pos in range(self.context_size + 1):
                    self.window_index.append((wf_idx, pos, gap_pos))
                pos += self.context_size

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        """
        Get a FIM training example.

        FIM (Fill-in-the-Middle) training with TWO position tokens:
        1. Window position (offset in workflow, stride=32)
        2. Gap position (which token to predict within window, 0 to window_size)

        - Input: [window_pos, gap_pos, tok_0, ..., tok_{gap-1}, PAD, tok_{gap+1}, ..., tok_{N-2}]
        - Target: [gap_pos, tok_0, tok_1, ..., tok_{gap-1}, tok_gap, tok_{gap+1}, ..., tok_{N-1}]

        The model learns to predict tok_gap from both left context (tok_0...tok_{gap-1})
        and right context (tok_{gap+1}...tok_{N-1}).

        Gap positions are pre-enumerated in window_index (not random).

        Returns:
            (x, y) tuple:
            - x: Input tokens [context_size+1] = [window_pos, gap_pos, content_with_gap_masked]
            - y: Target tokens [context_size+1] = [gap_pos, content_tokens]
        """
        wf_idx, start, gap_pos = self.window_index[idx]  # Gap position from index (not random!)
        tokens = self.tokenized_workflows[wf_idx]

        # Extract window
        window = tokens[start:start + self.context_size]
        actual_len = len(window)

        # Pad if needed (for workflows smaller than context_size)
        # Use token 8191 (new dedicated PAD token in vocab_size=8192)
        PAD_TOKEN = 8191  # New PAD token (vocab_size = 8192, PAD at 8191)
        if actual_len < self.context_size:
            # Pad at end with 8191 (dedicated PAD token)
            padding = torch.full((self.context_size - actual_len,), PAD_TOKEN, dtype=torch.long)
            window = torch.cat([window, padding])

        # Create window position token (map to vocab space)
        # Use stride 32 for position mapping to keep positions < 8192
        # For very long workflows (> 262k tokens), use modulo to wrap around
        window_position = (start // 32) % 8192

        # Create FIM window: mask the gap position with PAD (if gap is within window)
        # Gap positions: 0 to actual_len (includes "after last token" for generation)
        # Example: window=[tok_0, tok_1] → gap_pos ∈ {0, 1, 2}
        #   - gap_pos=0: predict tok_0 (no left context)
        #   - gap_pos=1: predict tok_1 from tok_0 (left context)
        #   - gap_pos=2: predict tok_2 from [tok_0, tok_1] (next-token generation)
        fim_window = window.clone()
        if gap_pos < actual_len:
            # Gap within window: mask token at gap_pos
            fim_window[gap_pos] = PAD_TOKEN
        # else: gap_pos == actual_len (after last token) → no masking, predict next token

        # Prepend TWO position tokens: [window_pos, gap_pos, content...]
        window_with_positions = torch.cat([
            torch.tensor([window_position, gap_pos], dtype=torch.long),
            fim_window
        ])

        # x: [window_pos, gap_pos, tok_0, ..., tok_{N-2}] (with possible gap masked)
        # y: [gap_pos, tok_0, tok_1, ..., tok_{N-1}] (unmasked targets)
        x = window_with_positions[:-1].clone()
        y = window_with_positions[1:].clone()

        # Mask padding positions in targets: set to -1 so cross_entropy ignores them
        # Padding is at END (last N positions where N = context_size - actual_len)
        # Note: targets start with gap_pos (position 0), then content starts at position 1
        if actual_len < self.context_size:
            # Padding starts at position (1 + actual_len) in targets (after gap_pos + real tokens)
            y[1 + actual_len:] = -1  # Mask padding at end

        return x, y


def create_fim_epoch_offset_dataloader(
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
    Create FIM dataloader with epoch-based offset windowing.

    Enumerates ALL gap positions for each window, creating context_size+1 training
    examples per window. For context=512, this creates 513× more examples than
    standard autoregressive training.

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
    dataset = FIMEpochOffsetWorkflowDataset(
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
