"""
Multi-Gap Fill-in-the-Middle (FIM) Epoch-Based Offset DataLoader for TCT Workflows

Implements randomized multi-gap FIM training with epoch-based offsets:
- Infinite data augmentation: Geometric distribution sampling of gap positions
- Decoder-only compatible: 0 gaps = regular autoregressive training
- Bidirectional context: Model learns from both left and right context

Multi-Gap FIM Training:
- 0 gaps: Decoder-only mode (regular next-token prediction)
- 1 gap: Single-token FIM (like BERT masked LM)
- 2+ gaps: Multi-token FIM (extended bidirectional learning)

Gap Sampling Strategy (v1.2):
- Number of gaps: Geometric distribution with p=0.5 (mean=1.0 gap)
  - P(0 gaps) = 50% (decoder-only)
  - P(1 gap) = 25% (single-gap FIM)
  - P(2 gaps) = 12.5%
  - P(k gaps) = (0.5)^k * 0.5 (exponential decay)
- Max gaps: 10% of window length (scales with context)
- Gap positions: Uniform random sampling within window
- Mask token: 8190 (MASK) at gap positions
- Prediction: Token at FIRST gap position only

Key benefits:
- Conservative masking (mean=1.0 gap) respects TCT's context-dependency
- Heavily favors decoder-only and simple FIM (75% of examples have 0-1 gaps)
- Percentage-based max scales naturally with window size
- Efficient caching (reuses tokenized workflows)

Format:
- Input: [window_pos, tok_0, ..., tok_i, MASK, tok_j, ..., MASK, ..., tok_{N-2}]
- Target: [tok_0, tok_1, ..., tok_i, tok_first_gap, tok_j, ..., tok_{N-1}]

Only the token at the FIRST gap position is predicted (critical for TCT's
context-dependent tokens where masked tokens change meaning of subsequent tokens).
"""

import sys
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Import TCT functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tct-github-workflow/target/wheels"))
import tct_github_workflow as tct


class MultiGapFIMEpochOffsetWorkflowDataset(Dataset):
    """
    Dataset that creates multi-gap FIM training examples with epoch-based offset.

    Each epoch uses a different offset (0, 32, 64, ..., 992) to window the workflows.
    Within each window, we randomly sample 0-10 gap positions for FIM training.
    This provides effectively infinite data augmentation through random sampling.
    """

    def __init__(self, workflow_files, context_size=512, offset_stride=16, split="train",
                 workflow_dir=None, train_split=0.9, geometric_p=0.5, seed=None, gap_mode="multi"):
        """
        Initialize multi-gap FIM dataset.

        Args:
            workflow_files: List of paths to JSON workflow files
            context_size: Size of context window (default: 512)
            offset_stride: Stride for epoch offset (default: 16, gives 32 different views)
            split: "train" or "val" (affects shuffling)
            workflow_dir: Path to workflow directory (for caching, optional)
            train_split: Train/val split ratio (for cache naming, default: 0.9)
            geometric_p: Geometric distribution parameter (default: 0.5, mean=1.0 gap, only used when gap_mode="multi")
            seed: Random seed for reproducibility (optional)
            gap_mode: "multi" (random sampling) or "single" (enumerate all positions, default: "multi")
        """
        self.workflow_files = list(workflow_files)
        self.context_size = context_size
        self.offset_stride = offset_stride
        self.split = split
        self.geometric_p = geometric_p
        self.gap_mode = gap_mode

        # Random number generator for gap sampling
        self.rng = random.Random(seed)

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
                tokens = tct.encode(json_str)
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

        print(f"Multi-Gap FIM Dataset initialized: {len(self)} windows (offset={self.current_offset}, geometric_p={geometric_p})")

    def set_epoch(self, epoch):
        """
        Set the current epoch, which determines the windowing offset.

        Args:
            epoch: Current training epoch
        """
        # Cycle through offsets: 0, 16, 32, ..., 496, 0, 16, ...
        self.current_offset = (epoch * self.offset_stride) % self.context_size
        self._rebuild_window_index()

        # Re-seed RNG for this epoch (deterministic across runs)
        self.rng.seed(epoch)

        print(f"Epoch {epoch}: offset={self.current_offset}, {len(self)} windows")

    def _rebuild_window_index(self):
        """
        Build index of all windows for current offset.

        Mode-dependent behavior:
        - gap_mode="multi": Each window generates ONE example with randomly sampled gaps
          Index format: (workflow_idx, window_start)
        - gap_mode="single": Each window generates context_size+1 examples (enumerate all gaps)
          Index format: (workflow_idx, window_start, gap_pos)

        For offset=0: [0-511], [512-1023], ...
        For offset=16: [16-527], [528-1039], ... (skip first 16 tokens)

        For workflows smaller than context_size:
        - If offset=0: Include window (pad to context_size)
        - If offset>0 and workflow too small: Skip (already seen in epoch 0)
        """
        self.window_index = []

        for wf_idx, tokens in enumerate(self.tokenized_workflows):
            # Handle workflows smaller than context_size
            if len(tokens) < self.context_size:
                # Only include in epoch 0 (offset=0), pad to context_size
                if self.current_offset == 0:
                    if self.gap_mode == "single":
                        # Enumerate all gap positions for short workflow
                        for gap_pos in range(len(tokens) + 1):
                            self.window_index.append((wf_idx, 0, gap_pos))
                    else:
                        # Multi-gap: one entry per window
                        self.window_index.append((wf_idx, 0))
                # Skip in other epochs (already saw full workflow in epoch 0)
                continue

            # Start from offset (skip first `offset` tokens)
            pos = self.current_offset

            # Create non-overlapping windows
            while pos + self.context_size <= len(tokens):
                if self.gap_mode == "single":
                    # Single-gap enumeration: create entry for each gap position
                    for gap_pos in range(self.context_size + 1):
                        self.window_index.append((wf_idx, pos, gap_pos))
                else:
                    # Multi-gap random: one entry per window
                    self.window_index.append((wf_idx, pos))
                pos += self.context_size

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        """
        Get a FIM training example (single-gap or multi-gap depending on gap_mode).

        Single-Gap Enumeration (gap_mode="single"):
        - Systematically enumerate all gap positions (0 to context_size)
        - gap_pos < actual_len: FIM mode (mask that position)
        - gap_pos >= actual_len: decoder-only mode (predict next token)

        Multi-Gap Random (gap_mode="multi"):
        - Sample random number of gaps using geometric distribution
        - 0 gaps = decoder-only, 1-k gaps = FIM mode
        - Predict token at FIRST gap position only

        Format:
        - Input: [window_pos, tok_0, ..., MASK, ..., tok_{N-2}]
        - Target: [tok_0, tok_1, ..., tok_gap, ..., tok_{N-1}]

        Returns:
            (x, y) tuple:
            - x: Input tokens [context_size+1]
            - y: Target tokens [context_size+1]
        """
        # Unpack index based on gap_mode
        if self.gap_mode == "single":
            wf_idx, start, gap_pos = self.window_index[idx]
        else:
            wf_idx, start = self.window_index[idx]

        tokens = self.tokenized_workflows[wf_idx]

        # Extract window
        end = min(start + self.context_size, len(tokens))
        actual_len = end - start

        # Determine gap positions based on mode
        if self.gap_mode == "single":
            # Single-gap enumeration: use the pre-determined gap_pos
            if gap_pos < actual_len:
                gap_positions = [gap_pos]  # FIM mode
            else:
                gap_positions = []  # Decoder-only (predict next token)
        else:
            # Multi-gap random sampling
            import numpy as np
            p = self.geometric_p
            probs = np.array([(1-p)**k * p for k in range(actual_len + 1)])
            probs /= probs.sum()  # Normalize

            num_gaps = np.random.choice(actual_len + 1, p=probs)

            # Sample random gap positions
            if num_gaps > 0:
                gap_positions = sorted(self.rng.sample(range(actual_len), num_gaps))
            else:
                gap_positions = []  # Decoder-only mode

        # Use TCT API to create FIM window with multiple masks
        # extract_window_fim(tokens, start, end, gap_positions) -> [window_pos, content_with_masks]
        # Important: end must not exceed len(tokens)
        end = min(start + self.context_size, len(tokens))

        # Extract original unmasked window for target creation
        window = tokens[start:end]

        fim_window = tct.extract_window_fim(
            tokens.tolist(),  # Convert tensor to list
            start,
            end,
            gap_positions  # Empty list for decoder-only, or list of positions
        )

        # Convert back to tensor
        fim_window = torch.tensor(fim_window, dtype=torch.long)

        # Pad if needed (for workflows smaller than context_size)
        # TCT format: [window_pos, content...] where content may be shorter than context_size
        # We need to pad to context_size+1 (1 for window_pos + context_size for content)
        PAD_TOKEN = 8191
        expected_len = self.context_size + 1  # +1 for window position token
        if len(fim_window) < expected_len:
            # Pad at end with PAD token
            padding = torch.full((expected_len - len(fim_window),), PAD_TOKEN, dtype=torch.long)
            fim_window = torch.cat([fim_window, padding])

        # Extract window position (first token in FIM window)
        # TCT format: [window_pos, content...]
        # Apply modulo to keep position within vocab bounds (0-8191)
        window_position = fim_window[0].item() % 8192
        fim_window[0] = window_position

        # Create unmasked target window (for y)
        # Need to get original window with padding
        target_window = window.clone()
        if actual_len < self.context_size:
            padding = torch.full((self.context_size - actual_len,), PAD_TOKEN, dtype=torch.long)
            target_window = torch.cat([target_window, padding])

        # Prepend window position to target
        target_with_position = torch.cat([
            torch.tensor([window_position], dtype=torch.long),
            target_window
        ])

        # x: [window_pos, tok_0, ..., MASK, ..., tok_{N-2}] (with gaps masked)
        # y: [tok_0, tok_1, ..., tok_gap, ..., tok_{N-1}] (unmasked targets)
        x = fim_window[:-1].clone()  # Everything except last token
        y = target_with_position[1:].clone()  # Skip window position, get content

        # Mask padding positions in targets: set to -1 so cross_entropy ignores them
        if actual_len < self.context_size:
            # Padding starts at position actual_len in targets (after real tokens)
            y[actual_len:] = -1  # Mask padding at end

        return x, y


def create_multigap_fim_epoch_offset_dataloader(
    workflow_dir,
    context_size=512,
    offset_stride=16,
    batch_size=20,
    split="train",
    train_split=0.9,
    geometric_p=0.5,
    num_workers=0,
    pin_memory=True,
    seed=None,
    gap_mode="multi",
):
    """
    Create FIM dataloader with epoch-based offset windowing.

    Supports two modes:
    - gap_mode="multi": Random gap sampling using geometric distribution (infinite augmentation)
    - gap_mode="single": Systematic enumeration of all gap positions (513× multiplier for context=512)

    Args:
        workflow_dir: Directory containing JSON workflow files
        context_size: Window size (default: 512)
        offset_stride: Stride for epoch offset (default: 16)
        batch_size: Batch size (default: 20)
        split: "train" or "val"
        train_split: Fraction of data for training (default: 0.9)
        geometric_p: Geometric distribution parameter (default: 0.5, mean=1.0 gap)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for reproducibility (optional)

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
    dataset = MultiGapFIMEpochOffsetWorkflowDataset(
        workflow_files,
        context_size=context_size,
        offset_stride=offset_stride,
        split=split,
        workflow_dir=workflow_dir,
        train_split=train_split,
        geometric_p=geometric_p,
        seed=seed,
        gap_mode=gap_mode,
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
