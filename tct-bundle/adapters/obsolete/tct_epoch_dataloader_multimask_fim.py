"""
Multi-Mask Fill-in-the-Middle (FIM) Epoch-Based Offset DataLoader for TCT Workflows

Implements randomized multi-mask FIM training with epoch-based offsets:
- Infinite data augmentation: Geometric distribution sampling of mask positions
- Decoder-only compatible: 0 masks = regular autoregressive training
- Bidirectional context: Model learns from both left and right context

Multi-Mask FIM Training:
- 0 masks: Decoder-only mode (regular next-token prediction)
- 1 mask: Single-token FIM (like BERT masked LM)
- 2+ masks: Multi-token FIM (extended bidirectional learning)

Mask Sampling Strategy (v1.2):
- Number of masks: Geometric distribution with p=0.5 (mean=1.0 mask)
  - P(0 masks) = 50% (decoder-only)
  - P(1 mask) = 25% (single-mask FIM)
  - P(2 masks) = 12.5%
  - P(k masks) = (0.5)^k * 0.5 (exponential decay)
- Max masks: 10% of window length (scales with context)
- Mask positions: Uniform random sampling within window
- Mask token: 8190 (MASK) at mask positions
- Prediction: Token at FIRST mask position only

Key benefits:
- Conservative masking (mean=1.0 mask) respects TCT's context-dependency
- Heavily favors decoder-only and simple FIM (75% of examples have 0-1 masks)
- Percentage-based max scales naturally with window size
- Efficient caching (reuses tokenized workflows)

Format:
- Input: [window_pos, tok_0, ..., tok_i, MASK, tok_j, ..., MASK, ..., tok_{N-2}]
- Target: [tok_0, tok_1, ..., tok_i, tok_first_mask, tok_j, ..., tok_{N-1}]

Only the token at the FIRST mask position is predicted (critical for TCT's
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


class MultiMaskFIMEpochOffsetWorkflowDataset(Dataset):
    """
    Dataset that creates multi-mask FIM training examples with epoch-based offset.

    Each epoch uses a different offset (0, 32, 64, ..., 992) to window the workflows.
    Within each window, we randomly sample 0-10 mask positions for FIM training.
    This provides effectively infinite data augmentation through random sampling.
    """

    def __init__(self, workflow_files, context_size=512, offset_stride=16, split="train",
                 workflow_dir=None, train_split=0.9, geometric_p=0.5, seed=None, mask_mode="randomized", mask_enum_max=1):
        """
        Initialize multi-mask FIM dataset.

        Args:
            workflow_files: List of paths to JSON workflow files
            context_size: Size of context window (default: 512)
            offset_stride: Stride for epoch offset (default: 16, gives 32 different views)
            split: "train" or "val" (affects shuffling)
            workflow_dir: Path to workflow directory (for caching, optional)
            train_split: Train/val split ratio (for cache naming, default: 0.9)
            geometric_p: Geometric distribution parameter (default: 0.5, mean=1.0 mask, only used when mask_mode="randomized")
            seed: Random seed for reproducibility (optional)
            mask_mode: "randomized" (random sampling) or "enum" (systematic enumeration, default: "randomized")
            mask_enum_max: Maximum masks to enumerate when mask_mode="enum" (1, 2, or 3, default: 1)
        """
        self.workflow_files = list(workflow_files)
        self.context_size = context_size
        self.offset_stride = offset_stride
        self.split = split
        self.geometric_p = geometric_p

        # Backward compatibility: map old modes to new modes
        if mask_mode == "multi":
            mask_mode = "randomized"
        elif mask_mode == "single":
            mask_mode = "enum"
            mask_enum_max = 1  # Single-mask enumeration

        self.mask_mode = mask_mode
        self.mask_enum_max = mask_enum_max

        # Random number generator for mask sampling
        self.rng = random.Random(seed)

        # Current epoch offset (set by set_epoch())
        self.current_offset = 0

        # Enumeration state (for mask_mode="enum")
        self.window_index = []  # List of (wf_idx, start, actual_len) for base windows
        self.enum_combinations_cache = {}  # Cache for mask combinations per window length
        self._cached_len = None  # Cache for dataset length
        self._cumulative_counts = None  # Precomputed cumulative counts for fast lookup

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
        if self.mask_mode == "enum":
            self._build_cumulative_counts()  # Build lookup table for fast index mapping

        if self.mask_mode == "enum":
            print(f"Multi-Mask FIM Dataset initialized: {len(self):,} enumerated examples (offset={self.current_offset}, mask_enum_max={self.mask_enum_max})")
        else:
            print(f"Multi-Mask FIM Dataset initialized: {len(self):,} windows (offset={self.current_offset}, geometric_p={geometric_p})")

    def set_epoch(self, epoch):
        """
        Set the current epoch, which determines the windowing offset.

        Args:
            epoch: Current training epoch
        """
        # Cycle through offsets: 0, 16, 32, ..., 496, 0, 16, ...
        self.current_offset = (epoch * self.offset_stride) % self.context_size
        self._rebuild_window_index()
        if self.mask_mode == "enum":
            self._build_cumulative_counts()  # Rebuild lookup table for new windows

        # Re-seed RNG for this epoch (deterministic across runs)
        self.rng.seed(epoch)

        print(f"Epoch {epoch}: offset={self.current_offset}, {len(self)} windows")

    def _rebuild_window_index(self):
        """
        Build index of all windows for current offset.

        Mode-dependent behavior:
        - mask_mode="randomized": Each window generates ONE example with randomly sampled masks
          Index format: (workflow_idx, window_start)
        - mask_mode="enum": Store base windows only, enumerate combinations on-the-fly
          Index format: (workflow_idx, window_start, actual_len)

        For offset=0: [0-511], [512-1023], ...
        For offset=16: [16-527], [528-1039], ... (skip first 16 tokens)

        For workflows smaller than context_size:
        - If offset=0: Include window (pad to context_size)
        - If offset>0 and workflow too small: Skip (already seen in epoch 0)
        """
        self.window_index = []
        self._cached_len = None  # Invalidate length cache
        self._cumulative_counts = None  # Invalidate cumulative counts

        for wf_idx, tokens in enumerate(self.tokenized_workflows):
            # Handle workflows smaller than context_size
            if len(tokens) < self.context_size:
                # Only include in epoch 0 (offset=0), pad to context_size
                if self.current_offset == 0:
                    actual_len = len(tokens)
                    if self.mask_mode == "enum":
                        # Store base window with actual length
                        self.window_index.append((wf_idx, 0, actual_len))
                    else:
                        # Randomized: one entry per window
                        self.window_index.append((wf_idx, 0))
                # Skip in other epochs (already saw full workflow in epoch 0)
                continue

            # Start from offset (skip first `offset` tokens)
            pos = self.current_offset

            # Create non-overlapping windows
            while pos + self.context_size <= len(tokens):
                if self.mask_mode == "enum":
                    # Store base window with actual length (always context_size for full windows)
                    self.window_index.append((wf_idx, pos, self.context_size))
                else:
                    # Randomized: one entry per window
                    self.window_index.append((wf_idx, pos))
                pos += self.context_size

    def _count_combinations(self, n, max_k):
        """
        Count total combinations for enumerating masks from k=0 to max_k.

        Returns sum of C(n, k) for k in [0, max_k].
        """
        from math import comb
        total = 0
        for k in range(min(max_k, n) + 1):
            total += comb(n, k)
        return total

    def _build_cumulative_counts(self):
        """Build cumulative counts for fast binary search in enumeration mode."""
        if self.mask_mode != "enum":
            return

        self._cumulative_counts = []
        cumulative = 0
        for wf_idx, start, actual_len in self.window_index:
            window_combinations = self._count_combinations(actual_len, self.mask_enum_max)
            cumulative += window_combinations
            self._cumulative_counts.append(cumulative)

    def _map_idx_to_combination(self, flat_idx):
        """
        Map flat index to (window_idx, mask_positions).

        For mask_mode="enum", converts a global flat index into:
        1. Which base window it belongs to
        2. Which k-mask combination within that window

        Returns: (window_idx, mask_positions)
            window_idx: Index into self.window_index
            mask_positions: List of mask positions (empty for decoder-only)
        """
        from math import comb
        from itertools import combinations
        import bisect

        # Build cumulative counts if not already built
        if self._cumulative_counts is None:
            self._build_cumulative_counts()

        # Binary search to find which window this index belongs to
        window_idx = bisect.bisect_right(self._cumulative_counts, flat_idx)

        if window_idx >= len(self.window_index):
            raise IndexError(f"Index {flat_idx} out of range (total: {self._cumulative_counts[-1] if self._cumulative_counts else 0})")

        # Get window info
        wf_idx, start, actual_len = self.window_index[window_idx]

        # Calculate local index within this window
        cumulative_before = self._cumulative_counts[window_idx - 1] if window_idx > 0 else 0
        local_idx = flat_idx - cumulative_before

        # Find which k and which combination
        for k in range(min(self.mask_enum_max, actual_len) + 1):
            k_combinations = comb(actual_len, k)

            if local_idx < k_combinations:
                # This is the k-th mask count
                if k == 0:
                    return (window_idx, [])  # Decoder-only (no masks)
                elif k == 1:
                    # Optimized: For single masks, directly map index to position
                    mask_positions = [local_idx]
                    return (window_idx, mask_positions)
                else:
                    # Generate the local_idx-th combination of k masks from actual_len positions
                    # Use itertools.combinations to enumerate
                    all_combinations = list(combinations(range(actual_len), k))
                    mask_positions = list(all_combinations[local_idx])
                    return (window_idx, mask_positions)

            local_idx -= k_combinations

        raise IndexError(f"Could not map index {flat_idx} within window {window_idx}")

    def __len__(self):
        if self._cached_len is not None:
            return self._cached_len

        if self.mask_mode == "enum":
            # Compute total combinations across all windows
            total = 0
            for entry in self.window_index:
                _, _, actual_len = entry
                total += self._count_combinations(actual_len, self.mask_enum_max)
            self._cached_len = total
            return total
        else:
            self._cached_len = len(self.window_index)
            return self._cached_len

    def __getitem__(self, idx):
        """
        Get a FIM training example.

        Enumeration Mode (mask_mode="enum"):
        - Systematically enumerate mask combinations (k=0 to mask_enum_max)
        - k=0: Decoder-only mode (no masks)
        - k=1: Single-mask FIM (enumerate all single positions)
        - k=2: Two-mask FIM (enumerate all pairs)
        - etc.

        Randomized Mode (mask_mode="randomized"):
        - Sample random number of masks using geometric distribution
        - 0 masks = decoder-only, 1-k masks = FIM mode
        - Predict token at FIRST mask position only

        Format:
        - Input: [window_pos, tok_0, ..., MASK, ..., tok_{N-2}]
        - Target: [tok_0, tok_1, ..., tok_mask, ..., tok_{N-1}]

        Returns:
            (x, y) tuple:
            - x: Input tokens [context_size+1]
            - y: Target tokens [context_size+1]
        """
        # Determine mask positions based on mode
        if self.mask_mode == "enum":
            # Enumeration: map flat index to window and mask combination
            window_idx, mask_positions = self._map_idx_to_combination(idx)
            wf_idx, start, actual_len = self.window_index[window_idx]
            tokens = self.tokenized_workflows[wf_idx]
        else:
            # Randomized: unpack window and sample masks
            wf_idx, start = self.window_index[idx]
            tokens = self.tokenized_workflows[wf_idx]

            # Extract window
            end = min(start + self.context_size, len(tokens))
            actual_len = end - start

            # Random sampling using geometric distribution
            import numpy as np
            p = self.geometric_p
            probs = np.array([(1-p)**k * p for k in range(actual_len + 1)])
            probs /= probs.sum()  # Normalize

            num_masks = np.random.choice(actual_len + 1, p=probs)

            # Sample random mask positions
            if num_masks > 0:
                mask_positions = sorted(self.rng.sample(range(actual_len), num_masks))
            else:
                mask_positions = []  # Decoder-only mode

        # Extract window (common for both modes)
        end = min(start + self.context_size, len(tokens))
        actual_len = end - start

        # Use TCT API to create FIM window with multiple masks
        # extract_window_fim(tokens, start, end, mask_positions) -> [window_pos, content_with_masks]
        # Important: end must not exceed len(tokens)
        end = min(start + self.context_size, len(tokens))

        # Extract original unmasked window for target creation
        window = tokens[start:end]

        fim_window = tct.extract_window_fim(
            tokens.tolist(),  # Convert tensor to list
            start,
            end,
            mask_positions  # Empty list for decoder-only, or list of positions
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

        # x: [window_pos, tok_0, ..., MASK, ..., tok_{N-2}] (with masks)
        # y: [tok_0, tok_1, ..., tok_mask, ..., tok_{N-1}] (unmasked targets)
        x = fim_window[:-1].clone()  # Everything except last token
        y = target_with_position[1:].clone()  # Skip window position, get content

        # Mask padding positions in targets: set to -1 so cross_entropy ignores them
        if actual_len < self.context_size:
            # Padding starts at position actual_len in targets (after real tokens)
            y[actual_len:] = -1  # Mask padding at end

        return x, y


def create_multimask_fim_epoch_offset_dataloader(
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
    mask_mode="randomized",
    mask_enum_max=1,
):
    """
    Create FIM dataloader with epoch-based offset windowing.

    Supports two modes:
    - mask_mode="randomized": Random mask sampling using geometric distribution (infinite augmentation)
    - mask_mode="enum": Systematic enumeration of mask combinations up to mask_enum_max
      - mask_enum_max=1: 513× multiplier (enumerate 0,1 masks per window, context=512)
      - mask_enum_max=2: 131k× multiplier (enumerate 0,1,2 masks)
      - mask_enum_max=3: 22M× multiplier (enumerate 0,1,2,3 masks)

    Args:
        workflow_dir: Directory containing JSON workflow files
        context_size: Window size (default: 512)
        offset_stride: Stride for epoch offset (default: 16)
        batch_size: Batch size (default: 20)
        split: "train" or "val"
        train_split: Fraction of data for training (default: 0.9)
        geometric_p: Geometric distribution parameter (default: 0.5, mean=1.0 mask, only for randomized mode)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        seed: Random seed for reproducibility (optional)
        mask_mode: "randomized" or "enum" (default: "randomized")
        mask_enum_max: Max masks to enumerate when mode="enum" (1, 2, or 3, default: 1)

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
    dataset = MultiMaskFIMEpochOffsetWorkflowDataset(
        workflow_files,
        context_size=context_size,
        offset_stride=offset_stride,
        split=split,
        workflow_dir=workflow_dir,
        train_split=train_split,
        geometric_p=geometric_p,
        seed=seed,
        mask_mode=mask_mode,
        mask_enum_max=mask_enum_max,
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
