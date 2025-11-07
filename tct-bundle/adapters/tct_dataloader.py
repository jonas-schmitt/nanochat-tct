"""
Prefix-Aware Fill-in-the-Middle (FIM) DataLoader for TCT Workflows

Addresses core training issues by enumerating/sampling ALL prefix lengths:
- Trains on contexts of length 1, 2, 3, ..., context_size
- Matches inference distribution (generation starts from empty context)
- Uses geometric FIM sampling for each prefix (balanced k=0/k=1)

Key Innovation:
Instead of fixed 512-token windows, create examples with varying prefix lengths:
  - Prefix 1: [] → tok0 (with geometric FIM)
  - Prefix 2: [tok0] → tok1 (with geometric FIM)
  - ...
  - Prefix 512: [tok0...tok510] → tok511 (with geometric FIM)

This fixes the train/test distribution mismatch where model only saw 512-token
contexts but inference needs to work with 0, 1, 2, ... contexts.

Prefix Sampling Modes:
- "all": Enumerate every prefix length 1-N (N examples per window)
- "log": Log-spaced sampling [1,2,4,8,16,...,N] (~log2(N) examples)
- "sample": Uniform random sample (configurable count)
- "hybrid": Enumerate short (1-64) + sample long (recommended)

Interface:
- Compatible with nanochat's training loop (same __getitem__ format)
- Drop-in replacement for old dataloader
- No changes to nanochat core required
"""

import sys
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np

# Import TCT functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tct-github-workflow/target/wheels"))
import tct_github_workflow as tct


class TCTEpochPrefixFIMDataset(Dataset):
    """
    Dataset with prefix-aware geometric FIM training.

    Creates multiple examples per base window, one for each sampled prefix length.
    For each prefix, uses geometric distribution to sample FIM masks.
    """

    def __init__(self, workflow_files, context_size=512, offset_stride=32,
                 split="train", workflow_dir=None, train_split=0.9,
                 geometric_p=0.5, seed=None,
                 prefix_mode="log", prefix_count=20, prefix_bias="uniform",
                 cache_file=None):
        """
        Initialize prefix-aware FIM dataset.

        Args:
            workflow_files: List of paths to JSON workflow files
            context_size: Maximum context size (default: 512)
            offset_stride: Stride for epoch offset (default: 32)
            split: "train" or "val"
            workflow_dir: Path for caching (optional)
            train_split: Train/val split ratio (default: 0.9)
            geometric_p: Geometric distribution parameter for FIM (default: 0.5)
                        p=1.0 → 100% decoder-only (k=0)
                        p=0.5 → 50% k=0, 50% k=1+
                        p=0.33 → 33% k=0, 67% k=1+
            seed: Random seed (optional)
            prefix_mode: Prefix sampling strategy:
                        - "all": Enumerate all prefix lengths 1-N (512 samples)
                        - "log": Log-spaced [1,2,4,8,16,32,...,N] (~10 samples)
                        - "linear": Linear-spaced using prefix_count samples
                        - "sample": Random sample prefix_count lengths
                        - "hybrid": Enumerate 1-64 + sample remaining (~164 samples)
            prefix_count: Number of prefixes for "linear"/"sample"/"hybrid" (default: 100)
            prefix_bias: "uniform" or "short" (bias toward short contexts)
            cache_file: Explicit cache file path (None = auto-detect, "none" = disable)
        """
        self.workflow_files = list(workflow_files)
        self.context_size = context_size
        self.offset_stride = offset_stride
        self.split = split
        self.geometric_p = geometric_p
        self.prefix_mode = prefix_mode
        self.prefix_count = prefix_count
        self.prefix_bias = prefix_bias

        # Random number generator
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Current epoch offset
        self.current_offset = 0

        # Index: (workflow_idx, window_start, prefix_length)
        self.prefix_index = []
        self._cached_len = None

        # Load and tokenize workflows (reuse caching logic from old dataloader)
        self.tokenized_workflows = []

        # Determine cache file to use
        actual_cache_file = None
        if cache_file == "none":
            # Explicitly disable cache
            print("Cache disabled (cache_file='none')")
        elif cache_file:
            # Use explicit cache file path
            actual_cache_file = Path(cache_file)
            if not actual_cache_file.exists():
                print(f"Warning: Specified cache file not found: {actual_cache_file}")
                actual_cache_file = None
        elif workflow_dir is not None:
            # Auto-detect cache file
            cache_dir = Path(workflow_dir) / ".cache"
            cache_dir.mkdir(exist_ok=True)
            split_pct = int(train_split * 100)
            num_files = len(self.workflow_files)
            auto_cache_file = cache_dir / f"tokenized_{split}_split{split_pct}_{num_files}files.pt"

            if auto_cache_file.exists():
                actual_cache_file = auto_cache_file
            else:
                # Try to find any cache file for this split (ignore file count)
                import glob
                cache_pattern = str(cache_dir / f"tokenized_{split}_split{split_pct}_*.pt")
                existing_caches = sorted(glob.glob(cache_pattern))
                if existing_caches:
                    # Use the most recent cache
                    actual_cache_file = Path(existing_caches[-1])
                    print(f"Warning: Exact cache not found, using latest: {actual_cache_file.name}")

        # Load cache if available
        if actual_cache_file and actual_cache_file.exists():
            print(f"Loading tokenized workflows from cache: {actual_cache_file}")
            self.tokenized_workflows = torch.load(actual_cache_file)
            print(f"Loaded {len(self.tokenized_workflows)} tokenized workflows from cache ✅")

        # Tokenize if not cached
        if len(self.tokenized_workflows) == 0:
            print(f"Tokenizing {len(self.workflow_files)} workflows...", flush=True)
            failed_count = 0
            removed_files = []
            for i, wf_file in enumerate(self.workflow_files):
                if (i + 1) % 10000 == 0:
                    print(f"  Progress: {i+1:,}/{len(self.workflow_files):,} workflows", flush=True)
                try:
                    with open(wf_file) as f:
                        workflow_json = f.read()
                    tokens = list(tct.encode(workflow_json))
                    self.tokenized_workflows.append(torch.tensor(tokens, dtype=torch.long))
                except Exception as e:
                    failed_count += 1
                    removed_files.append(wf_file)
                    if failed_count <= 10:  # Print first 10 failures
                        print(f"  ⚠️  Removing invalid workflow {Path(wf_file).name}: {str(e)[:100]}")
                    # Remove the invalid file
                    try:
                        Path(wf_file).unlink()
                    except Exception as rm_error:
                        print(f"  ⚠️  Failed to remove {Path(wf_file).name}: {rm_error}")

            if failed_count > 0:
                print(f"  Removed {failed_count} invalid workflows ({100*failed_count/len(self.workflow_files):.1f}%)", flush=True)
                print(f"  Successfully tokenized {len(self.tokenized_workflows)} workflows", flush=True)

            # Save cache (only if not explicitly disabled)
            if cache_file != "none" and workflow_dir is not None:
                # Save to auto-detected location (don't overwrite explicit cache_file)
                cache_dir = Path(workflow_dir) / ".cache"
                cache_dir.mkdir(exist_ok=True)
                split_pct = int(train_split * 100)
                num_files = len(self.workflow_files)
                save_cache_file = cache_dir / f"tokenized_{split}_split{split_pct}_{num_files}files.pt"

                print(f"Saving tokenized workflows to cache: {save_cache_file}")
                try:
                    torch.save(self.tokenized_workflows, save_cache_file)
                    print(f"Cache saved successfully ✅")
                except Exception as e:
                    print(f"Warning: Failed to save cache ({e})")

        # Build initial prefix index
        self._rebuild_prefix_index()

        # Print dataset info
        k0_pct = int(geometric_p * 100)
        k1_pct = 100 - k0_pct
        print(f"Prefix FIM Dataset initialized:", flush=True)
        print(f"  Total examples: {len(self):,}", flush=True)
        print(f"  Prefix mode: {prefix_mode}", flush=True)
        print(f"  FIM distribution: ~{k0_pct}% k=0, ~{k1_pct}% k=1+ (geometric_p={geometric_p})", flush=True)
        print(f"  Offset: {self.current_offset}", flush=True)

    def set_epoch(self, epoch):
        """Set current epoch (updates offset for windowing)."""
        self.current_offset = (epoch * self.offset_stride) % self.context_size
        self._rebuild_prefix_index()
        self.rng.seed(epoch)
        self.np_rng.seed(epoch)
        print(f"Epoch {epoch}: offset={self.current_offset}, {len(self):,} examples")

    def _rebuild_prefix_index(self):
        """Build index of (workflow_idx, window_start, prefix_length) tuples."""
        print(f"Building prefix index (mode={self.prefix_mode})...", flush=True)
        self.prefix_index = []
        self._cached_len = None

        for wf_idx, tokens in enumerate(self.tokenized_workflows):
            if (wf_idx + 1) % 10000 == 0:
                print(f"  Progress: {wf_idx+1:,}/{len(self.tokenized_workflows):,} workflows, {len(self.prefix_index):,} examples so far", flush=True)
            # Get window positions for this workflow
            window_positions = self._get_window_positions(len(tokens))

            for window_start in window_positions:
                # Determine max prefix length for this window
                max_prefix_len = min(self.context_size, len(tokens) - window_start)

                # Sample prefix lengths
                prefix_lengths = self._sample_prefix_lengths(max_prefix_len)

                # Add to index
                for prefix_len in prefix_lengths:
                    self.prefix_index.append((wf_idx, window_start, prefix_len))

    def _get_window_positions(self, workflow_len):
        """
        Get non-overlapping window start positions for a workflow.

        CRITICAL: In epochs with offset > 0, we MUST include an initial short
        window [0:offset] to ensure ALL tokens are seen in every epoch.
        Without this, tokens [0:offset) are never trained on in later epochs,
        creating systematic data gaps for beginning-of-workflow tokens.
        """
        positions = []

        # Handle workflows smaller than context_size
        if workflow_len < self.context_size:
            if self.current_offset == 0:
                positions.append(0)  # Only include in epoch 0
            return positions

        # CRITICAL FIX: Add initial short window when offset > 0
        # This window will be [0:offset] and padded to context_size in __getitem__
        # Ensures all tokens from the beginning of the workflow are trained on
        if self.current_offset > 0:
            positions.append(0)

        # Create non-overlapping windows starting from offset
        pos = self.current_offset
        while pos + self.context_size <= workflow_len:
            positions.append(pos)
            pos += self.context_size

        # CRITICAL FIX: Add final short window if there are remaining tokens
        # This ensures tokens at the end of the workflow are also trained on
        if pos < workflow_len:
            positions.append(pos)

        return positions

    def _sample_prefix_lengths(self, max_len):
        """Sample which prefix lengths to use for a window."""
        if self.prefix_mode == "all":
            # Enumerate all prefix lengths
            return list(range(1, max_len + 1))

        elif self.prefix_mode == "log":
            # Log-spaced sampling: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            lengths = []
            power = 0
            while True:
                length = 2 ** power
                if length > max_len:
                    break
                lengths.append(length)
                power += 1
            # Always include max_len if not already there
            if max_len not in lengths:
                lengths.append(max_len)
            return lengths

        elif self.prefix_mode == "linear":
            # Linear-spaced sampling: evenly distribute prefix_count samples
            count = min(self.prefix_count, max_len)
            if count >= max_len:
                # If count >= max_len, just enumerate all
                return list(range(1, max_len + 1))

            # Linspace from 1 to max_len
            # Always include 1 and max_len, distribute rest evenly
            if count <= 2:
                return [1, max_len]

            step = (max_len - 1) / (count - 1)
            samples = [int(round(1 + i * step)) for i in range(count)]

            # Ensure unique and sorted
            samples = sorted(set(samples))

            # Ensure we have 1 and max_len
            if 1 not in samples:
                samples = [1] + samples
            if max_len not in samples:
                samples.append(max_len)

            return samples

        elif self.prefix_mode == "sample":
            # Random uniform or biased sampling
            count = min(self.prefix_count, max_len)

            if self.prefix_bias == "short":
                # Bias toward shorter contexts (important for generation)
                weights = 1.0 / np.arange(1, max_len + 1)
                weights = weights / weights.sum()
                samples = self.np_rng.choice(max_len, size=count, replace=False, p=weights) + 1
            else:
                # Uniform sampling
                samples = self.np_rng.choice(max_len, size=count, replace=False) + 1

            return sorted(samples)

        elif self.prefix_mode == "hybrid":
            # Enumerate short (1-64), sample long (65-max)
            short_end = min(64, max_len)
            short = list(range(1, short_end + 1))

            if max_len > 64:
                # Sample from remaining range
                remaining_count = min(self.prefix_count - len(short), max_len - 64)
                long_samples = self.np_rng.choice(
                    range(65, max_len + 1),
                    size=remaining_count,
                    replace=False
                )
                return short + sorted(long_samples.tolist())

            return short

        else:
            raise ValueError(f"Unknown prefix_mode: {self.prefix_mode}")

    def __len__(self):
        """Return total number of examples."""
        if self._cached_len is None:
            self._cached_len = len(self.prefix_index)
        return self._cached_len

    def __getitem__(self, idx):
        """
        Get training example at index.

        Returns:
            (x, y) tuple:
            - x: Input tokens [context_size+1] with masks
            - y: Target tokens [context_size+1] unmasked
        """
        wf_idx, window_start, prefix_len = self.prefix_index[idx]
        tokens = self.tokenized_workflows[wf_idx]

        # Calculate window end
        window_end = min(window_start + prefix_len, len(tokens))
        actual_len = window_end - window_start

        # Geometric sample number of masks
        num_masks = self._geometric_sample_masks(actual_len)

        # Sample random mask positions
        if num_masks > 0:
            mask_positions = sorted(self.rng.sample(range(actual_len), num_masks))
        else:
            mask_positions = []  # Decoder-only mode

        # Extract original window for target
        window = tokens[window_start:window_end]

        # Use TCT API to create FIM window
        fim_window = tct.extract_window_fim(
            tokens.tolist(),
            window_start,
            window_end,
            mask_positions
        )

        # Convert to tensor
        fim_window = torch.tensor(fim_window, dtype=torch.long)

        # Pad to context_size+1 (1 for window position + context_size for content)
        PAD_TOKEN = 8191
        expected_len = self.context_size + 1
        if len(fim_window) < expected_len:
            padding = torch.full((expected_len - len(fim_window),), PAD_TOKEN, dtype=torch.long)
            fim_window = torch.cat([fim_window, padding])

        # Apply modulo to window position to keep in vocab bounds
        window_position = fim_window[0].item() % 8192
        fim_window[0] = window_position

        # Create target window
        target_window = window.clone()
        if actual_len < self.context_size:
            padding = torch.full((self.context_size - actual_len,), PAD_TOKEN, dtype=torch.long)
            target_window = torch.cat([target_window, padding])

        # Prepend window position to target
        target_with_position = torch.cat([
            torch.tensor([window_position], dtype=torch.long),
            target_window
        ])

        # Create (x, y) pair
        x = fim_window[:-1].clone()  # Everything except last token
        y = target_with_position[1:].clone()  # Skip window position

        # Mask padding in targets (set to -1 so loss ignores them)
        if actual_len < self.context_size:
            y[actual_len:] = -1

        return x, y

    def _geometric_sample_masks(self, max_masks):
        """Sample number of masks using geometric distribution."""
        # P(k masks) = (1-p)^k * p
        p = self.geometric_p
        probs = np.array([(1-p)**k * p for k in range(max_masks + 1)])
        probs = probs / probs.sum()  # Normalize

        num_masks = self.np_rng.choice(max_masks + 1, p=probs)
        return num_masks


def tokenizing_distributed_data_loader(device_batch_size, context_size=512, split="train",
                                      data_dir=None, train_split=0.9, geometric_p=0.5,
                                      prefix_mode="hybrid", prefix_count=100, prefix_bias="uniform",
                                      cache_file=None):
    """
    Create dataloader with prefix-aware FIM training.

    Drop-in replacement for old dataloader function.
    Compatible with nanochat's training loop.

    Args:
        cache_file: Explicit cache file path (None = auto-detect, "none" = disable cache)
    """
    import glob

    if data_dir is None:
        raise ValueError("data_dir must be specified")

    # Load workflow files
    workflow_files = sorted(glob.glob(f"{data_dir}/*.json"))

    if len(workflow_files) == 0:
        raise ValueError(f"No workflow files found in {data_dir}")

    # Split train/val
    split_idx = int(len(workflow_files) * train_split)
    if split == "train":
        workflow_files = workflow_files[:split_idx]
    else:
        workflow_files = workflow_files[split_idx:]

    print(f"Loading {split} split: {len(workflow_files)} workflows")

    # Create dataset
    dataset = TCTEpochPrefixFIMDataset(
        workflow_files=workflow_files,
        context_size=context_size,
        split=split,
        workflow_dir=data_dir,
        train_split=train_split,
        geometric_p=geometric_p,
        prefix_mode=prefix_mode,
        prefix_count=prefix_count,
        prefix_bias=prefix_bias,
        cache_file=cache_file,
    )

    return dataset
