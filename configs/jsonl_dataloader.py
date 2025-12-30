"""
Unified JSONL DataLoader for TCT Experiments.

Works with both TCT-BPE and UTF8-BPE tokenized data (same JSONL format).
Each sequence is an independent training example with no cross-sequence context.

Key Features:
- Works for all schemas (tsconfig, eslintrc, kubernetes)
- Works for both tokenizers (TCT-BPE, UTF8-BPE)
- PAD token is always 0
- Each batch contains B independent sequences (no concatenation)
- Inputs padded with 0 (PAD), targets padded with -1 (ignored in loss)
- Compatible with nanochat's training loop
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Iterator

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

# PAD token is always 0 in the new format
PAD_TOKEN_ID = 0


class JSONLDataset(Dataset):
    """
    Dataset for pre-encoded JSONL sequences.

    Each line in the JSONL file is a JSON array of token IDs.
    Sequences are padded/truncated to context_size for batching.
    """

    def __init__(
        self,
        jsonl_file: Path,
        context_size: int,
        max_len: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize dataset from pre-encoded JSONL file.

        Args:
            jsonl_file: Path to JSONL file (train.jsonl or validate.jsonl)
            context_size: Maximum sequence length (truncate if longer, pad if shorter)
            max_len: Filter out sequences longer than this (None = no filtering)
            verbose: Print loading statistics
        """
        self.context_size = context_size
        self.max_len = max_len
        self.verbose = verbose

        # Load all sequences into memory
        self.sequences = self._load_jsonl(jsonl_file)

        if verbose:
            self._print_stats()

    def _load_jsonl(self, jsonl_file: Path) -> list:
        """Load sequences from JSONL file."""
        sequences = []
        skipped = 0

        with open(jsonl_file, 'r') as f:
            for i, line in enumerate(f):
                tokens = json.loads(line)
                if self.max_len is None or len(tokens) <= self.max_len:
                    sequences.append(torch.tensor(tokens, dtype=torch.long))
                else:
                    skipped += 1

                # Progress indicator
                if self.verbose and (i + 1) % 100000 == 0:
                    print(f"  Loaded {i + 1:,} lines...")

        if self.verbose and skipped > 0:
            print(f"  Skipped {skipped:,} sequences longer than {self.max_len} tokens")

        return sequences

    def _print_stats(self):
        """Print dataset statistics."""
        if not self.sequences:
            print("Warning: Empty dataset!")
            return

        lengths = [len(s) for s in self.sequences]
        avg_len = sum(lengths) / len(lengths)
        truncated = sum(1 for l in lengths if l > self.context_size)
        padded = sum(1 for l in lengths if l < self.context_size)

        print(f"Loaded {len(self.sequences):,} sequences")
        print(f"  Average length: {avg_len:.1f} tokens")
        print(f"  Truncate (>{self.context_size}): {truncated:,} ({100*truncated/len(self.sequences):.1f}%)")
        print(f"  Pad (<{self.context_size}): {padded:,} ({100*padded/len(self.sequences):.1f}%)")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence as input/target pair for teacher forcing.

        Returns:
            inputs: [context_size] tensor, padded with 0
            targets: [context_size] tensor, padded with -1 (ignored in loss)
        """
        tokens = self.sequences[idx]

        # Truncate if too long
        if len(tokens) > self.context_size:
            tokens = tokens[:self.context_size]

        # Need context_size + 1 tokens for teacher forcing
        if len(tokens) < self.context_size + 1:
            needed = self.context_size + 1 - len(tokens)
            tokens = torch.cat([tokens, torch.full((needed,), PAD_TOKEN_ID, dtype=torch.long)])

        # Teacher forcing: shift by 1
        inputs = tokens[:-1].clone()   # First context_size tokens
        targets = tokens[1:].clone()   # Last context_size tokens

        # Replace padding in targets with -1 (ignored in loss via ignore_index)
        mask = inputs == PAD_TOKEN_ID
        targets[mask] = -1

        # Convert inputs to int32 (model expects this)
        inputs = inputs.to(dtype=torch.int32)

        return inputs, targets


def create_reshuffled_dataloaders(
    data_dir: Path,
    context_size: int,
    batch_size: int,
    train_ratio: float = 0.9,
    max_len: Optional[int] = None,
    device: str = "cuda",
    num_workers: int = 0,
    verbose: bool = True,
    seed: int = 42,
) -> tuple:
    """
    Load all data, shuffle randomly, and create train/val dataloaders.

    This fixes the sequential split issue by reshuffling the entire dataset.

    Args:
        data_dir: Directory containing train.jsonl and validate.jsonl
        context_size: Sequence length
        batch_size: Batch size
        train_ratio: Fraction for training (default 0.9)
        max_len: Filter sequences longer than this
        device: Device for tensors
        num_workers: DataLoader workers
        verbose: Print statistics
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader)
    """
    import random
    data_dir = Path(data_dir)

    # Load all sequences - prefer all.jsonl if available, otherwise combine train+val
    all_sequences = []
    all_jsonl = data_dir / "all.jsonl"

    if all_jsonl.exists():
        # Use all.jsonl directly
        with open(all_jsonl, 'r') as f:
            for line in f:
                tokens = json.loads(line)
                if max_len is None or len(tokens) <= max_len:
                    all_sequences.append(torch.tensor(tokens, dtype=torch.long))
        source = "all.jsonl"
    else:
        # Fallback: combine train.jsonl and validate.jsonl
        for jsonl_name in ["train.jsonl", "validate.jsonl"]:
            jsonl_file = data_dir / jsonl_name
            if jsonl_file.exists():
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        tokens = json.loads(line)
                        if max_len is None or len(tokens) <= max_len:
                            all_sequences.append(torch.tensor(tokens, dtype=torch.long))
        source = "train.jsonl + validate.jsonl"

    if verbose:
        print(f"Loaded {len(all_sequences):,} total sequences from {data_dir.name} ({source})")

    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(all_sequences)

    # Split
    split_idx = int(len(all_sequences) * train_ratio)
    train_sequences = all_sequences[:split_idx]
    val_sequences = all_sequences[split_idx:]

    if verbose:
        print(f"  Reshuffled split: {len(train_sequences):,} train, {len(val_sequences):,} val")

    # Create datasets directly from sequences
    class InMemoryDataset(Dataset):
        def __init__(self, sequences, context_size):
            self.sequences = sequences
            self.context_size = context_size

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            tokens = self.sequences[idx]
            if len(tokens) > self.context_size:
                tokens = tokens[:self.context_size]
            if len(tokens) < self.context_size + 1:
                needed = self.context_size + 1 - len(tokens)
                tokens = torch.cat([tokens, torch.full((needed,), PAD_TOKEN_ID, dtype=torch.long)])
            inputs = tokens[:-1].clone()
            targets = tokens[1:].clone()
            mask = inputs == PAD_TOKEN_ID
            targets[mask] = -1
            inputs = inputs.to(dtype=torch.int32)
            return inputs, targets

    def collate_fn(batch):
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.int64, non_blocking=True)
        return inputs, targets

    train_loader = DataLoader(
        InMemoryDataset(train_sequences, context_size),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        InMemoryDataset(val_sequences, context_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,
    )

    return train_loader, val_loader


def create_dataloader(
    data_dir: Path,
    context_size: int,
    batch_size: int,
    split: str = "train",
    max_len: Optional[int] = None,
    device: str = "cuda",
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    verbose: bool = True,
) -> DataLoader:
    """
    Create DataLoader for pre-encoded JSONL sequences.

    Args:
        data_dir: Directory containing train.jsonl and validate.jsonl
        context_size: Sequence length (from schema config)
        batch_size: Batch size
        split: "train" or "val"/"validate"
        max_len: Filter sequences longer than this (None = no filter)
        device: Device for tensors ("cuda" or "cpu")
        shuffle: Shuffle sequences (default: True for train, False for val)
        num_workers: DataLoader workers (0 = main process only)
        verbose: Print loading statistics

    Returns:
        DataLoader yielding (inputs, targets) batches of shape [B, context_size]

    Example:
        >>> from configs import get_schema_config
        >>> schema = get_schema_config("kubernetes")
        >>> loader = create_dataloader(
        ...     data_dir=schema["data_path_tct"],
        ...     context_size=schema["context_size"],
        ...     batch_size=16,
        ...     split="train",
        ... )
    """
    data_dir = Path(data_dir)

    # Normalize split name
    if split in ("val", "validate", "validation"):
        jsonl_file = data_dir / "validate.jsonl"
        split_name = "validate"
    else:
        jsonl_file = data_dir / "train.jsonl"
        split_name = "train"

    if not jsonl_file.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")

    # Load metadata if available
    metadata_file = data_dir / "metadata.json"
    if metadata_file.exists() and verbose:
        with open(metadata_file) as f:
            metadata = json.load(f)
        count_key = "train_count" if split_name == "train" else "validate_count"
        print(f"Dataset: {data_dir.name}")
        print(f"  Split: {split_name} ({metadata.get(count_key, '?'):,} sequences)")
        base_vocab = metadata.get('base_vocab_size', 0)
        full_vocab = base_vocab + 1 if base_vocab else '?'  # +1 for pad token
        print(f"  Vocab: {full_vocab:,}")

    # Create dataset
    dataset = JSONLDataset(
        jsonl_file,
        context_size=context_size,
        max_len=max_len,
        verbose=verbose,
    )

    # Auto-determine shuffle
    if shuffle is None:
        shuffle = (split_name == "train")

    # Collate function to move tensors to device
    def collate_fn(batch):
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])

        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.int64, non_blocking=True)

        return inputs, targets

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,  # Drop incomplete batches for consistent gradient accumulation
    )

    return loader


def get_epoch_steps(
    train_tokens: int,
    context_size: int,
    batch_size: int,
    gradient_accumulation: int = 1,
    world_size: int = 1,
) -> int:
    """
    Calculate number of training steps per epoch.

    Args:
        train_tokens: Total training tokens (from schema config)
        context_size: Context size (from schema config)
        batch_size: Micro batch size
        gradient_accumulation: Gradient accumulation steps
        world_size: Number of GPUs (for DDP)

    Returns:
        Number of optimizer steps per epoch
    """
    tokens_per_step = batch_size * context_size * gradient_accumulation * world_size
    return train_tokens // tokens_per_step


def get_warmup_steps(
    train_tokens: int,
    context_size: int,
    batch_size: int,
    gradient_accumulation: int = 1,
    world_size: int = 1,
    warmup_fraction: float = 0.05,
) -> int:
    """
    Calculate warmup steps as fraction of first epoch.

    Args:
        train_tokens: Total training tokens
        context_size: Context size
        batch_size: Micro batch size
        gradient_accumulation: Gradient accumulation steps
        world_size: Number of GPUs
        warmup_fraction: Fraction of epoch for warmup (default: 5%)

    Returns:
        Number of warmup steps
    """
    steps_per_epoch = get_epoch_steps(
        train_tokens, context_size, batch_size, gradient_accumulation, world_size
    )
    return max(1, int(steps_per_epoch * warmup_fraction))
