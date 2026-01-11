"""
Unified JSONL DataLoader for TCT Experiments.

Works with both TCT-BPE and UTF8-BPE tokenized data (same JSONL format).
Each sequence is an independent training example with no cross-sequence context.

Key Features:
- Works for all schemas (tsconfig, eslintrc, kubernetes)
- Works for both tokenizers (TCT-BPE, UTF8-BPE)
- PAD token is vocab_size - 1 (read from metadata.json)
- Each batch contains B independent sequences (no concatenation)
- Inputs padded with PAD, targets padded with -1 (ignored in loss)
- Compatible with nanochat's training loop
- Coordinated filtering: excludes manifests exceeding context window in EITHER tokenization
- Optional BOS token prepending for unconditional generation (use_bos=True)

IMPORTANT: PAD token must be the LAST token in vocab (vocab_size - 1), NOT 0!
Token 0 is a valid data token (e.g., start-of-object in TCT).

BOS Token Behavior (when use_bos=True):
- PAD token is prepended as BOS to each sequence
- Model learns P(first_token | BOS), enabling true unconditional generation
- During generation, start with [BOS] and sample the first token
- All PAD tokens in targets are masked (set to -1, ignored by cross_entropy)
"""

import json
from pathlib import Path
from typing import Optional, Tuple, Iterator, List

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset


def _load_sequence_lengths(data_dir: Path) -> List[int]:
    """Load sequence lengths from all.jsonl without keeping full sequences in memory.

    Used for coordinated filtering between TCT and UTF8 tokenizations.
    """
    lengths = []
    all_jsonl = data_dir / "all.jsonl"

    if all_jsonl.exists():
        with open(all_jsonl, 'r') as f:
            for line in f:
                tokens = json.loads(line)
                lengths.append(len(tokens))
    else:
        # Fallback: combine train + validate
        for jsonl_name in ["train.jsonl", "validate.jsonl"]:
            jsonl_file = data_dir / jsonl_name
            if jsonl_file.exists():
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        tokens = json.loads(line)
                        lengths.append(len(tokens))

    return lengths


def _load_all_sequences(data_dir: Path) -> List[List[int]]:
    """Load all sequences from all.jsonl (or train+validate fallback).

    Returns list of token lists (not tensors) for memory efficiency during filtering.
    """
    all_jsonl = data_dir / "all.jsonl"
    sequences = []

    if all_jsonl.exists():
        with open(all_jsonl, 'r') as f:
            for line in f:
                sequences.append(json.loads(line))
    else:
        # Fallback: combine train + validate
        for jsonl_name in ["train.jsonl", "validate.jsonl"]:
            jsonl_file = data_dir / jsonl_name
            if jsonl_file.exists():
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        sequences.append(json.loads(line))

    return sequences


def get_validation_sequences(
    data_dir: Path,
    max_len: Optional[int] = None,
    partner_data_dir: Optional[Path] = None,
    train_ratio: float = 0.95,
    seed: int = 42,
    max_samples: Optional[int] = None,
    verbose: bool = False,
    use_bos: bool = True,
) -> List[List[int]]:
    """Get validation sequences using the same logic as training.

    This ensures training and evaluation use identical validation splits.
    The process is:
    1. Load all sequences from all.jsonl
    2. Apply filtering (max_len, coordinated filtering with partner)
    3. Shuffle with fixed seed
    4. Return the validation portion (last 1-train_ratio)
    5. Optionally prepend BOS token (to match training behavior)

    Args:
        data_dir: Directory containing all.jsonl
        max_len: Filter sequences longer than this (None = no filtering)
        partner_data_dir: Partner data directory for coordinated filtering.
            When provided, sequences are excluded if they exceed max_len in
            EITHER this directory or partner_data_dir.
        train_ratio: Fraction for training (default 0.95), validation is 1-train_ratio
        seed: Random seed for reproducibility (default 42, same as training)
        max_samples: Limit number of validation samples returned (None = all)
        verbose: Print statistics
        use_bos: If True (default), prepend pad_token_id as BOS to match training.
            IMPORTANT: Training uses use_bos=True by default, so evaluation should too.

    Returns:
        List of token sequences (as lists, not tensors)
    """
    import random

    data_dir = Path(data_dir)

    # Load all sequences
    all_sequences = _load_all_sequences(data_dir)
    original_count = len(all_sequences)

    if verbose:
        print(f"  Loaded {original_count:,} sequences from {data_dir.name}")

    # Apply filtering
    if max_len is not None:
        if partner_data_dir is not None:
            partner_data_dir = Path(partner_data_dir)
            if partner_data_dir.exists():
                partner_lengths = _load_sequence_lengths(partner_data_dir)

                if len(partner_lengths) == len(all_sequences):
                    # Filter to indices valid in BOTH tokenizations
                    valid_indices = [
                        i for i in range(len(all_sequences))
                        if len(all_sequences[i]) <= max_len and partner_lengths[i] <= max_len
                    ]
                    all_sequences = [all_sequences[i] for i in valid_indices]

                    if verbose:
                        excluded = original_count - len(all_sequences)
                        print(f"  Coordinated filtering: kept {len(all_sequences):,}, excluded {excluded:,}")
                else:
                    # Length mismatch, fall back to standard filtering
                    all_sequences = [s for s in all_sequences if len(s) <= max_len]
                    if verbose:
                        print(f"  WARNING: Partner length mismatch, using standard filtering")
            else:
                all_sequences = [s for s in all_sequences if len(s) <= max_len]
        else:
            all_sequences = [s for s in all_sequences if len(s) <= max_len]

        if verbose and partner_data_dir is None:
            excluded = original_count - len(all_sequences)
            print(f"  Standard filtering: kept {len(all_sequences):,}, excluded {excluded:,}")

    # Shuffle with fixed seed (same as training)
    n = len(all_sequences)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    # Get validation indices (last portion after split)
    split_idx = int(n * train_ratio)
    val_indices = indices[split_idx:]

    # Extract validation sequences
    val_sequences = [all_sequences[i] for i in val_indices]

    if verbose:
        print(f"  Validation split: {len(val_sequences):,} sequences (seed={seed}, ratio={1-train_ratio:.0%})")

    # Limit samples if requested
    if max_samples is not None and len(val_sequences) > max_samples:
        val_sequences = val_sequences[:max_samples]
        if verbose:
            print(f"  Limited to {max_samples:,} samples")

    # Prepend BOS token if requested (to match training behavior)
    # Training uses use_bos=True by default, so evaluation should too
    if use_bos:
        pad_token_id = get_pad_token_id(data_dir)
        val_sequences = [[pad_token_id] + seq for seq in val_sequences]
        if verbose:
            print(f"  BOS token {pad_token_id} prepended to all sequences")

    return val_sequences


def get_pad_token_id(data_dir: Path) -> int:
    """Get pad token ID from metadata (vocab_size - 1).

    The pad token is always the last token in the vocabulary.
    This is read from metadata.json which stores base_vocab_size.
    """
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        # base_vocab_size is the actual vocab, pad token is the last one
        base_vocab = metadata.get("base_vocab_size", 256)
        return base_vocab  # This is vocab_size - 1 (0-indexed)
    # Fallback for old format
    return 256


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
        pad_token_id: int,
        max_len: Optional[int] = None,
        verbose: bool = True,
        use_bos: bool = True,
    ):
        """
        Initialize dataset from pre-encoded JSONL file.

        Args:
            jsonl_file: Path to JSONL file (train.jsonl or validate.jsonl)
            context_size: Maximum sequence length (truncate if longer, pad if shorter)
            pad_token_id: Token ID used for padding (should be vocab_size - 1)
            max_len: Filter out sequences longer than this (None = no filtering)
            verbose: Print loading statistics
            use_bos: If True (default), prepend pad_token_id as BOS to enable unconditional generation
        """
        self.context_size = context_size
        self.pad_token_id = pad_token_id
        self.max_len = max_len
        self.verbose = verbose
        self.use_bos = use_bos

        # Load all sequences into memory
        self.sequences = self._load_jsonl(jsonl_file)

        if verbose:
            self._print_stats()
            if use_bos:
                print(f"  BOS token: {pad_token_id} (prepended to all sequences)")

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
            inputs: [context_size] tensor, padded with pad_token_id
            targets: [context_size] tensor, padded with -1 (ignored in loss)

        When use_bos=True:
            - BOS (pad_token_id) is prepended to sequence
            - inputs[0] = BOS, targets[0] = first real token
            - Model learns P(first_token | BOS) for unconditional generation
            - All PAD tokens in targets are masked (set to -1)
        """
        tokens = self.sequences[idx]

        # Prepend BOS token if enabled
        if self.use_bos:
            tokens = torch.cat([torch.tensor([self.pad_token_id], dtype=torch.long), tokens])

        # Truncate if too long (need context_size + 1 for teacher forcing)
        if len(tokens) > self.context_size + 1:
            tokens = tokens[:self.context_size + 1]

        # Pad if too short
        if len(tokens) < self.context_size + 1:
            needed = self.context_size + 1 - len(tokens)
            tokens = torch.cat([tokens, torch.full((needed,), self.pad_token_id, dtype=torch.long)])

        # Teacher forcing: shift by 1
        inputs = tokens[:-1].clone()   # First context_size tokens
        targets = tokens[1:].clone()   # Last context_size tokens

        # Mask all PAD tokens in targets (ignored in loss via ignore_index=-1)
        # PAD tokens should never appear in real tokenized data, only as padding
        # This correctly handles the shift between inputs and targets
        targets[targets == self.pad_token_id] = -1

        # Convert inputs to int32 (model expects this)
        inputs = inputs.to(dtype=torch.int32)

        return inputs, targets


def create_reshuffled_dataloaders(
    data_dir: Path,
    context_size: int,
    batch_size: int,
    train_ratio: float = 0.95,
    max_len: Optional[int] = None,
    partner_data_dir: Optional[Path] = None,
    device: str = "cuda",
    num_workers: int = 0,
    verbose: bool = True,
    seed: int = 42,
    use_bos: bool = True,
) -> tuple:
    """
    Load all data, shuffle randomly, and create train/val dataloaders.

    This fixes the sequential split issue by reshuffling the entire dataset.

    Args:
        data_dir: Directory containing train.jsonl and validate.jsonl
        context_size: Sequence length
        batch_size: Batch size
        train_ratio: Fraction for training (default 0.95)
        max_len: Filter sequences longer than this
        partner_data_dir: Partner data directory for coordinated filtering.
            When provided, sequences are excluded if they exceed max_len in
            EITHER this directory or partner_data_dir. This ensures TCT and
            UTF8 models train on identical samples.
        device: Device for tensors
        num_workers: DataLoader workers
        verbose: Print statistics
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader)
    """
    import random
    data_dir = Path(data_dir)

    # Get the correct pad token from metadata
    pad_token_id = get_pad_token_id(data_dir)
    if verbose:
        print(f"  Pad token: {pad_token_id}")

    # Load all sequences - prefer all.jsonl if available, otherwise combine train+val
    # NOTE: We load ALL sequences first, then apply filtering (for coordinated filtering)
    all_sequences = []
    all_jsonl = data_dir / "all.jsonl"

    if all_jsonl.exists():
        # Use all.jsonl directly
        with open(all_jsonl, 'r') as f:
            for line in f:
                tokens = json.loads(line)
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
                        all_sequences.append(torch.tensor(tokens, dtype=torch.long))
        source = "train.jsonl + validate.jsonl"

    if verbose:
        print(f"Loaded {len(all_sequences):,} total sequences from {data_dir.name} ({source})")

    # Apply filtering (coordinated or standard)
    if max_len is not None:
        original_count = len(all_sequences)

        if partner_data_dir is not None:
            # Coordinated filtering: exclude if exceeds max_len in EITHER tokenization
            partner_data_dir = Path(partner_data_dir)
            if not partner_data_dir.exists():
                if verbose:
                    print(f"  WARNING: Partner dir not found: {partner_data_dir}")
                    print(f"  Falling back to standard filtering")
                all_sequences = [s for s in all_sequences if len(s) <= max_len]
            else:
                if verbose:
                    print(f"  Coordinated filtering with partner: {partner_data_dir.name}")

                partner_lengths = _load_sequence_lengths(partner_data_dir)

                if len(partner_lengths) != len(all_sequences):
                    if verbose:
                        print(f"  WARNING: Length mismatch (own={len(all_sequences)}, partner={len(partner_lengths)})")
                        print(f"  Falling back to standard filtering")
                    all_sequences = [s for s in all_sequences if len(s) <= max_len]
                else:
                    # Filter to indices valid in BOTH tokenizations
                    valid_indices = [
                        i for i in range(len(all_sequences))
                        if len(all_sequences[i]) <= max_len and partner_lengths[i] <= max_len
                    ]
                    all_sequences = [all_sequences[i] for i in valid_indices]

                    if verbose:
                        excluded = original_count - len(all_sequences)
                        print(f"  Coordinated filtering: kept {len(all_sequences):,}, excluded {excluded:,} (max_len={max_len})")
        else:
            # Standard filtering (no partner)
            all_sequences = [s for s in all_sequences if len(s) <= max_len]
            if verbose:
                excluded = original_count - len(all_sequences)
                print(f"  Standard filtering: kept {len(all_sequences):,}, excluded {excluded:,} (max_len={max_len})")

    # Check for empty dataset
    if len(all_sequences) == 0:
        raise ValueError(
            f"No sequences loaded from {data_dir}. "
            f"Expected 'all.jsonl' or 'train.jsonl'+'validate.jsonl' in {data_dir}. "
            f"Files found: {list(data_dir.glob('*.jsonl'))}"
        )

    # Shuffle indices with fixed seed for reproducibility
    # This ensures TCT and UTF8 get the same train/val split (same document indices)
    # IMPORTANT: Requires all.jsonl to have same documents in same order for both tokenizers
    n = len(all_sequences)
    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    # Split by shuffled indices (maintains random order within train/val)
    split_idx = int(n * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_sequences = [all_sequences[i] for i in train_indices]
    val_sequences = [all_sequences[i] for i in val_indices]

    if verbose:
        print(f"  Reshuffled split: {len(train_sequences):,} train, {len(val_sequences):,} val (seed={seed})")
        if use_bos:
            print(f"  BOS token: {pad_token_id} (prepended to all sequences)")

    # Create datasets directly from sequences
    class InMemoryDataset(Dataset):
        def __init__(self, sequences, context_size, pad_id, use_bos=False):
            self.sequences = sequences
            self.context_size = context_size
            self.pad_token_id = pad_id
            self.use_bos = use_bos

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            tokens = self.sequences[idx]

            # Prepend BOS token if enabled
            if self.use_bos:
                tokens = torch.cat([torch.tensor([self.pad_token_id], dtype=torch.long), tokens])

            # Truncate if too long
            if len(tokens) > self.context_size + 1:
                tokens = tokens[:self.context_size + 1]

            # Pad if too short
            if len(tokens) < self.context_size + 1:
                needed = self.context_size + 1 - len(tokens)
                tokens = torch.cat([tokens, torch.full((needed,), self.pad_token_id, dtype=torch.long)])

            inputs = tokens[:-1].clone()
            targets = tokens[1:].clone()

            # Mask all PAD tokens in targets (ignored in loss via ignore_index=-1)
            # PAD tokens should never appear in real tokenized data, only as padding
            targets[targets == self.pad_token_id] = -1

            inputs = inputs.to(dtype=torch.int32)
            return inputs, targets

    def collate_fn(batch):
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.int64, non_blocking=True)
        return inputs, targets

    train_loader = DataLoader(
        InMemoryDataset(train_sequences, context_size, pad_token_id, use_bos=use_bos),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        InMemoryDataset(val_sequences, context_size, pad_token_id, use_bos=use_bos),
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
    use_bos: bool = True,
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
        use_bos: If True (default), prepend pad_token_id as BOS to enable unconditional generation

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

    # Get the correct pad token from metadata
    pad_token_id = get_pad_token_id(data_dir)

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
        print(f"  Pad token: {pad_token_id}")

    # Create dataset
    dataset = JSONLDataset(
        jsonl_file,
        context_size=context_size,
        pad_token_id=pad_token_id,
        max_len=max_len,
        verbose=verbose,
        use_bos=use_bos,
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
