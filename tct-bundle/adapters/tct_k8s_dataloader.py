"""
Kubernetes Manifest DataLoader for TCT

Each manifest is a separate, independent training example with no cross-manifest context.
Manifests are padded/truncated to context_size, ensuring true manifest isolation.

Key Features:
- Each batch contains B independent manifests (no concatenation)
- Each manifest starts at position 0 (no prior manifest context)
- Inputs padded with 19999 (TCT_K8S_PAD_TOKEN_ID)
- Targets padded with -1 (ignored in loss via ignore_index)
- Compatible with nanochat's training loop
- Reuses existing tokenization caching
- Handles 10-split dataset structure (combines all splits)
"""

import sys
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Import TCT kubernetes tokenizer
import tct_kubernetes as tct

# Import TCT kubernetes constants
from tct_k8s_constants import TCT_K8S_PAD_TOKEN_ID, TCT_K8S_TOTAL_VOCAB_SIZE


# Module-level function for parallel tokenization (must be picklable)
def _tokenize_single_manifest(manifest_file):
    """Tokenize a single manifest file (worker function for multiprocessing)."""
    try:
        with open(manifest_file, 'r') as f:
            manifest_json = f.read()
        tokens = tct.encode(manifest_json)
        return (True, torch.tensor(tokens, dtype=torch.long))
    except Exception as e:
        return (False, str(e))


class TCTKubernetesDataset(Dataset):
    """
    Dataset where each Kubernetes manifest is an independent training example.

    No streaming, no concatenation - pure manifest isolation.
    """

    def __init__(self, manifest_files, context_size, split="train", train_split=0.9,
                 cache_file=None):
        """
        Initialize Kubernetes manifest dataset.

        Args:
            manifest_files: List of paths to JSON manifest files
            context_size: Maximum sequence length (truncate if longer, pad if shorter)
            split: "train" or "val"
            train_split: Train/val split ratio (default: 0.9)
            cache_file: Optional cache file path (None = auto-detect, "none" = disable)
        """
        self.manifest_files = list(manifest_files)
        self.context_size = context_size
        self.split = split

        # Determine cache file to use
        actual_cache_file = self._get_cache_file(cache_file, train_split)

        # Load or tokenize manifests
        if actual_cache_file and actual_cache_file.exists():
            print(f"Loading tokenized manifests from cache: {actual_cache_file}")
            tokenized_manifests = torch.load(actual_cache_file)
            print(f"✅ Loaded {len(tokenized_manifests)} tokenized manifests")
        else:
            print(f"Tokenizing {len(self.manifest_files)} manifests...")
            tokenized_manifests = self._tokenize_manifests()

            # Save cache if path provided
            if actual_cache_file:
                actual_cache_file.parent.mkdir(parents=True, exist_ok=True)
                torch.save(tokenized_manifests, actual_cache_file)
                print(f"✅ Saved cache: {actual_cache_file}")

        # Filter out problematic sequences
        print(f"\nFiltering sequences (max_len=8192, max_ascii_ratio=0.95)...")
        tokenized_manifests = self._filter_sequences(tokenized_manifests, max_len=8192, max_ascii_ratio=0.95)

        # Split train/val
        split_idx = int(len(tokenized_manifests) * train_split)
        if split == "train":
            self.manifests = tokenized_manifests[:split_idx]
        else:
            self.manifests = tokenized_manifests[split_idx:]

        print(f"\n{split.upper()} split: {len(self.manifests):,} manifests")

        # Statistics
        manifest_lengths = [len(m) for m in self.manifests]
        avg_len = sum(manifest_lengths) / len(manifest_lengths)
        truncated = sum(1 for l in manifest_lengths if l > context_size)
        padded = sum(1 for l in manifest_lengths if l < context_size)

        print(f"Average manifest length: {avg_len:.0f} tokens")
        print(f"Manifests truncated (>{context_size}): {truncated:,} ({100*truncated/len(self.manifests):.1f}%)")
        print(f"Manifests padded (<{context_size}): {padded:,} ({100*padded/len(self.manifests):.1f}%)")
        print()

    def _get_cache_file(self, cache_file, train_split):
        """Determine cache file path (matches other dataloaders for compatibility)."""
        if cache_file == "none":
            return None
        if cache_file:
            return Path(cache_file)

        # Auto-detect from manifest files
        if self.manifest_files:
            # Use parent of the split directory
            manifest_dir = Path(self.manifest_files[0]).parent.parent
            cache_dir = manifest_dir / ".cache"
            split_pct = int(train_split * 100)
            num_files = len(self.manifest_files)
            # Use split-agnostic cache (train and val share same tokenized data)
            auto_cache_file = cache_dir / f"tokenized_k8s_split{split_pct}_{num_files}files.pt"

            # Check for exact match first
            if auto_cache_file.exists():
                return auto_cache_file

            # Fallback: try to find any cache file (ignore file count)
            import glob
            cache_pattern = str(cache_dir / f"tokenized_k8s_split{split_pct}_*.pt")
            existing_caches = sorted(glob.glob(cache_pattern))
            if existing_caches:
                # Use the most recent cache
                fallback_cache = Path(existing_caches[-1])
                print(f"Warning: Exact cache not found, using latest: {fallback_cache.name}")
                return fallback_cache

            # No cache found, return path for new cache
            return auto_cache_file

        return None

    def _tokenize_manifests(self):
        """Tokenize all manifests using TCT kubernetes tokenizer (parallel)."""
        import multiprocessing as mp

        # Use 80% of CPU cores for tokenization (leave some for system)
        num_workers = max(1, int(mp.cpu_count() * 0.8))
        print(f"  Tokenizing with {num_workers} workers...")

        # Tokenize in parallel with progress reporting
        tokenized = []
        failed = 0
        failed_messages = []

        with mp.Pool(num_workers) as pool:
            results = pool.imap(_tokenize_single_manifest, self.manifest_files, chunksize=100)

            for i, result in enumerate(results, 1):
                success, data = result

                if success:
                    tokenized.append(data)
                else:
                    failed += 1
                    if failed <= 10:
                        failed_messages.append(f"    {self.manifest_files[i-1]}: {data}")

                # Progress reporting
                if i % 10000 == 0:
                    print(f"  Progress: {i:,}/{len(self.manifest_files):,} manifests")

        # Print failures
        if failed > 0:
            print(f"  ⚠️  Failed to tokenize {failed}/{len(self.manifest_files)} manifests")
            if failed_messages:
                print("  First failures:")
                for msg in failed_messages:
                    print(msg)

        print(f"  ✅ Successfully tokenized {len(tokenized)} manifests")
        return tokenized

    def _filter_sequences(self, tokenized_manifests, max_len=8192, max_ascii_ratio=0.95):
        """
        Filter out problematic sequences that hurt training:
        - Too long (>max_len tokens) - beyond practical context sizes
        - Too ASCII-heavy (>max_ascii_ratio) - model learns byte prediction, not structure

        Args:
            tokenized_manifests: List of tokenized manifest tensors
            max_len: Maximum sequence length (default: 8192)
            max_ascii_ratio: Maximum ratio of ASCII tokens (default: 0.95)

        Returns:
            Filtered list of manifests
        """
        filtered = []
        stats = {
            'too_long': 0,
            'too_ascii': 0,
            'both': 0,
            'kept': 0
        }

        for m in tokenized_manifests:
            length = len(m)

            # Count ASCII tokens (0-127)
            ascii_count = (m < 128).sum().item()
            ascii_ratio = ascii_count / length if length > 0 else 0

            # Check filters
            too_long = length > max_len
            too_ascii = ascii_ratio > max_ascii_ratio

            if too_long and too_ascii:
                stats['both'] += 1
            elif too_long:
                stats['too_long'] += 1
            elif too_ascii:
                stats['too_ascii'] += 1
            else:
                filtered.append(m)
                stats['kept'] += 1

        total = len(tokenized_manifests)
        removed = total - stats['kept']

        print(f"  Filtering results:")
        print(f"    Total sequences: {total:,}")
        print(f"    Kept: {stats['kept']:,} ({100*stats['kept']/total:.1f}%)")
        print(f"    Removed: {removed:,} ({100*removed/total:.1f}%)")
        print(f"      - Too long (>{max_len}): {stats['too_long']:,}")
        print(f"      - Too ASCII (>{max_ascii_ratio:.0%}): {stats['too_ascii']:,}")
        print(f"      - Both: {stats['both']:,}")
        print()

        return filtered

    def __len__(self):
        return len(self.manifests)

    def __getitem__(self, idx):
        """
        Get a single manifest as independent training example.

        Returns:
            inputs: [context_size] tensor, padded with 19999
            targets: [context_size] tensor, padded with -1
        """
        # Get manifest tokens
        tokens = self.manifests[idx]

        # Truncate if too long (take first context_size tokens)
        if len(tokens) > self.context_size:
            tokens = tokens[:self.context_size]

        # For teacher forcing, we need context_size + 1 total tokens
        # (to create context_size input/target pairs)
        if len(tokens) < self.context_size + 1:
            # Pad to context_size + 1
            # Inputs will use 19999, targets will use -1
            needed = self.context_size + 1 - len(tokens)
            # For now, pad with 19999 (we'll handle targets separately)
            tokens = torch.cat([tokens, torch.full((needed,), TCT_K8S_PAD_TOKEN_ID, dtype=torch.long)])

        # Teacher forcing: shift by 1
        inputs = tokens[:-1].clone()  # First context_size tokens
        targets = tokens[1:].clone()  # Last context_size tokens

        # Replace padding in targets with -1 (ignored in loss)
        # Padding occurs where inputs are 19999
        mask = inputs == TCT_K8S_PAD_TOKEN_ID
        targets[mask] = -1

        # Convert to int32 for inputs (model expects this)
        inputs = inputs.to(dtype=torch.int32)

        return inputs, targets


def tct_k8s_data_loader(B, T, split="train", data_dir=None, train_split=0.9,
                        device="cuda", cache_file=None, shuffle=True, num_workers=0):
    """
    Create DataLoader for independent Kubernetes manifest training.

    Args:
        B: Batch size
        T: Sequence length (context size)
        split: "train" or "val"
        data_dir: Directory containing k8s-split-XX subdirectories with JSON files
        train_split: Train/val split ratio
        device: Device for tensors ("cuda" or "cpu")
        cache_file: Optional cache file path
        shuffle: Shuffle manifests each epoch (default: True for train)
        num_workers: DataLoader workers (default: 0, single-threaded)

    Returns:
        DataLoader yielding (x, y) batches of shape [B, T]
    """
    if data_dir is None:
        data_dir = Path.home() / "Desktop/data"
    else:
        data_dir = Path(data_dir)

    # Load manifest files from all k8s-split-XX subdirectories
    manifest_files = []
    split_dirs = sorted(data_dir.glob("k8s-split-*"))

    if not split_dirs:
        raise ValueError(f"No k8s-split-* directories found in {data_dir}")

    print(f"Found {len(split_dirs)} split directories in {data_dir}")
    for split_dir in split_dirs:
        split_files = sorted(split_dir.glob("*.json"))
        manifest_files.extend(split_files)
        print(f"  {split_dir.name}: {len(split_files):,} files")

    print(f"Total: {len(manifest_files):,} manifest files")

    # Create dataset
    dataset = TCTKubernetesDataset(
        manifest_files,
        context_size=T,
        split=split,
        train_split=train_split,
        cache_file=cache_file
    )

    # Auto-determine shuffle
    if shuffle is True:
        shuffle = (split == "train")

    # Create DataLoader
    # Note: pin_memory=True for CUDA, collate_fn moves to device
    def collate_fn(batch):
        """Collate batch and move to device."""
        inputs = torch.stack([item[0] for item in batch])
        targets = torch.stack([item[1] for item in batch])

        # Move to device
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, dtype=torch.int64, non_blocking=True)

        return inputs, targets

    loader = DataLoader(
        dataset,
        batch_size=B,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=collate_fn,
    )

    return loader
