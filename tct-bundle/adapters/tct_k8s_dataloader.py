"""
Kubernetes Manifest DataLoader for TCT

Each manifest is a separate, independent training example with no cross-manifest context.
Manifests are padded/truncated to context_size, ensuring true manifest isolation.

Key Features:
- Each batch contains B independent manifests (no concatenation)
- Each manifest starts at position 0 (no prior manifest context)
- Inputs padded with PAD_TOKEN_ID
- Targets padded with -1 (ignored in loss via ignore_index)
- Compatible with nanochat's training loop
- Supports pre-encoded JSONL format (k8s-encoded/) with train/validate splits
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# Import TCT kubernetes constants
from tct_k8s_constants import TCT_K8S_PAD_TOKEN_ID, TCT_K8S_TOTAL_VOCAB_SIZE


class TCTKubernetesDataset(Dataset):
    """
    Dataset where each Kubernetes manifest is an independent training example.

    Loads pre-encoded JSONL files from k8s-encoded/ directory.
    No streaming, no concatenation - pure manifest isolation.
    """

    def __init__(self, jsonl_file, context_size, max_len=None):
        """
        Initialize Kubernetes manifest dataset from pre-encoded JSONL.

        Args:
            jsonl_file: Path to JSONL file (train.jsonl or validate.jsonl)
            context_size: Maximum sequence length (truncate if longer, pad if shorter)
            max_len: Maximum sequence length to keep (filter longer ones). None = context_size * 2
        """
        self.context_size = context_size
        self.max_len = max_len or (context_size * 2)

        # Load pre-encoded sequences from JSONL
        print(f"Loading pre-encoded sequences from: {jsonl_file}")
        self.manifests = self._load_jsonl(jsonl_file)

        # Statistics
        manifest_lengths = [len(m) for m in self.manifests]
        avg_len = sum(manifest_lengths) / len(manifest_lengths)
        truncated = sum(1 for l in manifest_lengths if l > context_size)
        padded = sum(1 for l in manifest_lengths if l < context_size)

        print(f"Loaded {len(self.manifests):,} sequences")
        print(f"Average length: {avg_len:.0f} tokens")
        print(f"Will truncate (>{context_size}): {truncated:,} ({100*truncated/len(self.manifests):.1f}%)")
        print(f"Will pad (<{context_size}): {padded:,} ({100*padded/len(self.manifests):.1f}%)")
        print()

    def _load_jsonl(self, jsonl_file):
        """Load sequences from JSONL file, filtering by max_len."""
        manifests = []
        skipped = 0

        with open(jsonl_file, 'r') as f:
            for i, line in enumerate(f):
                tokens = json.loads(line)
                if len(tokens) <= self.max_len:
                    manifests.append(torch.tensor(tokens, dtype=torch.long))
                else:
                    skipped += 1

                # Progress
                if (i + 1) % 50000 == 0:
                    print(f"  Loaded {i + 1:,} lines...")

        if skipped > 0:
            print(f"  Skipped {skipped:,} sequences longer than {self.max_len} tokens")

        return manifests

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


def tct_k8s_data_loader(B, T, split="train", data_dir=None, max_len=None,
                        device="cuda", shuffle=True, num_workers=0):
    """
    Create DataLoader for independent Kubernetes manifest training.

    Args:
        B: Batch size
        T: Sequence length (context size)
        split: "train" or "val"
        data_dir: Directory containing k8s-encoded/ with train.jsonl and validate.jsonl
        max_len: Maximum sequence length to keep (default: T * 2)
        device: Device for tensors ("cuda" or "cpu")
        shuffle: Shuffle manifests each epoch (default: True for train)
        num_workers: DataLoader workers (default: 0, single-threaded)

    Returns:
        DataLoader yielding (x, y) batches of shape [B, T]
    """
    if data_dir is None:
        data_dir = Path.home() / "Desktop/data"
    else:
        data_dir = Path(data_dir)

    # Find the k8s-encoded directory
    encoded_dir = data_dir / "k8s-encoded"
    if not encoded_dir.exists():
        raise ValueError(f"k8s-encoded directory not found in {data_dir}")

    # Select the right JSONL file based on split
    if split == "train":
        jsonl_file = encoded_dir / "train.jsonl"
    else:
        jsonl_file = encoded_dir / "validate.jsonl"

    if not jsonl_file.exists():
        raise ValueError(f"JSONL file not found: {jsonl_file}")

    # Load metadata if available
    metadata_file = encoded_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"Dataset: {metadata.get('encoded_files', '?')} encoded manifests")
        print(f"Split: {split} ({metadata.get('train_count' if split == 'train' else 'validate_count', '?')} sequences)")

    # Create dataset
    dataset = TCTKubernetesDataset(
        jsonl_file,
        context_size=T,
        max_len=max_len,
    )

    # Auto-determine shuffle
    if shuffle is True:
        shuffle = (split == "train")

    # Create DataLoader with collate function
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
