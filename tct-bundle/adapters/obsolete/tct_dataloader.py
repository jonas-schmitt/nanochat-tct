"""
TCT DataLoader for Workflow Training

Loads pre-prepared TCT windows and creates PyTorch DataLoader
compatible with nanochat's training loop.

Usage:
    from tct_dataloader import create_dataloader

    train_loader = create_dataloader(
        "data/prepared/train.pt",
        batch_size=32,
        shuffle=True
    )

    for x, y in train_loader:
        # x: [batch, context_size] - input tokens
        # y: [batch, context_size] - target tokens (shifted by 1)
        ...
"""

import torch
from torch.utils.data import Dataset, DataLoader


class TCTWindowDataset(Dataset):
    """
    Dataset for TCT windowed training data.

    Loads pre-prepared windows from prepare_training_data.py and
    creates (input, target) pairs for autoregressive training.
    """

    def __init__(self, data_path):
        """
        Initialize dataset.

        Args:
            data_path: Path to prepared windows tensor (e.g., train.pt)
        """
        self.windows = torch.load(data_path)

        # Verify shape: [num_windows, context_size+1]
        if len(self.windows.shape) != 2:
            raise ValueError(
                f"Expected 2D tensor [num_windows, context_size+1], "
                f"got shape {self.windows.shape}"
            )

        self.num_windows = self.windows.shape[0]
        self.window_size = self.windows.shape[1]

        print(f"Loaded {self.num_windows} windows (window size: {self.window_size})")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        """
        Get training example.

        Returns:
            (x, y) tuple:
            - x: Input tokens [context_size] (position + context, minus last token)
            - y: Target tokens [context_size] (context, minus position token)
        """
        window = self.windows[idx]

        # Window format: [position_token, tok_0, tok_1, ..., tok_N]
        # Input: [position_token, tok_0, ..., tok_N-1]
        # Target: [tok_0, tok_1, ..., tok_N]

        x = window[:-1]  # Remove last token
        y = window[1:]   # Remove position token

        return x, y


def create_dataloader(
    data_path,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
):
    """
    Create PyTorch DataLoader for TCT windows.

    Args:
        data_path: Path to prepared windows (train.pt or val.pt)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch

    Returns:
        DataLoader yielding (x, y) batches
    """
    dataset = TCTWindowDataset(data_path)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return loader


def create_train_val_loaders(
    train_path="data/prepared/train.pt",
    val_path="data/prepared/val.pt",
    batch_size=32,
    num_workers=4,
):
    """
    Create training and validation dataloaders.

    Args:
        train_path: Path to training windows
        val_path: Path to validation windows
        batch_size: Batch size for both loaders
        num_workers: Number of data loading workers

    Returns:
        (train_loader, val_loader) tuple
    """
    train_loader = create_dataloader(
        train_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = create_dataloader(
        val_path,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
    )

    return train_loader, val_loader


# =============================================================================
# Nanochat Compatibility Layer
# =============================================================================

def tokenizing_distributed_data_loader(
    device_batch_size,
    context_size,
    split="train",
    data_dir="data/prepared",
    device=None,
    num_workers=4,
):
    """
    Nanochat-compatible data loader interface.

    This function matches nanochat's data loader signature but loads
    pre-prepared TCT windows instead of tokenizing on-the-fly.

    Args:
        device_batch_size: Batch size per device
        context_size: Context window size (should match preparation)
        split: "train" or "val"
        data_dir: Directory containing prepared windows
        device: Target device (for compatibility, not used)
        num_workers: Number of data loading workers

    Yields:
        (x, y) batches compatible with nanochat training loop
    """
    import os

    data_path = os.path.join(data_dir, f"{split}.pt")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Prepared data not found: {data_path}\n"
            f"Run: python scripts/prepare_training_data.py "
            f"--input data/json/ --output {data_dir}/"
        )

    loader = create_dataloader(
        data_path,
        batch_size=device_batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
    )

    for x, y in loader:
        # Move to device if specified
        if device is not None:
            x = x.to(device)
            y = y.to(device)

        yield x, y


# =============================================================================
# Usage Example
# =============================================================================

if __name__ == "__main__":
    import os

    # Check if prepared data exists
    if not os.path.exists("data/prepared/train.pt"):
        print("⚠️  Prepared data not found!")
        print("Run: python scripts/prepare_training_data.py")
        exit(1)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_train_val_loaders(batch_size=4)

    # Test training loader
    print("\nTesting training loader:")
    for i, (x, y) in enumerate(train_loader):
        print(f"Batch {i}:")
        print(f"  x shape: {x.shape}  (input)")
        print(f"  y shape: {y.shape}  (target)")
        print(f"  x[0, :10]: {x[0, :10]}")
        print(f"  y[0, :10]: {y[0, :10]}")

        # Verify target is shifted input
        assert torch.equal(x[0, 1:], y[0, :-1]), "Target should be shifted input!"

        if i >= 2:  # Test only first 3 batches
            break

    print("\n✅ DataLoader test successful!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
