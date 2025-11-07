"""
Training Utilities for TCT and Nanochat

Common utilities extracted from training scripts to reduce code duplication:
- Learning rate scheduler (warmup + cosine decay)
- Model initialization (device placement, compilation)
- Checkpoint resume (model + optimizer state loading)

This module reduces ~246 LOC of duplication across training scripts.
"""

import re
import sys
import torch
from pathlib import Path
from typing import Optional, Tuple

# Import nanochat models
from nanochat.gpt import GPT, GPTConfig


def get_lr_multiplier(
    step: int,
    warmup_iters: int,
    max_iters: int,
    final_lr_frac: float = 0.1
) -> float:
    """
    Compute learning rate multiplier with linear warmup and cosine decay.

    Schedule:
        - [0, warmup_iters): Linear warmup from 0 to 1.0
        - [warmup_iters, max_iters]: Cosine decay from 1.0 to final_lr_frac

    Args:
        step: Current training step (0-indexed)
        warmup_iters: Number of warmup iterations
        max_iters: Total training iterations
        final_lr_frac: Final LR as fraction of initial (default: 0.1 = 10%)

    Returns:
        Learning rate multiplier in [final_lr_frac, 1.0]

    Example:
        >>> # Warmup for 1000 steps, train for 10000 steps, decay to 10% LR
        >>> lr_mult = get_lr_multiplier(step=500, warmup_iters=1000, max_iters=10000)
        >>> actual_lr = base_lr * lr_mult
    """
    if step < warmup_iters:
        # Linear warmup: 0 → 1.0
        return (step + 1) / warmup_iters
    else:
        # Cosine decay: 1.0 → final_lr_frac
        progress = (step - warmup_iters) / (max_iters - warmup_iters)
        cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        return final_lr_frac + (1.0 - final_lr_frac) * cosine


def create_gpt_model(
    vocab_size: int,
    context_size: int,
    n_layers: int,
    n_heads: int,
    d_model: int,
    device: str,
    compile: bool = True
) -> Tuple[GPT, GPT]:
    """
    Initialize GPT model with proper device placement and optional compilation.

    This function:
    1. Creates model on meta device (no memory allocation)
    2. Moves to target device and initializes weights
    3. Optionally applies torch.compile for speedup
    4. Returns both compiled and original models

    Args:
        vocab_size: Vocabulary size
        context_size: Maximum sequence length
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_model: Model embedding dimension
        device: Target device ('cuda', 'cpu', 'mps')
        compile: Whether to apply torch.compile (default: True)
                 Set to False when resuming to avoid OOM

    Returns:
        Tuple of (compiled_model, original_model):
            - compiled_model: For training/inference (may be same as original if compile=False)
            - original_model: For checkpointing (uncompiled, always saveable)

    Example:
        >>> model, orig_model = create_gpt_model(
        ...     vocab_size=8192,
        ...     context_size=512,
        ...     n_layers=10,
        ...     n_heads=12,
        ...     d_model=768,
        ...     device="cuda",
        ...     compile=True
        ... )
        >>> # Use `model` for training, `orig_model` for saving checkpoints
    """
    # Model configuration
    model_config_kwargs = dict(
        sequence_len=context_size,
        vocab_size=vocab_size,
        n_layer=n_layers,
        n_head=n_heads,
        n_kv_head=n_heads,  # 1:1 GQA ratio (standard attention)
        n_embd=d_model,
    )

    # Create on meta device first (no memory allocation)
    with torch.device("meta"):
        model_config = GPTConfig(**model_config_kwargs)
        model = GPT(model_config)

    # Move to target device and initialize weights
    model.to_empty(device=device)
    model.init_weights()

    # Keep reference to uncompiled model for checkpointing
    orig_model = model

    # Apply torch.compile if requested
    if compile:
        model = torch.compile(model, dynamic=False)

    return model, orig_model


def resume_from_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
    verbose: bool = True
) -> int:
    """
    Resume training from checkpoint.

    Loads model weights and optionally optimizer state from checkpoint files.
    Model checkpoint: model_XXXXX.pt
    Optimizer checkpoint: optim_XXXXX.pt (same directory, optional)

    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        model: Model to load weights into (should be uncompiled)
        optimizer: Optional optimizer to load state into
        device: Device for checkpoint loading (default: "cuda")
        verbose: Print loading progress (default: True)

    Returns:
        start_step: Training step to resume from (extracted from filename)

    Raises:
        SystemExit: If checkpoint file doesn't exist

    Example:
        >>> model, orig_model = create_gpt_model(...)
        >>> optimizer = torch.optim.AdamW(model.parameters())
        >>> start_step = resume_from_checkpoint(
        ...     checkpoint_path="~/Desktop/checkpoints/model_05000.pt",
        ...     model=orig_model,  # Use UNCOMPILED model
        ...     optimizer=optimizer,
        ...     device="cuda"
        ... )
        >>> # Continue training from start_step
    """
    resume_path = Path(checkpoint_path).expanduser()

    # Check if checkpoint exists
    if not resume_path.exists():
        if verbose:
            print(f"Error: Checkpoint not found: {resume_path}")
        sys.exit(1)

    if verbose:
        print(f"Resuming from checkpoint: {resume_path}")

    # Load model weights
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Load optimizer state if provided
    if optimizer is not None:
        optim_path = resume_path.parent / resume_path.name.replace("model_", "optim_")
        if optim_path.exists():
            if verbose:
                print(f"Loading optimizer state from: {optim_path}")
            try:
                optim_checkpoint = torch.load(optim_path, map_location=device)
                optimizer.load_state_dict(optim_checkpoint)
                if verbose:
                    print("Optimizer state loaded successfully")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not load optimizer state: {e}")
                    print("Continuing with fresh optimizer state")

    # Extract step number from checkpoint name (e.g., model_010000.pt -> 10000)
    match = re.search(r'_(\d+)\.pt$', resume_path.name)
    start_step = int(match.group(1)) if match else 0

    if verbose:
        print(f"Resuming from step: {start_step}")
        print()

    return start_step


# Export all utilities
__all__ = [
    "get_lr_multiplier",
    "create_gpt_model",
    "resume_from_checkpoint",
]
