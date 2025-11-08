"""
Model Configuration for Workflow Generation with TCT

Six preset configurations optimized for GitHub Actions workflow generation:

**Default (context=512, faster, recommended):**
- Small:  20M params, ~2h training, $10 budget
- Medium: 90M params, ~8h training, $45 budget  ⭐ RECOMMENDED
- Large:  183M params, ~12h training, $75 budget

**Alternative (context=1024, better coverage, slower):**
- Small-1024:  20M params, ~3h training, $15 budget
- Medium-1024: 90M params, ~12h training, $60 budget
- Large-1024:  183M params, ~18h training, $90 budget

**Context size tradeoffs:**
- 512:  Covers 30% of workflows entirely, 2× faster than 1024
- 1024: Covers 56% of workflows entirely, better for long workflows

**Recommendation:** Start with Medium (512) for production, use Medium-1024 if
workflows are frequently >512 tokens.
"""

# =============================================================================
# Small Configuration (20M params, context=512) - Quick baseline
# =============================================================================
SMALL_CONFIG = {
    # Model architecture
    "vocab_size": 8192,        # TCT vocabulary (8190 base + MASK + PAD)
    "context_size": 512,       # Context window
    "d_model": 384,            # Embedding dimension
    "n_layers": 8,             # Transformer layers
    "n_heads": 6,              # Attention heads (head_dim = 64)
    "dropout": 0.2,            # Strong regularization

    # Training
    "batch_size": 32,
    "gradient_accumulation": 4,  # Effective batch = 128
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "warmup_iters": 1000,
    "max_iters": 20000,        # Quick baseline

    # Optimization
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # Logging
    "eval_interval": 500,
    "log_interval": 10,
    "save_interval": 5000,

    # Estimated
    "parameters": "~20M",
    "training_time": "~2 hours on RTX 4090",
    "budget": "~$10",
}

# =============================================================================
# Small-1024 Configuration (20M params, context=1024) - Alternative baseline
# =============================================================================
SMALL_1024_CONFIG = {
    # Model architecture (same as Small, but 1024 context)
    "vocab_size": 8192,
    "context_size": 1024,      # 2× context vs small
    "d_model": 384,
    "n_layers": 8,
    "n_heads": 6,
    "dropout": 0.1,            # Light regularization (we only see ~3% of 110M examples in 100k steps)

    # Training (optimized for 100k steps, excellent generalization)
    "batch_size": 32,          # Doubled from 16 (fits in 8GB, more stable gradients)
    "gradient_accumulation": 4,  # Reduced to keep effective batch = 128
    "learning_rate": 3e-4,     # Increased for batch=32 and undertrained regime (only 2.9% data coverage)
    "weight_decay": 0.1,
    "warmup_iters": 5000,      # 5% warmup (industry standard for long training)
    "max_iters": 100000,       # Extended from 20k for better convergence

    # Optimization
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # Logging
    "eval_interval": 500,
    "log_interval": 10,
    "save_interval": 5000,

    # Estimated
    "parameters": "~20M",
    "training_time": "~3 hours on RTX 4090",
    "budget": "~$15",
}

# =============================================================================
# Medium Configuration (90M params, context=512) - RECOMMENDED PRODUCTION ⭐
# =============================================================================
MEDIUM_CONFIG = {
    # Model architecture (10 layers for hierarchical workflow structure)
    "vocab_size": 8192,
    "context_size": 512,
    "d_model": 768,            # GPT-2 Small standard
    "n_layers": 10,            # +2 layers vs previous for hierarchy
    "n_heads": 12,             # head_dim = 64
    "dropout": 0.2,

    # Training
    "batch_size": 20,
    "gradient_accumulation": 4,  # Effective batch = 80
    "learning_rate": 1.2e-4,   # Reduced from 2e-4 (4.5x more params than small)
    "weight_decay": 0.1,
    "warmup_iters": 2000,
    "max_iters": 40000,        # ~12 epochs (prevents overfitting)

    # Optimization
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # Logging
    "eval_interval": 500,
    "log_interval": 10,
    "save_interval": 5000,

    # Estimated
    "parameters": "~90M",
    "training_time": "~8 hours on RTX 4090",
    "budget": "~$45",
}

# =============================================================================
# Medium-1024 Configuration (90M params, context=1024) - Alternative production
# =============================================================================
MEDIUM_1024_CONFIG = {
    # Model architecture (same as Medium, but 1024 context)
    "vocab_size": 8192,
    "context_size": 1024,      # 2× context vs medium
    "d_model": 768,
    "n_layers": 10,
    "n_heads": 12,
    "dropout": 0.2,

    # Training (adjusted for larger context)
    "batch_size": 10,          # Reduced for memory
    "gradient_accumulation": 8,  # Effective batch = 80
    "learning_rate": 1.2e-4,   # Reduced from 2e-4 (4.5x more params than small)
    "weight_decay": 0.1,
    "warmup_iters": 2000,
    "max_iters": 40000,

    # Optimization
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # Logging
    "eval_interval": 500,
    "log_interval": 10,
    "save_interval": 5000,

    # Estimated
    "parameters": "~90M",
    "training_time": "~12 hours on RTX 4090",
    "budget": "~$60",
}

# =============================================================================
# Large Configuration (183M params, context=512) - Maximum quality
# =============================================================================
LARGE_CONFIG = {
    # Model architecture (maximum capacity for 8GB GPU)
    "vocab_size": 8192,
    "context_size": 512,
    "d_model": 1024,           # Large embeddings
    "n_layers": 12,            # Deep model
    "n_heads": 16,             # head_dim = 64
    "dropout": 0.2,

    # Training
    "batch_size": 20,
    "gradient_accumulation": 4,
    "learning_rate": 8e-5,     # Reduced from 2e-4 (9x more params than small)
    "weight_decay": 0.1,
    "warmup_iters": 2000,
    "max_iters": 50000,        # Reduced from 100k (overfitting observed)

    # Optimization
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # Logging
    "eval_interval": 500,
    "log_interval": 10,
    "save_interval": 5000,

    # Estimated
    "parameters": "~183M",
    "training_time": "~12 hours on RTX 4090",
    "budget": "~$75",
}

# =============================================================================
# Large-1024 Configuration (183M params, context=1024) - Maximum everything
# =============================================================================
LARGE_1024_CONFIG = {
    # Model architecture (maximum capacity + context)
    "vocab_size": 8192,
    "context_size": 1024,      # 2× context vs large
    "d_model": 1024,
    "n_layers": 12,
    "n_heads": 16,
    "dropout": 0.2,

    # Training (adjusted for larger context)
    "batch_size": 10,          # Reduced for memory
    "gradient_accumulation": 8,
    "learning_rate": 8e-5,     # Reduced from 2e-4 (9x more params than small)
    "weight_decay": 0.1,
    "warmup_iters": 2000,
    "max_iters": 50000,

    # Optimization
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # Logging
    "eval_interval": 500,
    "log_interval": 10,
    "save_interval": 5000,

    # Estimated
    "parameters": "~183M",
    "training_time": "~18 hours on RTX 4090",
    "budget": "~$90",
}

# =============================================================================
# Configuration Selection Helper
# =============================================================================

CONFIGS = {
    # Default: 512 context (faster, recommended)
    "small": SMALL_CONFIG,
    "medium": MEDIUM_CONFIG,
    "large": LARGE_CONFIG,

    # Alternative: 1024 context (better coverage)
    "small-1024": SMALL_1024_CONFIG,
    "medium-1024": MEDIUM_1024_CONFIG,
    "large-1024": LARGE_1024_CONFIG,
}

def get_config(size="medium"):
    """
    Get model configuration by size.

    Args:
        size: Config name - "small", "medium" (default), "large",
              or with -1024 suffix for 1024 context versions

    Returns:
        Configuration dictionary (copy, safe to modify)

    Raises:
        ValueError: If config name not found

    Example:
        >>> config = get_config("medium")  # 90M params, 512 context
        >>> config = get_config("medium-1024")  # 90M params, 1024 context
    """
    if size not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(
            f"Unknown config: '{size}'\n"
            f"Available configs: {available}\n"
            f"  Default (512 ctx): small, medium ⭐, large\n"
            f"  Alternative (1024 ctx): small-1024, medium-1024, large-1024"
        )
    return CONFIGS[size].copy()

def print_config_comparison():
    """Print comparison table of all configurations."""
    print("=" * 120)
    print("TCT Workflow Generation - Model Configurations")
    print("=" * 120)
    print()

    # Table header
    print(f"{'Config':<15} {'Params':<10} {'Context':<8} {'d_model':<8} {'Layers':<8} {'Heads':<8} {'Time':<20} {'Budget':<10}")
    print("-" * 120)

    # Default configs (512 context)
    print("DEFAULT (512 context, faster):")
    for name in ["small", "medium", "large"]:
        cfg = CONFIGS[name]
        print(f"  {name:<13} {cfg['parameters']:<10} {cfg['context_size']:<8} {cfg['d_model']:<8} "
              f"{cfg['n_layers']:<8} {cfg['n_heads']:<8} {cfg['training_time']:<20} {cfg['budget']:<10}")

    print()

    # Alternative configs (1024 context)
    print("ALTERNATIVE (1024 context, better coverage):")
    for name in ["small-1024", "medium-1024", "large-1024"]:
        cfg = CONFIGS[name]
        print(f"  {name:<13} {cfg['parameters']:<10} {cfg['context_size']:<8} {cfg['d_model']:<8} "
              f"{cfg['n_layers']:<8} {cfg['n_heads']:<8} {cfg['training_time']:<20} {cfg['budget']:<10}")

    print()
    print("=" * 120)
    print()
    print("Recommendations:")
    print("  ⭐ RECOMMENDED: medium (90M, 512 ctx) - Best balance for production")
    print("  - Use medium-1024 if workflows frequently >512 tokens")
    print("  - Use small for quick iteration/debugging")
    print("  - Use large if medium accuracy <95%")
    print()
    print("Context tradeoffs:")
    print("  - 512:  Covers 30% of workflows entirely, 2× faster than 1024")
    print("  - 1024: Covers 56% of workflows entirely, better for long workflows")
    print()

if __name__ == "__main__":
    print_config_comparison()
