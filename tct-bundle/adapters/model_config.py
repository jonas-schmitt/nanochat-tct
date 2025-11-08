"""
Model Configuration for Workflow Generation with TCT

Three preset configurations optimized for GitHub Actions workflow generation
with 1024 context (covers 56% of workflows entirely):

- Small:  20M params, 8 layers, ~3h training (100k steps), $15 budget
- Medium: 50M params, 14 layers, ~15h training (100k steps), $75 budget  ⭐ RECOMMENDED
- Large:  138M params, 16 layers + SwiGLU, ~30h training (100k steps), $175 budget

All configs emphasize depth over width for hierarchical workflow modeling.
All configs train for 100k steps (~44 window visits each) for optimal learning.
"""

# =============================================================================
# Small Configuration (20M params, context=1024)
# =============================================================================
SMALL_CONFIG = {
    # Model architecture
    "vocab_size": 8192,
    "context_size": 1024,
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
# Medium Configuration (50M params, context=1024) - RECOMMENDED ⭐
# =============================================================================
MEDIUM_CONFIG = {
    # Model architecture
    "vocab_size": 8192,
    "context_size": 1024,
    "d_model": 512,
    "n_layers": 14,
    "n_heads": 8,
    "dropout": 0.1,

    # Training
    "batch_size": 16,
    "gradient_accumulation": 8,  # Effective batch = 128
    "learning_rate": 2.5e-4,
    "weight_decay": 0.1,
    "warmup_iters": 5000,
    "max_iters": 100000,

    # Optimization
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # Logging
    "eval_interval": 500,
    "log_interval": 10,
    "save_interval": 5000,

    # Estimated
    "parameters": "~50M",
    "training_time": "~15 hours on RTX 4090",
    "budget": "~$75",
}

# =============================================================================
# Large Configuration (138M params, context=1024)
# =============================================================================
LARGE_CONFIG = {
    # Model architecture
    "vocab_size": 8192,
    "context_size": 1024,
    "d_model": 768,
    "n_layers": 16,
    "n_heads": 12,
    "dropout": 0.1,
    "use_swiglu": True,  # SwiGLU activation for +15% params, 3-8% quality boost

    # Training
    "batch_size": 10,
    "gradient_accumulation": 8,
    "learning_rate": 2e-4,
    "weight_decay": 0.1,
    "warmup_iters": 5000,
    "max_iters": 100000,

    # Optimization
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # Logging
    "eval_interval": 500,
    "log_interval": 10,
    "save_interval": 5000,

    # Estimated
    "parameters": "~138M",
    "training_time": "~30 hours on RTX 4090",
    "budget": "~$175",
}

# =============================================================================
# Configuration Selection Helper
# =============================================================================

CONFIGS = {
    "small": SMALL_CONFIG,
    "medium": MEDIUM_CONFIG,
    "large": LARGE_CONFIG,
}

def get_config(size="medium"):
    """
    Get model configuration by size.

    Args:
        size: Config name - "small", "medium" (default), or "large"
              All configs use 1024 context.

    Returns:
        Configuration dictionary (copy, safe to modify)

    Raises:
        ValueError: If config name not found

    Example:
        >>> config = get_config("medium")  # 90M params, 1024 context
    """
    if size not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(
            f"Unknown config: '{size}'\n"
            f"Available configs: {available}\n"
            f"  Recommended: medium (90M params, 1024 context)"
        )
    return CONFIGS[size].copy()

def print_config_comparison():
    """Print comparison table of all configurations."""
    print("=" * 120)
    print("TCT Workflow Generation - Model Configurations (1024 context)")
    print("=" * 120)
    print()

    # Table header
    print(f"{'Config':<10} {'Params':<10} {'Context':<8} {'d_model':<8} {'Layers':<8} {'Heads':<8} {'Time':<20} {'Budget':<10}")
    print("-" * 120)

    for name in ["small", "medium", "large"]:
        cfg = CONFIGS[name]
        marker = " ⭐" if name == "medium" else ""
        print(f"{name + marker:<10} {cfg['parameters']:<10} {cfg['context_size']:<8} {cfg['d_model']:<8} "
              f"{cfg['n_layers']:<8} {cfg['n_heads']:<8} {cfg['training_time']:<20} {cfg['budget']:<10}")

    print()
    print("=" * 120)
    print()
    print("Recommendation:")
    print("  ⭐ RECOMMENDED: medium (50M, 14 layers, 100k steps) - Best balance")
    print("  - Use small (20M, 8 layers) for baseline/debugging")
    print("  - Use large (138M, 16 layers + SwiGLU) for maximum quality")
    print()
    print("All configs emphasize depth over width for hierarchical workflow modeling")
    print("All configs train for 100k steps (~44 window visits each) for optimal learning")
    print()

if __name__ == "__main__":
    print_config_comparison()
