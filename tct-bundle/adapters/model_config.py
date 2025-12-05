"""
Model Configuration for Kubernetes Manifest Generation with TCT

Three preset configurations optimized for Kubernetes manifest generation
with 2048 context (balanced coverage/speed):

- Small:  33M params, 8 layers, ~2 days training (100k steps), $10 budget  ⭐ RECOMMENDED START
- Medium: 60M params, 14 layers, ~16h training (100k steps), $80 budget
- Large:  138M params, 16 layers + SwiGLU, ~30h training (100k steps), $175 budget

All configs emphasize depth over width for hierarchical manifest modeling.
All configs train for 100k steps for optimal learning on 265k manifests.
"""

# =============================================================================
# Small Configuration (33M params, context=2048) - RECOMMENDED START ⭐
# =============================================================================
SMALL_2048_CONFIG = {
    # Model architecture
    "vocab_size": 20000,        # Kubernetes vocab size
    "context_size": 2048,       # 2× context (covers ~70% of manifests entirely)
    "d_model": 384,
    "n_layers": 8,
    "n_heads": 6,
    "dropout": 0.1,

    # Training (adjusted for 2048 context - half of 4096)
    "batch_size": 16,            # Balanced for 2048 context
    "gradient_accumulation": 8,  # Effective batch = 128
    "learning_rate": 3e-4,
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
    "parameters": "~33M",
    "training_time": "~2 days on RTX 4070 Laptop",
    "budget": "~$10",
}

# =============================================================================
# Small-1024 Configuration (33M params, context=1024) - Fast iteration
# =============================================================================
SMALL_1024_CONFIG = {
    # Model architecture
    "vocab_size": 20000,
    "context_size": 1024,
    "d_model": 384,
    "n_layers": 8,
    "n_heads": 6,
    "dropout": 0.1,

    # Training (optimized for 100k steps)
    "batch_size": 32,          # Doubled from 16 (fits in 8GB, more stable gradients)
    "gradient_accumulation": 4,  # Reduced to keep effective batch = 128
    "learning_rate": 3e-4,
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
    "parameters": "~33M",
    "training_time": "~3 hours on RTX 4090",
    "budget": "~$15",
}

# =============================================================================
# Small-4096 Configuration (33M params, context=4096) - Full coverage
# =============================================================================
SMALL_4096_CONFIG = {
    # Model architecture
    "vocab_size": 20000,
    "context_size": 4096,       # 4× larger context (covers 82% of manifests entirely)
    "d_model": 384,
    "n_layers": 8,
    "n_heads": 6,
    "dropout": 0.1,

    # Training (adjusted for 4096 context - uses more memory)
    "batch_size": 8,             # Reduced from 32 due to 4× context size
    "gradient_accumulation": 16,  # Keep effective batch = 128
    "learning_rate": 3e-4,
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
    "parameters": "~33M",
    "training_time": "~4 hours on RTX 4090",
    "budget": "~$20",
}

# =============================================================================
# Medium Configuration (60M params, context=1024)
# =============================================================================
MEDIUM_1024_CONFIG = {
    # Model architecture
    "vocab_size": 20000,
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
    "parameters": "~60M",
    "training_time": "~15 hours on RTX 4090",
    "budget": "~$75",
}

# =============================================================================
# Medium-2048 Configuration (60M params, context=2048)
# =============================================================================
MEDIUM_2048_CONFIG = {
    # Model architecture
    "vocab_size": 20000,
    "context_size": 2048,
    "d_model": 512,
    "n_layers": 14,
    "n_heads": 8,
    "dropout": 0.1,

    # Training (adjusted for 2048 context)
    "batch_size": 4,              # Reduced due to 2048 context
    "gradient_accumulation": 32,   # Keep effective batch = 128
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
    "parameters": "~60M",
    "training_time": "~16 hours on RTX 4090",
    "budget": "~$80",
}

# =============================================================================
# Medium-4096 Configuration (87M params, context=4096)
# =============================================================================
MEDIUM_4096_CONFIG = {
    # Model architecture
    "vocab_size": 20000,
    "context_size": 4096,       # Full manifest coverage
    "d_model": 576,             # Wider for richer embeddings (72 dim per head)
    "n_layers": 16,             # Deep for hierarchical structure
    "n_heads": 8,
    "dropout": 0.1,

    # Training (adjusted for 4096 context + 87M model)
    "batch_size": 4,              # Reduced due to 4096 context + larger model
    "gradient_accumulation": 32,  # Keep effective batch = 128
    "learning_rate": 2e-4,        # Slightly lower for larger model
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
    "parameters": "~87M",
    "training_time": "~28 hours on RTX 4090",
    "budget": "~$140",
}

# =============================================================================
# Large Configuration (138M params, context=1024)
# =============================================================================
LARGE_1024_CONFIG = {
    # Model architecture
    "vocab_size": 20000,
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
    "small-1024": SMALL_1024_CONFIG,
    "small-2048": SMALL_2048_CONFIG,
    "small-4096": SMALL_4096_CONFIG,
    "medium-1024": MEDIUM_1024_CONFIG,
    "medium-2048": MEDIUM_2048_CONFIG,
    "medium-4096": MEDIUM_4096_CONFIG,
    "large-1024": LARGE_1024_CONFIG,
}

def get_config(size="small-2048"):
    """
    Get model configuration by size.

    Args:
        size: Config name - "small-2048" (default), "medium-2048", etc.

    Returns:
        Configuration dictionary (copy, safe to modify)

    Raises:
        ValueError: If config name not found

    Example:
        >>> config = get_config("small-2048")  # 33M params, 2048 context
    """
    if size not in CONFIGS:
        available = ", ".join(CONFIGS.keys())
        raise ValueError(
            f"Unknown config: '{size}'\n"
            f"Available configs: {available}\n"
            f"  Recommended: small-2048 (33M params, 2048 context)"
        )
    return CONFIGS[size].copy()

def print_config_comparison():
    """Print comparison table of all configurations."""
    print("=" * 120)
    print("TCT Kubernetes Manifest Generation - Model Configurations")
    print("=" * 120)
    print()

    # Table header
    print(f"{'Config':<15} {'Params':<10} {'Context':<8} {'d_model':<8} {'Layers':<8} {'Heads':<8} {'Time':<25} {'Budget':<10}")
    print("-" * 120)

    for name in ["small-1024", "small-2048", "small-4096", "medium-1024", "medium-2048", "medium-4096", "large-1024"]:
        cfg = CONFIGS[name]
        marker = " ⭐" if name == "small-2048" else ""
        print(f"{name + marker:<15} {cfg['parameters']:<10} {cfg['context_size']:<8} {cfg['d_model']:<8} "
              f"{cfg['n_layers']:<8} {cfg['n_heads']:<8} {cfg['training_time']:<25} {cfg['budget']:<10}")

    print()
    print("=" * 120)
    print()
    print("Recommendation:")
    print("  ⭐ RECOMMENDED START: small-2048 (33M, 8 layers, 100k steps) - Best balance for initial training")
    print("  - Use small-1024 for faster iteration/debugging")
    print("  - Use medium-2048 (60M, 14 layers) for production quality")
    print("  - Use large-1024 (138M, 16 layers + SwiGLU) for maximum quality")
    print()
    print("All configs emphasize depth over width for hierarchical manifest modeling")
    print("All configs train for 100k steps for optimal learning on 265k manifests")
    print()

if __name__ == "__main__":
    print_config_comparison()
