"""
Model Configuration for Workflow Generation with TCT

Three preset configurations optimized for GitHub Actions workflow generation:
- Small: 50M params, $20 budget
- Medium: 100M params, $50 budget
- Large: 200M params, $100 budget
"""

# =============================================================================
# Small Configuration (20M params) - Optimized for workflow generation
# =============================================================================
SMALL_CONFIG = {
    # Model architecture (optimized for hierarchical workflows)
    "vocab_size": 8192,        # TCT vocabulary
    "context_size": 512,       # Workflows rarely exceed 500 tokens
    "d_model": 384,            # Embedding dimension (narrower, efficient)
    "n_layers": 8,             # Transformer layers (depth for hierarchy)
    "n_heads": 6,              # Attention heads (64 head_dim - optimal)
    "dropout": 0.1,

    # Training
    "batch_size": 32,
    "gradient_accumulation": 4,  # Effective batch = 128
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_iters": 1000,
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
    "parameters": "~20M",
    "training_time": "~2 hours on 8×A100",
    "budget": "~$15",
}

# =============================================================================
# Medium Configuration (100M params) - Recommended for production
# =============================================================================
MEDIUM_CONFIG = {
    # Model architecture
    "vocab_size": 8192,
    "context_size": 512,
    "d_model": 768,            # Larger embeddings
    "n_layers": 8,             # More depth
    "n_heads": 12,             # More attention heads
    "dropout": 0.1,

    # Training
    "batch_size": 32,
    "gradient_accumulation": 4,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_iters": 2000,
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
    "parameters": "~103M",
    "training_time": "~7 hours on 8×A100",
    "budget": "~$50",
}

# =============================================================================
# Large Configuration (200M params) - Maximum quality
# =============================================================================
LARGE_CONFIG = {
    # Model architecture
    "vocab_size": 8192,
    "context_size": 512,
    "d_model": 1024,           # Large embeddings
    "n_layers": 12,            # Deep model
    "n_heads": 16,             # Many attention heads
    "dropout": 0.1,

    # Training
    "batch_size": 32,
    "gradient_accumulation": 4,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_iters": 2000,
    "max_iters": 150000,

    # Optimization
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,

    # Logging
    "eval_interval": 500,
    "log_interval": 10,
    "save_interval": 5000,

    # Estimated
    "parameters": "~205M",
    "training_time": "~15 hours on 8×A100",
    "budget": "~$100",
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
        size: "small", "medium", or "large"

    Returns:
        Configuration dictionary
    """
    if size not in CONFIGS:
        raise ValueError(f"Unknown config size: {size}. Choose from: {list(CONFIGS.keys())}")
    return CONFIGS[size].copy()

def print_config_comparison():
    """Print comparison table of all configurations"""
    print("=" * 80)
    print("TCT Workflow Generation - Model Configurations")
    print("=" * 80)
    print()
    print(f"{'Metric':<25} {'Small':<20} {'Medium':<20} {'Large':<20}")
    print("-" * 80)

    keys = ["vocab_size", "context_size", "d_model", "n_layers", "n_heads",
            "parameters", "training_time", "budget"]

    for key in keys:
        values = [CONFIGS[size].get(key, "N/A") for size in ["small", "medium", "large"]]
        print(f"{key:<25} {str(values[0]):<20} {str(values[1]):<20} {str(values[2]):<20}")

    print("=" * 80)
    print()
    print("Recommendations:")
    print("  - Small: Quick experiments, proof of concept")
    print("  - Medium: Production workflow generation (RECOMMENDED)")
    print("  - Large: Maximum quality, research")
    print()

if __name__ == "__main__":
    print_config_comparison()
