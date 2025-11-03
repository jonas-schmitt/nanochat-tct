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
    "vocab_size": 8192,        # TCT vocabulary (8192 base, stride=32 for position mapping)
    "context_size": 1024,      # Total window size (1 position + 1023 content tokens)
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
# Medium-Small Configuration (~35M params) - Deep for hierarchy, narrow for efficiency
# =============================================================================
MEDIUM_SMALL_CONFIG = {
    # Model architecture (depth-first: trust TCT position encoding for context)
    "vocab_size": 8193,        # TCT vocabulary (8192 base) + 1 dedicated PAD token (8192)
    "context_size": 1024,      # Total window size (1 position + 1023 content tokens)
    "d_model": 384,            # Same as small (position tokens handle context disambiguation)
    "n_layers": 12,            # +50% MORE DEPTH than small for workflow hierarchy
    "n_heads": 6,              # head_dim = 64 (optimal)
    "dropout": 0.1,

    # Training
    "batch_size": 32,
    "gradient_accumulation": 4,  # Effective batch = 128
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_iters": 1500,
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
    "parameters": "~35M",
    "training_time": "~2.5-3 hours on 8×A100",
    "budget": "~$20",
}

# =============================================================================
# Medium Configuration (100M params) - Recommended for production
# =============================================================================
MEDIUM_CONFIG = {
    # Model architecture
    "vocab_size": 8192,        # TCT vocabulary (8192 base, stride=32 for position mapping)
    "context_size": 1024,      # Total window size (1 position + 1023 content tokens)
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
    "vocab_size": 8192,        # TCT vocabulary (8192 base, stride=32 for position mapping)
    "context_size": 1024,      # Total window size (1 position + 1023 content tokens)
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
    "medium-small": MEDIUM_SMALL_CONFIG,
    "medium": MEDIUM_CONFIG,
    "large": LARGE_CONFIG,
}

def get_config(size="medium"):
    """
    Get model configuration by size.

    Args:
        size: "small", "medium-small", "medium", or "large"

    Returns:
        Configuration dictionary
    """
    if size not in CONFIGS:
        raise ValueError(f"Unknown config size: {size}. Choose from: {list(CONFIGS.keys())}")
    return CONFIGS[size].copy()

def print_config_comparison():
    """Print comparison table of all configurations"""
    print("=" * 100)
    print("TCT Workflow Generation - Model Configurations")
    print("=" * 100)
    print()
    print(f"{'Metric':<25} {'Small':<18} {'Medium-Small':<18} {'Medium':<18} {'Large':<18}")
    print("-" * 100)

    keys = ["vocab_size", "context_size", "d_model", "n_layers", "n_heads",
            "parameters", "training_time", "budget"]

    for key in keys:
        values = [CONFIGS[size].get(key, "N/A") for size in ["small", "medium-small", "medium", "large"]]
        print(f"{key:<25} {str(values[0]):<18} {str(values[1]):<18} {str(values[2]):<18} {str(values[3]):<18}")

    print("=" * 100)
    print()
    print("Recommendations:")
    print("  - Small: Quick experiments, proof of concept (~20M)")
    print("  - Medium-Small: Deep & efficient (384×12, ~35M) ⭐ RECOMMENDED first try with ASCII+position encoding")
    print("  - Medium: Production workflow generation (~103M)")
    print("  - Large: Maximum quality, research (~205M)")
    print()
    print("Note: Start with Medium-Small. If performance < 95%, scale to Medium (512×10 or 768×8).")
    print()

if __name__ == "__main__":
    print_config_comparison()
