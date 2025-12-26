"""
Schema-agnostic model configurations for TCT experiments.

Three preset architectures targeting standard GPT sizes:
- Small:  ~50M params with k8s vocab (d_model=512, 8 layers)
- Medium: ~125M params with k8s vocab (d_model=768, 12 layers)
- Large:  ~350M params with k8s vocab (d_model=1024, 24 layers)

The SAME architecture is used for both TCT-BPE and UTF8-BPE.
- TCT-BPE: Smaller vocab → fewer params
- UTF8-BPE: Larger vocab → more params (same transformer capacity)

Example parameter counts (kubernetes, context=2048):
| Model  | TCT-BPE (20k) | UTF8-BPE (24k) | Target |
|--------|---------------|----------------|--------|
| Small  | ~47M          | ~51M           | ~50M   |
| Medium | ~117M         | ~123M          | ~125M  |
| Large  | ~345M         | ~353M          | ~350M  |
"""

from typing import Dict, Any

# =============================================================================
# Core Model Architectures (vocab/context independent)
# =============================================================================

SMALL_ARCH = {
    "d_model": 512,
    "n_layers": 10,
    "n_heads": 8,
    "dropout": 0.1,
    "transformer_params": "~31M",
}

MEDIUM_ARCH = {
    "d_model": 768,
    "n_layers": 13,
    "n_heads": 12,
    "dropout": 0.1,
    "transformer_params": "~92M",
}

LARGE_ARCH = {
    "d_model": 1024,
    "n_layers": 24,
    "n_heads": 16,
    "dropout": 0.1,
    "transformer_params": "~302M",
}

ARCHITECTURES = {
    "small": SMALL_ARCH,
    "medium": MEDIUM_ARCH,
    "large": LARGE_ARCH,
}

# =============================================================================
# Training Hyperparameters (context-dependent batch sizes)
# =============================================================================

# Batch sizes optimized for RTX 4090 (24GB VRAM)
# Also works for RTX 3090/A5000 (same VRAM)
# Effective batch = batch_size × gradient_accumulation × world_size
# Target effective batch: ~128 for stability

TRAINING_PARAMS = {
    # Context 256 (tsconfig): fits easily, large batch
    256: {
        "small": {"batch_size": 64, "gradient_accumulation": 2},   # ~50M model
        "medium": {"batch_size": 32, "gradient_accumulation": 4},  # ~100M model
        "large": {"batch_size": 16, "gradient_accumulation": 8},   # ~323M model
    },
    # Context 512 (eslintrc): moderate batch
    512: {
        "small": {"batch_size": 32, "gradient_accumulation": 4},   # ~50M model
        "medium": {"batch_size": 16, "gradient_accumulation": 8},  # ~100M model
        "large": {"batch_size": 8, "gradient_accumulation": 16},   # ~323M model
    },
    # Context 2048 (kubernetes): smaller batch for memory
    2048: {
        "small": {"batch_size": 16, "gradient_accumulation": 8},   # ~47M model
        "medium": {"batch_size": 4, "gradient_accumulation": 32},  # ~117M model
        "large": {"batch_size": 2, "gradient_accumulation": 64},   # ~345M model
    },
}

# Common training hyperparameters (all configs)
COMMON_TRAINING = {
    "learning_rate": 3e-4,  # Will be slightly reduced for large model
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "eval_interval": 500,
    "log_interval": 10,
    "save_interval": 5000,
}

# Learning rate adjustments by model size (smaller for larger models)
LR_ADJUSTMENTS = {
    "small": 3e-4,   # ~50M params
    "medium": 2e-4,  # ~125M params
    "large": 1.5e-4, # ~350M params
}

# =============================================================================
# Full Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    "small": {
        **SMALL_ARCH,
        **COMMON_TRAINING,
        "learning_rate": LR_ADJUSTMENTS["small"],
        "description": "Small model (~33M with 20k vocab), fastest training",
    },
    "medium": {
        **MEDIUM_ARCH,
        **COMMON_TRAINING,
        "learning_rate": LR_ADJUSTMENTS["medium"],
        "description": "Medium model (~60M with 20k vocab), good quality/speed balance",
    },
    "large": {
        **LARGE_ARCH,
        **COMMON_TRAINING,
        "learning_rate": LR_ADJUSTMENTS["large"],
        "description": "Large model (~138M with 20k vocab), best quality",
    },
}


def estimate_params(model_size: str, vocab_size: int, context_size: int = 2048) -> int:
    """
    Estimate total model parameters.

    Args:
        model_size: "small", "medium", or "large"
        vocab_size: Vocabulary size for embedding layer
        context_size: Context size for position embeddings

    Returns:
        Estimated total parameters
    """
    arch = ARCHITECTURES[model_size]
    d_model = arch["d_model"]
    n_layers = arch["n_layers"]

    # Token embedding: vocab_size × d_model
    token_embedding = vocab_size * d_model

    # Position embedding: context_size × d_model
    position_embedding = context_size * d_model

    # Output projection (separate from embedding): d_model × vocab_size
    output_projection = d_model * vocab_size

    # Per-layer params: attention (4 × d_model²) + FFN (8 × d_model²)
    # Standard: 12 × d_model² per layer
    # SwiGLU: similar total with different structure
    params_per_layer = 12 * d_model * d_model
    transformer_params = n_layers * params_per_layer

    # Layer norms and biases (small contribution)
    norm_params = n_layers * 2 * d_model + 2 * d_model

    total = token_embedding + position_embedding + output_projection + transformer_params + norm_params
    return total


def get_model_config(
    model_size: str,
    vocab_size: int,
    context_size: int,
    epochs: int = 100,
) -> Dict[str, Any]:
    """
    Get full model configuration for training.

    Args:
        model_size: "small", "medium", or "large"
        vocab_size: Vocabulary size (from schema config)
        context_size: Context size (from schema config)
        epochs: Number of training epochs

    Returns:
        Complete configuration dictionary ready for training

    Example:
        >>> from configs import get_schema_config, get_model_config
        >>> schema = get_schema_config("kubernetes")
        >>> config = get_model_config(
        ...     model_size="small",
        ...     vocab_size=schema["tct_vocab_size"],
        ...     context_size=schema["context_size"],
        ...     epochs=100,
        ... )
    """
    if model_size not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model size: '{model_size}'. Available: {available}")

    config = MODEL_CONFIGS[model_size].copy()

    # Add vocab and context
    config["vocab_size"] = vocab_size
    config["context_size"] = context_size

    # Get context-specific batch sizes
    if context_size not in TRAINING_PARAMS:
        # Interpolate for unsupported context sizes
        closest = min(TRAINING_PARAMS.keys(), key=lambda x: abs(x - context_size))
        batch_config = TRAINING_PARAMS[closest][model_size]
    else:
        batch_config = TRAINING_PARAMS[context_size][model_size]

    # Apply batch multiplier for larger GPUs (set by job.sh based on VRAM)
    import os
    batch_multiplier = int(os.environ.get("TCT_BATCH_MULTIPLIER", "1"))

    config["batch_size"] = batch_config["batch_size"] * batch_multiplier
    config["gradient_accumulation"] = max(1, batch_config["gradient_accumulation"] // batch_multiplier)

    # Add training duration
    config["epochs"] = epochs

    # Warmup: 5% of first epoch (will be computed in training script)
    config["warmup_fraction"] = 0.05

    # Estimate parameters
    config["estimated_params"] = estimate_params(model_size, vocab_size, context_size)

    return config


def print_model_summary():
    """Print summary of model architectures."""
    print("=" * 80)
    print("Model Architectures for TCT Experiments")
    print("=" * 80)
    print()

    # Architecture table
    print(f"{'Size':<10} {'d_model':<10} {'Layers':<10} {'Heads':<10} {'Transformer Params':<20}")
    print("-" * 60)
    for name, arch in ARCHITECTURES.items():
        swiglu = " (SwiGLU)" if arch.get("use_swiglu") else ""
        print(f"{name:<10} {arch['d_model']:<10} {arch['n_layers']:<10} {arch['n_heads']:<10} {arch['transformer_params']}{swiglu}")

    print()

    # Parameter estimates for each schema
    print("Estimated Total Parameters by Schema:")
    print("-" * 80)

    schemas = [
        ("kubernetes", 20000, 23887, 2048),
        ("eslintrc", 10000, 18337, 512),
        ("tsconfig", 10000, 16197, 256),
    ]

    for schema, tct_vocab, utf8_vocab, context in schemas:
        print(f"\n{schema} (TCT={tct_vocab}, UTF8={utf8_vocab}, ctx={context}):")
        print(f"  {'Size':<10} {'TCT-BPE':<15} {'UTF8-BPE':<15}")
        for name in ARCHITECTURES:
            p_tct = estimate_params(name, tct_vocab, context)
            p_utf8 = estimate_params(name, utf8_vocab, context)
            print(f"  {name:<10} {p_tct:>12,} {p_utf8:>14,}")

    print()

    # Training params by context
    print("Batch Sizes by Context (single GPU):")
    print("-" * 60)
    print(f"{'Context':<10} {'Small':<20} {'Medium':<20} {'Large':<20}")
    for ctx in sorted(TRAINING_PARAMS.keys()):
        small = TRAINING_PARAMS[ctx]["small"]
        medium = TRAINING_PARAMS[ctx]["medium"]
        large = TRAINING_PARAMS[ctx]["large"]
        print(f"{ctx:<10} {small['batch_size']}×{small['gradient_accumulation']:<14} "
              f"{medium['batch_size']}×{medium['gradient_accumulation']:<14} "
              f"{large['batch_size']}×{large['gradient_accumulation']:<14}")

    print()


if __name__ == "__main__":
    print_model_summary()
