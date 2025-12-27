"""
Schema-agnostic model configurations for TCT experiments.

All schemas use context_size=2048.

Four preset architectures:
- Small:      d_model=512, 10 layers (~50M params)
- Small-deep: d_model=384, 20 layers (~50M params, deeper)
- Medium:     d_model=768, 13 layers (~125M params)
- Large:      d_model=1024, 24 layers (~350M params)

The SAME architecture is used for both TCT and UTF8 tokenizers.
- TCT: Smaller vocab → fewer embedding params
- UTF8: Larger vocab → more embedding params (same transformer capacity)
"""

from typing import Dict, Any

# =============================================================================
# Core Model Architectures (vocab/context independent)
# =============================================================================

SMALL_ARCH = {
    "d_model": 512,
    "n_layers": 10,
    "n_heads": 8,
    "dropout": 0.0,  # No dropout for pretraining (GPT-3, LLaMA use 0)
    "transformer_params": "~31M",
}

# Small-deep: Same ~50M params but narrower & deeper (better for complex patterns)
SMALL_DEEP_ARCH = {
    "d_model": 384,
    "n_layers": 20,
    "n_heads": 6,
    "dropout": 0.0,  # No dropout for pretraining
    "transformer_params": "~35M",
}

MEDIUM_ARCH = {
    "d_model": 768,
    "n_layers": 13,
    "n_heads": 12,
    "dropout": 0.0,  # No dropout for pretraining
    "transformer_params": "~92M",
}

LARGE_ARCH = {
    "d_model": 1024,
    "n_layers": 24,
    "n_heads": 16,
    "dropout": 0.0,  # No dropout for pretraining
    "transformer_params": "~302M",
}

ARCHITECTURES = {
    "small": SMALL_ARCH,
    "small-deep": SMALL_DEEP_ARCH,
    "medium": MEDIUM_ARCH,
    "large": LARGE_ARCH,
}

# =============================================================================
# Training Hyperparameters (context-dependent batch sizes)
# =============================================================================

# Batch sizes optimized for RTX 4090 (24GB VRAM)
# Also works for RTX 3090/A5000 (same VRAM)
# Effective batch = batch_size × gradient_accumulation × world_size
# Target effective batch: 32 (optimal for our data sizes: 10^7-10^8 tokens)

# All schemas use context_size=2048
# Smaller effective batch (32) works better for our data sizes (10^7-10^8 tokens)
# Research shows CBS scales with data size, not model size
# Maximize micro batch for GPU efficiency, use grad_accum to reach target eff batch
TRAINING_PARAMS = {
    2048: {
        "small": {"batch_size": 16, "gradient_accumulation": 2},        # ~50M model, eff=32
        "small-deep": {"batch_size": 16, "gradient_accumulation": 2},   # ~50M model (deeper), eff=32
        "medium": {"batch_size": 8, "gradient_accumulation": 4},        # ~125M model, eff=32
        "large": {"batch_size": 4, "gradient_accumulation": 8},         # ~350M model, eff=32
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
    "small": 3e-4,       # ~50M params
    "small-deep": 3e-4,  # ~50M params (deeper, same LR)
    "medium": 2e-4,      # ~125M params
    "large": 1.5e-4,     # ~350M params
}

# =============================================================================
# Full Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    "small": {
        **SMALL_ARCH,
        **COMMON_TRAINING,
        "learning_rate": LR_ADJUSTMENTS["small"],
        "description": "Small model (~50M with 20k vocab), fastest training",
    },
    "small-deep": {
        **SMALL_DEEP_ARCH,
        **COMMON_TRAINING,
        "learning_rate": LR_ADJUSTMENTS["small-deep"],
        "description": "Small-deep model (~50M with 20k vocab), narrower but deeper",
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

    # Apply batch multiplier for larger GPUs (set by run scripts based on VRAM)
    import os
    batch_multiplier = int(os.environ.get("TCT_BATCH_MULTIPLIER", "1"))
    batch_boost = int(os.environ.get("TCT_BATCH_SIZE_BOOST", "0"))  # For 32GB GPUs

    config["batch_size"] = batch_config["batch_size"] * batch_multiplier + batch_boost
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

    # Parameter estimates for each schema (all use context=2048)
    from configs.schema_configs import SCHEMA_CONFIGS

    print("Estimated Total Parameters by Schema (context=2048):")
    print("-" * 80)

    for schema_name, cfg in SCHEMA_CONFIGS.items():
        tct_vocab = cfg["tct_vocab_size"]
        utf8_vocab = cfg["utf8_vocab_size"]
        print(f"\n{schema_name} (TCT={tct_vocab}, UTF8={utf8_vocab}):")
        print(f"  {'Size':<12} {'TCT':<15} {'UTF8':<15}")
        for name in ARCHITECTURES:
            p_tct = estimate_params(name, tct_vocab, 2048)
            p_utf8 = estimate_params(name, utf8_vocab, 2048)
            print(f"  {name:<12} {p_tct:>12,} {p_utf8:>14,}")

    print()

    # Training params (context=2048 for all)
    print("Batch Sizes (context=2048, single RTX 4090):")
    print("-" * 70)
    print(f"{'Size':<12} {'Batch':<10} {'Grad Accum':<12} {'Effective Batch':<15}")
    params = TRAINING_PARAMS[2048]
    for size in ["small", "small-deep", "medium", "large"]:
        p = params[size]
        effective = p["batch_size"] * p["gradient_accumulation"]
        print(f"{size:<12} {p['batch_size']:<10} {p['gradient_accumulation']:<12} {effective:<15}")

    print()


if __name__ == "__main__":
    print_model_summary()
