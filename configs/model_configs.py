"""
Schema-agnostic model configurations for TCT experiments.

All schemas use context_size=2048.

Preset architectures (sized for vocab=1000):
- Tiny:   d_model=256, 6 layers, SwiGLU 3.0x   (~5M params)
- Mini:   d_model=384, 8 layers, SwiGLU 3.0x   (~15M params)
- Base:   d_model=512, 10 layers, SwiGLU 3.0x  (~34M params)
- Small:  d_model=512, 16 layers, SwiGLU 2.5x  (~50M params)
- Medium: d_model=768, 16 layers, SwiGLU 3.0x  (~125M params)
- Large:  d_model=1024, 24 layers, SwiGLU 3.25x (~350M params)

SwiGLU: Gated linear unit with 3 FFN matrices (gate, up, down) instead of 2.
Used by LLaMA, Mistral, etc. for better performance.

=== Model Selection Guide ===

Single-epoch token/param ratio determines overfitting risk:
- >10x: Capacity-limited (may underfit)
- 5-10x: Good balance
- 2-5x: Overparameterized (needs regularization)
- 1-2x: Severely overparameterized (will likely overfit)
- <1x: More parameters than tokens (extreme overfitting)

Note: Multi-epoch training (100 epochs) does NOT change these ratios.
Repeated data provides optimization benefit but no new unique information.

Single-epoch ratios by schema:

                    eslintrc    kubernetes/tsconfig
    Model   Params  (21.5M)     (117M)
    ------  ------  ----------  -------------------
    tiny    5M      4.3x        23x    ✓ recommended
    mini    15M     1.4x        7.8x   ✓ recommended
    base    34M     0.6x        3.4x   ~ borderline
    small   50M     0.4x        2.3x   ! overparameterized
    medium  125M    0.2x        0.9x   ✗ not recommended
    large   350M    0.06x       0.3x   ✗ not recommended

Recommendations:
- eslintrc: Use tiny only (even tiny is overparameterized at 4.3x)
- kubernetes/tsconfig: Use tiny or mini (base is borderline)
- medium/large: Not recommended for any schema (severe overfitting)

=== Regularization ===

Clean split based on overparameterization:
- Tiny/Mini/Base: dropout=0.0, weight_decay=0.0 (no regularization)
- Small+: dropout=0.2, weight_decay=0.1 (regularized)

Add regularization to smaller models only if validation curves show overfitting.

=== Architecture Notes ===

The SAME architecture is used for ALL schemas and tokenizers.
Reference: kubernetes (vocab=1000)
- Low vocab schemas (258-1000): params as designed
- High vocab schemas (20k+): More params from larger embeddings
"""

from typing import Dict, Any

# =============================================================================
# Core Model Architectures (vocab/context independent)
# =============================================================================

TINY_ARCH = {
    "d_model": 256,
    "n_layers": 6,
    "n_heads": 4,  # head_dim=64
    "ffn_mult": 3.0,  # SwiGLU multiplier (modern LLM practice)
    "use_swiglu": True,
    "dropout": 0.0,  # No dropout - Chinchilla optimal
    "weight_decay": 0.0,  # No weight decay - Chinchilla optimal
    "transformer_params": "~5M",
}

MINI_ARCH = {
    "d_model": 384,
    "n_layers": 8,
    "n_heads": 6,  # head_dim=64
    "ffn_mult": 3.0,  # SwiGLU multiplier (modern LLM practice)
    "use_swiglu": True,
    "dropout": 0.0,  # No dropout - good token/param ratio
    "weight_decay": 0.0,  # No weight decay - good token/param ratio
    "transformer_params": "~15M",
}

BASE_ARCH = {
    "d_model": 512,
    "n_layers": 10,
    "n_heads": 8,  # head_dim=64
    "ffn_mult": 3.0,  # SwiGLU multiplier (modern LLM practice)
    "use_swiglu": True,
    "dropout": 0.0,  # No dropout - test first, add if needed
    "weight_decay": 0.0,  # No weight decay - test first, add if needed
    "transformer_params": "~34M",
}

SMALL_ARCH = {
    "d_model": 512,
    "n_layers": 14,
    "n_heads": 8,  # head_dim=64
    "ffn_mult": 3.0,  # SwiGLU multiplier (modern LLM practice)
    "use_swiglu": True,
    "dropout": 0.2,  # Dropout for overparameterized model
    "weight_decay": 0.1,  # Weight decay for overparameterized model
    "transformer_params": "~48M",
}

# Wider but shallower variant of SMALL - same params, potentially faster
# d_model=768, n_layers=6 vs d_model=512, n_layers=16
# Fewer sequential layers = faster on GPU, more parallelism per layer
# Higher ffn_mult=3.0 (closer to modern LLMs like LLaMA ~2.67)
SMALL_WIDE_ARCH = {
    "d_model": 768,
    "n_layers": 6,
    "n_heads": 12,  # head_dim=64
    "ffn_mult": 3.0,  # Higher FFN capacity
    "use_swiglu": True,
    "dropout": 0.2,  # Dropout for overparameterized model
    "weight_decay": 0.1,  # Weight decay for overparameterized model
    "transformer_params": "~48M",  # 48.37M - similar to SMALL_ARCH
}

MEDIUM_ARCH = {
    "d_model": 768,
    "n_layers": 16,
    "n_heads": 12,  # head_dim=64
    "ffn_mult": 3.0,  # SwiGLU multiplier to hit ~125M target
    "use_swiglu": True,
    "dropout": 0.2,  # Dropout for overparameterized model
    "weight_decay": 0.1,  # Weight decay for overparameterized model
    "transformer_params": "~125M",
}

LARGE_ARCH = {
    "d_model": 1024,
    "n_layers": 24,
    "n_heads": 16,  # head_dim=64
    "ffn_mult": 3.25,  # SwiGLU multiplier to hit ~350M target
    "use_swiglu": True,
    "dropout": 0.2,  # Dropout for overparameterized model
    "weight_decay": 0.1,  # Weight decay for overparameterized model
    "transformer_params": "~350M",
}

ARCHITECTURES = {
    "tiny": TINY_ARCH,
    "mini": MINI_ARCH,
    "base": BASE_ARCH,
    "small": SMALL_ARCH,
    "small-wide": SMALL_WIDE_ARCH,
    "medium": MEDIUM_ARCH,
    "large": LARGE_ARCH,
}

# =============================================================================
# Training Hyperparameters (context-dependent batch sizes)
# =============================================================================

# =============================================================================
# Dynamic Batch Size Scaling
# =============================================================================
# Target effective batch sizes per model
# - tiny/mini: 128 (Chinchilla-optimal, benefit from larger stable batches)
# - base+: 64 (borderline/overparameterized, smaller batch adds regularization)

TARGET_EFFECTIVE_BATCH = {
    "tiny": 128,
    "mini": 128,
    "base": 64,
    "small": 64,
    "small-wide": 64,
    "medium": 64,
    "large": 64,
}

# Reference batch sizes: max micro batch that fits on 24GB VRAM (RTX 4090/3090)
# These scale linearly with available VRAM (computed in compute_batch_config)
# Smaller models (tiny/mini/base) use larger batches to maximize GPU utilization
# and eliminate gradient accumulation overhead
REFERENCE_VRAM_GB = 24
REFERENCE_BATCH_SIZES = {
    2048: {
        "tiny": 128,       # d=256, L=6, SwiGLU 3.0x, ~5M model - no grad_accum needed
        "mini": 64,        # d=384, L=8, SwiGLU 3.0x, ~15M model - no grad_accum needed
        "base": 32,        # d=512, L=10, SwiGLU 3.0x, ~34M model - minimal grad_accum
        "small": 16,       # d=512, L=16, SwiGLU 2.5x, ~50M model
        "small-wide": 16,  # d=768, L=6, SwiGLU 3.0x, ~48M model
        "medium": 8,       # d=768, L=16, SwiGLU 3.0x, ~125M model
        "large": 4,        # d=1024, L=24, SwiGLU 3.25x, ~350M model
    },
}


def get_gpu_memory_gb() -> float:
    """Detect GPU VRAM in GB. Returns 24 as default if detection fails."""
    try:
        import torch
        if torch.cuda.is_available():
            # Get memory of first GPU in bytes, convert to GB
            mem_bytes = torch.cuda.get_device_properties(0).total_memory
            return mem_bytes / (1024 ** 3)
    except Exception:
        pass
    return REFERENCE_VRAM_GB  # Default to reference


def compute_batch_config(model_size: str, context_size: int, gpu_memory_gb: float = None) -> dict:
    """
    Compute optimal batch size and gradient accumulation based on GPU memory.

    Uses power-of-2 batch sizes for clean division with target effective batch.

    Args:
        model_size: Model size name
        context_size: Context/sequence length
        gpu_memory_gb: GPU memory in GB (auto-detected if None)

    Returns:
        Dict with 'batch_size' and 'gradient_accumulation'
    """
    if gpu_memory_gb is None:
        gpu_memory_gb = get_gpu_memory_gb()

    # Get reference batch size for this model/context
    if context_size not in REFERENCE_BATCH_SIZES:
        closest = min(REFERENCE_BATCH_SIZES.keys(), key=lambda x: abs(x - context_size))
        ref_batch = REFERENCE_BATCH_SIZES[closest].get(model_size, 4)
    else:
        ref_batch = REFERENCE_BATCH_SIZES[context_size].get(model_size, 4)

    # Scale batch size proportionally to GPU memory (round to avoid truncation issues)
    scale_factor = gpu_memory_gb / REFERENCE_VRAM_GB
    max_batch = max(1, round(ref_batch * scale_factor))

    # Find largest power-of-2 batch size that:
    # 1. Fits in GPU memory (≤ max_batch)
    # 2. Divides target evenly OR equals target
    # Valid batch sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256
    valid_batches = [b for b in [256, 128, 64, 32, 16, 8, 4, 2, 1] if b <= max_batch]

    if not valid_batches:
        batch_size = 1
    else:
        batch_size = valid_batches[0]  # Largest that fits

    # Compute gradient accumulation to reach target (model-specific)
    target_eff_batch = TARGET_EFFECTIVE_BATCH.get(model_size, 64)
    grad_accum = max(1, target_eff_batch // batch_size)

    return {
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "gpu_memory_gb": gpu_memory_gb,
    }

# Common training hyperparameters (all configs)
# Note: weight_decay is model-specific (in ARCH dicts), not here
COMMON_TRAINING = {
    "learning_rate": 4e-4,  # Base LR (overridden by LR_ADJUSTMENTS per model size)
    "lr_schedule": "cosine",  # Cosine decay with warmup
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "eval_interval": 500,
    "log_interval": 10,
    "save_interval": 5000,
}

# Learning rate adjustments by model size (smaller for larger models)
LR_ADJUSTMENTS = {
    "tiny": 6e-4,        # ~5M params (higher LR for smaller model)
    "mini": 5e-4,        # ~15M params
    "base": 4e-4,        # ~32M params
    "small": 4e-4,       # ~50M params (base LR for batch 64)
    "small-wide": 4e-4,  # ~50M params (same as small)
    "medium": 3e-4,      # ~125M params (scaled down)
    "large": 2e-4,       # ~350M params (scaled down)
}

# =============================================================================
# Full Model Configurations
# =============================================================================

MODEL_CONFIGS = {
    "tiny": {
        **TINY_ARCH,
        **COMMON_TRAINING,
        "learning_rate": LR_ADJUSTMENTS["tiny"],
        "epochs_multiplier": 1.5,  # 150 epochs (extended training for small models)
        "description": "Tiny model (~5M with vocab=1k), 6 layers - best token/param ratio",
    },
    "mini": {
        **MINI_ARCH,
        **COMMON_TRAINING,
        "learning_rate": LR_ADJUSTMENTS["mini"],
        "epochs_multiplier": 1.5,  # 150 epochs (extended training for small models)
        "description": "Mini model (~15M with vocab=1k), 8 layers - good generalization",
    },
    "base": {
        **BASE_ARCH,
        **COMMON_TRAINING,
        "learning_rate": LR_ADJUSTMENTS["base"],
        "epochs_multiplier": 1.5,  # 150 epochs (extended training for small models)
        "description": "Base model (~32M with vocab=1k), 10 layers - balanced",
    },
    "small": {
        **SMALL_ARCH,
        **COMMON_TRAINING,
        "learning_rate": LR_ADJUSTMENTS["small"],
        "description": "Small model (~48M with vocab=1k), 14 layers",
    },
    "small-wide": {
        **SMALL_WIDE_ARCH,
        **COMMON_TRAINING,
        "learning_rate": LR_ADJUSTMENTS["small-wide"],
        "description": "Small-wide model (~48M with vocab=1k), 6 layers, d_model=768, ffn=3.0",
    },
    "medium": {
        **MEDIUM_ARCH,
        **COMMON_TRAINING,
        "learning_rate": LR_ADJUSTMENTS["medium"],
        "description": "Medium model (~126M with vocab=1k), good quality/speed balance",
    },
    "large": {
        **LARGE_ARCH,
        **COMMON_TRAINING,
        "learning_rate": LR_ADJUSTMENTS["large"],
        "description": "Large model (~357M with vocab=1k), best quality",
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
    ffn_mult = arch.get("ffn_mult", 4.0)
    use_swiglu = arch.get("use_swiglu", False)

    # Token embedding: vocab_size × d_model
    token_embedding = vocab_size * d_model

    # Position embedding: context_size × d_model
    position_embedding = context_size * d_model

    # Output projection (separate from embedding): d_model × vocab_size
    output_projection = d_model * vocab_size

    # Per-layer params: attention (4 × d_model²) + FFN
    # Standard FFN: 2 × ffn_mult × d_model² (up + down projections)
    # SwiGLU FFN: 3 × ffn_mult × d_model² (gate + up + down projections)
    attention_params = 4 * d_model * d_model
    ffn_matrices = 3 if use_swiglu else 2
    ffn_params = ffn_matrices * ffn_mult * d_model * d_model
    params_per_layer = attention_params + ffn_params
    transformer_params = n_layers * params_per_layer

    # Layer norms and biases (small contribution)
    norm_params = n_layers * 2 * d_model + 2 * d_model

    total = token_embedding + position_embedding + output_projection + transformer_params + norm_params
    return int(total)


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

    # Compute batch sizes dynamically based on GPU memory
    batch_config = compute_batch_config(model_size, context_size)
    config["batch_size"] = batch_config["batch_size"]
    config["gradient_accumulation"] = batch_config["gradient_accumulation"]
    config["gpu_memory_gb"] = batch_config["gpu_memory_gb"]

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
    print(f"{'Size':<10} {'d_model':<10} {'Layers':<10} {'Heads':<10} {'FFN':<12} {'Params':<15}")
    print("-" * 70)
    for name, arch in ARCHITECTURES.items():
        ffn_mult = arch.get("ffn_mult", 4.0)
        use_swiglu = arch.get("use_swiglu", False)
        ffn_str = f"SwiGLU {ffn_mult}x" if use_swiglu else f"{ffn_mult}x"
        print(f"{name:<10} {arch['d_model']:<10} {arch['n_layers']:<10} {arch['n_heads']:<10} {ffn_str:<12} {arch['transformer_params']}")

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

    # Training params for different GPUs (context=2048 for all)
    # Target GPUs: RTX 4090 (24GB), A40 (40GB), A100 40GB, A100 80GB
    target_gpus = [
        ("RTX 4090", 24),
        ("A40/A100-40", 40),
        ("A100-80", 80),
    ]

    print("Batch Sizes by GPU (context=2048, target eff_batch=64):")
    print("-" * 85)
    print(f"{'Model':<12}", end="")
    for gpu_name, _ in target_gpus:
        print(f" {gpu_name:>22}", end="")
    print()
    print(f"{'':12}", end="")
    for gpu_name, vram in target_gpus:
        print(f" {'(batch×accum=eff)':>22}", end="")
    print()
    print("-" * 85)

    for size in ["tiny", "mini", "base", "small", "medium", "large"]:
        print(f"{size:<12}", end="")
        for gpu_name, vram in target_gpus:
            cfg = compute_batch_config(size, 2048, gpu_memory_gb=vram)
            b, a, e = cfg["batch_size"], cfg["gradient_accumulation"], cfg["effective_batch"]
            print(f" {b:>6}×{a:<2}={e:<10}", end="")
        print()

    print()


if __name__ == "__main__":
    print_model_summary()
