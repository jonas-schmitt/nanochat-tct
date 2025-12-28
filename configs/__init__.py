"""
Configuration module for TCT experiments.

Contains:
- schema_configs: Schema-specific configurations (vocab, context, data paths)
- model_configs: Model architecture configurations (sizes, hyperparameters)
- jsonl_dataloader: Unified dataloader for pre-encoded JSONL sequences (requires torch)
"""

from .schema_configs import (
    SCHEMA_CONFIGS,
    get_schema_config,
    get_vocab_size,
    get_train_tokens,
)
from .model_configs import (
    MODEL_CONFIGS,
    ARCHITECTURES,
    get_model_config,
    estimate_params,
    compute_batch_config,
    get_gpu_memory_gb,
    TARGET_EFFECTIVE_BATCH,
)

# Dataloader requires torch, import lazily
def _import_dataloader():
    from .jsonl_dataloader import (
        create_dataloader,
        create_reshuffled_dataloaders,
        get_epoch_steps,
        get_warmup_steps,
        PAD_TOKEN_ID,
    )
    return create_dataloader, create_reshuffled_dataloaders, get_epoch_steps, get_warmup_steps, PAD_TOKEN_ID


# Re-export for convenience when torch is available
def __getattr__(name):
    if name in ("create_dataloader", "create_reshuffled_dataloaders", "get_epoch_steps", "get_warmup_steps", "PAD_TOKEN_ID"):
        create_dataloader, create_reshuffled_dataloaders, get_epoch_steps, get_warmup_steps, PAD_TOKEN_ID = _import_dataloader()
        globals()["create_dataloader"] = create_dataloader
        globals()["create_reshuffled_dataloaders"] = create_reshuffled_dataloaders
        globals()["get_epoch_steps"] = get_epoch_steps
        globals()["get_warmup_steps"] = get_warmup_steps
        globals()["PAD_TOKEN_ID"] = PAD_TOKEN_ID
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Schema configs
    "SCHEMA_CONFIGS",
    "get_schema_config",
    "get_vocab_size",
    "get_train_tokens",
    # Model configs
    "MODEL_CONFIGS",
    "ARCHITECTURES",
    "get_model_config",
    "estimate_params",
    "compute_batch_config",
    "get_gpu_memory_gb",
    "TARGET_EFFECTIVE_BATCH",
    # Dataloader (requires torch)
    "create_dataloader",
    "create_reshuffled_dataloaders",
    "get_epoch_steps",
    "get_warmup_steps",
    "PAD_TOKEN_ID",
]
