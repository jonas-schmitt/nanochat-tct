"""
Schema-specific configurations for TCT experiments.

Each schema has:
- context_size: 2048 for all (standardized)
- vocab sizes: For both TCT and UTF8 tokenizers
- training data statistics
- data directory paths

Data from TRAINING_DATA_REPORT.md (2025-12-25)
"""

from pathlib import Path

# Default data root (can be overridden)
DEFAULT_DATA_ROOT = Path.home() / "Desktop" / "data"

SCHEMA_CONFIGS = {
    # =========================================================================
    # TSCONFIG - Base encoding (no BPE), most data available
    # =========================================================================
    "tsconfig": {
        # Context: 2048 covers all sequences (P99=1108)
        "context_size": 2048,
        "default_epochs": 50,   # Converges very fast (ppl 1.19 at epoch 3)

        # Vocabulary sizes (includes pad token)
        "tct_vocab_size": 258,      # Base encoding (257) + pad
        "utf8_vocab_size": 277,     # Minimal BPE (276) + pad

        # Training data statistics
        "train_files": 320_202,
        "validate_files": 35_578,
        "total_files": 355_780,
        "train_tokens_tct": 117_000_000,
        "train_tokens_utf8": 103_000_000,
        "avg_tokens": 327,

        # Percentiles
        "p50": 252,
        "p90": 473,
        "p95": 561,
        "p99": 1108,

        # Data directories
        "data_dir_tct": "tsconfig-tct-base",
        "data_dir_utf8": "tsconfig-utf8-base-matched",

        # Schema info
        "complexity": "low",
        "description": "TypeScript compiler configuration files (base encoding)",
    },

    # =========================================================================
    # ESLINTRC - BPE-500, limited data
    # =========================================================================
    "eslintrc": {
        # Context: 2048 covers all sequences (P99=1858)
        "context_size": 2048,
        "default_epochs": 100,  # Fast convergence (ppl 1.54 at epoch 32)

        # Vocabulary sizes (includes pad token)
        "tct_vocab_size": 500,      # BPE-500 (499) + pad
        "utf8_vocab_size": 727,     # UTF8 BPE (726) + pad

        # Training data statistics
        "train_files": 114_499,
        "validate_files": 12_722,
        "total_files": 127_221,
        "train_tokens_tct": 21_500_000,
        "train_tokens_utf8": 21_600_000,
        "avg_tokens": 188,

        # Percentiles
        "p50": 93,
        "p90": 404,
        "p95": 713,
        "p99": 1858,

        # Data directories
        "data_dir_tct": "eslintrc-tct-bpe-500",
        "data_dir_utf8": "eslintrc-utf8-bpe-500",

        # Schema info
        "complexity": "medium",
        "description": "ESLint linter configuration files (BPE-500)",
    },

    # =========================================================================
    # KUBERNETES - BPE-20k, complex manifests
    # =========================================================================
    "kubernetes": {
        # Context: 2048 covers ~99% of sequences (P99=2622, some truncation)
        "context_size": 2048,
        "default_epochs": 200,  # Larger dataset, more epochs needed

        # Vocabulary sizes (includes pad token)
        "tct_vocab_size": 20_000,   # BPE-20k (19999) + pad
        "utf8_vocab_size": 23_887,  # UTF8 BPE (23886) + pad

        # Training data statistics
        "train_files": 221_794,
        "validate_files": 24_644,
        "total_files": 246_438,
        "train_tokens_tct": 42_000_000,
        "train_tokens_utf8": 42_000_000,
        "avg_tokens": 189,

        # Percentiles
        "p50": 37,
        "p90": 205,
        "p95": 418,
        "p99": 2622,

        # Data directories
        "data_dir_tct": "kubernetes-tct-bpe",
        "data_dir_utf8": "kubernetes-utf8-bpe",

        # Schema info
        "complexity": "high",
        "description": "Kubernetes manifest files (BPE-20k)",
    },
}


def get_schema_config(schema: str, data_root: Path = None) -> dict:
    """
    Get schema configuration with resolved data paths.

    Args:
        schema: Schema name ("tsconfig", "eslintrc", "kubernetes")
        data_root: Optional data root directory (default: ~/Desktop/data)

    Returns:
        Configuration dictionary with resolved paths

    Raises:
        ValueError: If schema not found
    """
    if schema not in SCHEMA_CONFIGS:
        available = ", ".join(SCHEMA_CONFIGS.keys())
        raise ValueError(f"Unknown schema: '{schema}'. Available: {available}")

    config = SCHEMA_CONFIGS[schema].copy()
    data_root = Path(data_root) if data_root else DEFAULT_DATA_ROOT

    # Resolve data paths
    config["data_path_tct"] = data_root / config["data_dir_tct"] if config.get("data_dir_tct") else None
    config["data_path_utf8"] = data_root / config["data_dir_utf8"] if config.get("data_dir_utf8") else None

    return config


def get_vocab_size(schema: str, tokenizer: str) -> int:
    """
    Get vocabulary size for schema and tokenizer type.

    Args:
        schema: Schema name
        tokenizer: "tct" or "utf8"

    Returns:
        Vocabulary size
    """
    config = SCHEMA_CONFIGS[schema]
    if tokenizer == "tct":
        return config["tct_vocab_size"]
    elif tokenizer == "utf8":
        return config["utf8_vocab_size"]
    else:
        raise ValueError(f"Unknown tokenizer: '{tokenizer}'. Use 'tct' or 'utf8'")


def get_train_tokens(schema: str, tokenizer: str) -> int:
    """Get number of training tokens for schema and tokenizer."""
    config = SCHEMA_CONFIGS[schema]
    if tokenizer == "tct":
        return config["train_tokens_tct"]
    elif tokenizer == "utf8":
        return config["train_tokens_utf8"]
    else:
        raise ValueError(f"Unknown tokenizer: '{tokenizer}'. Use 'tct' or 'utf8'")


def print_schema_summary():
    """Print summary of all schema configurations."""
    print("=" * 80)
    print("Schema Configurations for TCT Experiments")
    print("=" * 80)
    print()

    print(f"{'Schema':<12} {'Context':<8} {'Epochs':<8} {'TCT Vocab':<10} {'UTF8 Vocab':<11} {'Files':<10} {'Complexity'}")
    print("-" * 80)

    for name, cfg in SCHEMA_CONFIGS.items():
        print(f"{name:<12} {cfg['context_size']:<8} {cfg['default_epochs']:<8} "
              f"{cfg['tct_vocab_size']:<10} {cfg['utf8_vocab_size']:<11} "
              f"{cfg['total_files']:<10,} {cfg['complexity']}")

    print()
    print("All schemas use context=2048 for standardized comparison.")
    print()


if __name__ == "__main__":
    print_schema_summary()
