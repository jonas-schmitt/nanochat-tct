"""
Schema-specific configurations for TCT experiments.

Each schema has:
- context_size: Recommended context length (based on P99 percentile)
- vocab sizes: For both TCT-BPE and UTF8-BPE tokenizers
- training data statistics
- data directory paths

Data from TRAINING_DATA_REPORT.md (2025-12-23)
"""

from pathlib import Path

# Default data root (can be overridden)
DEFAULT_DATA_ROOT = Path.home() / "Desktop" / "data"

SCHEMA_CONFIGS = {
    "tsconfig": {
        # Context: 256 covers 99.3%+ of sequences (P99=148-159)
        "context_size": 256,
        "default_epochs": 100,

        # Vocabulary sizes (includes PAD token 0)
        "tct_vocab_size": 10000,
        "utf8_vocab_size": 16197,

        # Training data statistics
        "train_files": 320_202,
        "validate_files": 35_578,
        "total_files": 355_780,
        "train_tokens_tct": 7_877_907,
        "train_tokens_utf8": 7_889_868,
        "avg_tokens": 24.75,

        # Percentiles (for context selection)
        "p50": 11,
        "p90": 31,
        "p95": 47,
        "p99": 159,

        # Data directories
        "data_dir_tct": "tsconfig-tct-bpe-10k",
        "data_dir_utf8": "tsconfig-utf8-bpe-10k",

        # Schema complexity (for documentation)
        "complexity": "low",
        "description": "TypeScript compiler configuration files",
    },

    "tsconfig-base": {
        # Experiment: No BPE compression, longer sequences
        # Context: 2048 to fit P99=1108 with room to spare
        "context_size": 2048,
        "default_epochs": 150,

        # Base vocabulary (no BPE) - just TCT base tokens
        "tct_vocab_size": 257,
        "utf8_vocab_size": None,  # No UTF8 variant for base

        # Training data statistics (same files, different encoding)
        "train_files": 337_991,
        "validate_files": 17_789,
        "total_files": 355_780,
        "train_tokens_tct": 116_595_633,  # Much more tokens (no compression)
        "train_tokens_utf8": 0,
        "avg_tokens": 328,

        # Percentiles (base encoding, no BPE)
        "p50": 252,
        "p90": 473,
        "p95": 561,
        "p99": 1108,

        # Data directories
        "data_dir_tct": "tsconfig-tct-base",
        "data_dir_utf8": None,

        # Schema complexity
        "complexity": "low",
        "description": "TypeScript config - NO BPE (base encoding experiment)",
    },

    "eslintrc": {
        # Context: 512 covers 99.6%+ of sequences (P99=341-345)
        "context_size": 512,
        "default_epochs": 150,

        # Vocabulary sizes (includes PAD token 0)
        "tct_vocab_size": 10000,
        "utf8_vocab_size": 18337,

        # Training data statistics
        "train_files": 114_500,
        "validate_files": 12_722,
        "total_files": 127_222,
        "train_tokens_tct": 4_016_279,
        "train_tokens_utf8": 4_019_742,
        "avg_tokens": 35.19,

        # Percentiles
        "p50": 15,
        "p90": 74,
        "p95": 137,
        "p99": 345,

        # Data directories
        "data_dir_tct": "eslintrc-tct-bpe-10k",
        "data_dir_utf8": "eslintrc-utf8-bpe-10k",

        # Schema complexity
        "complexity": "medium",
        "description": "ESLint linter configuration files",
    },

    "kubernetes": {
        # Context: 2048 covers 99%+ of sequences (P99=2622-2653)
        "context_size": 2048,
        "default_epochs": 200,

        # Vocabulary sizes (includes PAD token 0)
        "tct_vocab_size": 20000,
        "utf8_vocab_size": 23887,

        # Training data statistics
        "train_files": 221_795,
        "validate_files": 24_643,
        "total_files": 246_438,
        "train_tokens_tct": 42_000_000,  # approximate
        "train_tokens_utf8": 42_000_000,  # approximate
        "avg_tokens": 189.07,

        # Percentiles
        "p50": 37,
        "p90": 205,
        "p95": 418,
        "p99": 2653,

        # Data directories
        "data_dir_tct": "kubernetes-tct-bpe",
        "data_dir_utf8": "kubernetes-utf8-bpe",

        # Schema complexity
        "complexity": "high",
        "description": "Kubernetes manifest files (YAML/JSON)",
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
    print("Context lengths based on P99 percentile for 99%+ coverage.")
    print()


if __name__ == "__main__":
    print_schema_summary()
