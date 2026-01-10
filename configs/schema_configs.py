"""
Schema-specific configurations for TCT experiments.

Each schema has:
- context_size: 2048 for all (standardized)
- vocab sizes: Read dynamically from data metadata.json (+ 1 for pad token)
- training data statistics
- data directory paths

Data from TRAINING_DATA_REPORT.md (2025-12-25)
"""

import json
from pathlib import Path

# Default data root (can be overridden)
DEFAULT_DATA_ROOT = Path.home() / "Desktop" / "data"


def _read_vocab_size_from_data(data_dir: Path) -> int | None:
    """Read vocab size from data metadata.json, adding 1 for pad token.

    The metadata stores base_vocab_size (max token + 1), and we add 1 for
    the pad token (token 0) that training uses.
    """
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
        # metadata stores base_vocab_size = max_token + 1
        # We add 1 for the pad token used in training
        return metadata.get("base_vocab_size", 0) + 1
    except (json.JSONDecodeError, KeyError):
        return None

SCHEMA_CONFIGS = {
    # =========================================================================
    # TSCONFIG - Base encoding (no BPE), most data available
    # =========================================================================
    "tsconfig": {
        # Context: 2048 covers all sequences (P99=1108)
        "context_size": 2048,
        "default_epochs": 150,  # Uniform epochs for fair comparison (best.pt for early stopping)

        # Vocabulary sizes (includes pad token)
        "tct_vocab_size": 258,      # Base encoding (257) + pad
        "utf8_vocab_size": 277,     # Minimal BPE (276) + pad

        # Training data statistics (95/5 split)
        "train_files": 337_991,  # 355,780 × 0.95
        "validate_files": 17_789,  # 355,780 × 0.05
        "total_files": 355_780,
        "train_tokens_tct": 117_000_000,
        "train_tokens_utf8": 103_000_000,
        "avg_tokens": 327,

        # Percentiles
        "p50": 252,
        "p90": 473,
        "p95": 561,
        "p99": 1108,

        # Data directories (bpe-matched baseline)
        "data_dir_tct": "tsconfig-tct-base",
        "data_dir_utf8": "tsconfig-utf8-base-matched",
        # Data directories (o200k-matched baseline)
        "data_dir_tct_o200k": "tsconfig-tct-o200k-matched",
        "data_dir_utf8_o200k": "tsconfig-utf8-o200k-matched",

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
        "default_epochs": 150,  # Uniform epochs for fair comparison (best.pt for early stopping)

        # Vocabulary sizes (includes pad token)
        "tct_vocab_size": 500,      # BPE-500 (499) + pad
        "utf8_vocab_size": 717,     # UTF8 BPE (716 from eslintrc-utf8-bpe-500) + pad

        # Training data statistics (95/5 split)
        "train_files": 120_859,  # 127,221 × 0.95
        "validate_files": 6_362,  # 127,221 × 0.05
        "total_files": 127_221,
        "train_tokens_tct": 21_500_000,
        "train_tokens_utf8": 21_600_000,
        "avg_tokens": 188,

        # Percentiles
        "p50": 93,
        "p90": 404,
        "p95": 713,
        "p99": 1858,

        # Data directories (bpe-matched baseline)
        "data_dir_tct": "eslintrc-tct-bpe-500",
        "data_dir_utf8": "eslintrc-utf8-bpe-500",
        # Data directories (o200k-matched baseline)
        "data_dir_tct_o200k": "eslintrc-tct-o200k-matched",
        "data_dir_utf8_o200k": "eslintrc-utf8-o200k-matched",

        # Schema info
        "complexity": "medium",
        "description": "ESLint linter configuration files (BPE-500)",
    },

    # =========================================================================
    # KUBERNETES - BPE-1k, good balance of compression and coverage
    # =========================================================================
    "kubernetes": {
        # Context: 2048 covers 97.3% of sequences (P95=1006)
        "context_size": 2048,
        "default_epochs": 150,  # Uniform epochs for fair comparison (best.pt for early stopping)

        # Vocabulary sizes (includes pad token)
        "tct_vocab_size": 1000,     # BPE-1k (999) + pad
        "utf8_vocab_size": 1527,    # UTF8 BPE (256 base + 1270 merges + 1 pad)

        # Training data statistics (95/5 split)
        "train_files": 234_117,  # 246,439 × 0.95
        "validate_files": 12_322,  # 246,439 × 0.05
        "total_files": 246_439,
        "train_tokens_tct": 114_000_000,  # 514 avg * 221k files
        "train_tokens_utf8": 114_000_000,
        "avg_tokens": 514,

        # Percentiles (measured)
        "p50": 92,
        "p90": 700,
        "p95": 1006,
        "p99": 7271,

        # Data directories (bpe-matched baseline)
        "data_dir_tct": "kubernetes-tct-bpe-1k",
        "data_dir_utf8": "kubernetes-utf8-bpe-1k",
        # Data directories (o200k-matched baseline)
        "data_dir_tct_o200k": "kubernetes-tct-o200k-matched",
        "data_dir_utf8_o200k": "kubernetes-utf8-o200k-matched",

        # Schema info
        "complexity": "high",
        "description": "Kubernetes manifest files (BPE-1k, 25% util, 97% coverage)",
    },
}


def get_schema_config(schema: str, data_root: Path = None) -> dict:
    """
    Get schema configuration with resolved data paths.

    Vocab sizes are read dynamically from data metadata.json when the data exists,
    ensuring the model vocab matches the actual tokenized data.

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

    # Resolve data paths (bpe-matched baseline)
    config["data_path_tct"] = data_root / config["data_dir_tct"] if config.get("data_dir_tct") else None
    config["data_path_utf8"] = data_root / config["data_dir_utf8"] if config.get("data_dir_utf8") else None
    # Resolve data paths (o200k-matched baseline)
    config["data_path_tct_o200k"] = data_root / config["data_dir_tct_o200k"] if config.get("data_dir_tct_o200k") else None
    config["data_path_utf8_o200k"] = data_root / config["data_dir_utf8_o200k"] if config.get("data_dir_utf8_o200k") else None

    # Override vocab sizes from data metadata when available
    # This ensures model vocab matches actual tokenized data
    if config["data_path_tct"]:
        actual_vocab = _read_vocab_size_from_data(config["data_path_tct"])
        if actual_vocab is not None:
            if actual_vocab != config["tct_vocab_size"]:
                print(f"[INFO] {schema} TCT vocab from metadata: {actual_vocab} (config had {config['tct_vocab_size']})")
            config["tct_vocab_size"] = actual_vocab

    if config["data_path_utf8"]:
        actual_vocab = _read_vocab_size_from_data(config["data_path_utf8"])
        if actual_vocab is not None:
            if actual_vocab != config["utf8_vocab_size"]:
                print(f"[INFO] {schema} UTF8 vocab from metadata: {actual_vocab} (config had {config['utf8_vocab_size']})")
            config["utf8_vocab_size"] = actual_vocab

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
