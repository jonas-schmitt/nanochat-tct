"""
Three-way evaluation script for TCT vs UTF8-BPE generation.

Compares:
1. TCT-BPE: Native schema-aware decoding (100% valid by construction)
2. UTF8-BPE: Unconstrained generation (validity varies)
3. UTF8-BPE + XGrammar: FSM-constrained generation (100% valid via XGrammar)

This evaluation addresses RQ2: Does TCT's advantage persist when BPE
is augmented with constrained decoding?

Supports two evaluation modes:
1. Validation set evaluation: Compute loss, perplexity, bits-per-byte on held-out data
2. Generation evaluation: Generate samples and measure validity, speed, diversity

Usage:
    # Validation set evaluation
    python -m scripts.eval_generation \\
        --schema kubernetes \\
        --tct_checkpoint checkpoints/kubernetes_tct_small/ \\
        --utf8_checkpoint checkpoints/kubernetes_utf8_small/ \\
        --eval_validation

    # Generation evaluation with XGrammar
    python -m scripts.eval_generation \\
        --schema kubernetes \\
        --tct_checkpoint checkpoints/kubernetes_tct_small/ \\
        --utf8_checkpoint checkpoints/kubernetes_utf8_small/ \\
        --eval_generation \\
        --xgrammar \\
        --num_samples 100

    # Both evaluations
    python -m scripts.eval_generation \\
        --schema kubernetes \\
        --tct_checkpoint checkpoints/kubernetes_tct_small/ \\
        --utf8_checkpoint checkpoints/kubernetes_utf8_small/ \\
        --eval_validation \\
        --eval_generation \\
        --xgrammar
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt import GPT, GPTConfig
from nanochat.xgrammar_tokenizer import UTF8BPEDecoder

# Import configs (for data loading)
try:
    from configs.jsonl_dataloader import create_dataloader
    from configs.schema_configs import get_schema_config
except ImportError:
    create_dataloader = None
    get_schema_config = None


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class GenerationResult:
    """Result from a single generation."""
    text: str
    tokens: List[int]
    time_seconds: float
    json_valid: bool
    schema_valid: bool
    has_required_fields: bool = False  # apiVersion, kind, metadata for K8s
    error: Optional[str] = None


@dataclass
class ValidationMetrics:
    """Metrics from validation set evaluation."""
    method: str
    num_samples: int
    num_batches: int
    total_tokens: int
    total_bytes: int
    avg_loss: float
    perplexity: float
    bits_per_byte: float
    token_accuracy: float
    top5_accuracy: float
    # Confidence intervals (95%)
    loss_ci: Tuple[float, float] = (0.0, 0.0)
    accuracy_ci: Tuple[float, float] = (0.0, 0.0)


@dataclass
class GenerationMetrics:
    """Aggregated metrics for generation evaluation."""
    method: str
    num_samples: int
    json_validity_rate: float
    schema_validity_rate: float
    field_coverage_rate: float  # Has required fields
    mean_time_seconds: float
    tokens_per_second: float
    mean_tokens: float
    min_tokens: int
    max_tokens: int
    std_tokens: float
    unique_rate: float  # % distinct outputs
    peak_memory_mb: float
    # Confidence intervals (95%)
    json_validity_ci: Tuple[float, float] = (0.0, 0.0)
    schema_validity_ci: Tuple[float, float] = (0.0, 0.0)


# =============================================================================
# Seed Management
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all models."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Statistical Utilities
# =============================================================================

def compute_bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a list of values."""
    if len(values) == 0:
        return (0.0, 0.0)

    values = np.array(values)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))

    lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)

    return (float(lower), float(upper))


def compute_bits_per_byte(
    total_loss: float,
    total_tokens: int,
    total_bytes: int,
) -> float:
    """Compute bits-per-byte for fair cross-tokenizer comparison.

    BPB = (total_loss * total_tokens) / (total_bytes * ln(2))

    This normalizes by raw byte count, making it comparable across
    different tokenization schemes.
    """
    if total_bytes == 0:
        return 0.0
    return (total_loss * total_tokens) / (total_bytes * math.log(2))


def measure_peak_memory() -> float:
    """Measure peak GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def reset_peak_memory():
    """Reset peak memory counter."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# Model Loading (Improved)
# =============================================================================

def load_model_from_checkpoint(
    checkpoint_dir: Path,
    checkpoint_name: str = "best.pt",
    device: str = "cuda",
) -> Tuple[GPT, dict]:
    """Load model from checkpoint directory with config.json.

    Args:
        checkpoint_dir: Directory containing config.json and checkpoint files
        checkpoint_name: Name of checkpoint file (default: best.pt)
        device: Device to load model to

    Returns:
        Tuple of (model, config_dict)

    Expected directory structure:
        checkpoints/{model_tag}/
        ├── config.json       # Model config, schema, tokenizer info
        ├── best.pt           # Best checkpoint (state_dict only)
        └── epoch_XXX.pt      # Epoch checkpoints
    """
    checkpoint_dir = Path(checkpoint_dir)

    # Load config.json
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config_dict = json.load(f)
    else:
        config_dict = {}

    # Get model config from nested structure or flat
    model_config = config_dict.get("model_config", config_dict)

    # Build GPTConfig
    gpt_config = GPTConfig(
        sequence_len=model_config.get("sequence_len", model_config.get("context_size", 2048)),
        vocab_size=model_config.get("vocab_size", 20000),
        n_layer=model_config.get("n_layer", model_config.get("n_layers", 10)),
        n_head=model_config.get("n_head", model_config.get("n_heads", 8)),
        n_kv_head=model_config.get("n_kv_head", model_config.get("n_head", model_config.get("n_heads", 8))),
        n_embd=model_config.get("n_embd", model_config.get("d_model", 512)),
    )

    # Create model
    model = GPT(gpt_config)

    # Load checkpoint
    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        # Try to find any checkpoint
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"  Using checkpoint: {checkpoint_path.name}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove '_orig_mod.' prefix from compiled models
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            cleaned_state_dict[k[10:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, config_dict


def get_tct_tokenizer(schema: str):
    """Get TCT tokenizer module for schema.

    Args:
        schema: Schema name (kubernetes, eslintrc, tsconfig)

    Returns:
        TCT tokenizer module with encode/decode/decode_prefix methods
    """
    if schema == "kubernetes":
        import tct_kubernetes_20k
        return tct_kubernetes_20k
    elif schema == "eslintrc":
        import tct_eslintrc_10k
        return tct_eslintrc_10k
    elif schema == "tsconfig":
        import tct_tsconfig_10k
        return tct_tsconfig_10k
    else:
        raise ValueError(f"Unknown schema: {schema}. Available: kubernetes, eslintrc, tsconfig")


def get_utf8_merge_table_path(schema: str) -> str:
    """Get UTF8-BPE merge table path for schema.

    Args:
        schema: Schema name (kubernetes, eslintrc, tsconfig)

    Returns:
        Path to merge table JSON file
    """
    merge_table_paths = {
        "kubernetes": "bpe-merges/kubernetes-utf8-bpe-matched.json",
        "eslintrc": "bpe-merges/eslintrc-utf8-bpe-matched.json",
        "tsconfig": "bpe-merges/tsconfig-utf8-bpe-matched.json",
    }
    if schema not in merge_table_paths:
        raise ValueError(f"Unknown schema: {schema}")
    return merge_table_paths[schema]


# =============================================================================
# Validation Set Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_validation_set(
    model: GPT,
    data_dir: Path,
    context_size: int,
    batch_size: int = 32,
    num_batches: Optional[int] = None,
    device: str = "cuda",
    method_name: str = "Model",
    raw_bytes: Optional[int] = None,
) -> ValidationMetrics:
    """Evaluate model on validation set.

    Computes:
    - Validation loss (cross-entropy)
    - Perplexity (exp(loss))
    - Bits-per-byte (for fair cross-tokenizer comparison)
    - Token accuracy (next-token prediction)
    - Top-5 accuracy

    Args:
        model: GPT model to evaluate
        data_dir: Directory containing validate.jsonl
        context_size: Context size for data loading
        batch_size: Batch size for evaluation
        num_batches: Number of batches to evaluate (None = all)
        device: Device to evaluate on
        method_name: Name for metrics (e.g., "TCT-BPE", "UTF8-BPE")
        raw_bytes: Total raw bytes in validation set (for bits-per-byte)

    Returns:
        ValidationMetrics with all computed metrics
    """
    if create_dataloader is None:
        raise ImportError("configs.jsonl_dataloader not available")

    print(f"Evaluating {method_name} on validation set...")
    print(f"  Data: {data_dir}")

    # Create validation dataloader
    val_loader = create_dataloader(
        data_dir=data_dir,
        context_size=context_size,
        batch_size=batch_size,
        split="val",
        device=device,
        shuffle=False,
        verbose=False,
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    total_top5_correct = 0
    batch_losses = []
    batch_count = 0

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        # Forward pass
        logits = model(inputs)  # [B, T, V]

        # Compute loss (cross-entropy)
        # Flatten for loss computation
        B, T, V = logits.shape
        logits_flat = logits.view(-1, V)
        targets_flat = targets.view(-1)

        # Mask out padding (targets == -1)
        valid_mask = targets_flat != -1
        valid_logits = logits_flat[valid_mask]
        valid_targets = targets_flat[valid_mask]

        if valid_targets.numel() == 0:
            continue

        # Cross-entropy loss
        loss = F.cross_entropy(valid_logits, valid_targets, reduction='mean')
        total_loss += loss.item() * valid_targets.numel()
        batch_losses.append(loss.item())

        # Token accuracy
        predictions = valid_logits.argmax(dim=-1)
        correct = (predictions == valid_targets).sum().item()
        total_correct += correct

        # Top-5 accuracy
        top5_preds = valid_logits.topk(5, dim=-1).indices
        top5_correct = (top5_preds == valid_targets.unsqueeze(-1)).any(dim=-1).sum().item()
        total_top5_correct += top5_correct

        total_tokens += valid_targets.numel()
        batch_count += 1

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}: loss={loss.item():.4f}")

    # Compute final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    top5_accuracy = total_top5_correct / total_tokens if total_tokens > 0 else 0.0

    # Bits-per-byte (use raw_bytes if provided, otherwise estimate)
    if raw_bytes is None:
        # Estimate: avg ~4 bytes per token (rough approximation)
        raw_bytes = total_tokens * 4
    bpb = compute_bits_per_byte(avg_loss, total_tokens, raw_bytes)

    # Confidence intervals
    loss_ci = compute_bootstrap_ci(batch_losses)
    acc_values = [total_correct / total_tokens] * batch_count  # Approximate per-batch
    accuracy_ci = compute_bootstrap_ci(acc_values)

    print(f"  Loss: {avg_loss:.4f}, PPL: {perplexity:.2f}, BPB: {bpb:.4f}")
    print(f"  Token Acc: {token_accuracy:.2%}, Top-5 Acc: {top5_accuracy:.2%}")

    return ValidationMetrics(
        method=method_name,
        num_samples=len(val_loader.dataset) if hasattr(val_loader, 'dataset') else batch_count * batch_size,
        num_batches=batch_count,
        total_tokens=total_tokens,
        total_bytes=raw_bytes,
        avg_loss=avg_loss,
        perplexity=perplexity,
        bits_per_byte=bpb,
        token_accuracy=token_accuracy,
        top5_accuracy=top5_accuracy,
        loss_ci=loss_ci,
        accuracy_ci=accuracy_ci,
    )


# =============================================================================
# Field Coverage and Uniqueness
# =============================================================================

def check_required_fields(parsed_json: dict, schema: str) -> bool:
    """Check if parsed JSON has required fields for the schema.

    Args:
        parsed_json: Parsed JSON object
        schema: Schema name

    Returns:
        True if all required fields are present
    """
    if schema == "kubernetes":
        # K8s requires: apiVersion, kind, metadata
        required = ["apiVersion", "kind", "metadata"]
        if not all(k in parsed_json for k in required):
            return False
        # metadata.name is also typically required
        if isinstance(parsed_json.get("metadata"), dict):
            if "name" not in parsed_json["metadata"]:
                return False
        return True
    elif schema == "eslintrc":
        # ESLint: must have at least one config key
        valid_keys = ["rules", "env", "extends", "plugins", "parserOptions", "settings"]
        return any(k in parsed_json for k in valid_keys)
    elif schema == "tsconfig":
        # TSConfig: must have compilerOptions or extends
        return "compilerOptions" in parsed_json or "extends" in parsed_json
    else:
        return True  # Unknown schema, assume valid


def compute_uniqueness(texts: List[str]) -> float:
    """Compute percentage of unique outputs.

    Uses SHA256 hash for comparison to handle large texts efficiently.
    """
    if len(texts) == 0:
        return 0.0

    hashes = set()
    for text in texts:
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        hashes.add(h)

    return len(hashes) / len(texts)


# =============================================================================
# Schema paths (relative to tct repo)
# =============================================================================

SCHEMA_PATHS = {
    "kubernetes": "/home/josch/git/tct/schemas/popular/kubernetes.json",
    "eslintrc": "/home/josch/git/tct/schemas/popular/eslintrc.json",
    "tsconfig": "/home/josch/git/tct/schemas/popular/tsconfig.json",
}


def load_schema(schema_name: str) -> dict:
    """Load JSON schema for validation."""
    schema_path = SCHEMA_PATHS.get(schema_name)
    if not schema_path or not os.path.exists(schema_path):
        raise ValueError(f"Schema not found: {schema_name}")

    with open(schema_path) as f:
        return json.load(f)


def load_tct_model(checkpoint_path: str, device: str = "cuda", schema: str = "kubernetes") -> Tuple[GPT, object]:
    """Load TCT-BPE model and tokenizer."""
    print(f"Loading TCT model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_dict = checkpoint.get("config", {})
    config = GPTConfig(
        sequence_len=config_dict.get("context_size", 2048),
        vocab_size=config_dict.get("vocab_size", 20000),
        n_layer=config_dict.get("n_layers", 10),
        n_head=config_dict.get("n_heads", 8),
        n_kv_head=config_dict.get("n_kv_head", config_dict.get("n_heads", 8)),
        n_embd=config_dict.get("d_model", 512),
    )

    model = GPT(config)

    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Import TCT tokenizer based on schema
    try:
        tokenizer = get_tct_tokenizer(schema)
    except ImportError as e:
        print(f"Warning: TCT tokenizer not installed: {e}")
        tokenizer = None

    return model, tokenizer


def validate_json(text: str) -> Tuple[bool, Optional[dict], Optional[str]]:
    """Validate if text is valid JSON."""
    try:
        parsed = json.loads(text)
        return True, parsed, None
    except json.JSONDecodeError as e:
        return False, None, str(e)


def validate_schema(parsed_json: dict, schema: dict) -> Tuple[bool, Optional[str]]:
    """Validate JSON against schema."""
    try:
        import jsonschema
        jsonschema.validate(parsed_json, schema)
        return True, None
    except jsonschema.ValidationError as e:
        return False, str(e)
    except ImportError:
        # Fallback: basic structure check for Kubernetes
        if "apiVersion" in parsed_json and "kind" in parsed_json:
            return True, None
        return False, "Missing apiVersion or kind"


def generate_tct(
    model: GPT,
    tokenizer,
    prompt_tokens: List[int],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> GenerationResult:
    """Generate using TCT-BPE model with streaming decode.

    Uses decode_prefix for truncation-tolerant decoding, which handles
    partial/incomplete token sequences gracefully.
    """
    device = model.get_device()

    start_time = time.perf_counter()
    # Start with token 0 if empty (TCT token 0 is typically "{" or start of structure)
    generated_tokens = list(prompt_tokens) if prompt_tokens else [0]

    for token in model.generate(
        tokens=generated_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=50,
    ):
        generated_tokens.append(token)
        # Stop on PAD token
        if token == tokenizer.vocab_size() - 1:
            break

    elapsed = time.perf_counter() - start_time

    # Decode with TCT using streaming decode (decode_prefix)
    # Returns: (text, consumed, is_complete)
    try:
        text, consumed, is_complete = tokenizer.decode_prefix(generated_tokens)
    except Exception as e:
        text = f"DECODE_ERROR: {e}"
        is_complete = False

    # Validate
    json_valid, parsed, error = validate_json(text)

    # For TCT, schema validity depends on both JSON validity and completeness
    schema_valid = json_valid and is_complete

    return GenerationResult(
        text=text,
        tokens=generated_tokens,
        time_seconds=elapsed,
        json_valid=json_valid,
        schema_valid=schema_valid,
        error=error if not json_valid else None,
    )


def generate_utf8_unconstrained(
    model: GPT,
    tokenizer,
    prompt_tokens: List[int],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> GenerationResult:
    """Generate using UTF8-BPE model without constraints.

    Uses the same tokenizer as XGrammar constrained generation for fair comparison.
    """
    device = model.get_device()

    start_time = time.perf_counter()
    # Start with token corresponding to '{' (byte 123) if empty
    # UTF8-BPE base vocab: tokens 0-255 are raw bytes, so '{' = 123
    generated_tokens = list(prompt_tokens) if prompt_tokens else [123]

    for token in model.generate(
        tokens=generated_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=50,
    ):
        generated_tokens.append(token)
        # Stop on PAD or EOS
        if token == tokenizer.vocab_size() - 1:
            break

    elapsed = time.perf_counter() - start_time

    # Decode with UTF8-BPE
    try:
        text = tokenizer.decode(generated_tokens)
    except Exception as e:
        text = f"DECODE_ERROR: {e}"

    # Validate
    json_valid, parsed, error = validate_json(text)
    schema_valid = False
    if json_valid and parsed:
        # Basic schema check
        schema_valid = "apiVersion" in parsed and "kind" in parsed

    return GenerationResult(
        text=text,
        tokens=generated_tokens,
        time_seconds=elapsed,
        json_valid=json_valid,
        schema_valid=schema_valid,
        error=error,
    )


def generate_utf8_xgrammar(
    model: GPT,
    tokenizer_info,
    compiled_grammar,
    utf8_tokenizer,
    prompt_tokens: List[int],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
) -> GenerationResult:
    """Generate using UTF8-BPE model with XGrammar constraints.

    Args:
        model: The GPT model
        tokenizer_info: XGrammar TokenizerInfo
        compiled_grammar: XGrammar CompiledGrammar
        utf8_tokenizer: UTF8-BPE tokenizer for decoding
        prompt_tokens: Initial tokens (usually empty)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter

    Returns:
        GenerationResult with constrained generation output
    """
    try:
        import xgrammar
        import torch.nn.functional as F

        start_time = time.perf_counter()

        # Create matcher for this generation
        matcher = xgrammar.GrammarMatcher(compiled_grammar)
        bitmask = xgrammar.allocate_token_bitmask(1, tokenizer_info.vocab_size)

        device = model.get_device()
        generated_tokens = list(prompt_tokens)

        for _ in range(max_tokens):
            # Check if generation is complete
            if matcher.is_terminated():
                break

            # Get allowed tokens from grammar
            xgrammar.reset_token_bitmask(bitmask)
            matcher.fill_next_token_bitmask(bitmask)

            # Get model logits
            if len(generated_tokens) == 0:
                # Start with a dummy token for the model
                input_ids = torch.tensor([[0]], device=device)
            else:
                input_ids = torch.tensor([generated_tokens], device=device)

            with torch.no_grad():
                logits = model(input_ids)
                logits = logits[0, -1, :]  # Last position

            # Apply grammar mask
            logits_batch = logits.unsqueeze(0)
            xgrammar.apply_token_bitmask_inplace(logits_batch, bitmask.to(device))
            logits = logits_batch[0]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Top-k sampling
            if top_k is not None and top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Accept token in grammar
            try:
                matcher.accept_token(next_token)
            except Exception as e:
                # Grammar rejection - shouldn't happen with proper masking
                break

            generated_tokens.append(next_token)

        elapsed = time.perf_counter() - start_time

        # Decode tokens
        try:
            text = utf8_tokenizer.decode(generated_tokens)
        except Exception as e:
            text = f"DECODE_ERROR: {e}"

        # Validate JSON
        json_valid, parsed, error = validate_json(text)

        return GenerationResult(
            text=text,
            tokens=generated_tokens,
            time_seconds=elapsed,
            json_valid=json_valid,
            schema_valid=json_valid,  # XGrammar guarantees schema validity
            error=error,
        )

    except ImportError:
        return GenerationResult(
            text="",
            tokens=[],
            time_seconds=0,
            json_valid=False,
            schema_valid=False,
            error="XGrammar not installed. Run: pip install xgrammar",
        )
    except Exception as e:
        import traceback
        return GenerationResult(
            text="",
            tokens=[],
            time_seconds=0,
            json_valid=False,
            schema_valid=False,
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


def compute_generation_metrics(
    results: List[GenerationResult],
    method: str,
    schema: str = "kubernetes",
) -> GenerationMetrics:
    """Compute comprehensive generation metrics."""
    num_samples = len(results)
    if num_samples == 0:
        return GenerationMetrics(
            method=method,
            num_samples=0,
            json_validity_rate=0.0,
            schema_validity_rate=0.0,
            field_coverage_rate=0.0,
            mean_time_seconds=0.0,
            tokens_per_second=0.0,
            mean_tokens=0.0,
            min_tokens=0,
            max_tokens=0,
            std_tokens=0.0,
            unique_rate=0.0,
            peak_memory_mb=0.0,
        )

    # Validity metrics
    json_valid_list = [1.0 if r.json_valid else 0.0 for r in results]
    schema_valid_list = [1.0 if r.schema_valid else 0.0 for r in results]

    # Field coverage (check required fields for schema)
    field_coverage_list = []
    for r in results:
        if r.json_valid:
            try:
                parsed = json.loads(r.text)
                has_fields = check_required_fields(parsed, schema)
                field_coverage_list.append(1.0 if has_fields else 0.0)
            except:
                field_coverage_list.append(0.0)
        else:
            field_coverage_list.append(0.0)

    # Token statistics
    token_counts = [len(r.tokens) for r in results]
    total_time = sum(r.time_seconds for r in results)
    total_tokens = sum(token_counts)

    # Uniqueness
    texts = [r.text for r in results]
    unique_rate = compute_uniqueness(texts)

    # Peak memory
    peak_memory = measure_peak_memory()

    # Confidence intervals
    json_ci = compute_bootstrap_ci(json_valid_list)
    schema_ci = compute_bootstrap_ci(schema_valid_list)

    return GenerationMetrics(
        method=method,
        num_samples=num_samples,
        json_validity_rate=sum(json_valid_list) / num_samples,
        schema_validity_rate=sum(schema_valid_list) / num_samples,
        field_coverage_rate=sum(field_coverage_list) / num_samples if field_coverage_list else 0.0,
        mean_time_seconds=total_time / num_samples,
        tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
        mean_tokens=total_tokens / num_samples,
        min_tokens=min(token_counts) if token_counts else 0,
        max_tokens=max(token_counts) if token_counts else 0,
        std_tokens=float(np.std(token_counts)) if token_counts else 0.0,
        unique_rate=unique_rate,
        peak_memory_mb=peak_memory,
        json_validity_ci=json_ci,
        schema_validity_ci=schema_ci,
    )


def print_validation_metrics(metrics_list: List[ValidationMetrics]):
    """Print validation set metrics comparison table."""
    if not metrics_list:
        return

    num_samples = metrics_list[0].num_samples if metrics_list else 0
    print("\n" + "=" * 100)
    print(f"VALIDATION SET EVALUATION ({num_samples:,} samples)")
    print("=" * 100)

    headers = ["Method", "Loss", "PPL", "Bits/Byte", "Tok-Acc", "Top5-Acc", "Tokens"]
    row_format = "{:<20} {:>8} {:>8} {:>10} {:>10} {:>10} {:>12}"

    print(row_format.format(*headers))
    print("-" * 100)

    for m in metrics_list:
        print(row_format.format(
            m.method,
            f"{m.avg_loss:.4f}",
            f"{m.perplexity:.2f}",
            f"{m.bits_per_byte:.4f}",
            f"{m.token_accuracy:.1%}",
            f"{m.top5_accuracy:.1%}",
            f"{m.total_tokens:,}",
        ))

    print("=" * 100)


def print_generation_metrics(metrics_list: List[GenerationMetrics], seed: int = 42):
    """Print generation metrics comparison table."""
    if not metrics_list:
        return

    num_samples = metrics_list[0].num_samples if metrics_list else 0
    print("\n" + "=" * 110)
    print(f"GENERATION EVALUATION ({num_samples} samples, seed={seed})")
    print("=" * 110)

    headers = ["Method", "JSON%", "Schema%", "Fields%", "Mean-Tok", "Tok/s", "Unique%", "Memory"]
    row_format = "{:<22} {:>8} {:>9} {:>9} {:>10} {:>8} {:>9} {:>10}"

    print(row_format.format(*headers))
    print("-" * 110)

    for m in metrics_list:
        print(row_format.format(
            m.method,
            f"{m.json_validity_rate:.1%}",
            f"{m.schema_validity_rate:.1%}",
            f"{m.field_coverage_rate:.1%}",
            f"{m.mean_tokens:.1f}",
            f"{m.tokens_per_second:.1f}",
            f"{m.unique_rate:.1%}",
            f"{m.peak_memory_mb:.0f}MB",
        ))

    print("=" * 110)
    print("Fields%: Has all required schema fields | Unique%: Distinct outputs")


def save_sample_outputs(
    results: List[GenerationResult],
    output_dir: Path,
    method_name: str,
    n: int = 10,
):
    """Save sample generated texts for qualitative analysis.

    Saves both best (valid) and worst (invalid) examples.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Separate valid and invalid
    valid = [r for r in results if r.json_valid]
    invalid = [r for r in results if not r.json_valid]

    # Save valid samples
    valid_samples = valid[:n]
    for i, r in enumerate(valid_samples):
        filepath = output_dir / f"{method_name}_valid_{i:03d}.json"
        with open(filepath, 'w') as f:
            f.write(r.text)

    # Save invalid samples (for error analysis)
    invalid_samples = invalid[:n]
    for i, r in enumerate(invalid_samples):
        filepath = output_dir / f"{method_name}_invalid_{i:03d}.txt"
        with open(filepath, 'w') as f:
            f.write(f"ERROR: {r.error}\n\n")
            f.write(r.text)

    print(f"  Saved {len(valid_samples)} valid and {len(invalid_samples)} invalid samples to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Three-way evaluation for TCT vs UTF8-BPE generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validation set evaluation only
  python -m scripts.eval_generation --schema kubernetes --eval_validation \\
      --tct_checkpoint checkpoints/k8s_tct_small/ \\
      --utf8_checkpoint checkpoints/k8s_utf8_small/

  # Generation evaluation with XGrammar
  python -m scripts.eval_generation --schema kubernetes --eval_generation \\
      --tct_checkpoint checkpoints/k8s_tct_small/ \\
      --utf8_checkpoint checkpoints/k8s_utf8_small/ \\
      --xgrammar --num_samples 100

  # Both evaluations
  python -m scripts.eval_generation --schema kubernetes \\
      --eval_validation --eval_generation --xgrammar \\
      --tct_checkpoint checkpoints/k8s_tct_small/ \\
      --utf8_checkpoint checkpoints/k8s_utf8_small/
        """
    )

    # Checkpoint arguments
    parser.add_argument("--tct_checkpoint", help="Path to TCT-BPE checkpoint directory")
    parser.add_argument("--utf8_checkpoint", help="Path to UTF8-BPE checkpoint directory")
    parser.add_argument("--checkpoint_name", default="best.pt",
                        help="Checkpoint filename within directory (default: best.pt)")

    # Schema and tokenizer
    parser.add_argument("--schema", default="kubernetes",
                        choices=["kubernetes", "eslintrc", "tsconfig"],
                        help="Schema to evaluate")
    parser.add_argument("--utf8_merge_table",
                        help="Path to UTF8-BPE merge table (for XGrammar). Auto-detected if not provided.")

    # Evaluation modes
    parser.add_argument("--eval_validation", action="store_true",
                        help="Run validation set evaluation (loss, perplexity, bits-per-byte)")
    parser.add_argument("--eval_generation", action="store_true",
                        help="Run generation evaluation (validity, speed, diversity)")
    parser.add_argument("--xgrammar", action="store_true",
                        help="Include XGrammar-constrained generation (requires --eval_generation)")

    # Validation parameters
    parser.add_argument("--num_val_batches", type=int, default=None,
                        help="Number of validation batches (default: all)")
    parser.add_argument("--val_batch_size", type=int, default=32,
                        help="Validation batch size (default: 32)")

    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate (default: 10)")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens per generation (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    # Output
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--save_samples", help="Directory to save sample outputs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")

    args = parser.parse_args()

    # Default: if neither eval mode specified, run both
    if not args.eval_validation and not args.eval_generation:
        args.eval_validation = True
        args.eval_generation = True

    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    # Get schema configuration
    schema_config = None
    if get_schema_config is not None:
        try:
            schema_config = get_schema_config(args.schema)
            print(f"\nSchema: {args.schema}")
            print(f"  Context size: {schema_config['context_size']}")
            print(f"  TCT vocab: {schema_config['tct_vocab_size']}")
            print(f"  UTF8 vocab: {schema_config['utf8_vocab_size']}")
        except Exception as e:
            print(f"Warning: Could not load schema config: {e}")
            schema_config = None

    # Load JSON schema for validation
    print(f"\nLoading {args.schema} JSON schema...")
    json_schema = load_schema(args.schema)

    # Auto-detect merge table if not provided
    if args.utf8_merge_table is None:
        args.utf8_merge_table = get_utf8_merge_table_path(args.schema)

    # Results storage
    validation_metrics = []
    generation_metrics = []
    all_results = {}

    # ==========================================================================
    # VALIDATION SET EVALUATION
    # ==========================================================================
    if args.eval_validation:
        print("\n" + "=" * 80)
        print("VALIDATION SET EVALUATION")
        print("=" * 80)

        context_size = schema_config["context_size"] if schema_config else 2048

        # Evaluate TCT model
        if args.tct_checkpoint:
            tct_dir = Path(args.tct_checkpoint)
            model, config = load_model_from_checkpoint(tct_dir, args.checkpoint_name, args.device)

            # Get data path
            if schema_config:
                data_dir = schema_config["data_path_tct"]
            else:
                data_dir = Path.home() / "Desktop" / "data" / f"{args.schema}-tct-bpe"

            if data_dir.exists():
                metrics = evaluate_validation_set(
                    model=model,
                    data_dir=data_dir,
                    context_size=context_size,
                    batch_size=args.val_batch_size,
                    num_batches=args.num_val_batches,
                    device=args.device,
                    method_name="TCT-BPE",
                )
                validation_metrics.append(metrics)
            else:
                print(f"  Warning: TCT data not found at {data_dir}")

            del model
            torch.cuda.empty_cache()

        # Evaluate UTF8 model
        if args.utf8_checkpoint:
            utf8_dir = Path(args.utf8_checkpoint)
            model, config = load_model_from_checkpoint(utf8_dir, args.checkpoint_name, args.device)

            # Get data path
            if schema_config:
                data_dir = schema_config["data_path_utf8"]
            else:
                data_dir = Path.home() / "Desktop" / "data" / f"{args.schema}-utf8-bpe"

            if data_dir.exists():
                metrics = evaluate_validation_set(
                    model=model,
                    data_dir=data_dir,
                    context_size=context_size,
                    batch_size=args.val_batch_size,
                    num_batches=args.num_val_batches,
                    device=args.device,
                    method_name="UTF8-BPE",
                )
                validation_metrics.append(metrics)
            else:
                print(f"  Warning: UTF8 data not found at {data_dir}")

            del model
            torch.cuda.empty_cache()

        # Print validation results
        print_validation_metrics(validation_metrics)

    # ==========================================================================
    # GENERATION EVALUATION
    # ==========================================================================
    if args.eval_generation:
        print("\n" + "=" * 80)
        print("GENERATION EVALUATION")
        print("=" * 80)

        # Reset seed before generation for fair comparison
        set_seed(args.seed)

        # Evaluate TCT-BPE
        if args.tct_checkpoint:
            print("\n" + "-" * 40)
            print("EVALUATING: TCT-BPE")
            print("-" * 40)

            reset_peak_memory()
            tct_dir = Path(args.tct_checkpoint)
            model, config = load_model_from_checkpoint(tct_dir, args.checkpoint_name, args.device)

            try:
                tokenizer = get_tct_tokenizer(args.schema)
            except ImportError as e:
                print(f"  ERROR: TCT tokenizer not available: {e}")
                raise

            tct_results = []
            for i in range(args.num_samples):
                result = generate_tct(
                    model, tokenizer, [],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                tct_results.append(result)
                print(f"  Sample {i+1}/{args.num_samples}: JSON={result.json_valid}, Schema={result.schema_valid}")

            metrics = compute_generation_metrics(tct_results, "TCT-BPE", args.schema)
            generation_metrics.append(metrics)
            all_results["tct"] = tct_results

            if args.save_samples:
                save_sample_outputs(tct_results, Path(args.save_samples), "tct", n=5)

            del model
            torch.cuda.empty_cache()

        # Evaluate UTF8-BPE (unconstrained)
        if args.utf8_checkpoint:
            print("\n" + "-" * 40)
            print("EVALUATING: UTF8-BPE (unconstrained)")
            print("-" * 40)

            # Reset seed for fair comparison
            set_seed(args.seed)
            reset_peak_memory()

            utf8_dir = Path(args.utf8_checkpoint)
            model, config = load_model_from_checkpoint(utf8_dir, args.checkpoint_name, args.device)

            # Use UTF8BPEDecoder for consistent tokenization with XGrammar
            print(f"  Using UTF8BPEDecoder from {args.utf8_merge_table}")
            utf8_decoder = UTF8BPEDecoder(args.utf8_merge_table)

            utf8_results = []
            for i in range(args.num_samples):
                result = generate_utf8_unconstrained(
                    model, utf8_decoder, [],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                utf8_results.append(result)
                print(f"  Sample {i+1}/{args.num_samples}: JSON={result.json_valid}, Schema={result.schema_valid}")

            metrics = compute_generation_metrics(utf8_results, "UTF8-BPE", args.schema)
            generation_metrics.append(metrics)
            all_results["utf8"] = utf8_results

            if args.save_samples:
                save_sample_outputs(utf8_results, Path(args.save_samples), "utf8", n=5)

            del model
            torch.cuda.empty_cache()

        # Evaluate UTF8-BPE + XGrammar (constrained)
        if args.xgrammar and args.utf8_checkpoint:
            print("\n" + "-" * 40)
            print("EVALUATING: UTF8-BPE + XGrammar")
            print("-" * 40)

            try:
                import xgrammar
                from nanochat.xgrammar_tokenizer import (
                    build_xgrammar_tokenizer_info,
                    compile_json_schema_grammar,
                )

                # Reset seed for fair comparison
                set_seed(args.seed)
                reset_peak_memory()

                # Build tokenizer info
                print(f"Building XGrammar tokenizer from {args.utf8_merge_table}...")
                tokenizer_info = build_xgrammar_tokenizer_info(args.utf8_merge_table)
                print(f"  Vocab size: {tokenizer_info.vocab_size}")

                # Compile grammar
                print(f"Compiling grammar for {args.schema} schema...")
                compiled_grammar = compile_json_schema_grammar(tokenizer_info, json_schema)
                print("  Grammar compiled successfully")

                # Load model
                utf8_dir = Path(args.utf8_checkpoint)
                model, config = load_model_from_checkpoint(utf8_dir, args.checkpoint_name, args.device)

                # Use UTF8BPEDecoder for consistent decoding
                utf8_decoder = UTF8BPEDecoder(args.utf8_merge_table)

                xgrammar_results = []
                for i in range(args.num_samples):
                    result = generate_utf8_xgrammar(
                        model,
                        tokenizer_info,
                        compiled_grammar,
                        utf8_decoder,
                        prompt_tokens=[],
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    xgrammar_results.append(result)
                    status = "ERROR" if result.error else f"JSON={result.json_valid}"
                    print(f"  Sample {i+1}/{args.num_samples}: {status}")

                metrics = compute_generation_metrics(xgrammar_results, "UTF8-BPE + XGrammar", args.schema)
                generation_metrics.append(metrics)
                all_results["xgrammar"] = xgrammar_results

                if args.save_samples:
                    save_sample_outputs(xgrammar_results, Path(args.save_samples), "xgrammar", n=5)

                del model
                torch.cuda.empty_cache()

            except ImportError as e:
                print(f"  ERROR: {e}")
                print("  XGrammar not available. Install with: pip install xgrammar")
            except Exception as e:
                import traceback
                print(f"  ERROR: {type(e).__name__}: {e}")
                traceback.print_exc()

        # Print generation results
        print_generation_metrics(generation_metrics, args.seed)

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    if args.output:
        results_dict = {
            "schema": args.schema,
            "seed": args.seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "num_samples": args.num_samples,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "val_batch_size": args.val_batch_size,
                "num_val_batches": args.num_val_batches,
            },
            "validation_metrics": [
                {
                    "method": m.method,
                    "num_samples": m.num_samples,
                    "num_batches": m.num_batches,
                    "total_tokens": m.total_tokens,
                    "total_bytes": m.total_bytes,
                    "avg_loss": m.avg_loss,
                    "perplexity": m.perplexity,
                    "bits_per_byte": m.bits_per_byte,
                    "token_accuracy": m.token_accuracy,
                    "top5_accuracy": m.top5_accuracy,
                    "loss_ci": m.loss_ci,
                    "accuracy_ci": m.accuracy_ci,
                }
                for m in validation_metrics
            ],
            "generation_metrics": [
                {
                    "method": m.method,
                    "num_samples": m.num_samples,
                    "json_validity_rate": m.json_validity_rate,
                    "schema_validity_rate": m.schema_validity_rate,
                    "field_coverage_rate": m.field_coverage_rate,
                    "mean_time_seconds": m.mean_time_seconds,
                    "tokens_per_second": m.tokens_per_second,
                    "mean_tokens": m.mean_tokens,
                    "min_tokens": m.min_tokens,
                    "max_tokens": m.max_tokens,
                    "std_tokens": m.std_tokens,
                    "unique_rate": m.unique_rate,
                    "peak_memory_mb": m.peak_memory_mb,
                    "json_validity_ci": m.json_validity_ci,
                    "schema_validity_ci": m.schema_validity_ci,
                }
                for m in generation_metrics
            ],
        }

        with open(args.output, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
