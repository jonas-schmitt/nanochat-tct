#!/usr/bin/env python3
"""
ICML 2026 Evaluation Script: TCT vs BPE+XGrammar

Runs evaluation metrics:
1. BPB (Bits-per-Byte) - Information-theoretic comparison with XGrammar masking
2. Generation Quality - Field value distribution comparison

Usage:
    # Full evaluation for kubernetes
    python -m scripts.eval_icml \
        --schema kubernetes \
        --tct_checkpoint checkpoints/kubernetes_tct_small/ \
        --utf8_checkpoint checkpoints/kubernetes_utf8_small/ \
        --num_samples 1000 \
        --output results/icml/kubernetes.json

    # BPB only (faster)
    python -m scripts.eval_icml \
        --schema kubernetes \
        --tct_checkpoint checkpoints/kubernetes_tct_small/ \
        --utf8_checkpoint checkpoints/kubernetes_utf8_small/ \
        --bpb_only \
        --output results/icml/kubernetes_bpb.json

    # Generation quality only
    python -m scripts.eval_icml \
        --schema kubernetes \
        --tct_checkpoint checkpoints/kubernetes_tct_small/ \
        --utf8_checkpoint checkpoints/kubernetes_utf8_small/ \
        --generation_only \
        --output results/icml/kubernetes_gen.json

    # Specific epochs
    python -m scripts.eval_icml \
        --schema kubernetes \
        --tct_checkpoint checkpoints/kubernetes_tct_small/ --tct_epoch 30 \
        --utf8_checkpoint checkpoints/kubernetes_utf8_small/ --utf8_epoch 30 \
        --bpb_only
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_generation_batch_size(model, max_tokens: int, verbose: bool = False) -> int:
    """Compute optimal batch size for generation based on model size and GPU memory.

    Uses similar approach to training batch sizing in configs/model_configs.py:
    - Reference batch sizes for 24GB VRAM
    - Scale proportionally to actual GPU memory
    - Account for KV-cache memory growth during generation

    Formula:
        KV-cache per sample = 2 * n_layers * seq_len * n_kv_heads * head_dim * 4 bytes
        Available = GPU_memory - model_weights - safety_margin
        max_batch = Available / KV_cache_per_sample

    Args:
        model: The GPT model (to get config and device)
        max_tokens: Maximum sequence length for generation
        verbose: Print calculation details

    Returns:
        Optimal batch size (power of 2, min 1, max 512)
    """
    import torch

    cfg = model.config
    device = next(model.parameters()).device

    # Get GPU memory
    if device.type == "cuda":
        gpu_memory = torch.cuda.get_device_properties(device).total_memory
        # Get current allocated memory
        allocated = torch.cuda.memory_allocated(device)
    else:
        # CPU fallback - use reasonable batch size
        return 16

    # Calculate model memory (already allocated)
    model_params = sum(p.numel() * p.element_size() for p in model.parameters())

    # Calculate KV-cache memory per sample
    # KV cache shape: [n_layers, 2, batch, n_kv_heads, seq_len, head_dim]
    # Each entry is float32 (4 bytes)
    head_dim = cfg.n_embd // cfg.n_head
    kv_cache_per_sample = (
        2  # K and V
        * cfg.n_layer
        * cfg.n_kv_head
        * max_tokens
        * head_dim
        * 4  # bytes per float32
    )

    # Available memory for KV-cache (leave 10% headroom for activations/overhead)
    safety_factor = 0.9
    available = (gpu_memory - allocated) * safety_factor

    # Calculate max batch size
    if kv_cache_per_sample > 0:
        max_batch = int(available / kv_cache_per_sample)
    else:
        max_batch = 512

    if verbose:
        print(f"    GPU memory: {gpu_memory/1e9:.1f}GB total, {allocated/1e9:.2f}GB allocated")
        print(f"    Available for KV-cache: {available/1e9:.2f}GB (with 10% headroom)")
        print(f"    KV-cache per sample: {kv_cache_per_sample/1e6:.1f}MB")
        print(f"    Max batch before capping: {max_batch}")

    # Round down to power of 2, clamp to [1, 512]
    # Small models can use very large batches for better GPU utilization
    valid_batches = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    for b in valid_batches:
        if b <= max_batch:
            return b

    return 1


def normalize_json(json_str: str) -> str:
    """Normalize JSON string to minified canonical form (sorted keys, no whitespace).

    This ensures fair comparison between TCT (minified) and UTF8 (formatted) outputs.
    Normalizes ISO 8601 timestamps:
    - Strips microseconds: .NNNNNN -> empty (TCT doesn't preserve them)
    - Normalizes timezone: +00:00 -> Z
    """
    try:
        parsed = json.loads(json_str)
        result = json.dumps(parsed, separators=(',', ':'), sort_keys=True)
        # Normalize ISO 8601 UTC timestamps:
        import re
        # 1. Strip microseconds: .NNNNNN -> empty (TCT doesn't preserve them)
        result = re.sub(r'(\d{2}:\d{2}:\d{2})\.\d+', r'\1', result)
        # 2. Normalize timezone: +00:00 -> Z
        result = re.sub(r'(\d{2}:\d{2}:\d{2})\+00:00', r'\1Z', result)
        result = re.sub(r'(\d{2}:\d{2}:\d{2})-00:00', r'\1Z', result)
        return result
    except json.JSONDecodeError:
        return json_str  # Return original if not valid JSON


def load_model(checkpoint_dir: Path, device: str = "cuda", epoch: int = None):
    """Load model from checkpoint directory.

    Args:
        checkpoint_dir: Path to checkpoint directory
        device: Device to load model on
        epoch: Specific epoch to load (default: None = auto-select best/latest)

    Returns the model and full config dict (including schema_config for data paths).
    Prioritizes: specified epoch > best.pt > checkpoint at min_val_loss epoch > latest checkpoint.
    """
    import torch
    from nanochat.gpt import GPT, GPTConfig

    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    # Handle nested config structure (model_config inside top-level config)
    model_cfg = config_dict.get("model_config", config_dict)

    model_config = GPTConfig(
        vocab_size=model_cfg.get("vocab_size", 20000),
        sequence_len=model_cfg.get("sequence_len", model_cfg.get("context_size", 2048)),
        n_layer=model_cfg.get("n_layer", 10),
        n_head=model_cfg.get("n_head", 6),
        n_kv_head=model_cfg.get("n_kv_head", model_cfg.get("n_head", 6)),
        n_embd=model_cfg.get("n_embd", 384),
        use_swiglu=model_cfg.get("use_swiglu", True),
        ffn_mult=model_cfg.get("ffn_mult", 2.5),
    )

    model = GPT(model_config)

    # Find checkpoint file
    checkpoint_path = None

    # If specific epoch requested, use that
    if epoch is not None:
        epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        if epoch_path.exists() and epoch_path.stat().st_size > 0:
            checkpoint_path = epoch_path
        else:
            raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found: {epoch_path}")

    # Otherwise, prioritize best.pt > config epoch > latest
    if checkpoint_path is None:
        checkpoint_path = checkpoint_dir / "best.pt"
        if not checkpoint_path.exists():
            # Try checkpoint at the epoch recorded in config (has the val_loss we want)
            config_epoch = config_dict.get("epoch")
            if config_epoch is not None:
                epoch_path = checkpoint_dir / f"epoch_{config_epoch:03d}.pt"
                if epoch_path.exists() and epoch_path.stat().st_size > 0:
                    checkpoint_path = epoch_path

    if checkpoint_path is None or not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        # Fall back to latest valid checkpoint
        checkpoint_files = [f for f in checkpoint_dir.glob("epoch_*.pt") if f.stat().st_size > 0]
        if not checkpoint_files:
            raise FileNotFoundError(f"No valid .pt files in {checkpoint_dir}")
        checkpoint_path = sorted(checkpoint_files)[-1]

    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle checkpoint format (may have model_state_dict key or be raw state_dict)
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, config_dict


def create_random_model(vocab_size: int, device: str = "cuda"):
    """Create a random model for testing."""
    from nanochat.gpt import GPT, GPTConfig

    model_config = GPTConfig(
        vocab_size=vocab_size,
        sequence_len=512,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=128,
    )
    model = GPT(model_config)
    model.to(device)
    model.eval()
    return model, {
        "vocab_size": vocab_size,
        "sequence_len": 512,
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
    }


def compute_valid_indices(
    utf8_tokens: List[List[int]],
    tct_tokens: List[List[int]],
    max_seq_len: int,
) -> List[int]:
    """Compute indices of samples that are valid for BOTH tokenizers.

    A sample is valid if its token count is <= max_seq_len in BOTH tokenizations.
    This ensures fair comparison by evaluating identical samples.
    """
    valid_indices = []
    for i in range(min(len(utf8_tokens), len(tct_tokens))):
        utf8_len = len(utf8_tokens[i])
        tct_len = len(tct_tokens[i])
        if utf8_len <= max_seq_len and tct_len <= max_seq_len:
            valid_indices.append(i)
    return valid_indices


def compute_first_token_distribution(data_dir: Path) -> Dict[int, float]:
    """Compute empirical distribution of first tokens from training data.

    TCT sequences start with different tokens depending on the schema variant
    (e.g., Deployment, Service, Pod). This distribution should be used to
    sample the first token during generation.

    Returns:
        Dict mapping token_id -> probability
    """
    from collections import Counter

    all_path = data_dir / "all.jsonl"
    if not all_path.exists():
        raise FileNotFoundError(f"Training data not found: {all_path}")

    first_tokens = Counter()
    with open(all_path) as f:
        for line in f:
            tokens = json.loads(line)
            if tokens:
                first_tokens[tokens[0]] += 1

    total = sum(first_tokens.values())
    return {token: count / total for token, count in first_tokens.items()}


def load_validation_tokens(data_dir: Path, max_samples: int = 1000) -> List[List[int]]:
    """Load validation tokens from JSONL file."""
    val_path = data_dir / "validate.jsonl"

    if not val_path.exists():
        # Try all.jsonl and use last 10% as validation
        all_path = data_dir / "all.jsonl"
        if all_path.exists():
            with open(all_path) as f:
                total_lines = sum(1 for _ in f)

            val_start = int(total_lines * 0.9)
            tokens = []
            with open(all_path) as f:
                for i, line in enumerate(f):
                    if i >= val_start:
                        tokens.append(json.loads(line))
                        if len(tokens) >= max_samples:
                            break
            return tokens
        else:
            raise FileNotFoundError(f"No validation file found in {data_dir}")

    tokens = []
    with open(val_path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            tokens.append(json.loads(line))
    return tokens


def load_validation_tokens_with_target(
    utf8_data_dir: Path,
    tct_data_dir: Optional[Path],
    target_valid: int,
    max_seq_len: int,
    max_iterations: int = 10,
) -> tuple:
    """Load validation tokens, auto-increasing until target valid samples is reached.

    Iteratively loads more samples until we have at least target_valid samples
    that pass the sequence length filter for both tokenizers.

    Args:
        utf8_data_dir: Path to UTF8 data directory
        tct_data_dir: Path to TCT data directory (or None for UTF8-only)
        target_valid: Target number of valid samples
        max_seq_len: Maximum sequence length filter
        max_iterations: Maximum attempts to reach target

    Returns:
        Tuple of (utf8_tokens, tct_tokens, valid_indices) where tokens are filtered lists
    """
    # Start with 20% overshoot to account for filtering
    current_request = int(target_valid * 1.2)

    for iteration in range(max_iterations):
        # Load samples
        utf8_tokens = load_validation_tokens(utf8_data_dir, current_request)

        if tct_data_dir:
            tct_tokens = load_validation_tokens(tct_data_dir, current_request)
            valid_indices = compute_valid_indices(utf8_tokens, tct_tokens, max_seq_len)
        else:
            tct_tokens = None
            # For UTF8-only, compute valid indices based on length
            valid_indices = [i for i, t in enumerate(utf8_tokens) if len(t) <= max_seq_len]

        num_valid = len(valid_indices)

        # Check if we've reached target
        if num_valid >= target_valid:
            # Trim to exactly target_valid
            valid_indices = valid_indices[:target_valid]
            utf8_tokens = [utf8_tokens[i] for i in valid_indices]
            if tct_tokens:
                tct_tokens = [tct_tokens[i] for i in valid_indices]
            print(f"  Loaded {current_request} samples, {num_valid} valid, using {target_valid}")
            return utf8_tokens, tct_tokens, valid_indices

        # Check if we've exhausted available samples
        if len(utf8_tokens) < current_request:
            # Can't load more, use what we have
            utf8_tokens = [utf8_tokens[i] for i in valid_indices]
            if tct_tokens:
                tct_tokens = [tct_tokens[i] for i in valid_indices]
            print(f"  WARNING: Only {num_valid} valid samples available (requested {target_valid})")
            return utf8_tokens, tct_tokens, valid_indices

        # Estimate how many more we need
        valid_ratio = num_valid / len(utf8_tokens) if utf8_tokens else 0.5
        if valid_ratio > 0:
            # Estimate total needed to get target_valid valid samples
            estimated_needed = int(target_valid / valid_ratio * 1.1)  # 10% safety margin
            current_request = max(current_request + target_valid, estimated_needed)
        else:
            current_request *= 2

        print(f"  Iteration {iteration + 1}: {num_valid} valid from {len(utf8_tokens)}, trying {current_request}...")

    # Fallback: return what we have
    utf8_tokens = [utf8_tokens[i] for i in valid_indices]
    if tct_tokens:
        tct_tokens = [tct_tokens[i] for i in valid_indices]
    print(f"  WARNING: After {max_iterations} iterations, only {len(valid_indices)} valid samples")
    return utf8_tokens, tct_tokens, valid_indices


def find_merge_table(schema: str, tokenizer: str = "utf8") -> Path:
    """Find BPE merge table for schema.

    Args:
        schema: Schema name (tsconfig, eslintrc, kubernetes)
        tokenizer: Tokenizer type ("utf8" or "tct") - prefers matching type

    Returns:
        Path to the merge table JSON file
    """
    bpe_dir = Path(__file__).parent.parent / "bpe-merges"

    # Define search order based on tokenizer type
    if tokenizer == "utf8":
        # Prefer UTF8 merge tables (matched variants first)
        suffixes = ["-utf8-bpe-1k-matched", "-utf8-base-matched", "-utf8-bpe-500", "-utf8-bpe"]
    else:
        # Prefer TCT merge tables
        suffixes = ["-tct-bpe-1k", "-tct-bpe-500", "-tct-base"]

    for suffix in suffixes:
        candidate = bpe_dir / f"{schema}{suffix}.json"
        if candidate.exists():
            return candidate

    # Fallback: any merge table for this schema
    candidates = list(bpe_dir.glob(f"{schema}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No merge table for {schema} in {bpe_dir}")
    return candidates[0]


def find_data_dir(schema: str, tokenizer: str = "utf8") -> Path:
    """Find data directory for schema using schema_configs.py definitions.

    Args:
        schema: Schema name (tsconfig, eslintrc, kubernetes)
        tokenizer: Tokenizer type ("utf8" or "tct")

    Raises:
        FileNotFoundError: If the expected data directory doesn't exist
    """
    from configs.schema_configs import get_schema_config

    config = get_schema_config(schema)

    if tokenizer == "tct":
        data_path = config.get("data_path_tct")
        dir_name = config.get("data_dir_tct")
    else:
        data_path = config.get("data_path_utf8")
        dir_name = config.get("data_dir_utf8")

    if data_path and data_path.exists():
        return data_path

    # Fail fast with clear error
    raise FileNotFoundError(
        f"Data directory not found for {schema} ({tokenizer}). "
        f"Expected: {data_path} (dir: {dir_name}). "
        f"Check schema_configs.py and ~/Desktop/data/"
    )


def run_constrained_bpb(
    model,
    schema: str,
    merge_table: Path,
    validation_tokens: List[List[int]],
    device: str,
    normalize_bytes: bool = True,
) -> Dict[str, Any]:
    """Run constrained BPB evaluation.

    Args:
        model: The model to evaluate
        schema: Schema name
        merge_table: Path to BPE merge table
        validation_tokens: Pre-filtered validation tokens (already filtered by max_seq_len)
        device: Device to use
        normalize_bytes: Whether to normalize bytes to minified JSON
    """
    from nanochat.xgrammar_tokenizer import (
        UTF8BPEDecoder,
        build_xgrammar_tokenizer_info,
        compile_json_schema_grammar,
        compute_constrained_bpb,
        load_schema,
    )

    print("\n" + "=" * 60)
    print("CONSTRAINED BPB EVALUATION")
    print("=" * 60)

    # Build tokenizer and grammar
    utf8_decoder = UTF8BPEDecoder(merge_table)
    print(f"  Merge table: {merge_table}")
    print(f"  Vocab size: {utf8_decoder.vocab_size()}")

    print("  Building XGrammar tokenizer info...")
    tokenizer_info = build_xgrammar_tokenizer_info(merge_table)

    print(f"  Loading schema: {schema}")
    schema_dict = load_schema(schema)

    print("  Compiling grammar...")
    compiled_grammar = compile_json_schema_grammar(tokenizer_info, schema_dict)

    print(f"  Validation sequences: {len(validation_tokens)} (pre-filtered)")

    # Compute constrained BPB
    print("\n  Computing constrained BPB...")
    result = compute_constrained_bpb(
        model=model,
        tokenizer_info=tokenizer_info,
        compiled_grammar=compiled_grammar,
        utf8_decoder=utf8_decoder,
        validation_tokens=validation_tokens,
        device=device,
        max_seq_len=None,  # Already pre-filtered
        show_progress=True,
        normalize_bytes=normalize_bytes,
    )

    print("\n  Results:")
    print(f"    Sequences:       {result.num_sequences}")
    print(f"    Tokens:          {result.total_tokens}")
    print(f"    Bytes:           {result.total_bytes}")
    print(f"    Raw BPB:         {result.raw_bpb:.4f}")
    print(f"    Constrained BPB: {result.constrained_bpb:.4f}")
    bpb_reduction = (result.raw_bpb - result.constrained_bpb) / result.raw_bpb * 100
    print(f"    BPB reduction:   {bpb_reduction:.1f}%")

    return {
        "raw_bpb": result.raw_bpb,
        "constrained_bpb": result.constrained_bpb,
        "bpb_reduction_percent": bpb_reduction,
        "num_sequences": result.num_sequences,
        "total_tokens": result.total_tokens,
        "total_bytes": result.total_bytes,
        "raw_loss_nats": result.raw_loss,
        "constrained_loss_nats": result.constrained_loss,
    }


def generate_samples_xgrammar(
    model,
    tokenizer_info,
    compiled_grammar,
    utf8_decoder,
    first_token_distribution: Dict[int, float],
    num_samples: int,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    show_progress: bool = True,
    batch_size: int = None,
    use_compile: bool = False,  # Disabled: torch.compile + KV-cache causes segfaults
) -> tuple:
    """Generate samples using UTF8-BPE model with XGrammar constraints.

    Uses KV-cache and batched generation for efficient GPU utilization.
    Each sample has its own grammar matcher state, but model forward passes
    are batched for better GPU throughput.

    Args:
        first_token_distribution: Dict mapping token_id -> probability for first token
        batch_size: Batch size for generation (None = auto-compute based on GPU memory)
        use_compile: Whether to use torch.compile for faster inference

    Returns:
        Tuple of (generated_texts, stats_dict) where stats_dict contains:
        - completed: Number of samples where grammar terminated normally
        - truncated: Number of samples that hit max_tokens before completion
        - empty: Number of samples that were empty
        - failed: Number of samples that failed to decode
    """
    import random
    import torch
    import torch.nn.functional as F
    import xgrammar
    from tqdm import tqdm
    from nanochat.engine import KVCache

    device = next(model.parameters()).device

    # torch.compile with KV-cache can cause segfaults - disabled by default
    # The issue is that reduce-overhead mode still triggers cudagraphs warnings
    # and can crash on some GPU/driver combinations
    if use_compile and not getattr(model, '_compiled', False):
        if show_progress:
            print("  Compiling model with torch.compile(mode='reduce-overhead')...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            model._compiled = True
        except Exception as e:
            if show_progress:
                print(f"  torch.compile failed: {e}, continuing without compilation")

    generated_texts = []
    completed_count = 0  # Grammar terminated normally
    truncated_count = 0  # Hit max_tokens before grammar termination
    empty_count = 0
    failed_count = 0

    # Get model config for KV cache
    cfg = model.config
    kv_kwargs = {
        "num_heads": cfg.n_kv_head,
        "head_dim": cfg.n_embd // cfg.n_head,
        "num_layers": cfg.n_layer,
    }

    # Auto-compute batch size based on model and GPU memory
    if batch_size is None:
        batch_size = compute_generation_batch_size(model, max_tokens, verbose=show_progress)
        if show_progress:
            print(f"  Auto batch size: {batch_size} (based on model={cfg.n_embd}d x {cfg.n_layer}L, seq={max_tokens})")

    # BOS token is pad token = vocab_size - 1 (matches training setup, same as TCT)
    bos_token = utf8_decoder.vocab_size() - 1

    # Process in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    iterator = tqdm(range(num_batches), desc="Generating samples") if show_progress else range(num_batches)

    total_tokens_generated = 0
    gen_start_time = time.time()

    for batch_idx in iterator:
        # Calculate actual batch size for this batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        try:
            # Create grammar matchers and bitmasks for each sample
            matchers = [xgrammar.GrammarMatcher(compiled_grammar) for _ in range(current_batch_size)]
            bitmasks = [xgrammar.allocate_token_bitmask(1, tokenizer_info.vocab_size)
                       for _ in range(current_batch_size)]
            kv_cache = KVCache(batch_size=current_batch_size, seq_len=max_tokens + 1, **kv_kwargs)

            # Initialize with BOS tokens (same approach as TCT)
            batch_tokens = [[] for _ in range(current_batch_size)]

            # Track which samples are still active (not terminated)
            active = [True] * current_batch_size
            terminated = [False] * current_batch_size  # Track if grammar terminated vs truncated

            # Prefill with BOS token (model predicts first real token)
            input_ids = torch.full((current_batch_size, 1), bos_token, device=device)
            with torch.no_grad():
                logits = model(input_ids, kv_cache=kv_cache)
                logits = logits[:, -1, :]  # [batch, vocab]

            for step in range(max_tokens):
                # Check if all samples terminated
                if not any(active):
                    break

                # Apply grammar masks per sample
                for i in range(current_batch_size):
                    if active[i]:
                        if matchers[i].is_terminated():
                            active[i] = False
                            terminated[i] = True  # Properly completed
                            continue
                        xgrammar.reset_token_bitmask(bitmasks[i])
                        matchers[i].fill_next_token_bitmask(bitmasks[i])
                        # Apply mask to this sample's logits
                        logits_i = logits[i:i+1, :]
                        xgrammar.apply_token_bitmask_inplace(logits_i, bitmasks[i].to(device))

                # Apply temperature and top-k (batched)
                if temperature > 0:
                    logits = logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                    logits[logits < v[:, -1:]] = float('-inf')

                # Sample next tokens
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)  # [batch, 1]
                next_tokens_list = next_tokens.squeeze(-1).tolist()

                # Accept tokens and update state
                for i, tok in enumerate(next_tokens_list):
                    if active[i]:
                        matchers[i].accept_token(tok)
                        batch_tokens[i].append(tok)

                # Get next logits using KV cache (batched)
                with torch.no_grad():
                    logits = model(next_tokens, kv_cache=kv_cache)
                    logits = logits[:, -1, :]

            # Decode and validate each sample
            for i, tokens in enumerate(batch_tokens):
                try:
                    text = utf8_decoder.decode(tokens)
                    import json
                    parsed = json.loads(text)
                    if parsed and parsed != {}:
                        generated_texts.append(text)
                        if terminated[i]:
                            completed_count += 1
                        else:
                            truncated_count += 1  # Valid JSON but hit max_tokens
                    else:
                        empty_count += 1
                except (json.JSONDecodeError, Exception):
                    failed_count += 1

        except Exception as e:
            failed_count += current_batch_size
            continue

        # Count tokens generated in this batch
        total_tokens_generated += sum(len(tokens) for tokens in batch_tokens)

    gen_elapsed_time = time.time() - gen_start_time
    tokens_per_second = total_tokens_generated / gen_elapsed_time if gen_elapsed_time > 0 else 0
    samples_per_second = num_samples / gen_elapsed_time if gen_elapsed_time > 0 else 0

    stats = {
        "completed": completed_count,
        "truncated": truncated_count,
        "empty": empty_count,
        "failed": failed_count,
        "total": num_samples,
        "completion_rate": completed_count / num_samples if num_samples > 0 else 0,
        "generation_time_seconds": gen_elapsed_time,
        "total_tokens_generated": total_tokens_generated,
        "tokens_per_second": tokens_per_second,
        "samples_per_second": samples_per_second,
    }

    if show_progress:
        print(f"  Completed: {completed_count}/{num_samples} ({stats['completion_rate']:.1%})")
        print(f"  Truncated: {truncated_count} (valid but hit max_tokens)")
        print(f"  Empty: {empty_count}, Failed: {failed_count}")
        print(f"  Throughput: {tokens_per_second:.0f} tokens/sec ({samples_per_second:.2f} samples/sec)")

    return generated_texts, stats


def generate_samples_utf8_raw(
    model,
    utf8_decoder,
    first_token_distribution: Dict[int, float],
    num_samples: int,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    show_progress: bool = True,
    batch_size: int = None,
    use_compile: bool = False,  # Disabled: torch.compile + KV-cache causes segfaults
) -> tuple:
    """Generate samples using UTF8-BPE model WITHOUT any grammar constraints.

    This is the raw model output - no XGrammar masking.
    Used as a baseline to show what the model generates without constraints.

    Args:
        first_token_distribution: Dict mapping token_id -> probability for first token
        batch_size: Batch size for generation (None = auto-compute based on GPU memory)
        use_compile: Whether to use torch.compile for faster inference

    Returns:
        Tuple of (generated_texts, stats_dict) where stats_dict contains:
        - valid_json: Number of samples that are valid JSON
        - valid_nonempty: Number of samples that are valid non-empty JSON
        - invalid: Number of samples that failed JSON parsing
        - empty: Number of samples that were empty objects
    """
    import random
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from nanochat.engine import KVCache

    device = next(model.parameters()).device

    # torch.compile with KV-cache can cause segfaults - disabled by default
    if use_compile and not getattr(model, '_compiled', False):
        if show_progress:
            print("  Compiling model with torch.compile(mode='reduce-overhead')...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            model._compiled = True
        except Exception as e:
            if show_progress:
                print(f"  torch.compile failed: {e}, continuing without compilation")

    generated_texts = []
    valid_json_count = 0
    valid_nonempty_count = 0
    invalid_count = 0
    empty_count = 0

    # Get model config for KV cache
    cfg = model.config
    kv_kwargs = {
        "num_heads": cfg.n_kv_head,
        "head_dim": cfg.n_embd // cfg.n_head,
        "num_layers": cfg.n_layer,
    }

    # Auto-compute batch size based on model and GPU memory
    if batch_size is None:
        batch_size = compute_generation_batch_size(model, max_tokens, verbose=show_progress)
        if show_progress:
            print(f"  Auto batch size: {batch_size}")

    # BOS token is pad token = vocab_size - 1 (matches training setup, same as TCT)
    bos_token = utf8_decoder.vocab_size() - 1
    # EOS token for this vocabulary
    eos_token = utf8_decoder.eos_token_id()

    # Process in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    iterator = tqdm(range(num_batches), desc="Generating raw UTF8 samples") if show_progress else range(num_batches)

    total_tokens_generated = 0
    gen_start_time = time.time()

    for batch_idx in iterator:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        try:
            kv_cache = KVCache(batch_size=current_batch_size, seq_len=max_tokens + 1, **kv_kwargs)

            # Initialize with BOS tokens (same approach as TCT)
            batch_tokens = [[] for _ in range(current_batch_size)]

            # Track which samples are still active (not hit EOS)
            active = [True] * current_batch_size

            # Prefill with BOS token (model predicts first real token)
            input_ids = torch.full((current_batch_size, 1), bos_token, device=device)
            with torch.no_grad():
                logits = model(input_ids, kv_cache=kv_cache)
                logits = logits[:, -1, :]

            for step in range(max_tokens):
                if not any(active):
                    break

                # Apply temperature and top-k (batched) - NO grammar masking
                if temperature > 0:
                    logits = logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                    logits[logits < v[:, -1:]] = float('-inf')

                # Sample next tokens
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
                next_tokens_list = next_tokens.squeeze(-1).tolist()

                # Update state
                for i, tok in enumerate(next_tokens_list):
                    if active[i]:
                        if tok == eos_token:
                            active[i] = False
                        else:
                            batch_tokens[i].append(tok)

                # Get next logits using KV cache (batched)
                with torch.no_grad():
                    logits = model(next_tokens, kv_cache=kv_cache)
                    logits = logits[:, -1, :]

            # Decode and validate each sample
            for tokens in batch_tokens:
                try:
                    text = utf8_decoder.decode(tokens)
                    import json
                    parsed = json.loads(text)
                    valid_json_count += 1
                    if parsed and parsed != {}:
                        generated_texts.append(text)
                        valid_nonempty_count += 1
                    else:
                        empty_count += 1
                except (json.JSONDecodeError, Exception):
                    invalid_count += 1

        except Exception as e:
            invalid_count += current_batch_size
            continue

        # Count tokens generated in this batch
        total_tokens_generated += sum(len(tokens) for tokens in batch_tokens)

    gen_elapsed_time = time.time() - gen_start_time
    tokens_per_second = total_tokens_generated / gen_elapsed_time if gen_elapsed_time > 0 else 0
    samples_per_second = num_samples / gen_elapsed_time if gen_elapsed_time > 0 else 0

    stats = {
        "valid_json": valid_json_count,
        "valid_nonempty": valid_nonempty_count,
        "invalid": invalid_count,
        "empty": empty_count,
        "total": num_samples,
        "validity_rate": valid_nonempty_count / num_samples if num_samples > 0 else 0,
        "generation_time_seconds": gen_elapsed_time,
        "total_tokens_generated": total_tokens_generated,
        "tokens_per_second": tokens_per_second,
        "samples_per_second": samples_per_second,
    }

    if show_progress:
        print(f"  Valid JSON: {valid_json_count}/{num_samples} ({valid_json_count/num_samples:.1%})")
        print(f"  Valid non-empty: {valid_nonempty_count}/{num_samples} ({stats['validity_rate']:.1%})")
        print(f"  Invalid: {invalid_count}, Empty: {empty_count}")
        print(f"  Throughput: {tokens_per_second:.0f} tokens/sec ({samples_per_second:.2f} samples/sec)")

    return generated_texts, stats


def generate_samples_tct(
    model,
    tct_module,
    num_samples: int,
    first_token_distribution: Dict[int, float],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    show_progress: bool = True,
    batch_size: int = None,
    use_compile: bool = False,  # Disabled: torch.compile + KV-cache causes segfaults
) -> tuple:
    """Generate samples using TCT model with batched generation.

    TCT is inherently constrained - all outputs are valid by construction.
    Unlike UTF8+XGrammar, TCT has no explicit EOS token - sequences are complete
    when they reach a target length. We generate max_tokens and decode.

    Uses KV-cache and batched generation for efficient GPU utilization.
    Batching provides ~10-30x throughput improvement on small models.

    The model was trained with BOS token (pad token = vocab_size - 1) prepended
    to all sequences. Generation starts with BOS and lets model predict first
    real token from scratch.

    Args:
        model: The GPT model
        tct_module: TCT tokenizer module (encode/decode/decode_prefix/vocab_size)
        num_samples: Number of samples to generate
        first_token_distribution: Dict mapping token_id -> probability for first token
            (used for UTF8 generation, kept for API compatibility but ignored here)
        max_tokens: Maximum tokens per sample (should match typical training lengths)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        show_progress: Whether to show progress bar
        batch_size: Batch size for generation (None = auto-compute based on GPU memory)
        use_compile: Whether to use torch.compile for faster inference

    Returns:
        Tuple of (generated_texts, stats_dict) where stats_dict contains:
        - completed: Number of samples where TCT decode completed fully
        - partial: Number of samples that decoded partially (hit max_tokens)
        - empty: Number of samples that decoded to empty objects
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    from nanochat.engine import KVCache

    device = next(model.parameters()).device

    # torch.compile with KV-cache can cause segfaults - disabled by default
    if use_compile and not getattr(model, '_compiled', False):
        if show_progress:
            print("  Compiling model with torch.compile(mode='reduce-overhead')...")
        try:
            model = torch.compile(model, mode="reduce-overhead")
            model._compiled = True
        except Exception as e:
            if show_progress:
                print(f"  torch.compile failed: {e}, continuing without compilation")

    generated_texts = []
    completed_count = 0  # TCT decode completed (is_complete=True)
    partial_count = 0    # Decoded but not complete (hit max_tokens)
    empty_count = 0

    # Get model config for KV cache
    cfg = model.config
    kv_kwargs = {
        "num_heads": cfg.n_kv_head,
        "head_dim": cfg.n_embd // cfg.n_head,
        "num_layers": cfg.n_layer,
    }

    # Auto-compute batch size based on model and GPU memory
    if batch_size is None:
        batch_size = compute_generation_batch_size(model, max_tokens, verbose=show_progress)
        if show_progress:
            print(f"  Auto batch size: {batch_size} (based on model={cfg.n_embd}d x {cfg.n_layer}L, seq={max_tokens})")

    # BOS token is pad token = vocab_size - 1 (matches training setup)
    bos_token = tct_module.vocab_size() - 1

    # Process in batches
    num_batches = (num_samples + batch_size - 1) // batch_size
    iterator = tqdm(range(num_batches), desc="Generating TCT samples") if show_progress else range(num_batches)

    all_generated_tokens = []
    total_tokens_generated = 0
    gen_start_time = time.time()

    for batch_idx in iterator:
        # Calculate actual batch size for this batch (last batch may be smaller)
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        current_batch_size = end_idx - start_idx

        # Create KV cache for this batch
        kv_cache = KVCache(batch_size=current_batch_size, seq_len=max_tokens + 1, **kv_kwargs)

        # Initialize with BOS tokens for all samples in batch
        input_ids = torch.full((current_batch_size, 1), bos_token, device=device)

        # Track generated tokens for each sample in batch
        batch_tokens = [[bos_token] for _ in range(current_batch_size)]

        # Prefill with BOS token
        with torch.no_grad():
            logits = model(input_ids, kv_cache=kv_cache)
            logits = logits[:, -1, :]  # [batch, vocab]

        # Generate max_tokens for all samples in parallel
        for step in range(max_tokens):
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-k (batched)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                logits[logits < v[:, -1:]] = float('-inf')

            # Sample next tokens for all samples
            probs = F.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  # [batch, 1]

            # Store generated tokens
            next_tokens_list = next_tokens.squeeze(-1).tolist()
            for i, tok in enumerate(next_tokens_list):
                batch_tokens[i].append(tok)

            # Get next logits using KV cache
            with torch.no_grad():
                logits = model(next_tokens, kv_cache=kv_cache)
                logits = logits[:, -1, :]

        # Store all generated token sequences from this batch
        all_generated_tokens.extend(batch_tokens)
        total_tokens_generated += current_batch_size * max_tokens

    gen_elapsed_time = time.time() - gen_start_time

    # Decode all samples (this is sequential, but decoding is fast)
    if show_progress:
        print(f"  Decoding {len(all_generated_tokens)} samples...")

    for generated_tokens in all_generated_tokens:
        try:
            tokens_to_decode = generated_tokens[1:]  # Skip BOS
            json_out, consumed, is_complete = tct_module.decode_prefix(tokens_to_decode)

            if is_complete and json_out and json_out != "{}":
                generated_texts.append(json_out)
                completed_count += 1
            elif is_complete:
                empty_count += 1
            elif json_out and json_out != "{}":
                partial_count += 1
            else:
                empty_count += 1
        except Exception:
            continue

    tokens_per_second = total_tokens_generated / gen_elapsed_time if gen_elapsed_time > 0 else 0
    samples_per_second = num_samples / gen_elapsed_time if gen_elapsed_time > 0 else 0

    stats = {
        "completed": completed_count,
        "partial": partial_count,
        "empty": empty_count,
        "total": num_samples,
        "completion_rate": completed_count / num_samples if num_samples > 0 else 0,
        "generation_time_seconds": gen_elapsed_time,
        "total_tokens_generated": total_tokens_generated,
        "tokens_per_second": tokens_per_second,
        "samples_per_second": samples_per_second,
    }

    if show_progress:
        print(f"  Completed: {completed_count}/{num_samples} ({stats['completion_rate']:.1%})")
        print(f"  Partial: {partial_count} (decoded but hit max_tokens)")
        print(f"  Empty: {empty_count}")
        print(f"  Throughput: {tokens_per_second:.0f} tokens/sec ({samples_per_second:.2f} samples/sec)")

    return generated_texts, stats


def run_generation_quality_utf8(
    model,
    schema: str,
    merge_table: Path,
    data_dir: Path,
    num_samples: int,
    device: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    normalize_json_output: bool = True,
) -> Dict[str, Any]:
    """Run generation quality evaluation for UTF8-BPE model with XGrammar.

    Args:
        normalize_json_output: If True, normalize ground truth and generated JSON
                              to minified form for consistent field extraction
    """
    from nanochat.field_extractors import get_extractor
    from nanochat.distribution_metrics import compare_extraction_results
    from nanochat.xgrammar_tokenizer import (
        UTF8BPEDecoder,
        build_xgrammar_tokenizer_info,
        compile_json_schema_grammar,
        load_schema,
    )

    print("\n" + "=" * 60)
    print("GENERATION QUALITY EVALUATION (UTF8-BPE + XGrammar)")
    print("=" * 60)

    # Get field extractor
    extractor = get_extractor(schema)
    print(f"  Schema: {schema}")
    print(f"  Fields defined: {len(extractor.field_definitions)}")

    # Build tokenizer and grammar for generation
    utf8_decoder = UTF8BPEDecoder(merge_table)
    tokenizer_info = build_xgrammar_tokenizer_info(merge_table)
    schema_dict = load_schema(schema)
    compiled_grammar = compile_json_schema_grammar(tokenizer_info, schema_dict)

    # Compute first token distribution for fair comparison with TCT
    print("  Computing first token distribution for generation...")
    first_token_dist = compute_first_token_distribution(data_dir)
    print(f"  First tokens: {len(first_token_dist)} unique, top: {list(first_token_dist.keys())[:5]}")

    # Load and decode validation samples for ground truth distribution
    print(f"\n  Extracting ground truth distribution from validation data...")
    validation_tokens = load_validation_tokens(data_dir, num_samples)
    val_texts = [utf8_decoder.decode(tokens) for tokens in validation_tokens]

    # Optionally normalize to canonical JSON for consistent comparison
    if normalize_json_output:
        val_texts = [normalize_json(t) for t in val_texts]

    real_result = extractor.extract_from_samples(val_texts)
    print(f"  Validation samples: {real_result.num_valid}/{real_result.num_samples} valid")

    # Report ground truth distribution
    print("\n  Ground truth field distributions:")
    for name, dist in sorted(real_result.field_distributions.items()):
        if dist.total > 0:
            print(f"    {name}: n={dist.total}, mode={dist.mode()}")

    # Generate samples from model
    print(f"\n  Generating {num_samples} samples with XGrammar constraints...")
    generated_texts, gen_stats = generate_samples_xgrammar(
        model=model,
        tokenizer_info=tokenizer_info,
        compiled_grammar=compiled_grammar,
        utf8_decoder=utf8_decoder,
        first_token_distribution=first_token_dist,
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        show_progress=True,
    )

    # Optionally normalize generated samples to canonical JSON
    if normalize_json_output:
        generated_texts = [normalize_json(t) for t in generated_texts]

    # Extract fields from generated samples
    print(f"\n  Extracting fields from {len(generated_texts)} generated samples...")
    gen_result = extractor.extract_from_samples(generated_texts)
    print(f"  Generated samples: {gen_result.num_valid}/{gen_result.num_samples} valid")

    # Report generated distribution
    print("\n  Generated field distributions:")
    for name, dist in sorted(gen_result.field_distributions.items()):
        if dist.total > 0:
            print(f"    {name}: n={dist.total}, mode={dist.mode()}")

    # Compare distributions
    print("\n  Comparing distributions...")
    comparison = compare_extraction_results(real_result, gen_result)

    print(f"\n  Distribution Comparison Summary:")
    print(f"    Mean KL divergence:  {comparison.mean_kl:.4f}")
    print(f"    Mean TV distance:    {comparison.mean_tv:.4f}")
    print(f"    Mean coverage:       {comparison.mean_coverage:.2%}")
    print(f"    Mode match rate:     {comparison.mode_match_rate:.2%}")

    # Per-field details
    print("\n  Per-field comparison:")
    for name, comp in sorted(comparison.field_comparisons.items()):
        match = "Y" if comp.mode_match else "N"
        print(f"    {name}: KL={comp.kl_divergence:.3f} TV={comp.total_variation:.3f} mode_match={match}")

    return {
        "schema": schema,
        "model_type": "utf8_xgrammar",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "validation_samples": real_result.num_valid,
        "generated_samples": gen_result.num_valid,
        "generation_stats": gen_stats,
        "ground_truth_distributions": real_result.to_dict()["field_distributions"],
        "generated_distributions": gen_result.to_dict()["field_distributions"],
        "comparison": comparison.to_dict(),
    }


def run_generation_quality_tct(
    model,
    schema: str,
    tct_module,
    validation_json_strings: List[str],
    first_token_distribution: Dict[int, float],
    num_samples: int,
    temperature: float = 0.7,
    max_tokens: int = 512,
    normalize_json_output: bool = True,
) -> Dict[str, Any]:
    """Run generation quality evaluation for TCT model.

    Args:
        model: The GPT model
        schema: Schema name ("tsconfig", "eslintrc", "kubernetes")
        tct_module: TCT tokenizer module
        validation_json_strings: Ground truth JSON strings from validation set
        first_token_distribution: Dict mapping token_id -> probability for first token
        num_samples: Number of samples to generate
        temperature: Sampling temperature
        max_tokens: Maximum tokens per sample
        normalize_json_output: If True, normalize ground truth and generated JSON
                              to minified form for consistent field extraction

    Returns:
        Dict with generation results and distribution comparison
    """
    from nanochat.field_extractors import get_extractor
    from nanochat.distribution_metrics import compare_extraction_results

    print("\n" + "=" * 60)
    print("GENERATION QUALITY EVALUATION (TCT)")
    print("=" * 60)

    # Get field extractor
    extractor = get_extractor(schema)
    print(f"  Schema: {schema}")
    print(f"  Fields defined: {len(extractor.field_definitions)}")

    # Extract ground truth distribution from validation samples
    print(f"\n  Extracting ground truth distribution from {len(validation_json_strings)} validation samples...")

    # Optionally normalize ground truth to canonical JSON
    if normalize_json_output:
        validation_json_strings = [normalize_json(s) for s in validation_json_strings]

    real_result = extractor.extract_from_samples(validation_json_strings)
    print(f"  Validation samples: {real_result.num_valid}/{real_result.num_samples} valid")

    # Report ground truth distribution
    print("\n  Ground truth field distributions:")
    for name, dist in sorted(real_result.field_distributions.items()):
        if dist.total > 0:
            print(f"    {name}: n={dist.total}, mode={dist.mode()}")

    # Generate samples from TCT model
    print(f"\n  Generating {num_samples} samples with TCT model...")
    generated_texts, gen_stats = generate_samples_tct(
        model=model,
        tct_module=tct_module,
        num_samples=num_samples,
        first_token_distribution=first_token_distribution,
        max_tokens=max_tokens,
        temperature=temperature,
        show_progress=True,
    )

    # Optionally normalize generated samples to canonical JSON
    if normalize_json_output:
        generated_texts = [normalize_json(t) for t in generated_texts]

    # Extract fields from generated samples
    print(f"\n  Extracting fields from {len(generated_texts)} generated samples...")
    gen_result = extractor.extract_from_samples(generated_texts)
    print(f"  Generated samples: {gen_result.num_valid}/{gen_result.num_samples} valid")

    # Report generated distribution
    print("\n  Generated field distributions:")
    for name, dist in sorted(gen_result.field_distributions.items()):
        if dist.total > 0:
            print(f"    {name}: n={dist.total}, mode={dist.mode()}")

    # Compare distributions
    print("\n  Comparing distributions...")
    comparison = compare_extraction_results(real_result, gen_result)

    print(f"\n  Distribution Comparison Summary:")
    print(f"    Mean KL divergence:  {comparison.mean_kl:.4f}")
    print(f"    Mean TV distance:    {comparison.mean_tv:.4f}")
    print(f"    Mean coverage:       {comparison.mean_coverage:.2%}")
    print(f"    Mode match rate:     {comparison.mode_match_rate:.2%}")

    # Per-field details
    print("\n  Per-field comparison:")
    for name, comp in sorted(comparison.field_comparisons.items()):
        match = "Y" if comp.mode_match else "N"
        print(f"    {name}: KL={comp.kl_divergence:.3f} TV={comp.total_variation:.3f} mode_match={match}")

    return {
        "schema": schema,
        "model_type": "tct",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "validation_samples": real_result.num_valid,
        "generated_samples": gen_result.num_valid,
        "generation_stats": gen_stats,
        "ground_truth_distributions": real_result.to_dict()["field_distributions"],
        "generated_distributions": gen_result.to_dict()["field_distributions"],
        "comparison": comparison.to_dict(),
    }


def run_generation_quality_utf8_raw(
    model,
    schema: str,
    merge_table: Path,
    data_dir: Path,
    num_samples: int,
    device: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
    normalize_json_output: bool = True,
) -> Dict[str, Any]:
    """Run generation quality evaluation for UTF8-BPE model WITHOUT XGrammar.

    This shows raw model output without any grammar constraints.
    """
    from nanochat.field_extractors import get_extractor
    from nanochat.distribution_metrics import compare_extraction_results
    from nanochat.xgrammar_tokenizer import UTF8BPEDecoder

    print("\n" + "=" * 60)
    print("GENERATION QUALITY EVALUATION (UTF8-BPE RAW - No XGrammar)")
    print("=" * 60)

    # Get field extractor
    extractor = get_extractor(schema)
    print(f"  Schema: {schema}")
    print(f"  Fields defined: {len(extractor.field_definitions)}")

    # Build decoder
    utf8_decoder = UTF8BPEDecoder(merge_table)

    # Compute first token distribution for fair comparison
    print("  Computing first token distribution for generation...")
    first_token_dist = compute_first_token_distribution(data_dir)
    print(f"  First tokens: {len(first_token_dist)} unique")

    # Load and decode validation samples for ground truth distribution
    print(f"\n  Extracting ground truth distribution from validation data...")
    validation_tokens = load_validation_tokens(data_dir, num_samples)
    val_texts = [utf8_decoder.decode(tokens) for tokens in validation_tokens]

    if normalize_json_output:
        val_texts = [normalize_json(t) for t in val_texts]

    real_result = extractor.extract_from_samples(val_texts)
    print(f"  Validation samples: {real_result.num_valid}/{real_result.num_samples} valid")

    # Generate samples from model (no grammar constraints!)
    print(f"\n  Generating {num_samples} samples WITHOUT grammar constraints...")
    generated_texts, gen_stats = generate_samples_utf8_raw(
        model=model,
        utf8_decoder=utf8_decoder,
        first_token_distribution=first_token_dist,
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        show_progress=True,
    )

    if normalize_json_output:
        generated_texts = [normalize_json(t) for t in generated_texts]

    # Extract fields from generated samples
    print(f"\n  Extracting fields from {len(generated_texts)} generated samples...")
    gen_result = extractor.extract_from_samples(generated_texts)
    print(f"  Generated samples: {gen_result.num_valid}/{gen_result.num_samples} valid")

    # Compare distributions (if we have any valid samples)
    if gen_result.num_valid > 0:
        print("\n  Comparing distributions...")
        comparison = compare_extraction_results(real_result, gen_result)

        print(f"\n  Distribution Comparison Summary:")
        print(f"    Mean KL divergence:  {comparison.mean_kl:.4f}")
        print(f"    Mean TV distance:    {comparison.mean_tv:.4f}")
        print(f"    Mean coverage:       {comparison.mean_coverage:.2%}")
        print(f"    Mode match rate:     {comparison.mode_match_rate:.2%}")

        comparison_dict = comparison.to_dict()
    else:
        print("\n  No valid samples generated - cannot compare distributions")
        comparison_dict = {"mean_kl": float('inf'), "mean_tv": 1.0, "mean_coverage": 0.0, "mode_match_rate": 0.0}

    return {
        "schema": schema,
        "model_type": "utf8_raw",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "validation_samples": real_result.num_valid,
        "generated_samples": gen_result.num_valid,
        "generation_stats": gen_stats,
        "ground_truth_distributions": real_result.to_dict()["field_distributions"],
        "generated_distributions": gen_result.to_dict()["field_distributions"],
        "comparison": comparison_dict,
    }


def generate_latex_tables(results: Dict[str, Any], output_dir: Optional[Path] = None) -> str:
    """Generate LaTeX tables from evaluation results.

    Returns:
        LaTeX string with all tables
    """
    lines = []
    has_tct = "tct_bpb" in results or "tct_generation" in results

    # Table 1: BPB Comparison (TCT vs UTF8)
    if "utf8_constrained_bpb" in results or "tct_bpb" in results:
        lines.extend([
            "% Table 1: BPB Comparison (TCT vs UTF8+XGrammar)",
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Bits-per-byte comparison: TCT vs UTF8-BPE with grammar constraints}",
            "\\label{tab:bpb-comparison}",
        ])

        if has_tct:
            lines.extend([
                "\\begin{tabular}{lrrrr}",
                "\\toprule",
                "Schema & TCT BPB & UTF8 Raw & UTF8 Constrained & Reduction \\\\",
                "\\midrule",
            ])
            tct_bpb = results.get("tct_bpb", {}).get("bpb", "-")
            utf8_bpb = results.get("utf8_constrained_bpb", {})
            raw = utf8_bpb.get("raw_bpb", "-")
            constrained = utf8_bpb.get("constrained_bpb", "-")
            reduction = utf8_bpb.get("bpb_reduction_percent", "-")

            tct_str = f"{tct_bpb:.4f}" if isinstance(tct_bpb, (int, float)) else "-"
            raw_str = f"{raw:.4f}" if isinstance(raw, (int, float)) else "-"
            constrained_str = f"{constrained:.4f}" if isinstance(constrained, (int, float)) else "-"
            reduction_str = f"{reduction:.1f}\\%" if isinstance(reduction, (int, float)) else "-"

            lines.append(f"{results['schema']} & {tct_str} & {raw_str} & {constrained_str} & {reduction_str} \\\\")
        else:
            bpb = results["utf8_constrained_bpb"]
            lines.extend([
                "\\begin{tabular}{lrrr}",
                "\\toprule",
                "Schema & Raw BPB & Constrained BPB & Reduction \\\\",
                "\\midrule",
                f"{results['schema']} & {bpb['raw_bpb']:.4f} & {bpb['constrained_bpb']:.4f} & {bpb['bpb_reduction_percent']:.1f}\\% \\\\",
            ])

        lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ])

    # Table 2: Generation Quality Comparison (TCT vs UTF8+XGrammar)
    if "utf8_generation" in results or "tct_generation" in results:
        utf8_comp = results.get("utf8_generation", {}).get("comparison", {})
        tct_comp = results.get("tct_generation", {}).get("comparison", {})

        lines.extend([
            "% Table 2: Generation Quality Comparison",
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Generation quality metrics: TCT vs UTF8+XGrammar}",
            "\\label{tab:generation-comparison}",
        ])

        if has_tct:
            lines.extend([
                "\\begin{tabular}{lrr}",
                "\\toprule",
                "Metric & UTF8+XGrammar & TCT \\\\",
                "\\midrule",
                f"Mean KL Divergence & {utf8_comp.get('mean_kl', 0):.4f} & {tct_comp.get('mean_kl', 0):.4f} \\\\",
                f"Mean TV Distance & {utf8_comp.get('mean_tv', 0):.4f} & {tct_comp.get('mean_tv', 0):.4f} \\\\",
                f"Coverage & {utf8_comp.get('mean_coverage', 0) * 100:.1f}\\% & {tct_comp.get('mean_coverage', 0) * 100:.1f}\\% \\\\",
                f"Mode Match Rate & {utf8_comp.get('mode_match_rate', 0) * 100:.1f}\\% & {tct_comp.get('mode_match_rate', 0) * 100:.1f}\\% \\\\",
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ])
        else:
            # Single model table (original format)
            gen = results.get("utf8_generation", results.get("tct_generation", {}))
            comp = gen.get("comparison", {})
            model_type = gen.get("model_type", "unknown")

            lines.extend([
                "\\begin{tabular}{lr}",
                "\\toprule",
                "Metric & Value \\\\",
                "\\midrule",
                f"Model Type & {model_type} \\\\",
                f"Mean KL Divergence & {comp.get('mean_kl', 0):.4f} \\\\",
                f"Mean TV Distance & {comp.get('mean_tv', 0):.4f} \\\\",
                f"Coverage & {comp.get('mean_coverage', 0) * 100:.1f}\\% \\\\",
                f"Mode Match Rate & {comp.get('mode_match_rate', 0) * 100:.1f}\\% \\\\",
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ])

    # Table 3: Per-Field Distribution Comparison (if both models present)
    if has_tct and "utf8_generation" in results and "tct_generation" in results:
        utf8_fields = results["utf8_generation"].get("comparison", {}).get("field_comparisons", {})
        tct_fields = results["tct_generation"].get("comparison", {}).get("field_comparisons", {})

        # Get all field names
        all_fields = set(utf8_fields.keys()) | set(tct_fields.keys())

        if all_fields:
            lines.extend([
                "% Table 3: Per-Field Distribution Comparison",
                "\\begin{table}[h]",
                "\\centering",
                "\\caption{Per-field KL divergence comparison (lower is better)}",
                "\\label{tab:field-comparison}",
                "\\begin{tabular}{lrr}",
                "\\toprule",
                "Field & UTF8+XGrammar & TCT \\\\",
                "\\midrule",
            ])

            for name in sorted(all_fields):
                utf8_kl = utf8_fields.get(name, {}).get("kl_divergence", "-")
                tct_kl = tct_fields.get(name, {}).get("kl_divergence", "-")
                utf8_str = f"{utf8_kl:.3f}" if isinstance(utf8_kl, (int, float)) else "-"
                tct_str = f"{tct_kl:.3f}" if isinstance(tct_kl, (int, float)) else "-"
                lines.append(f"{name.replace('_', '\\_')} & {utf8_str} & {tct_str} \\\\")

            # Add mean row
            utf8_mean = results["utf8_generation"].get("comparison", {}).get("mean_kl", 0)
            tct_mean = results["tct_generation"].get("comparison", {}).get("mean_kl", 0)
            lines.extend([
                "\\midrule",
                f"\\textbf{{Mean}} & \\textbf{{{utf8_mean:.3f}}} & \\textbf{{{tct_mean:.3f}}} \\\\",
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ])

    # Table for single-model field distribution (if only one model)
    elif "utf8_generation" in results and "comparison" in results["utf8_generation"] and not has_tct:
        comp = results["utf8_generation"]["comparison"]
        field_comps = comp.get("field_comparisons", {})

        lines.extend([
            "% Table 3: Field Distribution Comparison",
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Field value distribution comparison (generated vs validation)}",
            "\\label{tab:field-distribution}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Field & KL Div. & TV Dist. & Coverage & Mode Match \\\\",
            "\\midrule",
        ])

        for name, fc in sorted(field_comps.items()):
            match = "\\checkmark" if fc.get("mode_match", False) else ""
            coverage = fc.get("coverage", 0) * 100
            lines.append(
                f"{name.replace('_', '\\_')} & {fc.get('kl_divergence', 0):.3f} & "
                f"{fc.get('total_variation', 0):.3f} & {coverage:.0f}\\% & {match} \\\\"
            )

        lines.extend([
            "\\midrule",
            f"\\textbf{{Mean}} & \\textbf{{{comp.get('mean_kl', 0):.3f}}} & "
            f"\\textbf{{{comp.get('mean_tv', 0):.3f}}} & "
            f"\\textbf{{{comp.get('mean_coverage', 0) * 100:.0f}\\%}} & "
            f"\\textbf{{{comp.get('mode_match_rate', 0) * 100:.0f}\\%}} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ])

    latex_str = "\n".join(lines)

    # Optionally save to file
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        latex_path = output_dir / f"{results.get('schema', 'results')}_tables.tex"
        with open(latex_path, "w") as f:
            f.write(latex_str)
        print(f"  LaTeX tables saved to: {latex_path}")

    return latex_str


def main():
    parser = argparse.ArgumentParser(
        description="ICML 2026 Evaluation: TCT vs BPE+XGrammar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model options
    parser.add_argument("--tct_checkpoint", type=str, help="TCT model checkpoint directory")
    parser.add_argument("--utf8_checkpoint", type=str, help="UTF8-BPE model checkpoint directory")
    parser.add_argument("--tct_epoch", type=int, help="TCT checkpoint epoch (default: latest)")
    parser.add_argument("--utf8_epoch", type=int, help="UTF8 checkpoint epoch (default: latest)")
    parser.add_argument("--random_model", action="store_true", help="Use random model for testing")

    # Schema options
    parser.add_argument("--schema", type=str, required=True,
                        choices=["tsconfig", "eslintrc", "kubernetes"],
                        help="Schema to evaluate")

    # Data options
    parser.add_argument("--data_dir", type=str, help="Data directory (auto-detect if not specified)")
    parser.add_argument("--merge_table", type=str, help="BPE merge table (auto-detect if not specified)")

    # Evaluation options - what to run
    parser.add_argument("--bpb_only", action="store_true",
                        help="Only run BPB evaluation (skip generation quality)")
    parser.add_argument("--generation_only", action="store_true",
                        help="Only run generation quality evaluation (skip BPB)")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples for BPB evaluation")
    parser.add_argument("--num_gen_samples", type=int, default=100,
                        help="Number of samples to generate for distribution comparison")
    parser.add_argument("--max_seq_len", type=int, default=2048,
                        help="Maximum sequence length for BPB evaluation (default: 2048, training length)")
    parser.add_argument("--max_gen_tokens", type=int, default=2048,
                        help="Maximum tokens per generated sample")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for generation")
    parser.add_argument("--normalize_bytes", action="store_true", default=True,
                        help="Normalize bytes to minified JSON for fair comparison (default: True)")
    parser.add_argument("--no_normalize_bytes", dest="normalize_bytes", action="store_false",
                        help="Disable byte normalization (use raw decoded bytes)")

    # Output options
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    parser.add_argument("--latex_dir", type=str, help="Directory for LaTeX output (default: same as --output)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda/cpu)")

    args = parser.parse_args()

    # Validate conflicting options
    if args.bpb_only and args.generation_only:
        print("ERROR: Cannot use both --bpb_only and --generation_only")
        sys.exit(1)

    # Validate arguments
    if not args.random_model and not args.tct_checkpoint and not args.utf8_checkpoint:
        print("ERROR: Must specify at least one of --tct_checkpoint, --utf8_checkpoint, or --random_model")
        sys.exit(1)

    # Find merge table and data directories
    merge_table = Path(args.merge_table) if args.merge_table else find_merge_table(args.schema)
    utf8_data_dir = Path(args.data_dir) if args.data_dir else find_data_dir(args.schema, "utf8")

    print("=" * 60)
    print("ICML 2026 EVALUATION: TCT vs BPE+XGrammar")
    print("=" * 60)
    print(f"Schema:      {args.schema}")
    print(f"Merge table: {merge_table}")
    print(f"UTF8 data:   {utf8_data_dir}")
    print(f"Samples:     {args.num_samples}")
    print(f"Device:      {args.device}")
    print(f"Normalize bytes: {args.normalize_bytes} (minified JSON for fair comparison)")

    results = {
        "schema": args.schema,
        "num_samples": args.num_samples,
        "max_seq_len": args.max_seq_len,
        "normalize_bytes": args.normalize_bytes,
        "merge_table": str(merge_table),
        "utf8_data_dir": str(utf8_data_dir),
    }

    # Load UTF8-BPE decoder for vocab size
    from nanochat.xgrammar_tokenizer import UTF8BPEDecoder
    utf8_decoder = UTF8BPEDecoder(merge_table)
    vocab_size = utf8_decoder.vocab_size()

    # Try to find TCT data directory
    tct_data_dir = None
    if args.tct_checkpoint:
        try:
            tct_data_dir = find_data_dir(args.schema, "tct")
        except FileNotFoundError:
            pass

    # Pre-load validation tokens with auto-increase to reach target valid samples
    print(f"\n  Loading validation data (target: {args.num_samples} valid samples)...")
    utf8_validation_tokens, tct_validation_tokens, valid_indices = load_validation_tokens_with_target(
        utf8_data_dir=utf8_data_dir,
        tct_data_dir=tct_data_dir,
        target_valid=args.num_samples,
        max_seq_len=args.max_seq_len,
    )
    print(f"  UTF8 validation tokens: {len(utf8_validation_tokens)} samples")
    if tct_validation_tokens:
        print(f"  TCT validation tokens: {len(tct_validation_tokens)} samples")
        results["tct_data_dir"] = str(tct_data_dir)

    results["num_valid_samples"] = len(utf8_validation_tokens)

    # Evaluate UTF8-BPE model (with constrained BPB)
    if args.utf8_checkpoint or args.random_model:
        print("\n" + "-" * 60)
        print("UTF8-BPE MODEL")
        print("-" * 60)

        if args.random_model:
            print("Creating random model for testing...")
            model, config = create_random_model(vocab_size, args.device)
        else:
            print(f"Loading model from {args.utf8_checkpoint}")
            model, config = load_model(Path(args.utf8_checkpoint), args.device, args.utf8_epoch)

        print(f"  vocab_size: {config.get('vocab_size')}")
        print(f"  n_layer: {config.get('n_layer')}")

        # Run constrained BPB (unless --generation_only)
        if not args.generation_only:
            bpb_results = run_constrained_bpb(
                model=model,
                schema=args.schema,
                merge_table=merge_table,
                validation_tokens=utf8_validation_tokens,  # Pre-filtered for fair comparison
                device=args.device,
                normalize_bytes=args.normalize_bytes,
            )
            results["utf8_constrained_bpb"] = bpb_results

        # Run generation quality (unless --bpb_only)
        if not args.bpb_only:
            # Raw UTF8 generation (no grammar constraints)
            raw_gen_results = run_generation_quality_utf8_raw(
                model=model,
                schema=args.schema,
                merge_table=merge_table,
                data_dir=utf8_data_dir,
                num_samples=args.num_gen_samples,
                device=args.device,
                temperature=args.temperature,
                max_tokens=args.max_gen_tokens,
                normalize_json_output=args.normalize_bytes,
            )
            results["utf8_raw_generation"] = raw_gen_results

            # UTF8 + XGrammar generation (grammar constrained)
            gen_results = run_generation_quality_utf8(
                model=model,
                schema=args.schema,
                merge_table=merge_table,
                data_dir=utf8_data_dir,
                num_samples=args.num_gen_samples,
                device=args.device,
                temperature=args.temperature,
                max_tokens=args.max_gen_tokens,
                normalize_json_output=args.normalize_bytes,
            )
            results["utf8_generation"] = gen_results

        # Clean up
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Evaluate TCT model (generation quality only - no XGrammar needed)
    if args.tct_checkpoint:
        from nanochat.xgrammar_tokenizer import get_tct_module, compute_tct_bpb

        print("\n" + "-" * 60)
        print("TCT MODEL")
        print("-" * 60)

        # Load TCT module for this schema
        tct_module = get_tct_module(args.schema)
        tct_vocab_size = tct_module.vocab_size()
        print(f"  TCT vocab size: {tct_vocab_size}")

        # Load model
        print(f"Loading model from {args.tct_checkpoint}")
        model, full_config = load_model(Path(args.tct_checkpoint), args.device, args.tct_epoch)
        model_cfg = full_config.get("model_config", full_config)
        schema_cfg = full_config.get("schema_config", {})
        print(f"  vocab_size: {model_cfg.get('vocab_size')}")
        print(f"  n_layer: {model_cfg.get('n_layer')}")
        print(f"  val_loss: {full_config.get('val_loss', 'N/A')}")

        # Use pre-filtered tct_validation_tokens if available (from fair comparison section)
        # Otherwise, try to find data dir and load tokens
        if tct_validation_tokens is None:
            # Try to find TCT data dir from checkpoint config or discovery
            tct_data_dir_local = None
            if schema_cfg.get("data_path_tct"):
                candidate = Path(schema_cfg["data_path_tct"])
                if candidate.exists():
                    tct_data_dir_local = candidate
                else:
                    # Try relative to home/Desktop/data with the directory name
                    data_dir_name = schema_cfg.get("data_dir_tct")
                    if data_dir_name:
                        candidate = Path.home() / "Desktop" / "data" / data_dir_name
                        if candidate.exists():
                            tct_data_dir_local = candidate
            if tct_data_dir_local is None:
                try:
                    tct_data_dir_local = find_data_dir(args.schema, "tct")
                except FileNotFoundError:
                    tct_data_dir_local = None

            if tct_data_dir_local:
                tct_data_dir = tct_data_dir_local
                print(f"  TCT data dir: {tct_data_dir}")
                # Load tokens with auto-increase to match target (TCT-only path)
                tct_validation_tokens, _, _ = load_validation_tokens_with_target(
                    utf8_data_dir=tct_data_dir,  # Use TCT path as "utf8" (single tokenizer path)
                    tct_data_dir=None,  # No second tokenizer
                    target_valid=args.num_samples,
                    max_seq_len=args.max_seq_len,
                )
                print(f"  TCT validation tokens: {len(tct_validation_tokens)}")
            else:
                print(f"  WARNING: No TCT data directory found. TCT BPB cannot be computed.")

        if tct_validation_tokens is not None:
            print(f"  Using {len(tct_validation_tokens)} pre-filtered TCT validation sequences")

            # Compute TCT BPB (unless --generation_only)
            if not args.generation_only:
                print("\n  Computing TCT BPB...")
                tct_bpb_result = compute_tct_bpb(
                    model=model,
                    tct_module=tct_module,
                    validation_tokens=tct_validation_tokens,  # Pre-filtered for fair comparison
                    device=args.device,
                    max_seq_len=None,  # Already pre-filtered
                    show_progress=True,
                    normalize_bytes=args.normalize_bytes,
                )
                print(f"\n  TCT BPB Results:")
                print(f"    BPB:         {tct_bpb_result.bpb:.4f}")
                print(f"    Sequences:   {tct_bpb_result.num_sequences}")
                print(f"    Tokens:      {tct_bpb_result.total_tokens}")
                print(f"    Bytes:       {tct_bpb_result.total_bytes}")

                results["tct_bpb"] = {
                    "bpb": tct_bpb_result.bpb,
                    "num_sequences": tct_bpb_result.num_sequences,
                    "total_tokens": tct_bpb_result.total_tokens,
                    "total_bytes": tct_bpb_result.total_bytes,
                    "total_loss_nats": tct_bpb_result.total_loss,
                }

            # Decode TCT tokens to get validation JSON strings (for generation quality)
            val_json_strings = []
            for tokens in tct_validation_tokens[:args.num_gen_samples]:
                try:
                    json_out, consumed, surplus = tct_module.decode(tokens)
                    val_json_strings.append(json_out)
                except Exception:
                    continue
        else:
            print(f"  WARNING: No TCT validation tokens available. Using UTF8 validation data for generation comparison.")
            # Fallback: use UTF8 validation data decoded for comparison
            val_json_strings = [utf8_decoder.decode(tokens) for tokens in utf8_validation_tokens[:args.num_gen_samples]]

        # Run generation quality for TCT (unless --bpb_only)
        if not args.bpb_only:
            # Compute first token distribution for generation
            print("  Computing first token distribution for generation...")
            first_token_dist = compute_first_token_distribution(tct_data_dir)
            print(f"  First tokens: {len(first_token_dist)} unique, top: {list(first_token_dist.keys())[:5]}")

            tct_gen_results = run_generation_quality_tct(
                model=model,
                schema=args.schema,
                tct_module=tct_module,
                validation_json_strings=val_json_strings,
                first_token_distribution=first_token_dist,
                num_samples=args.num_gen_samples,
                temperature=args.temperature,
                max_tokens=args.max_gen_tokens,
                normalize_json_output=args.normalize_bytes,
            )
            results["tct_generation"] = tct_gen_results

        # Clean up
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # BPB Comparison Table (all three metrics)
    if "utf8_constrained_bpb" in results or "tct_bpb" in results:
        print("\n  BPB Comparison (lower is better):")
        print(f"  {'Method':<25} {'BPB':>10} {'Sequences':>12} {'Bytes':>12}")
        print(f"  {'-'*59}")

        if "utf8_constrained_bpb" in results:
            bpb = results["utf8_constrained_bpb"]
            # Raw BPE (no grammar constraints)
            print(f"  {'UTF8-BPE (raw)':<25} {bpb['raw_bpb']:>10.4f} {bpb['num_sequences']:>12} {bpb['total_bytes']:>12}")
            # BPE + XGrammar
            print(f"  {'UTF8-BPE + XGrammar':<25} {bpb['constrained_bpb']:>10.4f} {bpb['num_sequences']:>12} {bpb['total_bytes']:>12}")

        if "tct_bpb" in results:
            bpb = results["tct_bpb"]
            # TCT (inherently constrained)
            print(f"  {'TCT':<25} {bpb['bpb']:>10.4f} {bpb['num_sequences']:>12} {bpb['total_bytes']:>12}")

        # Fairness verification
        if "utf8_constrained_bpb" in results and "tct_bpb" in results:
            utf8_bpb = results["utf8_constrained_bpb"]
            tct_bpb = results["tct_bpb"]
            print(f"\n  Fairness Verification:")
            if utf8_bpb['num_sequences'] == tct_bpb['num_sequences']:
                print(f"    ✓ Same number of sequences: {utf8_bpb['num_sequences']}")
            else:
                print(f"    ✗ DIFFERENT sequence counts: UTF8={utf8_bpb['num_sequences']}, TCT={tct_bpb['num_sequences']}")
            if utf8_bpb['total_bytes'] == tct_bpb['total_bytes']:
                print(f"    ✓ Same total bytes: {utf8_bpb['total_bytes']}")
            else:
                print(f"    ✗ DIFFERENT byte counts: UTF8={utf8_bpb['total_bytes']}, TCT={tct_bpb['total_bytes']}")
                print(f"      (This is UNFAIR - bytes should match for normalized JSON)")

    if "utf8_generation" in results and "tct_generation" in results:
        utf8 = results["utf8_generation"].get("comparison", {})
        tct = results["tct_generation"].get("comparison", {})
        print(f"\nGeneration Quality Comparison:")
        print(f"  {'Metric':<20} {'UTF8+XGrammar':>15} {'TCT':>15}")
        print(f"  {'-'*50}")
        print(f"  {'Mean KL Divergence':<20} {utf8.get('mean_kl', 0):>15.4f} {tct.get('mean_kl', 0):>15.4f}")
        print(f"  {'Mean TV Distance':<20} {utf8.get('mean_tv', 0):>15.4f} {tct.get('mean_tv', 0):>15.4f}")
        print(f"  {'Coverage':<20} {utf8.get('mean_coverage', 0):>14.1%} {tct.get('mean_coverage', 0):>14.1%}")
        print(f"  {'Mode Match Rate':<20} {utf8.get('mode_match_rate', 0):>14.1%} {tct.get('mode_match_rate', 0):>14.1%}")

        # Throughput comparison
        utf8_stats = results["utf8_generation"].get("generation_stats", {})
        tct_stats = results["tct_generation"].get("generation_stats", {})
        if utf8_stats.get("tokens_per_second") and tct_stats.get("tokens_per_second"):
            print(f"\nGeneration Throughput:")
            print(f"  {'Metric':<20} {'UTF8+XGrammar':>15} {'TCT':>15}")
            print(f"  {'-'*50}")
            print(f"  {'Tokens/sec':<20} {utf8_stats.get('tokens_per_second', 0):>15.0f} {tct_stats.get('tokens_per_second', 0):>15.0f}")
            print(f"  {'Samples/sec':<20} {utf8_stats.get('samples_per_second', 0):>15.2f} {tct_stats.get('samples_per_second', 0):>15.2f}")
            print(f"  {'Time (sec)':<20} {utf8_stats.get('generation_time_seconds', 0):>15.1f} {tct_stats.get('generation_time_seconds', 0):>15.1f}")
            speedup = tct_stats.get('tokens_per_second', 0) / utf8_stats.get('tokens_per_second', 1) if utf8_stats.get('tokens_per_second', 0) > 0 else 0
            print(f"  {'TCT Speedup':<20} {speedup:>15.2f}x")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    # Generate LaTeX tables
    if args.latex:
        latex_dir = Path(args.latex_dir) if args.latex_dir else (Path(args.output).parent if args.output else Path("."))
        print("\n" + "=" * 60)
        print("LATEX TABLES")
        print("=" * 60)
        latex_str = generate_latex_tables(results, latex_dir)
        print("\nGenerated LaTeX:")
        print(latex_str[:500] + "..." if len(latex_str) > 500 else latex_str)

    print("\nDone!")


if __name__ == "__main__":
    main()
