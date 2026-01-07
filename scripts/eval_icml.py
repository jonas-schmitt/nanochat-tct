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
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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
    num_samples: int,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    show_progress: bool = True,
) -> List[str]:
    """Generate samples using UTF8-BPE model with XGrammar constraints.

    Returns:
        List of generated JSON strings (only valid ones)
    """
    import time
    import torch
    import torch.nn.functional as F
    import xgrammar
    from tqdm import tqdm

    device = next(model.parameters()).device
    generated_texts = []
    valid_count = 0
    empty_count = 0
    failed_count = 0

    iterator = tqdm(range(num_samples), desc="Generating samples") if show_progress else range(num_samples)

    for _ in iterator:
        try:
            # Create fresh matcher for each generation
            matcher = xgrammar.GrammarMatcher(compiled_grammar)
            bitmask = xgrammar.allocate_token_bitmask(1, tokenizer_info.vocab_size)

            # UTF8-BPE sequences start with token 123 (the '{' character)
            start_token = 123
            generated_tokens = [start_token]
            matcher.accept_token(start_token)

            for step in range(max_tokens - 1):
                if matcher.is_terminated():
                    break

                # Get allowed tokens from grammar
                xgrammar.reset_token_bitmask(bitmask)
                matcher.fill_next_token_bitmask(bitmask)

                # Get model logits
                input_ids = torch.tensor([generated_tokens], device=device)

                with torch.no_grad():
                    logits = model(input_ids)
                    logits = logits[0, -1, :]

                # Apply grammar mask
                logits_batch = logits.unsqueeze(0)
                xgrammar.apply_token_bitmask_inplace(logits_batch, bitmask.to(device))
                logits = logits_batch[0]

                # Apply temperature and top-k
                if temperature > 0:
                    logits = logits / temperature

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[-1]] = float('-inf')

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                # Accept token
                matcher.accept_token(next_token)
                generated_tokens.append(next_token)

            # Decode
            text = utf8_decoder.decode(generated_tokens)

            # Validate JSON and exclude empty objects
            try:
                import json
                parsed = json.loads(text)
                # Exclude empty objects for fair comparison with TCT
                if parsed and parsed != {}:
                    generated_texts.append(text)
                    valid_count += 1
                else:
                    empty_count += 1
            except json.JSONDecodeError:
                failed_count += 1

        except Exception as e:
            failed_count += 1
            continue

    if show_progress:
        print(f"  Generated: {valid_count} valid, {empty_count} empty (excluded), {failed_count} failed")

    return generated_texts


def generate_samples_tct(
    model,
    tct_module,
    num_samples: int,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_k: int = 50,
    show_progress: bool = True,
) -> List[str]:
    """Generate samples using TCT model.

    TCT is inherently constrained - all outputs are valid by construction.
    Unlike UTF8+XGrammar, TCT has no explicit EOS token - sequences are complete
    when they reach a target length. We generate max_tokens and decode.

    Args:
        model: The GPT model
        tct_module: TCT tokenizer module (encode/decode/decode_prefix/vocab_size)
        num_samples: Number of samples to generate
        max_tokens: Maximum tokens per sample (should match typical training lengths)
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        show_progress: Whether to show progress bar

    Returns:
        List of generated JSON strings (all valid by construction)
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    device = next(model.parameters()).device
    generated_texts = []
    decode_success = 0
    decode_partial = 0
    decode_empty = 0

    iterator = tqdm(range(num_samples), desc="Generating TCT samples") if show_progress else range(num_samples)

    for _ in iterator:
        # TCT sequences always start with token 0
        generated_tokens = [0]

        # Generate max_tokens - 1 more (we already have token 0)
        for step in range(max_tokens - 1):
            # Get model logits for next token
            input_ids = torch.tensor([generated_tokens], device=device)

            with torch.no_grad():
                logits = model(input_ids)
                logits = logits[0, -1, :]  # Last position

            # Apply temperature and top-k
            if temperature > 0:
                logits = logits / temperature

            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated_tokens.append(next_token)

        # Decode result using decode_prefix which handles extra tokens gracefully
        # (model may generate tokens beyond the complete JSON)
        try:
            json_out, consumed, is_complete = tct_module.decode_prefix(generated_tokens)

            if is_complete and json_out and json_out != "{}":
                # Complete, non-empty JSON - count as success
                generated_texts.append(json_out)
                decode_success += 1
            elif is_complete:
                # Complete but empty {} - track separately
                decode_empty += 1
            elif json_out and json_out != "{}":
                # Partial JSON (not complete) - track but don't include
                decode_partial += 1
            else:
                # Empty partial - count as empty
                decode_empty += 1
        except Exception:
            # Decode failed entirely
            continue

    if show_progress:
        print(f"  Generated: {decode_success} valid, {decode_empty} empty (excluded), {decode_partial} partial (excluded)")

    return generated_texts


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
    generated_texts = generate_samples_xgrammar(
        model=model,
        tokenizer_info=tokenizer_info,
        compiled_grammar=compiled_grammar,
        utf8_decoder=utf8_decoder,
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
        "ground_truth_distributions": real_result.to_dict()["field_distributions"],
        "generated_distributions": gen_result.to_dict()["field_distributions"],
        "comparison": comparison.to_dict(),
    }


def run_generation_quality_tct(
    model,
    schema: str,
    tct_module,
    validation_json_strings: List[str],
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
    generated_texts = generate_samples_tct(
        model=model,
        tct_module=tct_module,
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
        "model_type": "tct",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "validation_samples": real_result.num_valid,
        "generated_samples": gen_result.num_valid,
        "ground_truth_distributions": real_result.to_dict()["field_distributions"],
        "generated_distributions": gen_result.to_dict()["field_distributions"],
        "comparison": comparison.to_dict(),
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
    parser.add_argument("--max_gen_tokens", type=int, default=512,
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

    # Pre-load validation tokens and compute valid indices for fair comparison
    print("\n  Loading validation data...")
    utf8_validation_tokens = load_validation_tokens(utf8_data_dir, args.num_samples)
    print(f"  UTF8 validation tokens: {len(utf8_validation_tokens)} samples")

    # Try to find TCT data directory
    tct_data_dir = None
    if args.tct_checkpoint:
        try:
            tct_data_dir = find_data_dir(args.schema, "tct")
        except FileNotFoundError:
            pass

    # If both tokenizations available, compute valid indices for fair comparison
    if tct_data_dir:
        tct_validation_tokens = load_validation_tokens(tct_data_dir, args.num_samples)
        print(f"  TCT validation tokens: {len(tct_validation_tokens)} samples")

        # Compute valid indices (samples that fit in max_seq_len for BOTH tokenizers)
        valid_indices = compute_valid_indices(
            utf8_validation_tokens, tct_validation_tokens, args.max_seq_len
        )
        print(f"  Valid for both (max_seq_len={args.max_seq_len}): {len(valid_indices)} samples")

        # Filter to valid indices only
        utf8_validation_tokens = [utf8_validation_tokens[i] for i in valid_indices]
        tct_validation_tokens = [tct_validation_tokens[i] for i in valid_indices]

        results["num_valid_samples"] = len(valid_indices)
        results["tct_data_dir"] = str(tct_data_dir)
    else:
        tct_validation_tokens = None
        # For UTF8-only evaluation, filter by max_seq_len
        utf8_validation_tokens = [t for t in utf8_validation_tokens if len(t) <= args.max_seq_len]
        print(f"  After filtering (max_seq_len={args.max_seq_len}): {len(utf8_validation_tokens)} samples")
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
                # Load and filter tokens (UTF8-only comparison path)
                tct_validation_tokens = load_validation_tokens(tct_data_dir, args.num_samples)
                tct_validation_tokens = [t for t in tct_validation_tokens if len(t) <= args.max_seq_len]
                print(f"  TCT validation tokens: {len(tct_validation_tokens)} (filtered to max_seq_len={args.max_seq_len})")
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
            tct_gen_results = run_generation_quality_tct(
                model=model,
                schema=args.schema,
                tct_module=tct_module,
                validation_json_strings=val_json_strings,
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
