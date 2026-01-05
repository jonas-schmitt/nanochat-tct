#!/usr/bin/env python3
"""
ICML 2026 Evaluation Script: TCT vs BPE+XGrammar

Runs all evaluation metrics from EVALUATION_PLAN.md:
1. Constrained BPB - Information-theoretic comparison with XGrammar masking
2. Generation Quality - Field value distribution comparison
3. Semantic Accuracy - P(correct) at decision points (future work)

Usage:
    # Full evaluation for tsconfig
    python -m scripts.eval_icml \
        --schema tsconfig \
        --tct_checkpoint checkpoints/tsconfig_tct_medium/ \
        --utf8_checkpoint checkpoints/tsconfig_utf8_medium/ \
        --num_samples 1000 \
        --output results/icml/tsconfig.json

    # Just constrained BPB (faster)
    python -m scripts.eval_icml \
        --schema tsconfig \
        --utf8_checkpoint checkpoints/tsconfig_utf8_medium/ \
        --constrained_bpb_only \
        --output results/icml/tsconfig_bpb.json

    # Test with random model
    python -m scripts.eval_icml \
        --schema tsconfig \
        --random_model \
        --num_samples 100 \
        --device cpu
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model(checkpoint_dir: Path, device: str = "cuda"):
    """Load model from checkpoint directory.

    Returns the model and full config dict (including schema_config for data paths).
    Prioritizes: best.pt > checkpoint at min_val_loss epoch > latest checkpoint.
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

    # Find checkpoint file - prioritize best.pt, then checkpoint at config epoch
    checkpoint_path = checkpoint_dir / "best.pt"
    if not checkpoint_path.exists():
        # Try checkpoint at the epoch recorded in config (has the val_loss we want)
        config_epoch = config_dict.get("epoch")
        if config_epoch is not None:
            epoch_path = checkpoint_dir / f"epoch_{config_epoch:03d}.pt"
            if epoch_path.exists() and epoch_path.stat().st_size > 0:
                checkpoint_path = epoch_path

    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
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


def find_merge_table(schema: str) -> Path:
    """Find BPE merge table for schema."""
    bpe_dir = Path(__file__).parent.parent / "bpe-merges"
    candidates = list(bpe_dir.glob(f"{schema}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No merge table for {schema} in {bpe_dir}")
    return candidates[0]


def find_data_dir(schema: str, tokenizer: str = "utf8") -> Path:
    """Find data directory for schema.

    Args:
        schema: Schema name (tsconfig, eslintrc, kubernetes)
        tokenizer: Tokenizer type ("utf8" or "tct")
    """
    data_roots = [
        Path(__file__).parent.parent.parent / "data",
        Path.home() / "Desktop" / "data",
    ]

    if tokenizer == "tct":
        suffixes = ["-tct-bpe-500", "-tct-bpe-1k", "-tct-base", "-tct"]
    else:
        suffixes = ["-utf8-base-matched", "-utf8-bpe-500", "-utf8-bpe-1k", "-utf8"]

    candidates = []
    for root in data_roots:
        for suffix in suffixes:
            candidates.append(root / f"{schema}{suffix}")

    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No data directory for {schema} ({tokenizer}). Tried: {candidates}")


def run_constrained_bpb(
    model,
    schema: str,
    merge_table: Path,
    data_dir: Path,
    num_samples: int,
    max_seq_len: int,
    device: str,
) -> Dict[str, Any]:
    """Run constrained BPB evaluation."""
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

    # Load validation tokens
    print(f"  Loading validation tokens from {data_dir}...")
    validation_tokens = load_validation_tokens(data_dir, num_samples)
    print(f"  Loaded {len(validation_tokens)} sequences")

    # Compute constrained BPB
    print("\n  Computing constrained BPB...")
    result = compute_constrained_bpb(
        model=model,
        tokenizer_info=tokenizer_info,
        compiled_grammar=compiled_grammar,
        utf8_decoder=utf8_decoder,
        validation_tokens=validation_tokens,
        device=device,
        max_seq_len=max_seq_len,
        show_progress=True,
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

            # Validate JSON
            try:
                import json
                json.loads(text)
                generated_texts.append(text)
                valid_count += 1
            except json.JSONDecodeError:
                failed_count += 1

        except Exception as e:
            failed_count += 1
            continue

    if show_progress:
        print(f"  Generated: {valid_count} valid, {failed_count} failed")

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

        # Decode final result - try full decode first
        try:
            json_out, consumed, surplus = tct_module.decode(generated_tokens)
            generated_texts.append(json_out)
            decode_success += 1
        except Exception:
            # Full decode failed, get partial result via decode_prefix
            try:
                partial_json, consumed, is_complete = tct_module.decode_prefix(generated_tokens)
                if partial_json and partial_json != "{}":
                    generated_texts.append(partial_json)
                    decode_partial += 1
            except Exception:
                # Skip this sample entirely
                continue

    if show_progress:
        print(f"  Generated: {len(generated_texts)} samples ({decode_success} full decode, {decode_partial} partial)")

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
) -> Dict[str, Any]:
    """Run generation quality evaluation for UTF8-BPE model with XGrammar."""
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
    parser.add_argument("--random_model", action="store_true", help="Use random model for testing")

    # Schema options
    parser.add_argument("--schema", type=str, required=True,
                        choices=["tsconfig", "eslintrc", "kubernetes"],
                        help="Schema to evaluate")

    # Data options
    parser.add_argument("--data_dir", type=str, help="Data directory (auto-detect if not specified)")
    parser.add_argument("--merge_table", type=str, help="BPE merge table (auto-detect if not specified)")

    # Evaluation options
    parser.add_argument("--constrained_bpb_only", action="store_true",
                        help="Only run constrained BPB evaluation")
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples for BPB evaluation")
    parser.add_argument("--num_gen_samples", type=int, default=100,
                        help="Number of samples to generate for distribution comparison")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length for BPB evaluation")
    parser.add_argument("--max_gen_tokens", type=int, default=512,
                        help="Maximum tokens per generated sample")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for generation")

    # Output options
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    parser.add_argument("--latex_dir", type=str, help="Directory for LaTeX output (default: same as --output)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda/cpu)")

    args = parser.parse_args()

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

    results = {
        "schema": args.schema,
        "num_samples": args.num_samples,
        "max_seq_len": args.max_seq_len,
        "merge_table": str(merge_table),
        "utf8_data_dir": str(utf8_data_dir),
    }

    # Load UTF8-BPE decoder for vocab size
    from nanochat.xgrammar_tokenizer import UTF8BPEDecoder
    utf8_decoder = UTF8BPEDecoder(merge_table)
    vocab_size = utf8_decoder.vocab_size()

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
            model, config = load_model(Path(args.utf8_checkpoint), args.device)

        print(f"  vocab_size: {config.get('vocab_size')}")
        print(f"  n_layer: {config.get('n_layer')}")

        # Run constrained BPB
        bpb_results = run_constrained_bpb(
            model=model,
            schema=args.schema,
            merge_table=merge_table,
            data_dir=utf8_data_dir,
            num_samples=args.num_samples,
            max_seq_len=args.max_seq_len,
            device=args.device,
        )
        results["utf8_constrained_bpb"] = bpb_results

        # Run generation quality (if not constrained BPB only)
        if not args.constrained_bpb_only:
            gen_results = run_generation_quality_utf8(
                model=model,
                schema=args.schema,
                merge_table=merge_table,
                data_dir=utf8_data_dir,
                num_samples=args.num_gen_samples,
                device=args.device,
                temperature=args.temperature,
                max_tokens=args.max_gen_tokens,
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
        model, full_config = load_model(Path(args.tct_checkpoint), args.device)
        model_cfg = full_config.get("model_config", full_config)
        schema_cfg = full_config.get("schema_config", {})
        print(f"  vocab_size: {model_cfg.get('vocab_size')}")
        print(f"  n_layer: {model_cfg.get('n_layer')}")
        print(f"  val_loss: {full_config.get('val_loss', 'N/A')}")

        # For TCT BPB, use data path from checkpoint config, or fall back to discovery
        tct_data_dir = None
        if schema_cfg.get("data_path_tct"):
            candidate = Path(schema_cfg["data_path_tct"])
            if candidate.exists():
                tct_data_dir = candidate
            else:
                # Try relative to home/Desktop/data with the directory name
                data_dir_name = schema_cfg.get("data_dir_tct")
                if data_dir_name:
                    candidate = Path.home() / "Desktop" / "data" / data_dir_name
                    if candidate.exists():
                        tct_data_dir = candidate
        if tct_data_dir is None:
            try:
                tct_data_dir = find_data_dir(args.schema, "tct")
            except FileNotFoundError:
                tct_data_dir = None

        if tct_data_dir:
            print(f"  TCT data dir: {tct_data_dir}")
            # Load TCT validation tokens
            tct_validation_tokens = load_validation_tokens(tct_data_dir, args.num_samples)

            # Compute TCT BPB
            print("\n  Computing TCT BPB...")
            tct_bpb_result = compute_tct_bpb(
                model=model,
                tct_module=tct_module,
                validation_tokens=tct_validation_tokens,
                device=args.device,
                max_seq_len=args.max_seq_len,
                show_progress=True,
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

            # Decode TCT tokens to get validation JSON strings
            val_json_strings = []
            for tokens in tct_validation_tokens[:args.num_gen_samples]:
                try:
                    json_out, consumed, surplus = tct_module.decode(tokens)
                    val_json_strings.append(json_out)
                except Exception:
                    continue
        else:
            print(f"  WARNING: No TCT data directory found. Using UTF8 validation data for generation comparison.")
            # Fallback: use UTF8 validation data decoded for comparison
            from nanochat.xgrammar_tokenizer import UTF8BPEDecoder
            utf8_decoder = UTF8BPEDecoder(merge_table)
            utf8_validation_tokens = load_validation_tokens(utf8_data_dir, args.num_gen_samples)
            val_json_strings = [utf8_decoder.decode(tokens) for tokens in utf8_validation_tokens]

        # Run generation quality for TCT
        if not args.constrained_bpb_only:
            tct_gen_results = run_generation_quality_tct(
                model=model,
                schema=args.schema,
                tct_module=tct_module,
                validation_json_strings=val_json_strings,
                num_samples=args.num_gen_samples,
                temperature=args.temperature,
                max_tokens=args.max_gen_tokens,
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

    if "utf8_constrained_bpb" in results:
        bpb = results["utf8_constrained_bpb"]
        print(f"\nUTF8-BPE Constrained BPB:")
        print(f"  Raw BPB:         {bpb['raw_bpb']:.4f}")
        print(f"  Constrained BPB: {bpb['constrained_bpb']:.4f}")
        print(f"  Reduction:       {bpb['bpb_reduction_percent']:.1f}%")

    if "tct_bpb" in results:
        bpb = results["tct_bpb"]
        print(f"\nTCT BPB:")
        print(f"  BPB:             {bpb['bpb']:.4f}")
        print(f"  Sequences:       {bpb['num_sequences']}")

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
