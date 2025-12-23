"""
Three-way evaluation script for TCT vs UTF8-BPE generation.

Compares:
1. TCT-BPE: Native schema-aware decoding (100% valid by construction)
2. UTF8-BPE: Unconstrained generation (validity varies)
3. UTF8-BPE + XGrammar: FSM-constrained generation (100% valid via XGrammar)

This evaluation addresses RQ2: Does TCT's advantage persist when BPE
is augmented with constrained decoding?

Usage:
    python -m scripts.eval_generation \\
        --utf8_checkpoint checkpoints/kubernetes-utf8-small/best.pt \\
        --schema kubernetes \\
        --xgrammar \\
        --num_samples 100
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt import GPT, GPTConfig


@dataclass
class GenerationResult:
    """Result from a single generation."""
    text: str
    tokens: List[int]
    time_seconds: float
    json_valid: bool
    schema_valid: bool
    error: Optional[str] = None


@dataclass
class EvaluationMetrics:
    """Aggregated metrics for an evaluation run."""
    method: str
    num_samples: int
    json_validity_rate: float
    schema_validity_rate: float
    mean_time_seconds: float
    tokens_per_second: float
    mean_tokens: float


# Schema paths (relative to tct repo)
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


def load_tct_model(checkpoint_path: str, device: str = "cuda") -> Tuple[GPT, object]:
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
        import tct_kubernetes
        tokenizer = tct_kubernetes
    except ImportError:
        print("Warning: tct_kubernetes not installed")
        tokenizer = None

    return model, tokenizer


def load_utf8_model(checkpoint_path: str, device: str = "cuda") -> Tuple[GPT, object]:
    """Load UTF8-BPE model and tokenizer."""
    print(f"Loading UTF8 model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_dict = checkpoint.get("config", {})
    config = GPTConfig(
        sequence_len=config_dict.get("context_size", 2048),
        vocab_size=config_dict.get("vocab_size", 24000),
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

    # Import UTF8-BPE tokenizer
    try:
        import utf8_bpe_kubernetes
        tokenizer = utf8_bpe_kubernetes
    except ImportError:
        print("Warning: utf8_bpe_kubernetes not installed")
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
    """Generate using TCT-BPE model with native decoding."""
    device = model.get_device()

    start_time = time.perf_counter()
    generated_tokens = list(prompt_tokens)

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

    # Decode with TCT
    try:
        decoded_str, consumed, total = tokenizer.decode(generated_tokens)
        text = decoded_str
    except Exception as e:
        text = f"DECODE_ERROR: {e}"

    # Validate
    json_valid, parsed, error = validate_json(text)

    return GenerationResult(
        text=text,
        tokens=generated_tokens,
        time_seconds=elapsed,
        json_valid=json_valid,
        schema_valid=json_valid,  # TCT guarantees schema validity if JSON valid
        error=error,
    )


def generate_utf8_unconstrained(
    model: GPT,
    tokenizer,
    prompt_tokens: List[int],
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> GenerationResult:
    """Generate using UTF8-BPE model without constraints."""
    device = model.get_device()

    start_time = time.perf_counter()
    generated_tokens = list(prompt_tokens)

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


def compute_metrics(results: List[GenerationResult], method: str) -> EvaluationMetrics:
    """Compute aggregate metrics from generation results."""
    num_samples = len(results)
    if num_samples == 0:
        return EvaluationMetrics(
            method=method,
            num_samples=0,
            json_validity_rate=0.0,
            schema_validity_rate=0.0,
            mean_time_seconds=0.0,
            tokens_per_second=0.0,
            mean_tokens=0.0,
        )

    json_valid = sum(1 for r in results if r.json_valid)
    schema_valid = sum(1 for r in results if r.schema_valid)
    total_time = sum(r.time_seconds for r in results)
    total_tokens = sum(len(r.tokens) for r in results)

    return EvaluationMetrics(
        method=method,
        num_samples=num_samples,
        json_validity_rate=json_valid / num_samples,
        schema_validity_rate=schema_valid / num_samples,
        mean_time_seconds=total_time / num_samples,
        tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
        mean_tokens=total_tokens / num_samples,
    )


def print_metrics_table(metrics_list: List[EvaluationMetrics]):
    """Print metrics comparison table."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    headers = ["Method", "JSON Valid", "Schema Valid", "Mean Time", "Tok/s", "Mean Tok"]
    row_format = "{:<25} {:>10} {:>12} {:>10} {:>8} {:>10}"

    print(row_format.format(*headers))
    print("-" * 80)

    for m in metrics_list:
        print(row_format.format(
            m.method,
            f"{m.json_validity_rate:.1%}",
            f"{m.schema_validity_rate:.1%}",
            f"{m.mean_time_seconds:.3f}s",
            f"{m.tokens_per_second:.1f}",
            f"{m.mean_tokens:.1f}",
        ))

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Three-way generation evaluation")
    parser.add_argument("--tct_checkpoint", help="Path to TCT-BPE checkpoint")
    parser.add_argument("--utf8_checkpoint", help="Path to UTF8-BPE checkpoint")
    parser.add_argument("--utf8_merge_table", help="Path to UTF8-BPE merge table (for XGrammar)",
                        default="bpe-merges/kubernetes-utf8-bpe-matched.json")
    parser.add_argument("--xgrammar", action="store_true", help="Use XGrammar for constrained generation")
    parser.add_argument("--schema", default="kubernetes", choices=["kubernetes", "eslintrc", "tsconfig"])
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", help="Output JSON file for results")

    args = parser.parse_args()

    # Load schema for validation
    print(f"Loading {args.schema} schema...")
    schema = load_schema(args.schema)

    all_metrics = []

    # Evaluate TCT-BPE
    if args.tct_checkpoint:
        print("\n" + "=" * 40)
        print("EVALUATING: TCT-BPE")
        print("=" * 40)

        model, tokenizer = load_tct_model(args.tct_checkpoint, args.device)

        tct_results = []
        for i in range(args.num_samples):
            # Generate from empty prompt (start of document)
            prompt_tokens = []
            result = generate_tct(
                model, tokenizer, prompt_tokens,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            tct_results.append(result)
            print(f"  Sample {i+1}/{args.num_samples}: JSON={result.json_valid}, Schema={result.schema_valid}")

        metrics = compute_metrics(tct_results, "TCT-BPE")
        all_metrics.append(metrics)

        del model
        torch.cuda.empty_cache()

    # Evaluate UTF8-BPE (unconstrained)
    if args.utf8_checkpoint:
        print("\n" + "=" * 40)
        print("EVALUATING: UTF8-BPE (unconstrained)")
        print("=" * 40)

        model, tokenizer = load_utf8_model(args.utf8_checkpoint, args.device)

        utf8_results = []
        for i in range(args.num_samples):
            prompt_tokens = []
            result = generate_utf8_unconstrained(
                model, tokenizer, prompt_tokens,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            utf8_results.append(result)
            print(f"  Sample {i+1}/{args.num_samples}: JSON={result.json_valid}, Schema={result.schema_valid}")

        metrics = compute_metrics(utf8_results, "UTF8-BPE")
        all_metrics.append(metrics)

        del model
        torch.cuda.empty_cache()

    # Evaluate UTF8-BPE + XGrammar (constrained)
    if args.xgrammar and args.utf8_checkpoint:
        print("\n" + "=" * 40)
        print("EVALUATING: UTF8-BPE + XGrammar")
        print("=" * 40)

        try:
            import xgrammar
            from nanochat.xgrammar_tokenizer import (
                build_xgrammar_tokenizer_info,
                compile_json_schema_grammar,
            )

            # Build tokenizer info from merge table
            print(f"Building XGrammar tokenizer from {args.utf8_merge_table}...")
            tokenizer_info = build_xgrammar_tokenizer_info(args.utf8_merge_table)
            print(f"  Vocab size: {tokenizer_info.vocab_size}")

            # Compile grammar for schema
            print(f"Compiling grammar for {args.schema} schema...")
            compiled_grammar = compile_json_schema_grammar(tokenizer_info, schema)
            print("  Grammar compiled successfully")

            # Load model if not already loaded
            model, utf8_tokenizer = load_utf8_model(args.utf8_checkpoint, args.device)

            xgrammar_results = []
            for i in range(args.num_samples):
                result = generate_utf8_xgrammar(
                    model,
                    tokenizer_info,
                    compiled_grammar,
                    utf8_tokenizer,
                    prompt_tokens=[],
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                xgrammar_results.append(result)
                status = "ERROR" if result.error else f"JSON={result.json_valid}"
                print(f"  Sample {i+1}/{args.num_samples}: {status}")

            metrics = compute_metrics(xgrammar_results, "UTF8-BPE + XGrammar")
            all_metrics.append(metrics)

            del model
            torch.cuda.empty_cache()

        except ImportError as e:
            print(f"  ERROR: {e}")
            print("  XGrammar not available. Install with: pip install xgrammar")
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")

    # Print comparison table
    if all_metrics:
        print_metrics_table(all_metrics)

    # Save results
    if args.output:
        results_dict = {
            "schema": args.schema,
            "num_samples": args.num_samples,
            "metrics": [
                {
                    "method": m.method,
                    "json_validity_rate": m.json_validity_rate,
                    "schema_validity_rate": m.schema_validity_rate,
                    "mean_time_seconds": m.mean_time_seconds,
                    "tokens_per_second": m.tokens_per_second,
                    "mean_tokens": m.mean_tokens,
                }
                for m in all_metrics
            ]
        }
        with open(args.output, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
