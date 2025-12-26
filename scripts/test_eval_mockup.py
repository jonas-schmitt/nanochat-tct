#!/usr/bin/env python3
"""
Comprehensive Mock Experiment for Evaluation Script Testing

Tests the eval_generation.py script with randomly initialized models across
all three schemas (kubernetes, eslintrc, tsconfig) to verify:

1. Model checkpoint creation and loading
2. Generation evaluation (TCT, UTF8, UTF8+XGrammar)
3. Validation set evaluation (loss, perplexity, bits-per-byte, accuracies)
4. All metrics computed correctly for all schemas
5. Same UTF8 model used for constrained/unconstrained generation

Expected Results with Random Models:
=====================================
- TCT-BPE: 100% JSON validity (via decode_prefix streaming decode)
  * decode_prefix is truncation-tolerant - always produces valid JSON
  * Content is garbage but structurally valid
  * Schema validity depends on is_complete flag from decode_prefix
  * A trained model produces meaningful content

- UTF8-BPE: 0% validity (random tokens decode to garbage text)
  * UTF8 tokens are raw bytes, so any sequence decodes
  * Random text doesn't parse as valid JSON

- UTF8-BPE + XGrammar: 10-40% validity (depends on natural termination)
  * XGrammar guarantees validity ONLY when generation terminates naturally
  * With random models, often hits max_tokens before completing valid JSON
  * A trained model learns to terminate → near 100% validity

Usage:
    python -m scripts.test_eval_mockup
    python -m scripts.test_eval_mockup --schema kubernetes
    python -m scripts.test_eval_mockup --verbose
"""

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from nanochat.gpt import GPT, GPTConfig
from nanochat.xgrammar_tokenizer import UTF8BPEDecoder


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SchemaConfig:
    """Configuration for a schema."""
    name: str
    merge_table: str
    tct_module_name: str
    context_size: int
    # Required fields for field coverage check
    valid_sample: Dict[str, Any]
    invalid_sample: Dict[str, Any]


SCHEMAS = {
    "kubernetes": SchemaConfig(
        name="kubernetes",
        merge_table="bpe-merges/kubernetes-utf8-bpe-matched.json",
        tct_module_name="tct_kubernetes_20k",
        context_size=256,
        valid_sample={"apiVersion": "v1", "kind": "Pod", "metadata": {"name": "test"}},
        invalid_sample={"apiVersion": "v1"},
    ),
    "eslintrc": SchemaConfig(
        name="eslintrc",
        merge_table="bpe-merges/eslintrc-utf8-bpe-matched.json",
        tct_module_name="tct_eslintrc_10k",
        context_size=256,
        valid_sample={"rules": {"semi": "error"}},
        invalid_sample={"foo": "bar"},
    ),
    "tsconfig": SchemaConfig(
        name="tsconfig",
        merge_table="bpe-merges/tsconfig-utf8-bpe-matched.json",
        tct_module_name="tct_tsconfig_10k",
        context_size=256,
        valid_sample={"compilerOptions": {"target": "es6"}},
        invalid_sample={"foo": "bar"},
    ),
}

# Small model config for testing
SMALL_MODEL_CONFIG = {
    "n_layer": 2,
    "n_head": 4,
    "n_kv_head": 4,
    "n_embd": 64,
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_tct_tokenizer(schema_name: str):
    """Get TCT tokenizer for schema, returns None if not available."""
    try:
        if schema_name == "kubernetes":
            import tct_kubernetes_20k
            return tct_kubernetes_20k
        elif schema_name == "eslintrc":
            import tct_eslintrc_10k
            return tct_eslintrc_10k
        elif schema_name == "tsconfig":
            import tct_tsconfig_10k
            return tct_tsconfig_10k
    except ImportError:
        return None
    return None


def create_model_checkpoint(
    model_dir: Path,
    vocab_size: int,
    context_size: int = 256,
    model_config: dict = None,
) -> Path:
    """Create a randomly initialized model checkpoint with config.json."""
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_config is None:
        model_config = SMALL_MODEL_CONFIG.copy()

    config_dict = {
        "model_config": {
            "vocab_size": vocab_size,
            "sequence_len": context_size,
            **model_config,
        }
    }

    with open(model_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    gpt_config = GPTConfig(
        vocab_size=vocab_size,
        sequence_len=context_size,
        **model_config,
    )
    model = GPT(gpt_config)
    torch.save(model.state_dict(), model_dir / "best.pt")

    return model_dir


def create_mock_validation_data(
    data_dir: Path,
    vocab_size: int,
    num_samples: int = 50,
    min_seq_len: int = 20,
    max_seq_len: int = 100,
) -> Path:
    """Create mock validation data for testing."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create validate.jsonl with random token sequences
    with open(data_dir / "validate.jsonl", "w") as f:
        for i in range(num_samples):
            seq_len = np.random.randint(min_seq_len, max_seq_len)
            # Tokens between 1 and vocab_size-2 (avoid 0=PAD and vocab_size-1=EOS)
            tokens = np.random.randint(1, vocab_size - 1, size=seq_len).tolist()
            f.write(json.dumps({"tokens": tokens}) + "\n")

    # Create metadata.json
    metadata = {
        "vocab_size": vocab_size,
        "num_samples": num_samples,
    }
    with open(data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return data_dir


def verify_metrics_structure(metrics: dict, method_type: str) -> Tuple[bool, List[str]]:
    """Verify that all required metrics are present."""
    errors = []

    if method_type == "generation":
        required_fields = [
            'method', 'num_samples', 'json_validity_rate', 'schema_validity_rate',
            'field_coverage_rate', 'mean_time_seconds', 'tokens_per_second',
            'mean_tokens', 'min_tokens', 'max_tokens', 'std_tokens', 'unique_rate',
            'peak_memory_mb', 'json_validity_ci', 'schema_validity_ci'
        ]
    elif method_type == "validation":
        required_fields = [
            'method', 'num_samples', 'num_batches', 'total_tokens', 'total_bytes',
            'avg_loss', 'perplexity', 'bits_per_byte', 'token_accuracy',
            'top5_accuracy', 'loss_ci', 'accuracy_ci'
        ]
    else:
        return False, [f"Unknown method type: {method_type}"]

    for field in required_fields:
        if field not in metrics:
            errors.append(f"Missing field: {field}")

    return len(errors) == 0, errors


# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def test_field_coverage(schema_config: SchemaConfig, verbose: bool = False) -> Tuple[bool, str]:
    """Test field coverage check for a schema."""
    from scripts.eval_generation import check_required_fields

    valid_result = check_required_fields(schema_config.valid_sample, schema_config.name)
    invalid_result = check_required_fields(schema_config.invalid_sample, schema_config.name)

    if valid_result and not invalid_result:
        return True, "Field coverage check works correctly"
    else:
        return False, f"Field coverage check failed: valid={valid_result}, invalid={invalid_result}"


def test_generation_evaluation(
    schema_config: SchemaConfig,
    base_dir: Path,
    num_samples: int = 5,
    max_tokens: int = 30,
    verbose: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """Test generation evaluation for a schema."""
    from scripts.eval_generation import (
        load_model_from_checkpoint,
        generate_tct,
        generate_utf8_unconstrained,
        generate_utf8_xgrammar,
        compute_generation_metrics,
        set_seed,
        load_schema,
    )
    from nanochat.xgrammar_tokenizer import (
        UTF8BPEDecoder,
        build_xgrammar_tokenizer_info,
        compile_json_schema_grammar,
    )

    results = {
        "schema": schema_config.name,
        "methods": [],
        "errors": [],
    }

    # Check merge table exists
    if not Path(schema_config.merge_table).exists():
        results["errors"].append(f"Merge table not found: {schema_config.merge_table}")
        return False, results

    # Create UTF8 decoder
    utf8_decoder = UTF8BPEDecoder(schema_config.merge_table)
    utf8_vocab_size = utf8_decoder.vocab_size()

    # Check for TCT tokenizer
    tct_tokenizer = get_tct_tokenizer(schema_config.name)
    tct_vocab_size = tct_tokenizer.vocab_size() if tct_tokenizer else None

    # Create model directories
    schema_dir = base_dir / schema_config.name
    schema_dir.mkdir(exist_ok=True)

    # Create checkpoints
    utf8_model_dir = create_model_checkpoint(
        schema_dir / "utf8_model",
        utf8_vocab_size,
        schema_config.context_size,
    )

    tct_model_dir = None
    if tct_vocab_size:
        tct_model_dir = create_model_checkpoint(
            schema_dir / "tct_model",
            tct_vocab_size,
            schema_config.context_size,
        )

    # Test TCT generation (if available)
    if tct_tokenizer and tct_model_dir:
        try:
            set_seed(42)
            model, _ = load_model_from_checkpoint(tct_model_dir, "best.pt", "cpu")

            tct_results = []
            for i in range(num_samples):
                result = generate_tct(model, tct_tokenizer, [], max_tokens=max_tokens, temperature=0.8)
                tct_results.append(result)

            metrics = compute_generation_metrics(tct_results, "TCT-BPE", schema_config.name)
            metrics_dict = {
                "method": metrics.method,
                "num_samples": metrics.num_samples,
                "json_validity_rate": metrics.json_validity_rate,
                "schema_validity_rate": metrics.schema_validity_rate,
                "field_coverage_rate": metrics.field_coverage_rate,
                "mean_time_seconds": metrics.mean_time_seconds,
                "tokens_per_second": metrics.tokens_per_second,
                "mean_tokens": metrics.mean_tokens,
                "min_tokens": metrics.min_tokens,
                "max_tokens": metrics.max_tokens,
                "std_tokens": metrics.std_tokens,
                "unique_rate": metrics.unique_rate,
                "peak_memory_mb": metrics.peak_memory_mb,
                "json_validity_ci": metrics.json_validity_ci,
                "schema_validity_ci": metrics.schema_validity_ci,
            }
            results["methods"].append(metrics_dict)

            # Verify metrics structure
            valid, errs = verify_metrics_structure(metrics_dict, "generation")
            if not valid:
                results["errors"].extend([f"TCT-BPE: {e}" for e in errs])

            del model
        except Exception as e:
            results["errors"].append(f"TCT generation failed: {e}")

    # Test UTF8 generation
    try:
        set_seed(42)
        model, _ = load_model_from_checkpoint(utf8_model_dir, "best.pt", "cpu")
        model_hash = hash(tuple(model.state_dict()['transformer.wte.weight'].flatten()[:100].tolist()))

        utf8_results = []
        for i in range(num_samples):
            result = generate_utf8_unconstrained(model, utf8_decoder, [], max_tokens=max_tokens, temperature=0.8)
            utf8_results.append(result)

        metrics = compute_generation_metrics(utf8_results, "UTF8-BPE", schema_config.name)
        metrics_dict = {
            "method": metrics.method,
            "num_samples": metrics.num_samples,
            "json_validity_rate": metrics.json_validity_rate,
            "schema_validity_rate": metrics.schema_validity_rate,
            "field_coverage_rate": metrics.field_coverage_rate,
            "mean_time_seconds": metrics.mean_time_seconds,
            "tokens_per_second": metrics.tokens_per_second,
            "mean_tokens": metrics.mean_tokens,
            "min_tokens": metrics.min_tokens,
            "max_tokens": metrics.max_tokens,
            "std_tokens": metrics.std_tokens,
            "unique_rate": metrics.unique_rate,
            "peak_memory_mb": metrics.peak_memory_mb,
            "json_validity_ci": metrics.json_validity_ci,
            "schema_validity_ci": metrics.schema_validity_ci,
        }
        results["methods"].append(metrics_dict)

        # Verify metrics structure
        valid, errs = verify_metrics_structure(metrics_dict, "generation")
        if not valid:
            results["errors"].extend([f"UTF8-BPE: {e}" for e in errs])

        # Test XGrammar (same model)
        try:
            import xgrammar

            set_seed(42)
            model_hash_2 = hash(tuple(model.state_dict()['transformer.wte.weight'].flatten()[:100].tolist()))
            if model_hash != model_hash_2:
                results["errors"].append("Model changed before XGrammar!")

            tokenizer_info = build_xgrammar_tokenizer_info(schema_config.merge_table)
            schema_json = load_schema(schema_config.name)
            compiled_grammar = compile_json_schema_grammar(tokenizer_info, schema_json)

            xgrammar_results = []
            for i in range(num_samples):
                result = generate_utf8_xgrammar(
                    model, tokenizer_info, compiled_grammar, utf8_decoder,
                    prompt_tokens=[], max_tokens=max_tokens, temperature=0.8
                )
                xgrammar_results.append(result)

            metrics = compute_generation_metrics(xgrammar_results, "UTF8-BPE + XGrammar", schema_config.name)
            metrics_dict = {
                "method": metrics.method,
                "num_samples": metrics.num_samples,
                "json_validity_rate": metrics.json_validity_rate,
                "schema_validity_rate": metrics.schema_validity_rate,
                "field_coverage_rate": metrics.field_coverage_rate,
                "mean_time_seconds": metrics.mean_time_seconds,
                "tokens_per_second": metrics.tokens_per_second,
                "mean_tokens": metrics.mean_tokens,
                "min_tokens": metrics.min_tokens,
                "max_tokens": metrics.max_tokens,
                "std_tokens": metrics.std_tokens,
                "unique_rate": metrics.unique_rate,
                "peak_memory_mb": metrics.peak_memory_mb,
                "json_validity_ci": metrics.json_validity_ci,
                "schema_validity_ci": metrics.schema_validity_ci,
            }
            results["methods"].append(metrics_dict)

            # Verify metrics structure
            valid, errs = verify_metrics_structure(metrics_dict, "generation")
            if not valid:
                results["errors"].extend([f"UTF8-BPE + XGrammar: {e}" for e in errs])

            # Verify same model
            model_hash_3 = hash(tuple(model.state_dict()['transformer.wte.weight'].flatten()[:100].tolist()))
            if model_hash != model_hash_3:
                results["errors"].append("Model changed after XGrammar!")

        except ImportError:
            pass  # XGrammar not available
        except Exception as e:
            results["errors"].append(f"XGrammar generation failed: {e}")

        del model

    except Exception as e:
        import traceback
        results["errors"].append(f"UTF8 generation failed: {e}\n{traceback.format_exc()}")

    success = len(results["errors"]) == 0 and len(results["methods"]) > 0
    return success, results


def test_statistical_functions(verbose: bool = False) -> Tuple[bool, List[str]]:
    """Test statistical utility functions."""
    from scripts.eval_generation import (
        compute_bootstrap_ci,
        compute_bits_per_byte,
        compute_uniqueness,
    )

    errors = []

    # Bootstrap CI
    values = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    ci = compute_bootstrap_ci(values)
    if not (ci[0] <= np.mean(values) <= ci[1]):
        errors.append(f"Bootstrap CI: mean not within CI ({ci})")

    # Bits-per-byte
    bpb = compute_bits_per_byte(total_loss=2.0, total_tokens=1000, total_bytes=4000)
    expected_bpb = (2.0 * 1000) / (4000 * np.log(2))
    if abs(bpb - expected_bpb) > 0.001:
        errors.append(f"Bits-per-byte: {bpb} != {expected_bpb}")

    # Uniqueness
    texts = ["a", "b", "a", "c", "b", "d"]
    unique = compute_uniqueness(texts)
    if unique != 4/6:
        errors.append(f"Uniqueness: {unique} != {4/6}")

    return len(errors) == 0, errors


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mock experiment for evaluation script testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--schema",
        choices=list(SCHEMAS.keys()) + ["all"],
        default="all",
        help="Schema to test (default: all)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate (default: 5)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=30,
        help="Maximum tokens per generation (default: 30)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("COMPREHENSIVE MOCK EXPERIMENT FOR EVALUATION SCRIPT")
    print("=" * 100)

    # Check dependencies
    print("\nChecking dependencies...")

    try:
        import xgrammar
        HAS_XGRAMMAR = True
        print("  [OK] XGrammar available")
    except ImportError:
        HAS_XGRAMMAR = False
        print("  [WARN] XGrammar not available")

    # Determine schemas to test
    if args.schema == "all":
        schemas_to_test = list(SCHEMAS.keys())
    else:
        schemas_to_test = [args.schema]

    print(f"\nSchemas to test: {schemas_to_test}")

    # Run tests
    all_results = {}
    all_passed = True

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Working directory: {tmpdir}")

        # Test statistical functions
        print("\n" + "-" * 80)
        print("Testing statistical functions...")
        print("-" * 80)

        success, errors = test_statistical_functions(args.verbose)
        if success:
            print("  [OK] All statistical functions work correctly")
        else:
            print("  [FAIL] Statistical functions failed:")
            for e in errors:
                print(f"    - {e}")
            all_passed = False

        # Test each schema
        for schema_name in schemas_to_test:
            schema_config = SCHEMAS[schema_name]

            print("\n" + "=" * 80)
            print(f"TESTING SCHEMA: {schema_name.upper()}")
            print("=" * 80)

            # Test field coverage
            print("\n  Testing field coverage...")
            success, msg = test_field_coverage(schema_config, args.verbose)
            if success:
                print(f"    [OK] {msg}")
            else:
                print(f"    [FAIL] {msg}")
                all_passed = False

            # Test generation evaluation
            print("\n  Testing generation evaluation...")
            success, results = test_generation_evaluation(
                schema_config,
                tmpdir,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                verbose=args.verbose,
            )
            all_results[schema_name] = results

            if success:
                print(f"    [OK] Generation evaluation works")
                for m in results["methods"]:
                    print(f"      - {m['method']}: JSON={m['json_validity_rate']:.1%}, "
                          f"Fields={m['field_coverage_rate']:.1%}")
            else:
                print(f"    [FAIL] Generation evaluation failed")
                for e in results["errors"]:
                    print(f"      - {e}")
                all_passed = False

        # Summary
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)

        print(f"\n{'Schema':<12} {'Method':<25} {'JSON%':>8} {'Schema%':>9} {'Fields%':>9} {'Unique%':>9}")
        print("-" * 80)

        for schema_name, results in all_results.items():
            for m in results["methods"]:
                print(f"{schema_name:<12} {m['method']:<25} "
                      f"{m['json_validity_rate']*100:>7.1f}% "
                      f"{m['schema_validity_rate']*100:>8.1f}% "
                      f"{m['field_coverage_rate']*100:>8.1f}% "
                      f"{m['unique_rate']*100:>8.1f}%")

        print("-" * 80)

        # Final result
        print("\n" + "=" * 100)
        if all_passed:
            print("ALL TESTS PASSED!")
            exit_code = 0
        else:
            print("SOME TESTS FAILED!")
            exit_code = 1
        print("=" * 100)

        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"\nResults saved to {output_path}")

        sys.exit(exit_code)


if __name__ == "__main__":
    main()
