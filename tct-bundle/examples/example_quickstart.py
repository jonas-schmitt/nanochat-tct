#!/usr/bin/env python3
"""
TCT Quickstart Example

Demonstrates basic encode/decode with round-trip verification.
"""

import json
from tct_github_workflow import encode, decode, vocab_size


def main():
    print("=" * 60)
    print("TCT GitHub Workflow Tokenizer - Quickstart Example")
    print("=" * 60)
    print()

    # Example workflow
    workflow = {
        "name": "CI",
        "on": "push",
        "jobs": {
            "build": {
                "runs-on": "ubuntu-latest",
                "steps": [{"uses": "actions/checkout@v4"}],
            }
        },
    }

    # Convert to JSON string
    json_str = json.dumps(workflow)
    print(f"Original JSON ({len(json_str)} chars):")
    print(json.dumps(workflow, indent=2))
    print()

    # Encode
    tokens = encode(json_str)
    print(f"✅ Encoded: {len(tokens)} tokens (vocab: {vocab_size()})")
    print(f"   Token IDs: {tokens[:10]}..." if len(tokens) > 10 else f"   Token IDs: {tokens}")
    print()

    # Decode with consumption tracking
    decoded_json, consumed, total = decode(tokens)
    decoded = json.loads(decoded_json)

    print(f"✅ Decoded: {consumed}/{total} tokens consumed")
    if consumed < total:
        print(f"   ⚠️  Warning: {total - consumed} surplus tokens")
    print()

    # Verify round-trip
    if decoded == workflow:
        print("✅ Round-trip: SUCCESS (input == output)")
    else:
        print("❌ Round-trip: FAILED (mismatch!)")
        print(f"   Original: {workflow}")
        print(f"   Decoded:  {decoded}")
        return 1

    # Compression ratio
    chars_per_token = len(json_str) / len(tokens)
    print()
    print(f"📊 Compression ratio: {chars_per_token:.2f} chars/token")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
