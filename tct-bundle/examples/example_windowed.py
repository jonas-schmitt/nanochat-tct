#!/usr/bin/env python3
"""
TCT Decoder-Only Windowed Generation Example

Demonstrates windowed generation with single position token for
left-to-right autoregressive training (NOT fill-in-the-middle).
"""

import json
from tct_github_workflow import encode, extract_window, vocab_size


def main():
    print("=" * 70)
    print("TCT Decoder-Only Windowed Generation Example")
    print("=" * 70)
    print()

    # Larger workflow for demonstration
    workflow = {
        "name": "Production CI/CD Pipeline",
        "on": {
            "push": {"branches": ["main", "develop"]},
            "pull_request": {"branches": ["main"]},
        },
        "jobs": {
            "test": {
                "runs-on": "ubuntu-latest",
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {"name": "Install dependencies", "run": "npm install"},
                    {"name": "Run tests", "run": "npm test"},
                ],
            },
            "build": {
                "runs-on": "ubuntu-latest",
                "needs": ["test"],
                "steps": [
                    {"uses": "actions/checkout@v4"},
                    {"name": "Build", "run": "npm run build"},
                    {
                        "name": "Upload artifacts",
                        "uses": "actions/upload-artifact@v4",
                    },
                ],
            },
        },
    }

    # Encode
    json_str = json.dumps(workflow)
    tokens = encode(json_str)

    print(f"Full sequence: {len(tokens)} tokens")
    print(f"Token IDs: {tokens[:20]}...")
    print()

    # Extract windows with different context sizes
    CONTEXT_SIZE = 8  # Small for demonstration

    print(f"Extracting decoder-only windows (context_size={CONTEXT_SIZE}):")
    print()

    # Extract a few windows to demonstrate structure
    for window_idx in [0, 5, 10]:
        if window_idx + CONTEXT_SIZE >= len(tokens):
            break

        start = window_idx
        end = window_idx + CONTEXT_SIZE
        window = extract_window(tokens, start, end)

        print(f"Window {window_idx} (position {start}):")
        print(f"  Format: [position_token, content_tokens...]")
        print(f"  Length: {len(window)} tokens (1 position + {CONTEXT_SIZE} content)")
        print(f"  Position token: {window[0]} (should be {start})")
        print(f"  Content tokens: {window[1:]}")

        # Verify window structure
        assert len(window) == CONTEXT_SIZE + 1, f"Window length mismatch: {len(window)} != {CONTEXT_SIZE + 1}"
        assert window[0] == start, f"Position token mismatch: {window[0]} != {start}"
        assert window[1:] == tokens[start:end], "Content tokens mismatch"

        print(f"  ✅ Verified: Correct structure (1 position token + {CONTEXT_SIZE} content)")
        print()

    # Calculate total windows for training
    total_windows = len(tokens) - CONTEXT_SIZE
    print(f"📊 Statistics:")
    print(f"  Full sequence: {len(tokens)} tokens")
    print(f"  Context size: {CONTEXT_SIZE} tokens")
    print(f"  Total windows: {total_windows}")
    print(f"  Window format: [schema_pos, tok_0, tok_1, ..., tok_{CONTEXT_SIZE-1}]")
    print()

    # Demonstrate training data preparation
    print("Training data preparation:")
    print()

    # For transformer training, we need (input, target) pairs
    # Input: [position, tok_0, tok_1, ..., tok_N-1]
    # Target: [tok_0, tok_1, ..., tok_N]

    window = extract_window(tokens, start=0, end=CONTEXT_SIZE)
    input_tokens = window[:-1]  # Remove last token
    target_tokens = window[1:]  # Remove position token

    print(f"  Window: {window}")
    print(f"  Input (x): {input_tokens} (length: {len(input_tokens)})")
    print(f"  Target (y): {target_tokens} (length: {len(target_tokens)})")
    print()

    # Verify target is shifted input
    assert len(input_tokens) == len(target_tokens)
    assert list(input_tokens[1:]) == list(target_tokens[:-1])
    print("  ✅ Verified: Target is input shifted by 1 position")
    print()

    # Demonstrate larger context size (typical for training)
    TRAINING_CONTEXT = 256

    if len(tokens) > TRAINING_CONTEXT:
        training_windows = len(tokens) - TRAINING_CONTEXT
        print(f"Production training (context_size={TRAINING_CONTEXT}):")
        print(f"  Total windows: {training_windows}")
        print(f"  Window size: {TRAINING_CONTEXT + 1} tokens (1 position + {TRAINING_CONTEXT} content)")
        print()

        # Extract first training window
        window = extract_window(tokens, start=0, end=TRAINING_CONTEXT)
        print(f"  Example window 0:")
        print(f"    Position token: {window[0]}")
        print(f"    Content tokens: {len(window[1:])} tokens")
        print(f"    Total: {len(window)} tokens")
    else:
        print(f"⚠️  Sequence too short for context_size={TRAINING_CONTEXT}")
        print(f"   Need at least {TRAINING_CONTEXT + 1} tokens, have {len(tokens)}")

    print()
    print("=" * 70)
    print("✅ Decoder-only windowed generation example completed successfully!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
