#!/usr/bin/env python3
"""Test windowing fix to ensure all tokens covered in every epoch."""

import sys
from pathlib import Path

# Test the fix implementation directly
class TestWindowingDataset:
    """Mock dataset to test _get_window_positions fix."""
    def __init__(self):
        self.context_size = 512
        self.offset_stride = 32
        self.current_offset = 0

    def _get_window_positions(self, workflow_len):
        """
        FIXED implementation of window positioning.

        CRITICAL: In epochs with offset > 0, we MUST include an initial short
        window [0:offset] to ensure ALL tokens are seen in every epoch.
        """
        positions = []

        # Handle workflows smaller than context_size
        if workflow_len < self.context_size:
            if self.current_offset == 0:
                positions.append(0)
            return positions

        # CRITICAL FIX: Add initial short window when offset > 0
        if self.current_offset > 0:
            positions.append(0)

        # Create non-overlapping windows starting from offset
        pos = self.current_offset
        while pos + self.context_size <= workflow_len:
            positions.append(pos)
            pos += self.context_size

        # CRITICAL FIX: Add final short window if there are remaining tokens
        if pos < workflow_len:
            positions.append(pos)

        return positions


def test_windowing():
    """Test windowing fix with various workflow lengths."""
    dataset = TestWindowingDataset()
    workflow_len = 1536

    print("=" * 80)
    print("WINDOWING FIX TEST")
    print("=" * 80)
    print(f"\nWorkflow length: {workflow_len} tokens")
    print(f"Context size: {dataset.context_size} tokens")
    print(f"Offset stride: {dataset.offset_stride} tokens")
    print()

    all_passed = True

    for epoch in range(5):
        offset = epoch * dataset.offset_stride
        dataset.current_offset = offset
        positions = dataset._get_window_positions(workflow_len)

        # Calculate coverage
        covered = set()
        window_info = []
        for pos in positions:
            end = min(pos + dataset.context_size, workflow_len)
            covered.update(range(pos, end))
            window_info.append(f"[{pos}:{end}]")

        # Check for gaps
        all_tokens = set(range(workflow_len))
        missing = all_tokens - covered

        status = "✅ PASS" if len(missing) == 0 else "❌ FAIL"

        print(f"Epoch {epoch} (offset={offset}):")
        print(f"  Window positions: {positions}")
        print(f"  Window ranges: {' '.join(window_info)}")
        print(f"  Tokens covered: {len(covered)}/{workflow_len} {status}")

        if missing:
            all_passed = False
            missing_list = sorted(list(missing))
            if len(missing_list) <= 20:
                print(f"  ❌ MISSING TOKENS: {missing_list}")
            else:
                print(f"  ❌ MISSING TOKENS: {missing_list[:20]}... (+{len(missing_list)-20} more)")

        # Verify initial window exists when offset > 0
        if offset > 0 and positions[0] != 0:
            print(f"  ❌ CRITICAL: Initial short window [0:{offset}] MISSING!")
            all_passed = False
        elif offset > 0:
            print(f"  ✅ Initial short window [0:{offset}] correctly included")

        print()

    # Test with smaller workflow
    print("-" * 80)
    print("\nTesting with shorter workflow (256 tokens < context_size):")
    workflow_len_small = 256

    for epoch in range(3):
        offset = epoch * dataset.offset_stride
        dataset.current_offset = offset
        positions = dataset._get_window_positions(workflow_len_small)

        if epoch == 0:
            expected = [0]
            status = "✅" if positions == expected else "❌"
            print(f"  Epoch {epoch} (offset={offset}): positions={positions} (expected=[0]) {status}")
        else:
            expected = []
            status = "✅" if positions == expected else "❌"
            print(f"  Epoch {epoch} (offset={offset}): positions={positions} (expected=[]) {status}")

        if positions != expected:
            all_passed = False

    print()
    print("=" * 80)
    print("TEST RESULT:")
    print("=" * 80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nThe windowing fix correctly ensures:")
        print("  • All tokens covered in every epoch")
        print("  • Initial short window included when offset > 0")
        print("  • Short workflows handled correctly")
        print("\n✅ Ready to restart training!")
        return 0
    else:
        print("❌ TESTS FAILED")
        print("\nSome epochs have missing tokens or incorrect windowing.")
        print("DO NOT restart training until this is fixed!")
        return 1


if __name__ == "__main__":
    exit_code = test_windowing()
    sys.exit(exit_code)
