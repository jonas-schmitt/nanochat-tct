#!/usr/bin/env python3
"""
Convert JSON Kubernetes manifests to YAML for BPE training.

The input manifests are assumed to be pre-filtered (TCT-compatible).
Filtering happens in the TCT repo, not here.
"""

import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

import yaml


def convert_file(json_path: str) -> tuple[str, str | None, str | None]:
    """
    Convert a JSON file to YAML.

    Returns: (json_path, yaml_content, error_message)
    """
    try:
        with open(json_path) as f:
            data = json.load(f)

        yaml_content = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        return (json_path, yaml_content, None)

    except Exception as e:
        return (json_path, None, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON manifests to YAML for BPE training"
    )
    parser.add_argument(
        "--input-dir",
        default=os.path.expanduser("~/Desktop/data/k8s-tct-compatible"),
        help="Input directory with pre-filtered JSON manifests",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.expanduser("~/Desktop/data/k8s-yaml"),
        help="Output directory for YAML files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all JSON files
    json_files = sorted(input_dir.glob("*.json"))
    if args.limit:
        json_files = json_files[:args.limit]

    total = len(json_files)
    print(f"Converting {total} manifests from {input_dir}")
    print(f"Output directory: {output_dir}")

    success_count = 0
    fail_count = 0

    # Process in parallel
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_file, str(f)): f for f in json_files}

        for i, future in enumerate(as_completed(futures)):
            json_path, yaml_content, error = future.result()

            if yaml_content is not None:
                json_name = Path(json_path).stem
                yaml_path = output_dir / f"{json_name}.yaml"
                yaml_path.write_text(yaml_content)
                success_count += 1
            else:
                fail_count += 1
                print(f"  Failed: {json_path} - {error}", file=sys.stderr)

            if (i + 1) % 10000 == 0 or (i + 1) == total:
                print(f"Progress: {i+1}/{total} ({success_count} OK, {fail_count} failed)")

    print(f"\nDone! Converted {success_count} manifests to YAML")
    if fail_count:
        print(f"  {fail_count} files failed (JSON parse errors)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
