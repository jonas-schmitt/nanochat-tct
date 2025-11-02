#!/usr/bin/env python3
"""
YAML to JSON Converter for GitHub Actions Workflows

Converts YAML workflow files to JSON format required by TCT tokenizer.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed", file=sys.stderr)
    print("Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def yaml_to_json(yaml_path: Path) -> dict:
    """Convert YAML file to JSON dict."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data


def main():
    parser = argparse.ArgumentParser(description="Convert YAML workflows to JSON")
    parser.add_argument("input", nargs="+", help="Input YAML file(s)")
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory (if not specified, prints to stdout)",
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON (indented)"
    )

    args = parser.parse_args()

    # Create output directory if specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Convert files
    for input_file in args.input:
        input_path = Path(input_file)

        if not input_path.exists():
            print(f"Error: File not found: {input_path}", file=sys.stderr)
            continue

        try:
            # Convert YAML to dict
            data = yaml_to_json(input_path)

            # Convert to JSON string
            if args.pretty:
                json_str = json.dumps(data, indent=2)
            else:
                json_str = json.dumps(data)

            # Output
            if args.output:
                # Save to file
                output_path = output_dir / f"{input_path.stem}.json"
                with open(output_path, "w") as f:
                    f.write(json_str)
                print(f"✅ Converted: {input_path} → {output_path}")
            else:
                # Print to stdout
                print(json_str)

        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {input_path}", file=sys.stderr)
            print(f"  {e}", file=sys.stderr)
        except Exception as e:
            print(f"Error processing: {input_path}", file=sys.stderr)
            print(f"  {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    exit(main())
