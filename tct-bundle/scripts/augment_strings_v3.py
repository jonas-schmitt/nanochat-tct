#!/usr/bin/env python3
"""
Comprehensive String Augmentation for Workflow Training

Augments ALL string values throughout workflows:
- Workflow names
- Job IDs
- Step names
- Step IDs
- Input/output names
- Secret names
- Environment variable names
- Branch names
- Default values

This maximizes string diversity to teach UTF-8 encoding patterns.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Unicode character pools for augmentation
EMOJI = [
    "🚀", "✅", "❌", "🔧", "🔨", "🎯", "📦", "🏗️", "🧪", "🔍",
    "📊", "📈", "⚙️", "🛠️", "💻", "🌐", "📝", "🔐", "⚡", "🎉"
]

ACCENTED_CHARS = {
    'a': ['à', 'á', 'â', 'ã', 'ä', 'å', 'ā'],
    'e': ['è', 'é', 'ê', 'ë', 'ē'],
    'i': ['ì', 'í', 'î', 'ï', 'ī'],
    'o': ['ò', 'ó', 'ô', 'õ', 'ö', 'ō'],
    'u': ['ù', 'ú', 'û', 'ü', 'ū'],
    'c': ['ç', 'ć', 'č'],
    'n': ['ñ', 'ń'],
    's': ['š', 'ś'],
    'z': ['ž', 'ź'],
}

SPECIAL_CHARS = ['_', '-', '.', '~', '+', '@', '#']

# Fields that should NOT be mutated (functional/syntactic)
SKIP_FIELDS = {
    'uses',  # Action references like "actions/checkout@v3"
    'runs-on',  # Runner labels
    'if',  # Conditional expressions
    'shell',  # Shell types
    'type',  # Type specifications
}

def mutate_string(s: str, mutation_rate: float = 0.3, ascii_only: bool = False) -> str:
    """
    Apply character-level mutations to a string.

    Args:
        s: Original string
        mutation_rate: Probability of applying mutations (0-1)
        ascii_only: If True, only use ASCII-safe mutations (for IDs)

    Returns:
        Mutated string
    """
    if not s or random.random() > mutation_rate:
        return s

    if ascii_only:
        # Only ASCII-safe mutations for IDs
        mutation_type = random.choice([
            'case_mix',     # Mix case
            'special_char', # Add special characters (-, _)
            'truncate',     # Shorten string
            'extend',       # Add suffix
        ])
    else:
        # Full Unicode mutations for names/values
        mutation_type = random.choice([
            'unicode',      # Add Unicode characters
            'accents',      # Replace with accented versions
            'emoji_prefix', # Add emoji at start
            'emoji_suffix', # Add emoji at end
            'case_mix',     # Mix case
            'special_char', # Add special characters
            'truncate',     # Shorten string
            'extend',       # Add suffix
        ])

    if mutation_type == 'unicode':
        # Replace random character with accented version
        chars = list(s.lower())
        for i, c in enumerate(chars):
            if c in ACCENTED_CHARS and random.random() < 0.5:
                chars[i] = random.choice(ACCENTED_CHARS[c])
        return ''.join(chars)

    elif mutation_type == 'accents':
        # Multiple accent replacements
        chars = list(s.lower())
        for i, c in enumerate(chars):
            if c in ACCENTED_CHARS and random.random() < 0.3:
                chars[i] = random.choice(ACCENTED_CHARS[c])
        return ''.join(chars)

    elif mutation_type == 'emoji_prefix':
        return f"{random.choice(EMOJI)} {s}"

    elif mutation_type == 'emoji_suffix':
        return f"{s} {random.choice(EMOJI)}"

    elif mutation_type == 'case_mix':
        # Random case mixing
        return ''.join(
            c.upper() if random.random() < 0.3 else c.lower()
            for c in s
        )

    elif mutation_type == 'special_char':
        # Insert special character
        if len(s) > 3:
            pos = random.randint(1, len(s) - 1)
            if ascii_only:
                char = random.choice(['_', '-'])  # Only ASCII-safe chars for IDs
            else:
                char = random.choice(SPECIAL_CHARS)
            return s[:pos] + char + s[pos:]
        return s

    elif mutation_type == 'truncate':
        # Shorten by removing suffix
        if len(s) > 5:
            cut = random.randint(1, min(4, len(s) - 3))
            return s[:-cut]
        return s

    elif mutation_type == 'extend':
        # Add version or suffix
        suffixes = ['_v2', '_new', '_prod', '_beta', '-x', '-latest']
        return s + random.choice(suffixes)

    return s

def should_mutate_field(field_name: str) -> bool:
    """Check if a field should be mutated."""
    return field_name not in SKIP_FIELDS

def mutate_dict_strings(obj: Any, mutation_rate: float, path: str = "") -> Any:
    """
    Recursively mutate string values in a dict/list structure.

    Args:
        obj: The object to mutate (dict, list, str, or other)
        mutation_rate: Probability of mutations
        path: Current path in the object tree (for debugging)

    Returns:
        Mutated object
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            new_key = key  # Preserve keys by default

            # Special handling for job IDs (top-level keys in 'jobs')
            if path == 'jobs':
                # Mutate job ID with ASCII-only
                new_key = mutate_string(key, mutation_rate, ascii_only=True)
                # Ensure valid job ID
                new_key = ''.join(c for c in new_key if c.isalnum() or c in '-_')
                if not new_key or not (new_key[0].isalpha() or new_key[0] == '_'):
                    new_key = f"job_{new_key}" if new_key else key

            # Special handling for step IDs (keys in 'id' field)
            elif path.endswith('.steps') and key == 'id':
                # Step IDs should be ASCII-only (used in expressions)
                value = mutate_string(str(value), mutation_rate, ascii_only=True)
                # Ensure valid ID
                value = ''.join(c for c in value if c.isalnum() or c in '-_')
                if not value or not (value[0].isalpha() or value[0] == '_'):
                    value = f"step-{random.randint(1000, 9999)}"

            # Special handling for input/output names (keys in 'inputs', 'outputs', 'secrets')
            elif path.endswith('.inputs') or path.endswith('.outputs') or path.endswith('.secrets'):
                # These can have hyphens but should stay ASCII
                new_key = mutate_string(key, mutation_rate, ascii_only=True)
                new_key = ''.join(c for c in new_key if c.isalnum() or c in '-_')
                if not new_key:
                    new_key = key

            # Recursively process value
            new_path = f"{path}.{key}" if path else key
            new_value = mutate_dict_strings(value, mutation_rate, new_path)

            result[new_key] = new_value

        return result

    elif isinstance(obj, list):
        return [mutate_dict_strings(item, mutation_rate, f"{path}[]") for item in obj]

    elif isinstance(obj, str):
        # Mutate string values (not keys, those are handled above)
        # Skip template syntax and functional strings
        if obj.startswith('${{') or obj.startswith('actions/') or obj.startswith('uses:'):
            return obj  # Don't mutate template expressions or action references

        # Check field name context
        field_name = path.split('.')[-1] if '.' in path else path
        if not should_mutate_field(field_name):
            return obj

        # Mutate the string (Unicode allowed for most string values)
        return mutate_string(obj, mutation_rate, ascii_only=False)

    else:
        # Return other types unchanged (int, bool, None, etc.)
        return obj

def augment_workflow(
    workflow: Dict[str, Any],
    mutation_rate: float = 0.3
) -> Dict[str, Any]:
    """
    Comprehensively augment all strings in a workflow.

    Args:
        workflow: Original workflow dict
        mutation_rate: Probability of mutations (0-1)

    Returns:
        Augmented workflow dict
    """
    # Deep copy and recursively mutate all strings
    augmented = mutate_dict_strings(workflow, mutation_rate, path="")

    return augmented

def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    augmentation_factor: int = 3,
    mutation_rate: float = 0.3,
    max_files: int = None
):
    """
    Augment workflow dataset with comprehensive string mutations.

    Args:
        input_dir: Directory with original workflow JSON files
        output_dir: Directory to write augmented workflows
        augmentation_factor: Augmented copies per original
        mutation_rate: Probability of mutations (0-1)
        max_files: Maximum input files to process
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow_files = list(input_dir.glob('*.json'))
    if max_files:
        workflow_files = workflow_files[:max_files]

    print(f"Comprehensive String Augmentation V3")
    print(f"=" * 60)
    print(f"Augments: workflow names, job IDs, step names, step IDs,")
    print(f"          input/output names, secret names, env vars,")
    print(f"          branch names, default values, and more!")
    print(f"=" * 60)
    print(f"Input workflows: {len(workflow_files)}")
    print(f"Augmentation factor: {augmentation_factor}x")
    print(f"Mutation rate: {mutation_rate}")
    print(f"Total output: {len(workflow_files) * (1 + augmentation_factor)} workflows")
    print()

    total_written = 0

    for i, wf_file in enumerate(workflow_files):
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(workflow_files)} files...")

        try:
            with open(wf_file) as f:
                workflow = json.load(f)

            # Write original (ALWAYS keep original diversity)
            output_file = output_dir / f"{total_written:06d}.json"
            with open(output_file, 'w') as f:
                json.dump(workflow, f, separators=(',', ':'))
            total_written += 1

            # Generate augmented copies with mutations
            for aug_idx in range(augmentation_factor):
                augmented = augment_workflow(
                    workflow,
                    mutation_rate=mutation_rate
                )

                output_file = output_dir / f"{total_written:06d}.json"
                with open(output_file, 'w') as f:
                    json.dump(augmented, f, separators=(',', ':'))
                total_written += 1

        except Exception as e:
            print(f"Error processing {wf_file}: {e}")
            continue

    print()
    print(f"✅ Augmentation complete!")
    print(f"   Original workflows: {len(workflow_files)}")
    print(f"   Total output: {total_written}")
    print(f"   All string types augmented!")
    print(f"   Location: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive string augmentation for workflow training (v3)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with workflow JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for augmented workflows"
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=3,
        help="Augmentation factor (augmented copies per original, default: 3)"
    )
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.3,
        help="Probability of string mutations (0-1, default: 0.3)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum input files to process (default: all)"
    )

    args = parser.parse_args()

    augment_dataset(
        input_dir=args.input,
        output_dir=args.output,
        augmentation_factor=args.factor,
        mutation_rate=args.mutation_rate,
        max_files=args.max_files
    )
