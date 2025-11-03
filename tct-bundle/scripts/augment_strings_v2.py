#!/usr/bin/env python3
"""
Character-Level String Augmentation for Workflow Training

Generates synthetic workflows by mutating existing strings with:
- Unicode characters (emoji, accented chars)
- Special JSON characters (quotes, backslashes, newlines)
- Length variations
- Case variations
- Punctuation and special characters

This teaches the model UTF-8 encoding patterns, not just memorization.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, Any

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

def mutate_string(s: str, mutation_rate: float = 0.3, ascii_only: bool = False) -> str:
    """
    Apply character-level mutations to a string.

    Args:
        s: Original string
        mutation_rate: Probability of applying mutations (0-1)
        ascii_only: If True, only use ASCII-safe mutations (for job IDs)

    Returns:
        Mutated string
    """
    if not s or random.random() > mutation_rate:
        return s

    if ascii_only:
        # Only ASCII-safe mutations for job IDs
        mutation_type = random.choice([
            'case_mix',     # Mix case
            'special_char', # Add special characters (-, _)
            'truncate',     # Shorten string
            'extend',       # Add suffix
        ])
    else:
        # Full Unicode mutations for workflow/step names
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
                char = random.choice(['_', '-'])  # Only ASCII-safe chars for job IDs
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

def augment_workflow(
    workflow: Dict[str, Any],
    mutation_rate: float = 0.3,
    augment_names: bool = True,
    augment_jobs: bool = True
) -> Dict[str, Any]:
    """
    Augment a workflow by mutating string values.

    Args:
        workflow: Original workflow dict
        mutation_rate: Probability of mutations (0-1)
        augment_names: Mutate workflow name
        augment_jobs: Mutate job names

    Returns:
        Augmented workflow dict
    """
    augmented = workflow.copy()

    # Augment workflow name
    if augment_names and 'name' in augmented:
        original_name = augmented['name']
        augmented['name'] = mutate_string(original_name, mutation_rate)

    # Augment job names
    if augment_jobs and 'jobs' in augmented and isinstance(augmented['jobs'], dict):
        new_jobs = {}
        for old_name, job_config in augmented['jobs'].items():
            # Mutate the job ID (key) - ASCII only for GitHub Actions compliance
            mutated_name = mutate_string(old_name, mutation_rate, ascii_only=True)

            # Ensure valid job ID: ^[_a-zA-Z][a-zA-Z0-9_-]*$
            # Remove invalid chars and ensure it starts correctly
            mutated_name = ''.join(
                c for c in mutated_name
                if c.isalnum() or c in '-_'
            )

            if not mutated_name or not (mutated_name[0].isalpha() or mutated_name[0] == '_'):
                mutated_name = f"job_{mutated_name}" if mutated_name else old_name

            new_jobs[mutated_name] = job_config

        augmented['jobs'] = new_jobs

    return augmented

def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    augmentation_factor: int = 3,
    mutation_rate: float = 0.3,
    max_files: int = None
):
    """
    Augment workflow dataset with character-level string mutations.

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

    print(f"Character-Level String Augmentation")
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
                    mutation_rate=mutation_rate,
                    augment_names=True,
                    augment_jobs=True
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
    print(f"   Unique strings: ~{len(workflow_files) * 3} (original) + mutations")
    print(f"   Location: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Character-level string augmentation for workflow training"
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
