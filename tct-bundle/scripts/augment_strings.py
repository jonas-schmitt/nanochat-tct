#!/usr/bin/env python3
"""
String Augmentation for Workflow Training

Generates synthetic workflows with diverse strings to help the model
learn string encoding patterns. Takes existing workflows and:
1. Replaces workflow names with random variations
2. Replaces job names with random variations
3. Keeps structure and common patterns (actions, triggers, etc.)

This increases string diversity without changing workflow structure.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List

# String components for generating diverse names
PREFIXES = [
    "CI", "CD", "Build", "Test", "Deploy", "Release", "Lint", "Check",
    "Verify", "Validate", "Publish", "Package", "Install", "Update",
    "Scan", "Analyze", "Monitor", "Run", "Execute", "Process"
]

SUFFIXES = [
    "Pipeline", "Workflow", "Job", "Task", "Action", "Process",
    "Build", "Test", "Deploy", "Release", "Check", "Scan"
]

CONNECTORS = ["", "-", " ", " and ", "/"]

ENVIRONMENTS = [
    "dev", "prod", "staging", "test", "qa", "production",
    "development", "main", "master", "feature", "hotfix"
]

PLATFORMS = [
    "linux", "windows", "macos", "ubuntu", "debian", "alpine",
    "node", "python", "rust", "go", "java", "docker"
]


def generate_random_name() -> str:
    """Generate a random workflow/job name."""
    templates = [
        lambda: random.choice(PREFIXES),
        lambda: f"{random.choice(PREFIXES)}{random.choice(CONNECTORS)}{random.choice(SUFFIXES)}",
        lambda: f"{random.choice(PREFIXES)} {random.choice(ENVIRONMENTS)}",
        lambda: f"{random.choice(PLATFORMS)} {random.choice(PREFIXES)}",
        lambda: f"{random.choice(PREFIXES)}-{random.choice(ENVIRONMENTS)}-{random.choice(SUFFIXES)}",
    ]

    return random.choice(templates)()


def augment_workflow(workflow: dict, augment_names: bool = True, augment_jobs: bool = True) -> dict:
    """
    Augment a workflow by replacing string values.

    Args:
        workflow: Original workflow dict
        augment_names: Replace workflow name
        augment_jobs: Replace job names

    Returns:
        Augmented workflow dict
    """
    augmented = workflow.copy()

    # Augment workflow name
    if augment_names and 'name' in augmented:
        augmented['name'] = generate_random_name()

    # Augment job names
    if augment_jobs and 'jobs' in augmented and isinstance(augmented['jobs'], dict):
        new_jobs = {}
        for old_name, job_config in augmented['jobs'].items():
            new_name = generate_random_name().lower().replace(' ', '-')
            # Ensure valid job ID (must match ^[_a-zA-Z][a-zA-Z0-9_-]*$)
            new_name = ''.join(c for c in new_name if c.isalnum() or c in '-_')
            if not new_name or not (new_name[0].isalpha() or new_name[0] == '_'):
                new_name = f"job-{new_name}"
            new_jobs[new_name] = job_config
        augmented['jobs'] = new_jobs

    return augmented


def augment_dataset(
    input_dir: Path,
    output_dir: Path,
    augmentation_factor: int = 3,
    max_files: int = None
):
    """
    Augment workflow dataset with synthetic string variations.

    Args:
        input_dir: Directory with original workflow JSON files
        output_dir: Directory to write augmented workflows
        augmentation_factor: How many augmented copies per original
        max_files: Maximum number of input files to process
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow_files = list(input_dir.glob('*.json'))
    if max_files:
        workflow_files = workflow_files[:max_files]

    print(f"Augmenting {len(workflow_files)} workflows...")
    print(f"Augmentation factor: {augmentation_factor}x")
    print(f"Total output: {len(workflow_files) * (1 + augmentation_factor)} workflows")
    print()

    total_written = 0

    for i, wf_file in enumerate(workflow_files):
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(workflow_files)} files...")

        try:
            with open(wf_file) as f:
                workflow = json.load(f)

            # Write original
            output_file = output_dir / f"{total_written:06d}.json"
            with open(output_file, 'w') as f:
                json.dump(workflow, f, separators=(',', ':'))
            total_written += 1

            # Generate augmented copies
            for aug_idx in range(augmentation_factor):
                augmented = augment_workflow(
                    workflow,
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
    print(f"   Input:  {len(workflow_files)} workflows")
    print(f"   Output: {total_written} workflows")
    print(f"   Location: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment workflow dataset with string variations"
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
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of input files to process (default: all)"
    )

    args = parser.parse_args()

    augment_dataset(
        input_dir=args.input,
        output_dir=args.output,
        augmentation_factor=args.factor,
        max_files=args.max_files
    )
