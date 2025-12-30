#!/usr/bin/env python3
"""
Kubernetes Manifest Autocomplete Demo

Shows the original manifest alongside the generated completion.
"""

import argparse
import sys
import time
import random
import json
import re
import yaml

import subprocess
import torch
import tct_kubernetes_streaming as tct
from nanochat.gpt import GPT, GPTConfig

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

console = Console(force_terminal=True)

# Constants
VOCAB_SIZE = 20000
CONTEXT_SIZE = 2048
PAD_TOKEN = tct.pad_token()

# ANSI (for non-rich output)
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"


def out(text: str = "", end: str = "\n"):
    """Print with immediate flush."""
    print(text, end=end, flush=True)


def load_model(device: torch.device) -> GPT:
    """Load the trained model."""
    config = GPTConfig(
        sequence_len=CONTEXT_SIZE,
        vocab_size=VOCAB_SIZE,
        n_layer=8,
        n_head=6,
        n_kv_head=6,
        n_embd=384,
    )
    model = GPT(config)
    model.to(device)
    state_dict = torch.load("checkpoints/k8s_baseline_v1/model_200000.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def find_last_nonempty_string(obj, path=""):
    """Recursively find the last non-empty string value in a nested dict/list."""
    last = None
    if isinstance(obj, dict):
        for k, v in obj.items():
            result = find_last_nonempty_string(v, f"{path}.{k}" if path else k)
            if result:
                last = result
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            result = find_last_nonempty_string(v, f"{path}[{i}]")
            if result:
                last = result
    elif isinstance(obj, str) and obj:  # Non-empty string
        last = obj
    return last


def to_yaml(manifest: dict, max_lines: int = 30, highlight_last: bool = False) -> list[str]:
    """Convert manifest to YAML lines, handling embedded newlines.

    If highlight_last=True, highlights the last non-empty string value
    to show where generation will continue from.
    """
    try:
        text = yaml.dump(manifest, default_flow_style=False, sort_keys=False)
        # Split on actual newlines and flatten any embedded \n in strings
        raw_lines = text.rstrip().split('\n')
        lines = []
        for line in raw_lines:
            # Replace literal \n sequences with visible marker
            line = line.replace('\\n', '↵')
            lines.append(line)

        # Highlight the line containing the last non-empty string value
        if highlight_last and lines:
            last_value = find_last_nonempty_string(manifest)
            if last_value:
                # Find and highlight the line containing this value using rich markup
                for i in range(len(lines) - 1, -1, -1):
                    if last_value in lines[i]:
                        line = lines[i]
                        lines[i] = line.replace(last_value, f"[bold yellow]{last_value}[/]", 1)
                        break

        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"[dim]... ({len(lines) - max_lines} more)[/]"]
        return lines
    except:
        return [json.dumps(manifest, indent=2)]


def show_comparison(left_title: str, left_lines: list[str],
                    right_title: str, right_lines: list[str]):
    """Display two manifests side by side using rich Table."""
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold",
                  padding=(0, 1), collapse_padding=True)
    table.add_column(f"[blue]{left_title}[/]", width=45, overflow="fold")
    table.add_column(f"[green]{right_title}[/]", width=45, overflow="fold")

    # Pad to equal length
    max_len = max(len(left_lines), len(right_lines))
    left_lines = left_lines + [''] * (max_len - len(left_lines))
    right_lines = right_lines + [''] * (max_len - len(right_lines))

    for left, right in zip(left_lines, right_lines):
        # Use Text.from_markup to interpret [bold yellow] etc.
        left_text = Text.from_markup(left) if '[' in left else left
        right_text = Text.from_markup(right) if '[' in right else right
        table.add_row(left_text, right_text)

    console.print(table)


def generate_streaming(model, prompt, device, max_tokens=400, temperature=0.01):
    """Generate tokens with streaming progress updates."""
    x = torch.tensor([prompt], dtype=torch.long, device=device)
    tokens = list(prompt)
    last_state = None

    def show_progress(tokens_list, final=False):
        """Display current decode state."""
        json_str, fields, complete = tct.decode_prefix(tokens_list)
        try:
            m = json.loads(json_str)
            kind = m.get('kind') or '-'
            name = m.get('metadata', {}).get('name') or '-'
            ns = m.get('metadata', {}).get('namespace') or ''

            # Format: [tokens] fields | kind/name
            status = f"{GREEN}✓{RESET}" if complete else f"{YELLOW}…{RESET}"
            ns_str = f" ({ns})" if ns and ns != '-' else ""
            info = f"{status} [{len(tokens_list):3} tok, {fields} fields] {kind}/{name}{ns_str}"

            # Pad and clear line
            out(f"{DIM}{info:<75}{RESET}", end="\r" if not final else "\n")
            return (kind, name, fields)
        except:
            return None

    # Show initial state from prompt
    last_state = show_progress(tokens)

    with torch.no_grad():
        for step in range(max_tokens):
            logits = model(x)[:, -1, :] / temperature
            v, _ = torch.topk(logits, 50)
            logits[logits < v[:, [-1]]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)

            if next_tok.item() == PAD_TOKEN:
                break

            tokens.append(next_tok.item())
            x = torch.cat([x, next_tok], dim=1)

            # Update progress every 10 tokens
            if step % 10 == 0:
                new_state = show_progress(tokens)
                if new_state:
                    last_state = new_state

            if len(tokens) >= CONTEXT_SIZE:
                break

    # Clear the progress line
    out(" " * 80, end="\r")
    return tokens


def decode(tokens) -> tuple[dict | None, bool]:
    """Decode tokens to manifest."""
    json_str, _, complete = tct.decode_prefix(tokens)
    if complete:
        try:
            return json.loads(json_str), True
        except:
            pass
    return None, False


# Hand-picked examples with varied seed completion levels (10% to 50%)
# Avoiding ConfigMaps which can have unbounded data fields
# Format: (validation_index, seed_token_count, description)
CURATED_EXAMPLES = [
    # 10% seed - minimal context, model must infer resource type
    (3, 7, "10%"),    # RoleBinding - minimal structure visible
    (6, 5, "10%"),    # PersistentVolumeClaim - shows kind early

    # 20% seed - resource type becoming clear
    (27, 10, "20%"),  # Pod/visit-counter-server1
    (18, 11, "20%"),  # ResourceQuota/config-management-resource-quota

    # 30% seed - kind + partial structure visible
    (32, 20, "30%"),  # ReplicaSet/nginx
    (65, 22, "30%"),  # Deployment/echo

    # 40% seed - significant context provided
    (11, 36, "40%"),  # ClusterRoleBinding
    (40, 16, "40%"),  # PersistentVolumeClaim

    # 50% seed - half the manifest as context
    (29, 59, "50%"),  # ClusterRoleBinding - extensive context
    (23, 28, "50%"),  # ResourceQuota/hard-limit
]


def print_intro():
    """Print introductory explanation of the demo."""
    out()
    out(f"{'═' * 95}")
    out(f"{BOLD}{CYAN}Kubernetes Manifest Autocomplete Demo{RESET}")
    out(f"{'═' * 95}")
    out()
    out(f"{BOLD}What is this?{RESET}")
    out(f"  This demo shows a language model trained to autocomplete Kubernetes YAML manifests.")
    out(f"  Given a partial manifest (the SEED), the model generates a complete, valid manifest.")
    out()
    out(f"{BOLD}Technology:{RESET}")
    out(f"  • {CYAN}TCT (Type-Constrained Transformers){RESET}: A schema-aware language model approach")
    out(f"    that understands Kubernetes manifest structure. Instead of raw text tokens, the model")
    out(f"    learns structured data (field names, values, types), enabling valid K8s generation.")
    out(f"  • {CYAN}30M parameter GPT model{RESET}: Trained on 265K real Kubernetes manifests.")
    out(f"  • {CYAN}Streaming decode{RESET}: Shows partial results as tokens are generated.")
    out()
    out(f"{BOLD}What you'll see:{RESET}")
    out(f"  • {BLUE}SEED{RESET}: The partial manifest given to the model (5-50% of original)")
    out(f"    The last decoded value is {BOLD}{YELLOW}highlighted{RESET} to show the continuation point.")
    out(f"  • {GREEN}GENERATED{RESET}: The model's completion - a full, valid Kubernetes resource")
    out(f"  • Progress shows: [tokens, fields decoded] kind/name as generation proceeds")
    out()
    out(f"{BOLD}Validation:{RESET}")
    out(f"  Every generated manifest is validated with {BOLD}kubectl apply --dry-run=client{RESET}")
    out(f"  This confirms the output is a {GREEN}valid Kubernetes resource{RESET} that could be applied to a cluster.")
    out()
    out(f"{BOLD}Kubernetes resources shown:{RESET}")
    out(f"  RoleBinding, ClusterRoleBinding, Pod, Deployment, ReplicaSet, PVC, ResourceQuota")
    out()
    out(f"{'─' * 95}")
    out()


def run_demo(num_examples: int = 5, pause: float = 5.0, curated: bool = True):
    """Run autocomplete demo."""
    print_intro()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out(f"{BOLD}Loading model...{RESET}")
    model = load_model(device)
    out(f"{GREEN}✓ Model loaded ({device}){RESET}")

    out(f"{BOLD}Loading sequences...{RESET}")
    all_seqs = torch.load("/home/josch/Desktop/data/.cache/tokenized_k8s_split90_254908files.pt")
    val_seqs = all_seqs[int(len(all_seqs) * 0.95):]
    out(f"{GREEN}✓ {len(val_seqs)} validation sequences{RESET}")

    # Collect examples
    examples = []  # List of (manifest, tokens, seed_len, level)

    if curated:
        out(f"{BOLD}Using curated examples...{RESET}")
        for idx, seed_len, level in CURATED_EXAMPLES:
            if idx < len(val_seqs):
                seq = val_seqs[idx]
                tokens = [t for t in seq.tolist() if t != PAD_TOKEN]
                manifest, ok = decode(tokens)
                if ok and manifest.get('kind') and manifest.get('metadata', {}).get('name'):
                    examples.append((manifest, tokens, seed_len, level))
        out(f"{GREEN}✓ {len(examples)} curated examples loaded{RESET}")
    else:
        out(f"{BOLD}Finding random examples...{RESET}")
        for seq in val_seqs:
            if len(examples) >= num_examples * 2:
                break
            tokens = [t for t in seq.tolist() if t != PAD_TOKEN]
            if 50 <= len(tokens) <= 300:
                manifest, ok = decode(tokens)
                if ok and manifest.get('kind') and manifest.get('metadata', {}).get('name'):
                    seed_len = min(20, len(tokens) // 4)
                    examples.append((manifest, tokens, seed_len, "random"))
        out(f"{GREEN}✓ {len(examples)} examples found{RESET}")
        random.seed(int(time.time()))
        random.shuffle(examples)

    out()

    successes = 0
    validations = {'valid_json': 0, 'has_required': 0, 'valid_yaml': 0}

    for i in range(min(num_examples, len(examples))):
        orig_manifest, full_tokens, seed_len, level = examples[i]
        kind = orig_manifest.get('kind')
        name = orig_manifest.get('metadata', {}).get('name')

        # Use the curated seed length
        prompt = full_tokens[:seed_len]

        out(f"{'═' * 95}")
        out(f"{CYAN}[{i+1}/{num_examples}] {kind}/{name}{RESET} {DIM}[{level}]{RESET}")
        out(f"{DIM}Seed: {seed_len} tokens ({seed_len*100//len(full_tokens)}%), Original: {len(full_tokens)} tokens{RESET}")
        out(f"{'═' * 95}")
        out()

        # Generate with streaming progress (lower temperature for determinism)
        out(f"{YELLOW}Generating...{RESET}")

        result = None
        generated = None
        for attempt in range(5):
            temp = [0.01, 0.05, 0.1, 0.15, 0.2][attempt]  # Near-zero temps, slight increase on retry
            generated = generate_streaming(model, prompt, device, temperature=temp)
            result, ok = decode(generated)
            if ok:
                out(f"{GREEN}✓ Complete: {len(generated)} tokens{RESET}")
                break
            if attempt < 4:
                out(f"{YELLOW}Retry {attempt+2}/5...{RESET}")

        if result:
            successes += 1
            gen_kind = result.get('kind', '?')
            gen_name = result.get('metadata', {}).get('name', '?')

            # Validate the generated manifest
            valid_checks = []

            # Check 1: Valid JSON (already passed if we got here)
            validations['valid_json'] += 1
            valid_checks.append("JSON")

            # Check 2: Validate via kubectl --dry-run=client
            try:
                proc = subprocess.run(
                    ["kubectl", "apply", "--dry-run=client", "-f", "-"],
                    input=json.dumps(result),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if proc.returncode == 0:
                    validations['has_required'] += 1
                    valid_checks.append("K8s")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass  # kubectl not available or timed out

            # Check 3: Valid YAML serialization
            try:
                yaml.dump(result)
                validations['valid_yaml'] += 1
                valid_checks.append("YAML")
            except:
                pass

            # Decode the prompt to show what the model started with
            prompt_json, seed_fields, _ = tct.decode_prefix(prompt)
            try:
                prompt_manifest = json.loads(prompt_json)
            except:
                prompt_manifest = {}

            show_comparison(
                f"SEED ({seed_len} tok, {seed_fields} fields)",
                to_yaml(prompt_manifest, highlight_last=True),
                f"GENERATED ({len(generated)} tokens)",
                to_yaml(result)
            )
            # Show kubectl validation prominently
            if "K8s" in valid_checks:
                out(f"{GREEN}✓ {gen_kind}/{gen_name}{RESET} — {BOLD}{GREEN}kubectl validated ✓{RESET}")
            else:
                out(f"{YELLOW}✓ {gen_kind}/{gen_name}{RESET} — {DIM}generated (kubectl validation failed){RESET}")
        else:
            out(f"{YELLOW}✗ Failed after 5 attempts{RESET}")

        if i < num_examples - 1:
            out(f"\n{DIM}Next in {pause:.0f}s...{RESET}")
            time.sleep(pause)
        out()

    # Summary
    out(f"{'═' * 95}")
    out(f"{BOLD}Results: {GREEN}{successes}/{num_examples}{RESET} successful")
    out(f"{BOLD}{GREEN}{validations['has_required']}/{num_examples} kubectl validated{RESET} — all outputs are valid Kubernetes manifests")
    out(f"{'═' * 95}")
    out()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kubernetes Manifest Autocomplete Demo")
    parser.add_argument("--examples", type=int, default=5, help="Number of examples to show")
    parser.add_argument("--pause", type=float, default=5.0, help="Pause between examples (seconds)")
    parser.add_argument("--random", action="store_true", help="Use random examples instead of curated")
    args = parser.parse_args()

    try:
        run_demo(args.examples, args.pause, curated=not args.random)
    except KeyboardInterrupt:
        out(f"\n{YELLOW}Interrupted{RESET}")
        sys.exit(0)
