#!/usr/bin/env python3
"""
Real-time Kubernetes Manifest Autocompletion Demo

Shows original manifest alongside generated manifest in real-time.
Demonstrates both from-scratch generation and autocomplete modes.

Usage:
    python streaming_demo.py                    # Default settings
    python streaming_demo.py --pause 5          # 5 second pause between examples
    python streaming_demo.py --examples 10      # Run 10 examples
"""

import argparse
import torch
import json
import sys
import time
import random
import yaml

import tct_kubernetes_streaming as tct
from nanochat.gpt import GPT, GPTConfig

# Constants
VOCAB_SIZE = 20000
CONTEXT_SIZE = 2048
PAD_TOKEN = tct.pad_token()

# ANSI escape codes
CLEAR_SCREEN = "\033[2J\033[H"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"
BLUE = "\033[94m"


def load_model(device: torch.device) -> GPT:
    """Load the trained model."""
    model_config = GPTConfig(
        sequence_len=CONTEXT_SIZE,
        vocab_size=VOCAB_SIZE,
        n_layer=8,
        n_head=6,
        n_kv_head=6,
        n_embd=384,
    )
    model = GPT(model_config)
    model.to(device)
    state_dict = torch.load("checkpoints/k8s_baseline_v1/model_200000.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def format_yaml(manifest: dict) -> str:
    """Format manifest as YAML."""
    try:
        return yaml.dump(manifest, default_flow_style=False, sort_keys=False).rstrip()
    except:
        return json.dumps(manifest, indent=2)


def side_by_side(left_title: str, left_content: str, right_title: str, right_content: str, width: int = 38) -> str:
    """Format two panels side by side."""
    left_lines = left_content.split('\n')
    right_lines = right_content.split('\n')

    # Pad to same length
    max_lines = max(len(left_lines), len(right_lines))
    left_lines += [''] * (max_lines - len(left_lines))
    right_lines += [''] * (max_lines - len(right_lines))

    # Build output
    output = []
    output.append(f"{BOLD}{left_title:<{width}}{RESET} │ {BOLD}{right_title:<{width}}{RESET}")
    output.append(f"{'─' * width}─┼─{'─' * width}")

    for l, r in zip(left_lines, right_lines):
        # Truncate and pad
        l_clean = l[:width].ljust(width)
        r_clean = r[:width].ljust(width)
        output.append(f"{l_clean} │ {r_clean}")

    return '\n'.join(output)


def single_column(title: str, content: str, width: int = 78) -> str:
    """Format single column display."""
    lines = content.split('\n')
    output = []
    output.append(f"{BOLD}{title}{RESET}")
    output.append('─' * width)
    for line in lines:
        output.append(line[:width])
    return '\n'.join(output)


def generate_with_display(
    model: GPT,
    prompt_tokens: list[int],
    device: torch.device,
    original_manifest: dict,
    mode: str,
    max_tokens: int = 400,
    temperature: float = 0.7,
    top_k: int = 50,
) -> tuple[list[int], dict | None, bool]:
    """Generate with live side-by-side display."""
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = list(prompt_tokens)
    start_time = time.time()
    last_display_time = 0
    display_interval = 0.1  # Update every 100ms

    orig_kind = original_manifest.get('kind', '?')
    orig_name = original_manifest.get('metadata', {}).get('name', '?')

    # Decode prompt tokens to show what context the model has (for autocomplete mode)
    prompt_yaml = ""
    if mode == "autocomplete" and len(prompt_tokens) > 1:
        prompt_json, _, _ = tct.decode_prefix(prompt_tokens)
        try:
            prompt_manifest = json.loads(prompt_json)
            if prompt_manifest and (prompt_manifest.get('kind') or prompt_manifest.get('metadata')):
                prompt_yaml = format_yaml(prompt_manifest)
            else:
                prompt_yaml = f"{DIM}(partial prompt - {len(prompt_tokens)} tokens){RESET}"
        except:
            prompt_yaml = f"{DIM}(partial prompt - {len(prompt_tokens)} tokens){RESET}"

    with torch.no_grad():
        for step in range(max_tokens):
            logits = model(x)[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            if token_id == PAD_TOKEN:
                break

            generated.append(token_id)
            x = torch.cat([x, next_token], dim=1)

            # Throttled display
            now = time.time()
            if now - last_display_time >= display_interval:
                last_display_time = now
                json_str, fields, is_complete = tct.decode_prefix(generated)

                try:
                    result = json.loads(json_str)
                    gen_kind = result.get('kind', '')
                    gen_name = result.get('metadata', {}).get('name', '')
                    # Show YAML if we have real content, otherwise show progress
                    if gen_kind or gen_name or len(result) > 1:
                        gen_yaml = format_yaml(result)
                    else:
                        dots = "." * ((len(generated) // 10) % 4)
                        gen_yaml = f"{DIM}Generating{dots}{RESET}"
                except:
                    gen_kind = ''
                    gen_name = ''
                    result = {}
                    gen_yaml = f"{DIM}Generating...{RESET}"

                elapsed = now - start_time
                tps = len(generated) / elapsed if elapsed > 0 else 0

                # Progress bar
                progress = min(len(generated) / 150, 1.0)
                bar_w = 40
                filled = int(bar_w * progress)
                bar = f"{'█' * filled}{'░' * (bar_w - filled)}"

                status = f"{GREEN}● COMPLETE{RESET}" if is_complete else f"{YELLOW}○ generating...{RESET}"
                header = "FROM SCRATCH" if mode == "scratch" else "AUTOCOMPLETE"

                # Build content section based on mode
                if mode == "scratch":
                    content_section = single_column(
                        f"{GREEN}GENERATED{RESET} ({gen_kind or '?'}/{gen_name or '?'})",
                        gen_yaml
                    )
                else:
                    content_section = side_by_side(
                        f"{BLUE}PROMPT{RESET} ({len(prompt_tokens)} tokens)",
                        prompt_yaml,
                        f"{GREEN}GENERATED{RESET} ({gen_kind or '?'}/{gen_name or '?'})",
                        gen_yaml
                    )

                display = f"""{CLEAR_SCREEN}
{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗
║                    KUBERNETES MANIFEST GENERATION - {header:^14}                ║
╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}

  {CYAN}Status:{RESET}   {status}
  {CYAN}Progress:{RESET} [{bar}] {len(generated):3d} tokens @ {tps:.0f}/s

{content_section}
"""
                print(display, end='', flush=True)
                time.sleep(0.02)

                if is_complete:
                    # Show final display
                    gen_yaml = format_yaml(result)
                    if mode == "scratch":
                        content_section = single_column(
                            f"{GREEN}GENERATED{RESET} ({gen_kind}/{gen_name})",
                            gen_yaml
                        )
                    else:
                        content_section = side_by_side(
                            f"{BLUE}PROMPT{RESET} ({len(prompt_tokens)} tokens)",
                            prompt_yaml,
                            f"{GREEN}GENERATED{RESET} ({gen_kind}/{gen_name})",
                            gen_yaml
                        )

                    final_display = f"""{CLEAR_SCREEN}
{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗
║                    KUBERNETES MANIFEST GENERATION - {header:^14}                ║
╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}

  {CYAN}Status:{RESET}   {GREEN}● COMPLETE{RESET}
  {CYAN}Progress:{RESET} [{'█' * bar_w}] {len(generated):3d} tokens @ {tps:.0f}/s

{content_section}
"""
                    print(final_display, end='', flush=True)
                    return generated, result, True

            if len(generated) >= CONTEXT_SIZE:
                break

    # Final decode
    json_str, _, is_complete = tct.decode_prefix(generated)
    try:
        result = json.loads(json_str)
        gen_kind = result.get('kind', '?')
        gen_name = result.get('metadata', {}).get('name', '?')
        gen_yaml = format_yaml(result)
        elapsed = time.time() - start_time
        tps = len(generated) / elapsed if elapsed > 0 else 0
        bar_w = 40
        header = "FROM SCRATCH" if mode == "scratch" else "AUTOCOMPLETE"

        if mode == "scratch":
            content_section = single_column(
                f"{GREEN}GENERATED{RESET} ({gen_kind}/{gen_name})",
                gen_yaml
            )
        else:
            content_section = side_by_side(
                f"{BLUE}PROMPT{RESET} ({len(prompt_tokens)} tokens)",
                prompt_yaml,
                f"{GREEN}GENERATED{RESET} ({gen_kind}/{gen_name})",
                gen_yaml
            )

        final_display = f"""{CLEAR_SCREEN}
{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗
║                    KUBERNETES MANIFEST GENERATION - {header:^14}                ║
╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}

  {CYAN}Status:{RESET}   {YELLOW}○ max tokens reached{RESET}
  {CYAN}Progress:{RESET} [{'█' * bar_w}] {len(generated):3d} tokens @ {tps:.0f}/s

{content_section}
"""
        print(final_display, end='', flush=True)
        return generated, result, is_complete
    except:
        return generated, None, False


def sample_until_complete(model, prompt, device, max_attempts=5, max_tokens=400):
    """Sample until complete."""
    temps = [0.7, 0.6, 0.8, 0.65, 0.75]
    for i in range(max_attempts):
        x = torch.tensor([prompt], dtype=torch.long, device=device)
        generated = list(prompt)

        with torch.no_grad():
            for _ in range(max_tokens):
                logits = model(x)[:, -1, :] / temps[i % len(temps)]
                v, _ = torch.topk(logits, 50)
                logits[logits < v[:, [-1]]] = -float('inf')
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                if next_token.item() == PAD_TOKEN:
                    break
                generated.append(next_token.item())
                x = torch.cat([x, next_token], dim=1)
                if len(generated) >= CONTEXT_SIZE:
                    break

        json_str, _, complete = tct.decode_prefix(generated)
        if complete:
            try:
                return json.loads(json_str), i + 1
            except:
                pass
    return None, max_attempts


def run_demo(pause_between: float = 5.0, num_examples: int = 5, result_pause: float = 2.0):
    """Run the demo.

    Args:
        pause_between: Seconds to pause between examples
        num_examples: Number of examples to run
        result_pause: Seconds to pause after showing result
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{BOLD}Loading model...{RESET}")
    model = load_model(device)
    print(f"{GREEN}✓ Model loaded on {device}{RESET}")

    print(f"{BOLD}Loading sequences...{RESET}")
    all_sequences = torch.load("/home/josch/Desktop/data/.cache/tokenized_k8s_split90_254908files.pt")
    # Use validation set (last 10%)
    split_idx = int(len(all_sequences) * 0.9)
    sequences = all_sequences[split_idx:]
    print(f"{GREEN}✓ Loaded {len(sequences)} validation sequences (from {len(all_sequences)} total){RESET}")

    # Find examples
    print(f"{BOLD}Finding examples...{RESET}")
    examples = {'Deployment': [], 'Service': [], 'ConfigMap': [], 'Secret': [],
                'Job': [], 'StatefulSet': [], 'DaemonSet': [], 'Ingress': []}

    for i, seq in enumerate(sequences):
        if all(len(v) >= 2 for v in examples.values()):
            break
        seq_list = [t for t in seq.tolist() if t != PAD_TOKEN]
        if 30 <= len(seq_list) <= 400:
            try:
                json_str, _, complete = tct.decode_prefix(seq_list)
                if complete:
                    manifest = json.loads(json_str)
                    kind = manifest.get('kind', '')
                    name = manifest.get('metadata', {}).get('name', '')
                    if kind in examples and name and len(examples[kind]) < 3:
                        examples[kind].append((i, manifest, seq_list))
            except:
                pass

    all_examples = [(i, m, s) for exs in examples.values() for (i, m, s) in exs]
    print(f"{GREEN}✓ Found {len(all_examples)} examples{RESET}")
    time.sleep(1)

    # Demo scenarios - alternate between scratch and autocomplete
    random.seed(int(time.time()))
    kinds = ["Deployment", "Service", "ConfigMap", "Job", "Secret", "StatefulSet", "DaemonSet"]
    demos = []
    for i in range(num_examples):
        if i % 2 == 0:
            demos.append(("scratch", kinds[i % len(kinds)]))
        else:
            demos.append(("autocomplete", None))

    success_count = 0

    for demo_num, (mode, target_kind) in enumerate(demos):
        print(f"\n{CYAN}{'─' * 70}{RESET}")

        if mode == "scratch":
            # Find matching example
            matching = [(i, m, s) for (i, m, s) in all_examples if m.get('kind') == target_kind]
            if not matching:
                matching = all_examples
            idx, orig_manifest, full_seq = random.choice(matching)

            # Start from scratch - use only the first token as seed (resource type marker)
            prompt = full_seq[:1]

            print(f"{CYAN}Demo {demo_num + 1}/{len(demos)}: Generating from scratch...{RESET}")

        else:
            # Autocomplete with longer prompt
            idx, orig_manifest, full_seq = random.choice(all_examples)
            kind = orig_manifest.get('kind', '?')
            name = orig_manifest.get('metadata', {}).get('name', '?')

            # 25-35% prompt
            prompt_len = max(15, min(50, len(full_seq) // 3))
            prompt = full_seq[:prompt_len]

            print(f"{CYAN}Demo {demo_num + 1}/{len(demos)}: Autocompleting {kind}/{name}...{RESET}")

        time.sleep(1)

        generated, manifest, is_complete = generate_with_display(
            model, prompt, device, orig_manifest, mode,
            max_tokens=350, temperature=0.7
        )

        if is_complete and manifest:
            gen_kind = manifest.get('kind', '?')
            gen_name = manifest.get('metadata', {}).get('name', '?')
            success_count += 1

            time.sleep(result_pause)
            print(f"\n{GREEN}{'━' * 70}")
            print(f"  ✅ SUCCESS!")
            print(f"     Generated: {gen_kind}/{gen_name} ({len(generated)} tokens)")
            print(f"{'━' * 70}{RESET}")
        else:
            print(f"\n{YELLOW}First attempt incomplete, sampling alternatives...{RESET}")
            result, attempts = sample_until_complete(model, prompt, device)

            if result:
                success_count += 1
                print(f"\n{GREEN}{'━' * 70}")
                print(f"  ✅ SUCCESS on attempt {attempts}!")
                print(f"     Generated: {result.get('kind')}/{result.get('metadata', {}).get('name')}")
                print(f"{'━' * 70}{RESET}")
            else:
                print(f"\n{RED}  ❌ Failed after {attempts} attempts{RESET}")

        if demo_num < len(demos) - 1:
            print(f"\n{DIM}Next demo in {pause_between:.0f} seconds...{RESET}")
            time.sleep(pause_between)

    # Summary
    print(f"""
{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗
║                                   DEMO COMPLETE                                   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                   ║
║   Success: {GREEN}{success_count}/{len(demos)} manifests{RESET}                                                        ║
║                                                                                   ║
║   Features:                                                                       ║
║   • Side-by-side comparison (prompt vs generated)                                 ║
║   • From-scratch generation (1 seed token)                                        ║
║   • Autocomplete mode (partial manifest prompt)                                   ║
║   • Real-time streaming with decode_prefix()                                      ║
║                                                                                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}
""")


def parse_args():
    parser = argparse.ArgumentParser(description="Kubernetes Manifest Generation Demo")
    parser.add_argument("--pause", type=float, default=5.0,
                        help="Seconds to pause between examples (default: 5)")
    parser.add_argument("--examples", type=int, default=5,
                        help="Number of examples to run (default: 5)")
    parser.add_argument("--result-pause", type=float, default=2.0,
                        help="Seconds to pause after showing result (default: 2)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_demo(pause_between=args.pause, num_examples=args.examples, result_pause=args.result_pause)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Demo interrupted{RESET}")
        sys.exit(0)
