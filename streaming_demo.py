#!/usr/bin/env python3
"""
Real-time Kubernetes Manifest Autocompletion Demo

Shows original manifest alongside generated manifest in real-time.
Demonstrates both from-scratch generation and autocomplete modes.
"""

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
    orig_yaml = format_yaml(original_manifest)

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
                    # Show YAML if we have real content, otherwise show animated progress
                    if gen_kind or gen_name or len(result) > 1:
                        gen_yaml = format_yaml(result)
                    else:
                        # Animated waiting indicator
                        dots = "." * ((len(generated) // 10) % 4)
                        gen_yaml = f"""{DIM}
    ╭─────────────────────────────╮
    │                             │
    │   Generating tokens{dots:<4}    │
    │                             │
    │   {len(generated):3d} / ~150 tokens        │
    │                             │
    ╰─────────────────────────────╯{RESET}"""
                except:
                    gen_kind = ''
                    gen_name = ''
                    result = {}
                    gen_yaml = f"{DIM}Parsing...{RESET}"

                elapsed = now - start_time
                tps = len(generated) / elapsed if elapsed > 0 else 0

                # Progress bar
                progress = min(len(generated) / 150, 1.0)
                bar_w = 40
                filled = int(bar_w * progress)
                bar = f"{'█' * filled}{'░' * (bar_w - filled)}"

                status = f"{GREEN}● COMPLETE{RESET}" if is_complete else f"{YELLOW}○ generating...{RESET}"
                header = "FROM SCRATCH" if mode == "scratch" else "AUTOCOMPLETE"

                display = f"""{CLEAR_SCREEN}
{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗
║                    KUBERNETES MANIFEST GENERATION - {header:^14}                ║
╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}

  {CYAN}Mode:{RESET}     {mode.capitalize()} ({len(prompt_tokens)} prompt tokens)
  {CYAN}Status:{RESET}   {status}
  {CYAN}Progress:{RESET} [{bar}] {len(generated):3d} tokens @ {tps:.0f}/s

{BOLD}┌──────────────────────────────────────────────────────────────────────────────────┐{RESET}
{side_by_side(
    f"{BLUE}ORIGINAL{RESET} ({orig_kind}/{orig_name})",
    orig_yaml,
    f"{GREEN}GENERATED{RESET} ({gen_kind or '?'}/{gen_name or '?'})",
    gen_yaml
)}
{BOLD}└──────────────────────────────────────────────────────────────────────────────────┘{RESET}
"""
                print(display, end='', flush=True)
                time.sleep(0.02)

                if is_complete:
                    # Show final side-by-side comparison
                    gen_yaml = format_yaml(result)
                    final_display = f"""{CLEAR_SCREEN}
{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗
║                    KUBERNETES MANIFEST GENERATION - {header:^14}                ║
╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}

  {CYAN}Mode:{RESET}     {mode.capitalize()} ({len(prompt_tokens)} prompt tokens)
  {CYAN}Status:{RESET}   {GREEN}● COMPLETE{RESET}
  {CYAN}Progress:{RESET} [{'█' * bar_w}] {len(generated):3d} tokens @ {tps:.0f}/s

{BOLD}┌──────────────────────────────────────────────────────────────────────────────────┐{RESET}
{side_by_side(
    f"{BLUE}ORIGINAL{RESET} ({orig_kind}/{orig_name})",
    orig_yaml,
    f"{GREEN}GENERATED{RESET} ({gen_kind}/{gen_name})",
    gen_yaml
)}
{BOLD}└──────────────────────────────────────────────────────────────────────────────────┘{RESET}
"""
                    print(final_display, end='', flush=True)
                    return generated, result, True

            if len(generated) >= CONTEXT_SIZE:
                break

    # Final decode
    json_str, _, is_complete = tct.decode_prefix(generated)
    try:
        result = json.loads(json_str)
        # Show final comparison even if not flagged complete
        gen_kind = result.get('kind', '?')
        gen_name = result.get('metadata', {}).get('name', '?')
        gen_yaml = format_yaml(result)
        elapsed = time.time() - start_time
        tps = len(generated) / elapsed if elapsed > 0 else 0
        bar_w = 40
        final_display = f"""{CLEAR_SCREEN}
{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗
║                    KUBERNETES MANIFEST GENERATION - {header:^14}                ║
╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}

  {CYAN}Mode:{RESET}     {mode.capitalize()} ({len(prompt_tokens)} prompt tokens)
  {CYAN}Status:{RESET}   {YELLOW}○ max tokens reached{RESET}
  {CYAN}Progress:{RESET} [{'█' * bar_w}] {len(generated):3d} tokens @ {tps:.0f}/s

{BOLD}┌──────────────────────────────────────────────────────────────────────────────────┐{RESET}
{side_by_side(
    f"{BLUE}ORIGINAL{RESET} ({orig_kind}/{orig_name})",
    orig_yaml,
    f"{GREEN}GENERATED{RESET} ({gen_kind}/{gen_name})",
    gen_yaml
)}
{BOLD}└──────────────────────────────────────────────────────────────────────────────────┘{RESET}
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


def run_demo():
    """Run the demo."""
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

    # Demo scenarios
    random.seed(int(time.time()))
    demos = [
        ("scratch", "Deployment"),
        ("autocomplete", None),
        ("scratch", "Service"),
        ("autocomplete", None),
        ("scratch", "Job"),
    ]

    success_count = 0

    for demo_num, (mode, target_kind) in enumerate(demos):
        print(f"\n{CYAN}{'─' * 70}{RESET}")

        if mode == "scratch":
            # Find matching example
            matching = [(i, m, s) for (i, m, s) in all_examples if m.get('kind') == target_kind]
            if not matching:
                matching = all_examples
            idx, orig_manifest, full_seq = random.choice(matching)

            # Very short prompt (5-10 tokens)
            prompt_len = random.randint(5, 10)
            prompt = full_seq[:prompt_len]

            print(f"{CYAN}Demo {demo_num + 1}/5: Generating {target_kind} from scratch...{RESET}")

        else:
            # Autocomplete with longer prompt
            idx, orig_manifest, full_seq = random.choice(all_examples)
            kind = orig_manifest.get('kind', '?')
            name = orig_manifest.get('metadata', {}).get('name', '?')

            # 25-35% prompt
            prompt_len = max(15, min(50, len(full_seq) // 3))
            prompt = full_seq[:prompt_len]

            print(f"{CYAN}Demo {demo_num + 1}/5: Autocompleting {kind}/{name}...{RESET}")

        time.sleep(1)

        generated, manifest, is_complete = generate_with_display(
            model, prompt, device, orig_manifest, mode,
            max_tokens=350, temperature=0.7
        )

        if is_complete and manifest:
            gen_kind = manifest.get('kind', '?')
            gen_name = manifest.get('metadata', {}).get('name', '?')
            success_count += 1

            time.sleep(0.3)
            print(f"\n{GREEN}{'━' * 70}")
            print(f"  ✅ SUCCESS!")
            print(f"     Original: {orig_manifest.get('kind')}/{orig_manifest.get('metadata', {}).get('name')}")
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
            print(f"\n{DIM}Next demo in 3 seconds...{RESET}")
            time.sleep(3)

    # Summary
    print(f"""
{BOLD}╔══════════════════════════════════════════════════════════════════════════════════╗
║                                   DEMO COMPLETE                                   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                   ║
║   Success: {GREEN}{success_count}/{len(demos)} manifests{RESET}                                                        ║
║                                                                                   ║
║   Features:                                                                       ║
║   • Side-by-side comparison (original vs generated)                               ║
║   • From-scratch generation (5-10 token prompt)                                   ║
║   • Autocomplete mode (25-35% prompt)                                             ║
║   • Real-time streaming with decode_prefix()                                      ║
║                                                                                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝{RESET}
""")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Demo interrupted{RESET}")
        sys.exit(0)
