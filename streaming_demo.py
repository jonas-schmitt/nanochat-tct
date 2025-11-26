#!/usr/bin/env python3
"""
Real-time Kubernetes Manifest Autocompletion Demo

Shows manifests being generated token-by-token with live decode_prefix updates.
Includes both from-scratch generation and autocomplete modes.
"""

import torch
import json
import sys
import time
import random
from pathlib import Path

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


def format_yaml_preview(manifest: dict, max_lines: int = 28) -> str:
    """Format manifest as YAML-like display."""
    try:
        import yaml
        formatted = yaml.dump(manifest, default_flow_style=False, sort_keys=False)
        lines = formatted.split('\n')
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f'{DIM}  ... ({len(lines) - max_lines} more lines){RESET}']
        return '\n'.join(lines)
    except:
        return json.dumps(manifest, indent=2)[:800]


def generate_with_display(
    model: GPT,
    prompt_tokens: list[int],
    device: torch.device,
    mode: str,
    target_kind: str,
    target_name: str = "",
    max_tokens: int = 400,
    temperature: float = 0.7,
    top_k: int = 50,
    display_delay: float = 0.03,  # Delay between display updates
) -> tuple[list[int], dict | None, bool]:
    """Generate with live display."""
    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = list(prompt_tokens)
    start_time = time.time()
    last_display_time = 0
    display_interval = 0.08  # Update display every 80ms

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

            # Throttled display updates
            now = time.time()
            if now - last_display_time >= display_interval:
                last_display_time = now
                json_str, fields, is_complete = tct.decode_prefix(generated)

                try:
                    result = json.loads(json_str)
                    kind = result.get('kind', '')
                    name = result.get('metadata', {}).get('name', '')
                except:
                    kind = ''
                    name = ''
                    result = {}

                elapsed = now - start_time
                tps = len(generated) / elapsed if elapsed > 0 else 0
                status = f"{GREEN}● COMPLETE{RESET}" if is_complete else f"{YELLOW}○ generating{RESET}"

                # Build progress bar
                progress = min(len(generated) / 200, 1.0)  # Assume ~200 tokens typical
                bar_width = 30
                filled = int(bar_width * progress)
                bar = f"[{'█' * filled}{'░' * (bar_width - filled)}]"

                header = "FROM SCRATCH" if mode == "scratch" else "AUTOCOMPLETE"

                display = f"""{CLEAR_SCREEN}
{BOLD}╔═══════════════════════════════════════════════════════════════════════════════╗
║             KUBERNETES MANIFEST GENERATION - {header:^15}             ║
╚═══════════════════════════════════════════════════════════════════════════════╝{RESET}

  {CYAN}Target:{RESET}    {target_kind}{f'/{target_name}' if target_name else ''}
  {CYAN}Mode:{RESET}      {mode.capitalize()} ({len(prompt_tokens)} prompt tokens)
  {CYAN}Status:{RESET}    {status}

  {BOLD}Progress:{RESET}  {bar} {len(generated):3d} tokens
  {BOLD}Speed:{RESET}     {tps:.0f} tokens/sec
  {BOLD}Elapsed:{RESET}   {elapsed:.1f}s

{BOLD}┌─ Generated Manifest ─────────────────────────────────────────────────────────┐{RESET}
  {BOLD}Kind:{RESET}      {kind or DIM + '(pending)' + RESET}
  {BOLD}Name:{RESET}      {name or DIM + '(pending)' + RESET}

{format_yaml_preview(result, max_lines=20) if result else DIM + '  Waiting for complete structure...' + RESET}
{BOLD}└───────────────────────────────────────────────────────────────────────────────┘{RESET}
"""
                print(display, end='', flush=True)
                time.sleep(display_delay)  # Small delay for smooth animation

                if is_complete:
                    return generated, result, True

            if len(generated) >= CONTEXT_SIZE:
                break

    # Final decode
    json_str, _, is_complete = tct.decode_prefix(generated)
    try:
        return generated, json.loads(json_str), is_complete
    except:
        return generated, None, False


def sample_until_complete(model, prompt, device, max_attempts=5, max_tokens=400):
    """Sample multiple times until we get a complete manifest."""
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
    """Run the interactive demo."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{BOLD}Loading model...{RESET}")
    model = load_model(device)
    print(f"{GREEN}✓ Model loaded on {device}{RESET}")

    print(f"{BOLD}Loading sequences...{RESET}")
    sequences = torch.load("/home/josch/Desktop/data/.cache/tokenized_k8s_split90_254908files.pt")
    print(f"{GREEN}✓ Loaded {len(sequences)} sequences{RESET}")

    # Find examples of different types
    print(f"{BOLD}Finding examples...{RESET}")
    examples = {'Deployment': [], 'Job': [], 'Ingress': [], 'ClusterRole': [],
                'DaemonSet': [], 'StatefulSet': [], 'CronJob': []}

    for i, seq in enumerate(sequences):
        if all(len(v) >= 5 for v in examples.values()):
            break
        seq_list = [t for t in seq.tolist() if t != PAD_TOKEN]
        if 60 <= len(seq_list) <= 350:
            try:
                json_str, _, complete = tct.decode_prefix(seq_list)
                if complete:
                    manifest = json.loads(json_str)
                    kind = manifest.get('kind', '')
                    name = manifest.get('metadata', {}).get('name', '')
                    if kind in examples and name and len(examples[kind]) < 5:
                        examples[kind].append((i, kind, name, len(seq_list)))
            except:
                pass

    all_examples = [ex for exs in examples.values() for ex in exs]
    print(f"{GREEN}✓ Found {len(all_examples)} examples{RESET}")
    time.sleep(1)

    # Demo scenarios
    random.seed(int(time.time()))
    demos = [
        ("scratch", "Deployment"),  # From scratch
        ("autocomplete", None),      # Random autocomplete
        ("scratch", "Job"),          # From scratch
        ("autocomplete", None),      # Random autocomplete
        ("scratch", "Ingress"),      # From scratch
    ]

    success_count = 0

    for demo_num, (mode, target_kind) in enumerate(demos):
        print(f"\n{CYAN}{'─' * 60}{RESET}")

        if mode == "scratch":
            # Generate from scratch - use minimal prompt (just first few tokens of a matching example)
            matching = [ex for ex in all_examples if ex[1] == target_kind]
            if not matching:
                matching = all_examples
            idx, kind, name, seq_len = random.choice(matching)
            full_seq = [t for t in sequences[idx].tolist() if t != PAD_TOKEN]

            # Very short prompt - just enough to hint at the kind (5-10 tokens)
            prompt_len = random.randint(5, 10)
            prompt = full_seq[:prompt_len]

            print(f"{CYAN}Demo {demo_num + 1}/5: Generating {target_kind} from scratch...{RESET}")
            time.sleep(1)

            generated, manifest, is_complete = generate_with_display(
                model, prompt, device, "scratch", target_kind, "",
                max_tokens=400, temperature=0.7, display_delay=0.02
            )

        else:
            # Autocomplete - use longer prompt
            idx, kind, name, seq_len = random.choice(all_examples)
            full_seq = [t for t in sequences[idx].tolist() if t != PAD_TOKEN]

            # Longer prompt for autocomplete (25-35% of original)
            prompt_len = max(20, min(60, len(full_seq) // 3))
            prompt = full_seq[:prompt_len]

            print(f"{CYAN}Demo {demo_num + 1}/5: Autocompleting {kind}/{name}...{RESET}")
            time.sleep(1)

            generated, manifest, is_complete = generate_with_display(
                model, prompt, device, "autocomplete", kind, name,
                max_tokens=400, temperature=0.7, display_delay=0.02
            )

        # Handle result
        if is_complete and manifest:
            gen_kind = manifest.get('kind', '?')
            gen_name = manifest.get('metadata', {}).get('name', '?')
            success_count += 1

            time.sleep(0.5)
            print(f"\n{GREEN}{'━' * 60}")
            print(f"  ✅ SUCCESS! Generated {gen_kind}/{gen_name}")
            print(f"     {len(generated)} tokens in {mode} mode")
            print(f"{'━' * 60}{RESET}")
        else:
            # Try sampling more
            print(f"\n{YELLOW}First attempt incomplete, sampling alternatives...{RESET}")
            result, attempts = sample_until_complete(model, prompt, device)

            if result:
                gen_kind = result.get('kind', '?')
                gen_name = result.get('metadata', {}).get('name', '?')
                success_count += 1
                print(f"\n{GREEN}{'━' * 60}")
                print(f"  ✅ SUCCESS on attempt {attempts}!")
                print(f"     Generated {gen_kind}/{gen_name}")
                print(f"{'━' * 60}{RESET}")
            else:
                print(f"\n{RED}  ❌ Could not complete after {attempts} attempts{RESET}")

        if demo_num < len(demos) - 1:
            print(f"\n{DIM}Next demo in 3 seconds...{RESET}")
            time.sleep(3)

    # Final summary
    print(f"""
{BOLD}╔═══════════════════════════════════════════════════════════════════════════════╗
║                              DEMO COMPLETE                                    ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║   Success Rate: {GREEN}{success_count}/{len(demos)} manifests generated{RESET}                                   ║
║                                                                               ║
║   Features demonstrated:                                                      ║
║   • From-scratch generation (minimal prompt)                                  ║
║   • Autocomplete (partial manifest → complete)                                ║
║   • Real-time streaming decode with decode_prefix()                           ║
║   • Retry sampling for robustness                                             ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝{RESET}
""")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Demo interrupted{RESET}")
        sys.exit(0)
