#!/usr/bin/env python3
"""
Real-time Kubernetes Manifest Autocompletion Demo

Shows prompt alongside generated manifest in real-time.

Usage:
    python streaming_demo.py                    # Default settings
    python streaming_demo.py --pause 5          # 5 second pause between examples
    python streaming_demo.py --examples 10      # Run 10 examples
"""

import argparse
import io
import os
import sys
import time
import random
import json
import yaml

import torch
import tct_kubernetes_streaming as tct
from nanochat.gpt import GPT, GPTConfig

# Constants
VOCAB_SIZE = 20000
CONTEXT_SIZE = 2048
PAD_TOKEN = tct.pad_token()


class TerminalDisplay:
    """Handle terminal output with proper buffering."""

    # ANSI codes
    CLEAR = "\033[2J\033[H"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"

    def __init__(self):
        self.buffer = io.StringIO()

    def clear(self):
        """Clear screen."""
        self.buffer = io.StringIO()
        self.buffer.write(self.CLEAR)

    def write(self, text: str):
        """Write to buffer."""
        self.buffer.write(text)

    def writeln(self, text: str = ""):
        """Write line to buffer."""
        self.buffer.write(text + "\n")

    def flush(self):
        """Flush buffer to stdout."""
        sys.stdout.write(self.buffer.getvalue())
        sys.stdout.flush()
        self.buffer = io.StringIO()

    def header(self, title: str):
        """Write header box."""
        width = 80
        self.writeln(f"{self.BOLD}{'═' * width}{self.RESET}")
        self.writeln(f"{self.BOLD}{title:^{width}}{self.RESET}")
        self.writeln(f"{self.BOLD}{'═' * width}{self.RESET}")

    def status_line(self, label: str, value: str):
        """Write a status line."""
        self.writeln(f"  {self.CYAN}{label}:{self.RESET} {value}")

    def progress_bar(self, current: int, total: int, width: int = 40) -> str:
        """Generate progress bar string."""
        pct = min(current / total, 1.0) if total > 0 else 0
        filled = int(width * pct)
        return f"[{'█' * filled}{'░' * (width - filled)}]"

    def side_by_side(self, left_title: str, left_lines: list,
                      right_title: str, right_lines: list, col_width: int = 38):
        """Write two columns side by side."""
        # Pad to same length
        max_lines = max(len(left_lines), len(right_lines))
        left_lines = left_lines + [''] * (max_lines - len(left_lines))
        right_lines = right_lines + [''] * (max_lines - len(right_lines))

        # Header
        self.writeln()
        self.writeln(f"{self.BOLD}{left_title:<{col_width}}{self.RESET} │ {self.BOLD}{right_title:<{col_width}}{self.RESET}")
        self.writeln(f"{'─' * col_width}─┼─{'─' * col_width}")

        # Content
        for left, right in zip(left_lines, right_lines):
            l = left[:col_width].ljust(col_width)
            r = right[:col_width].ljust(col_width)
            self.writeln(f"{l} │ {r}")


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


def format_yaml_lines(manifest: dict, max_lines: int = 30) -> list:
    """Format manifest as YAML lines."""
    try:
        text = yaml.dump(manifest, default_flow_style=False, sort_keys=False)
        lines = text.rstrip().split('\n')
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more)"]
        return lines
    except:
        return [json.dumps(manifest, indent=2)]


def generate_with_display(
    model: GPT,
    prompt_tokens: list,
    prompt_yaml_lines: list,
    device: torch.device,
    display: TerminalDisplay,
    max_tokens: int = 400,
    temperature: float = 0.7,
    top_k: int = 50,
) -> tuple:
    """Generate tokens with live display."""

    x = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated = list(prompt_tokens)
    start_time = time.time()
    last_update = 0
    update_interval = 0.15

    with torch.no_grad():
        for step in range(max_tokens):
            # Generate next token
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

            # Throttled display update
            now = time.time()
            if now - last_update >= update_interval:
                last_update = now

                # Try to decode current state
                json_str, fields, is_complete = tct.decode_prefix(generated)
                try:
                    result = json.loads(json_str)
                    gen_kind = result.get('kind', '')
                    gen_name = result.get('metadata', {}).get('name', '')
                    if gen_kind or gen_name:
                        gen_lines = format_yaml_lines(result)
                    else:
                        gen_lines = [f"{display.DIM}Generating...{display.RESET}"]
                except:
                    gen_kind, gen_name = '', ''
                    gen_lines = [f"{display.DIM}Generating...{display.RESET}"]
                    result = {}

                elapsed = now - start_time
                tps = len(generated) / elapsed if elapsed > 0 else 0

                # Render display
                display.clear()
                display.header("KUBERNETES MANIFEST AUTOCOMPLETE")
                display.writeln()

                if is_complete:
                    display.status_line("Status", f"{display.GREEN}● COMPLETE{display.RESET}")
                else:
                    display.status_line("Status", f"{display.YELLOW}○ generating...{display.RESET}")

                bar = display.progress_bar(len(generated), 150)
                display.status_line("Progress", f"{bar} {len(generated):3d} tokens @ {tps:.0f}/s")

                # Side by side comparison
                prompt_title = f"{display.BLUE}PROMPT{display.RESET} ({len(prompt_tokens)} tokens)"
                gen_title = f"{display.GREEN}GENERATED{display.RESET}"
                if gen_kind:
                    gen_title += f" ({gen_kind})"

                display.side_by_side(prompt_title, prompt_yaml_lines, gen_title, gen_lines)
                display.flush()

                if is_complete:
                    return generated, result, True

            if len(generated) >= CONTEXT_SIZE:
                break

    # Final decode
    json_str, _, is_complete = tct.decode_prefix(generated)
    try:
        result = json.loads(json_str)
        return generated, result, is_complete
    except:
        return generated, None, False


def sample_until_complete(model, prompt, device, max_attempts=5, max_tokens=400):
    """Retry sampling with varied temperatures."""
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
    """Run the autocomplete demo."""

    display = TerminalDisplay()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{display.BOLD}Loading model...{display.RESET}")
    model = load_model(device)
    print(f"{display.GREEN}✓ Model loaded on {device}{display.RESET}")

    print(f"{display.BOLD}Loading validation sequences...{display.RESET}")
    all_sequences = torch.load("/home/josch/Desktop/data/.cache/tokenized_k8s_split90_254908files.pt")
    split_idx = int(len(all_sequences) * 0.9)
    sequences = all_sequences[split_idx:]
    print(f"{display.GREEN}✓ Loaded {len(sequences)} validation sequences{display.RESET}")

    # Find suitable examples
    print(f"{display.BOLD}Finding examples...{display.RESET}")
    examples = []
    for seq in sequences:
        if len(examples) >= num_examples * 3:
            break
        seq_list = [t for t in seq.tolist() if t != PAD_TOKEN]
        if 50 <= len(seq_list) <= 300:
            try:
                json_str, _, complete = tct.decode_prefix(seq_list)
                if complete:
                    manifest = json.loads(json_str)
                    kind = manifest.get('kind', '')
                    name = manifest.get('metadata', {}).get('name', '')
                    if kind and name:
                        examples.append((manifest, seq_list))
            except:
                pass

    print(f"{display.GREEN}✓ Found {len(examples)} suitable examples{display.RESET}")
    time.sleep(1)

    random.seed(int(time.time()))
    random.shuffle(examples)

    success_count = 0

    for demo_num in range(min(num_examples, len(examples))):
        orig_manifest, full_seq = examples[demo_num]
        kind = orig_manifest.get('kind', '?')
        name = orig_manifest.get('metadata', {}).get('name', '?')

        # Use 25-35% of sequence as prompt
        prompt_len = min(20, len(full_seq) // 4)
        prompt = full_seq[:prompt_len]

        # Try to decode prompt for display
        prompt_json, _, _ = tct.decode_prefix(prompt)
        try:
            prompt_manifest = json.loads(prompt_json)
            prompt_lines = format_yaml_lines(prompt_manifest)
        except:
            prompt_lines = [f"{display.DIM}({len(prompt)} tokens){display.RESET}"]

        print(f"\n{display.CYAN}{'─' * 70}{display.RESET}")
        print(f"{display.CYAN}Demo {demo_num + 1}/{num_examples}: {kind}/{name}{display.RESET}")
        time.sleep(1)

        generated, manifest, is_complete = generate_with_display(
            model, prompt, prompt_lines, device, display,
            max_tokens=350, temperature=0.7
        )

        if is_complete and manifest:
            gen_kind = manifest.get('kind', '?')
            gen_name = manifest.get('metadata', {}).get('name', '?')
            success_count += 1

            time.sleep(result_pause)
            print(f"\n{display.GREEN}{'━' * 70}")
            print(f"  ✅ SUCCESS!")
            print(f"     Generated: {gen_kind}/{gen_name} ({len(generated)} tokens)")
            print(f"{'━' * 70}{display.RESET}")
        else:
            print(f"\n{display.YELLOW}Retrying with varied temperature...{display.RESET}")
            result, attempts = sample_until_complete(model, prompt, device)

            if result:
                success_count += 1
                gen_kind = result.get('kind', '?')
                gen_name = result.get('metadata', {}).get('name', '?')
                print(f"\n{display.GREEN}{'━' * 70}")
                print(f"  ✅ SUCCESS on attempt {attempts}!")
                print(f"     Generated: {gen_kind}/{gen_name}")
                print(f"{'━' * 70}{display.RESET}")
            else:
                print(f"\n{display.YELLOW}  ⚠ Incomplete after {attempts} attempts{display.RESET}")

        if demo_num < num_examples - 1:
            print(f"\n{display.DIM}Next demo in {pause_between:.0f}s...{display.RESET}")
            time.sleep(pause_between)

    # Summary
    print(f"\n{display.BOLD}{'═' * 70}")
    print(f"{'DEMO COMPLETE':^70}")
    print(f"{'═' * 70}{display.RESET}")
    print(f"\n  Success: {display.GREEN}{success_count}/{num_examples}{display.RESET} manifests")
    print(f"\n  • Autocomplete with streaming decode")
    print(f"  • Retry sampling for robustness")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Kubernetes Manifest Autocomplete Demo")
    parser.add_argument("--pause", type=float, default=5.0,
                        help="Seconds between examples (default: 5)")
    parser.add_argument("--examples", type=int, default=5,
                        help="Number of examples (default: 5)")
    parser.add_argument("--result-pause", type=float, default=2.0,
                        help="Seconds after result (default: 2)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_demo(args.pause, args.examples, args.result_pause)
    except KeyboardInterrupt:
        print(f"\n\033[93mDemo interrupted\033[0m")
        sys.exit(0)
