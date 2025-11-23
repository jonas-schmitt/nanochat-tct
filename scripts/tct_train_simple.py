#!/usr/bin/env python3
"""
Kubernetes Manifest TCT Training Script

Train TCT kubernetes manifest generation model using standard teacher forcing.
Trains on 265k kubernetes manifests with context size 2048 (recommended).

Usage:
    python -m scripts.tct_train_simple --model_size=small-2048

For distributed training:
    torchrun --nproc_per_node=8 -m scripts.tct_train_simple --model_size=small-2048
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F

# Add TCT adapters to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tct-bundle" / "adapters"))

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, compute_cleanup, print0, print_banner, autodetect_device_type
from model_config import get_config
from tct_k8s_dataloader import tct_k8s_data_loader

print_banner()

# -----------------------------------------------------------------------------
# User settings
device_type = ""  # cuda|cpu|mps (empty => autodetect)
model_size = "small-2048"  # small-1024, small-2048, small-4096, medium-2048, etc.
data_dir = "/home/josch/Desktop/data-test"  # Test with 10k subset
train_split = 0.9  # train/val split
num_iterations = 10  # SMOKE TEST: Just 10 iterations
eval_every = 5  # eval interval
save_every = 5000  # checkpoint interval
model_tag = ""  # optional tag for checkpoint directory
cache_file = None  # optional cache file path (None => auto-detect)
warmup_iters = 5000  # LR warmup steps
grad_clip = 1.0  # gradient clipping
device_batch_size = 4  # SMOKE TEST: Small batch

# CLI override
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# Load model config
config = get_config(model_size)
B = device_batch_size if device_batch_size is not None else config["batch_size"]
T = config["context_size"]
grad_accum = config["gradient_accumulation"]
learning_rate = config["learning_rate"]
weight_decay = config["weight_decay"]
beta1 = config["beta1"]
beta2 = config["beta2"]

print0(f"Model: {model_size}")
print0(f"Vocab size: {config['vocab_size']}")
print0(f"Context size: {T}")
print0(f"Batch size: {B}")
print0(f"Gradient accumulation: {grad_accum}")
print0(f"Effective batch size: {B * grad_accum * ddp_world_size}")
print0(f"Learning rate: {learning_rate}")

# Initialize model
model_config_kwargs = dict(
    sequence_len=T,
    vocab_size=config["vocab_size"],
    n_layer=config["n_layers"],
    n_head=config["n_heads"],
    n_kv_head=config["n_heads"],
    n_embd=config["d_model"],
)

with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)

model.to_empty(device=device)
model.init_weights()
orig_model = model
model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")

# Initialize optimizer (simple AdamW)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
)

# Initialize dataloaders
print0("\nInitializing dataloaders...")
train_loader = tct_k8s_data_loader(
    B=B,
    T=T,
    split="train",
    data_dir=data_dir,
    train_split=train_split,
    device=device,
    cache_file=cache_file,
    shuffle=True,
)

# Validation loader builder (created on-demand)
def build_val_loader():
    return tct_k8s_data_loader(
        B=B,
        T=T,
        split="val",
        data_dir=data_dir,
        train_split=train_split,
        device=device,
        cache_file=cache_file,
        shuffle=False,
    )

# Create infinite iterator (cycles through epochs)
import itertools
train_iter = itertools.cycle(train_loader)

# Prefetch first batch
x, y = next(train_iter)
print0(f"First batch shape: x={x.shape}, y={y.shape}")
print0(f"First batch dtype: x={x.dtype}, y={y.dtype}")
print0()

# Learning rate scheduler
def get_lr(step):
    # Warmup
    if step < warmup_iters:
        return learning_rate * (step + 1) / warmup_iters
    # Constant LR after warmup
    return learning_rate

# Checkpoint saving
def save_checkpoint(step, val_loss):
    if not master_process:
        return

    output_dirname = model_tag if model_tag else f"{model_size}_{step:06d}"
    checkpoint_dir = Path("checkpoints") / output_dirname
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"model_{step:06d}.pt"
    torch.save(orig_model.state_dict(), checkpoint_path)

    # Save config
    config_path = checkpoint_dir / "config.json"
    import json
    with open(config_path, 'w') as f:
        json.dump({
            "model_size": model_size,
            "step": step,
            "val_loss": val_loss,
            "model_config": model_config_kwargs,
            "user_config": user_config,
        }, f, indent=2)

    print0(f"✅ Checkpoint saved: {checkpoint_path}")

# Validation
@torch.no_grad()
def evaluate(eval_steps=100):
    model.eval()
    val_loader = build_val_loader()
    val_iter = iter(val_loader)
    total_loss = 0.0
    actual_steps = 0

    for _ in range(eval_steps):
        try:
            x, y = next(val_iter)
            with autocast_ctx:
                loss = model(x, y)
            total_loss += loss.item()
            actual_steps += 1
        except StopIteration:
            # Exhausted validation set
            break

    avg_loss = total_loss / actual_steps if actual_steps > 0 else 0.0
    model.train()
    return avg_loss

# -----------------------------------------------------------------------------
# Training loop
print0("=" * 80)
print0("STARTING TRAINING")
print0("=" * 80)
print0()

min_val_loss = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0

for step in range(num_iterations + 1):
    last_step = step == num_iterations

    # Evaluation
    if last_step or step % eval_every == 0:
        val_loss = evaluate(eval_steps=100)
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        print0(f"Step {step:05d} | Val loss: {val_loss:.4f} | Val ppl: {val_ppl:.2f}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            print0(f"  ✨ New best validation loss!")

    # Checkpointing
    if master_process and (last_step or (step > 0 and step % save_every == 0)):
        save_checkpoint(step, val_loss if 'val_loss' in locals() else 0.0)

    if last_step:
        break

    # Training step
    synchronize()
    t0 = time.time()

    # Gradient accumulation
    for micro_step in range(grad_accum):
        with autocast_ctx:
            loss = model(x, y)

        train_loss = loss.detach()
        loss = loss / grad_accum  # normalize for accumulation
        loss.backward()

        # Prefetch next batch
        x, y = next(train_iter)

    # Gradient clipping
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)

    # Optimizer step with LR schedule
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    model.zero_grad(set_to_none=True)

    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # Logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(B * T * grad_accum * ddp_world_size / dt)

    if step > 10:
        total_training_time += dt

    if step % 10 == 0:
        train_ppl = torch.exp(torch.tensor(debiased_smooth_loss)).item()
        print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.1f}%) | "
               f"loss: {debiased_smooth_loss:.4f} | ppl: {train_ppl:.1f} | "
               f"lr: {lr:.2e} | dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,} | "
               f"time: {total_training_time/60:.1f}m")

print0()
print0("=" * 80)
print0("TRAINING COMPLETE")
print0("=" * 80)
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.0f}MiB")
print0(f"Total time: {total_training_time/60:.1f}m ({total_training_time/3600:.2f}h)")
print0(f"Min val loss: {min_val_loss:.4f}")
print0()

# Cleanup
compute_cleanup()
