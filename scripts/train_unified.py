#!/usr/bin/env python3
"""
Unified Training Script for TCT Experiments

Train language models on pre-encoded JSONL data for any schema and tokenizer.
Works with both TCT-BPE and UTF8-BPE tokenized data (same JSONL format).

Usage:
    # Train TCT-BPE on kubernetes
    python -m scripts.train_unified --schema kubernetes --tokenizer tct --model_size small

    # Train UTF8-BPE on eslintrc
    python -m scripts.train_unified --schema eslintrc --tokenizer utf8 --model_size small

    # Distributed training
    torchrun --nproc_per_node=8 -m scripts.train_unified --schema kubernetes --tokenizer tct

Configuration loaded from configs/ module:
    - Schema config (vocab size, context length, data paths)
    - Model config (architecture, training hyperparameters)
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import sys
import time
import itertools
from pathlib import Path
from contextlib import nullcontext

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, compute_cleanup, print0, print_banner, autodetect_device_type
from configs import (
    get_schema_config,
    get_model_config,
    create_dataloader,
    create_reshuffled_dataloaders,
    get_epoch_steps,
    get_warmup_steps,
)

print_banner()

# -----------------------------------------------------------------------------
# User settings (can be overridden via CLI)
device_type = ""        # cuda|cpu|mps (empty => autodetect)
schema = "kubernetes"   # tsconfig, eslintrc, kubernetes
tokenizer = "tct"       # tct or utf8
model_size = "small"    # small, medium, large
epochs = None           # None => use schema default
data_root = None        # None => ~/Desktop/data
model_tag = ""          # optional tag for checkpoint directory
warmup_fraction = 0.05  # warmup as fraction of first epoch
grad_clip = 1.0         # gradient clipping
device_batch_size = None  # None => use config default
gradient_accumulation_override = None  # None => use config default
eff_batch = None          # None => use config default (64), or override effective batch size
lr_schedule = None        # None => use config default (cosine), or "constant"/"cosine"
dropout = None          # None => use config default (0.0), or 0.0-0.5
learning_rate_override = None  # None => use config default, or e.g. 3e-4
resume_from_epoch = 0   # resume training from this epoch (0 = start fresh)
eval_every_epoch = 1    # evaluate every N epochs
save_every_pct = None   # None => auto (2% on RunPod, 5% otherwise), or override
num_eval_batches = 100  # number of batches for validation
reshuffle_data = True   # reshuffle train+val data randomly (fixes sequential split)

# Auto-detect RunPod for more frequent checkpoints (preemptible instances)
is_runpod = os.environ.get("RUNPOD_POD_ID") is not None
if save_every_pct is None:
    save_every_pct = 2 if is_runpod else 5

# CLI override
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, type(None)))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Load schema config
schema_cfg = get_schema_config(schema, data_root)

# Select tokenizer-specific settings
if tokenizer == "tct":
    vocab_size = schema_cfg["tct_vocab_size"]
    data_path = schema_cfg["data_path_tct"]
elif tokenizer == "utf8":
    vocab_size = schema_cfg["utf8_vocab_size"]
    data_path = schema_cfg["data_path_utf8"]
else:
    raise ValueError(f"Unknown tokenizer: '{tokenizer}'. Use 'tct' or 'utf8'")

context_size = schema_cfg["context_size"]
base_epochs = epochs if epochs is not None else schema_cfg["default_epochs"]
train_tokens = schema_cfg.get(f"train_tokens_{tokenizer}", schema_cfg["train_tokens_tct"])

# Load model config (pass base_epochs, will apply multiplier inside)
model_cfg = get_model_config(model_size, vocab_size, context_size, base_epochs)

# Apply epochs multiplier if present (e.g., small-long uses 2x epochs)
epochs_multiplier = model_cfg.get("epochs_multiplier", 1)
num_epochs = base_epochs * epochs_multiplier

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# Training parameters
B = device_batch_size if device_batch_size is not None else model_cfg["batch_size"]
T = context_size
# Calculate gradient accumulation: explicit override > eff_batch > config default
if gradient_accumulation_override is not None:
    grad_accum = gradient_accumulation_override
elif eff_batch is not None:
    grad_accum = max(1, eff_batch // B)
else:
    grad_accum = model_cfg["gradient_accumulation"]

# Learning rate with automatic batch size scaling (sqrt rule)
# Base LRs calibrated for batch 64: small=4e-4, medium=3e-4, large=2e-4
# Scales with sqrt(actual_batch/64) if batch differs from reference
REFERENCE_BATCH = 64  # LR values in config are calibrated for this batch size
actual_eff_batch = B * grad_accum * ddp_world_size
base_lr = model_cfg["learning_rate"]  # Model-specific base LR
if learning_rate_override is not None:
    learning_rate = learning_rate_override
elif actual_eff_batch != REFERENCE_BATCH:
    # Scale LR with sqrt of batch ratio
    import math
    scale = math.sqrt(actual_eff_batch / REFERENCE_BATCH)
    learning_rate = base_lr * scale
else:
    learning_rate = base_lr
weight_decay = model_cfg["weight_decay"]
beta1 = model_cfg["beta1"]
beta2 = model_cfg["beta2"]

# Calculate training steps
steps_per_epoch = get_epoch_steps(train_tokens, T, B, grad_accum, ddp_world_size)
total_steps = steps_per_epoch * num_epochs
warmup_steps = get_warmup_steps(train_tokens, T, B, grad_accum, ddp_world_size, warmup_fraction)
save_interval = max(1, total_steps * save_every_pct // 100)

print0("=" * 80)
print0(f"EXPERIMENT: {schema} / {tokenizer}-BPE / {model_size}")
print0("=" * 80)
print0(f"Schema: {schema}")
print0(f"Tokenizer: {tokenizer}-BPE")
print0(f"Data path: {data_path}")
print0(f"Vocab size: {vocab_size:,}")
print0(f"Context size: {T}")
print0(f"Model size: {model_size} ({model_cfg['estimated_params']:,} params)")
dropout_effective = dropout if dropout is not None else model_cfg.get("dropout", 0.0)
print0(f"Dropout: {dropout_effective}")
print0()
print0(f"Batch size: {B}")
print0(f"Gradient accumulation: {grad_accum}")
print0(f"Effective batch size: {B * grad_accum * ddp_world_size}")
lr_sched_effective = lr_schedule if lr_schedule else model_cfg.get("lr_schedule", "constant")
lr_sched_desc = "constant" if lr_sched_effective == "constant" else f"cosine decay to {learning_rate * 0.1:.1e}"
lr_scaled_note = f" (scaled from {base_lr:.1e} for batch {actual_eff_batch})" if actual_eff_batch != REFERENCE_BATCH and learning_rate_override is None else ""
print0(f"Learning rate: {learning_rate:.4e} ({lr_sched_desc}){lr_scaled_note}")
print0()
print0(f"Epochs: {num_epochs}")
print0(f"Steps per epoch: {steps_per_epoch}")
print0(f"Total steps: {total_steps}")
print0(f"Warmup steps: {warmup_steps}")
runpod_note = " (RunPod detected)" if is_runpod else ""
print0(f"Save interval: every {save_interval} steps ({save_every_pct}%){runpod_note}")
print0()

# Initialize model
# Auto-enable gradient checkpointing when dropout > 0 (dropout + torch.compile needs more memory)
use_grad_checkpoint = dropout_effective > 0
if use_grad_checkpoint:
    print0(f"Gradient checkpointing: ENABLED (dropout={dropout_effective} requires extra memory)")
else:
    print0("Gradient checkpointing: disabled")

model_config_kwargs = dict(
    sequence_len=T,
    vocab_size=vocab_size,
    n_layer=model_cfg["n_layers"],
    n_head=model_cfg["n_heads"],
    n_kv_head=model_cfg["n_heads"],
    n_embd=model_cfg["d_model"],
    dropout=dropout_effective,
    use_swiglu=model_cfg.get("use_swiglu", False),
    ffn_mult=model_cfg.get("ffn_mult", 4.0),
    gradient_checkpointing=use_grad_checkpoint,
)

with torch.device("meta"):
    gpt_config = GPTConfig(**model_config_kwargs)
    model = GPT(gpt_config)

model.to_empty(device=device)
model.init_weights()

# Resume from checkpoint if specified
start_step = 0
if resume_from_epoch > 0:
    checkpoint_dir = Path("checkpoints") / (model_tag if model_tag else f"{schema}_{tokenizer}_{model_size}")
    checkpoint_path = checkpoint_dir / f"epoch_{resume_from_epoch:03d}.pt"
    if checkpoint_path.exists():
        print0(f"Resuming from checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        start_step = resume_from_epoch * steps_per_epoch
        del state_dict
        gc.collect()
        if device_type == "cuda":
            torch.cuda.empty_cache()
    else:
        print0(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

orig_model = model
model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")

# Initialize optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(beta1, beta2),
    weight_decay=weight_decay,
)

# Initialize dataloaders
print0("\nInitializing dataloaders...")
if reshuffle_data:
    print0("Reshuffling data (combining train+val, random split)...")
    train_loader, val_loader_static = create_reshuffled_dataloaders(
        data_dir=data_path,
        context_size=T,
        batch_size=B,
        train_ratio=0.9,
        device=device,
        verbose=master_process,
        seed=42,
    )

    def build_val_loader():
        return val_loader_static
else:
    train_loader = create_dataloader(
        data_dir=data_path,
        context_size=T,
        batch_size=B,
        split="train",
        device=device,
        shuffle=True,
        verbose=master_process,
    )

    def build_val_loader():
        return create_dataloader(
            data_dir=data_path,
            context_size=T,
            batch_size=B,
            split="val",
            device=device,
            shuffle=False,
            verbose=False,
        )

# Create infinite iterator
train_iter = itertools.cycle(train_loader)

# Prefetch first batch
x, y = next(train_iter)
print0(f"First batch shape: x={x.shape}, y={y.shape}")
print0()

# Learning rate scheduler
import math
min_lr = learning_rate * 0.1  # Decay to 10% of max LR
# Use CLI override if provided, otherwise use config default
lr_schedule_actual = lr_schedule if lr_schedule else model_cfg.get("lr_schedule", "constant")

def get_lr(step):
    # Warmup phase
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    # After warmup: constant or cosine decay
    if lr_schedule_actual == "constant":
        return learning_rate
    # Cosine decay phase
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * progress))

# Checkpoint saving
def save_checkpoint(step, epoch, val_loss, is_best=False):
    if not master_process:
        return

    output_dirname = model_tag if model_tag else f"{schema}_{tokenizer}_{model_size}"
    checkpoint_dir = Path("checkpoints") / output_dirname
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save epoch checkpoint
    checkpoint_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
    torch.save(orig_model.state_dict(), checkpoint_path)

    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best.pt"
        torch.save(orig_model.state_dict(), best_path)
        print0(f"  New best model saved!")

    # Save config
    config_path = checkpoint_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "schema": schema,
            "tokenizer": tokenizer,
            "model_size": model_size,
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "model_config": model_config_kwargs,
            "schema_config": {k: str(v) if isinstance(v, Path) else v for k, v in schema_cfg.items()},
            "user_config": user_config,
        }, f, indent=2)

    print0(f"Checkpoint saved: {checkpoint_path}")

# Validation
@torch.no_grad()
def evaluate():
    model.eval()
    val_loader = build_val_loader()
    val_iter = iter(val_loader)
    total_loss = 0.0
    actual_batches = 0

    for _ in range(num_eval_batches):
        try:
            x_val, y_val = next(val_iter)
            with autocast_ctx:
                loss = model(x_val, y_val)
            total_loss += loss.item()
            actual_batches += 1
        except StopIteration:
            break

    avg_loss = total_loss / actual_batches if actual_batches > 0 else 0.0
    model.train()

    del val_loader, val_iter
    gc.collect()
    if device_type == "cuda":
        torch.cuda.empty_cache()

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

for step in range(start_step, total_steps + 1):
    current_epoch = step // steps_per_epoch
    epoch_step = step % steps_per_epoch
    last_step = step == total_steps

    # Evaluation at epoch boundaries
    if last_step or (epoch_step == 0 and current_epoch > 0 and current_epoch % eval_every_epoch == 0):
        val_loss = evaluate()
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        is_best = val_loss < min_val_loss
        if is_best:
            min_val_loss = val_loss
        print0(f"Epoch {current_epoch:3d} | Step {step:6d} | Val loss: {val_loss:.4f} | Val ppl: {val_ppl:.2f}" +
               (" (best)" if is_best else ""))

    # Checkpointing
    if master_process and (last_step or (step > 0 and step % save_interval == 0)):
        save_checkpoint(step, current_epoch, val_loss if 'val_loss' in locals() else 0.0,
                       is_best=val_loss < min_val_loss if 'val_loss' in locals() else False)

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
        loss = loss / grad_accum
        loss.backward()

        x, y = next(train_iter)

    # Gradient clipping
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)

    # Optimizer step
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
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step - start_step + 1))

    if step > start_step + 10:
        total_training_time += dt

    if step % 10 == 0:
        pct_done = 100 * step / total_steps
        tok_per_sec = int(B * T * grad_accum * ddp_world_size / dt)
        train_ppl = torch.exp(torch.tensor(debiased_smooth_loss)).item()
        print0(f"E{current_epoch:02d} step {step:06d}/{total_steps:06d} ({pct_done:.1f}%) | "
               f"loss: {debiased_smooth_loss:.4f} | ppl: {train_ppl:.1f} | "
               f"lr: {lr:.2e} | dt: {dt*1000:.0f}ms | tok/s: {tok_per_sec:,}")

    # Periodic memory cleanup to prevent fragmentation
    if step % 1000 == 0 and device_type == "cuda":
        torch.cuda.empty_cache()

print0()
print0("=" * 80)
print0("TRAINING COMPLETE")
print0("=" * 80)
print0(f"Schema: {schema}")
print0(f"Tokenizer: {tokenizer}-BPE")
print0(f"Model: {model_size}")
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.0f}MiB")
print0(f"Total time: {total_training_time/60:.1f}m ({total_training_time/3600:.2f}h)")
print0(f"Min val loss: {min_val_loss:.4f}")
print0()

# Cleanup
del model, orig_model, optimizer, train_loader
gc.collect()
if device_type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
compute_cleanup()
