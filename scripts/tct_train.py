"""
TCT Training Script for GitHub Actions Workflow Generation

Train a decoder-only transformer on GitHub workflows using TCT tokenization.

Usage:
    # Smoke test (10 iterations)
    python -m scripts.tct_train --num_iterations 10 --model_size small

    # Full training
    python -m scripts.tct_train --model_size medium --data_dir ~/Desktop/data/prepared/

    # Distributed training
    torchrun --nproc_per_node=8 -m scripts.tct_train --model_size medium
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
from pathlib import Path
from contextlib import nullcontext

import wandb
import torch

# Add tct-bundle adapters to path
sys.path.insert(0, str(Path(__file__).parent.parent / "tct-bundle" / "adapters"))

from nanochat.gpt import GPT, GPTConfig
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb

# TCT imports
from model_config import get_config
from tct_tokenizer_adapter import TCTTokenizer
from tct_dataloader import tokenizing_distributed_data_loader

print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name ("dummy" = no logging)
# Runtime
device_type = "" # cuda|cpu|mps (empty = autodetect)
# Model
model_size = "medium" # small|medium|large (512 ctx) or small-1024|medium-1024|large-1024
data_dir = str(Path.home() / "Desktop/data/workflows/json") # Workflow JSON directory
# Training
num_iterations = 50000 # number of optimization steps (-1 = use from config)
device_batch_size = 20 # per-device batch size (768×8 with context=512 should fit in 8GB)
# Optimization (will use defaults from model config if not overridden)
learning_rate = -1.0 # learning rate (-1 = use config default)
grad_clip = 1.0 # gradient clipping
warmup_iters = -1 # warmup iterations (-1 = use config default)
# Evaluation
eval_every = 5000 # evaluate val loss every N steps
eval_max_batches = 50 # max batches for validation
# Checkpointing
save_every = 5000 # save checkpoint every N steps
checkpoint_dir = "" # checkpoint directory (empty = auto)
model_tag = "" # optional tag for checkpoint directory
resume_from = "" # No resume - train from scratch

# CLI config override
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Load TCT model configuration
print0(f"Loading {model_size.upper()} model configuration...")
config = get_config(model_size)

# Override with CLI args if provided
if num_iterations > 0:
    config["max_iters"] = num_iterations
if learning_rate > 0:
    config["learning_rate"] = learning_rate
if warmup_iters >= 0:
    config["warmup_iters"] = warmup_iters

print0(f"Model configuration:")
print0(f"  Vocab size: {config['vocab_size']:,}")
print0(f"  Context size: {config['context_size']}")
print0(f"  Model dim: {config['d_model']}")
print0(f"  Layers: {config['n_layers']}")
print0(f"  Heads: {config['n_heads']}")
print0(f"  Max iterations: {config['max_iters']:,}")
print0(f"  Learning rate: {config['learning_rate']}")
print0()

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-tct", name=run, config={**user_config, **config})

# Initialize TCT tokenizer
print0("Initializing TCT tokenizer...")
tokenizer = TCTTokenizer()
print0(f"TCT vocab size: {tokenizer.get_vocab_size():,}")
print0()

# Initialize model
print0("Initializing model...")
model_config_kwargs = dict(
    sequence_len=config["context_size"],
    vocab_size=config["vocab_size"],
    n_layer=config["n_layers"],
    n_head=config["n_heads"],
    n_kv_head=config["n_heads"], # 1:1 GQA ratio
    n_embd=config["d_model"],
)

with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)

model.to_empty(device=device)
model.init_weights()
orig_model = model # uncompiled model for saving
model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
print0()

# Initialize optimizer (simplified - just use AdamW for now)
print0("Initializing optimizer...")
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["learning_rate"],
    betas=(config["beta1"], config["beta2"]),
    weight_decay=config["weight_decay"],
)
print0()

# Resume from checkpoint if specified
start_step = 0
if resume_from:
    resume_path = Path(resume_from).expanduser()
    if not resume_path.exists():
        print0(f"Error: Checkpoint not found: {resume_path}")
        sys.exit(1)

    print0(f"Resuming from checkpoint: {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    orig_model.load_state_dict(checkpoint)

    # Load optimizer state
    optim_path = resume_path.parent / resume_path.name.replace("model_", "optim_")
    if optim_path.exists():
        print0(f"Loading optimizer state from: {optim_path}")
        try:
            optim_checkpoint = torch.load(optim_path, map_location=device)
            optimizer.load_state_dict(optim_checkpoint)
            print0("Optimizer state loaded successfully")
        except Exception as e:
            print0(f"Warning: Could not load optimizer state: {e}")
            print0("Continuing with fresh optimizer state")

    # Extract step number from checkpoint name (e.g., model_010000.pt -> 10000)
    import re
    match = re.search(r'_(\d+)\.pt$', resume_path.name)
    if match:
        start_step = int(match.group(1))
        print0(f"Resuming from step: {start_step}")

    # Recompile model after loading weights
    model = torch.compile(orig_model, dynamic=False)
    print0()

# Initialize dataloaders with epoch-based offset windowing
print0(f"Loading workflows from {data_dir}...")

if not Path(data_dir).exists():
    print0(f"Error: Workflow directory not found: {data_dir}")
    sys.exit(1)

print0("Creating epoch-based offset dataloaders...")
# For context=512, offset_stride=16 gives 32 epochs (512/16=32)
# Create prefix-aware datasets (decoder-only mode)
train_dataset = tokenizing_distributed_data_loader(
    device_batch_size=device_batch_size,
    context_size=config["context_size"],
    split="train",
    data_dir=data_dir,
    train_split=0.9,
    geometric_p=1.0,  # Pure decoder-only (no FIM)
    prefix_mode="log",  # Efficient prefix sampling
    prefix_count=20,
    prefix_bias="uniform",
)

val_dataset = tokenizing_distributed_data_loader(
    device_batch_size=device_batch_size,
    context_size=config["context_size"],
    split="val",
    data_dir=data_dir,
    train_split=0.9,
    geometric_p=1.0,  # Pure decoder-only (no FIM)
    prefix_mode="log",
    prefix_count=20,
    prefix_bias="uniform",
)
has_val = True

# Create dataloaders
from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=device_batch_size,
    shuffle=False,  # Don't shuffle - epoch offset provides diversity
    num_workers=0,
    pin_memory=(device_type == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=device_batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=(device_type == "cuda"),
)

print0(f"Training windows (epoch 0): {len(train_dataset)}")
print0(f"Validation windows (epoch 0): {len(val_dataset)}")
print0()

# Create infinite iterator for training with epoch updates
def infinite_dataloader_with_epochs(loader, dataset):
    epoch = 0
    while True:
        dataset.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1

train_iter = infinite_dataloader_with_epochs(train_loader, train_dataset)

# Learning rate scheduler
def get_lr_multiplier(step):
    warmup = config["warmup_iters"]
    max_steps = config["max_iters"]

    if step < warmup:
        return (step + 1) / warmup
    else:
        # Cosine decay after warmup
        progress = (step - warmup) / (max_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

# -----------------------------------------------------------------------------
# Training loop
print0("Starting training...")
print0(f"  Iterations: {config['max_iters']:,}")
print0(f"  Batch size: {device_batch_size}")
print0(f"  Gradient clip: {grad_clip}")
print0()

min_val_loss = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0

for step in range(start_step + 1, config["max_iters"] + 1):
    last_step = step == config["max_iters"]

    # Evaluate validation loss
    if has_val and (last_step or step % eval_every == 0):
        model.eval()
        val_losses = []

        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                if i >= eval_max_batches:
                    break
                x, y = x.to(device), y.to(device)
                with autocast_ctx:
                    loss = model(x, y)
                val_losses.append(loss.item())

        val_loss = sum(val_losses) / len(val_losses)
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        print0(f"Step {step:05d} | Val loss: {val_loss:.4f} | Val perplexity: {val_perplexity:.2f}")

        if val_loss < min_val_loss:
            min_val_loss = val_loss

        wandb_run.log({
            "step": step,
            "val/loss": val_loss,
            "val/perplexity": val_perplexity,
            "total_training_time": total_training_time,
        })

        model.train()

    # Save checkpoint periodically and at the end
    should_save = (step % save_every == 0 and step > 0) or last_step
    if master_process and should_save:
        output_dirname = model_tag if model_tag else f"tct_{model_size}"
        if checkpoint_dir:
            ckpt_dir = Path(checkpoint_dir)
        else:
            ckpt_dir = Path.home() / "Desktop" / "checkpoints" / output_dirname

        ckpt_dir.mkdir(parents=True, exist_ok=True)

        save_checkpoint(
            str(ckpt_dir),
            step,
            orig_model.state_dict(),
            [optimizer.state_dict()],
            {
                "step": step,
                "val_loss": val_loss if has_val else None,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "tct_config": config,
            }
        )
        print0(f"Saved checkpoint to {ckpt_dir / f'model_{step:05d}.pt'}")

    if last_step:
        break

    # Training step
    synchronize()
    t0 = time.time()

    # Get batch
    x, y = next(train_iter)
    x, y = x.to(device), y.to(device)

    # Forward & backward
    with autocast_ctx:
        loss = model(x, y)

    train_loss = loss.detach().item()
    loss.backward()

    # Gradient clipping
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)

    # Optimizer step
    lrm = get_lr_multiplier(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = config["learning_rate"] * lrm

    optimizer.step()
    model.zero_grad(set_to_none=True)

    synchronize()
    t1 = time.time()
    dt = t1 - t0

    # Logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss
    debiased_loss = smooth_train_loss / (1 - ema_beta**(step + 1))

    if step > 10:
        total_training_time += dt

    tokens_per_sec = int(device_batch_size * config["context_size"] / dt)
    pct_done = 100 * step / config["max_iters"]
    train_perplexity = torch.exp(torch.tensor(debiased_loss)).item()

    if step % 10 == 0:
        print0(f"step {step:05d}/{config['max_iters']:05d} ({pct_done:.1f}%) | "
               f"loss: {debiased_loss:.4f} | ppl: {train_perplexity:.2f} | "
               f"lr: {config['learning_rate'] * lrm:.2e} | dt: {dt*1000:.1f}ms | tok/s: {tokens_per_sec:,}")

    if step % 100 == 0:
        wandb_run.log({
            "step": step,
            "train/loss": debiased_loss,
            "train/perplexity": train_perplexity,
            "train/lr": config["learning_rate"] * lrm,
            "train/dt": dt,
            "train/tok_per_sec": tokens_per_sec,
            "total_training_time": total_training_time,
        })

# Final stats
print0()
print0(f"Peak memory: {get_max_memory() / 1024 / 1024:.1f} MiB")
print0(f"Total training time: {total_training_time/60:.1f}m")
if has_val:
    print0(f"Min validation loss: {min_val_loss:.4f}")
print0()
print0("Training complete!")

# Cleanup
wandb_run.finish()
compute_cleanup()
