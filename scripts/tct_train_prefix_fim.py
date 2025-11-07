"""
TCT Prefix-Aware FIM (Fill-in-the-Middle) Training Script for GitHub Actions Workflow Generation

Train a decoder-only transformer on GitHub workflows using TCT tokenization with prefix-aware geometric FIM.

ADDRESSES CORE TRAINING ISSUES:
- Trains on contexts of ALL lengths: 1, 2, 3, ..., context_size (not just fixed 512)
- Matches inference distribution (generation starts from empty/short context)
- Uses geometric FIM sampling for each prefix (balanced k=0/k=1 distribution)

Key Innovation:
Instead of fixed 512-token windows, creates examples with varying prefix lengths:
  - Prefix 1: [] → tok0 (with geometric FIM)
  - Prefix 2: [tok0] → tok1 (with geometric FIM)
  - ...
  - Prefix 512: [tok0...tok510] → tok511 (with geometric FIM)

This fixes train/test distribution mismatch where model only saw 512-token
contexts but inference needs to work with 0, 1, 2, ... contexts.

Usage:
    # Smoke test (10 iterations)
    python -m scripts.tct_train_prefix_fim --num_iterations 10

    # Full prefix-aware FIM training
    python -m scripts.tct_train_prefix_fim --data_dir ~/Desktop/data/workflows-100k/json

    # Pure decoder-only (no FIM)
    python -m scripts.tct_train_prefix_fim --geometric_p 1.0

    # Balanced decoder-only + FIM (recommended)
    python -m scripts.tct_train_prefix_fim --geometric_p 0.5
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
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

# TCT Prefix-Aware FIM imports
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
model_size = "small-512" # small-512|medium-512|large-512 (start small for testing)
data_dir = str(Path.home() / "Desktop/data/workflows/json") # Workflow JSON directory
# Training
num_iterations = 20000 # number of optimization steps (-1 = use from config)
device_batch_size = 32 # per-device batch size (smaller due to more diverse batch sizes)
# FIM parameters
geometric_p = 0.5   # geometric distribution parameter (0.5 = balanced decoder/FIM)
                    # p=1.0 → 100% decoder-only (k=0)
                    # p=0.5 → 50% k=0, 50% k=1+
                    # p=0.33 → 33% k=0, 67% k=1+
# Prefix sampling parameters
prefix_mode = "log" # "all"|"log"|"sample"|"hybrid" (log = powers of 2, much faster)
prefix_count = 20     # Number of prefixes for "sample"/"hybrid" modes (only used for long prefixes)
prefix_bias = "uniform" # "uniform" or "short" (bias toward short contexts)
# Optimization (will use defaults from model config if not overridden)
learning_rate = -1.0 # learning rate (-1 = use config default)
grad_clip = 1.0 # gradient clipping
warmup_iters = -1 # warmup iterations (-1 = use config default)
# Evaluation
eval_every = 2000 # evaluate val loss every N steps (more frequent for early feedback)
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

# Load model configuration
print0(f"Loading {model_size.upper()} Prefix-Aware FIM model configuration...")
config = get_config(model_size)

# Override with CLI args if provided
if num_iterations > 0:
    config["max_iters"] = num_iterations
if learning_rate > 0:
    config["learning_rate"] = learning_rate
if warmup_iters >= 0:
    config["warmup_iters"] = warmup_iters

print0(f"Model configuration:")
print0(f"  Vocab size: {config['vocab_size']:,} (8190 base + MASK at 8190 + PAD at 8191)")
print0(f"  Context size: {config['context_size']}")
print0(f"  Model dim: {config['d_model']}")
print0(f"  Layers: {config['n_layers']}")
print0(f"  Heads: {config['n_heads']}")
print0(f"  Prefix mode: {prefix_mode}")
k0_pct = int(geometric_p * 100)
k1_pct = 100 - k0_pct
print0(f"  FIM distribution: ~{k0_pct}% k=0, ~{k1_pct}% k=1+ (geometric_p={geometric_p})")
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
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-tct-prefix-fim", name=run, config={**user_config, **config})

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
# Only compile if not resuming (to avoid OOM during resume)
if not resume_from:
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

    # Note: Running without torch.compile when resuming to avoid OOM
    # This is slower but necessary with limited GPU memory
    model = orig_model
    print0()

# Initialize Prefix-Aware FIM datasets
print0(f"Loading workflows from {data_dir}...")

if not Path(data_dir).exists():
    print0(f"Error: Workflow directory not found: {data_dir}")
    sys.exit(1)

print0("Creating Prefix-Aware FIM datasets...")
# For context=512, offset_stride=16 gives 32 epochs (512/16=32)
offset_stride = config["context_size"] // 32

train_dataset = tokenizing_distributed_data_loader(
    device_batch_size=device_batch_size,
    context_size=config["context_size"],
    split="train",
    data_dir=data_dir,
    train_split=0.9,
    geometric_p=geometric_p,
    prefix_mode=prefix_mode,
    prefix_count=prefix_count,
    prefix_bias=prefix_bias,
)

val_dataset = tokenizing_distributed_data_loader(
    device_batch_size=device_batch_size,
    context_size=config["context_size"],
    split="val",
    data_dir=data_dir,
    train_split=0.9,
    geometric_p=geometric_p,
    prefix_mode=prefix_mode,
    prefix_count=prefix_count,
    prefix_bias=prefix_bias,
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

print0(f"Training examples (epoch 0): {len(train_dataset):,}")
print0(f"Validation examples (epoch 0): {len(val_dataset):,}")
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
print0("Starting Prefix-Aware FIM training...")
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
        output_dirname = model_tag if model_tag else f"tct_prefix_fim_{model_size}_p{int(geometric_p*100)}"
        ckpt_dir = Path(checkpoint_dir if checkpoint_dir else "checkpoints") / output_dirname
        save_checkpoint(
            str(ckpt_dir),
            step,
            orig_model.state_dict(),
            [optimizer.state_dict()],
            {"config": config, "model_size": model_size, "geometric_p": geometric_p, "prefix_mode": prefix_mode}
        )
        print0(f"Saved checkpoint to {ckpt_dir}")

    # Training step
    t0 = time.time()

    # Get batch
    x, y = next(train_iter)
    x, y = x.to(device), y.to(device)

    # Forward pass
    with autocast_ctx:
        loss = model(x, y)

    # Backward pass
    loss.backward()

    # Gradient clipping
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Learning rate schedule
    lr_multiplier = get_lr_multiplier(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = config["learning_rate"] * lr_multiplier

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    synchronize()
    t1 = time.time()
    dt = t1 - t0
    total_training_time += dt

    # Logging
    train_loss = loss.item()
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss
    train_perplexity = torch.exp(torch.tensor(smooth_train_loss)).item()

    if step % config["log_interval"] == 0:
        lr = optimizer.param_groups[0]["lr"]
        tokens_per_sec = device_batch_size * config["context_size"] / dt
        print0(f"step {step:05d}/{config['max_iters']:05d} ({100*step/config['max_iters']:.1f}%) | "
               f"loss: {train_loss:.4f} | ppl: {train_perplexity:.2f} | "
               f"lr: {lr:.2e} | dt: {dt*1000:.1f}ms | tok/s: {tokens_per_sec:,.0f}")

        wandb_run.log({
            "step": step,
            "train/loss": train_loss,
            "train/perplexity": train_perplexity,
            "train/lr": lr,
            "train/dt_ms": dt * 1000,
            "train/tokens_per_sec": tokens_per_sec,
            "total_training_time": total_training_time,
        })

print0("Training complete!")
print0(f"Total training time: {total_training_time/3600:.2f} hours")
print0(f"Best validation loss: {min_val_loss:.4f}")

# Cleanup
compute_cleanup()
wandb_run.finish()
