# RunPod Training Guide

## Prerequisites

1. RunPod account with GPU access
2. Training data uploaded to the pod

## Quick Start

### 1. Create Pod

Create a new pod on RunPod with:
- Template: PyTorch 2.1 (or similar with CUDA 12.x)
- GPU: RTX 4090 (recommended) or A100
- Container Disk: 50GB+
- Volume: 100GB+ (for data and checkpoints)

### 2. Upload Training Data

From your local machine:

```bash
# Upload single schema
rsync -avz --progress ~/Desktop/data/kubernetes-tct-bpe/ \
    root@YOUR_POD_IP:/workspace/data/kubernetes-tct-bpe/

# Upload all data for one schema (both tokenizers)
rsync -avz --progress ~/Desktop/data/kubernetes-tct-bpe/ \
    root@YOUR_POD_IP:/workspace/data/kubernetes-tct-bpe/
rsync -avz --progress ~/Desktop/data/kubernetes-utf8-bpe/ \
    root@YOUR_POD_IP:/workspace/data/kubernetes-utf8-bpe/
```

### 3. Clone Repository

SSH into pod and clone:

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/nanochat-tct.git
cd nanochat-tct
pip install -e .
```

### 4. Run Training

```bash
# Train TCT-BPE on kubernetes
./scripts/runpod/train_runpod.sh kubernetes tct small

# Train UTF8-BPE on kubernetes
./scripts/runpod/train_runpod.sh kubernetes utf8 small

# Or run directly
python -m scripts.train_unified \
    --schema=kubernetes \
    --tokenizer=tct \
    --model_size=small \
    --data_root=/workspace/data
```

### 5. Download Checkpoints

From your local machine:

```bash
rsync -avz root@YOUR_POD_IP:/workspace/nanochat-tct/checkpoints/ ./checkpoints/
```

## Running All Experiments

To run all 18 experiments (3 schemas × 3 sizes × 2 tokenizers):

```bash
# Screen/tmux recommended for long training
screen -S tct

# Run experiments sequentially
for schema in tsconfig eslintrc kubernetes; do
    for tokenizer in tct utf8; do
        for size in small medium large; do
            echo "Training: ${schema} ${tokenizer} ${size}"
            ./scripts/runpod/train_runpod.sh ${schema} ${tokenizer} ${size}
        done
    done
done
```

## Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Watch training logs
tail -f logs/*.log
```

## Cost Estimates

| GPU | $/hr | Small | Medium | Large |
|-----|------|-------|--------|-------|
| RTX 4090 | ~$0.40 | ~$5 | ~$15 | ~$50 |
| A100 40GB | ~$1.50 | ~$10 | ~$30 | ~$100 |

Total for all 18 experiments: ~$200-400 depending on GPU choice.
