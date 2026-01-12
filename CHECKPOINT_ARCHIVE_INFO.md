# Checkpoint Archive

## Archive Details
- File: checkpoints.tar.xz
- Size: 4.4GB (compressed with xz)
- Format: tar + xz compression
- Contents: All 24 model checkpoints (3 schemas × 2 tokenizers × 4 sizes)

## S3 Location
```
s3://m4rt30bdh2/nanochat-tct/checkpoints.tar.xz
Region: eu-ro-1
Endpoint: https://s3api-eu-ro-1.runpod.io
```

## Download from S3
```bash
aws s3 cp s3://m4rt30bdh2/nanochat-tct/checkpoints.tar.xz . \
    --region eu-ro-1 \
    --endpoint-url https://s3api-eu-ro-1.runpod.io
```

## Unpack
```bash
# Use --no-same-owner to avoid permission warnings
tar -xJf checkpoints.tar.xz --no-same-owner
```

This will extract the `checkpoints/` directory with all models.

## Verify After Unpacking
```bash
# Should show 24 model directories
ls checkpoints/ | wc -l

# Test with eval script
python -m scripts.eval_icml --schema kubernetes --model_size mini --bpb_only --num_samples 10
```

## Archive Contents
Each checkpoint directory contains:
- epoch_XXX.pt - Best checkpoint for that model
- config.json - Model configuration

Models included:
- kubernetes_{tct,utf8}_{tiny,mini,base,small}
- tsconfig_{tct,utf8}_{tiny,mini,base,small}
- eslintrc_{tct,utf8}_{tiny,mini,base,small}
