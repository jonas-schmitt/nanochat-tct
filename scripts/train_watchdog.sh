#!/bin/bash
# Watchdog script that automatically restarts BPE training on failure
# Saves checkpoints every 5000 steps, resumes from latest

MODEL_TAG="bpe_169_full"
CHECKPOINT_DIR="checkpoints-bpe/$MODEL_TAG"
LOG_FILE="logs/bpe_169_full_training.log"
MAX_ITERS=200000

get_latest_checkpoint() {
    # Find latest checkpoint step number
    latest=$(ls -1 $CHECKPOINT_DIR/model_*.pt 2>/dev/null | sed 's/.*model_0*\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
    echo ${latest:-0}
}

while true; do
    RESUME_STEP=$(get_latest_checkpoint)
    echo "[$(date)] Starting training from step $RESUME_STEP..."

    if [ "$RESUME_STEP" -ge "$MAX_ITERS" ]; then
        echo "[$(date)] Training complete!"
        break
    fi

    if [ "$RESUME_STEP" -eq 0 ]; then
        python -m scripts.bpe_train_simple \
            --model_size=bpe-169-small-2048 \
            --num_iterations=$MAX_ITERS \
            --eval_every=5000 \
            --model_tag=$MODEL_TAG >> $LOG_FILE 2>&1
    else
        python -m scripts.bpe_train_simple \
            --model_size=bpe-169-small-2048 \
            --num_iterations=$MAX_ITERS \
            --eval_every=5000 \
            --model_tag=$MODEL_TAG \
            --resume_from_step=$RESUME_STEP >> $LOG_FILE 2>&1
    fi

    EXIT_CODE=$?
    echo "[$(date)] Training exited with code $EXIT_CODE at step $(get_latest_checkpoint)"

    if [ "$EXIT_CODE" -eq 0 ]; then
        echo "[$(date)] Training completed successfully!"
        break
    fi

    echo "[$(date)] Training crashed, restarting in 10 seconds..."
    sleep 10
done
