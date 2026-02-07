#!/bin/bash
# Quick training test to verify pipeline works

set -e

echo "=== Quick Training Test ==="
echo ""
echo "This will:"
echo "  1. Train a tiny model (3 blocks, 64 channels) for 2 epochs"
echo "  2. Save checkpoints to models/test_checkpoints/"
echo "  3. Verify training completes successfully"
echo ""

# Use virtual environment
source .venv/bin/activate

# Train with minimal settings
.venv/bin/python tools/train_model.py \
    --dataset data/training_test.h5 \
    --blocks 3 \
    --channels 64 \
    --epochs 2 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --checkpoint-dir models/test_checkpoints \
    --save-every 1 \
    --device cpu \
    --num-workers 0 \
    --verbose

echo ""
echo "=== Test Complete ==="
echo "Check models/test_checkpoints/ for saved models"
