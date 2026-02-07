#!/bin/bash
# Medium training configuration
#
# This trains a medium-sized model (5 blocks, 128 channels) on the full dataset
# Expected to take several hours depending on dataset size

set -e

echo "=== Medium Model Training ==="
echo ""
echo "Model: 5 blocks, 128 channels (~2M parameters)"
echo "Training: 50 epochs with early stopping"
echo "Device: CPU (use --device cuda if GPU available)"
echo ""

# Check if dataset exists
if [ ! -f "data/training.h5" ]; then
    echo "Error: Training dataset not found at data/training.h5"
    echo "Please generate it first with tools/generate_dataset.py"
    exit 1
fi

# Use virtual environment
source .venv/bin/activate

# Train
.venv/bin/python tools/train_model.py \
    --dataset data/training.h5 \
    --blocks 5 \
    --channels 128 \
    --epochs 50 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --checkpoint-dir models/medium_5b128ch \
    --save-every 5 \
    --patience 10 \
    --lr-patience 5 \
    --lr-factor 0.5 \
    --device cpu \
    --num-workers 2 \
    --verbose

echo ""
echo "=== Training Complete ==="
echo "Best model saved to: models/medium_5b128ch/best_model.pt"
echo ""
echo "Next steps:"
echo "  1. Benchmark model:"
echo "     python tools/benchmark_model.py --compare --depth 5"
echo "  2. Compare with classical baseline"
echo ""
