#!/bin/bash

set -e

cd ${PROJECT_ROOT}/thinking_summarizer

echo "================================================================================"
echo "LoRA model training - Thinking Summarizer"
echo "================================================================================"

# Use GPU 4,5,6,7
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Training configuration
MODEL_PATH="${PROJECT_ROOT}/Qwen2.5-3B-Instruct"
TRAIN_FILE="./data/training_dataset_6770/train.json"
VAL_FILE="./data/training_dataset_6770/val.json"
OUTPUT_DIR="./outputs/thinking_summarizer_6770"

echo "Base model: $MODEL_PATH"
echo "Training data: $TRAIN_FILE (6093 samples)"
echo "Validation data: $VAL_FILE (677 samples)"
echo "Output directory: $OUTPUT_DIR"
echo "GPU: 4,5,6,7"
echo "================================================================================"

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p ./logs

# Multi-GPU training (using torchrun)
torchrun --nproc_per_node=4 --master_port=29501 \
    scripts/train_summarizer.py \
    --model-path $MODEL_PATH \
    --train-file $TRAIN_FILE \
    --val-file $VAL_FILE \
    --output-dir $OUTPUT_DIR \
    --num-epochs 3 \
    --batch-size 2 \
    --gradient-accumulation 4 \
    --learning-rate 2e-4 \
    --warmup-ratio 0.1 \
    --lora-rank 32 \
    --lora-alpha 64 \
    --lora-dropout 0.05 \
    --max-length 2048 \
    --logging-steps 10 \
    --save-steps 200

echo ""
echo "================================================================================"
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR/final_model"
echo "================================================================================"
