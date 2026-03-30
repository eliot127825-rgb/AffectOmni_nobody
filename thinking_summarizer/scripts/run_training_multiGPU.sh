#!/bin/bash

# Thinking Summarizer multi-GPU training launch script
# Distributed training with 4 GPUs

# Set GPUs to use
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch multi-GPU training with torchrun
torchrun \
  --nproc_per_node=4 \
  --master_port=29500 \
  scripts/train_summarizer.py \
  --model-path ${PROJECT_ROOT}/Qwen2.5-3B-Instruct \
  --train-file ./data/training_dataset/train.json \
  --val-file ./data/training_dataset/val.json \
  --output-dir ./outputs/thinking_summarizer \
  --max-length 2048 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --lora-dropout 0.05 \
  --num-epochs 3 \
  --batch-size 2 \
  --gradient-accumulation 2 \
  --learning-rate 3e-4 \
  --warmup-ratio 0.1 \
  --save-steps 50 \
  --logging-steps 10

# Notes:
# - nproc_per_node=4: use 4 GPUs
# - batch-size=2: per-GPU batch size of 2
# - gradient-accumulation=2: 2-step gradient accumulation
# - effective batch size = 4 GPUs x 2 batch x 2 accumulation = 16
# - ~3-3.5x speedup compared to single-GPU training
