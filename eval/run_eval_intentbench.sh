#!/bin/bash
# IntentBench evaluation script
# Evaluate Stage 4 trained model on IntentBench test set

echo "=========================================="
echo "IntentBench Evaluation"
echo "=========================================="

# Set environment variables
export PYTHONPATH=./

# Configuration parameters
NPROC_PER_NODE=4  # use 4 GPUs
MASTER_PORT=29502

# Model path (modify as needed)
MODEL_PATH=${1:-"../outputs/stage4_debug_no_audio_v2/checkpoint-380"}
FILE_NAME=${2:-"stage4_eval"}

echo "Model path: $MODEL_PATH"
echo "Result filename: $FILE_NAME"
echo "Test set: IntentBench (2689 samples)"
echo "=========================================="

# Run evaluation
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node $NPROC_PER_NODE \
    --master-port $MASTER_PORT \
    --nnodes 1 \
    eval/eval_humanomniv2.py \
    --model-path $MODEL_PATH \
    --file-name $FILE_NAME \
    --dataset ib

echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: eval_results/ib_${FILE_NAME}.json"
echo "=========================================="
