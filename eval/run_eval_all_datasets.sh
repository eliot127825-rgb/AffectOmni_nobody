#!/bin/bash
# Evaluate model: Daily-Omni, IntentBench, WorldSense

echo "=========================================="
echo "Batch evaluation script - Daily-Omni & IntentBench & WorldSense"
echo "=========================================="

# Set environment variables
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Configuration parameters
NPROC_PER_NODE=4  # use 4 GPUs
MASTER_PORT=29502

# Model path (modify as needed)
MODEL_PATH=${1:-"${OUTPUT_DIR}/stage4_improved_v1/checkpoint-518"}
FILE_PREFIX=${2:-"stage4_improved_v1"}

echo "Model path: $MODEL_PATH"
echo "Result file prefix: $FILE_PREFIX"
echo "=========================================="

# # Evaluate Daily-Omni (commented out)
# echo ""
# echo ">>> Evaluating Daily-Omni dataset..."
# python -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node $NPROC_PER_NODE \
#     --master-port $MASTER_PORT \
#     --nnodes 1 \
#     eval/eval_humanomniv2.py \
#     --model-path $MODEL_PATH \
#     --file-name ${FILE_PREFIX} \
#     --dataset daily

# echo ""
# echo "Daily-Omni evaluation complete! Results: eval_results/${FILE_PREFIX}/daily_${FILE_PREFIX}.json"
# echo "=========================================="

# # Evaluate IntentBench (commented out)
# echo ""
# echo ">>> Evaluating IntentBench dataset..."
# python -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node $NPROC_PER_NODE \
#     --master-port $MASTER_PORT \
#     --nnodes 1 \
#     eval/eval_humanomniv2.py \
#     --model-path $MODEL_PATH \
#     --file-name ${FILE_PREFIX} \
#     --dataset ib

# echo ""
# echo "IntentBench evaluation complete! Results: eval_results/${FILE_PREFIX}/ib_${FILE_PREFIX}.json"
# echo "=========================================="

# Evaluate WorldSense
echo ""
echo ">>> Evaluating WorldSense dataset..."
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node $NPROC_PER_NODE \
    --master-port $MASTER_PORT \
    --nnodes 1 \
    eval/eval_humanomniv2.py \
    --model-path $MODEL_PATH \
    --file-name ${FILE_PREFIX} \
    --dataset world

echo ""
echo "WorldSense evaluation complete! Results: eval_results/${FILE_PREFIX}/world_${FILE_PREFIX}.json"
echo "=========================================="

echo ""
echo "All evaluations complete!"
echo ""
echo "Result files:"
echo "  - Daily-Omni:  eval_results/${FILE_PREFIX}/daily_${FILE_PREFIX}.json"
echo "  - IntentBench: eval_results/${FILE_PREFIX}/ib_${FILE_PREFIX}.json"
echo "  - WorldSense:  eval_results/${FILE_PREFIX}/world_${FILE_PREFIX}.json"
echo "=========================================="
