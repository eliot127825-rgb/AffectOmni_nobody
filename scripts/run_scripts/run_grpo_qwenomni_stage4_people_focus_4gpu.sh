#!/bin/bash
# Stage 4: People-focus enhanced GRPO training (4-GPU version)
# Based on the final model, enhance people-focus ability via GRPO
# Optimization: reduce candidates to avoid OOM
#
# Usage:
#   Test mode: bash run_grpo_qwenomni_stage4_people_focus_4gpu.sh test
#   Full training: bash run_grpo_qwenomni_stage4_people_focus_4gpu.sh

# Check if test mode
TEST_MODE=${1:-""}
if [ "$TEST_MODE" = "test" ]; then
    echo "Test mode: using single-sample data"
    DATA_CONFIG="data_config/stage4_test_single.yaml"
    RUN_NAME="stage4_test_single"
    MAX_STEPS=5  # test mode runs only 5 steps
else
    echo "Full training mode"
    DATA_CONFIG="data_config/stage4_people_focus.yaml"
    RUN_NAME="stage4_people_focus_4gpu"
    MAX_STEPS=-1  # -1 means run entire epoch
fi

ARG_WORLD_SIZE=${2:-1}
ARG_NPROC_PER_NODE=${3:-4}  # default 4 GPUs
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

# Create log directory
LOG_DIR="log"
mkdir -p "$LOG_DIR"

# Generate log filename with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${RUN_NAME}_${TIMESTAMP}.log"

echo "Log will be saved to: $LOG_FILE"

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

export LOG_PATH="./debug_log_$RUN_NAME.txt"

# Environment variables
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Create output directory
mkdir -p ../outputs/$RUN_NAME
cp $0 ../outputs/$RUN_NAME

# Model path: start from the trained model
MODEL_PATH="${PROJECT_ROOT}/models/base_model"

echo "=========================================="
echo "Stage 4: People-focus enhanced GRPO training (4-GPU version)"
echo "=========================================="
echo "Run mode: $([[ "$TEST_MODE" == "test" ]] && echo "test mode" || echo "full training")"
echo "Base model: $MODEL_PATH"
echo "Data config: $DATA_CONFIG"
echo "Reward functions: format + accuracy + people_focus"
echo "Training strategy: conservative (low lr + large grad accumulation + KL penalty)"
echo "Memory optimization: 4 candidates (8 for 8-GPU version)"
echo "Checkpoint: save every 100 steps"
echo "=========================================="
echo ""

torchrun --nproc_per_node 4 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/grpo_qwenomni.py \
    --deepspeed run_scripts/zero3_offload.json \
    --output_dir ../outputs/$RUN_NAME \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_CONFIG \
    \
    `# Generation config (4-GPU optimization: reduce candidates)` \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --num_generations 2 \
    \
    `# Training config (4-GPU optimization)` \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 12 \
    --num_train_epochs 1 \
    --max_steps $MAX_STEPS \
    --freeze_vision_modules true \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to none \
    --scale_rewards false \
    \
    `# Reward config` \
    --reward_funcs format accuracy people_focus \
    \
    `# Other config` \
    --use_audio_in_video true \
    --gradient_checkpointing true \
    --log_completions true \
    --attn_implementation flash_attention_2 \
    --run_name $RUN_NAME \
    \
    `# Checkpoint config` \
    --save_steps 100 \
    --save_only_model false \
    2>&1 | tee ../outputs/$RUN_NAME/train.log | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Output directory: ../outputs/$RUN_NAME"
echo "Training log: ../outputs/$RUN_NAME/train.log"
echo "Detailed log: $LOG_FILE"
echo "=========================================="

exit ${PIPESTATUS[0]}
