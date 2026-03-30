#!/bin/bash
# Stage 5 quick test script: verify comparative scoring logic

echo "Stage 5 quick test: verify training pipeline"

DATA_CONFIG="data_config/stage5_test.yaml"
RUN_NAME="stage5_test"

ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-4}  # use 4 GPUs for testing
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16669
ARG_RANK=0

LOG_DIR="log"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/test_${RUN_NAME}_${TIMESTAMP}.log"
echo "Log will be saved to: $LOG_FILE"

if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

export CUDA_VISIBLE_DEVICES=4,5,6,7
export LOG_PATH="./train_log_$RUN_NAME.txt"
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export USE_API_REWARD=true
export USE_COMBINED_REWARD=false
export DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:-"your_api_key_here"}

# Test output directory
OUTPUT_BASE_DIR="${OUTPUT_DIR}"
mkdir -p $OUTPUT_BASE_DIR/$RUN_NAME

# Use checkpoint-380 as starting point
MODEL_PATH="./outputs/stage4_debug_no_audio_v2/checkpoint-380"

echo "=========================================="
echo "Stage 5 test configuration"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_CONFIG (15 test samples)"
echo ""
echo "Test parameters:"
echo "  - num_generations: 4 (GRPO)"
echo "  - max_steps: 5 (quick test)"
echo "  - reward: accuracy + thinking_focus + people_focus + temporal_order"
echo "  - comparative scoring: people_focus + temporal_order"
echo "=========================================="
echo ""

torchrun --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/grpo_qwenomni_stage5.py \
    --deepspeed run_scripts/zero3_offload.json \
    --output_dir $OUTPUT_BASE_DIR/$RUN_NAME \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_CONFIG \
    \
    `# Generation config` \
    --max_prompt_length 2048 \
    --max_completion_length 512 \
    --num_generations 4 \
    \
    `# Test config - run a few steps to verify` \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_steps 5 \
    --learning_rate 1e-6 \
    \
    `# GRPO config` \
    --num_iterations 1 \
    --beta 0.02 \
    --epsilon 0.2 \
    \
    `# Reward config - Stage 5 new combination` \
    --reward_funcs accuracy thinking_focus people_focus temporal_order \
    --reward_weights 0.4 0.2 0.2 0.2 \
    --scale_rewards false \
    \
    `# Optimizer config` \
    --freeze_vision_modules true \
    --gradient_checkpointing true \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    \
    `# Other config` \
    --use_audio_in_video false \
    --data_seed 42 \
    --logging_steps 1 \
    --log_completions true \
    --report_to none \
    --run_name $RUN_NAME \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --save_only_model true \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Test succeeded! Ready for full training"
else
    echo "Test failed, please check error logs"
fi
echo "Output directory: $OUTPUT_BASE_DIR/$RUN_NAME"
echo "Training log: $LOG_FILE"
echo "=========================================="

exit ${PIPESTATUS[0]}
