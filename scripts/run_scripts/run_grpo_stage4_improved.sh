#!/bin/bash
# Stage 4: People-focus enhanced GRPO training - improved version
# Optimized config based on stage4_debug_no_audio_v2.sh

echo "Stage 4 improved training: optimized configuration"

DATA_CONFIG="data_config/stage4_people_focus.yaml"
RUN_NAME="stage4_improved_v1"

ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-4}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16668
ARG_RANK=0

LOG_DIR="log"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${RUN_NAME}_${TIMESTAMP}.log"
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

export LOG_PATH="./train_log_$RUN_NAME.txt"
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export USE_API_REWARD=true
export USE_COMBINED_REWARD=false
export DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:-"your_api_key_here"}

# Output directory
OUTPUT_BASE_DIR="${OUTPUT_DIR}"
mkdir -p $OUTPUT_BASE_DIR/$RUN_NAME

# Continue training from checkpoint-380
MODEL_PATH="./outputs/stage4_debug_no_audio_v2/checkpoint-380"

echo "=========================================="
echo "Stage 4: improved training config (from checkpoint-380)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_CONFIG"
echo ""
echo "Key improvements:"
echo "  - num_generations: 4 -> 12 (increase candidate diversity)"
echo "  - gradient_accumulation: 4 -> 16 (effective batch=64)"
echo "  - max_completion: 512 -> 1024 (longer reasoning chain)"
echo "  - num_epochs: 1 -> 2 (more thorough training)"
echo "  - learning_rate: default -> 2e-6 (faster convergence)"
echo "  - num_iterations: 1 -> 2 (GRPO multi-round optimization)"
echo "  - beta: default 0.04 -> 0.02 (reduce KL penalty)"
echo "  - reward: added temporal_order (temporal analysis constraint)"
echo "=========================================="
echo ""

torchrun --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/grpo_qwenomni.py \
    --deepspeed run_scripts/zero3_offload.json \
    --output_dir $OUTPUT_BASE_DIR/$RUN_NAME \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_CONFIG \
    \
    `# Generation config - core improvements` \
    --max_prompt_length 2048 \
    --max_completion_length 512 \
    --num_generations 4 \
    \
    `# Training config - increase effective batch size and epochs` \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 2 \
    --learning_rate 2e-6 \
    \
    `# GRPO-specific config - optimize reinforcement learning` \
    --num_iterations 2 \
    --beta 0.02 \
    --epsilon 0.2 \
    \
    `# Reward config` \
    --reward_funcs format accuracy people_focus temporal_order \
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
    --save_steps 50 \
    --save_total_limit 3 \
    --save_only_model true \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Training complete!"
echo "Output directory: $OUTPUT_BASE_DIR/$RUN_NAME"
echo "Training log: $LOG_FILE"
echo "=========================================="

exit ${PIPESTATUS[0]}
