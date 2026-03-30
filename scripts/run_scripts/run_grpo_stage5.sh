#!/bin/bash
# Stage 5 experiment: outcome reward optimization (Outcome Reward + Thinking Focus)
# Continue training from checkpoint-380, verify effectiveness of new reward combination

echo "Stage 5 experiment: outcome reward optimization"

DATA_CONFIG="data_config/outcome_reward_experiment.yaml"
RUN_NAME="stage5_outcome_reward"

ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-4}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16670
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

export CUDA_VISIBLE_DEVICES=4,5,6,7
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
echo "Stage 5 experiment: outcome reward optimization (from checkpoint-380)"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_CONFIG"
echo ""
echo "Experiment configuration:"
echo "  [Dataset]"
echo "    - Social-IQ full set: 2737 samples (70%)"
echo "    - EMER full set: 150 samples (20%)"
echo "    - NExT-QA subset: 300 samples (10%, general data)"
echo "  [Training parameters]"
echo "    - learning_rate: 1e-6 (conservative, avoid degrading prior optimization)"
echo "    - num_epochs: 2-3 (quick verification)"
echo "    - gradient_accumulation: 16 (effective batch=64)"
echo "    - num_generations: 4 (GRPO candidates)"
echo "  [Reward functions]"
echo "    - accuracy (0.4): answer accuracy"
echo "    - thinking_focus (0.2): thinking focuses on correct answer"
echo "    - people_focus (0.2): people attention (comparative scoring)"
echo "    - temporal_order (0.2): temporal analysis (comparative scoring)"
echo "  [Target]"
echo "    - IntentBench: 69.36% -> 71%+"
echo "    - Daily-Omni: 62.57% -> 64%+"
echo "=========================================="
echo ""

torchrun --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/grpo_qwenomni_stage5.py \
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
    --num_train_epochs 3 \
    --learning_rate 1e-6 \
    \
    `# GRPO-specific config - optimize reinforcement learning` \
    --num_iterations 2 \
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
