#!/bin/bash
# Ablation experiment: lambda_p=0.3, lambda_t=0.3

# Fixed parameters
LAMBDA_P=0.3        # people_focus weight
LAMBDA_T=0.3        # temporal_order weight
LAMBDA_ACC=0.4      # accuracy weight
LAMBDA_THINK=0.2    # thinking_focus weight

echo "=========================================="
echo "Ablation experiment: lp=0.3, lt=0.3"
echo "=========================================="
echo "λ_accuracy     = $LAMBDA_ACC"
echo "λ_thinking     = $LAMBDA_THINK"
echo "λp (people)    = $LAMBDA_P"
echo "λt (temporal)  = $LAMBDA_T"
echo "=========================================="

DATA_CONFIG=./src/data_config/outcome_reward_experiment.yaml
RUN_NAME="ablation_lp0.3_lt0.3"

ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-4}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16669
ARG_RANK=0

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/ablation_${RUN_NAME}_${TIMESTAMP}.log"
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

export CUDA_VISIBLE_DEVICES=0,1,2,3
export LOG_PATH="${OUTPUT_DIR}/logs/train_log_$RUN_NAME.txt"
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export USE_API_REWARD=true
export USE_COMBINED_REWARD=false
export DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:-"your_api_key_here"}

# Output directory
OUTPUT_BASE_DIR="${OUTPUT_DIR}"
mkdir -p $OUTPUT_BASE_DIR/$RUN_NAME

# Use checkpoint-380 as starting point
MODEL_PATH="./outputs/stage4_debug_no_audio_v2/checkpoint-380"

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
    `# Training config` \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    \
    `# GRPO config` \
    --num_iterations 1 \
    --beta 0.02 \
    --epsilon 0.2 \
    \
    `# Reward config - using provided weights` \
    --reward_funcs accuracy thinking_focus people_focus temporal_order \
    --reward_weights $LAMBDA_ACC $LAMBDA_THINK $LAMBDA_P $LAMBDA_T \
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
    echo "Experiment complete!"
else
    echo "Experiment failed, please check error logs"
fi
echo "Experiment config: lp=$LAMBDA_P, lt=$LAMBDA_T"
echo "Output directory: $OUTPUT_BASE_DIR/$RUN_NAME"
echo "Training log: $LOG_FILE"
echo "=========================================="

exit ${PIPESTATUS[0]}
