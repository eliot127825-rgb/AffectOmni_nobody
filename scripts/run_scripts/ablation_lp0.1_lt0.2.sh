#!/bin/bash
# 消融实验：λp=0.1, λt=0.2

# 固定参数
LAMBDA_P=0.1        # people_focus 权重
LAMBDA_T=0.2        # temporal_order 权重
LAMBDA_ACC=0.4      # accuracy 权重
LAMBDA_THINK=0.2    # thinking_focus 权重

echo "=========================================="
echo "🔬 消融实验：λp=0.1, λt=0.2"
echo "=========================================="
echo "λ_accuracy     = $LAMBDA_ACC"
echo "λ_thinking     = $LAMBDA_THINK"
echo "λp (people)    = $LAMBDA_P"
echo "λt (temporal)  = $LAMBDA_T"
echo "=========================================="

DATA_CONFIG=./src/data_config/outcome_reward_experiment.yaml
RUN_NAME="ablation_lp0.1_lt0.2"

ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-4}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16670
ARG_RANK=0

LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/ablation_${RUN_NAME}_${TIMESTAMP}.log"
echo "📝 日志将保存到: $LOG_FILE"

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

# 输出目录
OUTPUT_BASE_DIR="${OUTPUT_DIR}"
mkdir -p $OUTPUT_BASE_DIR/$RUN_NAME

# 使用checkpoint-380作为起点
MODEL_PATH="./outputs/stage4_debug_no_audio_v2/checkpoint-380"

torchrun --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/grpo_qwenomni_stage5.py \
    --deepspeed run_scripts/zero3_offload.json \
    --output_dir $OUTPUT_BASE_DIR/$RUN_NAME \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_CONFIG \
    \
    `# 生成配置` \
    --max_prompt_length 2048 \
    --max_completion_length 512 \
    --num_generations 4 \
    \
    `# 训练配置` \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    \
    `# GRPO配置` \
    --num_iterations 1 \
    --beta 0.02 \
    --epsilon 0.2 \
    \
    `# Reward配置 - 使用传入的权重` \
    --reward_funcs accuracy thinking_focus people_focus temporal_order \
    --reward_weights $LAMBDA_ACC $LAMBDA_THINK $LAMBDA_P $LAMBDA_T \
    --scale_rewards false \
    \
    `# 优化器配置` \
    --freeze_vision_modules true \
    --gradient_checkpointing true \
    --bf16 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    \
    `# 其他配置` \
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
    echo "✅ 实验完成！"
else
    echo "❌ 实验失败，请检查错误日志"
fi
echo "实验配置: λp=$LAMBDA_P, λt=$LAMBDA_T"
echo "输出目录: $OUTPUT_BASE_DIR/$RUN_NAME"
echo "训练日志: $LOG_FILE"
echo "=========================================="

exit ${PIPESTATUS[0]}
