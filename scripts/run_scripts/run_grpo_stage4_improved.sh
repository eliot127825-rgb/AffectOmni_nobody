#!/bin/bash
# Stage 4: 人物关注增强 GRPO 训练 - 改进版
# 基于 stage4_debug_no_audio_v2.sh 的优化配置

echo "🚀 Stage 4 改进训练：提升效果的优化配置"

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

export LOG_PATH="./train_log_$RUN_NAME.txt"
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export USE_API_REWARD=true
export USE_COMBINED_REWARD=false
export DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:-"your_api_key_here"}

# 输出到data3避免磁盘空间不足
OUTPUT_BASE_DIR="${OUTPUT_DIR}"
mkdir -p $OUTPUT_BASE_DIR/$RUN_NAME

# 使用训练好的checkpoint-380继续训练
MODEL_PATH="./outputs/stage4_debug_no_audio_v2/checkpoint-380"

echo "=========================================="
echo "Stage 4: 改进训练配置（基于checkpoint-380）"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "数据: $DATA_CONFIG"
echo ""
echo "📊 关键改进："
echo "  ✓ num_generations: 4 → 12 (增加候选多样性)"
echo "  ✓ gradient_accumulation: 4 → 16 (有效batch=64)"
echo "  ✓ max_completion: 512 → 1024 (更长推理链)"
echo "  ✓ num_epochs: 1 → 2 (充分训练)"
echo "  ✓ learning_rate: 默认 → 2e-6 (加快收敛)"
echo "  ✓ num_iterations: 1 → 2 (GRPO多轮优化)"
echo "  ✓ beta: 默认0.04 → 0.02 (降低KL惩罚)"
echo "  ✓ reward: 新增 temporal_order (时序分析约束)"
echo "=========================================="
echo ""

torchrun --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/grpo_qwenomni.py \
    --deepspeed run_scripts/zero3_offload.json \
    --output_dir $OUTPUT_BASE_DIR/$RUN_NAME \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_CONFIG \
    \
    `# 生成配置 - 核心改进` \
    --max_prompt_length 2048 \
    --max_completion_length 512 \
    --num_generations 4 \
    \
    `# 训练配置 - 提升有效batch size和epochs` \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 2 \
    --learning_rate 2e-6 \
    \
    `# GRPO特有配置 - 优化强化学习` \
    --num_iterations 2 \
    --beta 0.02 \
    --epsilon 0.2 \
    \
    `# Reward配置` \
    --reward_funcs format accuracy people_focus temporal_order \
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
    --save_steps 50 \
    --save_total_limit 3 \
    --save_only_model true \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "输出目录: $OUTPUT_BASE_DIR/$RUN_NAME"
echo "训练日志: $LOG_FILE"
echo "=========================================="

exit ${PIPESTATUS[0]}
