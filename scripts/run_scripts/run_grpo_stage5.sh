#!/bin/bash
# Stage 5实验：结果奖励优化（Outcome Reward + Thinking Focus）
# 基于 checkpoint-380 继续训练，验证新reward组合的有效性

echo "🚀 Stage 5实验：结果奖励优化（目标71%+）"

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

export CUDA_VISIBLE_DEVICES=4,5,6,7
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
echo "Stage 5实验：结果奖励优化（基于checkpoint-380）"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "数据: $DATA_CONFIG"
echo ""
echo "📊 实验配置："
echo "  【数据集】"
echo "    - Social-IQ全集: 2737样本 (70%)"
echo "    - EMER全集: 150样本 (20%)"
echo "    - NExT-QA子集: 300样本 (10%, 新增通用数据)"
echo "  【训练参数】"
echo "    - learning_rate: 1e-6 (更保守，避免破坏已有优化)"
echo "    - num_epochs: 2-3 (快速验证)"
echo "    - gradient_accumulation: 16 (有效batch=64)"
echo "    - num_generations: 4 (GRPO候选数)"
echo "  【Reward函数】"
echo "    - accuracy (0.4): 答案准确率"
echo "    - thinking_focus (0.2): thinking聚焦正确答案"
echo "    - people_focus (0.2): 人物关注度（对比打分✨）"
echo "    - temporal_order (0.2): 时序分析（对比打分✨）"
echo "  【目标】"
echo "    - IntentBench: 69.36% → 71%+ (提升1.64%+)"
echo "    - Daily-Omni: 62.57% → 64%+ (提升1.43%+)"
echo "=========================================="
echo ""

torchrun --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/grpo_qwenomni_stage5.py \
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
    --num_train_epochs 3 \
    --learning_rate 1e-6 \
    \
    `# GRPO特有配置 - 优化强化学习` \
    --num_iterations 2 \
    --beta 0.02 \
    --epsilon 0.2 \
    \
    `# Reward配置 - Stage 5新组合` \
    --reward_funcs accuracy thinking_focus people_focus temporal_order \
    --reward_weights 0.4 0.2 0.2 0.2 \
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
