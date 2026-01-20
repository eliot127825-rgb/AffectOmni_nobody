#!/bin/bash
# Stage 4: 人物关注增强 GRPO 训练 (4卡版本)
# 基于 HumanOmniV2 最终模型，通过 GRPO 强化人物关注能力
# 优化：降低候选数避免显存不足
#
# 使用方法:
#   测试模式: bash run_grpo_qwenomni_stage4_people_focus_4gpu.sh test
#   正式训练: bash run_grpo_qwenomni_stage4_people_focus_4gpu.sh

# 检查是否为测试模式
TEST_MODE=${1:-""}
if [ "$TEST_MODE" = "test" ]; then
    echo "⚠️  测试模式：使用单样本数据"
    DATA_CONFIG="data_config/stage4_test_single.yaml"
    RUN_NAME="stage4_test_single"
    MAX_STEPS=5  # 测试模式只跑5步
else
    echo "🚀 正式训练模式"
    DATA_CONFIG="data_config/stage4_people_focus.yaml"
    RUN_NAME="stage4_people_focus_4gpu"
    MAX_STEPS=-1  # -1表示跑完整个epoch
fi

ARG_WORLD_SIZE=${2:-1}
ARG_NPROC_PER_NODE=${3:-4}  # 默认4卡
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

# 创建log目录
LOG_DIR="log"
mkdir -p "$LOG_DIR"

# 生成日志文件名（带时间戳）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${RUN_NAME}_${TIMESTAMP}.log"

echo "📝 日志将保存到: $LOG_FILE"

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

# 环境变量
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# 创建输出目录
mkdir -p ../outputs/$RUN_NAME
cp $0 ../outputs/$RUN_NAME

# 模型路径：从训练好的 HumanOmniV2 开始
MODEL_PATH="${PROJECT_ROOT}/models/HumanOmniV2"

echo "=========================================="
echo "Stage 4: 人物关注增强 GRPO 训练 (4卡版本)"
echo "=========================================="
echo "运行模式: $([[ "$TEST_MODE" == "test" ]] && echo "测试模式" || echo "正式训练")"
echo "模型起点: $MODEL_PATH"
echo "数据配置: $DATA_CONFIG"
echo "Reward 函数: format + accuracy + people_focus"
echo "训练策略: 极保守（低学习率 + 大梯度累积 + KL惩罚）"
echo "显存优化: 4个候选 (8卡版本为8个)"
echo "Checkpoint: 每100步保存一次"
echo "=========================================="
echo ""

torchrun --nproc_per_node 4 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/grpo_qwenomni.py \
    --deepspeed run_scripts/zero3_offload.json \
    --output_dir ../outputs/$RUN_NAME \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_CONFIG \
    \
    `# 生成配置（4卡优化：降低候选数）` \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --num_generations 2 \
    \
    `# 训练配置（4卡优化）` \
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
    `# Reward 配置` \
    --reward_funcs format accuracy people_focus \
    \
    `# 其他配置` \
    --use_audio_in_video true \
    --gradient_checkpointing true \
    --log_completions true \
    --attn_implementation flash_attention_2 \
    --run_name $RUN_NAME \
    \
    `# Checkpoint 配置` \
    --save_steps 100 \
    --save_only_model false \
    2>&1 | tee ../outputs/$RUN_NAME/train.log | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "训练完成！"
echo "输出目录: ../outputs/$RUN_NAME"
echo "训练日志: ../outputs/$RUN_NAME/train.log"
echo "详细日志: $LOG_FILE"
echo "=========================================="

exit ${PIPESTATUS[0]}
