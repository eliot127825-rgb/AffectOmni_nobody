#!/bin/bash
# Stage 4: 调试脚本 - 使用 Zero2 替代 Zero3
# 目的：验证乱码问题是否与 Zero3 参数分片相关

echo "🔧 调试模式：使用 Zero2 替代 Zero3"

DATA_CONFIG="data_config/stage4_test_single.yaml"
RUN_NAME="stage4_debug_zero2"

ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-4}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16668
ARG_RANK=0

LOG_DIR="log"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/debug_${RUN_NAME}_${TIMESTAMP}.log"
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

export LOG_PATH="./debug_log_$RUN_NAME.txt"
export DEBUG_MODE="true"
export NCCL_SOCKET_TIMEOUT=3600
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

mkdir -p ../outputs/$RUN_NAME

MODEL_PATH="${PROJECT_ROOT}/models/HumanOmniV2"

echo "=========================================="
echo "调试: Zero2 测试"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "数据: $DATA_CONFIG"
echo "关键设置: DeepSpeed Zero2 (而非 Zero3)"
echo "=========================================="
echo ""

torchrun --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/grpo_qwenomni.py \
    --deepspeed run_scripts/zero2_offload.json \
    --output_dir ../outputs/$RUN_NAME \
    --model_name_or_path $MODEL_PATH \
    --dataset_name $DATA_CONFIG \
    --max_prompt_length 2048 \
    --max_completion_length 512 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --max_steps 3 \
    --freeze_vision_modules true \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to none \
    --scale_rewards false \
    --reward_funcs format accuracy \
    --use_audio_in_video true \
    --gradient_checkpointing true \
    --log_completions true \
    --attn_implementation flash_attention_2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model false \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "调试完成！检查日志: $LOG_FILE"
echo "=========================================="

exit ${PIPESTATUS[0]}

