ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=0

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

RUN_NAME="qwenomni-sft"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

# 设置 PYTHONPATH，使 Python 能找到 open_r1 模块
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

mkdir -p output/$RUN_NAME/

torchrun  --nproc_per_node $NPROC_PER_NODE --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    src/open_r1/sft.py \
    --deepspeed run_scripts/zero3_offload.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path ${PROJECT_ROOT}/Qwen2.5-Omni-7B-Thinker \
    --dataset_name data_config/stage1.yaml \
    --freeze_vision_modules true \
    --use_audio_in_video true \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --learning_rate 2.0e-5 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to none \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_seq_length 32768 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --log_level info \
    --save_only_model true 2>&1 | tee output/$RUN_NAME/train.log

