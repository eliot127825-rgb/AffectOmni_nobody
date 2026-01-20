#!/bin/bash
# IntentBench 评估脚本
# 用于评估 Stage 4 训练的模型在 IntentBench 测试集上的表现

echo "=========================================="
echo "IntentBench 评估"
echo "=========================================="

# 设置环境变量
export PYTHONPATH=./

# 配置参数
NPROC_PER_NODE=4  # 使用4个GPU
MASTER_PORT=29502

# 模型路径（根据需要修改）
MODEL_PATH=${1:-"../outputs/stage4_debug_no_audio_v2/checkpoint-380"}
FILE_NAME=${2:-"stage4_eval"}

echo "模型路径: $MODEL_PATH"
echo "结果文件名: $FILE_NAME"
echo "测试集: IntentBench (2689 samples)"
echo "=========================================="

# 运行评估
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node $NPROC_PER_NODE \
    --master-port $MASTER_PORT \
    --nnodes 1 \
    eval/eval_humanomniv2.py \
    --model-path $MODEL_PATH \
    --file-name $FILE_NAME \
    --dataset ib

echo "=========================================="
echo "评估完成！"
echo "结果保存在: eval_results/ib_${FILE_NAME}.json"
echo "=========================================="
