#!/bin/bash
# 评估模型：Daily-Omni, IntentBench, WorldSense

echo "=========================================="
echo "批量评估脚本 - Daily-Omni & IntentBench & WorldSense"
echo "=========================================="

# 设置环境变量
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 配置参数
NPROC_PER_NODE=4  # 使用4个GPU
MASTER_PORT=29502

# 模型路径（根据需要修改）
MODEL_PATH=${1:-"${OUTPUT_DIR}/stage4_improved_v1/checkpoint-518"}
FILE_PREFIX=${2:-"stage4_improved_v1"}

echo "模型路径: $MODEL_PATH"
echo "结果文件前缀: $FILE_PREFIX"
echo "=========================================="

# # 评估 Daily-Omni (已注释)
# echo ""
# echo ">>> 开始评估 Daily-Omni 数据集..."
# python -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node $NPROC_PER_NODE \
#     --master-port $MASTER_PORT \
#     --nnodes 1 \
#     eval/eval_humanomniv2.py \
#     --model-path $MODEL_PATH \
#     --file-name ${FILE_PREFIX} \
#     --dataset daily

# echo ""
# echo "Daily-Omni 评估完成！结果: eval_results/${FILE_PREFIX}/daily_${FILE_PREFIX}.json"
# echo "=========================================="

# # 评估 IntentBench (已注释)
# echo ""
# echo ">>> 开始评估 IntentBench 数据集..."
# python -m torch.distributed.launch \
#     --use_env \
#     --nproc_per_node $NPROC_PER_NODE \
#     --master-port $MASTER_PORT \
#     --nnodes 1 \
#     eval/eval_humanomniv2.py \
#     --model-path $MODEL_PATH \
#     --file-name ${FILE_PREFIX} \
#     --dataset ib

# echo ""
# echo "IntentBench 评估完成！结果: eval_results/${FILE_PREFIX}/ib_${FILE_PREFIX}.json"
# echo "=========================================="

# 评估 WorldSense
echo ""
echo ">>> 开始评估 WorldSense 数据集..."
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node $NPROC_PER_NODE \
    --master-port $MASTER_PORT \
    --nnodes 1 \
    eval/eval_humanomniv2.py \
    --model-path $MODEL_PATH \
    --file-name ${FILE_PREFIX} \
    --dataset world

echo ""
echo "WorldSense 评估完成！结果: eval_results/${FILE_PREFIX}/world_${FILE_PREFIX}.json"
echo "=========================================="

echo ""
echo "🎉 所有评估完成！"
echo ""
echo "结果文件:"
echo "  - Daily-Omni:  eval_results/${FILE_PREFIX}/daily_${FILE_PREFIX}.json"
echo "  - IntentBench: eval_results/${FILE_PREFIX}/ib_${FILE_PREFIX}.json"
echo "  - WorldSense:  eval_results/${FILE_PREFIX}/world_${FILE_PREFIX}.json"
echo "=========================================="
