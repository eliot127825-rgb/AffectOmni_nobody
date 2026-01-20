#!/bin/bash

# 使用GPU 4和5评估WorldSense数据集
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=4,5

python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node 2 \
    --master-port 29503 \
    --nnodes 1 \
    eval/eval_humanomniv2.py \
    --model-path ../outputs/stage4_debug_no_audio_v2/checkpoint-380 \
    --file-name people_focus_world \
    --dataset world
