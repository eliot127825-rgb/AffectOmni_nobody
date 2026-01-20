#!/usr/bin/env python
"""
测试 DeepSpeed Zero3 + Qwen2.5-Omni 生成
运行: CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 scripts/test_zero3_generate.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import deepspeed
from accelerate import Accelerator
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, GenerationConfig
from trl.models import unwrap_model_for_generation

# 获取当前 rank
local_rank = int(os.environ.get('LOCAL_RANK', 0))
is_main = local_rank == 0

if is_main:
    print('=' * 60)
    print('测试 DeepSpeed Zero3 + Qwen2.5-Omni 生成')
    print('=' * 60)

# 创建 accelerator（会自动检测 DeepSpeed 配置）
accelerator = Accelerator()

model_path = '${PROJECT_ROOT}/models/HumanOmniV2'

if is_main:
    print(f'[Rank {local_rank}] 加载模型...')
    print(f'DeepSpeed 状态: {accelerator.state.deepspeed_plugin}')

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)

# 启用 gradient checkpointing（模拟训练脚本）
if is_main:
    print('启用 gradient_checkpointing...')
model.gradient_checkpointing_enable()
model.config.use_cache = False

# 准备模型
model = accelerator.prepare_model(model)

# 测试数据
text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n'
inputs = processor(text=[text], return_tensors='pt', padding=True)
inputs = {k: v.to(accelerator.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

if is_main:
    print(f'Input shape: {inputs["input_ids"].shape}')
    print(f'\n=== 测试1: 使用 unwrap_model_for_generation (TRL 方式) ===')

# 测试1: 使用 TRL 的 unwrap_model_for_generation
gen_config = GenerationConfig(
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
)

with torch.no_grad():
    with unwrap_model_for_generation(model, accelerator) as unwrapped:
        outputs1 = unwrapped.generate(**inputs, generation_config=gen_config)

if is_main:
    result1 = processor.batch_decode(outputs1, skip_special_tokens=True)[0]
    print(f'结果 (TRL unwrap):\n{result1[:300]}')
    if 'system' * 3 in result1.lower():
        print('\n⚠️  检测到乱码!')
    else:
        print('\n✅ 输出正常')

# 测试2: 手动使用 GatheredParameters
if is_main:
    print(f'\n=== 测试2: 手动使用 GatheredParameters ===')

with torch.no_grad():
    with deepspeed.zero.GatheredParameters(model.parameters()):
        unwrapped2 = accelerator.unwrap_model(model)
        outputs2 = unwrapped2.generate(**inputs, generation_config=gen_config)

if is_main:
    result2 = processor.batch_decode(outputs2, skip_special_tokens=True)[0]
    print(f'结果 (手动 GatheredParameters):\n{result2[:300]}')
    if 'system' * 3 in result2.lower():
        print('\n⚠️  检测到乱码!')
    else:
        print('\n✅ 输出正常')

if is_main:
    print('\n' + '=' * 60)
    print('测试完成')
    print('=' * 60)

