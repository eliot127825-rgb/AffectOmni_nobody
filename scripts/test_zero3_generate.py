#!/usr/bin/env python
"""
Test DeepSpeed Zero3 + model generation
Usage: CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 scripts/test_zero3_generate.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import deepspeed
from accelerate import Accelerator
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, GenerationConfig
from trl.models import unwrap_model_for_generation

# Get current rank
local_rank = int(os.environ.get('LOCAL_RANK', 0))
is_main = local_rank == 0

if is_main:
    print('=' * 60)
    print('Test DeepSpeed Zero3 + model generation')
    print('=' * 60)

# Create accelerator (auto-detects DeepSpeed config)
accelerator = Accelerator()

model_path = '${PROJECT_ROOT}/models/HumanOmniV2'

if is_main:
    print(f'[Rank {local_rank}] Loading model...')
    print(f'DeepSpeed status: {accelerator.state.deepspeed_plugin}')

model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)

# Enable gradient checkpointing (simulating training script)
if is_main:
    print('Enabling gradient_checkpointing...')
model.gradient_checkpointing_enable()
model.config.use_cache = False

# Prepare model
model = accelerator.prepare_model(model)

# Test data
text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n'
inputs = processor(text=[text], return_tensors='pt', padding=True)
inputs = {k: v.to(accelerator.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

if is_main:
    print(f'Input shape: {inputs["input_ids"].shape}')
    print(f'\n=== Test 1: Using unwrap_model_for_generation (TRL method) ===')

# Test 1: Using TRL's unwrap_model_for_generation
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
    print(f'Result (TRL unwrap):\n{result1[:300]}')
    if 'system' * 3 in result1.lower():
        print('\n⚠️  Garbled output detected!')
    else:
        print('\n✅ Output is normal')

# Test 2: Manually using GatheredParameters
if is_main:
    print(f'\n=== Test 2: Manually using GatheredParameters ===')

with torch.no_grad():
    with deepspeed.zero.GatheredParameters(model.parameters()):
        unwrapped2 = accelerator.unwrap_model(model)
        outputs2 = unwrapped2.generate(**inputs, generation_config=gen_config)

if is_main:
    result2 = processor.batch_decode(outputs2, skip_special_tokens=True)[0]
    print(f'Result (manual GatheredParameters):\n{result2[:300]}')
    if 'system' * 3 in result2.lower():
        print('\n⚠️  Garbled output detected!')
    else:
        print('\n✅ Output is normal')

if is_main:
    print('\n' + '=' * 60)
    print('Test completed')
    print('=' * 60)

