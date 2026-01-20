#!/usr/bin/env python
"""
测试 Accelerator + DeepSpeed 环境下 gradient checkpointing 检测问题

运行方式:
  CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --config_file /dev/null scripts/test_accelerator_gc.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from accelerate import Accelerator
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, GenerationConfig
from trl.models import unwrap_model_for_generation
import json


def main():
    # DeepSpeed config
    ds_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": True},
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
        },
        "train_batch_size": 2,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
    }
    
    # 创建临时 DS config 文件
    ds_config_path = "/tmp/test_ds_config.json"
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f)
    
    # 设置环境变量使 accelerate 使用 DeepSpeed
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
    os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] = ds_config_path
    
    accelerator = Accelerator()
    is_main = accelerator.is_main_process
    
    if is_main:
        print('=' * 70)
        print('测试 Accelerator + DeepSpeed 环境下 gradient checkpointing 检测')
        print('=' * 70)
        print(f'DeepSpeed plugin: {accelerator.state.deepspeed_plugin}')
        if accelerator.state.deepspeed_plugin:
            print(f'  Zero stage: {accelerator.state.deepspeed_plugin.zero_stage}')
    
    model_path = '${PROJECT_ROOT}/models/HumanOmniV2'
    
    if is_main:
        print('\n[1] 加载模型...')
    
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 检查原始模型状态
    if is_main:
        print(f'\n[2] 原始模型状态:')
        print(f'  model.is_gradient_checkpointing: {model.is_gradient_checkpointing}')
        print(f'  model.config.use_cache: {model.config.use_cache}')
    
    # 启用 gradient checkpointing (模拟 trainer)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    if is_main:
        print(f'\n[3] 启用 gradient_checkpointing 后:')
        print(f'  model.is_gradient_checkpointing: {model.is_gradient_checkpointing}')
        print(f'  model.config.use_cache: {model.config.use_cache}')
    
    # 使用 Accelerator prepare
    if is_main:
        print(f'\n[4] 使用 Accelerator 准备模型...')
    
    # 创建 optimizer (需要for deepspeed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    model_wrapped, optimizer = accelerator.prepare(model, optimizer)
    
    # 检查 prepare 后的状态
    if is_main:
        print(f'\n[5] Accelerator prepare 后的模型状态:')
        print(f'  type(model_wrapped): {type(model_wrapped)}')
        print(f'  hasattr model_wrapped.is_gradient_checkpointing: {hasattr(model_wrapped, "is_gradient_checkpointing")}')
        if hasattr(model_wrapped, 'is_gradient_checkpointing'):
            print(f'  model_wrapped.is_gradient_checkpointing: {model_wrapped.is_gradient_checkpointing}')
        
        # 检查 unwrap
        unwrapped = accelerator.unwrap_model(model_wrapped)
        print(f'\n  检查 accelerator.unwrap_model:')
        print(f'  type(unwrapped): {type(unwrapped)}')
        print(f'  hasattr unwrapped.is_gradient_checkpointing: {hasattr(unwrapped, "is_gradient_checkpointing")}')
        if hasattr(unwrapped, 'is_gradient_checkpointing'):
            print(f'  unwrapped.is_gradient_checkpointing: {unwrapped.is_gradient_checkpointing}')
        print(f'  unwrapped.config.use_cache: {unwrapped.config.use_cache}')
    
    # 模拟 unwrap_model_for_generation
    if is_main:
        print(f'\n[6] 测试 unwrap_model_for_generation:')
    
    with torch.no_grad():
        with unwrap_model_for_generation(model_wrapped, accelerator) as unwrapped_model:
            if is_main:
                print(f'  type(unwrapped_model): {type(unwrapped_model)}')
                print(f'  hasattr unwrapped_model.is_gradient_checkpointing: {hasattr(unwrapped_model, "is_gradient_checkpointing")}')
                if hasattr(unwrapped_model, 'is_gradient_checkpointing'):
                    print(f'  unwrapped_model.is_gradient_checkpointing: {unwrapped_model.is_gradient_checkpointing}')
                else:
                    print(f'  ⚠️ unwrapped_model 没有 is_gradient_checkpointing 属性!')
                
                print(f'  unwrapped_model.config.use_cache: {unwrapped_model.config.use_cache}')
            
            # 测试禁用 gradient checkpointing
            gc_was_enabled = unwrapped_model.is_gradient_checkpointing if hasattr(unwrapped_model, 'is_gradient_checkpointing') else False
            
            if is_main:
                print(f'\n[7] gc_was_enabled = {gc_was_enabled}')
            
            if gc_was_enabled:
                if is_main:
                    print('  正在禁用 gradient checkpointing...')
                unwrapped_model.gradient_checkpointing_disable()
                unwrapped_model.config.use_cache = True
                if is_main:
                    print(f'  禁用后 is_gradient_checkpointing: {unwrapped_model.is_gradient_checkpointing}')
                    print(f'  禁用后 config.use_cache: {unwrapped_model.config.use_cache}')
            else:
                if is_main:
                    print('  ⚠️ gc_was_enabled 为 False，不会禁用 gradient checkpointing!')
            
            # 测试生成
            if is_main:
                print(f'\n[8] 测试纯文本生成...')
            
            gen_config = GenerationConfig(
                max_new_tokens=50,
                do_sample=True,
                temperature=1.0,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            
            text = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n'
            inputs = processor(text=[text], return_tensors='pt', padding=True)
            inputs = {k: v.to(accelerator.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            outputs = unwrapped_model.generate(**inputs, generation_config=gen_config)
            
            if is_main:
                result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                print(f'  结果: {result[:200]}')
                
                # 检查是否有乱码
                if 'system' * 3 in result.lower():
                    print('  ❌ 检测到乱码!')
                else:
                    print('  ✅ 输出正常')
            
            # 恢复状态
            if gc_was_enabled:
                unwrapped_model.gradient_checkpointing_enable()
                unwrapped_model.config.use_cache = False
    
    if is_main:
        print('\n' + '=' * 70)
        print('测试完成!')
        print('=' * 70)


if __name__ == "__main__":
    main()

