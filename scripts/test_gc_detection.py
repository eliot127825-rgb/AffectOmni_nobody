#!/usr/bin/env python
"""
测试 DeepSpeed 环境下 gradient checkpointing 检测问题

运行方式:
  CUDA_VISIBLE_DEVICES=2,3 deepspeed --num_gpus 2 scripts/test_gc_detection.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import deepspeed
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, GenerationConfig


def main():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_main = local_rank == 0
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    ds_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": True},
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
        },
        "train_batch_size": world_size,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
    }
    
    if is_main:
        print('=' * 70)
        print('测试 DeepSpeed 环境下 gradient checkpointing 检测')
        print('=' * 70)
    
    deepspeed.init_distributed()
    
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
    
    # 初始化 DeepSpeed
    if is_main:
        print(f'\n[4] 初始化 DeepSpeed Zero3...')
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, 
        config=ds_config,
        model_parameters=model.parameters(),
    )
    
    # 检查 DeepSpeed 包装后的状态
    if is_main:
        print(f'\n[5] DeepSpeed 包装后的模型状态:')
        print(f'  type(model_engine): {type(model_engine)}')
        print(f'  hasattr model_engine.is_gradient_checkpointing: {hasattr(model_engine, "is_gradient_checkpointing")}')
        if hasattr(model_engine, 'is_gradient_checkpointing'):
            print(f'  model_engine.is_gradient_checkpointing: {model_engine.is_gradient_checkpointing}')
        
        print(f'\n  检查 model_engine.module:')
        print(f'  type(model_engine.module): {type(model_engine.module)}')
        print(f'  hasattr module.is_gradient_checkpointing: {hasattr(model_engine.module, "is_gradient_checkpointing")}')
        if hasattr(model_engine.module, 'is_gradient_checkpointing'):
            print(f'  model_engine.module.is_gradient_checkpointing: {model_engine.module.is_gradient_checkpointing}')
        print(f'  model_engine.module.config.use_cache: {model_engine.module.config.use_cache}')
    
    # 模拟 unwrap_model_for_generation
    if is_main:
        print(f'\n[6] 测试 GatheredParameters + unwrap:')
    
    with torch.no_grad():
        with deepspeed.zero.GatheredParameters(model_engine.parameters()):
            unwrapped_model = model_engine.module
            
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
            inputs = {k: v.to(model_engine.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
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

