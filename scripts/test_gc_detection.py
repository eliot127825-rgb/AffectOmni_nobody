#!/usr/bin/env python
"""
Test gradient checkpointing detection under DeepSpeed environment

Usage:
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
        print('Test gradient checkpointing detection under DeepSpeed')
        print('=' * 70)
    
    deepspeed.init_distributed()
    
    model_path = '${PROJECT_ROOT}/models/HumanOmniV2'
    
    if is_main:
        print('\n[1] Loading model...')
    
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Check original model state
    if is_main:
        print(f'\n[2] Original model state:')
        print(f'  model.is_gradient_checkpointing: {model.is_gradient_checkpointing}')
        print(f'  model.config.use_cache: {model.config.use_cache}')
    
    # Enable gradient checkpointing (simulating trainer)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    if is_main:
        print(f'\n[3] After enabling gradient_checkpointing:')
        print(f'  model.is_gradient_checkpointing: {model.is_gradient_checkpointing}')
        print(f'  model.config.use_cache: {model.config.use_cache}')
    
    # Initialize DeepSpeed
    if is_main:
        print(f'\n[4] Initializing DeepSpeed Zero3...')
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, 
        config=ds_config,
        model_parameters=model.parameters(),
    )
    
    # Check state after DeepSpeed wrapping
    if is_main:
        print(f'\n[5] Model state after DeepSpeed wrapping:')
        print(f'  type(model_engine): {type(model_engine)}')
        print(f'  hasattr model_engine.is_gradient_checkpointing: {hasattr(model_engine, "is_gradient_checkpointing")}')
        if hasattr(model_engine, 'is_gradient_checkpointing'):
            print(f'  model_engine.is_gradient_checkpointing: {model_engine.is_gradient_checkpointing}')
        
        print(f'\n  Checking model_engine.module:')
        print(f'  type(model_engine.module): {type(model_engine.module)}')
        print(f'  hasattr module.is_gradient_checkpointing: {hasattr(model_engine.module, "is_gradient_checkpointing")}')
        if hasattr(model_engine.module, 'is_gradient_checkpointing'):
            print(f'  model_engine.module.is_gradient_checkpointing: {model_engine.module.is_gradient_checkpointing}')
        print(f'  model_engine.module.config.use_cache: {model_engine.module.config.use_cache}')
    
    # Simulate unwrap_model_for_generation
    if is_main:
        print(f'\n[6] Testing GatheredParameters + unwrap:')
    
    with torch.no_grad():
        with deepspeed.zero.GatheredParameters(model_engine.parameters()):
            unwrapped_model = model_engine.module
            
            if is_main:
                print(f'  type(unwrapped_model): {type(unwrapped_model)}')
                print(f'  hasattr unwrapped_model.is_gradient_checkpointing: {hasattr(unwrapped_model, "is_gradient_checkpointing")}')
                if hasattr(unwrapped_model, 'is_gradient_checkpointing'):
                    print(f'  unwrapped_model.is_gradient_checkpointing: {unwrapped_model.is_gradient_checkpointing}')
                else:
                    print(f'  WARNING: unwrapped_model does not have is_gradient_checkpointing attribute!')
                
                print(f'  unwrapped_model.config.use_cache: {unwrapped_model.config.use_cache}')
            
            # Test disabling gradient checkpointing
            gc_was_enabled = unwrapped_model.is_gradient_checkpointing if hasattr(unwrapped_model, 'is_gradient_checkpointing') else False
            
            if is_main:
                print(f'\n[7] gc_was_enabled = {gc_was_enabled}')
            
            if gc_was_enabled:
                if is_main:
                    print('  Disabling gradient checkpointing...')
                unwrapped_model.gradient_checkpointing_disable()
                unwrapped_model.config.use_cache = True
                if is_main:
                    print(f'  After disabling, is_gradient_checkpointing: {unwrapped_model.is_gradient_checkpointing}')
                    print(f'  After disabling, config.use_cache: {unwrapped_model.config.use_cache}')
            else:
                if is_main:
                    print('  ⚠️ gc_was_enabled is False, will not disable gradient checkpointing!')
            
            # Test generation
            if is_main:
                print(f'\n[8] Testing text-only generation...')
            
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
                print(f'  Result: {result[:200]}')
                
                # Check for garbled output
                if 'system' * 3 in result.lower():
                    print('  ❌ Garbled output detected!')
                else:
                    print('  ✅ Output is normal')
            
            # Restore state
            if gc_was_enabled:
                unwrapped_model.gradient_checkpointing_enable()
                unwrapped_model.config.use_cache = False
    
    if is_main:
        print('\n' + '=' * 70)
        print('Test completed!')
        print('=' * 70)


if __name__ == "__main__":
    main()

