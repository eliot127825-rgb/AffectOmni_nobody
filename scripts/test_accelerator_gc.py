#!/usr/bin/env python
"""
Test gradient checkpointing detection under Accelerator + DeepSpeed environment

Usage:
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
    
    # Create temporary DS config file
    ds_config_path = "/tmp/test_ds_config.json"
    with open(ds_config_path, "w") as f:
        json.dump(ds_config, f)
    
    # Set environment variables to make accelerate use DeepSpeed
    os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
    os.environ["ACCELERATE_DEEPSPEED_CONFIG_FILE"] = ds_config_path
    
    accelerator = Accelerator()
    is_main = accelerator.is_main_process
    
    if is_main:
        print('=' * 70)
        print('Test gradient checkpointing detection under Accelerator + DeepSpeed')
        print('=' * 70)
        print(f'DeepSpeed plugin: {accelerator.state.deepspeed_plugin}')
        if accelerator.state.deepspeed_plugin:
            print(f'  Zero stage: {accelerator.state.deepspeed_plugin.zero_stage}')
    
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
    
    # Use Accelerator prepare
    if is_main:
        print(f'\n[4] Preparing model with Accelerator...')
    
    # Create optimizer (needed for deepspeed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    model_wrapped, optimizer = accelerator.prepare(model, optimizer)
    
    # Check state after prepare
    if is_main:
        print(f'\n[5] Model state after Accelerator prepare:')
        print(f'  type(model_wrapped): {type(model_wrapped)}')
        print(f'  hasattr model_wrapped.is_gradient_checkpointing: {hasattr(model_wrapped, "is_gradient_checkpointing")}')
        if hasattr(model_wrapped, 'is_gradient_checkpointing'):
            print(f'  model_wrapped.is_gradient_checkpointing: {model_wrapped.is_gradient_checkpointing}')
        
        # Check unwrap
        unwrapped = accelerator.unwrap_model(model_wrapped)
        print(f'\n  Checking accelerator.unwrap_model:')
        print(f'  type(unwrapped): {type(unwrapped)}')
        print(f'  hasattr unwrapped.is_gradient_checkpointing: {hasattr(unwrapped, "is_gradient_checkpointing")}')
        if hasattr(unwrapped, 'is_gradient_checkpointing'):
            print(f'  unwrapped.is_gradient_checkpointing: {unwrapped.is_gradient_checkpointing}')
        print(f'  unwrapped.config.use_cache: {unwrapped.config.use_cache}')
    
    # Simulate unwrap_model_for_generation
    if is_main:
        print(f'\n[6] Testing unwrap_model_for_generation:')
    
    with torch.no_grad():
        with unwrap_model_for_generation(model_wrapped, accelerator) as unwrapped_model:
            if is_main:
                print(f'  type(unwrapped_model): {type(unwrapped_model)}')
                print(f'  hasattr unwrapped_model.is_gradient_checkpointing: {hasattr(unwrapped_model, "is_gradient_checkpointing")}')
                if hasattr(unwrapped_model, 'is_gradient_checkpointing'):
                    print(f'  unwrapped_model.is_gradient_checkpointing: {unwrapped_model.is_gradient_checkpointing}')
                else:
                    print(f'  ⚠️ unwrapped_model does not have is_gradient_checkpointing attribute!')
                
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
            inputs = {k: v.to(accelerator.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
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

