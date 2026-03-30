#!/usr/bin/env python
"""
Test DeepSpeed Zero3 + multimodal generation (simulating GRPO trainer environment)

This script simulates the GRPO trainer environment:
1. Initialize with DeepSpeed Zero3
2. Create optimizer (simulating training flow)
3. Test generation after remove_hooks/add_hooks

Usage:
  CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus 2 scripts/test_zero3_multimodal_with_hooks.py
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import deepspeed
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, GenerationConfig


def remove_hooks(model):
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    import itertools
    
    def get_all_parameters(sub_module, recurse=False):
        return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())

    def iter_params(module, recurse=False):
        return [param for _, param in get_all_parameters(module, recurse)]
    
    if not hasattr(model, "optimizer"):
        print("  [remove_hooks] Model has no optimizer, skipping")
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []
    print("  [remove_hooks] Hooks removed")


def add_hooks(model):
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    from packaging import version
    
    if not hasattr(model, "optimizer"):
        print("  [add_hooks] Model has no optimizer, skipping")
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")
    
    if version.parse(deepspeed.__version__) >= version.parse("0.16.4"):
        optimizer_offload._register_deepspeed_module(optimizer_offload.module)
    else:
        optimizer_offload._register_hooks_recursively(optimizer_offload.module)
    print("  [add_hooks] Hooks restored")


def check_garbled(text):
    """Check for garbled output"""
    if 'system' * 3 in text.lower():
        print('⚠️  Garbled output detected!')
        return True
    else:
        print('✅ Output is normal')
        return False


def main():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    ds_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": True},
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True,
        },
        "train_batch_size": world_size,  # dynamically computed
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
    }
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_main = local_rank == 0
    
    if is_main:
        print('=' * 70)
        print('Test DeepSpeed Zero3 + multimodal (simulating GRPO trainer environment)')
        print('=' * 70)
    
    deepspeed.init_distributed()
    
    model_path = '${PROJECT_ROOT}/models/HumanOmniV2'
    test_video = "${PROJECT_ROOT}/data/videos/MER24/sample_00000033.mp4"
    
    if is_main:
        print('\n[1] Loading model...')
    
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Simulate trainer setup
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    if is_main:
        print('\n[2] Initializing DeepSpeed...')
    
    # Create optimizer to simulate training environment
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, 
        config=ds_config,
        model_parameters=model.parameters(),  # needed to create optimizer
    )
    
    if is_main:
        print(f'  DeepSpeed version: {deepspeed.__version__}')
        print(f'  Model has optimizer: {hasattr(model_engine, "optimizer")}')
        if hasattr(model_engine, "optimizer"):
            print(f'  Optimizer type: {type(model_engine.optimizer)}')
    
    gen_config = GenerationConfig(
        max_new_tokens=100,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    
    # =====================================================
    # Test 1: Text-only, no hooks operation
    # =====================================================
    if is_main:
        print('\n' + '=' * 70)
        print('Test 1: Text-only generation (no hooks operation)')
        print('=' * 70)
    
    text1 = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n'
    inputs1 = processor(text=[text1], return_tensors='pt', padding=True)
    inputs1 = {k: v.to(model_engine.device) if hasattr(v, 'to') else v for k, v in inputs1.items()}
    
    with torch.no_grad():
        with deepspeed.zero.GatheredParameters(model_engine.parameters()):
            unwrapped = model_engine.module
            outputs1 = unwrapped.generate(**inputs1, generation_config=gen_config)
    
    if is_main:
        result1 = processor.batch_decode(outputs1, skip_special_tokens=True)[0]
        print(f'Result: {result1[:300]}')
        check_garbled(result1)
    
    # =====================================================
    # Test 2: Text-only, with hooks operation (simulating TRL's unwrap_model_for_generation)
    # =====================================================
    if is_main:
        print('\n' + '=' * 70)
        print('Test 2: Text-only generation (with remove_hooks/add_hooks)')
        print('=' * 70)
    
    inputs2 = processor(text=[text1], return_tensors='pt', padding=True)
    inputs2 = {k: v.to(model_engine.device) if hasattr(v, 'to') else v for k, v in inputs2.items()}
    
    with torch.no_grad():
        with deepspeed.zero.GatheredParameters(model_engine.parameters()):
            remove_hooks(model_engine)
            unwrapped = model_engine.module
            outputs2 = unwrapped.generate(**inputs2, generation_config=gen_config)
            add_hooks(model_engine)
    
    if is_main:
        result2 = processor.batch_decode(outputs2, skip_special_tokens=True)[0]
        print(f'Result: {result2[:300]}')
        check_garbled(result2)
    
    # =====================================================
    # Test 3: Multimodal (video), no hooks operation
    # =====================================================
    if is_main:
        print('\n' + '=' * 70)
        print('Test 3: Multimodal generation (video, no hooks operation)')
        print('=' * 70)
    
    # Note: multimodal inputs need to be synced to all ranks
    text3 = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>\nDescribe what you see in the video.<|im_end|>\n<|im_start|>assistant\n'
    
    try:
        inputs3 = processor(
            text=[text3], 
            videos=[test_video],
            return_tensors='pt', 
            padding=True,
            use_audio_in_video=True,
        )
        inputs3 = {k: v.to(model_engine.device) if hasattr(v, 'to') else v for k, v in inputs3.items()}
        inputs3['use_audio_in_video'] = True
        
        if is_main:
            print(f'  inputs3 keys: {list(inputs3.keys())}')
            for k, v in inputs3.items():
                if hasattr(v, 'shape'):
                    print(f'    {k}: shape={v.shape}, dtype={v.dtype}')
        
        with torch.no_grad():
            with deepspeed.zero.GatheredParameters(model_engine.parameters()):
                unwrapped = model_engine.module
                outputs3 = unwrapped.generate(**inputs3, generation_config=gen_config)
        
        if is_main:
            result3 = processor.batch_decode(outputs3, skip_special_tokens=True)[0]
            print(f'Result: {result3[:500]}')
            check_garbled(result3)
            
    except Exception as e:
        if is_main:
            print(f'❌ Multimodal test failed: {e}')
            import traceback
            traceback.print_exc()
    
    # =====================================================
    # Test 4: Multimodal (video), with hooks operation
    # =====================================================
    if is_main:
        print('\n' + '=' * 70)
        print('Test 4: Multimodal generation (video, with remove_hooks/add_hooks)')
        print('=' * 70)
    
    try:
        inputs4 = processor(
            text=[text3], 
            videos=[test_video],
            return_tensors='pt', 
            padding=True,
            use_audio_in_video=True,
        )
        inputs4 = {k: v.to(model_engine.device) if hasattr(v, 'to') else v for k, v in inputs4.items()}
        inputs4['use_audio_in_video'] = True
        
        with torch.no_grad():
            with deepspeed.zero.GatheredParameters(model_engine.parameters()):
                remove_hooks(model_engine)
                unwrapped = model_engine.module
                outputs4 = unwrapped.generate(**inputs4, generation_config=gen_config)
                add_hooks(model_engine)
        
        if is_main:
            result4 = processor.batch_decode(outputs4, skip_special_tokens=True)[0]
            print(f'Result: {result4[:500]}')
            check_garbled(result4)
            
    except Exception as e:
        if is_main:
            print(f'❌ Multimodal test (with hooks) failed: {e}')
            import traceback
            traceback.print_exc()
    
    # =====================================================
    # Test 5: Using qwen_omni_utils.process_mm_info (simulating trainer data processing)
    # =====================================================
    if is_main:
        print('\n' + '=' * 70)
        print('Test 5: Processing data with qwen_omni_utils.process_mm_info')
        print('=' * 70)
    
    try:
        from qwen_omni_utils import process_mm_info
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "video", "video": test_video},
                {"type": "text", "text": "Describe what you see."}
            ]},
        ]
        
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        
        if is_main:
            print(f'  audios: {type(audios)}, count: {len(audios) if audios else 0}')
            print(f'  videos: {type(videos)}, count: {len(videos) if videos else 0}')
        
        # Apply chat template
        text5 = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # Use preprocessed audio/video
        inputs5 = processor(
            text=[text5], 
            videos=videos,
            audios=audios,
            return_tensors='pt', 
            padding=True,
        )
        inputs5 = {k: v.to(model_engine.device) if hasattr(v, 'to') else v for k, v in inputs5.items()}
        inputs5['use_audio_in_video'] = True
        
        if is_main:
            print(f'  inputs5 keys: {list(inputs5.keys())}')
            for k, v in inputs5.items():
                if hasattr(v, 'shape'):
                    print(f'    {k}: shape={v.shape}, dtype={v.dtype}')
        
        with torch.no_grad():
            with deepspeed.zero.GatheredParameters(model_engine.parameters()):
                remove_hooks(model_engine)
                unwrapped = model_engine.module
                outputs5 = unwrapped.generate(**inputs5, generation_config=gen_config)
                add_hooks(model_engine)
        
        if is_main:
            result5 = processor.batch_decode(outputs5, skip_special_tokens=True)[0]
            print(f'Result: {result5[:500]}')
            check_garbled(result5)
            
    except Exception as e:
        if is_main:
            print(f'❌ process_mm_info test failed: {e}')
            import traceback
            traceback.print_exc()
    
    if is_main:
        print('\n' + '=' * 70)
        print('All tests completed!')
        print('=' * 70)


if __name__ == "__main__":
    main()

