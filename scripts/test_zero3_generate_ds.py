#!/usr/bin/env python
"""
测试 DeepSpeed Zero3 + Qwen2.5-Omni 生成（使用真正的 DeepSpeed 配置）
单卡测试多模态:
  python scripts/test_zero3_generate_ds.py --mode single_multimodal
  
多卡 Zero3 测试:
  CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus 2 scripts/test_zero3_generate_ds.py --mode zero3
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, GenerationConfig

def test_single_multimodal():
    """单卡多模态测试"""
    print('=' * 60)
    print('测试模式: 单卡多模态')
    print('=' * 60)
    
    model_path = '${PROJECT_ROOT}/models/HumanOmniV2'
    test_video = "${PROJECT_ROOT}/data/videos/MER24/sample_00000033.mp4"
    
    print('加载模型...')
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda:0',
        trust_remote_code=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 模拟训练脚本的设置
    print('启用 gradient_checkpointing...')
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    print(f'use_cache: {model.config.use_cache}')
    
    gen_config = GenerationConfig(
        max_new_tokens=100,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    
    # 测试1: 纯文本
    print('\n=== 测试1: 纯文本 ===')
    text1 = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n'
    inputs1 = processor(text=[text1], return_tensors='pt', padding=True)
    inputs1 = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs1.items()}
    
    with torch.no_grad():
        outputs1 = model.generate(**inputs1, generation_config=gen_config)
    result1 = processor.batch_decode(outputs1, skip_special_tokens=True)[0]
    print(f'结果: {result1[:200]}')
    check_garbled(result1)
    
    # 测试2: 带视频（使用 use_audio_in_video=True）
    print('\n=== 测试2: 带视频 (use_audio_in_video=True) ===')
    text2 = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>\nDescribe what you see.<|im_end|>\n<|im_start|>assistant\n'
    
    inputs2 = processor(
        text=[text2], 
        videos=[test_video],
        return_tensors='pt', 
        padding=True,
        use_audio_in_video=True,
    )
    inputs2 = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs2.items()}
    inputs2['use_audio_in_video'] = True
    
    print(f'inputs2 keys: {list(inputs2.keys())}')
    for k, v in inputs2.items():
        if hasattr(v, 'shape'):
            print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
    
    with torch.no_grad():
        outputs2 = model.generate(**inputs2, generation_config=gen_config)
    result2 = processor.batch_decode(outputs2, skip_special_tokens=True)[0]
    print(f'结果: {result2[:500]}')
    check_garbled(result2)
    
    print('\n' + '=' * 60)
    print('单卡多模态测试完成')
    print('=' * 60)


def test_zero3():
    """使用 DeepSpeed Zero3 测试"""
    import deepspeed
    
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
        "train_batch_size": 2,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
    }
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_main = local_rank == 0
    
    if is_main:
        print('=' * 60)
        print('测试 DeepSpeed Zero3 (仅纯文本)')
        print('=' * 60)
    
    deepspeed.init_distributed()
    
    model_path = '${PROJECT_ROOT}/models/HumanOmniV2'
    
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    model_engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    
    gen_config = GenerationConfig(
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    
    # 纯文本测试
    if is_main:
        print('\n=== 纯文本测试 ===')
    
    text1 = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n'
    inputs1 = processor(text=[text1], return_tensors='pt', padding=True)
    inputs1 = {k: v.to(model_engine.device) if hasattr(v, 'to') else v for k, v in inputs1.items()}
    
    with torch.no_grad():
        with deepspeed.zero.GatheredParameters(model_engine.parameters()):
            unwrapped = model_engine.module
            outputs1 = unwrapped.generate(**inputs1, generation_config=gen_config)
    
    if is_main:
        result1 = processor.batch_decode(outputs1, skip_special_tokens=True)[0]
        print(f'结果: {result1[:300]}')
        check_garbled(result1)
        print('\n' + '=' * 60)


def check_garbled(text):
    """检查是否有乱码"""
    if 'system' * 3 in text.lower():
        print('⚠️  检测到乱码!')
        return True
    else:
        print('✅ 输出正常')
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single_multimodal", "zero3"], default="single_multimodal")
    args = parser.parse_args()
    
    if args.mode == "single_multimodal":
        test_single_multimodal()
    elif args.mode == "zero3":
        test_zero3()
