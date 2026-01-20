#!/usr/bin/env python
"""
诊断脚本：测试 DeepSpeed Zero3 环境下 Qwen2.5-Omni 的生成问题
运行方式：
  单卡测试（无 DeepSpeed）:
    python scripts/debug_zero3_generate.py --mode single
  
  多卡测试（使用 DeepSpeed Zero3）:
    torchrun --nproc_per_node 2 scripts/debug_zero3_generate.py --mode zero3
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

def test_single_gpu():
    """单卡测试，不使用 DeepSpeed"""
    print("=" * 60)
    print("测试模式: 单卡（无 DeepSpeed）")
    print("=" * 60)
    
    model_path = "${PROJECT_ROOT}/models/HumanOmniV2"
    
    # 加载模型和处理器
    print("加载模型...")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 简单文本测试
    print("\n--- 简单文本测试 ---")
    text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    print(f"Input keys: {list(inputs.keys())}")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
        )
    
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\n生成结果:\n{result}")
    
    # 检查是否为乱码
    if "system" * 3 in result.lower():
        print("\n⚠️  警告: 检测到疑似乱码输出!")
    else:
        print("\n✅ 输出看起来正常")
    
    return result

def test_zero3():
    """使用 DeepSpeed Zero3 测试"""
    import deepspeed
    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin
    
    print("=" * 60)
    print("测试模式: DeepSpeed Zero3")
    print("=" * 60)
    
    # DeepSpeed 配置
    ds_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
    }
    
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=3,
        gradient_accumulation_steps=1,
    )
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
    
    model_path = "${PROJECT_ROOT}/models/HumanOmniV2"
    
    # 加载模型
    print(f"[Rank {accelerator.process_index}] 加载模型...")
    with accelerator.main_process_first():
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 准备模型
    model = accelerator.prepare_model(model)
    
    # 简单文本测试
    if accelerator.is_main_process:
        print("\n--- 简单文本测试 (Zero3) ---")
    
    text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(accelerator.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    if accelerator.is_main_process:
        print(f"Input keys: {list(inputs.keys())}")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
    
    # 测试1: 使用 GatheredParameters
    if accelerator.is_main_process:
        print("\n--- 测试1: 使用 GatheredParameters ---")
    
    with torch.no_grad():
        with deepspeed.zero.GatheredParameters(model.parameters()):
            unwrapped_model = accelerator.unwrap_model(model)
            outputs = unwrapped_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
            )
    
    if accelerator.is_main_process:
        result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"\n生成结果 (GatheredParameters):\n{result[:500]}")
        
        if "system" * 3 in result.lower():
            print("\n⚠️  警告: 检测到疑似乱码输出!")
        else:
            print("\n✅ 输出看起来正常")
    
    # 测试2: 不使用 GatheredParameters（直接 unwrap）
    if accelerator.is_main_process:
        print("\n--- 测试2: 不使用 GatheredParameters ---")
    
    with torch.no_grad():
        unwrapped_model = accelerator.unwrap_model(model)
        try:
            outputs2 = unwrapped_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
            )
            if accelerator.is_main_process:
                result2 = processor.batch_decode(outputs2, skip_special_tokens=True)[0]
                print(f"\n生成结果 (直接 unwrap):\n{result2[:500]}")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"\n❌ 错误: {e}")

def test_gradient_checkpointing():
    """测试 gradient checkpointing 对生成的影响"""
    print("=" * 60)
    print("测试模式: Gradient Checkpointing 影响")
    print("=" * 60)
    
    model_path = "${PROJECT_ROOT}/models/HumanOmniV2"
    
    # 加载模型
    print("加载模型...")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n"
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    # 测试1: 不启用 gradient checkpointing
    print("\n--- 测试1: 不启用 gradient checkpointing ---")
    print(f"model.config.use_cache: {model.config.use_cache}")
    
    with torch.no_grad():
        outputs1 = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
    result1 = processor.batch_decode(outputs1, skip_special_tokens=True)[0]
    print(f"结果: {result1[:300]}")
    
    # 测试2: 启用 gradient checkpointing
    print("\n--- 测试2: 启用 gradient checkpointing ---")
    model.gradient_checkpointing_enable()
    print(f"model.config.use_cache after gradient_checkpointing_enable: {model.config.use_cache}")
    
    with torch.no_grad():
        outputs2 = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
    result2 = processor.batch_decode(outputs2, skip_special_tokens=True)[0]
    print(f"结果: {result2[:300]}")
    
    # 测试3: 禁用 gradient checkpointing 后再生成
    print("\n--- 测试3: 禁用 gradient checkpointing 后再生成 ---")
    model.gradient_checkpointing_disable()
    model.config.use_cache = True  # 手动恢复
    print(f"model.config.use_cache after disable: {model.config.use_cache}")
    
    with torch.no_grad():
        outputs3 = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
    result3 = processor.batch_decode(outputs3, skip_special_tokens=True)[0]
    print(f"结果: {result3[:300]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "zero3", "gc"], default="single",
                       help="测试模式: single=单卡, zero3=DeepSpeed Zero3, gc=gradient checkpointing")
    args = parser.parse_args()
    
    if args.mode == "single":
        test_single_gpu()
    elif args.mode == "zero3":
        test_zero3()
    elif args.mode == "gc":
        test_gradient_checkpointing()

