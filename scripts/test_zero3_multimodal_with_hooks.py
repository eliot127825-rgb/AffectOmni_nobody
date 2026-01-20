#!/usr/bin/env python
"""
测试 DeepSpeed Zero3 + Qwen2.5-Omni 多模态生成（模拟 GRPO trainer 环境）

这个脚本模拟 GRPO trainer 的环境：
1. 使用 DeepSpeed Zero3 初始化
2. 创建 optimizer（模拟训练流程）
3. 测试 remove_hooks/add_hooks 后的生成

运行方式:
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
        print("  [remove_hooks] 模型没有 optimizer，跳过")
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
    print("  [remove_hooks] hooks 已移除")


def add_hooks(model):
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    from packaging import version
    
    if not hasattr(model, "optimizer"):
        print("  [add_hooks] 模型没有 optimizer，跳过")
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
    print("  [add_hooks] hooks 已恢复")


def check_garbled(text):
    """检查是否有乱码"""
    if 'system' * 3 in text.lower():
        print('⚠️  检测到乱码!')
        return True
    else:
        print('✅ 输出正常')
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
        "train_batch_size": world_size,  # 动态计算
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
    }
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_main = local_rank == 0
    
    if is_main:
        print('=' * 70)
        print('测试 DeepSpeed Zero3 + 多模态 (模拟 GRPO trainer 环境)')
        print('=' * 70)
    
    deepspeed.init_distributed()
    
    model_path = '${PROJECT_ROOT}/models/HumanOmniV2'
    test_video = "${PROJECT_ROOT}/data/videos/MER24/sample_00000033.mp4"
    
    if is_main:
        print('\n[1] 加载模型...')
    
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # 模拟 trainer 设置
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    
    if is_main:
        print('\n[2] 初始化 DeepSpeed...')
    
    # 创建 optimizer 来模拟训练环境
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, 
        config=ds_config,
        model_parameters=model.parameters(),  # 添加这个来创建 optimizer
    )
    
    if is_main:
        print(f'  DeepSpeed 版本: {deepspeed.__version__}')
        print(f'  模型是否有 optimizer: {hasattr(model_engine, "optimizer")}')
        if hasattr(model_engine, "optimizer"):
            print(f'  Optimizer 类型: {type(model_engine.optimizer)}')
    
    gen_config = GenerationConfig(
        max_new_tokens=100,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    
    # =====================================================
    # 测试 1: 纯文本，无 hooks 操作
    # =====================================================
    if is_main:
        print('\n' + '=' * 70)
        print('测试 1: 纯文本生成 (无 hooks 操作)')
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
        print(f'结果: {result1[:300]}')
        check_garbled(result1)
    
    # =====================================================
    # 测试 2: 纯文本，带 hooks 操作（模拟 TRL 的 unwrap_model_for_generation）
    # =====================================================
    if is_main:
        print('\n' + '=' * 70)
        print('测试 2: 纯文本生成 (带 remove_hooks/add_hooks)')
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
        print(f'结果: {result2[:300]}')
        check_garbled(result2)
    
    # =====================================================
    # 测试 3: 多模态（视频），无 hooks 操作
    # =====================================================
    if is_main:
        print('\n' + '=' * 70)
        print('测试 3: 多模态生成 (视频, 无 hooks 操作)')
        print('=' * 70)
    
    # 注意: 多模态输入需要同步到所有 rank
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
            print(f'结果: {result3[:500]}')
            check_garbled(result3)
            
    except Exception as e:
        if is_main:
            print(f'❌ 多模态测试失败: {e}')
            import traceback
            traceback.print_exc()
    
    # =====================================================
    # 测试 4: 多模态（视频），带 hooks 操作
    # =====================================================
    if is_main:
        print('\n' + '=' * 70)
        print('测试 4: 多模态生成 (视频, 带 remove_hooks/add_hooks)')
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
            print(f'结果: {result4[:500]}')
            check_garbled(result4)
            
    except Exception as e:
        if is_main:
            print(f'❌ 多模态测试 (带 hooks) 失败: {e}')
            import traceback
            traceback.print_exc()
    
    # =====================================================
    # 测试 5: 使用 qwen_omni_utils.process_mm_info（模拟 trainer 的数据处理）
    # =====================================================
    if is_main:
        print('\n' + '=' * 70)
        print('测试 5: 使用 qwen_omni_utils.process_mm_info 处理数据')
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
        
        # 应用 chat template
        text5 = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        # 使用预处理的 audio/video
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
            print(f'结果: {result5[:500]}')
            check_garbled(result5)
            
    except Exception as e:
        if is_main:
            print(f'❌ process_mm_info 测试失败: {e}')
            import traceback
            traceback.print_exc()
    
    if is_main:
        print('\n' + '=' * 70)
        print('所有测试完成!')
        print('=' * 70)


if __name__ == "__main__":
    main()

