#!/usr/bin/env python
"""
简化版 Zero3 + 多模态测试（不使用 CPU offload，加速测试）

运行:
  CUDA_VISIBLE_DEVICES=2,3,4,5 deepspeed --num_gpus 4 scripts/test_zero3_simple.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import deepspeed
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor, GenerationConfig


def check_garbled(text, test_name):
    """检查是否有乱码"""
    if 'systemsystem' in text.lower() or 'contextsystem' in text.lower():
        print(f'⚠️  [{test_name}] 检测到乱码!')
        print(f'    输出样本: {text[:200]}')
        return True
    else:
        print(f'✅ [{test_name}] 输出正常')
        return False


def main():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_main = local_rank == 0
    
    # 不使用 CPU offload，速度更快
    ds_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
        },
        "train_batch_size": world_size,
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
    }
    
    if is_main:
        print('=' * 60)
        print('Zero3 + 多模态简化测试 (无 CPU offload)')
        print('=' * 60)
    
    deepspeed.init_distributed()
    
    model_path = '${PROJECT_ROOT}/models/HumanOmniV2'
    test_video = "${PROJECT_ROOT}/data/videos/MER24/sample_00000033.mp4"
    
    if is_main:
        print('\n加载模型...')
    
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
        print('初始化 DeepSpeed Zero3...')
    
    model_engine, _, _, _ = deepspeed.initialize(
        model=model, 
        config=ds_config,
        model_parameters=model.parameters(),
    )
    
    gen_config = GenerationConfig(
        max_new_tokens=50,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )
    
    # =====================================================
    # 测试 1: 纯文本
    # =====================================================
    if is_main:
        print('\n--- 测试 1: 纯文本生成 ---')
    
    text1 = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nSay hello in one sentence.<|im_end|>\n<|im_start|>assistant\n'
    inputs1 = processor(text=[text1], return_tensors='pt', padding=True)
    inputs1 = {k: v.to(model_engine.device) if hasattr(v, 'to') else v for k, v in inputs1.items()}
    
    with torch.no_grad():
        with deepspeed.zero.GatheredParameters(model_engine.parameters()):
            outputs1 = model_engine.module.generate(**inputs1, generation_config=gen_config)
    
    if is_main:
        result1 = processor.batch_decode(outputs1, skip_special_tokens=True)[0]
        print(f'  结果: {result1[:200]}')
        check_garbled(result1, "纯文本")
    
    torch.distributed.barrier()
    
    # =====================================================
    # 测试 2: 视频 (使用 processor 直接处理)
    # =====================================================
    if is_main:
        print('\n--- 测试 2: 视频生成 (processor直接处理) ---')
    
    try:
        text2 = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>\nWhat do you see? Reply briefly.<|im_end|>\n<|im_start|>assistant\n'
        
        inputs2 = processor(
            text=[text2], 
            videos=[test_video],
            return_tensors='pt', 
            padding=True,
            use_audio_in_video=True,
        )
        inputs2 = {k: v.to(model_engine.device) if hasattr(v, 'to') else v for k, v in inputs2.items()}
        inputs2['use_audio_in_video'] = True
        
        if is_main:
            print(f'  keys: {list(inputs2.keys())}')
        
        with torch.no_grad():
            with deepspeed.zero.GatheredParameters(model_engine.parameters()):
                outputs2 = model_engine.module.generate(**inputs2, generation_config=gen_config)
        
        if is_main:
            result2 = processor.batch_decode(outputs2, skip_special_tokens=True)[0]
            print(f'  结果: {result2[:300]}')
            check_garbled(result2, "视频")
            
    except Exception as e:
        if is_main:
            print(f'  ❌ 错误: {e}')
            import traceback
            traceback.print_exc()
    
    torch.distributed.barrier()
    
    # =====================================================
    # 测试 3: 使用 qwen_omni_utils (模拟 trainer 数据处理)
    # =====================================================
    if is_main:
        print('\n--- 测试 3: qwen_omni_utils 处理 ---')
    
    try:
        from qwen_omni_utils import process_mm_info
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "video", "video": test_video},
                {"type": "text", "text": "What is happening? Reply in one sentence."}
            ]},
        ]
        
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        
        if is_main:
            print(f'  audios: {len(audios) if audios else 0}, videos: {len(videos) if videos else 0}')
        
        text3 = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        inputs3 = processor(
            text=[text3], 
            videos=videos,
            audios=audios,
            return_tensors='pt', 
            padding=True,
        )
        inputs3 = {k: v.to(model_engine.device) if hasattr(v, 'to') else v for k, v in inputs3.items()}
        inputs3['use_audio_in_video'] = True
        
        if is_main:
            print(f'  keys: {list(inputs3.keys())}')
            for k, v in inputs3.items():
                if hasattr(v, 'shape'):
                    print(f'    {k}: {v.shape}')
        
        with torch.no_grad():
            with deepspeed.zero.GatheredParameters(model_engine.parameters()):
                outputs3 = model_engine.module.generate(**inputs3, generation_config=gen_config)
        
        if is_main:
            result3 = processor.batch_decode(outputs3, skip_special_tokens=True)[0]
            print(f'  结果: {result3[:300]}')
            check_garbled(result3, "qwen_omni_utils")
            
    except Exception as e:
        if is_main:
            print(f'  ❌ 错误: {e}')
            import traceback
            traceback.print_exc()
    
    if is_main:
        print('\n' + '=' * 60)
        print('测试完成!')
        print('=' * 60)


if __name__ == "__main__":
    main()

