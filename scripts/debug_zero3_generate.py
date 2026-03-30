#!/usr/bin/env python
"""
Diagnostic script: Test generation issues under DeepSpeed Zero3 environment
Usage:
  Single-GPU test (no DeepSpeed):
    python scripts/debug_zero3_generate.py --mode single
  
  Multi-GPU test (using DeepSpeed Zero3):
    torchrun --nproc_per_node 2 scripts/debug_zero3_generate.py --mode zero3
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor

def test_single_gpu():
    """Single-GPU test, without DeepSpeed"""
    print("=" * 60)
    print("Test mode: Single-GPU (no DeepSpeed)")
    print("=" * 60)
    
    model_path = "${PROJECT_ROOT}/models/HumanOmniV2"
    
    # Load model and processor
    print("Loading model...")
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Simple text test
    print("\n--- Simple text test ---")
    text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    print(f"Input keys: {list(inputs.keys())}")
    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
        )
    
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\nGeneration result:\n{result}")
    
    # Check for garbled output
    if "system" * 3 in result.lower():
        print("\n⚠️  Warning: Suspected garbled output detected!")
    else:
        print("\n✅ Output looks normal")
    
    return result

def test_zero3():
    """Test with DeepSpeed Zero3"""
    import deepspeed
    from accelerate import Accelerator
    from accelerate.utils import DeepSpeedPlugin
    
    print("=" * 60)
    print("Test mode: DeepSpeed Zero3")
    print("=" * 60)
    
    # DeepSpeed config
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
    
    # Load model
    print(f"[Rank {accelerator.process_index}] Loading model...")
    with accelerator.main_process_first():
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        processor = Qwen2_5OmniProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Prepare model
    model = accelerator.prepare_model(model)
    
    # Simple text test
    if accelerator.is_main_process:
        print("\n--- Simple text test (Zero3) ---")
    
    text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    inputs = {k: v.to(accelerator.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    
    if accelerator.is_main_process:
        print(f"Input keys: {list(inputs.keys())}")
        print(f"Input IDs shape: {inputs['input_ids'].shape}")
    
    # Test 1: Using GatheredParameters
    if accelerator.is_main_process:
        print("\n--- Test 1: Using GatheredParameters ---")
    
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
        print(f"\nGeneration result (GatheredParameters):\n{result[:500]}")
        
        if "system" * 3 in result.lower():
            print("\n⚠️  Warning: Suspected garbled output detected!")
        else:
            print("\n✅ Output looks normal")
    
    # Test 2: Without GatheredParameters (direct unwrap)
    if accelerator.is_main_process:
        print("\n--- Test 2: Without GatheredParameters ---")
    
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
                print(f"\nGeneration result (direct unwrap):\n{result2[:500]}")
        except Exception as e:
            if accelerator.is_main_process:
                print(f"\n❌ Error: {e}")

def test_gradient_checkpointing():
    """Test the impact of gradient checkpointing on generation"""
    print("=" * 60)
    print("Test mode: Gradient Checkpointing Impact")
    print("=" * 60)
    
    model_path = "${PROJECT_ROOT}/models/HumanOmniV2"
    
    # Load model
    print("Loading model...")
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
    
    # Test 1: Without gradient checkpointing
    print("\n--- Test 1: Without gradient checkpointing ---")
    print(f"model.config.use_cache: {model.config.use_cache}")
    
    with torch.no_grad():
        outputs1 = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
    result1 = processor.batch_decode(outputs1, skip_special_tokens=True)[0]
    print(f"Result: {result1[:300]}")
    
    # Test 2: With gradient checkpointing enabled
    print("\n--- Test 2: With gradient checkpointing enabled ---")
    model.gradient_checkpointing_enable()
    print(f"model.config.use_cache after gradient_checkpointing_enable: {model.config.use_cache}")
    
    with torch.no_grad():
        outputs2 = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
    result2 = processor.batch_decode(outputs2, skip_special_tokens=True)[0]
    print(f"Result: {result2[:300]}")
    
    # Test 3: Generate after disabling gradient checkpointing
    print("\n--- Test 3: Generate after disabling gradient checkpointing ---")
    model.gradient_checkpointing_disable()
    model.config.use_cache = True  # manually restore
    print(f"model.config.use_cache after disable: {model.config.use_cache}")
    
    with torch.no_grad():
        outputs3 = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
    result3 = processor.batch_decode(outputs3, skip_special_tokens=True)[0]
    print(f"Result: {result3[:300]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "zero3", "gc"], default="single",
                       help="Test mode: single=single GPU, zero3=DeepSpeed Zero3, gc=gradient checkpointing")
    args = parser.parse_args()
    
    if args.mode == "single":
        test_single_gpu()
    elif args.mode == "zero3":
        test_zero3()
    elif args.mode == "gc":
        test_gradient_checkpointing()

