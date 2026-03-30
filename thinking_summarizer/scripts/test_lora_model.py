#!/usr/bin/env python3
"""
Test the trained LoRA model.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Configuration
BASE_MODEL = os.environ.get("BASE_MODEL_PATH", "./Qwen2.5-3B-Instruct")
LORA_MODEL = os.environ.get("LORA_MODEL_PATH", "./thinking_summarizer/outputs/final_model")
TEST_DATA = os.environ.get("TEST_DATA_PATH", "./data/training_dataset/val.json")

print("=" * 80)
print("Testing LoRA Model - Thinking Summarizer")
print("=" * 80)
print(f"Base model: {BASE_MODEL}")
print(f"LoRA model: {LORA_MODEL}")
print("=" * 80)

import os

# Load model
print("\n1. Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, LORA_MODEL)
model.eval()
print("Model loaded successfully")

# Load test data
print("\n2. Loading test data...")
with open(TEST_DATA, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
print(f"Loaded {len(test_data)} validation samples")

# Test function
def generate_summary(thinking_text, instruction):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{instruction}\n\nThinking Process:\n{thinking_text}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )
    
    summary = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return summary

# Test 3 samples
print("\n3. Testing model output...")
print("=" * 80)

for i in range(min(3, len(test_data))):
    sample = test_data[i]
    thinking = sample['input']
    instruction = sample['instruction']
    expected = sample['output']
    
    print(f"\n[Test Sample {i+1}]")
    print(f"Video ID: {sample['metadata'].get('video_id', 'N/A')}")
    print(f"Source: {sample['metadata'].get('source', 'N/A')}")
    print(f"\nThinking (first 200 chars):")
    print(f"  {thinking[:200]}...")
    
    # Generate summary
    generated = generate_summary(thinking, instruction)
    
    print(f"\nGenerated Summary:")
    print("-" * 40)
    print(generated[:800])
    print("-" * 40)
    
    print(f"\nExpected Summary (first 300 chars):")
    print(f"  {expected[:300]}...")
    print("=" * 80)

print("\nTesting complete!")
