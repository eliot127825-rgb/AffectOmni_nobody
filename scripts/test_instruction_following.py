#!/usr/bin/env python3
"""
Instruction following capability test script
Test whether the model truly follows instructions by using different prompts on the same video
"""

import os
import sys
import json
import yaml
import random
import torch
import re
from pathlib import Path
from datetime import datetime

# Add src path to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "src"))

from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from qwen_omni_utils import process_mm_info

# Set random seed
random.seed(42)
torch.manual_seed(42)


# ==================== Different test instructions (3 representative tests) ====================
TEST_PROMPTS = {
    "count_3_points": {
        "name": "Specified count: 3 key points",
        "instruction": "Please summarize this video in exactly 3 key points.",
        "expected": "Should output 3 key points"
    },
    
    "focus_people": {
        "name": "Specified focus: people",
        "instruction": "Describe the people in this video, focusing on their actions, expressions, and interactions.",
        "expected": "Should describe people in detail"
    },
    
    "one_sentence": {
        "name": "Specified length: one sentence",
        "instruction": "Describe this video in one single sentence.",
        "expected": "Should be only one sentence"
    },
}


def load_dataset(yaml_path):
    """Load training dataset"""
    print(f"Loading dataset: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    datasets = yaml_data.get('datasets', [])
    all_samples = []
    
    for dataset_config in datasets:
        json_path = dataset_config.get('json_path')
        data_root = dataset_config.get('data_root')
        
        print(f"  Loading: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Prepend data_root path
        if data_root:
            for sample in data:
                if 'path' in sample:
                    sample['path'] = os.path.join(data_root, sample['path'])
        
        all_samples.extend(data)
        print(f"  Loaded {len(data)} samples")
    
    print(f"\nTotal loaded: {len(all_samples)} samples\n")
    return all_samples


def create_messages_with_custom_prompt(sample, custom_instruction, system_prompt):
    """Create conversation messages with custom instruction"""
    
    # Construct messages
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": sample.get('data_type', 'video'),
                    sample.get('data_type', 'video'): sample['path'],
                    "max_frames": 32,
                    "max_pixels": 602112
                },
                {
                    "type": "text",
                    "text": custom_instruction
                }
            ]
        }
    ]
    
    return messages


def extract_tags(text):
    """Extract <context>, <think>, <answer> tag contents"""
    context_match = re.search(r'<context>(.*?)</context>', text, re.DOTALL)
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
    return {
        'context': context_match.group(1).strip() if context_match else None,
        'think': think_match.group(1).strip() if think_match else None,
        'answer': answer_match.group(1).strip() if answer_match else None
    }


def count_sentences(text):
    """Count number of sentences"""
    if not text:
        return 0
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def count_list_items(text):
    """Count number of list items"""
    if not text:
        return 0
    # Match numbered lists (1. 2. 3.) or bullet points (- * •)
    patterns = [
        r'^\d+\.',  # 1. 2. 3.
        r'^[-*•]',  # - * •
    ]
    
    lines = text.split('\n')
    count = 0
    for line in lines:
        line = line.strip()
        for pattern in patterns:
            if re.match(pattern, line):
                count += 1
                break
    return count


def analyze_output(generated_text, prompt_config):
    """Analyze whether output follows the instruction"""
    extracted = extract_tags(generated_text)
    analysis = {
        "has_context": extracted['context'] is not None,
        "has_think": extracted['think'] is not None,
        "has_answer": extracted['answer'] is not None,
    }
    
    answer_text = extracted.get('answer', '')
    
    # Analyze based on different prompt types
    prompt_key = prompt_config.get('key', '')
    
    if 'count_3' in prompt_key:
        list_count = count_list_items(answer_text)
        analysis['list_items'] = list_count
        analysis['follows_instruction'] = (list_count == 3)
        analysis['note'] = f"Required 3 key points, got {list_count}"
    
    elif 'count_5' in prompt_key:
        list_count = count_list_items(answer_text)
        analysis['list_items'] = list_count
        analysis['follows_instruction'] = (list_count == 5)
        analysis['note'] = f"Required 5 observations, got {list_count}"
    
    elif 'one_sentence' in prompt_key:
        sent_count = count_sentences(answer_text)
        analysis['sentence_count'] = sent_count
        analysis['follows_instruction'] = (sent_count == 1)
        analysis['note'] = f"Required 1 sentence, got {sent_count}"
    
    elif 'focus_people' in prompt_key:
        people_keywords = ['person', 'people', 'man', 'woman', 'he', 'she', 'they', 'facial', 'expression', 'gesture', 'interaction']
        keyword_count = sum(1 for kw in people_keywords if kw in answer_text.lower())
        analysis['people_keyword_count'] = keyword_count
        analysis['follows_instruction'] = (keyword_count >= 5)
        analysis['note'] = f"People-related keywords appeared {keyword_count} times"
    
    elif 'focus_environment' in prompt_key:
        env_keywords = ['background', 'setting', 'location', 'environment', 'room', 'outdoor', 'indoor', 'place']
        people_keywords = ['person', 'people', 'man', 'woman']
        env_count = sum(1 for kw in env_keywords if kw in answer_text.lower())
        people_count = sum(1 for kw in people_keywords if kw in answer_text.lower())
        analysis['environment_keyword_count'] = env_count
        analysis['people_keyword_count'] = people_count
        analysis['follows_instruction'] = (env_count > people_count)
        analysis['note'] = f"Environment words {env_count} vs people words {people_count}"
    
    elif 'timeline' in prompt_key:
        timeline_keywords = ['first', 'then', 'next', 'after', 'finally', 'initially', 'subsequently']
        keyword_count = sum(1 for kw in timeline_keywords if kw in answer_text.lower())
        analysis['timeline_keyword_count'] = keyword_count
        analysis['follows_instruction'] = (keyword_count >= 3)
        analysis['note'] = f"Temporal order words appeared {keyword_count} times"
    
    else:
        analysis['follows_instruction'] = None
        analysis['note'] = "General response, no specific requirement"
    
    return analysis, extracted


def generate_with_prompt(model, processor, sample, prompt_config, system_prompt):
    """Generate output with a specific prompt (processing flow consistent with test_base_model.py)"""
    
    # Create messages
    messages = create_messages_with_custom_prompt(
        sample, 
        prompt_config['instruction'],
        system_prompt
    )
    
    # Apply chat template
    texts = processor.apply_chat_template(
        [messages],
        tokenize=False,
        add_generation_prompt=True
    )
    text = texts[0]
    
    # Process multimodal inputs (consistent with test_base_model: use_audio_in_video=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=True,  # consistent with test_base_model
        max_length=32768
    )
    
    inputs = inputs.to(model.device)
    
    # Check sequence length (safety measure)
    seq_len = inputs['input_ids'].shape[1]
    if seq_len > 32768:
        raise AssertionError(f"Sequence too long: {seq_len} > 32768")
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,  # greedy decoding for stable results
            temperature=1.0,
            top_p=0.9
        )
    
    # Only take generated part
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    generated_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return generated_text


def main():
    print("=" * 80)
    print("Instruction Following Capability Test")
    print("=" * 80)
    print()
    
    # ==================== Configuration ====================
    MODEL_PATH = "${PROJECT_ROOT}/models/HumanOmniV2"
    BASE_MODEL_PATH = "${PROJECT_ROOT}/Qwen2.5-Omni-7B-Thinker"
    DATASET_PATH = "../configs/test_samples.yaml"
    
    # Use the same system prompt as test_base_model
    SYSTEM_PROMPT = """You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question. 

When analyzing videos, YOU MUST reference specific frame numbers and timestamps for key events and observations.
Format: "observation [Frame N: T.XXs]"

Examples of correct temporal references:
- The woman picks up the rose [Frame 3: 3.00s]
- She smiles at the man [Frame 5: 5.00s]
- The man receives the rose [Frame 12: 12.00s]

Pay special attention to the temporal progression of events. Always connect your visual observations to their corresponding frame numbers and timestamps.

In reasoning, It is encouraged to incorporate self-reflection and verification into your reasoning process. You are encouraged to review the image, video, audio, or other context information to ensure the answer accuracy.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.
"""
    
    # ==================== Load Model ====================
    print("Loading model...")
    print(f"  Model weights path: {MODEL_PATH}")
    
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    if hasattr(processor, 'image_processor'):
        processor.image_processor.max_pixels = 6422528
        processor.image_processor.min_pixels = 3136
    
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print("Model loaded successfully!")
    print()
    
    # ==================== Load Data ====================
    all_samples = load_dataset(DATASET_PATH)
    
    # Randomly select a video sample
    sample = random.choice(all_samples)
    
    print("Test sample:")
    print(f"  Video path: {sample.get('path', 'unknown')}")
    print(f"  Data type: {sample.get('data_type', 'unknown')}")
    print()
    
    # ==================== Test Different Instructions ====================
    print("=" * 80)
    print("Testing different instructions...")
    print("=" * 80)
    print()
    
    results = {}
    
    for prompt_key, prompt_config in TEST_PROMPTS.items():
        print(f"\n{'='*80}")
        print(f"Testing: {prompt_config['name']}")
        print(f"{'='*80}")
        print(f"Instruction: {prompt_config['instruction']}")
        print(f"Expected: {prompt_config['expected']}")
        print()
        
        # Generate output
        print("Generating...")
        generated_text = generate_with_prompt(
            model, processor, sample, prompt_config, SYSTEM_PROMPT
        )
        
        # Analyze output
        prompt_config['key'] = prompt_key
        analysis, extracted = analyze_output(generated_text, prompt_config)
        
        # Display results
        print("Generation complete")
        print()
        print("[Analysis Results]")
        if analysis.get('follows_instruction') is not None:
            status = "FOLLOWED" if analysis['follows_instruction'] else "NOT FOLLOWED"
            print(f"  Instruction following: {status}")
        print(f"  Note: {analysis['note']}")
        
        print()
        print("[Generated Answer]")
        answer = extracted.get('answer', 'None')
        print(f"{answer[:500]}{'...' if len(answer) > 500 else ''}")
        print()
        
        # Save results
        results[prompt_key] = {
            "prompt": prompt_config,
            "generated_text": generated_text,
            "extracted": extracted,
            "analysis": analysis
        }
    
    # ==================== Summary Comparison ====================
    print("\n" + "=" * 80)
    print("Instruction Following Summary")
    print("=" * 80)
    print()
    
    follow_count = 0
    total_testable = 0
    
    print(f"{'Instruction Type':<25} {'Status':<15} {'Details'}")
    print("-" * 80)
    
    for prompt_key, result in results.items():
        name = result['prompt']['name']
        follows = result['analysis'].get('follows_instruction')
        note = result['analysis'].get('note', '')
        
        if follows is not None:
            total_testable += 1
            if follows:
                follow_count += 1
                status = "FOLLOWED"
            else:
                status = "NOT FOLLOWED"
        else:
            status = "N/A"
        
        print(f"{name:<25} {status:<15} {note}")
    
    print("-" * 80)
    if total_testable > 0:
        follow_rate = (follow_count / total_testable) * 100
        print(f"\nInstruction following rate: {follow_count}/{total_testable} ({follow_rate:.1f}%)")
    
    # ==================== Save Results ====================
    log_dir = Path("../logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = log_dir / f"instruction_following_test_{timestamp}.json"
    
    test_result = {
        "timestamp": timestamp,
        "model_path": MODEL_PATH,
        "video_path": sample.get('path', 'unknown'),
        "results": results,
        "summary": {
            "total_prompts": len(TEST_PROMPTS),
            "testable_prompts": total_testable,
            "followed_prompts": follow_count,
            "follow_rate": f"{follow_rate:.1f}%" if total_testable > 0 else "N/A"
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)
    
    print(f"\nTest results saved to: {output_file}")
    
    # ==================== Conclusion ====================
    print("\n" + "=" * 80)
    print("Test Conclusion")
    print("=" * 80)
    print()
    
    if total_testable == 0:
        print("Cannot evaluate instruction following (all tests indeterminate)")
    elif follow_rate >= 80:
        print("Instruction following capability: **Strong**")
        print("   The model can understand and execute different types of instructions well")
    elif follow_rate >= 50:
        print("Instruction following capability: **Moderate**")
        print("   The model understands some instructions but execution is not precise enough")
    else:
        print("Instruction following capability: **Weak**")
        print("   The model struggles to follow specific instruction requirements")
        print("   May be overfitting to a specific Q&A format")
    
    print()


if __name__ == "__main__":
    main()
