#!/usr/bin/env python3
"""
Simple model evaluation script
Randomly sample training data and test model output
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
from open_r1.vlm_modules.qwenomni_module import QwenOmniModule
from qwen_omni_utils import process_mm_info


# Set random seed
random.seed(42)
torch.manual_seed(42)


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


def format_question(sample):
    """Format the question (same as during training)"""
    if sample['problem_type'] in ['multiple choice', 'emer_ov_mc']:
        question = sample['problem'] + "\nOptions:\n"
        for option in sample.get('options', []):
            question += option + "\n"
    else:
        question = sample['problem']
    
    return question


def create_messages(sample, system_prompt, timestamp_info=None):
    """Create conversation messages (same format as during training)
    
    Args:
        sample: Sample data
        system_prompt: System prompt
        timestamp_info: Optional timestamp info string; if provided, appended to user message
    """
    question = format_question(sample)
    
    # TYPE_TEMPLATE
    TYPE_TEMPLATES = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "emer_ov": " Please provide the words to describe emotions within the  <answer> </answer> tags.",
        "emer_ov_mc": " Please provide only the single or multiple option letter (e.g., A for single option or A,E for multi option, etc.) within the <answer> </answer> tags.",
    }
    
    text_prompt = question + "\n" + TYPE_TEMPLATES.get(sample['problem_type'], "")
    
    # If timestamp info is provided, add it to the prompt
    if timestamp_info:
        text_prompt += "\n\n" + timestamp_info
    
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
                    "max_frames": 32,  # consistent with training
                    "max_pixels": 602112
                },
                {
                    "type": "text",
                    "text": text_prompt
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


def check_format(generated_text):
    """Check if output format is correct"""
    has_context = '<context>' in generated_text and '</context>' in generated_text
    has_think = '<think>' in generated_text and '</think>' in generated_text
    has_answer = '<answer>' in generated_text and '</answer>' in generated_text
    
    return {
        'has_context': has_context,
        'has_think': has_think,
        'has_answer': has_answer,
        'all_correct': has_context and has_think and has_answer
    }


def main():
    print("=" * 80)
    print("Base Model Test Script")
    print("=" * 80)
    print()
    
    # ==================== Configuration ====================
    MODEL_PATH = "${PROJECT_ROOT}/models/checkpoint"  # trained model
    BASE_MODEL_PATH = "${PROJECT_ROOT}/Qwen2.5-Omni-7B-Thinker"  # base model (for loading processor)
    DATASET_PATH = "../configs/test_samples.yaml"  # test dataset config
    
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
    print(f"  Processor path: {BASE_MODEL_PATH}")
    
    # Load processor from base model (only model weights are saved during training)
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    # Override global config (safety measure)
    if hasattr(processor, 'image_processor'):
        processor.image_processor.max_pixels = 6422528
        processor.image_processor.min_pixels = 3136
    
    # Load model weights from training output
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print("Model loaded successfully!")
    print(f"  Device: {model.device}")
    print()
    
    # ==================== Load Data ====================
    all_samples = load_dataset(DATASET_PATH)
    
    # ==================== Random Sample Selection ====================
    sample = random.choice(all_samples)
    
    print("Randomly selected sample:")
    print(f"  Problem type: {sample.get('problem_type', 'unknown')}")
    print(f"  Data type: {sample.get('data_type', 'unknown')}")
    print(f"  File path: {sample.get('path', 'unknown')}")
    print(f"  Question: {format_question(sample)[:200]}...")
    print()
    
    # ==================== Step 1: Preprocess video to get timestamp info ====================
    print("Step 1: Preprocess video to get timestamp info...")
    
    # First process once without timestamp messages to get actual frame count and interval
    temp_messages = create_messages(sample, SYSTEM_PROMPT, timestamp_info=None)
    temp_texts = processor.apply_chat_template(
        [temp_messages],
        tokenize=False,
        add_generation_prompt=True
    )
    temp_text = temp_texts[0]
    
    # Process multimodal inputs
    audios, images, videos = process_mm_info(temp_messages, use_audio_in_video=False)
    
    # Temporarily process to get timestamp info
    temp_inputs = processor(
        text=[temp_text],
        images=images,
        videos=videos,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32768
    )
    
    # Extract timestamp info
    timestamp_info_str = None
    if 'video_grid_thw' in temp_inputs and temp_inputs['video_grid_thw'] is not None:
        video_grid = temp_inputs['video_grid_thw']
        num_frames = video_grid[0][0].item()
        
        if 'video_second_per_grid' in temp_inputs and temp_inputs['video_second_per_grid'] is not None:
            second_per_grid = temp_inputs['video_second_per_grid']
            
            # Get time interval
            if second_per_grid.dim() == 0:
                interval = second_per_grid.item()
            elif second_per_grid.dim() == 1 and len(second_per_grid) == 1:
                interval = second_per_grid[0].item()
            else:
                interval = second_per_grid.flatten()[0].item()
            
            # Calculate timestamp per frame
            frame_timestamps = [i * interval for i in range(num_frames)]
            
            # Construct timestamp info string
            timestamp_info_str = "[Video Frame Information]\n"
            timestamp_info_str += f"This video has been sampled into {num_frames} frames at {interval:.2f}-second intervals.\n"
            timestamp_info_str += "Available frame timestamps:\n"
            timestamp_info_str += ", ".join([f"Frame {i}: {ts:.2f}s" for i, ts in enumerate(frame_timestamps)])
            timestamp_info_str += "\n\n"
            timestamp_info_str += "IMPORTANT: In your <think> section, you MUST reference specific frame numbers for each key event or observation.\n"
            timestamp_info_str += "Use the exact format: \"your observation [Frame N: T.XXs]\"\n"
            timestamp_info_str += "Example: The woman smiles [Frame 5: 5.00s], indicating happiness."
            
            print(f"  Extracted {num_frames} frames, interval {interval:.2f}s/frame")
    
    # ==================== Step 2: Reconstruct full input with timestamp info ====================
    print("Step 2: Constructing full input with timestamps...")
    messages = create_messages(sample, SYSTEM_PROMPT, timestamp_info=timestamp_info_str)
    
    # Apply chat template
    texts = processor.apply_chat_template(
        [messages],
        tokenize=False,
        add_generation_prompt=True
    )
    text = texts[0]
    
    # Reprocess multimodal inputs (using the same data)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    
    # Debug: display extracted multimodal data
    print(f"  Multimodal data extraction:")
    print(f"     - Audio: {len(audios) if audios else 0}")
    print(f"     - Images: {len(images) if images else 0}")
    print(f"     - Videos: {len(videos) if videos else 0}")
    
    # Read actual total video duration
    video_duration = None
    if videos and len(videos) > 0:
        video_path = sample['path']
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fps > 0:
                    video_duration = frame_count / fps
                cap.release()
        except Exception as e:
            print(f"Warning: unable to read video duration: {e}")
    
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=True,  # safety: truncation protection
        max_length=32768
    )
    
    inputs = inputs.to(model.device)
    
    # Safety: assertion + debug info
    seq_len = inputs['input_ids'].shape[1]
    print(f"Input preparation complete")
    print(f"  Input token count: {seq_len}")
    
    # Debug: print actual size of video pixel data
    if 'pixel_values_videos' in inputs and inputs['pixel_values_videos'] is not None:
        vid_shape = inputs['pixel_values_videos'].shape
        vid_size_gb = inputs['pixel_values_videos'].element_size() * inputs['pixel_values_videos'].nelement() / (1024**3)
        print(f"  Video pixel data shape: {vid_shape}")
        print(f"  Video pixel data size: {vid_size_gb:.2f} GB")
    
    # Print video frame count and timestamp info
    if 'video_grid_thw' in inputs and inputs['video_grid_thw'] is not None:
        video_grid = inputs['video_grid_thw']
        num_frames = video_grid[0][0].item()  # T dimension is frame count
        print(f"  Video analysis info:")
        if video_duration is not None:
            print(f"     - Total video duration: {video_duration:.2f}s")
        print(f"     - Sampled frames: {num_frames}")
        print(f"     - Grid dimensions (T*H*W): {video_grid[0][0].item()}x{video_grid[0][1].item()}x{video_grid[0][2].item()}")
        
        # Print timestamps per frame
        if 'video_second_per_grid' in inputs and inputs['video_second_per_grid'] is not None:
            second_per_grid = inputs['video_second_per_grid']
            
            # video_second_per_grid is seconds per temporal grid (interval), not a timestamp list
            if second_per_grid.dim() == 0:
                interval = second_per_grid.item()
            elif second_per_grid.dim() == 1 and len(second_per_grid) == 1:
                interval = second_per_grid[0].item()
            else:
                # If multiple values, take the first
                interval = second_per_grid.flatten()[0].item()
            
            # Calculate timestamps for each frame from frame count and interval
            frame_timestamps = [i * interval for i in range(num_frames)]
            
            print(f"     - Time interval: {interval:.2f}s/frame")
            print(f"     - Sampling coverage: {frame_timestamps[0]:.2f}s ~ {frame_timestamps[-1]:.2f}s")
            print(f"     - Sampling span: {frame_timestamps[-1] - frame_timestamps[0]:.2f}s")
            
            # Show all frame timestamps
            timestamps_str = [f'{t:.2f}s' for t in frame_timestamps]
            print(f"     - Frame timestamps ({num_frames} frames): {timestamps_str}")
    
    # Print audio info
    if 'input_features' in inputs and inputs['input_features'] is not None:
        audio_features = inputs['input_features']
        print(f"  Audio analysis info:")
        print(f"     - Audio feature shape: {audio_features.shape}")
        
        if 'audio_feature_lengths' in inputs and inputs['audio_feature_lengths'] is not None:
            audio_lengths = inputs['audio_feature_lengths']
            print(f"     - Audio feature lengths: {audio_lengths}")
            # Audio sample rate is typically 16kHz, each feature corresponds to a certain duration
            # Audio processing: ~50 feature frames per second
            if audio_lengths.numel() > 0:
                total_audio_frames = audio_lengths[0].item() if audio_lengths.dim() > 0 else audio_lengths.item()
                # Assuming ~50 audio feature frames per second (common for Whisper-like models)
                audio_duration_estimate = total_audio_frames / 50.0
                print(f"     - Estimated audio duration: {audio_duration_estimate:.2f}s (based on feature frames)")
    
    if seq_len > 32768:
        raise AssertionError(f"Sequence too long: {seq_len} > 32768")
    print()
    
    # ==================== Generate Output ====================
    print("Starting generation...")
    print("-" * 80)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,  # greedy decoding for stable results
            temperature=1.0,
            top_p=0.9
        )
    
    # Only take generated part (remove input)
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    generated_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    print("Generation complete!")
    print("-" * 80)
    print()
    
    # ==================== Analyze Output ====================
    print("Output analysis:")
    print("=" * 80)
    
    # 1. Format check
    format_check = check_format(generated_text)
    print("\n[Format Check]")
    print(f"  Has <context>: {'PASS' if format_check['has_context'] else 'FAIL'}")
    print(f"  Has <think>:   {'PASS' if format_check['has_think'] else 'FAIL'}")
    print(f"  Has <answer>:  {'PASS' if format_check['has_answer'] else 'FAIL'}")
    print(f"  Format valid:  {'PASS' if format_check['all_correct'] else 'FAIL'}")
    
    # 2. Extract content
    extracted = extract_tags(generated_text)
    
    print("\n[Generated Content]")
    if extracted['context']:
        print(f"\n<context>")
        print(f"{extracted['context'][:300]}...")
        print(f"</context>")
    
    if extracted['think']:
        print(f"\n<think>")
        print(f"{extracted['think'][:300]}...")
        print(f"</think>")
    
    if extracted['answer']:
        print(f"\n<answer>")
        print(f"{extracted['answer']}")
        print(f"</answer>")
    
    # 3. Compare with ground truth
    ground_truth_solution = sample.get('solution', '')
    ground_truth_answer = sample.get('answer', '')
    
    print("\n[Ground Truth Comparison]")
    print(f"  Ground Truth Answer: {ground_truth_answer}")
    if extracted['answer']:
        print(f"  Generated Answer:    {extracted['answer']}")
        
        # Simple answer matching
        if extracted['answer'].strip() == ground_truth_answer.strip():
            print(f"  Match result: EXACT MATCH")
        elif ground_truth_answer.strip() in extracted['answer'].strip():
            print(f"  Match result: PARTIAL MATCH")
        else:
            print(f"  Match result: NO MATCH")
    else:
        print(f"  Generated Answer:    FAILED TO EXTRACT")
    
    # 4. Full output
    print("\n[Full Generated Text]")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)
    
    # ==================== Save Results ====================
    # Create logs directory
    log_dir = Path("../logs")
    log_dir.mkdir(exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = log_dir / f"basemodel_test_result_{timestamp}.json"
    
    # Prepare data to save
    test_result = {
        "timestamp": timestamp,
        "model_path": MODEL_PATH,
        "sample_info": {
            "problem_type": sample.get('problem_type', 'unknown'),
            "data_type": sample.get('data_type', 'unknown'),
            "video_path": sample.get('path', 'unknown'),
            "question": format_question(sample),
            "ground_truth_answer": ground_truth_answer
        },
        "generated_output": {
            "full_text": generated_text,
            "context": extracted.get('context', ''),
            "think": extracted.get('think', ''),
            "answer": extracted.get('answer', '')
        },
        "evaluation": {
            "format_check": format_check,
            "answer_match": extracted['answer'].strip() == ground_truth_answer.strip() if extracted['answer'] else False
        }
    }
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)
    
    print(f"\nTest results saved to: {output_file}")
    
    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    
    print("\nNotes:")
    print("  - If format is correct, the model has learned the output structure")
    print("  - If answer matches, the model understands the task")
    print("  - Run the script multiple times to test different samples")
    print()


if __name__ == "__main__":
    main()
