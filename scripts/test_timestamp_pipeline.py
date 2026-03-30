"""
Full timestamp post-processing pipeline test script
Integrates all modules: video frame sampling + model inference + event extraction + CLIP matching + timestamp insertion
"""

import sys
import os
import random
import yaml
import json
import torch
from datetime import datetime
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tools'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/src'))

from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration

# Import our modules
from tools.video_utils import sample_frames, get_video_info
from tools.clip_matcher import CLIPMatcher, match_with_monotonic_constraint
from extract_events import extract_events, events_to_queries
from insert_timestamps import insert_timestamps, verify_insertions


def load_test_samples(dataset_path: str):
    """Load test samples (consistent with test_base_model.py)"""
    print(f"Loading dataset: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
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
    """Create conversation messages (consistent with test_base_model.py)"""
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


def run_inference(sample, model, processor, system_prompt):
    """Run model inference (without timestamps, consistent with test_base_model.py)"""
    from qwen_omni_utils import process_mm_info
    
    # Construct messages (without timestamps)
    messages = create_messages(sample, system_prompt, timestamp_info=None)
    
    # Apply chat template
    texts = processor.apply_chat_template(
        [messages],
        tokenize=False,
        add_generation_prompt=True
    )
    text = texts[0]
    
    # Process multimodal inputs
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    
    # Process inputs
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32768
    ).to(model.device)
    
    # Generate (parameters consistent with test_base_model.py)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    # Decode
    generated_text = processor.batch_decode(
        outputs[:, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )[0]
    
    return generated_text, inputs


def parse_think_section(text: str) -> str:
    """Extract <think> section from generated text"""
    import re
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def main():
    print("=" * 80)
    print("Timestamp Post-Processing Pipeline Test")
    print("=" * 80)
    print()
    
    # ==================== Configuration ====================
    MODEL_PATH = "${PROJECT_ROOT}/models/HumanOmniV2"
    BASE_MODEL_PATH = "${PROJECT_ROOT}/Qwen2.5-Omni-7B-Thinker"
    DATASET_PATH = "../configs/test_samples.yaml"
    MAX_FRAMES = 16  # reduced frame count for faster processing
    CLIP_MODEL = "ViT-B-32"
    USE_MONOTONIC_CONSTRAINT = True  # maintain temporal constraint
    LAMBDA_SMOOTH = 0.01  # minimal smoothness constraint, mainly relying on CLIP similarity
    
    SYSTEM_PROMPT = """You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."""
    
    # ==================== Load Model ====================
    print("Loading model...")
    print(f"  Model path: {MODEL_PATH}")
    print(f"  Processor path: {BASE_MODEL_PATH}")
    
    # Load processor from base model
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    # Load model with custom architecture
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded successfully (device: {model.device})")
    print()
    
    # ==================== Load CLIP ====================
    print(f"Loading CLIP model: {CLIP_MODEL}...")
    # Use original CLIP (offline-friendly, no HuggingFace download needed)
    clip_matcher = CLIPMatcher(model_name=CLIP_MODEL, device="cuda", use_original_clip=True)
    print()
    
    # ==================== Load Test Samples ====================
    print(f"Loading test data: {DATASET_PATH}")
    all_samples = load_test_samples(DATASET_PATH)
    sample = random.choice(all_samples)
    
    print(f"Randomly selected sample:")
    print(f"  Video path: {sample['path']}")
    print(f"  Question: {sample['problem'][:100]}...")
    print()
    
    # ==================== Phase 1: Sample Video Frames ====================
    print("=" * 80)
    print("Phase 1: Sample Video Frames")
    print("=" * 80)
    
    video_path = sample['path']
    video_info = get_video_info(video_path)
    print(f"Video info:")
    print(f"  Total frames: {video_info['total_frames']}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Duration: {video_info['duration']:.2f}s")
    
    frames_pil, frame_ids, timestamps, fps = sample_frames(
        video_path, max_frames=MAX_FRAMES, strategy="uniform"
    )
    print(f"Sampling complete: {len(frames_pil)} frames")
    print(f"  Frame range: {frame_ids[0]} ~ {frame_ids[-1]}")
    print(f"  Time range: {timestamps[0]:.2f}s ~ {timestamps[-1]:.2f}s")
    print()
    
    # ==================== Phase 2: Model Inference ====================
    print("=" * 80)
    print("Phase 2: Model Inference (without timestamps)")
    print("=" * 80)
    
    generated_text, inputs = run_inference(sample, model, processor, SYSTEM_PROMPT)
    think_text = parse_think_section(generated_text)
    
    print("Inference complete")
    print(f"Generated length: {len(generated_text)} characters")
    print(f"\n[Original <think> content]")
    print("-" * 80)
    print(think_text[:500] + "..." if len(think_text) > 500 else think_text)
    print("-" * 80)
    print()
    
    # ==================== Phase 3: Extract Events ====================
    print("=" * 80)
    print("Phase 3: Extract Key Events")
    print("=" * 80)
    
    # Try LLM method
    print("Attempting LLM-based event extraction...")
    events = extract_events(
        think_text,
        method="llm",
        model=model,
        processor=processor,
        max_events=10
    )
    
    if not events:
        print("LLM extraction failed, using rule-based method")
        events = extract_events(think_text, method="rule", max_events=10)
    
    print(f"Extracted {len(events)} events:")
    for i, event in enumerate(events, 1):
        print(f"  {i}. anchor: {event.anchor[:60]}...")
        print(f"     query:  {event.query}")
    print()
    
    # ==================== Phase 4: CLIP Matching ====================
    print("=" * 80)
    print("Phase 4: CLIP Event-Frame Matching")
    print("=" * 80)
    
    queries = events_to_queries(events)
    
    if USE_MONOTONIC_CONSTRAINT:
        print("Using monotonic constraint DP...")
        similarity_matrix = clip_matcher.get_similarity_matrix(queries, frames_pil)
        best_frames = match_with_monotonic_constraint(
            similarity_matrix,
            lambda_smooth=LAMBDA_SMOOTH
        )
        frame_matches = {q: f for q, f in zip(queries, best_frames)}
    else:
        print("Using independent matching...")
        frame_matches = clip_matcher.match_events_to_frames(queries, frames_pil)
    
    print(f"Matching complete:")
    for i, (event, frame_id) in enumerate(zip(events, best_frames if USE_MONOTONIC_CONSTRAINT else [frame_matches[q] for q in queries]), 1):
        timestamp = timestamps[frame_id]
        print(f"  {i}. {event.query[:40]:<40} → Frame {frame_id:2d} ({timestamp:5.2f}s)")
    print()
    
    # ==================== Phase 5: Insert Timestamps ====================
    print("=" * 80)
    print("Phase 5: Insert Timestamps")
    print("=" * 80)
    
    think_with_timestamps = insert_timestamps(
        think_text,
        events,
        frame_matches,
        timestamps,
        format_style="frame_and_time"
    )
    
    # Verify insertion results
    verification = verify_insertions(think_text, think_with_timestamps, len(events))
    print(f"Insertion complete:")
    print(f"  Expected: {verification['expected_count']}")
    print(f"  Actual: {verification['inserted_count']}")
    print(f"  Insertion rate: {verification['insertion_rate']:.1%}")
    print()
    
    print(f"[<think> content with timestamps]")
    print("=" * 80)
    print(think_with_timestamps)
    print("=" * 80)
    print()
    
    # ==================== Save Results ====================
    log_dir = Path("../logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = log_dir / f"timestamp_pipeline_{timestamp_str}.json"
    
    # Convert numpy types to native Python types
    def convert_to_native(obj):
        """Recursively convert numpy types to native Python types"""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    result = {
        "video_path": video_path,
        "question": sample['problem'],
        "video_info": convert_to_native(video_info),
        "num_frames_sampled": len(frames_pil),
        "num_events_extracted": len(events),
        "events": [e.to_dict() for e in events],
        "frame_matches": {e.query: int(frame_matches[e.query]) for e in events},
        "original_think": think_text,
        "think_with_timestamps": think_with_timestamps,
        "verification": verification,
        "config": {
            "max_frames": MAX_FRAMES,
            "clip_model": CLIP_MODEL,
            "use_monotonic_constraint": USE_MONOTONIC_CONSTRAINT,
            "lambda_smooth": LAMBDA_SMOOTH
        }
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {result_file}")
    print()
    
    # ==================== Summary ====================
    print("=" * 80)
    print("Pipeline execution complete")
    print("=" * 80)
    print(f"  Sampled frames: {len(frames_pil)}")
    print(f"  Extracted events: {len(events)}")
    print(f"  Timestamp insertion rate: {verification['insertion_rate']:.1%}")
    print()
    print("Next steps:")
    print("  1. Check if timestamps are reasonable (match video content)")
    print("  2. Tune parameters (max_frames, lambda_smooth) to optimize results")
    print("  3. Test generalization on more samples")
    print()


if __name__ == "__main__":
    main()
