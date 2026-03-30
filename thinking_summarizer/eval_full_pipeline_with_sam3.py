#!/usr/bin/env python3
"""
Full Pipeline: Omni Thinker -> Summarizer -> SAM3
Uses SAM3 official API for video segmentation.
"""
import os
import sys
import json
import argparse
import re
from typing import List, Dict
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import numpy as np
from PIL import Image

# Add SAM3 path
sys.path.insert(0, os.environ.get('SAM3_CODE_PATH', './sam3/sam3_code'))

# Add eval path
sys.path.insert(0, os.environ.get('EVAL_PATH', './eval'))
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from qwen_omni_utils import process_mm_info

# Import SAM3
from sam3.model_builder import build_sam3_video_predictor

def simplify_prompt_for_sam3(prompt: str) -> str:
    """
    Simplify a complex prompt to a SAM3-friendly simple category name.
    
    Args:
        prompt: Original complex prompt, e.g. "President John F. Kennedy"
    
    Returns:
        Simplified prompt, e.g. "person"
    """
    prompt_lower = prompt.lower()
    
    # People-related
    if any(keyword in prompt_lower for keyword in [
        'president', 'speaker', 'presenter', 'doctor', 'dr.', 'man', 'woman',
        'person', 'people', 'kennedy', 'host', 'announcer', 'reporter'
    ]):
        return "person"
    
    # Screen/display devices
    if any(keyword in prompt_lower for keyword in [
        'television', 'tv', 'monitor', 'display', 'screen', 'broadcast'
    ]):
        return "screen"
    
    # Products/objects
    if any(keyword in prompt_lower for keyword in [
        'product', 'bottle', 'container', 'device', 'tool', 'equipment'
    ]):
        return "object"
    
    # Text/captions
    if any(keyword in prompt_lower for keyword in [
        'text', 'overlay', 'subtitle', 'caption', 'label', 'title'
    ]):
        return "text"
    
    # Furniture
    if any(keyword in prompt_lower for keyword in [
        'table', 'chair', 'desk', 'shelf', 'bookshelf', 'furniture'
    ]):
        return "furniture"
    
    # If no match, return original prompt with modifiers removed to get core noun
    # Remove adjectives and possessives, keep core nouns
    simplified = re.sub(r'\b(simulated|vintage|old|new|large|small|big|red|blue)\b', '', prompt_lower).strip()
    simplified = re.sub(r"'s\b", '', simplified).strip()
    simplified = re.sub(r'\s+', ' ', simplified)  # Collapse extra spaces
    
    return simplified if simplified else prompt

def extract_context(output_str):
    """Extract content within <context> tags"""
    pattern = r'<context>\s*(.*?)\s*</context>'
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_think(output_str):
    """Extract content within <think> tags"""
    pattern = r'<think>\s*(.*?)\s*</think>'
    match = re.search(pattern, output_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_answer(text):
    """Extract content within <answer> tags"""
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def parse_sam3_instructions(summary):
    """Extract SAM3 segmentation instructions from the summary"""
    pattern = r'## SAM3 Segmentation Instructions\s*\n\s*Please segment the following in the video:\s*(.+?)(?:\n\n|\Z)'
    match = re.search(pattern, summary, re.DOTALL | re.IGNORECASE)
    if match:
        objects_str = match.group(1).strip()
        # Split the object list
        objects = [obj.strip() for obj in objects_str.split(',')]
        return objects
    return []

class SummarizerModel:
    """Summarizer model wrapper"""
    def __init__(self, base_model_path, lora_path):
        print("Loading Summarizer model...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.model.eval()
        print("Summarizer loaded successfully")
    
    def summarize(self, thinking_text):
        """Summarize the thinking process"""
        instruction = """Analyze the following reasoning process and extract structured information.

Output format:
## Key Points Analysis
1. [First key point]
2. [Second key point]
3. [Third key point]

## Video Focus Objects
- People: [List important people with appearance and emotional features]
- Objects: [List important objects]
- Scenes: [List important scene elements]

## Emotional Indicators (if applicable)
- [List specific visual cues indicating emotions]

## SAM3 Segmentation Instructions
Please segment the following in the video: [object1], [object2], [object3]"""
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{instruction}\n\nThinking Process:\n{thinking_text}"}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                top_p=0.9,
                do_sample=True
            )
        
        summary = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return summary

class SAM3Segmenter:
    """SAM3 video segmentation wrapper"""
    def __init__(self, checkpoint_path=None, gpu_id=1):
        print(f"Loading SAM3 video predictor (GPU {gpu_id})...")
        self.gpu_id = gpu_id
        
        # Temporarily switch to specified GPU for model loading
        import torch
        with torch.cuda.device(gpu_id):
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"  Using local checkpoint: {checkpoint_path}")
                self.predictor = build_sam3_video_predictor(checkpoint_path=checkpoint_path)
            else:
                print("  Using HuggingFace auto-download")
                self.predictor = build_sam3_video_predictor()
            print("SAM3 loaded successfully")
    
    def segment_video(self, video_path, text_prompts, max_frames=30):
        """
        Perform SAM3 segmentation on a video.
        
        Args:
            video_path: Path to the video file
            text_prompts: List of text prompts, e.g. ["person", "car"]
            max_frames: Maximum number of frames to process (default 30)
        
        Returns:
            Segmentation results dict, with results per prompt
        """
        print(f"\nProcessing video: {video_path}")
        
        results_per_prompt = {}
        
        for prompt_idx, text_prompt in enumerate(text_prompts):
            print(f"\n  [{prompt_idx+1}/{len(text_prompts)}] Segmenting object: '{text_prompt}'")
            
            try:
                # 1. Start session
                response = self.predictor.handle_request(
                    request=dict(
                        type="start_session",
                        resource_path=video_path,
                    )
                )
                session_id = response["session_id"]
                
                # 2. Add text prompt at frame 0
                response = self.predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        text=text_prompt,
                    )
                )
                print(f"    [debug] add_prompt response: frame_index={response.get('frame_index')}, outputs keys={list(response.get('outputs', {}).keys())}")
                if 'outputs' in response and 'out_obj_ids' in response['outputs']:
                    print(f"    [debug] add_prompt returned out_obj_ids: {response['outputs']['out_obj_ids']}")
                
                # 3. Propagate segmentation to entire video (key step)
                all_outputs = {}
                frame_count = 0
                for result_dict in self.predictor.handle_stream_request(
                    request=dict(
                        type="propagate_in_video",
                        session_id=session_id,
                        propagation_direction="forward",  # forward propagation only
                        start_frame_idx=0,  # start from frame 0
                        max_frame_num_to_track=max_frames,  # limit number of frames
                    )
                ):
                    frame_idx = result_dict["frame_index"]
                    outputs = result_dict["outputs"]
                    all_outputs[frame_idx] = outputs
                    frame_count += 1
                    # Print debug info for first 3 frames
                    if frame_count <= 3:
                        print(f"    [debug] frame {frame_idx}: outputs keys={list(outputs.keys())}")
                
                # 4. Collect statistics
                all_object_ids = set()
                for outputs in all_outputs.values():
                    if "out_obj_ids" in outputs:
                        all_object_ids.update(outputs["out_obj_ids"])
                
                num_objects = len(all_object_ids)
                num_frames = len(all_outputs)
                
                # Debug: print outputs structure of first frame
                if 0 in all_outputs:
                    print(f"    [debug] frame 0 output keys: {list(all_outputs[0].keys())}")
                    if 'out_obj_ids' in all_outputs[0]:
                        print(f"    [debug] out_obj_ids: {all_outputs[0]['out_obj_ids']}")
                    if 'out_binary_masks' in all_outputs[0]:
                        print(f"    [debug] out_binary_masks shape: {all_outputs[0]['out_binary_masks'].shape if hasattr(all_outputs[0]['out_binary_masks'], 'shape') else 'N/A'}")
                
                results_per_prompt[text_prompt] = {
                    'session_id': session_id,
                    'num_objects': num_objects,
                    'num_frames': num_frames,
                    'object_ids': list(all_object_ids),
                    'outputs': all_outputs  # Full output including masks etc.
                }
                
                print(f"    Detected {num_objects} objects")
                print(f"    Processed {num_frames} frames")
                
                # Visualization: save all frames containing objects
                if num_objects > 0:
                    frames_with_objects = []
                    for frame_idx in all_outputs.keys():
                        if 'out_obj_ids' in all_outputs[frame_idx] and len(all_outputs[frame_idx]['out_obj_ids']) > 0:
                            frames_with_objects.append(frame_idx)
                    
                    if frames_with_objects:
                        print(f"    [visualization] Saving segmentation masks for {len(frames_with_objects)} frames")
                        self._save_all_frames_visualization(video_path, all_outputs, frames_with_objects, text_prompt)
                
            except Exception as e:
                print(f"    Segmentation failed: {e}")
                results_per_prompt[text_prompt] = {'error': str(e)}
        
        return results_per_prompt
    
    def _save_all_frames_visualization(self, video_path, all_outputs, frame_indices, prompt):
        """Save visualization for all frames with objects, using ffmpeg for correct brightness"""
        import subprocess
        import tempfile
        
        vis_dir = './outputs/sam3_visualizations'
        os.makedirs(vis_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Use temp directory to store ffmpeg-extracted frames
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Extract all frames with objects using ffmpeg
            for frame_idx in frame_indices:
                # Extract specified frame using ffmpeg
                output_frame = os.path.join(tmp_dir, f"frame_{frame_idx}.jpg")
                cmd = [
                    'ffmpeg', '-i', video_path,
                    '-vf', f'select=eq(n\\,{frame_idx})',
                    '-vframes', '1',
                    output_frame,
                    '-y', '-loglevel', 'quiet'
                ]
                subprocess.run(cmd, check=True)
                
                # Read the ffmpeg-extracted frame
                frame = cv2.imread(output_frame)
                if frame is None:
                    continue
                
                # Get masks for this frame
                outputs = all_outputs[frame_idx]
                if 'out_binary_masks' not in outputs or len(outputs['out_binary_masks']) == 0:
                    continue
                
                masks = outputs['out_binary_masks']
                
                # Save mask overlay for each object
                for obj_idx, mask in enumerate(masks):
                    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
                    
                    # Create overlay: original frame + mask highlight
                    overlay = frame.copy()
                    color = np.array([0, 255, 0], dtype=np.float32)
                    mask_bool = mask_np > 0
                    overlay[mask_bool] = (overlay[mask_bool] * 0.5 + color * 0.5).astype(np.uint8)
                    
                    # Save
                    output_path = os.path.join(vis_dir, f"{video_name}_frame{frame_idx}_{prompt}_obj{obj_idx}.jpg")
                    cv2.imwrite(output_path, overlay)
                
                print(f"      Frame {frame_idx}: {len(masks)} objects")
    
    def _save_visualization(self, video_path, outputs, prompt, frame_idx):
        """Save segmentation mask visualization for a specified frame"""
        try:
            # Read specified frame
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"    Cannot read frame {frame_idx}")
                return
            
            # Use original frame directly without TV range conversion
            
            # Get masks
            if 'out_binary_masks' not in outputs or len(outputs['out_binary_masks']) == 0:
                return
            
            masks = outputs['out_binary_masks']  # shape: (N, H, W) N objects
            
            # Create output directory
            vis_dir = './outputs/sam3_visualizations'
            os.makedirs(vis_dir, exist_ok=True)
            
            # Save mask for each object
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            for obj_idx, mask in enumerate(masks):
                # Convert mask to color overlay
                mask_np = mask.cpu().numpy() 
                
                # Create visualization: aggressively brighten background + mask highlight
                # Brighten original frame 4x to handle dark scenes
                overlay = np.clip(frame.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
                color = np.array([0, 255, 0], dtype=np.float32)  # green
                
                # Overlay highlight color on mask region
                mask_bool = mask_np > 0
                overlay[mask_bool] = (overlay[mask_bool] * 0.4 + color * 0.6).astype(np.uint8)
                
                # Save
                output_path = os.path.join(vis_dir, f"{video_name}_{prompt}_obj{obj_idx}.jpg")
                cv2.imwrite(output_path, overlay)
                print(f"    Saved visualization: {output_path}")
        except Exception as e:
            print(f"    Visualization save failed: {e}")

def load_test_samples(data_path, num_samples=3):
    """Load test samples"""
    with open(data_path) as f:
        data = json.load(f)
    
    # Get data directory (for constructing full video paths)
    data_dir = os.path.dirname(data_path)
    
    samples = []
    for item in data[:num_samples]:
        # Support two formats: video_path (full path) or video (filename)
        video_path = item.get('video_path', '')
        if not video_path:
            video_name = item.get('video', '')
            if video_name:
                # Try to find video in data directory
                video_path = os.path.join(data_dir, video_name)
        
        # Support two formats: question or problem + options
        question = item.get('question', '')
        if not question:
            problem = item.get('problem', '')
            options = item.get('options', [])
            if problem and options:
                options_text = '\n'.join(options)
                question = f"{problem} Options:\n{options_text}\nPlease provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
        
        samples.append({
            'qid': item.get('qid', ''),
            'video_path': video_path,
            'question': question,
            'problem_type': item.get('problem_type', ''),
            'solution': item.get('solution', item.get('answer', ''))
        })
    
    return samples

def run_full_pipeline(args):
    """Run the full pipeline"""
    
    # 1. Load Omni Thinker model
    print("="*80)
    print("Step 1: Loading Omni Thinker model")
    print("="*80)
    processor = Qwen2_5OmniProcessor.from_pretrained(args.humanomniv2_model)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.humanomniv2_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("Omni Thinker model loaded\n")
    
    # 2. Load Summarizer model
    print("="*80)
    print("Step 2: Loading Thinking Summarizer model")
    print("="*80)
    summarizer = SummarizerModel(args.base_model, args.lora_model)
    print()
    
    # 3. SAM3 will be loaded after main model inference (to avoid VRAM conflicts)
    print("="*80)
    print("Step 3: SAM3 segmentation model (will load after inference)")
    print("="*80)
    sam3 = None  # Lazy loading
    sam3_checkpoint_path = None
    if args.enable_sam3:
        sam3_checkpoint_path = os.path.join(args.sam3_model, "sam3.pt")
        if not os.path.exists(sam3_checkpoint_path):
            print(f"Local checkpoint does not exist: {sam3_checkpoint_path}")
            sam3_checkpoint_path = None
        print("SAM3 will load after main model releases VRAM")
    else:
        print("Skipping SAM3")
    print()
    
    # 4. Load test samples
    print("="*80)
    print(f"Step 4: Loading test samples (first {args.num_samples})")
    print("="*80)
    test_samples = load_test_samples(args.data_path, args.num_samples)
    print(f"Loaded {len(test_samples)} test samples\n")
    
    # 5. Run full pipeline
    print("="*80)
    print("Step 5: Running full pipeline")
    print("="*80)
    
    results = []
    
    for idx, sample in enumerate(test_samples):
        print(f"\n{'='*80}")
        print(f"Sample {idx+1}/{len(test_samples)}")
        print(f"{'='*80}")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Video: {sample['video_path']}")
        
        # 5.1 Build input (limit video frames to prevent token overflow)
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": sample['video_path'], "nframes": args.video_max_frames},
                    {"type": "text", "text": sample['question']}
                ]
            }
        ]
        
        print(f"  Video max frames: {args.video_max_frames}")
        audios, images, videos = process_mm_info(message, use_audio_in_video=False)
        text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        model_inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        
        # 5.2 Generate thinking
        print("\n[1/3] Generating thinking...")
        with torch.inference_mode():
            text_ids = model.generate(
                **model_inputs.to(model.device).to(model.dtype),
                use_audio_in_video=False,
                max_new_tokens=2048
            )
        
        full_output = processor.decode(text_ids[0][model_inputs.input_ids.size(1):], skip_special_tokens=True)
        context = extract_context(full_output)
        thinking = extract_think(full_output)
        answer = extract_answer(full_output)
        
        print(f"Context generated (length: {len(context)} chars)")
        print(f"Thinking generated (length: {len(thinking)} chars)")
        print(f"Answer: {answer}")
        
        # 5.3 Summarize thinking
        print("\n[2/3] Summarizing thinking...")
        summary = summarizer.summarize(thinking)
        print(f"Summary generated (length: {len(summary)} chars)")
        
        # Extract SAM3 instructions
        sam3_objects = parse_sam3_instructions(summary)
        print(f"Extracted SAM3 segmentation objects: {sam3_objects}")
        
        # Simplify prompts to SAM3-friendly categories
        sam3_objects_simplified = []
        if sam3_objects:
            print("\nSimplifying prompts:")
            for obj in sam3_objects:
                simplified = simplify_prompt_for_sam3(obj)
                sam3_objects_simplified.append(simplified)
                print(f"  '{obj}' -> '{simplified}'")
            # Deduplicate
            sam3_objects_simplified = list(dict.fromkeys(sam3_objects_simplified))
            print(f"Simplified objects: {sam3_objects_simplified}")
            
            # Process only the first object to prevent OOM
            if len(sam3_objects_simplified) > 1:
                print(f"To prevent OOM, only processing the first object: '{sam3_objects_simplified[0]}'")
                sam3_objects_simplified = [sam3_objects_simplified[0]]
        
        # 5.4 Store SAM3 segmentation tasks (to execute later)
        sam3_results = None
        if sam3_objects_simplified and args.enable_sam3:
            print(f"\n[3/3] SAM3 segmentation task recorded (will execute after main model release)")
        else:
            print(f"\n[3/3] Skipping SAM3 segmentation (enable_sam3={args.enable_sam3}, has_objects={bool(sam3_objects)})")
        
        # Save results
        result = {
            'sample_id': idx + 1,
            'qid': sample['qid'],
            'question': sample['question'],
            'video_path': sample['video_path'],
            'ground_truth': sample['solution'],
            'full_output': full_output,  # Full model output (with <context>, <think> and <answer> tags)
            'context': context,
            'thinking': thinking,
            'answer': answer,
            'summary': summary,
            'sam3_objects': sam3_objects,
            'sam3_results_summary': {
                prompt: {
                    'num_objects': data.get('num_objects', 0),
                    'num_frames': data.get('num_frames', 0),
                    'object_ids': data.get('object_ids', []),
                    'error': data.get('error')
                }
                for prompt, data in (sam3_results or {}).items()
            } if sam3_results else None
        }
        results.append(result)
        
        print(f"\nSample {idx+1} processing complete")
    
    # 6. Release main model VRAM, execute SAM3 segmentation
    if args.enable_sam3 and sam3_checkpoint_path:
        print("\n" + "="*80)
        print("Step 6: Releasing main model VRAM, loading SAM3 for segmentation")
        print("="*80)
        
        # Release main model and Summarizer VRAM
        print("Releasing main model VRAM...")
        del model
        del processor
        del summarizer
        import gc
        gc.collect()
        
        # Thoroughly clean CUDA state
        print("Cleaning up CUDA state...")
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Reset default device to SAM3 GPU
        torch.cuda.set_device(args.sam3_gpu)
        print(f"Main model VRAM released, CUDA default device set to GPU {args.sam3_gpu}")
        
        # Load SAM3
        print(f"\nLoading SAM3 model (GPU {args.sam3_gpu})...")
        sam3 = SAM3Segmenter(checkpoint_path=sam3_checkpoint_path, gpu_id=args.sam3_gpu)
        
        # Execute SAM3 segmentation for all samples
        print("\nExecuting SAM3 segmentation tasks...")
        for idx, result in enumerate(results):
            sam3_objects = result.get('sam3_objects', [])
            if not sam3_objects:
                continue
            
            # Simplify prompts
            sam3_objects_simplified = []
            for obj in sam3_objects:
                simplified = simplify_prompt_for_sam3(obj)
                sam3_objects_simplified.append(simplified)
            sam3_objects_simplified = list(dict.fromkeys(sam3_objects_simplified))
            
            # Only process the first object
            if len(sam3_objects_simplified) > 1:
                sam3_objects_simplified = [sam3_objects_simplified[0]]
            
            print(f"\nProcessing sample {idx+1}: {sam3_objects_simplified}")
            try:
                sam3_results = sam3.segment_video(
                    result['video_path'],
                    sam3_objects_simplified,
                    max_frames=args.max_frames
                )
                    # Update results
                results[idx]['sam3_results_summary'] = {
                    prompt: {
                        'num_objects': data.get('num_objects', 0),
                        'num_frames': data.get('num_frames', 0),
                        'object_ids': data.get('object_ids', []),
                        'error': data.get('error')
                    }
                    for prompt, data in (sam3_results or {}).items()
                }
            except Exception as e:
                print(f"SAM3 segmentation failed: {e}")
                results[idx]['sam3_results_summary'] = {'error': str(e)}
        
        print("\nSAM3 segmentation complete")
    
    # 7. Save results (remove non-JSON-serializable output fields)
    output_file = os.path.join(args.output_dir, "full_pipeline_results.json")
    
    # Clean results, remove large data like masks from outputs
    cleaned_results = []
    for result in results:
        cleaned_result = result.copy()
        if 'sam3_results_summary' in cleaned_result and cleaned_result['sam3_results_summary']:
            cleaned_summary = {}
            for prompt, data in cleaned_result['sam3_results_summary'].items():
                cleaned_data = {
                    'num_objects': int(data.get('num_objects', 0)),
                    'num_frames': int(data.get('num_frames', 0)),
                    'object_ids': [int(x) for x in data.get('object_ids', [])],
                    'error': data.get('error')
                }
                cleaned_summary[prompt] = cleaned_data
            cleaned_result['sam3_results_summary'] = cleaned_summary
        cleaned_results.append(cleaned_result)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("Pipeline complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Processed {len(results)} samples")

def main():
    parser = argparse.ArgumentParser(description="Full Pipeline: Omni Thinker + Summarizer + SAM3")
    
    # Omni Thinker model
    parser.add_argument('--humanomniv2-model', type=str, 
                        default=os.environ.get('MODEL_CHECKPOINT_DIR', './checkpoints'))
    
    # Summarizer model
    parser.add_argument('--base-model', type=str,
                        default=os.environ.get('SUMMARIZER_BASE_MODEL', './Qwen2.5-3B-Instruct'))
    parser.add_argument('--lora-model', type=str,
                        default=os.environ.get('SUMMARIZER_LORA_MODEL', './thinking_summarizer/outputs/final_model'))
    
    # SAM3 model
    parser.add_argument('--sam3-model', type=str,
                        default=os.environ.get('SAM3_CHECKPOINT_DIR', './sam3/checkpoints'))
    parser.add_argument('--enable-sam3', action='store_true', default=False,
                        help='Enable SAM3 segmentation')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Max frames for SAM3 processing (default: all frames)')
    parser.add_argument('--sam3-gpu', type=int, default=1,
                        help='GPU ID for SAM3 (default: 1)')
    
    # Video processing
    parser.add_argument('--video-max-frames', type=int, default=64,
                        help='Max video frames for Omni Thinker (default: 64)')
    
    # Data
    parser.add_argument('--data-path', type=str,
                        default='./data/demo_test_data.json')
    parser.add_argument('--num-samples', type=int, default=1,
                        help='Number of test samples')
    
    # Output
    parser.add_argument('--output-dir', type=str,
                        default='./outputs/full_pipeline')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Visualize SAM3 results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_full_pipeline(args)

if __name__ == '__main__':
    main()
