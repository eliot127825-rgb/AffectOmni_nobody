#!/usr/bin/env python3
"""
Integrated evaluation script: Omni Thinker model generates thinking + Summarizer compression.
"""
import os
import sys
import json
import re
import torch
import argparse
from tqdm import tqdm
from pathlib import Path

# Add eval path
sys.path.insert(0, os.environ.get('EVAL_PATH', './eval'))
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from qwen_omni_utils import process_mm_info

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

def load_test_samples(data_path, video_dir=None, num_samples=5):
    """Load test samples, supporting multiple data formats"""
    with open(data_path) as f:
        data = json.load(f)
    
    # Take first num_samples samples
    samples = []
    for item in data[:num_samples]:
        # Get video path
        video = item.get('video_path') or item.get('video', '')
        if video_dir and not os.path.isabs(video):
            video = os.path.join(video_dir, video)
        
        # Build question text (supports multiple formats)
        if 'question' in item:
            question = item['question']
        else:
            # Use problem + options format
            question = item.get('problem', '')
            if 'options' in item:
                options_text = '\n'.join(item['options'])
                question = f"{question} Options:\n{options_text}\n Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
        
        samples.append({
            'qid': item.get('qid', ''),
            'video_path': video,
            'question': question,
            'problem_type': item.get('problem_type', ''),
            'solution': item.get('solution', '')
        })
    
    return samples

def run_evaluation(args):
    """Run the full evaluation pipeline"""
    
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
    
    # 3. Load test samples
    print("="*80)
    print(f"Step 3: Loading test samples (first {args.num_samples})")
    print("="*80)
    test_samples = load_test_samples(args.data_path, args.video_dir, args.num_samples)
    print(f"Loaded {len(test_samples)} test samples\n")
    
    # 4. Run pipeline
    print("="*80)
    print("Step 4: Running full pipeline")
    print("="*80)
    
    results = []
    
    for idx, sample in enumerate(tqdm(test_samples, desc="Processing samples")):
        print(f"\n{'='*80}")
        print(f"Sample {idx+1}/{len(test_samples)}")
        print(f"{'='*80}")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Video: {sample['video_path']}")
        
        # 4.1 Build input
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": sample['video_path']},
                    {"type": "text", "text": sample['question']}
                ]
            }
        ]
        
        # process_mm_info returns (audios, images, videos)
        audios, images, videos = process_mm_info(message, use_audio_in_video=False)
        
        # Apply chat template
        text = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Build model inputs
        model_inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        
        # 4.2 Generate thinking
        print("\nGenerating thinking...")
        with torch.inference_mode():
            text_ids = model.generate(
                **model_inputs.to(model.device).to(model.dtype),
                use_audio_in_video=False,
                max_new_tokens=2048
            )
        
        full_output = processor.decode(text_ids[0][model_inputs.input_ids.size(1):], skip_special_tokens=True)
        thinking = extract_think(full_output)
        answer = extract_answer(full_output)
        
        print(f"Thinking generated (length: {len(thinking)} chars)")
        print(f"Answer: {answer}")
        
        # Print full thinking
        print("\n" + "="*80)
        print("[Full Model Output]")
        print("="*80)
        print(f"\n>>> Full Output (length: {len(full_output)} chars):\n")
        print(full_output)
        print("\n" + "-"*80)
        print(f">>> Thinking (length: {len(thinking)} chars):\n")
        print(thinking)
        print("\n" + "-"*80)
        print(f">>> Answer:\n")
        print(answer)
        print("="*80)
        
        # 4.3 Summarize thinking
        print("\nSummarizing thinking...")
        summary = summarizer.summarize(thinking)
        print(f"Summary generated (length: {len(summary)} chars)")
        
        # Print full summary
        print("\n" + "="*80)
        print("[Summarizer Model Output]")
        print("="*80)
        print(summary)
        print("="*80)
        
        # Save results
        result = {
            'qid': sample['qid'],
            'question': sample['question'],
            'video_path': sample['video_path'],
            'ground_truth': sample['solution'],
            'full_output': full_output,
            'thinking': thinking,
            'answer': answer,
            'summary': summary
        }
        results.append(result)
    
    # 5. Save results
    output_file = args.output_file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("Pipeline complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Processed {len(results)} samples")
    
    # Statistics
    print(f"\n{'='*80}")
    print("Statistics:")
    print(f"{'='*80}")
    avg_thinking_len = sum(len(r['thinking']) for r in results) / len(results)
    avg_summary_len = sum(len(r['summary']) for r in results) / len(results)
    compression_ratio = (1 - avg_summary_len / avg_thinking_len) * 100 if avg_thinking_len > 0 else 0
    
    print(f"Average Thinking length: {avg_thinking_len:.0f} chars")
    print(f"Average Summary length: {avg_summary_len:.0f} chars")
    print(f"Compression ratio: {compression_ratio:.1f}%")
    print(f"\nDetailed results saved to JSON file.")

def main():
    parser = argparse.ArgumentParser(description="Integrated evaluation: Omni Thinker + Summarizer")
    
    # Omni Thinker model
    parser.add_argument(
        '--humanomniv2-model',
        type=str,
        default=os.environ.get('MODEL_CHECKPOINT_DIR', './checkpoints'),
        help='Path to Omni Thinker model'
    )
    
    # Summarizer model
    parser.add_argument(
        '--base-model',
        type=str,
        default=os.environ.get('SUMMARIZER_BASE_MODEL', './Qwen2.5-3B-Instruct'),
        help='Summarizer base model path'
    )
    parser.add_argument(
        '--lora-model',
        type=str,
        default=os.environ.get('SUMMARIZER_LORA_MODEL', './thinking_summarizer/outputs/final_model'),
        help='Summarizer LoRA weights path'
    )
    
    # Data
    parser.add_argument(
        '--data-path',
        type=str,
        default='./data/demo_test_data.json',
        help='Test data path'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default='./data/demo_videos',
        help='Video files directory'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of test samples'
    )
    
    # Output
    parser.add_argument(
        '--output-file',
        type=str,
        default='./outputs/eval_with_summarizer_results.json',
        help='Output results file'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    run_evaluation(args)

if __name__ == '__main__':
    main()
