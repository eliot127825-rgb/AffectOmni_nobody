#!/usr/bin/env python3
"""
Call API to generate thinking summaries and SAM3 instructions.
Supports OpenAI API, Qwen API, etc.
"""

import os
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import requests

SUMMARIZE_PROMPT_TEMPLATE = """You are an expert in text content analysis and object segmentation. Analyze the following reasoning process and extract structured information for SAM3 video segmentation.

Input Thinking:
{thinking_text}

**CRITICAL REQUIREMENTS FOR SAM3 SEGMENTATION:**

1. **Object Naming Rules** (MUST FOLLOW):
   - Use STANDARD category names: person, car, bicycle, dog, cat, chair, table, bed, laptop, bottle, cup, etc.
   - For people: use "person" as category, BUT add key appearance features (e.g., "person: man in red shirt", "person: woman with glasses")
   - For objects: use base category WITH important attributes if needed (e.g., "car: red sports car on street")
   - DO include descriptive details that help SAM3 identify the specific object
   - NEVER use abstract concepts like "dimly lit room" or "romantic atmosphere"

2. **Object Priority**:
   - **Primary Objects**: People or key objects directly related to the question (MUST segment)
   - **Secondary Objects**: Concrete background objects (OPTIONAL)

**Output Format:**

## Key Points Analysis
1. [First key observation - max 25 words]
2. [Second key observation - max 25 words]
3. [Third key observation - max 25 words]

## Video Focus Objects (Maximum 2 objects)

### Primary Objects (Priority 1) - Choose the 2 MOST important
- person: [man/woman in [color] [clothing], with [key features]]
- person/object: [detailed description with color and location]

## Emotional Indicators (if applicable)
- [Specific visual cues: facial expressions, gestures, body language]

## SAM3 Segmentation Instructions
Please segment the following in the video: [detailed object 1], [detailed object 2]

**Rules:**
- Use standard category names (person, car, table, etc.)
- Include KEY appearance features for better identification (e.g., "man in red shirt", "woman with black top")
- Prioritize people and key objects mentioned in the question
- **Maximum 2 objects total** - choose only the most important ones
- All objects must be concrete and segmentable
"""

def call_openai_api(prompt: str, api_key: str, model: str = "gpt-4") -> str:
    """Call OpenAI API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert assistant in analyzing reasoning processes and extracting structured information."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data,
        timeout=60
    )
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"API call failed: {response.status_code}, {response.text}")

def call_qwen_api(prompt: str, api_key: str, model: str = "qwen-max") -> str:
    """Call Qwen API"""
    import dashscope
    from dashscope import Generation
    
    dashscope.api_key = api_key
    
    response = Generation.call(
        model=model,
        messages=[
            {'role': 'system', 'content': 'You are an expert assistant in analyzing reasoning processes and extracting structured information. You strictly follow naming conventions and output formats.'},
            {'role': 'user', 'content': prompt}
        ],
        result_format='message',
        temperature=0.2,
        top_p=0.8
    )
    
    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        raise Exception(f"API call failed: {response.status_code}, {response.message}")

def parse_summary(summary_text: str) -> Dict:
    """Parse the summary text returned by the API"""
    result = {
        'key_points': [],
        'focus_objects': {
            'people': [],
            'objects': [],
            'scenes': []
        },
        'emotional_indicators': [],
        'sam3_instruction': ''
    }
    
    # Extract key points
    import re
    points_pattern = r'## Key Points Analysis\n(.*?)(?=\n##|\Z)'
    points_match = re.search(points_pattern, summary_text, re.DOTALL)
    if points_match:
        points_text = points_match.group(1)
        points = re.findall(r'\d+\.\s*(.+)', points_text)
        result['key_points'] = [p.strip() for p in points]
    
    # Extract focus objects (new format: Primary/Secondary Objects)
    # Extract Primary Objects
    primary_pattern = r'### Primary Objects.*?\n(.*?)(?=\n###|\n##|\Z)'
    primary_match = re.search(primary_pattern, summary_text, re.DOTALL)
    if primary_match:
        primary_text = primary_match.group(1)
        primary_objects = re.findall(r'-\s*(\w+):\s*(.+)', primary_text)
        result['focus_objects']['people'] = [f"{obj[0]}: {obj[1].strip()}" for obj in primary_objects]
    
    # Extract Secondary Objects
    secondary_pattern = r'### Secondary Objects.*?\n(.*?)(?=\n##|\Z)'
    secondary_match = re.search(secondary_pattern, summary_text, re.DOTALL)
    if secondary_match:
        secondary_text = secondary_match.group(1)
        secondary_objects = re.findall(r'-\s*(\w+):\s*(.+)', secondary_text)
        result['focus_objects']['objects'] = [f"{obj[0]}: {obj[1].strip()}" for obj in secondary_objects]
    
    # Extract emotional indicators
    emotion_pattern = r'## Emotional Indicators.*?\n(.*?)(?=\n##|\Z)'
    emotion_match = re.search(emotion_pattern, summary_text, re.DOTALL)
    if emotion_match:
        emotion_text = emotion_match.group(1)
        emotions = re.findall(r'-\s*(.+)', emotion_text)
        result['emotional_indicators'] = [e.strip() for e in emotions]
    
    # Extract SAM3 instructions
    sam3_pattern = r'## SAM3 Segmentation Instructions\n(.+?)(?=\n##|\Z)'
    sam3_match = re.search(sam3_pattern, summary_text, re.DOTALL)
    if sam3_match:
        result['sam3_instruction'] = sam3_match.group(1).strip()
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Generate summaries using API')
    parser.add_argument('--input-path', type=str, required=True,
                        help='Input JSON file with thinking data')
    parser.add_argument('--output-path', type=str, default='./thinking_summaries',
                        help='Output directory for summaries')
    parser.add_argument('--api-type', type=str, choices=['openai', 'qwen'], default='qwen',
                        help='API type to use')
    parser.add_argument('--api-key', type=str, required=True,
                        help='API key')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name (default: gpt-4 for openai, qwen-max for qwen)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    # Set default model
    if args.model is None:
        args.model = 'gpt-4' if args.api_type == 'openai' else 'qwen-max'
    
    # Create output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("API Summary Config")
    print("=" * 80)
    print(f"Input file: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"API type: {args.api_type}")
    print(f"Model: {args.model}")
    print("=" * 80)
    
    # Load thinking data
    print("\nLoading thinking data...")
    with open(args.input_path, 'r', encoding='utf-8') as f:
        thinking_data = json.load(f)
    
    if args.max_samples:
        thinking_data = thinking_data[:args.max_samples]
    
    print(f"Loaded {len(thinking_data)} samples")
    
    # Select API call function
    api_call_func = call_openai_api if args.api_type == 'openai' else call_qwen_api
    
    # Batch processing
    results = []
    failed = []
    
    print("\nStarting summary generation...")
    for i, item in enumerate(tqdm(thinking_data)):
        try:
            
            # Build prompt
            thinking_text = item.get('thinking', '')
            if not thinking_text:
                print(f"\n{item['video_id']} has no thinking content, skipping")
                continue
            
            prompt = SUMMARIZE_PROMPT_TEMPLATE.format(thinking_text=thinking_text)
            
            # Call API
            summary_text = api_call_func(prompt, args.api_key, args.model)
            
            # Parse results
            parsed_summary = parse_summary(summary_text)
            
            result = {
                'video_id': item['video_id'],
                'video_path': item.get('video_path', ''),
                'original_thinking': thinking_text,
                'summary_text': summary_text,
                'parsed_summary': parsed_summary,
                'metadata': item.get('metadata', {})
            }
            
            results.append(result)
            
            # API rate limiting
            time.sleep(0.5)
            
            # Periodically save
            if (i + 1) % 50 == 0:
                batch_file = output_dir / f'summaries_batch_{i+1}.json'
                with open(batch_file, 'w', encoding='utf-8') as f:
                    json.dump(results[-50:], f, ensure_ascii=False, indent=2)
        
        except Exception as e:
            print(f"\nError processing {item['video_id']}: {str(e)}")
            failed.append({
                'video_id': item['video_id'],
                'error': str(e)
            })
            continue
    
    # Save final results
    output_file = output_dir / 'summaries_all.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save failure records
    if failed:
        failed_file = output_dir / 'failed_samples.json'
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Done!")
    print(f"Successfully processed: {len(results)} samples")
    print(f"Failed: {len(failed)} samples")
    print(f"Output file: {output_file}")
    if failed:
        print(f"Failure records: {failed_file}")
    print("=" * 80)
    
    # Statistics
    if results:
        key_points_counts = [len(r['parsed_summary']['key_points']) for r in results]
        print(f"\nKey points statistics:")
        print(f"  Average count: {sum(key_points_counts) / len(key_points_counts):.1f}")
        print(f"  Minimum: {min(key_points_counts)}")
        print(f"  Maximum: {max(key_points_counts)}")

if __name__ == '__main__':
    main()
