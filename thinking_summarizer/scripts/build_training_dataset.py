#!/usr/bin/env python3
"""
Convert API summary data into training datasets.
Format: instruction-input-output, suitable for fine-tuning LLMs.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

INSTRUCTION_TEMPLATE = """You are an expert in analyzing reasoning processes and extracting structured information. Given a detailed thinking process about video content analysis, extract key points and generate SAM3 segmentation instructions.

Please output in the following format:

## Key Points Analysis
1. [First key point]
2. [Second key point]
3. [Third key point]

## Video Focus Objects
- People: [List all important people with their appearance features]
- Objects: [List all important objects]
- Scenes: [List important scene elements]

## Emotional Indicators (if applicable)
- [List specific visual cues that indicate emotions, such as facial expressions, body language, tone, gestures, etc.]

## SAM3 Segmentation Instructions
Please segment the following in the video: [object1], [object2], [object3]

Note: List exactly 3 objects separated by commas on a single line.

Requirements:
1. Key points should be concise and clear, each within 30 words
2. Extract only content relevant to visual segmentation
3. For people analysis, include:
   - Physical appearance (e.g., "woman in red dress")
   - Facial expressions and emotions (e.g., "smiling man", "frustrated woman with furrowed brows")
   - Body language indicators (e.g., "person with crossed arms", "nodding")
4. Object descriptions should include location information (e.g., "vase on the table")
5. If the video involves people or character scenes, prioritize listing visual details that help infer emotions (facial expressions, posture, gestures, tone indicators)
6. **IMPORTANT**: SAM3 Segmentation Instructions must list ONLY the top 3 most important objects/people to segment. Choose the most critical elements from the analysis.
7. Output must strictly follow the above format"""

def format_summary_as_output(summary_text: str) -> str:
    """Format the API-returned summary text as training output"""
    # Directly use the full summary text returned by the API
    return summary_text.strip()

def build_training_sample(item: Dict) -> Dict:
    """Build a single training sample"""
    return {
        "instruction": INSTRUCTION_TEMPLATE,
        "input": item["original_thinking"],
        "output": item["summary_text"],
        "metadata": {
            "video_id": item["video_id"],
            "source": item.get("metadata", {}).get("source", ""),
            "type": item.get("metadata", {}).get("Type", ""),
        }
    }

def main():
    parser = argparse.ArgumentParser(description='Build training dataset from API summaries')
    parser.add_argument('--input-file', type=str, required=True,
                        help='Input summaries_all.json file')
    parser.add_argument('--output-dir', type=str, default='./data/training_dataset',
                        help='Output directory for training data')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='Ratio of training samples (default: 0.9)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splitting')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Build Training Dataset")
    print("=" * 80)
    print(f"Input file: {args.input_file}")
    print(f"Output path: {args.output_dir}")
    print(f"Training ratio: {args.train_ratio}")
    print("=" * 80)
    
    # Load API summary data
    print("\nLoading API summary data...")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
    
    print(f"Loaded {len(summaries)} summary samples")
    
    # Build training samples
    print("\nBuilding training samples...")
    training_samples = []
    
    for item in summaries:
        try:
            sample = build_training_sample(item)
            training_samples.append(sample)
        except Exception as e:
            print(f"Error processing {item.get('video_id', 'unknown')}: {str(e)}")
            continue
    
    print(f"Successfully built {len(training_samples)} training samples")
    
    # Random shuffle
    random.shuffle(training_samples)
    
    # Split into training and validation sets
    split_idx = int(len(training_samples) * args.train_ratio)
    train_samples = training_samples[:split_idx]
    val_samples = training_samples[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_samples)} samples")
    print(f"  Validation: {len(val_samples)} samples")
    
    # Save training set
    train_file = output_dir / 'train.json'
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    print(f"\nTraining set saved: {train_file}")
    
    # Save validation set
    val_file = output_dir / 'val.json'
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    print(f"Validation set saved: {val_file}")
    
    # Statistics
    print("\n" + "=" * 80)
    print("Dataset Statistics")
    print("=" * 80)
    
    # Count data sources
    sources = {}
    for sample in training_samples:
        source = sample['metadata']['source']
        sources[source] = sources.get(source, 0) + 1
    
    print("\nData source distribution:")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count} ({count/len(training_samples)*100:.1f}%)")
    
    # Compute input/output length statistics
    input_lengths = [len(s['input']) for s in training_samples]
    output_lengths = [len(s['output']) for s in training_samples]
    
    print(f"\nInput length statistics:")
    print(f"  Average: {sum(input_lengths)/len(input_lengths):.0f} chars")
    print(f"  Shortest: {min(input_lengths)} chars")
    print(f"  Longest: {max(input_lengths)} chars")
    
    print(f"\nOutput length statistics:")
    print(f"  Average: {sum(output_lengths)/len(output_lengths):.0f} chars")
    print(f"  Shortest: {min(output_lengths)} chars")
    print(f"  Longest: {max(output_lengths)} chars")
    
    print("\n" + "=" * 80)
    print("Dataset construction complete!")
    print("=" * 80)
    
    # Save an example
    example_file = output_dir / 'example.json'
    with open(example_file, 'w', encoding='utf-8') as f:
        json.dump(train_samples[0], f, ensure_ascii=False, indent=2)
    print(f"\nExample sample saved: {example_file}")

if __name__ == '__main__':
    main()
