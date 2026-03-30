#!/usr/bin/env python3
"""
Sample from multiple datasets and generate a unified format dataset for thinking generation.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict

def load_intentbench(data_path: str) -> List[Dict]:
    """Load IntentBench dataset"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        # IntentBench format conversion
        question = item['problem']
        if item['problem_type'] == 'multiple choice':
            question = question + " Options:\n"
            for op in item['options']:
                question += op + "\n"
        
        samples.append({
            'video_id': item['video'].replace('.mp4', '').replace('_trimmed-out', ''),
            'video_path': os.path.join(os.environ.get('DATA_ROOT', './data'), f"IntentBench/videos/{item['video']}"),
            'question': question,
            'problem_type': item['problem_type'],
            'data_type': item['data_type'],
            'metadata': {
                'source': 'IntentBench',
                'Type': item.get('Type', ''),
                'qid': item.get('qid', ''),
                'original_video': item['video']
            }
        })
    return samples

def load_daily_omni(data_path: str) -> List[Dict]:
    """Load Daily-Omni dataset"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = []
    for item in data:
        video_id = item['video_id']
        question = item['Question']
        choices = item.get('Choice', [])
        
        # Build complete question
        if choices:
            question = question + " Options:\n"
            for choice in choices:
                question += choice + "\n"
        
        samples.append({
            'video_id': video_id,
            'video_path': os.path.join(os.environ.get('DATA_ROOT', './data'), f"Videos/{video_id}/{video_id}_video.mp4"),
            'question': question,
            'problem_type': 'multiple choice',
            'data_type': 'video',
            'metadata': {
                'source': 'Daily-Omni',
                'Type': item.get('Type', ''),
                'video_category': item.get('video_category', '')
            }
        })
    return samples

def load_intent_train(base_path: str) -> List[Dict]:
    """Load IntentTrain dataset"""
    samples = []
    
    # Load emer_rewrite.json
    emer_path = Path(base_path) / 'emer_rewrite.json'
    if emer_path.exists():
        with open(emer_path, 'r', encoding='utf-8') as f:
            emer_data = json.load(f)
        
        for item in emer_data:
            question = item['problem']
            if item['problem_type'] == 'emer_ov_mc':
                question = question + " Options:\n"
                for op in item['options']:
                    question += op + "\n"
            
            samples.append({
                'video_id': item['path'].replace('/', '_').replace('.mp4', ''),
                'video_path': f"${PROJECT_ROOT}/data/IntentTrain/videos/{item['path']}",
                'question': question,
                'problem_type': item['problem_type'],
                'data_type': item['data_type'],
                'metadata': {
                    'source': 'IntentTrain-emer',
                    'original_path': item['path']
                }
            })
    
    # Load social_iq_v2_rewrite.json
    social_path = Path(base_path) / 'social_iq_v2_rewrite.json'
    if social_path.exists():
        with open(social_path, 'r', encoding='utf-8') as f:
            social_data = json.load(f)
        
        for item in social_data:
            question = item['problem']
            if item['problem_type'] == 'multiple choice':
                question = question + " Options:\n"
                for op in item['options']:
                    question += op + "\n"
            
            samples.append({
                'video_id': item['path'].replace('/', '_').replace('.mp4', ''),
                'video_path': os.path.join(os.environ.get('DATA_ROOT', './data'), f"IntentTrain/videos/{item['path']}"),
                'question': question,
                'problem_type': item['problem_type'],
                'data_type': item['data_type'],
                'metadata': {
                    'source': 'IntentTrain-social',
                    'original_path': item['path']
                }
            })
    
    return samples

def sample_data(all_samples: Dict[str, List[Dict]], sample_counts: Dict[str, int], seed: int = 42) -> List[Dict]:
    """Sample from each dataset"""
    random.seed(seed)
    
    sampled = []
    for dataset_name, count in sample_counts.items():
        if dataset_name in all_samples:
            dataset_samples = all_samples[dataset_name]
            if len(dataset_samples) >= count:
                sampled.extend(random.sample(dataset_samples, count))
            else:
                print(f"  {dataset_name}: requested {count} samples, but only {len(dataset_samples)} available, using all")
                sampled.extend(dataset_samples)
    
    # Shuffle the order
    random.shuffle(sampled)
    return sampled

def main():
    parser = argparse.ArgumentParser(description='Sample data from multiple datasets')
    parser.add_argument('--intentbench-path', type=str, 
                        default='./data/IntentBench/qa.json',
                        help='Path to IntentBench qa.json')
    parser.add_argument('--daily-path', type=str,
                        default='./data/DailyOmni/qa.json',
                        help='Path to Daily-Omni qa.json')
    parser.add_argument('--intenttrain-path', type=str,
                        default='./data/IntentTrain',
                        help='Path to IntentTrain directory')
    parser.add_argument('--output-path', type=str, required=True,
                        help='Output path for sampled data')
    parser.add_argument('--intentbench-count', type=int, default=400,
                        help='Number of samples from IntentBench')
    parser.add_argument('--daily-count', type=int, default=400,
                        help='Number of samples from Daily-Omni')
    parser.add_argument('--intenttrain-count', type=int, default=200,
                        help='Number of samples from IntentTrain')
    parser.add_argument('--use-all', action='store_true',
                        help='Use all available data instead of sampling (ignores count arguments)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Multi-Dataset Sampling Tool")
    print("=" * 80)
    
    # Load datasets
    all_samples = {}
    
    print("\nLoading IntentBench...")
    intentbench_samples = load_intentbench(args.intentbench_path)
    all_samples['IntentBench'] = intentbench_samples
    print(f"  Loaded {len(intentbench_samples)} samples")
    
    print("\nLoading Daily-Omni...")
    daily_samples = load_daily_omni(args.daily_path)
    all_samples['Daily'] = daily_samples
    print(f"  Loaded {len(daily_samples)} samples")
    
    print("\nLoading IntentTrain...")
    intenttrain_samples = load_intent_train(args.intenttrain_path)
    all_samples['IntentTrain'] = intenttrain_samples
    print(f"  Loaded {len(intenttrain_samples)} samples")
    
    # Sample or merge all
    print("\n" + "=" * 80)
    if args.use_all:
        print("Full data merge mode (--use-all)")
        print("=" * 80)
        # Directly merge all data
        sampled_data = []
        for dataset_name, dataset_samples in all_samples.items():
            print(f"  {dataset_name}: {len(dataset_samples)} samples")
            sampled_data.extend(dataset_samples)
        # Shuffle order
        random.seed(args.seed)
        random.shuffle(sampled_data)
    else:
        print("Sampling config:")
        print(f"  IntentBench: {args.intentbench_count} samples")
        print(f"  Daily-Omni:  {args.daily_count} samples")
        print(f"  IntentTrain: {args.intenttrain_count} samples")
        print(f"  Random seed: {args.seed}")
        print("=" * 80)
        
        sample_counts = {
            'IntentBench': args.intentbench_count,
            'Daily': args.daily_count,
            'IntentTrain': args.intenttrain_count
        }
        
        sampled_data = sample_data(all_samples, sample_counts, args.seed)
    
    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDone! Sampled {len(sampled_data)} samples total")
    print(f"Output file: {output_path}")
    
    # Statistics
    print("\nData source statistics:")
    source_counts = {}
    for item in sampled_data:
        source = item['metadata']['source']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} samples")

if __name__ == '__main__':
    main()
