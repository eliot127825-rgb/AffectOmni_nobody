#!/usr/bin/env python3
"""
Merge thinking outputs from multiple shards.
"""

import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Merge thinking output shards')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing shard files')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output merged file path')
    parser.add_argument('--num-shards', type=int, required=True,
                        help='Number of shards to merge')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    
    print("=" * 80)
    print("Merge Thinking Output Files")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Number of shards: {args.num_shards}")
    print("=" * 80)
    
    # Read all shards
    all_results = []
    for i in range(args.num_shards):
        shard_file = input_dir / f'thinking_data_shard_{i}.json'
        
        if not shard_file.exists():
            print(f"Shard {i} does not exist: {shard_file}")
            continue
        
        print(f"\nReading Shard {i}...")
        with open(shard_file, 'r', encoding='utf-8') as f:
            shard_data = json.load(f)
        
        print(f"  Loaded {len(shard_data)} samples")
        all_results.extend(shard_data)
    
    # Deduplicate (based on video_id + question, since same video may have multiple questions)
    print(f"\nBefore merge: {len(all_results)} samples")
    seen_keys = set()
    unique_results = []
    for item in all_results:
        # Use video_id + question as unique key
        # Prefer qid if available
        qid = item.get('metadata', {}).get('qid', '')
        if qid:
            key = qid
        else:
            key = (item['video_id'], item.get('question', ''))
        
        if key not in seen_keys:
            seen_keys.add(key)
            unique_results.append(item)
    
    print(f"After dedup: {len(unique_results)} samples (removed {len(all_results) - len(unique_results)} duplicates)")
    
    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unique_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print(f"Done! Merged {len(unique_results)} thinking samples")
    print(f"Output file: {output_path}")
    print("=" * 80)
    
    # Statistics
    thinking_lengths = [len(r['thinking']) for r in unique_results if r['thinking']]
    if thinking_lengths:
        print(f"\nThinking statistics:")
        print(f"  Average length: {sum(thinking_lengths) // len(thinking_lengths)} chars")
        print(f"  Shortest: {min(thinking_lengths)} chars")
        print(f"  Longest: {max(thinking_lengths)} chars")
    
    # Data source statistics
    source_counts = {}
    for item in unique_results:
        source = item['metadata'].get('source', 'unknown')
        source_counts[source] = source_counts.get(source, 0) + 1
    
    if source_counts:
        print("\nData source statistics:")
        for source, count in sorted(source_counts.items()):
            print(f"  {source}: {count} samples")

if __name__ == '__main__':
    main()
