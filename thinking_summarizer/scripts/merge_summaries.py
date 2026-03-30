#!/usr/bin/env python3
"""Merge summary results from 4 workers"""

import os
import json
from pathlib import Path

base_dir = Path(os.environ.get('SUMMARIES_DIR', './data/thinking_summaries'))

all_summaries = []
for i in range(4):
    file = base_dir / f'workers/worker_{i}/summaries_all.json'
    if file.exists():
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_summaries.extend(data)
            print(f'Worker {i}: {len(data)} entries')

output_file = base_dir / 'summaries_all.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_summaries, f, indent=2, ensure_ascii=False)

print(f'\nMerge complete, total {len(all_summaries)} entries')
print(f'Output file: {output_file}')

# Statistics
print(f'\nData source statistics:')
sources = {}
for item in all_summaries:
    src = item.get('metadata', {}).get('source', 'unknown')
    sources[src] = sources.get(src, 0) + 1
for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
    print(f'   {src}: {cnt}')
