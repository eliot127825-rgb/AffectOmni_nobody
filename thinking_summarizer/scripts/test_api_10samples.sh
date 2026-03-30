#!/bin/bash

set -e

cd ${PROJECT_ROOT}/thinking_summarizer

echo "================================================================================"
echo "Test API call - 10 samples"
echo "================================================================================"

# 1. Create test data (first 10 entries)
echo "1. Creating test data..."
python3 << 'EOF'
import json

with open('./data/thinking_outputs_all_6772/thinking_data_all.json', 'r') as f:
    data = json.load(f)

test_data = data[:10]

with open('./data/test_10_samples.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)

print(f'Created test data: {len(test_data)} samples')
for i, item in enumerate(test_data):
    print(f'  {i+1}. {item["video_id"]} - {item["metadata"].get("source", "unknown")}')
EOF

# 2. Run API test
echo ""
echo "2. Calling API to generate summaries..."
echo "================================================================================"

python3 scripts/call_api_summarize.py \
    --input-path ./data/test_10_samples.json \
    --output-path ./data/test_summaries_10 \
    --api-type qwen \
    --api-key "${API_KEY}" \
    --model qwen-max

echo ""
echo "================================================================================"
echo "Test complete!"
echo "================================================================================"
echo "Result file: ./data/test_summaries_10/summaries_all.json"
echo ""
echo "View results:"
echo "  cat ./data/test_summaries_10/summaries_all.json | head -100"
