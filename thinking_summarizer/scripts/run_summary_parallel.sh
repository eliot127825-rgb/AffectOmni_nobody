#!/bin/bash

set -e

INPUT_FILE="./data/thinking_outputs_all_6772/thinking_data_all.json"
OUTPUT_BASE="./data/thinking_summaries_6772"
API_KEY="${API_KEY:-your_api_key_here}"
MODEL="qwen-max"
TOTAL_SAMPLES=6772
NUM_WORKERS=4

echo "================================================================================"
echo "4-worker parallel summary generation script"
echo "================================================================================"
echo "Input file: $INPUT_FILE"
echo "Output path: $OUTPUT_BASE"
echo "Total samples: $TOTAL_SAMPLES"
echo "Workers: $NUM_WORKERS"
echo "================================================================================"

# Calculate samples per worker
SAMPLES_PER_WORKER=$((TOTAL_SAMPLES / NUM_WORKERS))
echo "Samples per worker: ~$SAMPLES_PER_WORKER"

# Create output directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$OUTPUT_BASE/logs"
mkdir -p "$OUTPUT_BASE/workers"

echo ""
echo "Splitting input data..."
python3 << EOF
import json

with open('$INPUT_FILE', 'r') as f:
    data = json.load(f)

total = len(data)
chunk_size = (total + $NUM_WORKERS - 1) // $NUM_WORKERS

for i in range($NUM_WORKERS):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, total)
    chunk = data[start:end]
    
    output_file = f'$OUTPUT_BASE/workers/input_worker_{i}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, indent=2, ensure_ascii=False)
    
    print(f"Worker {i}: {len(chunk)} samples (index {start}-{end-1})")

print(f"Data split complete, {total} samples total")
EOF

echo ""
echo "Starting $NUM_WORKERS parallel workers..."

for WORKER_ID in $(seq 0 $((NUM_WORKERS - 1))); do
    INPUT_WORKER="$OUTPUT_BASE/workers/input_worker_${WORKER_ID}.json"
    OUTPUT_WORKER="$OUTPUT_BASE/workers/worker_${WORKER_ID}"
    LOG_FILE="$OUTPUT_BASE/logs/worker_${WORKER_ID}.log"
    
    echo "  Starting Worker $WORKER_ID..."
    
    nohup python3 scripts/call_api_summarize.py \
        --input-path "$INPUT_WORKER" \
        --output-path "$OUTPUT_WORKER" \
        --api-type qwen \
        --api-key "$API_KEY" \
        --model "$MODEL" \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "    PID: $PID, Log: $LOG_FILE"
    
    sleep 2
done

echo ""
echo "================================================================================"
echo "All workers started"
echo "================================================================================"
echo ""
echo "Monitoring commands:"
echo "  View all logs: tail -f $OUTPUT_BASE/logs/worker_*.log"
echo "  View single worker: tail -f $OUTPUT_BASE/logs/worker_0.log"
echo "  Check processes: ps aux | grep call_api_summarize.py"
echo ""
echo "After all workers complete, run merge command:"
echo "  python3 << 'MERGE_EOF'"
echo "  import json"
echo "  from pathlib import Path"
echo "  "
echo "  all_summaries = []"
echo "  for i in range($NUM_WORKERS):"
echo "      file = Path('$OUTPUT_BASE/workers/worker_{i}/summaries_all.json'.format(i=i))"
echo "      if file.exists():"
echo "          with open(file, 'r') as f:"
echo "              data = json.load(f)"
echo "              all_summaries.extend(data)"
echo "              print(f'Worker {i}: {len(data)} samples')"
echo "  "
echo "  output_file = '$OUTPUT_BASE/summaries_all.json'"
echo "  with open(output_file, 'w', encoding='utf-8') as f:"
echo "      json.dump(all_summaries, f, indent=2, ensure_ascii=False)"
echo "  "
echo "  print(f'Merge complete, {len(all_summaries)} samples total')"
echo "  print(f'Output file: {output_file}')"
echo "  MERGE_EOF"
echo ""
echo "================================================================================"
