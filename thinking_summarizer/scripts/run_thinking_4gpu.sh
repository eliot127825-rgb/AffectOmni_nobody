#!/bin/bash

set -e

MODEL_PATH="${PROJECT_ROOT}/outputs/stage4_debug_no_audio_v2/checkpoint-380"
DATA_PATH="./data/sampled_mixed_all.json"
OUTPUT_PATH="./data/thinking_outputs_all_2gpu"
MAX_SAMPLES=999999  # set large enough to process all samples
NUM_SHARDS=2

echo "================================================================================"
echo "4-GPU parallel thinking generation script"
echo "================================================================================"
echo "Model path: $MODEL_PATH"
echo "Data path: $DATA_PATH"
echo "Output path: $OUTPUT_PATH"
echo "Total samples: $MAX_SAMPLES"
echo "Number of GPUs: $NUM_SHARDS"
echo "================================================================================"

mkdir -p "$OUTPUT_PATH"
mkdir -p "$OUTPUT_PATH/logs"

echo "Cleaning old log files..."
rm -f "$OUTPUT_PATH"/logs/shard_*.log

echo ""
echo "Starting 2 GPU processes (GPU 4,5)..."

for SHARD_ID in 0 1; do
    GPU_ID=$((SHARD_ID + 4))  # GPU 4, 5
    LOG_FILE="$OUTPUT_PATH/logs/shard_${SHARD_ID}.log"
    
    echo "  Starting GPU $GPU_ID (Shard $SHARD_ID)..."
    
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python3 scripts/generate_thinking.py \
        --model-path "$MODEL_PATH" \
        --data-path "$DATA_PATH" \
        --output-path "$OUTPUT_PATH" \
        --shard-id $SHARD_ID \
        --num-shards $NUM_SHARDS \
        --max-samples $MAX_SAMPLES \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "    PID: $PID, Log: $LOG_FILE"
    
    sleep 2
done

echo ""
echo "================================================================================"
echo "All processes started"
echo "================================================================================"
echo ""
echo "Monitoring commands:"
echo "  View all logs: tail -f $OUTPUT_PATH/logs/shard_*.log"
echo "  View single GPU: tail -f $OUTPUT_PATH/logs/shard_0.log"
echo "  View progress: watch -n 30 'ls -lh $OUTPUT_PATH/thinking_data_shard_*.json'"
echo "  Check processes: ps aux | grep generate_thinking.py"
echo ""
echo "After all processes complete, run merge command:"
echo "  python3 scripts/merge_shards.py \\"
echo "    --input-dir $OUTPUT_PATH \\"
echo "    --output-file $OUTPUT_PATH/thinking_data_all.json"
echo ""
echo "================================================================================"
