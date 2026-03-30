#!/bin/bash
# 4-GPU parallel thinking output generation

# Configuration
MODEL_PATH="${PROJECT_ROOT}/outputs/stage4_debug_no_audio_v2/checkpoint-380"
DATA_PATH="./data/sampled_mixed_1000.json"
OUTPUT_PATH="./data/thinking_outputs_mixed_1000_4gpu"
NUM_SHARDS=4

echo "========================================"
echo "4-GPU parallel thinking generation"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_PATH"
echo "Num shards: $NUM_SHARDS"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Start 4 processes, each using a different GPU
for i in {0..3}; do
    echo "Starting GPU $i (Shard $i)..."
    CUDA_VISIBLE_DEVICES=$i python generate_thinking.py \
        --model-path "$MODEL_PATH" \
        --data-path "$DATA_PATH" \
        --output-path "$OUTPUT_PATH" \
        --shard-id $i \
        --num-shards $NUM_SHARDS \
        --extract-thinking-only \
        > "$OUTPUT_PATH/shard_${i}.log" 2>&1 &
    
    # Record process ID
    echo $! >> "$OUTPUT_PATH/pids.txt"
    
    # Brief delay to avoid simultaneous model loading
    sleep 5
done

echo ""
echo "All processes started!"
echo "Process IDs saved to: $OUTPUT_PATH/pids.txt"
echo ""
echo "View progress:"
echo "  tail -f $OUTPUT_PATH/shard_0.log"
echo "  tail -f $OUTPUT_PATH/shard_1.log"
echo "  tail -f $OUTPUT_PATH/shard_2.log"
echo "  tail -f $OUTPUT_PATH/shard_3.log"
echo ""
echo "Monitor all processes:"
echo "  watch -n 5 'ps aux | grep generate_thinking.py'"
echo ""
echo "Wait for all processes to complete:"
echo "  wait \$(cat $OUTPUT_PATH/pids.txt)"
