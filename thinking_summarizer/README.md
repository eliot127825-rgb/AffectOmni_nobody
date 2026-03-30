# Thinking Summarizer

A lightweight module that compresses long-form reasoning chains into structured summaries, which can then drive downstream visual grounding (e.g., SAM3 video segmentation).

## Overview

The full pipeline works as follows:

```
Video + Question
       |
       v
 Omni Thinker Model        (multimodal reasoning, generates long thinking chains)
       |
       v
 Thinking Summarizer        (extracts key points, focus objects, and segmentation instructions)
       |
       v
 SAM3 Video Segmentation    (produces per-frame masks for identified objects)
       |
       v
 Visualized Video + Masks
```

The Thinking Summarizer is trained via LoRA on a small language model, using structured summaries distilled from a larger teacher model as supervision.

## Project Structure

```
thinking_summarizer/
├── eval_full_pipeline_with_sam3.py   # End-to-end pipeline evaluation
├── eval_with_summarizer.py           # Summarizer-only evaluation
├── create_masked_video.sh            # Compose mask frames into video
└── scripts/
    ├── generate_thinking.py          # Generate thinking data from the thinker model
    ├── call_api_summarize.py         # Call LLM API to produce structured summaries
    ├── build_training_dataset.py     # Build instruction-tuning dataset
    ├── train_summarizer.py           # LoRA fine-tuning script
    ├── sample_datasets.py            # Dataset sampling utilities
    ├── merge_shards.py               # Merge parallel generation shards
    ├── merge_summaries.py            # Merge parallel API outputs
    ├── test_lora_model.py            # Quick LoRA model test
    └── run_*.sh                      # Various launch scripts
```

## Quick Start

### 1. Generate Thinking Data

```bash
python scripts/generate_thinking.py \
  --model-path /path/to/thinker/checkpoint \
  --data-path /path/to/eval/data.json \
  --output-path ./data/thinking_outputs \
  --max-samples 1000
```

### 2. Generate Structured Summaries

```bash
python scripts/call_api_summarize.py \
  --input-path ./data/thinking_outputs/thinking_data_all.json \
  --output-path ./data/thinking_summaries \
  --api-type qwen \
  --api-key $API_KEY \
  --max-samples 1000
```

### 3. Build Training Dataset

```bash
python scripts/build_training_dataset.py \
  --summary-path ./data/thinking_summaries/summaries_all.json \
  --output-path ./data/training_data \
  --train-ratio 0.8
```

### 4. Train Summarizer (LoRA)

```bash
python scripts/train_summarizer.py \
  --model_name_or_path /path/to/base/model \
  --data_path ./data/training_data/train.json \
  --eval_data_path ./data/training_data/val.json \
  --output_dir ./outputs/thinking_summarizer \
  --bf16 True
```

### 5. Run Full Pipeline

```bash
python eval_full_pipeline_with_sam3.py \
  --data-path /path/to/test/data.json \
  --num-samples 10
```

## Output

- **JSON results**: `./outputs/full_pipeline/full_pipeline_results.json`
- **Mask images**: `./outputs/sam3_visualizations/*.jpg`
- **Mask video**: run `./create_masked_video.sh` to compose frames into a video
