"""
完整的时间戳后处理 Pipeline 测试脚本
整合所有模块：视频采帧 + 模型推理 + 事件提取 + CLIP 匹配 + 时间戳插入
"""

import sys
import os
import random
import yaml
import json
import torch
from datetime import datetime
from pathlib import Path

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../tools'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src/src'))

from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration

# 导入我们的模块
from tools.video_utils import sample_frames, get_video_info
from tools.clip_matcher import CLIPMatcher, match_with_monotonic_constraint
from extract_events import extract_events, events_to_queries
from insert_timestamps import insert_timestamps, verify_insertions


def load_test_samples(dataset_path: str):
    """加载测试样本（与test_base_model.py一致）"""
    print(f"📂 加载数据集: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    datasets = yaml_data.get('datasets', [])
    all_samples = []
    
    for dataset_config in datasets:
        json_path = dataset_config.get('json_path')
        data_root = dataset_config.get('data_root')
        
        print(f"  ├─ 加载: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 添加data_root路径
        if data_root:
            for sample in data:
                if 'path' in sample:
                    sample['path'] = os.path.join(data_root, sample['path'])
        
        all_samples.extend(data)
        print(f"  └─ 加载了 {len(data)} 个样本")
    
    print(f"\n✅ 总共加载 {len(all_samples)} 个样本\n")
    return all_samples


def format_question(sample):
    """格式化问题（和训练时一样）"""
    if sample['problem_type'] in ['multiple choice', 'emer_ov_mc']:
        question = sample['problem'] + "\nOptions:\n"
        for option in sample.get('options', []):
            question += option + "\n"
    else:
        question = sample['problem']
    
    return question


def create_messages(sample, system_prompt, timestamp_info=None):
    """创建对话消息（和test_base_model.py一致）"""
    question = format_question(sample)
    
    # TYPE_TEMPLATE
    TYPE_TEMPLATES = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "emer_ov": " Please provide the words to describe emotions within the  <answer> </answer> tags.",
        "emer_ov_mc": " Please provide only the single or multiple option letter (e.g., A for single option or A,E for multi option, etc.) within the <answer> </answer> tags.",
    }
    
    text_prompt = question + "\n" + TYPE_TEMPLATES.get(sample['problem_type'], "")
    
    # 如果提供了时间戳信息，添加到提示中
    if timestamp_info:
        text_prompt += "\n\n" + timestamp_info
    
    # 构造messages
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": sample.get('data_type', 'video'),
                    sample.get('data_type', 'video'): sample['path'],
                    "max_frames": 32,  # 与训练时保持一致
                    "max_pixels": 602112
                },
                {
                    "type": "text",
                    "text": text_prompt
                }
            ]
        }
    ]
    
    return messages


def run_inference(sample, model, processor, system_prompt):
    """运行模型推理（不带时间戳，与test_base_model.py一致）"""
    from qwen_omni_utils import process_mm_info
    
    # 构造消息（不带时间戳）
    messages = create_messages(sample, system_prompt, timestamp_info=None)
    
    # 应用 chat template
    texts = processor.apply_chat_template(
        [messages],
        tokenize=False,
        add_generation_prompt=True
    )
    text = texts[0]
    
    # 处理多模态
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    
    # 处理输入
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32768
    ).to(model.device)
    
    # 生成（与test_base_model.py参数一致）
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    # 解码
    generated_text = processor.batch_decode(
        outputs[:, inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )[0]
    
    return generated_text, inputs


def parse_think_section(text: str) -> str:
    """从生成文本中提取 <think> 部分"""
    import re
    match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def main():
    print("=" * 80)
    print("🎬 时间戳后处理 Pipeline 测试")
    print("=" * 80)
    print()
    
    # ==================== 配置 ====================
    MODEL_PATH = "${PROJECT_ROOT}/models/HumanOmniV2"
    BASE_MODEL_PATH = "${PROJECT_ROOT}/Qwen2.5-Omni-7B-Thinker"
    DATASET_PATH = "../configs/test_samples.yaml"
    MAX_FRAMES = 16  # 减少帧数以加快速度
    CLIP_MODEL = "ViT-B-32"
    USE_MONOTONIC_CONSTRAINT = True  # 保持时序约束
    LAMBDA_SMOOTH = 0.01  # 极小的平滑约束，主要依赖CLIP相似度
    
    SYSTEM_PROMPT = """You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."""
    
    # ==================== 加载模型 ====================
    print("🔧 加载模型...")
    print(f"  模型路径: {MODEL_PATH}")
    print(f"  Processor路径: {BASE_MODEL_PATH}")
    
    # Processor 从基座模型加载
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    # 模型使用自定义架构加载
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print(f"✅ 模型加载成功 (设备: {model.device})")
    print()
    
    # ==================== 加载 CLIP ====================
    print(f"🔧 加载 CLIP 模型: {CLIP_MODEL}...")
    # 使用原版 CLIP（离线友好，不需要从 HuggingFace 下载）
    clip_matcher = CLIPMatcher(model_name=CLIP_MODEL, device="cuda", use_original_clip=True)
    print()
    
    # ==================== 加载测试样本 ====================
    print(f"📂 加载测试数据: {DATASET_PATH}")
    all_samples = load_test_samples(DATASET_PATH)
    sample = random.choice(all_samples)
    
    print(f"✅ 随机抽取样本:")
    print(f"  视频路径: {sample['path']}")
    print(f"  问题: {sample['problem'][:100]}...")
    print()
    
    # ==================== 阶段1: 采样视频帧 ====================
    print("=" * 80)
    print("📹 阶段1: 采样视频帧")
    print("=" * 80)
    
    video_path = sample['path']
    video_info = get_video_info(video_path)
    print(f"视频信息:")
    print(f"  总帧数: {video_info['total_frames']}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  总时长: {video_info['duration']:.2f}秒")
    
    frames_pil, frame_ids, timestamps, fps = sample_frames(
        video_path, max_frames=MAX_FRAMES, strategy="uniform"
    )
    print(f"✅ 采样完成: {len(frames_pil)} 帧")
    print(f"  帧号范围: {frame_ids[0]} ~ {frame_ids[-1]}")
    print(f"  时间范围: {timestamps[0]:.2f}s ~ {timestamps[-1]:.2f}s")
    print()
    
    # ==================== 阶段2: 模型推理 ====================
    print("=" * 80)
    print("🤖 阶段2: 模型推理（不带时间戳）")
    print("=" * 80)
    
    generated_text, inputs = run_inference(sample, model, processor, SYSTEM_PROMPT)
    think_text = parse_think_section(generated_text)
    
    print("✅ 推理完成")
    print(f"生成长度: {len(generated_text)} 字符")
    print(f"\n【原始 <think> 内容】")
    print("-" * 80)
    print(think_text[:500] + "..." if len(think_text) > 500 else think_text)
    print("-" * 80)
    print()
    
    # ==================== 阶段3: 提取事件 ====================
    print("=" * 80)
    print("🔍 阶段3: 提取关键事件")
    print("=" * 80)
    
    # 尝试 LLM 方法
    print("尝试使用 LLM 提取事件...")
    events = extract_events(
        think_text,
        method="llm",
        model=model,
        processor=processor,
        max_events=10
    )
    
    if not events:
        print("⚠️  LLM 提取失败，使用基于规则的方法")
        events = extract_events(think_text, method="rule", max_events=10)
    
    print(f"✅ 提取到 {len(events)} 个事件:")
    for i, event in enumerate(events, 1):
        print(f"  {i}. anchor: {event.anchor[:60]}...")
        print(f"     query:  {event.query}")
    print()
    
    # ==================== 阶段4: CLIP 匹配 ====================
    print("=" * 80)
    print("🎯 阶段4: CLIP 事件-帧匹配")
    print("=" * 80)
    
    queries = events_to_queries(events)
    
    if USE_MONOTONIC_CONSTRAINT:
        print("使用单调约束 DP...")
        similarity_matrix = clip_matcher.get_similarity_matrix(queries, frames_pil)
        best_frames = match_with_monotonic_constraint(
            similarity_matrix,
            lambda_smooth=LAMBDA_SMOOTH
        )
        frame_matches = {q: f for q, f in zip(queries, best_frames)}
    else:
        print("使用独立匹配...")
        frame_matches = clip_matcher.match_events_to_frames(queries, frames_pil)
    
    print(f"✅ 匹配完成:")
    for i, (event, frame_id) in enumerate(zip(events, best_frames if USE_MONOTONIC_CONSTRAINT else [frame_matches[q] for q in queries]), 1):
        timestamp = timestamps[frame_id]
        print(f"  {i}. {event.query[:40]:<40} → Frame {frame_id:2d} ({timestamp:5.2f}s)")
    print()
    
    # ==================== 阶段5: 插入时间戳 ====================
    print("=" * 80)
    print("✏️  阶段5: 插入时间戳")
    print("=" * 80)
    
    think_with_timestamps = insert_timestamps(
        think_text,
        events,
        frame_matches,
        timestamps,
        format_style="frame_and_time"
    )
    
    # 验证插入结果
    verification = verify_insertions(think_text, think_with_timestamps, len(events))
    print(f"✅ 插入完成:")
    print(f"  预期插入: {verification['expected_count']} 个")
    print(f"  实际插入: {verification['inserted_count']} 个")
    print(f"  插入率: {verification['insertion_rate']:.1%}")
    print()
    
    print(f"【带时间戳的 <think> 内容】")
    print("=" * 80)
    print(think_with_timestamps)
    print("=" * 80)
    print()
    
    # ==================== 保存结果 ====================
    log_dir = Path("../logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = log_dir / f"timestamp_pipeline_{timestamp_str}.json"
    
    # 转换numpy类型为Python原生类型
    def convert_to_native(obj):
        """递归转换numpy类型为Python原生类型"""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    result = {
        "video_path": video_path,
        "question": sample['problem'],
        "video_info": convert_to_native(video_info),
        "num_frames_sampled": len(frames_pil),
        "num_events_extracted": len(events),
        "events": [e.to_dict() for e in events],
        "frame_matches": {e.query: int(frame_matches[e.query]) for e in events},
        "original_think": think_text,
        "think_with_timestamps": think_with_timestamps,
        "verification": verification,
        "config": {
            "max_frames": MAX_FRAMES,
            "clip_model": CLIP_MODEL,
            "use_monotonic_constraint": USE_MONOTONIC_CONSTRAINT,
            "lambda_smooth": LAMBDA_SMOOTH
        }
    }
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"💾 结果已保存到: {result_file}")
    print()
    
    # ==================== 总结 ====================
    print("=" * 80)
    print("✅ Pipeline 执行完成")
    print("=" * 80)
    print(f"✓ 采样帧数: {len(frames_pil)}")
    print(f"✓ 提取事件: {len(events)}")
    print(f"✓ 时间戳插入率: {verification['insertion_rate']:.1%}")
    print()
    print("💡 下一步:")
    print("  1. 检查时间戳是否合理（是否符合视频内容）")
    print("  2. 调整参数（max_frames, lambda_smooth）优化效果")
    print("  3. 在更多样本上测试泛化性能")
    print()


if __name__ == "__main__":
    main()
