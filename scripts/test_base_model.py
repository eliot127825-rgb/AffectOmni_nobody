#!/usr/bin/env python3
"""
简单的Stage 1模型评估脚本
随机抽取训练样本，测试模型输出
"""

import os
import sys
import json
import yaml
import random
import torch
import re
from pathlib import Path
from datetime import datetime

# 添加src路径到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "src"))

from transformers import AutoProcessor, Qwen2_5OmniThinkerForConditionalGeneration
from open_r1.vlm_modules.qwenomni_module import QwenOmniModule
from qwen_omni_utils import process_mm_info


# 设置随机种子
random.seed(42)
torch.manual_seed(42)


def load_dataset(yaml_path):
    """加载训练数据集"""
    print(f"📂 加载数据集: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
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
    """创建对话消息（和训练时一样的格式）
    
    Args:
        sample: 样本数据
        system_prompt: 系统提示
        timestamp_info: 可选的时间戳信息字符串，如果提供则添加到用户消息中
    """
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


def extract_tags(text):
    """提取<context>, <think>, <answer>标签内容"""
    context_match = re.search(r'<context>(.*?)</context>', text, re.DOTALL)
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    
    return {
        'context': context_match.group(1).strip() if context_match else None,
        'think': think_match.group(1).strip() if think_match else None,
        'answer': answer_match.group(1).strip() if answer_match else None
    }


def check_format(generated_text):
    """检查输出格式是否正确"""
    has_context = '<context>' in generated_text and '</context>' in generated_text
    has_think = '<think>' in generated_text and '</think>' in generated_text
    has_answer = '<answer>' in generated_text and '</answer>' in generated_text
    
    return {
        'has_context': has_context,
        'has_think': has_think,
        'has_answer': has_answer,
        'all_correct': has_context and has_think and has_answer
    }


def main():
    print("=" * 80)
    print("🧪 HumanOmniV2 基座模型测试脚本")
    print("=" * 80)
    print()
    
    # ==================== 配置 ====================
    MODEL_PATH = "${PROJECT_ROOT}/models/HumanOmniV2"  # 训练好的HumanOmniV2模型
    BASE_MODEL_PATH = "${PROJECT_ROOT}/Qwen2.5-Omni-7B-Thinker"  # 基座模型（用于加载processor）
    DATASET_PATH = "../configs/test_samples.yaml"  # 测试数据集配置
    
    SYSTEM_PROMPT = """You are a helpful assistant. Your primary goal is to deeply analyze and interpret information from available various modalities (image, video, audio, text context) to answer questions with human-like depth and a clear, traceable thought process.

Begin by thoroughly understanding the image, video, audio or other available context information, and then proceed with an in-depth analysis related to the question. 

When analyzing videos, YOU MUST reference specific frame numbers and timestamps for key events and observations.
Format: "observation [Frame N: T.XXs]"

Examples of correct temporal references:
- The woman picks up the rose [Frame 3: 3.00s]
- She smiles at the man [Frame 5: 5.00s]
- The man receives the rose [Frame 12: 12.00s]

Pay special attention to the temporal progression of events. Always connect your visual observations to their corresponding frame numbers and timestamps.

In reasoning, It is encouraged to incorporate self-reflection and verification into your reasoning process. You are encouraged to review the image, video, audio, or other context information to ensure the answer accuracy.

Provide your understanding of the image, video, and audio between the <context> </context> tags, detail the reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags.
"""
    
    # ==================== 加载模型 ====================
    print("🔧 加载模型...")
    print(f"  模型权重路径: {MODEL_PATH}")
    print(f"  Processor路径: {BASE_MODEL_PATH}")
    
    # Processor从基座模型加载（因为训练时只保存了模型权重）
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    # 覆盖全局配置（防御措施3）
    if hasattr(processor, 'image_processor'):
        processor.image_processor.max_pixels = 6422528
        processor.image_processor.min_pixels = 3136
    
    # 模型权重从训练输出加载
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print("✅ 模型加载成功！")
    print(f"  设备: {model.device}")
    print()
    
    # ==================== 加载数据 ====================
    all_samples = load_dataset(DATASET_PATH)
    
    # ==================== 随机抽取样本 ====================
    sample = random.choice(all_samples)
    
    print("🎲 随机抽取的样本:")
    print(f"  问题类型: {sample.get('problem_type', 'unknown')}")
    print(f"  数据类型: {sample.get('data_type', 'unknown')}")
    print(f"  文件路径: {sample.get('path', 'unknown')}")
    print(f"  问题: {format_question(sample)[:200]}...")
    print()
    
    # ==================== 第一步：先处理视频获取时间戳信息 ====================
    print("📝 第一步：预处理视频获取时间戳...")
    
    # 先用不包含时间戳的messages处理一次，获取实际的帧数和时间间隔
    temp_messages = create_messages(sample, SYSTEM_PROMPT, timestamp_info=None)
    temp_texts = processor.apply_chat_template(
        [temp_messages],
        tokenize=False,
        add_generation_prompt=True
    )
    temp_text = temp_texts[0]
    
    # 处理多模态输入
    audios, images, videos = process_mm_info(temp_messages, use_audio_in_video=False)
    
    # 临时处理获取时间戳信息
    temp_inputs = processor(
        text=[temp_text],
        images=images,
        videos=videos,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32768
    )
    
    # 提取时间戳信息
    timestamp_info_str = None
    if 'video_grid_thw' in temp_inputs and temp_inputs['video_grid_thw'] is not None:
        video_grid = temp_inputs['video_grid_thw']
        num_frames = video_grid[0][0].item()
        
        if 'video_second_per_grid' in temp_inputs and temp_inputs['video_second_per_grid'] is not None:
            second_per_grid = temp_inputs['video_second_per_grid']
            
            # 获取时间间隔
            if second_per_grid.dim() == 0:
                interval = second_per_grid.item()
            elif second_per_grid.dim() == 1 and len(second_per_grid) == 1:
                interval = second_per_grid[0].item()
            else:
                interval = second_per_grid.flatten()[0].item()
            
            # 计算每帧的时间戳
            frame_timestamps = [i * interval for i in range(num_frames)]
            
            # 构造时间戳信息字符串
            timestamp_info_str = "[Video Frame Information]\n"
            timestamp_info_str += f"This video has been sampled into {num_frames} frames at {interval:.2f}-second intervals.\n"
            timestamp_info_str += "Available frame timestamps:\n"
            timestamp_info_str += ", ".join([f"Frame {i}: {ts:.2f}s" for i, ts in enumerate(frame_timestamps)])
            timestamp_info_str += "\n\n"
            timestamp_info_str += "IMPORTANT: In your <think> section, you MUST reference specific frame numbers for each key event or observation.\n"
            timestamp_info_str += "Use the exact format: \"your observation [Frame N: T.XXs]\"\n"
            timestamp_info_str += "Example: The woman smiles [Frame 5: 5.00s], indicating happiness."
            
            print(f"  ✅ 提取到 {num_frames} 帧，时间间隔 {interval:.2f}秒/帧")
    
    # ==================== 第二步：用时间戳信息重新构造完整输入 ====================
    print("📝 第二步：构造包含时间戳的完整输入...")
    messages = create_messages(sample, SYSTEM_PROMPT, timestamp_info=timestamp_info_str)
    
    # 应用chat template
    texts = processor.apply_chat_template(
        [messages],
        tokenize=False,
        add_generation_prompt=True
    )
    text = texts[0]
    
    # 重新处理多模态输入（使用相同的数据）
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    
    # 调试：显示提取的多模态数据
    print(f"  多模态数据提取:")
    print(f"     - 音频数据: {len(audios) if audios else 0} 个")
    print(f"     - 图像数据: {len(images) if images else 0} 个")
    print(f"     - 视频数据: {len(videos) if videos else 0} 个")
    
    # 读取视频的实际总时长
    video_duration = None
    if videos and len(videos) > 0:
        video_path = sample['path']
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if fps > 0:
                    video_duration = frame_count / fps
                cap.release()
        except Exception as e:
            print(f"⚠️  无法读取视频时长: {e}")
    
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=True,  # 防御措施2：truncation保护
        max_length=32768
    )
    
    inputs = inputs.to(model.device)
    
    # 防御措施1：实测断言 + 调试信息
    seq_len = inputs['input_ids'].shape[1]
    print(f"✅ 输入准备完成")
    print(f"  输入token数: {seq_len}")
    
    # 调试：打印视频像素数据的实际大小
    if 'pixel_values_videos' in inputs and inputs['pixel_values_videos'] is not None:
        vid_shape = inputs['pixel_values_videos'].shape
        vid_size_gb = inputs['pixel_values_videos'].element_size() * inputs['pixel_values_videos'].nelement() / (1024**3)
        print(f"  视频像素数据shape: {vid_shape}")
        print(f"  视频像素数据大小: {vid_size_gb:.2f} GB")
    
    # 打印视频帧数和时间戳信息
    if 'video_grid_thw' in inputs and inputs['video_grid_thw'] is not None:
        video_grid = inputs['video_grid_thw']
        num_frames = video_grid[0][0].item()  # T维度就是帧数
        print(f"  📹 视频分析信息:")
        if video_duration is not None:
            print(f"     - 视频总时长: {video_duration:.2f}秒")
        print(f"     - 采样帧数: {num_frames} 帧")
        print(f"     - 网格维度 (T×H×W): {video_grid[0][0].item()}×{video_grid[0][1].item()}×{video_grid[0][2].item()}")
        
        # 打印每帧的时间戳
        if 'video_second_per_grid' in inputs and inputs['video_second_per_grid'] is not None:
            second_per_grid = inputs['video_second_per_grid']
            
            # video_second_per_grid 是每个时间网格的秒数（间隔），不是时间戳列表
            if second_per_grid.dim() == 0:
                interval = second_per_grid.item()
            elif second_per_grid.dim() == 1 and len(second_per_grid) == 1:
                interval = second_per_grid[0].item()
            else:
                # 如果是多个值，取第一个
                interval = second_per_grid.flatten()[0].item()
            
            # 根据帧数和时间间隔计算每帧的时间戳
            frame_timestamps = [i * interval for i in range(num_frames)]
            
            print(f"     - 时间间隔: {interval:.2f}秒/帧")
            print(f"     - 采样覆盖范围: {frame_timestamps[0]:.2f}秒 ~ {frame_timestamps[-1]:.2f}秒")
            print(f"     - 采样跨度: {frame_timestamps[-1] - frame_timestamps[0]:.2f}秒")
            
            # 显示所有帧的时间戳
            timestamps_str = [f'{t:.2f}s' for t in frame_timestamps]
            print(f"     - 各帧时间戳 ({num_frames}帧): {timestamps_str}")
    
    # 打印音频信息
    if 'input_features' in inputs and inputs['input_features'] is not None:
        audio_features = inputs['input_features']
        print(f"  🎵 音频分析信息:")
        print(f"     - 音频特征shape: {audio_features.shape}")
        
        if 'audio_feature_lengths' in inputs and inputs['audio_feature_lengths'] is not None:
            audio_lengths = inputs['audio_feature_lengths']
            print(f"     - 音频特征长度: {audio_lengths}")
            # 音频采样率通常是16kHz，每个特征对应一定时长
            # Qwen2.5-Omni的音频处理：每秒约50个特征帧
            if audio_lengths.numel() > 0:
                total_audio_frames = audio_lengths[0].item() if audio_lengths.dim() > 0 else audio_lengths.item()
                # 假设每秒50个音频特征帧（这是Whisper等模型的常见设置）
                audio_duration_estimate = total_audio_frames / 50.0
                print(f"     - 音频时长估计: {audio_duration_estimate:.2f}秒 (基于特征帧数)")
    
    if seq_len > 32768:
        raise AssertionError(f"序列太长: {seq_len} > 32768")
    print()
    
    # ==================== 生成输出 ====================
    print("🤖 开始生成输出...")
    print("-" * 80)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,  # 贪婪解码，确保结果稳定
            temperature=1.0,
            top_p=0.9
        )
    
    # 只取生成的部分（去掉输入）
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    generated_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    print("生成完成！")
    print("-" * 80)
    print()
    
    # ==================== 分析输出 ====================
    print("📊 输出分析:")
    print("=" * 80)
    
    # 1. 格式检查
    format_check = check_format(generated_text)
    print("\n【格式检查】")
    print(f"  ✓ 包含 <context>: {'✅' if format_check['has_context'] else '❌'}")
    print(f"  ✓ 包含 <think>:   {'✅' if format_check['has_think'] else '❌'}")
    print(f"  ✓ 包含 <answer>:  {'✅' if format_check['has_answer'] else '❌'}")
    print(f"  ✓ 格式完整:       {'✅ 正确' if format_check['all_correct'] else '❌ 错误'}")
    
    # 2. 提取内容
    extracted = extract_tags(generated_text)
    
    print("\n【生成内容】")
    if extracted['context']:
        print(f"\n<context>")
        print(f"{extracted['context'][:300]}...")
        print(f"</context>")
    
    if extracted['think']:
        print(f"\n<think>")
        print(f"{extracted['think'][:300]}...")
        print(f"</think>")
    
    if extracted['answer']:
        print(f"\n<answer>")
        print(f"{extracted['answer']}")
        print(f"</answer>")
    
    # 3. 与标准答案对比
    ground_truth_solution = sample.get('solution', '')
    ground_truth_answer = sample.get('answer', '')
    
    print("\n【标准答案对比】")
    print(f"  Ground Truth Answer: {ground_truth_answer}")
    if extracted['answer']:
        print(f"  Generated Answer:    {extracted['answer']}")
        
        # 简单的答案匹配
        if extracted['answer'].strip() == ground_truth_answer.strip():
            print(f"  匹配结果: ✅ 完全匹配")
        elif ground_truth_answer.strip() in extracted['answer'].strip():
            print(f"  匹配结果: ⚠️ 部分匹配")
        else:
            print(f"  匹配结果: ❌ 不匹配")
    else:
        print(f"  Generated Answer:    ❌ 未能提取")
    
    # 4. 完整输出
    print("\n【完整生成文本】")
    print("-" * 80)
    print(generated_text)
    print("-" * 80)
    
    # ==================== 保存结果 ====================
    # 创建logs目录
    log_dir = Path("../logs")
    log_dir.mkdir(exist_ok=True)
    
    # 生成时间戳文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = log_dir / f"basemodel_test_result_{timestamp}.json"
    
    # 准备保存的数据
    test_result = {
        "timestamp": timestamp,
        "model_path": MODEL_PATH,
        "sample_info": {
            "problem_type": sample.get('problem_type', 'unknown'),
            "data_type": sample.get('data_type', 'unknown'),
            "video_path": sample.get('path', 'unknown'),
            "question": format_question(sample),
            "ground_truth_answer": ground_truth_answer
        },
        "generated_output": {
            "full_text": generated_text,
            "context": extracted.get('context', ''),
            "think": extracted.get('think', ''),
            "answer": extracted.get('answer', '')
        },
        "evaluation": {
            "format_check": format_check,
            "answer_match": extracted['answer'].strip() == ground_truth_answer.strip() if extracted['answer'] else False
        }
    }
    
    # 保存到JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 测试结果已保存到: {output_file}")
    
    # ==================== 总结 ====================
    print("\n" + "=" * 80)
    print("✅ 评估完成！")
    print("=" * 80)
    
    print("\n💡 提示:")
    print("  - 如果格式正确，说明模型学会了输出结构")
    print("  - 如果答案匹配，说明模型理解了任务")
    print("  - 可以多次运行脚本测试不同样本")
    print()


if __name__ == "__main__":
    main()
