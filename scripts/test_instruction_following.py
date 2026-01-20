#!/usr/bin/env python3
"""
指令遵循能力测试脚本
对同一个视频使用不同的prompt，测试模型是否真的遵循指令
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
from qwen_omni_utils import process_mm_info

# 设置随机种子
random.seed(42)
torch.manual_seed(42)


# ==================== 测试用的不同指令（3种代表性测试）====================
TEST_PROMPTS = {
    "count_3_points": {
        "name": "指定数量：3个要点",
        "instruction": "Please summarize this video in exactly 3 key points.",
        "expected": "应该输出3个要点"
    },
    
    "focus_people": {
        "name": "指定关注点：人物",
        "instruction": "Describe the people in this video, focusing on their actions, expressions, and interactions.",
        "expected": "应该详细描述人物"
    },
    
    "one_sentence": {
        "name": "指定长度：一句话",
        "instruction": "Describe this video in one single sentence.",
        "expected": "应该只有一句话"
    },
}


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


def create_messages_with_custom_prompt(sample, custom_instruction, system_prompt):
    """创建自定义指令的对话消息"""
    
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
                    "max_frames": 32,
                    "max_pixels": 602112
                },
                {
                    "type": "text",
                    "text": custom_instruction
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


def count_sentences(text):
    """统计句子数量"""
    if not text:
        return 0
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def count_list_items(text):
    """统计列表项数量"""
    if not text:
        return 0
    # 匹配编号列表 (1. 2. 3.) 或 项目符号 (- * •)
    patterns = [
        r'^\d+\.',  # 1. 2. 3.
        r'^[-*•]',  # - * •
    ]
    
    lines = text.split('\n')
    count = 0
    for line in lines:
        line = line.strip()
        for pattern in patterns:
            if re.match(pattern, line):
                count += 1
                break
    return count


def analyze_output(generated_text, prompt_config):
    """分析输出是否符合指令"""
    extracted = extract_tags(generated_text)
    analysis = {
        "has_context": extracted['context'] is not None,
        "has_think": extracted['think'] is not None,
        "has_answer": extracted['answer'] is not None,
    }
    
    answer_text = extracted.get('answer', '')
    
    # 根据不同的prompt类型进行分析
    prompt_key = prompt_config.get('key', '')
    
    if 'count_3' in prompt_key:
        list_count = count_list_items(answer_text)
        analysis['list_items'] = list_count
        analysis['follows_instruction'] = (list_count == 3)
        analysis['note'] = f"要求3个要点，实际{list_count}个"
    
    elif 'count_5' in prompt_key:
        list_count = count_list_items(answer_text)
        analysis['list_items'] = list_count
        analysis['follows_instruction'] = (list_count == 5)
        analysis['note'] = f"要求5个观察点，实际{list_count}个"
    
    elif 'one_sentence' in prompt_key:
        sent_count = count_sentences(answer_text)
        analysis['sentence_count'] = sent_count
        analysis['follows_instruction'] = (sent_count == 1)
        analysis['note'] = f"要求1句话，实际{sent_count}句"
    
    elif 'focus_people' in prompt_key:
        people_keywords = ['person', 'people', 'man', 'woman', 'he', 'she', 'they', 'facial', 'expression', 'gesture', 'interaction']
        keyword_count = sum(1 for kw in people_keywords if kw in answer_text.lower())
        analysis['people_keyword_count'] = keyword_count
        analysis['follows_instruction'] = (keyword_count >= 5)
        analysis['note'] = f"人物相关关键词出现{keyword_count}次"
    
    elif 'focus_environment' in prompt_key:
        env_keywords = ['background', 'setting', 'location', 'environment', 'room', 'outdoor', 'indoor', 'place']
        people_keywords = ['person', 'people', 'man', 'woman']
        env_count = sum(1 for kw in env_keywords if kw in answer_text.lower())
        people_count = sum(1 for kw in people_keywords if kw in answer_text.lower())
        analysis['environment_keyword_count'] = env_count
        analysis['people_keyword_count'] = people_count
        analysis['follows_instruction'] = (env_count > people_count)
        analysis['note'] = f"环境词{env_count}次 vs 人物词{people_count}次"
    
    elif 'timeline' in prompt_key:
        timeline_keywords = ['first', 'then', 'next', 'after', 'finally', 'initially', 'subsequently']
        keyword_count = sum(1 for kw in timeline_keywords if kw in answer_text.lower())
        analysis['timeline_keyword_count'] = keyword_count
        analysis['follows_instruction'] = (keyword_count >= 3)
        analysis['note'] = f"时间顺序词出现{keyword_count}次"
    
    else:
        analysis['follows_instruction'] = None
        analysis['note'] = "通用回答，无特定要求"
    
    return analysis, extracted


def generate_with_prompt(model, processor, sample, prompt_config, system_prompt):
    """使用特定prompt生成输出（与test_base_model.py保持一致的处理流程）"""
    
    # 创建消息
    messages = create_messages_with_custom_prompt(
        sample, 
        prompt_config['instruction'],
        system_prompt
    )
    
    # 应用chat template
    texts = processor.apply_chat_template(
        [messages],
        tokenize=False,
        add_generation_prompt=True
    )
    text = texts[0]
    
    # 处理多模态输入（与test_base_model保持一致：use_audio_in_video=False）
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        audio=audios,
        return_tensors="pt",
        padding=True,
        truncation=True,  # 与test_base_model保持一致
        max_length=32768
    )
    
    inputs = inputs.to(model.device)
    
    # 检查序列长度（防御措施）
    seq_len = inputs['input_ids'].shape[1]
    if seq_len > 32768:
        raise AssertionError(f"序列太长: {seq_len} > 32768")
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,  # 贪婪解码，确保结果稳定
            temperature=1.0,
            top_p=0.9
        )
    
    # 只取生成的部分
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    generated_text = processor.batch_decode(
        generated_ids, 
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    return generated_text


def main():
    print("=" * 80)
    print("🧪 HumanOmniV2 指令遵循能力测试")
    print("=" * 80)
    print()
    
    # ==================== 配置 ====================
    MODEL_PATH = "${PROJECT_ROOT}/models/HumanOmniV2"
    BASE_MODEL_PATH = "${PROJECT_ROOT}/Qwen2.5-Omni-7B-Thinker"
    DATASET_PATH = "../configs/test_samples.yaml"
    
    # 使用与test_base_model相同的system prompt
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
    
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    if hasattr(processor, 'image_processor'):
        processor.image_processor.max_pixels = 6422528
        processor.image_processor.min_pixels = 3136
    
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    print("✅ 模型加载成功！")
    print()
    
    # ==================== 加载数据 ====================
    all_samples = load_dataset(DATASET_PATH)
    
    # 随机选择一个视频样本
    sample = random.choice(all_samples)
    
    print("🎲 测试样本:")
    print(f"  视频路径: {sample.get('path', 'unknown')}")
    print(f"  数据类型: {sample.get('data_type', 'unknown')}")
    print()
    
    # ==================== 测试不同的指令 ====================
    print("=" * 80)
    print("开始测试不同指令...")
    print("=" * 80)
    print()
    
    results = {}
    
    for prompt_key, prompt_config in TEST_PROMPTS.items():
        print(f"\n{'='*80}")
        print(f"🔹 测试 {prompt_config['name']}")
        print(f"{'='*80}")
        print(f"指令: {prompt_config['instruction']}")
        print(f"预期: {prompt_config['expected']}")
        print()
        
        # 生成输出
        print("⏳ 生成中...")
        generated_text = generate_with_prompt(
            model, processor, sample, prompt_config, SYSTEM_PROMPT
        )
        
        # 分析输出
        prompt_config['key'] = prompt_key
        analysis, extracted = analyze_output(generated_text, prompt_config)
        
        # 显示结果
        print("✅ 生成完成")
        print()
        print("【分析结果】")
        if analysis.get('follows_instruction') is not None:
            status = "✅ 遵循" if analysis['follows_instruction'] else "❌ 未遵循"
            print(f"  指令遵循: {status}")
        print(f"  说明: {analysis['note']}")
        
        print()
        print("【生成的答案】")
        answer = extracted.get('answer', '无')
        print(f"{answer[:500]}{'...' if len(answer) > 500 else ''}")
        print()
        
        # 保存结果
        results[prompt_key] = {
            "prompt": prompt_config,
            "generated_text": generated_text,
            "extracted": extracted,
            "analysis": analysis
        }
    
    # ==================== 总结对比 ====================
    print("\n" + "=" * 80)
    print("📊 指令遵循能力总结")
    print("=" * 80)
    print()
    
    follow_count = 0
    total_testable = 0
    
    print(f"{'指令类型':<25} {'遵循状态':<15} {'详细说明'}")
    print("-" * 80)
    
    for prompt_key, result in results.items():
        name = result['prompt']['name']
        follows = result['analysis'].get('follows_instruction')
        note = result['analysis'].get('note', '')
        
        if follows is not None:
            total_testable += 1
            if follows:
                follow_count += 1
                status = "✅ 遵循"
            else:
                status = "❌ 未遵循"
        else:
            status = "➖ 无法判断"
        
        print(f"{name:<25} {status:<15} {note}")
    
    print("-" * 80)
    if total_testable > 0:
        follow_rate = (follow_count / total_testable) * 100
        print(f"\n📈 指令遵循率: {follow_count}/{total_testable} ({follow_rate:.1f}%)")
    
    # ==================== 保存结果 ====================
    log_dir = Path("../logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = log_dir / f"instruction_following_test_{timestamp}.json"
    
    test_result = {
        "timestamp": timestamp,
        "model_path": MODEL_PATH,
        "video_path": sample.get('path', 'unknown'),
        "results": results,
        "summary": {
            "total_prompts": len(TEST_PROMPTS),
            "testable_prompts": total_testable,
            "followed_prompts": follow_count,
            "follow_rate": f"{follow_rate:.1f}%" if total_testable > 0 else "N/A"
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 测试结果已保存到: {output_file}")
    
    # ==================== 结论 ====================
    print("\n" + "=" * 80)
    print("🎯 测试结论")
    print("=" * 80)
    print()
    
    if total_testable == 0:
        print("⚠️  无法评估指令遵循能力（所有测试都无法判断）")
    elif follow_rate >= 80:
        print("✅ 指令遵循能力 **强**")
        print("   模型能够很好地理解和执行不同类型的指令")
    elif follow_rate >= 50:
        print("⚠️  指令遵循能力 **中等**")
        print("   模型能理解部分指令，但执行不够精确")
    else:
        print("❌ 指令遵循能力 **弱**")
        print("   模型难以遵循具体的指令要求")
        print("   可能过拟合到特定的问答格式")
    
    print()


if __name__ == "__main__":
    main()
