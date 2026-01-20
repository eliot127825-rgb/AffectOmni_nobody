"""
Outcome Reward: 基于held-out验证集的结果导向奖励

Week 1实现：直接优化答题准确率
- 使用Daily-Omni held-out验证集（239样本）
- 答对=1.0，答错=0.0
- 避免过拟合测试集
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
import torch

# 全局缓存held-out数据集
_HELD_OUT_INDEX = None


def load_held_out_index():
    """加载held-out验证集索引"""
    global _HELD_OUT_INDEX
    
    if _HELD_OUT_INDEX is not None:
        return _HELD_OUT_INDEX
    
    held_out_path = Path(__file__).parent.parent.parent.parent / "data" / "outcome_reward_data" / "daily_held_out.json"
    
    if not held_out_path.exists():
        print(f"⚠️  held-out验证集不存在: {held_out_path}")
        print(f"   outcome_reward将返回0.0")
        _HELD_OUT_INDEX = {}
        return _HELD_OUT_INDEX
    
    with open(held_out_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 构建索引：video_name -> ground_truth
    _HELD_OUT_INDEX = {}
    for item in data:
        video_path = item.get('path', item.get('video_path', ''))
        if video_path:
            video_name = Path(video_path).name
            ground_truth = item.get('final_answer', item.get('answer', ''))
            _HELD_OUT_INDEX[video_name] = ground_truth.strip().upper()
    
    print(f"✓ 加载held-out验证集索引: {len(_HELD_OUT_INDEX)}条记录")
    
    return _HELD_OUT_INDEX


def extract_answer(text: str) -> str:
    """从生成文本中提取答案"""
    # 方法1: <answer>标签
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # 提取首字母
        if len(answer) > 0 and answer[0].isalpha():
            return answer[0].upper()
        return answer.upper()
    
    # 方法2: 常见答案模式
    patterns = [
        r'(?:答案是|选择|answer is)\s*[:：]?\s*([A-E])',
        r'(?:最终答案|final answer)\s*[:：]?\s*([A-E])',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 方法3: 最后一个单独的大写字母
    letters = re.findall(r'\b([A-E])\b', text)
    if letters:
        return letters[-1].upper()
    
    return ""


def outcome_reward(completions, solution=None, **kwargs):
    """
    计算outcome reward
    
    Args:
        completions: 生成的文本列表
        solution: ground truth答案列表（优先使用）
        **kwargs: 可能包含video_paths等信息
        
    Returns:
        reward tensor
    """
    held_out_index = load_held_out_index()
    
    rewards = []
    
    for idx, completion in enumerate(completions):
        # 提取生成的文本
        if isinstance(completion, str):
            generated_text = completion
        elif isinstance(completion, dict):
            generated_text = completion.get('content', str(completion))
        elif isinstance(completion, list) and len(completion) > 0:
            if isinstance(completion[0], dict):
                generated_text = completion[0].get('content', '')
            else:
                generated_text = str(completion[0])
        else:
            generated_text = str(completion)
        
        # 提取预测答案
        predicted_answer = extract_answer(generated_text)
        
        # 获取ground truth
        ground_truth = ""
        
        # 优先使用solution参数
        if solution and idx < len(solution):
            gt = solution[idx]
            if isinstance(gt, str):
                ground_truth = gt.strip().upper()
                # 提取首字母
                if len(ground_truth) > 0 and ground_truth[0].isalpha():
                    ground_truth = ground_truth[0]
        
        # 如果没有solution，尝试从held-out索引查找
        if not ground_truth and held_out_index:
            video_paths = kwargs.get('video_paths', [])
            if video_paths and idx < len(video_paths):
                video_name = Path(video_paths[idx]).name
                ground_truth = held_out_index.get(video_name, '')
        
        # 计算reward
        if predicted_answer and ground_truth:
            reward = 1.0 if predicted_answer == ground_truth else 0.0
        else:
            # 无法判断，给0分（保守策略）
            reward = 0.0
        
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float32)


if __name__ == "__main__":
    # 测试
    print("测试outcome_reward...")
    
    index = load_held_out_index()
    print(f"\n索引包含{len(index)}条记录")
    
    if index:
        print("\n前5条:")
        for i, (video, answer) in enumerate(list(index.items())[:5]):
            print(f"  {video} -> {answer}")
    
    # 测试reward计算
    test_completions = [
        "<think>分析...</think><answer>A</answer>",
        "<think>推理...</think><answer>B. 选项B</answer>",
        "没有明确答案",
    ]
    test_solution = ["A", "B", "C"]
    
    rewards = outcome_reward(test_completions, solution=test_solution)
    print(f"\n测试rewards: {rewards}")
