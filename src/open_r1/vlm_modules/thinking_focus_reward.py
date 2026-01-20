"""
Thinking Focus Reward: 评估thinking是否聚焦正确答案

Week 1实现：关键词匹配法
- 统计正确答案关键词 vs 错误答案关键词
- 聚焦正确答案 → 高分
- 模糊不清/偏向错误答案 → 低分

Week 2-3可选升级：API评估（更精确）
"""

import re
from typing import List, Dict, Any
import torch


def extract_keywords(text: str, min_length: int = 2) -> List[str]:
    """提取关键词"""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        '的', '了', '是', '在', '有', '和', '就', '不', '人', '都', '一',
        '我', '他', '她', '们', '这', '那', '与', '或', '但', '而',
    }
    
    return [w for w in words if len(w) >= min_length and w not in stop_words]


def extract_answer_text(option: str) -> str:
    """从选项中提取答案文本（去除A. B.等前缀）"""
    return re.sub(r'^[A-E][\.．、]\s*', '', option.strip())


def count_keywords_in_text(text: str, keywords: List[str]) -> int:
    """统计关键词在文本中出现的次数"""
    text_lower = text.lower()
    return sum(text_lower.count(kw.lower()) for kw in keywords)


def thinking_focus_reward(completions, question=None, options=None, solution=None, **kwargs):
    """
    计算thinking聚焦度reward
    
    Args:
        completions: 生成的文本列表
        question: 问题文本（可选）
        options: 选项列表 ["A. ...", "B. ...", ...]
        solution: 正确答案（字母，如"A"）
        
    Returns:
        reward tensor
    """
    if not options or not solution:
        # 缺少必要信息，返回中性分
        return torch.tensor([0.5] * len(completions), dtype=torch.float32)
    
    # 处理solution（可能是列表）
    if isinstance(solution, list):
        if len(solution) == 0:
            return torch.tensor([0.5] * len(completions), dtype=torch.float32)
        solution_letter = solution[0] if isinstance(solution[0], str) else str(solution[0])
    else:
        solution_letter = str(solution)
    
    solution_letter = solution_letter.strip().upper()
    
    # 提取首字母
    if len(solution_letter) > 0 and solution_letter[0].isalpha():
        solution_letter = solution_letter[0]
    
    solution_idx = ord(solution_letter) - ord('A')
    
    if solution_idx < 0 or solution_idx >= len(options):
        return torch.tensor([0.5] * len(completions), dtype=torch.float32)
    
    # 提取正确和错误选项的关键词
    correct_option = options[solution_idx]
    wrong_options = [opt for i, opt in enumerate(options) if i != solution_idx]
    
    correct_text = extract_answer_text(correct_option)
    correct_keywords = list(set(extract_keywords(correct_text)))
    
    wrong_keywords = []
    for opt in wrong_options:
        wrong_text = extract_answer_text(opt)
        wrong_keywords.extend(extract_keywords(wrong_text))
    wrong_keywords = list(set(wrong_keywords))
    
    # 移除重叠关键词
    overlap = set(correct_keywords) & set(wrong_keywords)
    correct_keywords = [kw for kw in correct_keywords if kw not in overlap]
    wrong_keywords = [kw for kw in wrong_keywords if kw not in overlap]
    
    rewards = []
    
    for completion in completions:
        # 提取生成文本
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
        
        # 提取<think>标签内容
        think_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL | re.IGNORECASE)
        thinking_text = think_match.group(1) if think_match else generated_text
        
        # 统计关键词
        correct_count = count_keywords_in_text(thinking_text, correct_keywords)
        wrong_count = count_keywords_in_text(thinking_text, wrong_keywords)
        
        # 计算reward
        if correct_count == 0 and wrong_count == 0:
            reward = 0.3  # 通用推理，中性分
        elif correct_count > wrong_count * 1.5:
            reward = 1.0  # 明显聚焦正确答案
        elif correct_count > wrong_count:
            reward = 0.7  # 略微偏向正确答案
        elif correct_count == wrong_count:
            reward = 0.3  # 中性
        else:
            reward = 0.0  # 偏向错误答案
        
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float32)


if __name__ == "__main__":
    # 测试
    print("测试thinking_focus_reward...")
    
    test_options = [
        "A. Romantic affection",
        "B. Friendship",
        "C. Professional relationship"
    ]
    test_solution = "A"
    
    test_completions = [
        "<think>男士展现了romantic affection，他的眼神很gentle</think><answer>A</answer>",
        "<think>可能是friendship，也可能是romantic关系</think><answer>A</answer>",
        "<think>这是professional的互动</think><answer>A</answer>",
    ]
    
    rewards = thinking_focus_reward(
        test_completions,
        options=test_options,
        solution=test_solution
    )
    
    print(f"\nThinking聚焦度rewards: {rewards}")
