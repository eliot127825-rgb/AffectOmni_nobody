"""
事件抽取模块
从模型的 <think> 输出中提取关键事件
"""

import re
import json
from typing import List, Dict, Optional
import warnings


class Event:
    """事件数据结构"""
    def __init__(self, anchor: str, query: str):
        """
        Args:
            anchor: 用于在原文中定位的锚点（尽量是原句/片段）
            query: 用于 CLIP 匹配的查询短语（更短更"图像化"）
        """
        self.anchor = anchor
        self.query = query
    
    def __repr__(self):
        return f"Event(anchor='{self.anchor}', query='{self.query}')"
    
    def to_dict(self):
        return {"anchor": self.anchor, "query": self.query}


def extract_events_with_llm(
    think_text: str,
    model,
    processor,
    max_events: int = 10
) -> List[Event]:
    """
    使用 LLM 从 think 中提取事件（推荐路线）
    
    Args:
        think_text: 原始 <think> 文本
        model: 语言模型（HumanOmniV2 或其他）
        processor: 模型的 processor
        max_events: 最多提取的事件数
    
    Returns:
        events: Event 对象列表
    
    策略：
        让模型输出"事件要点列表"，而不是时间戳
        这不改变模型的写作范式，成功率高
    """
    prompt = f"""Based on the following reasoning text, extract key visual events as short phrases.

Reasoning text:
{think_text}

Requirements:
1. Extract {max_events} or fewer key visual events that can be observed in video frames
2. For each event, provide:
   - anchor: The original phrase from the reasoning text (keep it as close to original as possible)
   - query: A short, visual description for image matching (3-8 words, noun-focused)
3. Events should be in chronological order as they appear in the reasoning
4. Output ONLY valid JSON in this format:

{{"events": [
  {{"anchor": "the woman picks up the rose", "query": "woman picking up red rose"}},
  {{"anchor": "she smiles at the man", "query": "woman smiling at man"}}
]}}

JSON output:"""

    # 构造消息
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant that extracts key visual events from text."}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }
    ]
    
    # 应用 chat template
    try:
        texts = processor.apply_chat_template(
            [messages],
            tokenize=False,
            add_generation_prompt=True
        )
        text = texts[0]
        
        # 处理输入
        inputs = processor(
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # 生成
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=False
            )
        
        # 解码
        generated_text = processor.batch_decode(
            outputs[:, inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )[0]
        
        # 解析 JSON
        events = _parse_events_json(generated_text)
        return events[:max_events]
        
    except Exception as e:
        warnings.warn(f"LLM extraction failed: {e}, falling back to rule-based")
        return extract_events_rule_based(think_text, max_events)


def extract_events_rule_based(
    think_text: str,
    max_events: int = 10
) -> List[Event]:
    """
    基于规则的事件提取（Fallback）
    
    策略：
        1. 句子切分
        2. 关键词过滤（动作词、视觉词）
        3. 提取短语
    """
    # 清理文本
    text = think_text.strip()
    
    # 句子切分（简单版）
    sentences = re.split(r'[.!?]\s+', text)
    
    # 视觉动作关键词（更具体）
    visual_action_keywords = [
        'pick', 'hold', 'give', 'receive', 'grab', 'touch', 'point',
        'smile', 'frown', 'laugh', 'cry', 'nod', 'shake', 'turn',
        'wear', 'dress', 'put on', 'take off',
        'stand', 'sit', 'walk', 'run', 'jump', 'lean', 'bend',
        'raise', 'lower', 'open', 'close', 'wave', 'gesture',
        'kiss', 'hug', 'push', 'pull', 'throw', 'catch'
    ]
    
    # 视觉对象关键词
    visual_object_keywords = [
        'woman', 'man', 'person', 'people', 'child', 'baby',
        'hair', 'face', 'eyes', 'eyebrows', 'hand', 'arm', 'leg',
        'dress', 'suit', 'jacket', 'shirt', 'hat', 'glasses',
        'rose', 'flower', 'book', 'phone', 'bag', 'backpack',
        'table', 'chair', 'door', 'window', 'car', 'room'
    ]
    
    # 需要过滤的推理性词汇（扩展）
    reasoning_keywords = [
        'think', 'consider', 'seem', 'suggest', 'indicate', 'imply',
        'therefore', 'so', 'thus', 'hence', 'because', 'since',
        'option', 'choice', 'answer', 'question', 'let me', 'okay',
        "i'm", "i see", "i hear", "i notice", "now,", "first,", 
        "looking at", "listening to", "focus on", "take a look",
        "makes me think", "makes me", "probably", "definitely",
        "if i", "best fit", "based on", "whole place", "clearly"
    ]
    
    events = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20:  # 更高的最小长度
            continue
        
        sentence_lower = sentence.lower()
        
        # 过滤掉推理性句子（更严格）
        if any(kw in sentence_lower for kw in reasoning_keywords):
            continue
        
        # 必须同时包含动作词和对象词（更严格）
        has_action = any(kw in sentence_lower for kw in visual_action_keywords)
        has_object = any(kw in sentence_lower for kw in visual_object_keywords)
        
        # 只有同时包含动作和对象的句子才保留
        if has_action and has_object:
            anchor = sentence
            query = _simplify_to_query(sentence)
            events.append(Event(anchor, query))
            
            if len(events) >= max_events:
                break
    
    # 如果提取的事件太少，降低标准（至少要有动作或对象）
    if len(events) < 3:
        events = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 15:
                continue
            
            sentence_lower = sentence.lower()
            
            # 仍然过滤推理性句子
            if any(kw in sentence_lower for kw in reasoning_keywords):
                continue
            
            has_action = any(kw in sentence_lower for kw in visual_action_keywords)
            has_object = any(kw in sentence_lower for kw in visual_object_keywords)
            
            if has_action or has_object:
                anchor = sentence
                query = _simplify_to_query(sentence)
                events.append(Event(anchor, query))
                
                if len(events) >= max_events:
                    break
    
    return events


def _simplify_to_query(sentence: str) -> str:
    """
    将句子简化为适合 CLIP 的查询短语
    
    策略：
        - 去掉副词、连接词
        - 保留主要的名词和动词
        - 限制长度
    """
    # 简单版本：保留前8个词
    words = sentence.split()[:8]
    
    # 去掉一些停用词
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'that', 'this', 'it'}
    filtered = [w for w in words if w.lower() not in stop_words]
    
    return ' '.join(filtered[:6])  # 限制6个词


def _parse_events_json(text: str) -> List[Event]:
    """
    从生成的文本中解析 JSON 格式的事件
    
    支持的格式：
        {"events": [...]}
        或者直接是数组 [...]
    """
    # 提取 JSON 部分（可能在 markdown 代码块中）
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # 尝试直接提取 JSON 对象
        json_match = re.search(r'\{.*"events".*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError("No valid JSON found in generated text")
    
    # 解析 JSON
    data = json.loads(json_str)
    
    if isinstance(data, dict) and 'events' in data:
        events_data = data['events']
    elif isinstance(data, list):
        events_data = data
    else:
        raise ValueError("Invalid JSON structure")
    
    # 转换为 Event 对象
    events = []
    for item in events_data:
        if isinstance(item, dict) and 'anchor' in item and 'query' in item:
            events.append(Event(item['anchor'], item['query']))
        else:
            warnings.warn(f"Invalid event format: {item}")
    
    return events


def extract_events(
    think_text: str,
    method: str = "llm",
    model=None,
    processor=None,
    max_events: int = 10
) -> List[Event]:
    """
    统一的事件提取接口
    
    Args:
        think_text: <think> 文本
        method: "llm" 或 "rule"
        model, processor: LLM 方法需要
        max_events: 最多提取的事件数
    
    Returns:
        events: Event 对象列表
    """
    if method == "llm":
        if model is None or processor is None:
            warnings.warn("model/processor not provided, falling back to rule-based")
            return extract_events_rule_based(think_text, max_events)
        return extract_events_with_llm(think_text, model, processor, max_events)
    elif method == "rule":
        return extract_events_rule_based(think_text, max_events)
    else:
        raise ValueError(f"Unknown method: {method}")


# 便捷函数
def events_to_queries(events: List[Event]) -> List[str]:
    """提取所有 query 字符串"""
    return [e.query for e in events]


def events_to_dict_list(events: List[Event]) -> List[Dict]:
    """转换为字典列表（用于保存/调试）"""
    return [e.to_dict() for e in events]
