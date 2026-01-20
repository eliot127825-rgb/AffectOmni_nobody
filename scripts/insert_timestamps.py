"""
时间戳插入模块
将匹配好的时间戳插入到原始 think 文本中
"""

import re
from typing import List, Dict
from extract_events import Event


def insert_timestamps(
    think_text: str,
    events: List[Event],
    frame_matches: Dict[str, int],
    timestamps: List[float],
    format_style: str = "frame_and_time"
) -> str:
    """
    将时间戳插入到原始 think 文本中
    
    Args:
        think_text: 原始 <think> 文本
        events: 事件列表（包含 anchor 和 query）
        frame_matches: {event.query: frame_id} 映射
        timestamps: 每帧的时间戳列表
        format_style: 时间戳格式
            - "frame_and_time": [Frame 3: 3.00s]
            - "frame_only": [Frame 3]
            - "time_only": [3.00s]
    
    Returns:
        think_with_timestamps: 插入时间戳后的文本
    
    策略：
        1. 对每个事件，找到其 anchor 在原文中的位置
        2. 在 anchor 后插入时间戳
        3. 处理同一 anchor 多次出现的情况（使用"第一次未插入的位置"）
        4. 如果 anchor 找不到，尝试降级策略
    """
    result = think_text
    inserted_positions = set()  # 记录已插入的位置，避免重复
    
    for event in events:
        anchor = event.anchor.strip()
        query = event.query
        
        # 获取匹配的帧号
        if query not in frame_matches:
            continue
        
        frame_id = frame_matches[query]
        timestamp = timestamps[frame_id] if frame_id < len(timestamps) else 0.0
        
        # 生成时间戳字符串
        timestamp_str = _format_timestamp(frame_id, timestamp, format_style)
        
        # 在原文中查找 anchor
        success = _insert_at_anchor(
            result, anchor, timestamp_str, inserted_positions
        )
        
        if success:
            result = success
        else:
            # Fallback: 尝试模糊匹配
            result = _insert_with_fuzzy_match(
                result, anchor, query, timestamp_str, inserted_positions
            )
    
    return result


def _format_timestamp(
    frame_id: int,
    timestamp: float,
    format_style: str
) -> str:
    """生成时间戳字符串"""
    if format_style == "frame_and_time":
        return f" [Frame {frame_id}: {timestamp:.2f}s]"
    elif format_style == "frame_only":
        return f" [Frame {frame_id}]"
    elif format_style == "time_only":
        return f" [{timestamp:.2f}s]"
    else:
        return f" [Frame {frame_id}: {timestamp:.2f}s]"


def _insert_at_anchor(
    text: str,
    anchor: str,
    timestamp_str: str,
    inserted_positions: set
) -> str:
    """
    在指定 anchor 位置插入时间戳
    
    处理同一 anchor 多次出现的情况：
        - 使用"第一次未插入的位置"
    
    Returns:
        插入后的文本，如果找不到 anchor 则返回 None
    """
    # 不区分大小写查找
    anchor_lower = anchor.lower()
    text_lower = text.lower()
    
    # 查找所有匹配位置
    positions = []
    start = 0
    while True:
        pos = text_lower.find(anchor_lower, start)
        if pos == -1:
            break
        positions.append(pos)
        start = pos + 1
    
    if not positions:
        return None
    
    # 找到第一个未插入的位置
    for pos in positions:
        if pos not in inserted_positions:
            # 插入时间戳（在 anchor 后）
            insert_pos = pos + len(anchor)
            result = text[:insert_pos] + timestamp_str + text[insert_pos:]
            
            # 标记已插入
            inserted_positions.add(pos)
            return result
    
    # 所有位置都已插入
    return None


def _insert_with_fuzzy_match(
    text: str,
    anchor: str,
    query: str,
    timestamp_str: str,
    inserted_positions: set
) -> str:
    """
    降级策略：模糊匹配
    
    如果 anchor 完全找不到，尝试：
        1. 查找 query 中的关键词
        2. 查找 anchor 的部分短语
        3. 最后：直接追加到句末
    """
    # 策略1: 查找 query 的关键词
    query_words = query.split()
    for word in query_words:
        if len(word) > 3:  # 跳过太短的词
            match = re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE)
            if match:
                pos = match.start()
                if pos not in inserted_positions:
                    # 找到词的结尾
                    end_pos = match.end()
                    result = text[:end_pos] + timestamp_str + text[end_pos:]
                    inserted_positions.add(pos)
                    return result
    
    # 策略2: 查找 anchor 的前半部分
    anchor_half = anchor[:len(anchor)//2]
    if len(anchor_half) > 10:
        pos = text.lower().find(anchor_half.lower())
        if pos != -1 and pos not in inserted_positions:
            insert_pos = pos + len(anchor_half)
            result = text[:insert_pos] + timestamp_str + text[insert_pos:]
            inserted_positions.add(pos)
            return result
    
    # 策略3: 追加到文本末尾（带上下文）
    # 这是最后的手段，尽量避免
    result = text + f"\n(Event: {query}{timestamp_str})"
    return result


def batch_insert_timestamps(
    think_text: str,
    events: List[Event],
    frame_matches: List[int],
    timestamps: List[float],
    format_style: str = "frame_and_time"
) -> str:
    """
    批量插入时间戳（events 和 frame_matches 是对齐的列表）
    
    Args:
        think_text: 原始文本
        events: 事件列表
        frame_matches: 帧号列表（与 events 对齐）
        timestamps: 时间戳列表
        format_style: 格式样式
    
    Returns:
        插入时间戳后的文本
    """
    # 转换为字典格式
    frame_dict = {
        event.query: frame_matches[i]
        for i, event in enumerate(events)
        if i < len(frame_matches)
    }
    
    return insert_timestamps(
        think_text, events, frame_dict, timestamps, format_style
    )


def verify_insertions(
    original: str,
    modified: str,
    expected_count: int
) -> Dict[str, any]:
    """
    验证时间戳插入的结果
    
    Returns:
        {
            'success': bool,
            'inserted_count': int,
            'expected_count': int,
            'missing': int
        }
    """
    # 统计插入的时间戳数量
    timestamp_pattern = r'\[Frame \d+: \d+\.\d+s\]'
    inserted = len(re.findall(timestamp_pattern, modified))
    
    return {
        'success': inserted >= expected_count * 0.7,  # 70% 成功率即可
        'inserted_count': inserted,
        'expected_count': expected_count,
        'missing': max(0, expected_count - inserted),
        'insertion_rate': inserted / expected_count if expected_count > 0 else 0
    }


# 便捷函数
def quick_insert(
    think_text: str,
    event_queries: List[str],
    frame_ids: List[int],
    timestamps: List[float]
) -> str:
    """
    快速插入接口（当你已经有简单的 query 列表时）
    
    注意：这个函数假设 query 可以直接在文本中找到
    """
    from extract_events import Event
    
    # 构造简单的 Event 对象（anchor = query）
    events = [Event(anchor=q, query=q) for q in event_queries]
    
    frame_dict = {q: fid for q, fid in zip(event_queries, frame_ids)}
    
    return insert_timestamps(
        think_text, events, frame_dict, timestamps
    )
