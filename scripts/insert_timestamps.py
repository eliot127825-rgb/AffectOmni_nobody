"""
Timestamp insertion module
Insert matched timestamps into the original think text
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
    Insert timestamps into the original think text
    
    Args:
        think_text: Original <think> text
        events: List of events (containing anchor and query)
        frame_matches: {event.query: frame_id} mapping
        timestamps: List of timestamps per frame
        format_style: Timestamp format
            - "frame_and_time": [Frame 3: 3.00s]
            - "frame_only": [Frame 3]
            - "time_only": [3.00s]
    
    Returns:
        think_with_timestamps: Text with timestamps inserted
    
    Strategy:
        1. For each event, find the position of its anchor in the original text
        2. Insert timestamp after the anchor
        3. Handle cases where the same anchor appears multiple times (use "first un-inserted position")
        4. If anchor is not found, try fallback strategies
    """
    result = think_text
    inserted_positions = set()  # track inserted positions to avoid duplicates
    
    for event in events:
        anchor = event.anchor.strip()
        query = event.query
        
        # Get matched frame number
        if query not in frame_matches:
            continue
        
        frame_id = frame_matches[query]
        timestamp = timestamps[frame_id] if frame_id < len(timestamps) else 0.0
        
        # Generate timestamp string
        timestamp_str = _format_timestamp(frame_id, timestamp, format_style)
        
        # Find anchor in the original text
        success = _insert_at_anchor(
            result, anchor, timestamp_str, inserted_positions
        )
        
        if success:
            result = success
        else:
            # Fallback: try fuzzy matching
            result = _insert_with_fuzzy_match(
                result, anchor, query, timestamp_str, inserted_positions
            )
    
    return result


def _format_timestamp(
    frame_id: int,
    timestamp: float,
    format_style: str
) -> str:
    """Generate timestamp string"""
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
    Insert timestamp at the specified anchor position
    
    Handles cases where the same anchor appears multiple times:
        - Uses "first un-inserted position"
    
    Returns:
        Text after insertion, or None if anchor is not found
    """
    # Case-insensitive search
    anchor_lower = anchor.lower()
    text_lower = text.lower()
    
    # Find all matching positions
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
    
    # Find the first un-inserted position
    for pos in positions:
        if pos not in inserted_positions:
            # Insert timestamp (after anchor)
            insert_pos = pos + len(anchor)
            result = text[:insert_pos] + timestamp_str + text[insert_pos:]
            
            # Mark as inserted
            inserted_positions.add(pos)
            return result
    
    # All positions already inserted
    return None


def _insert_with_fuzzy_match(
    text: str,
    anchor: str,
    query: str,
    timestamp_str: str,
    inserted_positions: set
) -> str:
    """
    Fallback strategy: fuzzy matching
    
    If anchor cannot be found at all, try:
        1. Search for keywords from the query
        2. Search for partial phrases from the anchor
        3. Last resort: append to end of text
    """
    # Strategy 1: search for query keywords
    query_words = query.split()
    for word in query_words:
        if len(word) > 3:  # skip words that are too short
            match = re.search(r'\b' + re.escape(word) + r'\b', text, re.IGNORECASE)
            if match:
                pos = match.start()
                if pos not in inserted_positions:
                    # Find end of word
                    end_pos = match.end()
                    result = text[:end_pos] + timestamp_str + text[end_pos:]
                    inserted_positions.add(pos)
                    return result
    
    # Strategy 2: search for the first half of anchor
    anchor_half = anchor[:len(anchor)//2]
    if len(anchor_half) > 10:
        pos = text.lower().find(anchor_half.lower())
        if pos != -1 and pos not in inserted_positions:
            insert_pos = pos + len(anchor_half)
            result = text[:insert_pos] + timestamp_str + text[insert_pos:]
            inserted_positions.add(pos)
            return result
    
    # Strategy 3: append to end of text (with context)
    # This is a last resort, try to avoid
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
    Batch insert timestamps (events and frame_matches are aligned lists)
    
    Args:
        think_text: Original text
        events: List of events
        frame_matches: List of frame numbers (aligned with events)
        timestamps: List of timestamps
        format_style: Format style
    
    Returns:
        Text with timestamps inserted
    """
    # Convert to dict format
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
    Verify the results of timestamp insertion
    
    Returns:
        {
            'success': bool,
            'inserted_count': int,
            'expected_count': int,
            'missing': int
        }
    """
    # Count inserted timestamps
    timestamp_pattern = r'\[Frame \d+: \d+\.\d+s\]'
    inserted = len(re.findall(timestamp_pattern, modified))
    
    return {
        'success': inserted >= expected_count * 0.7,  # 70% success rate is acceptable
        'inserted_count': inserted,
        'expected_count': expected_count,
        'missing': max(0, expected_count - inserted),
        'insertion_rate': inserted / expected_count if expected_count > 0 else 0
    }


# Convenience function
def quick_insert(
    think_text: str,
    event_queries: List[str],
    frame_ids: List[int],
    timestamps: List[float]
) -> str:
    """
    Quick insert interface (when you already have a simple query list)
    
    Note: This function assumes queries can be found directly in the text
    """
    from extract_events import Event
    
    # Construct simple Event objects (anchor = query)
    events = [Event(anchor=q, query=q) for q in event_queries]
    
    frame_dict = {q: fid for q, fid in zip(event_queries, frame_ids)}
    
    return insert_timestamps(
        think_text, events, frame_dict, timestamps
    )
