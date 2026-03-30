"""
Outcome Reward: Result-oriented reward based on held-out validation set.

Directly optimizes answer accuracy:
- Uses held-out validation set
- Correct = 1.0, Wrong = 0.0
- Avoids overfitting to the test set
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any
import torch

# Global cache for held-out dataset
_HELD_OUT_INDEX = None


def load_held_out_index():
    """Load held-out validation set index"""
    global _HELD_OUT_INDEX
    
    if _HELD_OUT_INDEX is not None:
        return _HELD_OUT_INDEX
    
    held_out_path = Path(__file__).parent.parent.parent.parent / "data" / "outcome_reward_data" / "daily_held_out.json"
    
    if not held_out_path.exists():
        print(f"Warning: held-out validation set not found: {held_out_path}")
        print(f"   outcome_reward will return 0.0")
        _HELD_OUT_INDEX = {}
        return _HELD_OUT_INDEX
    
    with open(held_out_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Build index: video_name -> ground_truth
    _HELD_OUT_INDEX = {}
    for item in data:
        video_path = item.get('path', item.get('video_path', ''))
        if video_path:
            video_name = Path(video_path).name
            ground_truth = item.get('final_answer', item.get('answer', ''))
            _HELD_OUT_INDEX[video_name] = ground_truth.strip().upper()
    
    print(f"Loaded held-out validation index: {len(_HELD_OUT_INDEX)} records")
    
    return _HELD_OUT_INDEX


def extract_answer(text: str) -> str:
    """Extract answer from generated text"""
    # Method 1: <answer> tag
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        # Extract first letter
        if len(answer) > 0 and answer[0].isalpha():
            return answer[0].upper()
        return answer.upper()
    
    # Method 2: Common answer patterns
    patterns = [
        r'(?:answer is|the answer is)\s*[:：]?\s*([A-E])',
        r'(?:final answer)\s*[:：]?\s*([A-E])',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # Method 3: Last standalone uppercase letter
    letters = re.findall(r'\b([A-E])\b', text)
    if letters:
        return letters[-1].upper()
    
    return ""


def outcome_reward(completions, solution=None, **kwargs):
    """
    Compute outcome reward.
    
    Args:
        completions: list of generated texts
        solution: ground truth answer list (used first if available)
        **kwargs: may contain video_paths etc.
        
    Returns:
        reward tensor
    """
    held_out_index = load_held_out_index()
    
    rewards = []
    
    for idx, completion in enumerate(completions):
        # Extract generated text
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
        
        # Extract predicted answer
        predicted_answer = extract_answer(generated_text)
        
        # Get ground truth
        ground_truth = ""
        
        # Prefer solution parameter
        if solution and idx < len(solution):
            gt = solution[idx]
            if isinstance(gt, str):
                ground_truth = gt.strip().upper()
                # Extract first letter
                if len(ground_truth) > 0 and ground_truth[0].isalpha():
                    ground_truth = ground_truth[0]
        
        # If no solution, try to look up from held-out index
        if not ground_truth and held_out_index:
            video_paths = kwargs.get('video_paths', [])
            if video_paths and idx < len(video_paths):
                video_name = Path(video_paths[idx]).name
                ground_truth = held_out_index.get(video_name, '')
        
        # Compute reward
        if predicted_answer and ground_truth:
            reward = 1.0 if predicted_answer == ground_truth else 0.0
        else:
            # Cannot determine, assign 0 (conservative strategy)
            reward = 0.0
        
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float32)


if __name__ == "__main__":
    # Test
    print("Testing outcome_reward...")
    
    index = load_held_out_index()
    print(f"\nIndex contains {len(index)} records")
    
    if index:
        print("\nFirst 5 entries:")
        for i, (video, answer) in enumerate(list(index.items())[:5]):
            print(f"  {video} -> {answer}")
    
    # Test reward computation
    test_completions = [
        "<think>Analyzing...</think><answer>A</answer>",
        "<think>Reasoning...</think><answer>B. Option B</answer>",
        "No clear answer",
    ]
    test_solution = ["A", "B", "C"]
    
    rewards = outcome_reward(test_completions, solution=test_solution)
    print(f"\nTest rewards: {rewards}")
