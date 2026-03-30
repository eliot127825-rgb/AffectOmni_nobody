"""
Thinking Focus Reward: Evaluate whether thinking focuses on the correct answer

Keyword matching approach:
- Count correct answer keywords vs wrong answer keywords
- Focused on correct answer -> high score
- Ambiguous / biased toward wrong answer -> low score
"""

import re
from typing import List, Dict, Any
import torch


def extract_keywords(text: str, min_length: int = 2) -> List[str]:
    """Extract keywords"""
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'should', 'could', 'may', 'might', 'must', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our',
        'their', 'what', 'which', 'who', 'whom', 'when', 'where', 'how', 'not',
        'no', 'nor', 'so', 'if', 'then', 'than', 'too', 'very', 'just', 'about',
    }
    
    return [w for w in words if len(w) >= min_length and w not in stop_words]


def extract_answer_text(option: str) -> str:
    """Extract answer text from an option (strip A. B. etc. prefixes)"""
    return re.sub(r'^[A-E][\.．、]\s*', '', option.strip())


def count_keywords_in_text(text: str, keywords: List[str]) -> int:
    """Count occurrences of keywords in text"""
    text_lower = text.lower()
    return sum(text_lower.count(kw.lower()) for kw in keywords)


def thinking_focus_reward(completions, question=None, options=None, solution=None, **kwargs):
    """
    Compute thinking focus reward
    
    Args:
        completions: list of generated texts
        question: question text (optional)
        options: option list ["A. ...", "B. ...", ...]
        solution: correct answer (letter, e.g. "A")
        
    Returns:
        reward tensor
    """
    if not options or not solution:
        # Missing required information, return neutral score
        return torch.tensor([0.5] * len(completions), dtype=torch.float32)
    
    # Handle solution (may be a list)
    if isinstance(solution, list):
        if len(solution) == 0:
            return torch.tensor([0.5] * len(completions), dtype=torch.float32)
        solution_letter = solution[0] if isinstance(solution[0], str) else str(solution[0])
    else:
        solution_letter = str(solution)
    
    solution_letter = solution_letter.strip().upper()
    
    # Extract first letter
    if len(solution_letter) > 0 and solution_letter[0].isalpha():
        solution_letter = solution_letter[0]
    
    solution_idx = ord(solution_letter) - ord('A')
    
    if solution_idx < 0 or solution_idx >= len(options):
        return torch.tensor([0.5] * len(completions), dtype=torch.float32)
    
    # Extract keywords from correct and wrong options
    correct_option = options[solution_idx]
    wrong_options = [opt for i, opt in enumerate(options) if i != solution_idx]
    
    correct_text = extract_answer_text(correct_option)
    correct_keywords = list(set(extract_keywords(correct_text)))
    
    wrong_keywords = []
    for opt in wrong_options:
        wrong_text = extract_answer_text(opt)
        wrong_keywords.extend(extract_keywords(wrong_text))
    wrong_keywords = list(set(wrong_keywords))
    
    # Remove overlapping keywords
    overlap = set(correct_keywords) & set(wrong_keywords)
    correct_keywords = [kw for kw in correct_keywords if kw not in overlap]
    wrong_keywords = [kw for kw in wrong_keywords if kw not in overlap]
    
    rewards = []
    
    for completion in completions:
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
        
        # Extract <think> tag content
        think_match = re.search(r'<think>(.*?)</think>', generated_text, re.DOTALL | re.IGNORECASE)
        thinking_text = think_match.group(1) if think_match else generated_text
        
        # Count keywords
        correct_count = count_keywords_in_text(thinking_text, correct_keywords)
        wrong_count = count_keywords_in_text(thinking_text, wrong_keywords)
        
        # Compute reward
        if correct_count == 0 and wrong_count == 0:
            reward = 0.3  # Generic reasoning, neutral score
        elif correct_count > wrong_count * 1.5:
            reward = 1.0  # Clearly focused on correct answer
        elif correct_count > wrong_count:
            reward = 0.7  # Slightly biased toward correct answer
        elif correct_count == wrong_count:
            reward = 0.3  # Neutral
        else:
            reward = 0.0  # Biased toward wrong answer
        
        rewards.append(reward)
    
    return torch.tensor(rewards, dtype=torch.float32)


if __name__ == "__main__":
    # Test
    print("Testing thinking_focus_reward...")
    
    test_options = [
        "A. Romantic affection",
        "B. Friendship",
        "C. Professional relationship"
    ]
    test_solution = "A"
    
    test_completions = [
        "<think>The man shows romantic affection, his gaze is very gentle</think><answer>A</answer>",
        "<think>Could be friendship, or possibly a romantic relationship</think><answer>A</answer>",
        "<think>This is a professional interaction</think><answer>A</answer>",
    ]
    
    rewards = thinking_focus_reward(
        test_completions,
        options=test_options,
        solution=test_solution
    )
    
    print(f"\nThinking focus rewards: {rewards}")
