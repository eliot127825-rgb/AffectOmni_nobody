"""
Temporal Order Reward Function
Used in Stage 4 GRPO training to evaluate whether the model analyzes content
in chronological video order.
"""

import re
import os
import time

# LLM API configuration
api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("API_KEY", "")

def call_qwen_api(prompt, model_name="qwen-max", max_retries=20):
    """Call LLM API for evaluation (using DashScope SDK)"""
    try:
        from dashscope import Generation
        import dashscope
        dashscope.api_key = api_key
    except ImportError:
        print("Warning: dashscope not installed, falling back to simplified reward")
        return None
    
    for attempt in range(max_retries):
        try:
            response = Generation.call(
                model=model_name,
                prompt=prompt
            )
            if response.status_code == 200:
                return response.output.text
            else:
                print(f"API error (attempt {attempt+1}/{max_retries}): {response.message}")
        except Exception as e:
            print(f"API call failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(1)
    
    return None


def temporal_order_reward_simple(completions, **kwargs):
    """
    Simplified temporal order reward (keyword-based detection, no API needed).
    Suitable for fast training and debugging.
    
    Evaluation criteria:
    - Detect presence and distribution of temporal marker words
    - Score range: 0.0 - 1.0
    """
    
    def extract_thinking(text):
        """Extract <think> section"""
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def check_temporal_order(text):
        """Detect temporal analysis features"""
        text_lower = text.lower()
        
        # Temporal marker words (strong temporal sense)
        temporal_markers = [
            # Beginning
            'first', 'initially', 'at the beginning', 'at the start', 'opening',
            # In progress
            'then', 'next', 'after', 'following', 'subsequently', 'later',
            'meanwhile', 'during', 'while', 'as', 'when',
            # Ending
            'finally', 'eventually', 'at the end', 'lastly', 'concluding',
            # Time points
            'second', 'minute', 'moment', 'timestamp',
            # Sequence
            'before', 'after', 'sequence', 'progression', 'chronological'
        ]
        
        # Time period descriptions
        time_phrases = [
            'at 0:', 'at 1:', 'at 2:', 'at 3:', 'at 4:', 'at 5:',  # timestamps
            'in the first', 'in the second', 'in the third',
            'early in', 'middle of', 'towards the end',
            'throughout the video'
        ]
        
        # Non-temporal words (will lower score)
        non_temporal = [
            'overall', 'in general', 'static', 'always', 'entire',
            'whole video', 'throughout without change'
        ]
        
        # Count temporal markers
        temporal_count = sum(1 for marker in temporal_markers if marker in text_lower)
        time_phrase_count = sum(1 for phrase in time_phrases if phrase in text_lower)
        non_temporal_count = sum(1 for word in non_temporal if word in text_lower)
        
        # Detect explicit paragraph segmentation (step-by-step analysis)
        step_indicators = len(re.findall(r'\n\s*\d+[\.\):]|\n\s*-\s+', text))
        
        # Compute score
        # temporal markers + time phrases + step segmentation - non-temporal penalty
        temporal_score = (temporal_count * 0.8 + time_phrase_count * 1.5 + step_indicators * 0.3) / 15.0
        non_temporal_penalty = non_temporal_count * 0.1
        
        score = max(0.0, min(1.0, temporal_score - non_temporal_penalty))
        
        return score
    
    # Process each completion
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in contents:
        thinking = extract_thinking(content)
        score = check_temporal_order(thinking)
        rewards.append(score)
    
    return rewards


def temporal_order_reward_api(completions, **kwargs):
    """
    Advanced temporal order reward (using LLM API for evaluation).
    More accurate but slower, requires API environment variables.
    
    Environment variables:
    - DASHSCOPE_API_KEY: API key
    
    Evaluation criteria:
    - Use LLM to judge whether reasoning follows chronological order
    - Score range: 0.0 - 1.0
    """
    
    # Check API configuration
    if not api_key:
        print("Warning: DASHSCOPE_API_KEY not configured, falling back to simplified reward")
        return temporal_order_reward_simple(completions, **kwargs)
    
    def extract_thinking(text):
        """Extract <think> section"""
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def evaluate_temporal_order_comparative(thinkings_list):
        """Use LLM API for comparative evaluation of temporal analysis quality
        across all candidates.
        
        Args:
            thinkings_list: List[str], thinking text from all candidates
            
        Returns:
            scores: List[float], temporal analysis scores (0-1)
        """
        num_candidates = len(thinkings_list)
        
        # Build comparative evaluation prompt
        candidates_text = ""
        for i, thinking in enumerate(thinkings_list, 1):
            candidates_text += f"\n[Candidate {i}]\n{thinking[:600]}\n"
        
        prompt = f"""Please comparatively evaluate the following {num_candidates} candidate answers on the **temporal analysis** dimension.

{candidates_text}

[Evaluation Criteria]
Evaluate which answer better analyzes content **in chronological order of the video**.

Please score each candidate (0-10), referencing the following detailed criteria:

- **10**: Very clearly analyzes in chronological order (beginning -> middle -> end), uses explicit temporal markers (e.g., "first", "then", "next", "finally"), provides step-by-step descriptions of content at different time periods
- **7-9**: Good temporal structure, analyzes changes across different video phases, has some temporal markers
- **4-6**: Mentions some time-related content, but analysis is disorganized without clear temporal thread
- **1-3**: Almost no temporal analysis, mainly static descriptions or overall summaries
- **0**: No temporal order at all, purely static analysis

**Key evaluation points**:
1. Whether different time segments of the video are clearly distinguished (beginning/middle/end)
2. Whether temporal connectors are used (first, then, next, finally, afterwards, etc.)
3. Whether changes or action sequences over time are described
4. Whether purely static overall descriptions are avoided

**Important notes**:
1. Please compare relatively based on the above criteria, scores should have clear differentiation
2. The best answer should be close to 10, the worst close to 0, with middle answers distributed by quality
3. Avoid clustering all scores in the 5-7 range

Please return in the following format (one candidate per line, score only):
Answer 1: score
Answer 2: score
...

Example:
Answer 1: 9.0
Answer 2: 6.0
Answer 3: 2.0
Answer 4: 7.0"""

        try:
            response = call_qwen_api(prompt)
            if response:
                # Parse scores for all candidates
                scores = []
                
                # Extract score from each line
                lines = response.strip().split('\n')
                for line in lines:
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) >= 1:
                        score = max(0, min(10, int(numbers[0]))) / 10.0
                        scores.append(score)
                
                # If enough scores were successfully parsed
                if len(scores) == num_candidates:
                    return scores
                    
        except Exception as e:
            print(f"API comparative evaluation failed: {e}")
        
        # Return medium scores on failure
        return [0.5] * num_candidates
    
    def check_temporal_simple(text):
        """Simplified fallback when API fails"""
        text_lower = text.lower()
        temporal_keywords = ['first', 'then', 'next', 'after', 'finally', 
                            'initially', 'subsequently', 'beginning', 'end']
        count = sum(1 for kw in temporal_keywords if kw in text_lower)
        return min(1.0, count / 8.0)
    
    # Process all completions - comparative evaluation
    contents = [completion[0]["content"] for completion in completions]
    thinkings = [extract_thinking(content) for content in contents]
    
    print(f"Calling API for comparative evaluation of {len(contents)} candidates (temporal analysis)...")
    
    # Evaluate all answers at once
    rewards = evaluate_temporal_order_comparative(thinkings)
    
    print(f"Temporal analysis evaluation complete")
    
    return rewards


# Default to simplified version (faster, no API needed)
def temporal_order_reward(completions, **kwargs):
    """
    Temporal order reward (defaults to simplified version).
    
    To use the API version, set environment variables:
    - export USE_API_REWARD=true
    """
    use_api = os.environ.get("USE_API_REWARD", "false").lower() == "true"
    
    if use_api:
        return temporal_order_reward_api(completions, **kwargs)
    else:
        return temporal_order_reward_simple(completions, **kwargs)
