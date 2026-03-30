"""
Combined Reward Function
Single API call evaluates both people focus and temporal analysis,
improving efficiency and reducing cost.

Uses global caching to ensure only one API call per batch.
"""

import re
import os
import time
from functools import lru_cache
import hashlib

# LLM API configuration
api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("API_KEY", "")

# Global cache: stores the most recent API call results
_global_cache = {
    "batch_hash": None,
    "people_rewards": None,
    "temporal_rewards": None
}

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


def _compute_batch_hash(completions):
    """Compute hash for current batch, used for cache identification (includes count)"""
    content_str = str(len(completions)) + "_" + str([completion[0]["content"] for completion in completions])
    return hashlib.md5(content_str.encode()).hexdigest()


def combined_reward_api(completions, **kwargs):
    """
    Combined reward (single API call evaluates both people focus and temporal analysis).
    
    Uses global caching so the same batch only triggers one API call.
    
    Returns:
    - Tuple of two reward lists: (people_focus_rewards, temporal_order_rewards)
    """
    global _global_cache
    
    # Compute hash for current batch
    batch_hash = _compute_batch_hash(completions)
    
    # Check cache (also verify count consistency)
    if (_global_cache["batch_hash"] == batch_hash and 
        _global_cache["people_rewards"] is not None and
        len(_global_cache["people_rewards"]) == len(completions)):
        print("Using cached API evaluation results (saving API call)")
        return _global_cache["people_rewards"], _global_cache["temporal_rewards"]
    
    # Check API configuration
    if not api_key:
        print("Warning: DASHSCOPE_API_KEY not configured, cannot use combined reward")
        # Return default scores
        num_completions = len(completions)
        return ([0.5] * num_completions, [0.5] * num_completions)
    
    def extract_thinking(text):
        """Extract <think> section"""
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def evaluate_combined(thinking_text):
        """Use LLM API to simultaneously evaluate people focus and temporal analysis"""
        prompt = f"""Please simultaneously evaluate the following reasoning text on two dimensions:

[Dimension 1: People Focus]
Evaluate whether the reasoning text sufficiently focuses on the **people** in the video (actions, expressions, body language, interactions).

Scoring criteria (0-10):
- 10: Very detailed descriptions of people's actions, expressions, body language, and interactions; almost every observation is people-related
- 7-9: Significant focus on people, describes multiple people-related details
- 4-6: Mentions people, but also focuses substantially on environment, objects, and other non-people factors
- 1-3: Rarely mentions people, mainly describes environment, objects, or other content
- 0: No focus on people at all

[Dimension 2: Temporal Analysis]
Evaluate whether the reasoning text analyzes content **in chronological order of the video**.

Scoring criteria (0-10):
- 10: Very clearly analyzes in chronological order (beginning -> middle -> end), uses explicit temporal markers (e.g., "first", "then", "next", "finally"), provides step-by-step descriptions of content at different time periods
- 7-9: Good temporal structure, analyzes changes across different video phases, has some temporal markers
- 4-6: Mentions some time-related content, but analysis is disorganized without clear temporal thread
- 1-3: Almost no temporal analysis, mainly static descriptions or overall summaries
- 0: No temporal order at all, purely static analysis

Reasoning text:
{thinking_text[:800]}

Please return scores in the following format (only two numbers, separated by comma):
people_focus_score,temporal_analysis_score

Example: 8,7"""

        try:
            response = call_qwen_api(prompt)
            if response:
                numbers = re.findall(r'\d+', response)
                if len(numbers) >= 2:
                    people_score = max(0, min(10, int(numbers[0]))) / 10.0
                    temporal_score = max(0, min(10, int(numbers[1]))) / 10.0
                    return people_score, temporal_score
        except Exception as e:
            print(f"API evaluation failed: {e}")
        
        # Return medium scores on failure
        return 0.5, 0.5
    
    # Process each completion
    contents = [completion[0]["content"] for completion in completions]
    people_rewards = []
    temporal_rewards = []
    
    print(f"Calling API for evaluation of {len(contents)} candidates (people focus + temporal analysis)...")
    
    for idx, content in enumerate(contents):
        thinking = extract_thinking(content)
        people_score, temporal_score = evaluate_combined(thinking)
        people_rewards.append(people_score)
        temporal_rewards.append(temporal_score)
        if (idx + 1) % 5 == 0:
            print(f"  Completed {idx + 1}/{len(contents)} evaluations")
    
    # Update cache
    _global_cache["batch_hash"] = batch_hash
    _global_cache["people_rewards"] = people_rewards
    _global_cache["temporal_rewards"] = temporal_rewards
    
    print(f"API evaluation complete, results cached")
    
    return people_rewards, temporal_rewards


def people_focus_reward_combined(completions, **kwargs):
    """
    People focus reward (using combined API evaluation).
    Gets the first dimension scores from combined_reward_api.
    """
    use_api = os.environ.get("USE_API_REWARD", "false").lower() == "true"
    
    if use_api:
        people_rewards, _ = combined_reward_api(completions, **kwargs)
        return people_rewards
    else:
        # Fall back to simplified version
        from open_r1.vlm_modules.people_focus_reward import people_focus_reward_simple
        return people_focus_reward_simple(completions, **kwargs)


def temporal_order_reward_combined(completions, **kwargs):
    """
    Temporal order reward (using combined API evaluation).
    Gets the second dimension scores from combined_reward_api.
    """
    use_api = os.environ.get("USE_API_REWARD", "false").lower() == "true"
    
    if use_api:
        _, temporal_rewards = combined_reward_api(completions, **kwargs)
        return temporal_rewards
    else:
        # Fall back to simplified version
        from open_r1.vlm_modules.temporal_order_reward import temporal_order_reward_simple
        return temporal_order_reward_simple(completions, **kwargs)
