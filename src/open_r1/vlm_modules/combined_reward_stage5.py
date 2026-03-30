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
    
    def evaluate_combined_comparative(thinkings_list):
        """Use LLM API for comparative evaluation of people focus and temporal analysis
        across all candidates.
        
        Args:
            thinkings_list: List[str], thinking text from all candidates
            
        Returns:
            people_scores: List[float], people focus scores (0-1)
            temporal_scores: List[float], temporal analysis scores (0-1)
        """
        num_candidates = len(thinkings_list)
        
        # Build comparative evaluation prompt
        candidates_text = ""
        for i, thinking in enumerate(thinkings_list, 1):
            candidates_text += f"\n[Candidate {i}]\n{thinking[:600]}\n"
        
        prompt = f"""Please comparatively evaluate the following {num_candidates} candidate answers on two dimensions, providing relative ranking and scores.

{candidates_text}

[Dimension 1: People Focus]
Evaluate which answer more thoroughly focuses on the **people** in the video (actions, expressions, body language, interactions).

[Dimension 2: Temporal Analysis]
Evaluate which answer better analyzes content **in chronological order of the video**.

Please score each candidate on both dimensions (0-10), scores should reflect relative quality:
- Best answers close to 10
- Medium quality answers 5-7
- Poor answers close to 0

**Important: Scores should have clear differentiation, avoid giving similar scores to all**

Please return in the following format (one candidate per line, two dimension scores separated by comma):
Answer 1: people_score,temporal_score
Answer 2: people_score,temporal_score
...

Example:
Answer 1: 8,7
Answer 2: 5,6
Answer 3: 3,4
Answer 4: 7,8"""

        try:
            response = call_qwen_api(prompt)
            if response:
                # Parse scores for all candidates
                people_scores = []
                temporal_scores = []
                
                # Extract scores from each line
                lines = response.strip().split('\n')
                for line in lines:
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) >= 2:
                        people_score = max(0, min(10, int(numbers[0]))) / 10.0
                        temporal_score = max(0, min(10, int(numbers[1]))) / 10.0
                        people_scores.append(people_score)
                        temporal_scores.append(temporal_score)
                
                # If enough scores were successfully parsed
                if len(people_scores) == num_candidates:
                    return people_scores, temporal_scores
                    
        except Exception as e:
            print(f"API comparative evaluation failed: {e}")
        
        # Return medium scores on failure
        return [0.5] * num_candidates, [0.5] * num_candidates
    
    # Process all completions - comparative evaluation
    contents = [completion[0]["content"] for completion in completions]
    thinkings = [extract_thinking(content) for content in contents]
    
    print(f"Calling API for comparative evaluation of {len(contents)} candidates (people focus + temporal analysis)...")
    
    # Evaluate all answers at once
    people_rewards, temporal_rewards = evaluate_combined_comparative(thinkings)
    
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
