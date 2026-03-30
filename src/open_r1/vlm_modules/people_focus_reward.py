"""
People Focus Reward Function
Used in Stage 4 GRPO training to evaluate whether model output
sufficiently focuses on people.
"""

import re
import requests
import os
import time

# LLM API configuration (for evaluating people focus)
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


def people_focus_reward_simple(completions, **kwargs):
    """
    Simplified people focus reward (keyword-based statistics, no API needed).
    Suitable for fast training and debugging.
    
    Evaluation criteria:
    - Detect count and density of people-related keywords
    - Score range: 0.0 - 1.0
    """
    
    def extract_thinking(text):
        """Extract <think> section"""
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def count_people_focus(text):
        """Count people focus related features"""
        text_lower = text.lower()
        
        # People-related keywords (high weight)
        people_keywords = [
            'person', 'people', 'man', 'woman', 'men', 'women',
            'he', 'she', 'they', 'his', 'her', 'their',
            'facial', 'expression', 'face', 'smile', 'frown',
            'gesture', 'hand', 'body language', 'posture',
            'interaction', 'talking', 'speaking', 'listening',
            'emotion', 'feeling', 'mood', 'tone',
            'looking', 'watching', 'gazing', 'staring',
            'wearing', 'dressed', 'clothing'
        ]
        
        # Action words (people-related)
        action_keywords = [
            'walk', 'run', 'sit', 'stand', 'move',
            'talk', 'speak', 'say', 'ask', 'answer',
            'hold', 'touch', 'point', 'wave',
            'laugh', 'cry', 'nod', 'shake'
        ]
        
        # Environment words (low weight, too many will lower score)
        environment_keywords = [
            'background', 'setting', 'location', 'place',
            'room', 'building', 'outdoor', 'indoor',
            'sky', 'ground', 'wall', 'floor'
        ]
        
        # Count keywords
        people_count = sum(1 for kw in people_keywords if kw in text_lower)
        action_count = sum(1 for kw in action_keywords if kw in text_lower)
        env_count = sum(1 for kw in environment_keywords if kw in text_lower)
        
        # Compute score
        # people words + action words - excess environment words
        people_score = (people_count * 1.0 + action_count * 0.5) / 20.0  # normalize
        env_penalty = max(0, (env_count - 3) * 0.1)  # penalize when env words > 3
        
        score = max(0.0, min(1.0, people_score - env_penalty))
        
        return score
    
    # Process each completion
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in contents:
        thinking = extract_thinking(content)
        score = count_people_focus(thinking)
        rewards.append(score)
    
    return rewards


def people_focus_reward_api(completions, **kwargs):
    """
    Advanced people focus reward (using LLM API for evaluation).
    More accurate but slower, requires API environment variables.
    
    Environment variables:
    - DASHSCOPE_API_KEY: API key
    
    Evaluation criteria:
    - Use LLM to judge whether reasoning focuses on people
    - Score range: 0.0 - 1.0
    """
    
    # Check API configuration
    if not api_key:
        print("Warning: DASHSCOPE_API_KEY not configured, falling back to simplified reward")
        return people_focus_reward_simple(completions, **kwargs)
    
    def extract_thinking(text):
        """Extract <think> section"""
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def evaluate_people_focus(thinking_text):
        """Use LLM API to evaluate people focus"""
        prompt = f"""Please evaluate whether the following reasoning text sufficiently focuses on the **people** in the video (actions, expressions, body language, interactions).

Scoring criteria (0-10):
- 10: Very detailed descriptions of people's actions, expressions, body language, and interactions; almost every observation is people-related
- 7-9: Significant focus on people, describes multiple people-related details
- 4-6: Mentions people, but also focuses substantially on environment, objects, and other non-people factors
- 1-3: Rarely mentions people, mainly describes environment, objects, or other content
- 0: No focus on people at all

Reasoning text:
{thinking_text[:800]}

Please return only the score (integer 0-10), no other text."""

        try:
            response = call_qwen_api(prompt)
            if response:
                score_match = re.search(r'\d+', response)
                if score_match:
                    score = int(score_match.group())
                    return max(0, min(10, score)) / 10.0  # Normalize to [0, 1]
        except Exception as e:
            print(f"API evaluation failed: {e}")
        
        # Fall back to simplified version on failure
        return count_people_focus_simple(thinking_text)
    
    def count_people_focus_simple(text):
        """Simplified fallback when API fails"""
        text_lower = text.lower()
        people_keywords = ['person', 'people', 'man', 'woman', 'facial', 'expression', 
                          'gesture', 'interaction', 'emotion']
        count = sum(1 for kw in people_keywords if kw in text_lower)
        return min(1.0, count / 10.0)
    
    # Process each completion
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in contents:
        thinking = extract_thinking(content)
        score = evaluate_people_focus(thinking)
        rewards.append(score)
    
    return rewards


# Default to simplified version (faster, no API needed)
def people_focus_reward(completions, **kwargs):
    """
    People focus reward (defaults to simplified version).
    
    To use the API version, set environment variables:
    - export USE_API_REWARD=true
    - export API=<api_endpoint>
    - export API_KEY=<your_api_key>
    """
    use_api = os.environ.get("USE_API_REWARD", "false").lower() == "true"
    
    if use_api:
        return people_focus_reward_api(completions, **kwargs)
    else:
        return people_focus_reward_simple(completions, **kwargs)
