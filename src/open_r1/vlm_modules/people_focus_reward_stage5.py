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
    
    def evaluate_people_focus_comparative(thinkings_list):
        """Use LLM API for comparative evaluation of people focus quality
        across all candidates.
        
        Args:
            thinkings_list: List[str], thinking text from all candidates
            
        Returns:
            scores: List[float], people focus scores (0-1)
        """
        num_candidates = len(thinkings_list)
        
        # Build comparative evaluation prompt
        candidates_text = ""
        for i, thinking in enumerate(thinkings_list, 1):
            candidates_text += f"\n[Candidate {i}]\n{thinking[:600]}\n"
        
        prompt = f"""Please comparatively evaluate the following {num_candidates} candidate answers on the **people focus** dimension.

{candidates_text}

[Evaluation Criteria]
Evaluate which answer more thoroughly focuses on the **people** in the video (actions, expressions, body language, interactions).

Please score each candidate (0-10), referencing the following detailed criteria:

- **10**: Very detailed descriptions of people's actions, expressions, body language, and interactions; almost every observation is people-related
- **7-9**: Significant focus on people, describes multiple people-related details
- **4-6**: Mentions people, but also focuses substantially on environment, objects, and other non-people factors
- **1-3**: Rarely mentions people, mainly describes environment, objects, or other content
- **0**: No focus on people at all

**Important notes**:
1. Please compare relatively based on the above criteria, scores should have clear differentiation
2. The best answer should be close to 10, the worst close to 0, with middle answers distributed by quality
3. Avoid clustering all scores in the 5-7 range

Please return in the following format (one candidate per line, score only):
Answer 1: score
Answer 2: score
...

Example:
Answer 1: 9.2
Answer 2: 5.3
Answer 3: 2.4
Answer 4: 7.6"""

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
    
    def count_people_focus_simple(text):
        """Simplified fallback when API fails"""
        text_lower = text.lower()
        people_keywords = ['person', 'people', 'man', 'woman', 'facial', 'expression', 
                          'gesture', 'interaction', 'emotion']
        count = sum(1 for kw in people_keywords if kw in text_lower)
        return min(1.0, count / 10.0)
    
    # Process all completions - comparative evaluation
    contents = [completion[0]["content"] for completion in completions]
    thinkings = [extract_thinking(content) for content in contents]
    
    print(f"Calling API for comparative evaluation of {len(contents)} candidates (people focus)...")
    
    # Evaluate all answers at once
    rewards = evaluate_people_focus_comparative(thinkings)
    
    print(f"People focus evaluation complete")
    
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
