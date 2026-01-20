"""
合并版 Reward 函数
一次API调用同时评估人物关注度和时序分析，提升效率降低成本

使用全局缓存机制，确保每个batch只调用一次API
"""

import re
import os
import time
from functools import lru_cache
import hashlib

# Qwen API 配置
api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("API_KEY", "")

# 全局缓存：存储最近一次API调用的结果
_global_cache = {
    "batch_hash": None,
    "people_rewards": None,
    "temporal_rewards": None
}

def call_qwen_api(prompt, model_name="qwen-max", max_retries=20):
    """调用 Qwen API 进行评估（使用 DashScope SDK）"""
    try:
        from dashscope import Generation
        import dashscope
        dashscope.api_key = api_key
    except ImportError:
        print("警告：未安装 dashscope，降级使用简化版 reward")
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
                print(f"Qwen API错误 (尝试 {attempt+1}/{max_retries}): {response.message}")
        except Exception as e:
            print(f"Qwen API调用失败 (尝试 {attempt+1}/{max_retries}): {e}")
            time.sleep(1)
    
    return None


def _compute_batch_hash(completions):
    """计算当前batch的哈希值，用于缓存识别（包含数量信息）"""
    content_str = str(len(completions)) + "_" + str([completion[0]["content"] for completion in completions])
    return hashlib.md5(content_str.encode()).hexdigest()


def combined_reward_api(completions, **kwargs):
    """
    合并版 reward（一次API调用同时评估人物关注度和时序分析）
    
    使用全局缓存机制，同一个batch只调用一次API
    
    返回格式：
    - 返回两个reward列表的元组：(people_focus_rewards, temporal_order_rewards)
    """
    global _global_cache
    
    # 计算当前batch的哈希值
    batch_hash = _compute_batch_hash(completions)
    
    # 检查缓存（同时验证数量一致性）
    if (_global_cache["batch_hash"] == batch_hash and 
        _global_cache["people_rewards"] is not None and
        len(_global_cache["people_rewards"]) == len(completions)):
        print("✅ 使用缓存的API评估结果（节省API调用）")
        return _global_cache["people_rewards"], _global_cache["temporal_rewards"]
    
    # 检查 API 配置
    if not api_key:
        print("⚠️  警告：未配置 DASHSCOPE_API_KEY，无法使用合并版 reward")
        # 返回默认分数
        num_completions = len(completions)
        return ([0.5] * num_completions, [0.5] * num_completions)
    
    def extract_thinking(text):
        """提取 <think> 部分"""
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def evaluate_combined_comparative(thinkings_list):
        """使用 Qwen API 对比评估所有答案的人物关注度和时序分析
        
        Args:
            thinkings_list: List[str], 所有候选答案的thinking文本
            
        Returns:
            people_scores: List[float], 人物关注度分数（0-1）
            temporal_scores: List[float], 时序分析分数（0-1）
        """
        num_candidates = len(thinkings_list)
        
        # 构建对比评估prompt
        candidates_text = ""
        for i, thinking in enumerate(thinkings_list, 1):
            candidates_text += f"\n【候选答案{i}】\n{thinking[:600]}\n"
        
        prompt = f"""请对比评估以下{num_candidates}个候选答案在两个维度上的质量，给出相对排序和分数。

{candidates_text}

【维度1：人物关注度】
评估哪个答案更充分地关注了视频中的**人物**（动作、表情、肢体语言、交互关系）。

【维度2：时序分析】
评估哪个答案更好地**按照视频的时间顺序**进行分析。

请为每个候选答案在两个维度上打分（0-10分），分数要体现相对质量：
- 最好的答案接近10分
- 中等质量的答案5-7分
- 较差的答案接近0分

**重要：分数要有区分度，不要都打相近的分**

请按以下格式返回（每行一个候选答案，用逗号分隔两个维度的分数）：
答案1: 人物分数,时序分数
答案2: 人物分数,时序分数
...

示例：
答案1: 8,7
答案2: 5,6
答案3: 3,4
答案4: 7,8"""

        try:
            response = call_qwen_api(prompt)
            if response:
                # 解析所有候选答案的分数
                people_scores = []
                temporal_scores = []
                
                # 提取每一行的分数
                lines = response.strip().split('\n')
                for line in lines:
                    # 匹配格式：答案X: 数字,数字
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) >= 2:
                        people_score = max(0, min(10, int(numbers[0]))) / 10.0
                        temporal_score = max(0, min(10, int(numbers[1]))) / 10.0
                        people_scores.append(people_score)
                        temporal_scores.append(temporal_score)
                
                # 如果成功解析了足够的分数
                if len(people_scores) == num_candidates:
                    return people_scores, temporal_scores
                    
        except Exception as e:
            print(f"API对比评估失败: {e}")
        
        # 失败时返回中等分数
        return [0.5] * num_candidates, [0.5] * num_candidates
    
    # 处理所有completions - 对比评估
    contents = [completion[0]["content"] for completion in completions]
    thinkings = [extract_thinking(content) for content in contents]
    
    print(f"🔄 正在调用API对比评估 {len(contents)} 个候选答案（人物关注度 + 时序分析）...")
    
    # 一次性对比评估所有答案
    people_rewards, temporal_rewards = evaluate_combined_comparative(thinkings)
    
    # 更新缓存
    _global_cache["batch_hash"] = batch_hash
    _global_cache["people_rewards"] = people_rewards
    _global_cache["temporal_rewards"] = temporal_rewards
    
    print(f"✅ API评估完成，已缓存结果")
    
    return people_rewards, temporal_rewards


def people_focus_reward_combined(completions, **kwargs):
    """
    人物关注度 reward（使用合并API评估）
    从combined_reward_api获取第一个维度的分数
    """
    use_api = os.environ.get("USE_API_REWARD", "false").lower() == "true"
    
    if use_api:
        people_rewards, _ = combined_reward_api(completions, **kwargs)
        return people_rewards
    else:
        # 降级到简化版
        from open_r1.vlm_modules.people_focus_reward import people_focus_reward_simple
        return people_focus_reward_simple(completions, **kwargs)


def temporal_order_reward_combined(completions, **kwargs):
    """
    时序分析 reward（使用合并API评估）
    从combined_reward_api获取第二个维度的分数
    """
    use_api = os.environ.get("USE_API_REWARD", "false").lower() == "true"
    
    if use_api:
        _, temporal_rewards = combined_reward_api(completions, **kwargs)
        return temporal_rewards
    else:
        # 降级到简化版
        from open_r1.vlm_modules.temporal_order_reward import temporal_order_reward_simple
        return temporal_order_reward_simple(completions, **kwargs)
