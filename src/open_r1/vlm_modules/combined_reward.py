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
    
    def evaluate_combined(thinking_text):
        """使用 Qwen API 同时评估人物关注度和时序分析"""
        prompt = f"""请同时评估以下推理文本的两个维度：

【维度1：人物关注度】
评估推理文本是否充分关注了视频中的**人物**（动作、表情、肢体语言、交互关系）。

评分标准（0-10分）：
- 10分：非常详细地描述人物的动作、表情、肢体语言、交互关系，几乎每个观察都与人物相关
- 7-9分：较多地关注人物，描述了多个人物相关的细节
- 4-6分：有提到人物，但同时关注了较多环境、物体等非人物因素
- 1-3分：很少提到人物，主要描述环境、物体或其他内容
- 0分：完全没有关注人物

【维度2：时序分析】
评估推理文本是否**按照视频的时间顺序**进行分析。

评分标准（0-10分）：
- 10分：非常清晰地按照时间顺序（开始→中间→结束）分析，使用了明确的时序标记（如"首先"、"然后"、"接着"、"最后"），对视频不同时间段的内容进行了分步描述
- 7-9分：较好地体现了时序性，分析了视频不同阶段的变化，有一定的时序标记
- 4-6分：有提到时间相关的内容，但分析较为混乱，没有清晰的时间线索
- 1-3分：基本没有时序分析，主要是静态描述或整体概括
- 0分：完全没有体现时间顺序，纯静态分析

推理文本：
{thinking_text[:800]}

请按照以下格式返回评分（只返回两个数字，用逗号分隔）：
人物关注度分数,时序分析分数

示例：8,7"""

        try:
            response = call_qwen_api(prompt)
            if response:
                # 提取两个数字
                numbers = re.findall(r'\d+', response)
                if len(numbers) >= 2:
                    people_score = max(0, min(10, int(numbers[0]))) / 10.0
                    temporal_score = max(0, min(10, int(numbers[1]))) / 10.0
                    return people_score, temporal_score
        except Exception as e:
            print(f"API评估失败: {e}")
        
        # 失败时返回中等分数
        return 0.5, 0.5
    
    # 处理每个completion
    contents = [completion[0]["content"] for completion in completions]
    people_rewards = []
    temporal_rewards = []
    
    print(f"🔄 正在调用API评估 {len(contents)} 个候选答案（人物关注度 + 时序分析）...")
    
    for idx, content in enumerate(contents):
        thinking = extract_thinking(content)
        people_score, temporal_score = evaluate_combined(thinking)
        people_rewards.append(people_score)
        temporal_rewards.append(temporal_score)
        if (idx + 1) % 5 == 0:
            print(f"  已完成 {idx + 1}/{len(contents)} 个评估")
    
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
