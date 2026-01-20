"""
时序分析 Reward 函数
用于 Stage 4 GRPO 训练，评估模型是否按照视频时间顺序分析内容
"""

import re
import os
import time

# Qwen API 配置
api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("API_KEY", "")

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


def temporal_order_reward_simple(completions, **kwargs):
    """
    简化版时序分析 reward（基于时序关键词检测，不需要API）
    适合快速训练和调试
    
    评估标准：
    - 检测时序标记词的出现和分布
    - 评分范围：0.0 - 1.0
    """
    
    def extract_thinking(text):
        """提取 <think> 部分"""
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def check_temporal_order(text):
        """检测时序分析特征"""
        text_lower = text.lower()
        
        # 时序标记词（强时序感）
        temporal_markers = [
            # 开始
            'first', 'initially', 'at the beginning', 'at the start', 'opening',
            # 进行中
            'then', 'next', 'after', 'following', 'subsequently', 'later',
            'meanwhile', 'during', 'while', 'as', 'when',
            # 结束
            'finally', 'eventually', 'at the end', 'lastly', 'concluding',
            # 时间点
            'second', 'minute', 'moment', 'timestamp',
            # 序列
            'before', 'after', 'sequence', 'progression', 'chronological'
        ]
        
        # 时间段描述
        time_phrases = [
            'at 0:', 'at 1:', 'at 2:', 'at 3:', 'at 4:', 'at 5:',  # 时间戳
            'in the first', 'in the second', 'in the third',
            'early in', 'middle of', 'towards the end',
            'throughout the video'
        ]
        
        # 非时序词（会降低分数）
        non_temporal = [
            'overall', 'in general', 'static', 'always', 'entire',
            'whole video', 'throughout without change'
        ]
        
        # 统计时序标记
        temporal_count = sum(1 for marker in temporal_markers if marker in text_lower)
        time_phrase_count = sum(1 for phrase in time_phrases if phrase in text_lower)
        non_temporal_count = sum(1 for word in non_temporal if word in text_lower)
        
        # 检测是否有明确的段落划分（分步分析）
        step_indicators = len(re.findall(r'\n\s*\d+[\.\):]|\n\s*-\s+', text))  # 数字列表或破折号
        
        # 计算分数
        # 时序标记词 + 时间短语 + 步骤划分 - 非时序词惩罚
        temporal_score = (temporal_count * 0.8 + time_phrase_count * 1.5 + step_indicators * 0.3) / 15.0
        non_temporal_penalty = non_temporal_count * 0.1
        
        score = max(0.0, min(1.0, temporal_score - non_temporal_penalty))
        
        return score
    
    # 处理每个completion
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in contents:
        thinking = extract_thinking(content)
        score = check_temporal_order(thinking)
        rewards.append(score)
    
    return rewards


def temporal_order_reward_api(completions, **kwargs):
    """
    高级版时序分析 reward（使用 Qwen API 评估）
    更准确但更慢，需要配置 API 环境变量
    
    环境变量：
    - DASHSCOPE_API_KEY: Qwen API key
    
    评估标准：
    - 使用大模型判断推理是否按照时间顺序展开
    - 评分范围：0.0 - 1.0
    """
    
    # 检查 API 配置
    if not api_key:
        print("⚠️  警告：未配置 DASHSCOPE_API_KEY，降级使用简化版 reward")
        return temporal_order_reward_simple(completions, **kwargs)
    
    def extract_thinking(text):
        """提取 <think> 部分"""
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def evaluate_temporal_order(thinking_text):
        """使用 Qwen API 评估时序分析质量"""
        prompt = f"""请评估以下推理文本是否**按照视频的时间顺序**进行分析。

评分标准（0-10分）：
- 10分：非常清晰地按照时间顺序（开始→中间→结束）分析，使用了明确的时序标记（如"首先"、"然后"、"接着"、"最后"），对视频不同时间段的内容进行了分步描述
- 7-9分：较好地体现了时序性，分析了视频不同阶段的变化，有一定的时序标记
- 4-6分：有提到时间相关的内容，但分析较为混乱，没有清晰的时间线索
- 1-3分：基本没有时序分析，主要是静态描述或整体概括
- 0分：完全没有体现时间顺序，纯静态分析

关键考察点：
1. 是否明确区分了视频的不同时间段（开始/中间/结束）
2. 是否使用了时序连接词（首先、然后、接着、最后、之后等）
3. 是否描述了随时间发生的变化或动作序列
4. 是否避免了纯静态的整体描述

推理文本：
{thinking_text[:800]}

请只返回分数（0-10的整数），不要有其他文字。"""

        try:
            response = call_qwen_api(prompt)
            if response:
                # 提取数字
                score_match = re.search(r'\d+', response)
                if score_match:
                    score = int(score_match.group())
                    return max(0, min(10, score)) / 10.0  # 归一化到 [0, 1]
        except Exception as e:
            print(f"API评估失败: {e}")
        
        # 失败时降级到简化版
        return check_temporal_simple(thinking_text)
    
    def check_temporal_simple(text):
        """简化版（API失败时的fallback）"""
        text_lower = text.lower()
        temporal_keywords = ['first', 'then', 'next', 'after', 'finally', 
                            'initially', 'subsequently', 'beginning', 'end']
        count = sum(1 for kw in temporal_keywords if kw in text_lower)
        return min(1.0, count / 8.0)
    
    # 处理每个completion
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in contents:
        thinking = extract_thinking(content)
        score = evaluate_temporal_order(thinking)
        rewards.append(score)
    
    return rewards


# 默认使用简化版（更快，不需要API）
def temporal_order_reward(completions, **kwargs):
    """
    时序分析 reward（默认使用简化版）
    
    如果需要使用 API 版本，请设置环境变量：
    - export USE_API_REWARD=true
    """
    use_api = os.environ.get("USE_API_REWARD", "false").lower() == "true"
    
    if use_api:
        return temporal_order_reward_api(completions, **kwargs)
    else:
        return temporal_order_reward_simple(completions, **kwargs)
