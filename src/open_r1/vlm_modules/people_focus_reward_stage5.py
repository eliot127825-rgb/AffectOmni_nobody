"""
人物关注度 Reward 函数
用于 Stage 4 GRPO 训练，评估模型输出是否充分关注人物
"""

import re
import requests
import os
import time

# Qwen API 配置（用于评估人物关注度）
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


def people_focus_reward_simple(completions, **kwargs):
    """
    简化版人物关注度 reward（基于关键词统计，不需要API）
    适合快速训练和调试
    
    评估标准：
    - 检测人物相关关键词的数量和密度
    - 评分范围：0.0 - 1.0
    """
    
    def extract_thinking(text):
        """提取 <think> 部分"""
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def count_people_focus(text):
        """统计人物关注度相关特征"""
        text_lower = text.lower()
        
        # 人物相关关键词（权重高）
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
        
        # 动作词（人物相关）
        action_keywords = [
            'walk', 'run', 'sit', 'stand', 'move',
            'talk', 'speak', 'say', 'ask', 'answer',
            'hold', 'touch', 'point', 'wave',
            'laugh', 'cry', 'nod', 'shake'
        ]
        
        # 环境词（权重低，过多会降低分数）
        environment_keywords = [
            'background', 'setting', 'location', 'place',
            'room', 'building', 'outdoor', 'indoor',
            'sky', 'ground', 'wall', 'floor'
        ]
        
        # 统计关键词
        people_count = sum(1 for kw in people_keywords if kw in text_lower)
        action_count = sum(1 for kw in action_keywords if kw in text_lower)
        env_count = sum(1 for kw in environment_keywords if kw in text_lower)
        
        # 计算分数
        # 人物词 + 动作词 - 过多环境词
        people_score = (people_count * 1.0 + action_count * 0.5) / 20.0  # 归一化
        env_penalty = max(0, (env_count - 3) * 0.1)  # 环境词超过3个开始惩罚
        
        score = max(0.0, min(1.0, people_score - env_penalty))
        
        return score
    
    # 处理每个completion
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in contents:
        thinking = extract_thinking(content)
        score = count_people_focus(thinking)
        rewards.append(score)
    
    return rewards


def people_focus_reward_api(completions, **kwargs):
    """
    高级版人物关注度 reward（使用 Qwen API 评估）
    更准确但更慢，需要配置 API 环境变量
    
    环境变量：
    - DASHSCOPE_API_KEY: Qwen API key
    
    评估标准：
    - 使用大模型判断推理过程是否关注人物
    - 评分范围：0.0 - 1.0
    """
    
    # 检查 API 配置
    if not api_key:
        print("⚠️  警告：未配置 DASHSCOPE_API_KEY，降级使用简化版 reward")
        return people_focus_reward_simple(completions, **kwargs)
    
    def extract_thinking(text):
        """提取 <think> 部分"""
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def evaluate_people_focus_comparative(thinkings_list):
        """使用 Qwen API 对比评估所有答案的人物关注度
        
        Args:
            thinkings_list: List[str], 所有候选答案的thinking文本
            
        Returns:
            scores: List[float], 人物关注度分数（0-1）
        """
        num_candidates = len(thinkings_list)
        
        # 构建对比评估prompt
        candidates_text = ""
        for i, thinking in enumerate(thinkings_list, 1):
            candidates_text += f"\n【候选答案{i}】\n{thinking[:600]}\n"
        
        prompt = f"""请对比评估以下{num_candidates}个候选答案在**人物关注度**维度上的质量。

{candidates_text}

【评估标准】
评估哪个答案更充分地关注了视频中的**人物**（动作、表情、肢体语言、交互关系）。

请为每个候选答案打分（0-10分），参考以下详细标准：

- **10分**：非常详细地描述人物的动作、表情、肢体语言、交互关系，几乎每个观察都与人物相关
- **7-9分**：较多地关注人物，描述了多个人物相关的细节
- **4-6分**：有提到人物，但同时关注了较多环境、物体等非人物因素
- **1-3分**：很少提到人物，主要描述环境、物体或其他内容
- **0分**：完全没有关注人物

**重要提示**：
1. 请基于以上标准进行相对比较，分数要有明显区分度
2. 最好的答案应接近10分，最差的应接近0分，中间答案按质量分布
3. 避免所有答案分数都集中在5-7分

请按以下格式返回（每行一个候选答案，只返回分数）：
答案1: 分数
答案2: 分数
...

示例：
答案1: 9.2
答案2: 5.3
答案3: 2.4
答案4: 7.6"""

        try:
            response = call_qwen_api(prompt)
            if response:
                # 解析所有候选答案的分数
                scores = []
                
                # 提取每一行的分数
                lines = response.strip().split('\n')
                for line in lines:
                    # 匹配格式：答案X: 数字
                    numbers = re.findall(r'\d+', line)
                    if len(numbers) >= 1:
                        score = max(0, min(10, int(numbers[0]))) / 10.0
                        scores.append(score)
                
                # 如果成功解析了足够的分数
                if len(scores) == num_candidates:
                    return scores
                    
        except Exception as e:
            print(f"API对比评估失败: {e}")
        
        # 失败时返回中等分数
        return [0.5] * num_candidates
    
    def count_people_focus_simple(text):
        """简化版（API失败时的fallback）"""
        text_lower = text.lower()
        people_keywords = ['person', 'people', 'man', 'woman', 'facial', 'expression', 
                          'gesture', 'interaction', 'emotion']
        count = sum(1 for kw in people_keywords if kw in text_lower)
        return min(1.0, count / 10.0)
    
    # 处理所有completions - 对比评估
    contents = [completion[0]["content"] for completion in completions]
    thinkings = [extract_thinking(content) for content in contents]
    
    print(f"🔄 正在调用API对比评估 {len(contents)} 个候选答案（人物关注度）...")
    
    # 一次性对比评估所有答案
    rewards = evaluate_people_focus_comparative(thinkings)
    
    print(f"✅ 人物关注度评估完成")
    
    return rewards


# 默认使用简化版（更快，不需要API）
def people_focus_reward(completions, **kwargs):
    """
    人物关注度 reward（默认使用简化版）
    
    如果需要使用 API 版本，请设置环境变量：
    - export USE_API_REWARD=true
    - export API=<qwen_api_endpoint>
    - export API_KEY=<your_api_key>
    """
    use_api = os.environ.get("USE_API_REWARD", "false").lower() == "true"
    
    if use_api:
        return people_focus_reward_api(completions, **kwargs)
    else:
        return people_focus_reward_simple(completions, **kwargs)
