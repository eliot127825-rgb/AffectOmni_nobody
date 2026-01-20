# Stage 4: 人物关注增强 GRPO 训练

## 📋 概述

基于训练好的 HumanOmniV2 模型，通过 GRPO 强化学习进一步增强模型对视频中**人物**的关注能力。

### 训练目标
- ✅ 提升模型对人物动作、表情、肢体语言的描述能力
- ✅ 增强人物交互和社交关系的理解
- ✅ 提高 IntentBench 等人物中心测试集的分数

### 训练策略
- **数据组成**: 90% 人物中心数据 + 10% 通用数据（防止遗忘）
- **Reward 函数**: `format` + `accuracy` + `people_focus`（新增）
- **训练参数**: 极保守策略（低学习率 + 大梯度累积）

---

## 📁 文件结构

```
spatio-temporal-reasoner/
├── src/
│   ├── data_config/
│   │   └── stage4_people_focus.yaml          # Stage 4 数据配置
│   ├── run_scripts/
│   │   └── run_grpo_qwenomni_stage4_people_focus.sh  # 训练脚本
│   └── src/open_r1/vlm_modules/
│       ├── qwenomni_module.py                # 主模块（已集成 people_focus）
│       └── people_focus_reward.py            # 人物关注度 reward 函数
├── outputs/                                  # 训练输出目录
└── STAGE4_README.md                          # 本文档
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
cd /data2/youle/HumanOmniV2/spatio-temporal-reasoner/src
conda activate humanomniv2
```

### 2. 数据准备（已完成）

数据配置文件 `data_config/stage4_people_focus.yaml` 已自动包含：
- **Social-IQ** (50%): 社交互动理解
- **EMER** (30%): 情绪识别
- **Video-R1 sample** (20%): 通用能力保持

### 3. 启动训练

```bash
# 8卡训练（推荐）
bash run_scripts/run_grpo_qwenomni_stage4_people_focus.sh

# 自定义卡数（例如4卡）
bash run_scripts/run_grpo_qwenomni_stage4_people_focus.sh 1 4
```

### 4. 监控训练

```bash
# 查看实时日志
tail -f ../outputs/stage4_people_focus/train.log

# 检查GPU使用
nvidia-smi
```

---

## ⚙️ 配置说明

### 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `model_name_or_path` | `/data2/youle/HumanOmniV2/models/HumanOmniV2` | 起点模型 |
| `learning_rate` | `5e-7` | **极低学习率**（防止破坏原有能力） |
| `num_train_epochs` | `1` | 只训练1个epoch |
| `gradient_accumulation_steps` | `8` | 大梯度累积（稳定训练） |
| `per_device_train_batch_size` | `1` | 每卡batch size |
| `num_generations` | `8` | 每个问题生成8个候选答案 |

### Reward 函数

```python
total_reward = format_reward + accuracy_reward + people_focus_reward
```

- **`format`**: 检查输出是否包含 `<context><think><answer>` 标签
- **`accuracy`**: 检查答案是否正确
- **`people_focus`**: **新增！** 评估是否关注人物

---

## 🎯 People Focus Reward 说明

### 简化版（默认，推荐）

基于关键词统计，**不需要 API**，速度快。

**评估标准**:
- 统计人物相关关键词（person, facial, gesture, interaction...）
- 统计动作词（talk, walk, smile...）
- 惩罚过多环境词（background, setting...）

**评分范围**: 0.0 - 1.0

### API 版（可选，更准确）

使用 Qwen API 评估人物关注度。

**启用方法**:
```bash
export USE_API_REWARD=true
export API=<qwen_api_endpoint>
export API_KEY=<your_api_key>
```

**评估标准**:
- 使用大模型判断推理过程是否关注人物
- 10分制评分，归一化到 [0, 1]

---

## 📊 预期效果

| 指标 | 当前 | Stage 4 目标 | 提升 |
|------|------|------------|------|
| **IntentBench 准确率** | 基线 | +10-15% | ✅ |
| **人物关注度评分** | 6/10 | 8.5/10 | ✅ |
| **通用指令能力** | 33% | 保持不降 | ✅ |

### 训练时间估算

```
数据量: ~1500 样本
训练硬件: 8x A800 (80GB)
预计时间: 3-5 天
```

---

## 🔍 训练监控

### 关键指标

```bash
# 在训练日志中查找
grep "Reward" ../outputs/stage4_people_focus/train.log

# 关注以下指标：
- Total Reward: 应该逐步上升
- People Focus Reward: 应该从 ~0.3 上升到 ~0.7+
- Accuracy: 应该保持稳定或上升
```

### Checkpoint 保存

```
outputs/stage4_people_focus/
├── checkpoint-100/     # 第100步
├── checkpoint-200/     # 第200步
├── checkpoint-300/     # 第300步
└── ...
```

---

## ⚠️ 常见问题

### Q1: 显存不足

**解决方案**:
```bash
# 减少 num_generations
--num_generations 4  # 从8改为4

# 或增加梯度累积
--gradient_accumulation_steps 16  # 从8改为16
```

### Q2: 训练不稳定

**解决方案**:
```bash
# 降低学习率
--learning_rate 1e-7  # 从5e-7改为1e-7

# 或增加梯度累积
--gradient_accumulation_steps 16
```

### Q3: People Focus Reward 一直很低

**检查**:
1. 数据是否正确加载（应该是人物中心的数据）
2. Reward 函数是否正确导入
3. 尝试启用 API 版 reward（更准确）

**调试**:
```bash
# 启用调试模式
export DEBUG_MODE=true
export LOG_PATH=./debug_log_stage4.txt

# 查看详细日志
tail -f debug_log_stage4.txt
```

---

## 📈 评估训练结果

### 1. 快速评估（指令遵循测试）

```bash
cd /data2/youle/HumanOmniV2/spatio-temporal-reasoner/scripts

# 修改 test_instruction_following.py 中的模型路径
# MODEL_PATH = "/data2/youle/HumanOmniV2/spatio-temporal-reasoner/outputs/stage4_people_focus/checkpoint-XXX"

python test_instruction_following.py
```

### 2. 完整评估（IntentBench）

```bash
cd /data2/youle/HumanOmniV2/spatio-temporal-reasoner/src

# 评估最终模型
python eval/eval_humanomniv2.py \
    --model-path ../outputs/stage4_people_focus \
    --dataset ib
```

---

## 💡 优化建议

### 如果人物关注度提升不明显

1. **增加人物数据比例**:
   ```yaml
   # 修改 data_config/stage4_people_focus.yaml
   - Social-IQ: 60% → 70%
   - Video-R1: 20% → 10%
   ```

2. **提高 people_focus reward 权重**:
   ```python
   # 在 trainer 中调整权重（需要修改源码）
   total_reward = 0.2*format + 0.3*accuracy + 0.5*people_focus
   ```

3. **使用 API 版 reward**:
   ```bash
   export USE_API_REWARD=true
   ```

### 如果通用能力下降

1. **增加通用数据比例**:
   ```yaml
   - Video-R1: 20% → 30%
   ```

2. **降低学习率**:
   ```bash
   --learning_rate 1e-7
   ```

---

## 📚 参考文献

- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [HumanOmniV2 Paper](https://arxiv.org/abs/2506.21277)
- [Qwen2.5-Omni](https://github.com/QwenLM/Qwen2.5-Omni)

---

## 🎬 下一步

训练完成后：

1. ✅ 评估 IntentBench 分数
2. ✅ 对比原模型的人物关注度
3. ✅ 撰写论文实验部分
4. ✅ 准备模型发布

---

**训练愉快！如有问题，请检查日志或联系开发者。** 🚀
