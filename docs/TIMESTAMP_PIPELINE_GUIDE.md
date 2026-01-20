# 时间戳后处理 Pipeline 使用指南

## 🎯 概述

这是一个**零标注、零训练**的时间戳自动对齐方案，通过后处理为模型生成的推理文本自动添加帧级时间戳。

**核心流程**：
```
视频 → 采帧 → 模型推理 → 事件提取 → CLIP匹配 → 时间戳插入
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 核心依赖
pip install open-clip-torch  # CLIP 模型
pip install decord           # 快速视频读取（推荐）
# 或者使用 opencv-python 作为备选

# 如果使用原版 CLIP
pip install git+https://github.com/openai/CLIP.git
```

### 2. 运行测试

```bash
cd /data2/youle/HumanOmniV2/spatio-temporal-reasoner/scripts
python test_timestamp_pipeline.py
```

**预期输出**：
- 视频信息和采样帧数
- 原始 `<think>` 内容
- 提取的事件列表
- 每个事件匹配的帧号和时间戳
- 带时间戳的最终 `<think>` 内容

### 3. 查看结果

结果会保存在 `../logs/timestamp_pipeline_YYYYMMDD_HHMMSS.json`

## 📚 模块说明

### tools/video_utils.py

**功能**：统一的视频采帧

```python
from tools.video_utils import sample_frames

frames_pil, frame_ids, timestamps, fps = sample_frames(
    video_path="video.mp4",
    max_frames=32,      # 与模型推理一致
    strategy="uniform"  # 均匀采样
)
```

**关键**：`max_frames` 必须与模型推理时一致！

---

### tools/clip_matcher.py

**功能**：CLIP 图文匹配 + 单调约束

```python
from tools.clip_matcher import CLIPMatcher, match_with_monotonic_constraint

# 初始化
matcher = CLIPMatcher(model_name="ViT-B-32", device="cuda")

# 方法1: 独立匹配
matches = matcher.match_events_to_frames(
    events=["woman picks up rose", "man smiles"],
    frames_pil=frames_pil
)
# 结果: {"woman picks up rose": 3, "man smiles": 8}

# 方法2: 单调约束（推荐）
similarity_matrix = matcher.get_similarity_matrix(events, frames_pil)
best_frames = match_with_monotonic_constraint(
    similarity_matrix,
    lambda_smooth=0.3  # 平滑系数
)
# 结果: [3, 8]（保证非递减）
```

**参数调优**：
- `lambda_smooth=0.1~0.5`: 越大越平滑（避免跳跃）
- CLIP 模型：`ViT-B-32`（快）vs `ViT-L-14`（准）

---

### scripts/extract_events.py

**功能**：从 `<think>` 提取关键事件

```python
from extract_events import extract_events

# 方法1: LLM 提取（推荐）
events = extract_events(
    think_text=think,
    method="llm",
    model=model,
    processor=processor,
    max_events=10
)

# 方法2: 规则提取（Fallback）
events = extract_events(
    think_text=think,
    method="rule",
    max_events=10
)

# 结果: [Event(anchor="...", query="..."), ...]
```

**Event 结构**：
- `anchor`: 用于在原文中定位（保持原句）
- `query`: 用于 CLIP 匹配（短、视觉化）

---

### scripts/insert_timestamps.py

**功能**：将时间戳插入原文

```python
from insert_timestamps import insert_timestamps

result = insert_timestamps(
    think_text=original_think,
    events=events,
    frame_matches={"woman picks up rose": 3, ...},
    timestamps=[0.0, 1.0, 2.0, ...],
    format_style="frame_and_time"  # [Frame 3: 3.00s]
)
```

**格式选项**：
- `"frame_and_time"`: `[Frame 3: 3.00s]`
- `"frame_only"`: `[Frame 3]`
- `"time_only"`: `[3.00s]`

---

## 🔧 参数配置

### 推荐配置（test_timestamp_pipeline.py）

```python
MAX_FRAMES = 32              # 视频采样帧数（与推理一致）
CLIP_MODEL = "ViT-B-32"      # CLIP 模型
USE_MONOTONIC_CONSTRAINT = True  # 使用单调约束
LAMBDA_SMOOTH = 0.3          # 平滑系数
```

### 调优建议

| 问题 | 调整 |
|------|------|
| 时间戳跳跃太大 | 增大 `LAMBDA_SMOOTH` (0.3 → 0.5) |
| 匹配不准确 | 换用 `ViT-L-14` 或增加 `max_frames` |
| 事件提取不全 | 增大 `max_events` 或改进事件提取 Prompt |
| 插入失败率高 | 检查 anchor 是否在原文中（可能需要改进事件提取）|

---

## 📊 效果评估

### 评估指标

1. **时间戳插入率**: 成功插入的事件 / 提取的事件
   - 目标: ≥ 70%
   
2. **时序一致性**: 检查是否有时间倒流
   - 使用单调约束应 100% 满足

3. **匹配准确性**: 人工检查帧号是否合理
   - 粗粒度（±2帧）: 80-90%
   - 精确（准确帧）: 60-70%

### 测试命令

```bash
# 单个样本测试
python test_timestamp_pipeline.py

# 批量测试（需自己实现）
python batch_test_timestamps.py --num_samples 20
```

---

## 🐛 常见问题

### Q1: 事件提取失败怎么办？

**A**: 检查 `<think>` 内容：
- 如果 `<think>` 太短或不包含事件描述 → 改进 System Prompt
- 如果 LLM 提取失败 → 会自动 fallback 到规则方法
- 如果规则方法也不行 → 可能需要手动调整关键词列表

### Q2: CLIP 匹配不准确？

**A**: 可能的原因和解决方案：
1. **事件描述太抽象** → 改进 query 短语（更具体、更视觉化）
2. **视频质量差** → 增加采样帧数 `max_frames`
3. **CLIP 模型不够强** → 换用 `ViT-L-14`

### Q3: 时间戳插入失败？

**A**: 检查日志中的 `verification` 结果：
- 如果 `insertion_rate < 0.5` → anchor 与原文不匹配
  - 解决：改进事件提取的 anchor 质量
  - 或者使用更宽松的模糊匹配

### Q4: 没有 decord 怎么办？

**A**: 自动 fallback 到 cv2（opencv-python）
```bash
pip install opencv-python
```

### Q5: 内存不足？

**A**: 
1. 减少 `max_frames` (32 → 16)
2. 使用更小的 CLIP 模型 (`ViT-B-32` → `RN50`)
3. 批量处理时逐个处理样本

---

## 🎨 进阶使用

### 自定义事件提取 Prompt

编辑 `scripts/extract_events.py` 中的 `extract_events_with_llm` 函数：

```python
prompt = f"""Your custom prompt here...

Extract visual events from:
{think_text}

Output JSON:
{{"events": [...]}}
"""
```

### 集成到生产流程

```python
# 在你的推理脚本中
from tools.video_utils import sample_frames
from tools.clip_matcher import CLIPMatcher
from extract_events import extract_events
from insert_timestamps import insert_timestamps

# 1. 推理
think = run_inference(...)

# 2. 采帧
frames, _, timestamps, _ = sample_frames(video_path, max_frames=32)

# 3. 事件提取
events = extract_events(think, method="llm", model=model, processor=processor)

# 4. CLIP 匹配
matcher = CLIPMatcher()
matches = matcher.match_events_to_frames([e.query for e in events], frames)

# 5. 插入时间戳
result = insert_timestamps(think, events, matches, timestamps)
```

---

## 📈 性能参考

**测试环境**: A100 40GB

| 阶段 | 耗时 | 备注 |
|------|------|------|
| 视频采帧 (32 帧) | ~0.2s | decord |
| 模型推理 | ~5-10s | HumanOmniV2 |
| 事件提取 (LLM) | ~2-3s | 生成短文本 |
| CLIP 匹配 (10 事件) | ~0.1s | ViT-B-32 |
| 时间戳插入 | <0.01s | 纯字符串操作 |
| **总计** | **~8-13s** | 单个样本 |

---

## 📝 TODO

- [ ] 批量测试脚本
- [ ] 自动化评估指标
- [ ] 可视化工具（显示帧+时间戳）
- [ ] SAM3 mask 集成
- [ ] 性能优化（批处理、缓存）

---

## 🤝 贡献

如果你发现问题或有改进建议，欢迎提交 Issue 或 PR！

主要改进方向：
1. 更好的事件提取算法
2. 更准确的 CLIP 匹配
3. 更鲁棒的时间戳插入
