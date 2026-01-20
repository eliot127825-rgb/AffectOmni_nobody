# Stage 4 QwenOmni GRPO 训练输出乱码问题

## 问题概述

**日期**: 2023-12-23  
**严重程度**: 🔴 Critical  
**状态**: 🔍 调查中

### 核心问题
Stage 4 QwenOmni GRPO 训练过程中，模型生成的输出全部为乱码，具体表现为：
- 模型疯狂重复输出 `"system"` token
- Reward 始终为 0.10（仅 people_focus_reward，format 和 accuracy 均为 0）
- Prompt 显示正常，但 Completion 完全无意义

### 问题示例
```
Prompt: <|im_start|>system\nYou are a helpful assistant...
Completion: <contextsystemsystemsystemsystem schönesystemsystemsystemsystem...
Reward: 0.10
```

---

## 环境信息

### 训练配置
- **模型**: `/data2/youle/HumanOmniV2/models/HumanOmniV2`
- **训练脚本**: `run_scripts/run_grpo_qwenomni_stage4_people_focus_4gpu.sh`
- **主训练代码**: `src/open_r1/grpo_qwenomni.py`
- **Trainer**: `src/open_r1/trainer/grpo_trainer.py`
- **VLM Module**: `src/open_r1/vlm_modules/qwenomni_module.py`

### 训练参数
```bash
--num_generations 2
--per_device_train_batch_size 1
--gradient_accumulation_steps 12
--max_prompt_length 2048
--max_completion_length 1024
--use_audio_in_video true
--reward_funcs format accuracy people_focus
```

### 数据类型
- **视频**: Social-IQ v2 数据集
- **音频**: 从视频中提取
- **多模态 Token**: `<|VIDEO|>`, `<|AUDIO|>`

---

## 已完成的排查和修复

### ✅ 修复 1: `prepare_prompt` 的 `[0]` 索引问题
**文件**: `src/open_r1/trainer/grpo_trainer.py:696`

**问题**: 
```python
prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)[0]
```
只取了列表的第一个元素，导致批处理时只使用第一个 prompt。

**修复**:
```python
prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)
```

### ✅ 修复 2: `use_audio_in_video` 参数传递链
**涉及文件**:
- `src/open_r1/grpo_qwenomni.py:398-401`
- `src/open_r1/trainer/grpo_trainer.py:325-351, 695-697`

**问题**: 多处硬编码 `use_audio_in_video=False`

**修复**: 
1. Trainer 的 `__init__` 方法接收 `use_audio_in_video` 参数并保存为实例属性
2. `grpo_qwenomni.py` 在初始化 Trainer 时传入 `use_audio_in_video` 参数
3. `_generate_and_score_completions` 和 `process_mm_info` 使用实例属性而非硬编码

### ✅ 修复 3: 多模态数据传递
**文件**: `src/open_r1/trainer/vllm_grpo_trainer.py:505-532`

**问题**: `_prepare_inputs` 方法只提取了 `images`，没有提取 `videos` 和 `audios`

**修复**: 添加了 `videos` 和 `audios` 的提取和传递

---

## 当前调试状态

### 已添加的调试信息

#### 1. GRPO Trainer 调试 (`grpo_trainer.py:713-721`)
```python
if self.state.global_step == 0 and self.accelerator.is_main_process:
    print(f"\n=== GRPO Trainer Debug ===")
    print(f"use_audio_in_video: {use_audio_in_video}")
    print(f"images: {type(images)}, count: {len(images) if images else 0}")
    print(f"videos: {type(videos)}, count: {len(videos) if videos else 0}")
    print(f"audios: {type(audios)}, count: {len(audios) if audios else 0}")
    print(f"prompts_text[0][:300]: {prompts_text[0][:300]}")
    print(f"===========================\n")
```

**输出结果**:
```
use_audio_in_video: True ✓
images: <class 'NoneType'>, count: 0
videos: <class 'list'>, count: 1 ✓
audios: <class 'list'>, count: 2 ✓
prompts_text[0][:300]: <|im_start|>system\nYou are a helpful assistant... ✓
```

#### 2. QwenOmni Module 调试 (`qwenomni_module.py:103-113`)
```python
if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print(f"\n=== QwenOmni Module Debug ===")
    print(f"use_audio_in_video: {use_audio_in_video}")
    print(f"images type: {type(images)}, value: {images}")
    print(f"videos type: {type(videos)}, count: {len(videos) if videos else 0}")
    print(f"audios type: {type(audios)}, count: {len(audios) if audios else 0}")
    print(f"prompts_text[0] contains VIDEO: {'VIDEO' in prompts_text[0]}")
    print(f"prompts_text[0] contains AUDIO: {'AUDIO' in prompts_text[0]}")
    print(f"===========================\n")
```

#### 3. Conversation 格式调试 (`grpo_trainer.py:697-705`)
```python
if self.state.global_step == 0 and self.accelerator.is_main_process:
    print(f"\n=== DEBUG: Input Conversation ===")
    print(f"Type of inputs[0]['prompt']: {type(inputs[0]['prompt'])}")
    if isinstance(inputs[0]['prompt'], list):
        print(f"First message: {inputs[0]['prompt'][0]}")
        if len(inputs[0]['prompt']) > 1:
            print(f"Second message keys: {inputs[0]['prompt'][1].keys()}")
    print(f"================================\n")
```

---

## 问题分析

### 观察到的现象

1. ✅ **数据预处理正常**: 
   - `use_audio_in_video` 参数正确传递
   - 视频和音频数据正确提取（1个视频，2个音频）
   - Prompt 文本完整且格式正确

2. ❌ **模型生成异常**:
   - 输出全是 "system" token 的重复
   - 夹杂大量非英语词汇碎片（schöne, sistema, など）
   - 完全无法解析出结构化标签（`<context>`, `<think>`, `<answer>`）

3. ❌ **Reward 异常**:
   - `format_reward`: 0.0（因为没有正确的标签结构）
   - `accuracy_reward`: 0.0（因为没有正确答案）
   - `people_focus_reward`: 0.10（默认值）

### 可能的根因

#### 假设 1: 多模态 Embedding 未正确传入模型
**证据**:
- Prompt 显示正常说明文本部分处理正确
- 但模型生成乱码说明可能缺少关键信息

**需要验证**:
- Processor 是否正确处理了 VIDEO/AUDIO token
- 多模态 embedding 是否被正确注入到模型输入中

#### 假设 2: `maybe_apply_chat_template` 处理多模态消息有问题
**证据**:
- Conversation 包含复杂的多模态内容（字典列表格式）
- `maybe_apply_chat_template` 可能无法正确处理这种格式

**需要验证**:
- `prompts_text` 是否包含完整的多模态 token
- Processor 收到的 `text` 参数是否正确

#### 假设 3: Processor 参数不匹配
**证据**:
- 测试脚本使用 `audio=audios` (单数)
- 训练脚本使用 `audio=audios` (单数) ✓

**状态**: 已确认参数名一致

---

## 日志和相关文件

### 训练日志位置
```
src/log/train_stage4_test_single_YYYYMMDD_HHMMSS.log
../outputs/stage4_test_single/train.log
```

### 关键代码文件
```
src/open_r1/grpo_qwenomni.py                    # 主训练脚本
src/open_r1/trainer/grpo_trainer.py             # GRPO Trainer 实现
src/open_r1/trainer/vllm_grpo_trainer.py        # vLLM Trainer (备用)
src/open_r1/vlm_modules/qwenomni_module.py      # QwenOmni 模块
scripts/test_base_model.py                      # 正常工作的测试脚本（对比参考）
```

### 对比参考
- **正常工作**: `scripts/test_base_model.py` - 能正确推理并输出结构化内容
- **异常工作**: Stage 4 GRPO 训练 - 输出乱码

---

## 下一步行动

### 🔍 待验证
1. **查看完整日志中的调试输出**
   - Conversation 的具体结构
   - Processor 接收到的参数
   - VIDEO/AUDIO token 是否存在于 `prompts_text` 中

2. **对比测试脚本和训练脚本的差异**
   - 消息构造方式
   - Processor 调用方式
   - 多模态数据传递方式

3. **检查 Processor 内部处理**
   - 是否正确识别多模态 token
   - 是否正确加载视频/音频文件
   - 是否正确生成多模态 embedding

### 🛠️ 可能的修复方向
1. **修改消息格式**: 确保 `maybe_apply_chat_template` 能正确处理多模态消息
2. **直接传递原始消息**: 跳过 `maybe_apply_chat_template`，直接使用测试脚本的消息格式
3. **添加更多调试**: 在 Processor 调用后检查生成的 `input_ids` 和 `attention_mask`

---


## 附录：警告信息

训练过程中的警告（可能无关）:
```
WARNING:root:System prompt modified, audio output may not work as expected.
UserWarning: None of the inputs have requires_grad=True. Gradients will be None
```
