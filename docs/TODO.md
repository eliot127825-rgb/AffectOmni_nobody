# 时空增强改进 - 任务清单

## ✅ 已完成

- [x] 创建项目目录结构
- [x] 编写改进方案文档 (IMPROVEMENT_PLAN.md)

---

## 📋 待完成任务

### Stage 0: 环境准备

- [ ] 检查GPU资源 (8×A100)
- [ ] 确认基座模型路径
- [ ] 检查Python依赖
- [ ] 安装SAM3工具

### Stage 1: 数据重构

- [ ] 复制原始数据集到 `data/` 目录
  - [ ] emer_rewrite.json
  - [ ] social_iq_v2_rewrite.json
- [ ] 编写数据标注脚本 (`scripts/annotate_data.py`)
- [ ] 运行自动标注
- [ ] 人工验证10-20个样本
- [ ] 修正标注错误

### Stage 2: 模型微调

- [ ] 创建训练配置文件
  - [ ] `configs/spatiotemporal_stage1.yaml`
  - [ ] `configs/train_config.sh`
- [ ] 修改训练脚本
- [ ] 启动LoRA SFT训练
- [ ] 监控训练进度
- [ ] 保存最佳checkpoint

### Stage 3: 工具集成

- [ ] 安装SAM3
  - [ ] 克隆仓库
  - [ ] 安装依赖
  - [ ] 下载模型权重
- [ ] 编写工具封装
  - [ ] `tools/sam3_tool.py`
  - [ ] `scripts/parse_reasoning.py`
- [ ] 实现推理pipeline
  - [ ] `scripts/inference_with_tools.py`
- [ ] 测试工具集成

### Stage 4: 测试评估

- [ ] 功能测试
  - [ ] 测试时空标记生成
  - [ ] 测试SAM3分割
  - [ ] 测试完整pipeline
- [ ] 效果评估
  - [ ] 编写评估脚本 (`scripts/evaluate.py`)
  - [ ] 运行评估
  - [ ] 分析结果
- [ ] 问题修复

### Stage 5: 优化部署 (可选)

- [ ] GRPO强化学习
  - [ ] 生成Rollout数据
  - [ ] 运行GRPO训练
- [ ] 模型合并
- [ ] 性能优化

---

## 📝 当前进度

**当前阶段**: Stage 0 - 环境准备

**下一步**: 检查环境和安装SAM3

---

## 🐛 问题记录

(暂无)

---

## 💡 备注

- 训练预计耗时: 2周
- GPU需求: 8×A100
- 数据量: ~150个样本 (初期验证)
