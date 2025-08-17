# CS336 Spring 2025 Assignment 5: Alignment

## 项目背景

本项目是CS336 Spring 2025的第五次作业，专注于Alignment（对齐）技术的研究和应用。主要目标是探索多种alignment方法来改善语言模型在数学推理任务上的表现，包括监督微调（SFT）、Expert Iteration（EI）和GRPO等算法。

## 作业要求

### 主要任务
- 使用 gsm8k 数据集 [[Link]](https://huggingface.co/datasets/openai/gsm8k)
- 基于 Qwen 2.5 Math 1.5B base model
- 实现和比较多种alignment方法

### 实验方法
1. **监督微调 (SFT)**: 使用不同数据集大小 {128, 256, 512, 1024} + 完整数据集
2. **Expert Iteration (EI)**: 通过迭代优化提升模型性能
3. **GRPO**: 实现GRPO算法进行模型对齐

### 目标
- 调整学习率和批量大小
- 比较不同alignment方法的效果
- 交付物：不同方法在不同数据集大小下的验证准确率曲线

## 数据集

### 数据转换
源数据为GSM8K数据集，使用`####`分隔符格式。我们将其转换为结构化的`<think>推理过程</think> <answer>答案</answer>`格式，便于模型学习和评估。

转换脚本位于`data/prepare_gsm8k.py`，输出目录为`data/math/`。

### 数据分布
- 训练集：6792样本
- 验证集：1000样本  
- 测试集：1000样本

所有数据都采用统一的`<think>`格式，与原始GSM8K格式完全分离。

## 评估

### Baseline评估结果
基于Qwen 2.5 Math 1.5B的零样本评估结果：
- 数据集：MATH test set (1000样本)
- 结果文件：`evaluations/math_baseline.jsonl`
- 格式准确率：18.80%
- 答案准确率：3.60%
- 总体准确率：0.00%

### 问题分析
1. 格式问题：模型经常无法输出正确的`<think>...</think><answer>...</answer>`格式
2. 答案问题：即使格式正确，数学推理也经常出错
3. 数据格式：已完全转换为`<think>`格式，与GSM8K无关

## 代码结构

### 主要模块
- `alignment/math_baseline/`: Baseline评估
- `alignment/math_sft/`: 监督微调 (SFT)
- `alignment/math_ei/`: Expert Iteration (EI)
- `alignment/math_grpo/`: GRPO算法
- `alignment/shared/`: 共享配置和工具
- `data/math/`: 转换后的MATH数据集
- `evaluations/`: 评估结果
- `models/`: 训练后的模型

### 关键文件
- `alignment/shared/config.py`: 全局配置
- `alignment/shared/drgrpo_grader.py`: 评估函数（给定，不可修改）
- `alignment/shared/math_evaluation_utils.py`: 评估工具

## 使用方法

### 1. 数据准备
```bash
cd data
python prepare_gsm8k.py
```

### 2. Baseline评估
```bash
python -m alignment.math_baseline.eval
```

### 3. 结果分析
```bash
python -m alignment.math_baseline.analyze
```

### 4. 监督微调 (SFT)
```bash
python -m alignment.math_sft.run_sft
```

### 5. Expert Iteration (EI)
```bash
python -m alignment.math_ei.run_ei
```

### 6. GRPO训练
```bash
python -m alignment.math_grpo.run_grpo
```

## 技术细节

### 评估函数
使用给定的`r1_zero_reward_fn`，支持`<think>...</think><answer>...</answer>`格式，分别评估格式正确性和答案正确性。

### 数据格式
- 问题：数学问题文本
- 答案：`<think>推理过程</think><answer>最终答案</answer>`
- 推理过程：包含中间计算步骤和结果

### 模型配置
- 基础模型：Qwen 2.5 Math 1.5B
- 训练框架：PyTorch + Transformers
- 优化器：AdamW
- 混合精度：bfloat16

## 下一步计划

1. **SFT实验**：使用不同数据集大小进行训练，调整学习率和批量大小，监控训练过程中的准确率变化
2. **Expert Iteration**：实现EI算法，通过迭代优化提升模型性能
3. **GRPO实现**：完成GRPO算法的实现和训练
4. **方法对比**：比较不同alignment方法的效果，分析各自的优势和适用场景
5. **结果分析**：总结最佳实践和发现，为后续研究提供指导

---

*项目状态：数据集准备完成，Baseline评估完成，准备开始多种alignment方法的实验*

