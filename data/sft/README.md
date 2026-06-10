# HotpotQA-MAS SFT 数据集

## 概述

本数据集包含 **3071** 个 SFT 训练样本，从 **375** 个成功的 HotpotQA 多智能体任务中生成。

## 生成方法

### 核心原则：无 Oracle 信息
- **Main** 生成子任务时**不指定文档 ID**
- **Sub** 通过**多轮搜索**动态发现相关文档
- 只保留 Sub 成功找到所有 support docs 的任务（成功率 75%）

### 搜索策略
1. Sub 使用完整问题作为初始搜索查询
2. 从搜索结果中选择标题重叠度最高的文档读取
3. 从已读文档内容中提取新关键词（答案词或 support title 词）
4. 用新关键词继续搜索，直到找到所有 support docs

## 数据格式

每个样本是一个 JSON 对象，包含：
- `messages`: 3 轮对话 [system, user, assistant]
- `category`: "main" 或 "sub"
- `stage`: "plan", "action", "summary", 或 "answer"
- `task_type`: "bridge" 或 "comparison"

## 样本分布

| Category | Stage | 数量 |
|----------|-------|------|
| main | plan | 375 |
| main | answer | 375 |
| sub | action | 1943 |
| sub | summary | 378 |

| Task Type | 数量 |
|-----------|------|
| bridge | 2672 |
| comparison | 399 |

## 统计信息

- 总样本数: 3071
- 成功任务数: 375/500 (75.0%)
- 平均每任务样本数: 8.2
- 总字符数: 6,090,070
- 估计总 token 数: ~1,522,517
- Search/Read 比例: 0.83:1
- 每任务平均搜索轮数: 2.35

## 使用方式

```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='data/sft/hotpotqa_correct_sft_data.jsonl')

# 按 category/stage 过滤
plan_data = dataset.filter(lambda x: x['category'] == 'main' and x['stage'] == 'plan')
action_data = dataset.filter(lambda x: x['category'] == 'sub' and x['stage'] == 'action')
```

## 训练建议

1. **SFT 阶段**: 使用全部 3071 个样本进行监督微调
2. **GRPO 阶段**: 在 SFT 基础上，用剩余 125 个困难任务进行强化学习
3. **数据增强**: 可以对成功任务进行子任务重组，生成更多变体
