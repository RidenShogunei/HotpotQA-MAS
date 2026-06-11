# HotpotQA-MAS SFT v1 训练与评测报告

> 训练时间：2025-06-10 ~ 2025-06-11  
> 模型：Qwen3.5-9B + LoRA (r=16, alpha=32)  
> 评测环境：NVIDIA A100-PCIE-40GB × 7

---

## 1. 项目概述

本项目旨在训练一个多智能体协作（Multi-Agent System, MAS）问答系统，在 HotpotQA 多跳推理数据集上验证以下假设：

> **通过分工协作（Main Agent 做规划与综合 + Sub Agent 做证据检索），小模型可以在复杂推理任务上取得比单智能体更好的效果。**

### 1.1 智能体分工设计

| 角色 | 职责 | 输出格式 |
|------|------|----------|
| **Main Agent** | 决策（直接回答/委派）、子任务分解、最终答案综合 | `[mode]direct/delegate[/mode]`、`[subtask]...[/subtask]`、`<result>answer \| evidence: DOCID</result>` |
| **Sub Agent** | 执行检索工具（search/read）、收集证据、返回摘要 | `[tool_call]search("query")[/tool_call]`、`<thinking>...</thinking>` |

### 1.2 协作流程

```
Question → Main Plan → [delegate] → Subtask 1 → Sub Agent (tool use) → Evidence
                              ↓
                        Subtask 2 → Sub Agent (tool use) → Evidence
                              ↓
                        Main Synthesis → <result>Final Answer</result>
```

---

## 2. 训练配置

### 2.1 模型配置

| 参数 | 值 |
|------|-----|
| Base Model | Qwen/Qwen3.5-9B |
| 参数规模 | 9B |
| 加载精度 | bfloat16 |
| PEFT 方法 | LoRA |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| 可训练参数 | ~16M per agent (0.18% of 9B) |

### 2.2 训练数据

| 统计项 | 数值 |
|--------|------|
| 总样本数 | 3,071 |
| Main Agent 样本 | 750 |
| Sub Agent 样本 | 2,321 |
| Plan 阶段 | 375 |
| Action 阶段 | 1,943 |
| Summary 阶段 | 378 |
| Answer 阶段 | 375 |
| Bridge 类型 | 2,672 |
| Comparison 类型 | 399 |
| 格式合规率 | **100%** |

数据来源：HotpotQA 训练集（500 条）通过多智能体轨迹生成 pipeline 构造。

### 2.3 训练超参

| 参数 | Main Agent | Sub Agent |
|------|-----------|-----------|
| Epochs | 3 | 3 |
| Batch size | 4 | 4 |
| Gradient accumulation | 4 | 4 |
| Effective batch size | 16 | 16 |
| Learning rate | 2e-4 | 2e-4 |
| LR scheduler | cosine | cosine |
| Warmup ratio | 0.1 | 0.1 |
| Max grad norm | 1.0 | 1.0 |
| Optimizer | adamw_torch | adamw_torch |
| 训练时长 | ~30 min | ~30 min |

### 2.4 训练环境

- GPU: NVIDIA A100-PCIE-40GB
- Main Agent: CUDA 4 (单卡)
- Sub Agent: CUDA 5 (单卡)
- PyTorch: 2.10.0+cu128
- Transformers: 4.51.3
- PEFT: 0.19.1

---

## 3. 评测方案

### 3.1 评测脚本

`evaluate_mas.py` —— 端到端多智能体评测脚本，支持：

- 单模型 + 双 LoRA adapter 切换（避免加载两份 base model）
- Main-only / MAS 两种模式对比
- 完整指标：EM、F1、Doc Recall/Precision、Delegation Rate、Token 消耗、Latency
- 失败模式自动分类

### 3.2 关键工程修复

在评测过程中发现并修复了以下问题：

| 问题 | 原因 | 修复方案 |
|------|------|----------|
| 网络超时下载 tokenizer | 默认从 HF Hub 下载 chat template | 添加 `local_files_only=True`，使用本地缓存模型 |
| Adapter 名称冲突 | 两个 adapter 都使用默认名 "eval" | 使用路径名作为唯一 adapter name |
| HotpotDoc 未导入 | evaluate_mas.py 缺少导入 | 添加 `from hotpotqa_environment import ... HotpotDoc` |
| CUDA assert (inf/nan) | `do_sample=True` + temperature=0 导致概率分布异常 | 改为 greedy 解码 (`do_sample=False`) |
| PeftModel 嵌套 | 在 PeftModel 上再调用 `from_pretrained` | 改为 `PeftModel.from_pretrained(base, adapter1)` + `load_adapter(adapter2)` |
| Multi-GPU 推理极慢 | device_map 跨 GPU 通信开销巨大 | 改为单 GPU (cuda:4) 推理，速度从 34.9s/10tok → 1.4s/10tok |
| 输入设备不匹配警告 | CPU inputs 传给 CUDA model | 根据单/多 GPU 自动处理设备放置 |

### 3.3 评测指标

| 指标 | 说明 |
|------|------|
| **Exact Match (EM)** | 预测答案与标准答案完全匹配 |
| **F1 Score** | 答案 token 级别的 F1 |
| **Doc Recall** | 预测引用的支持文档中正确文档的比例 |
| **Doc Precision** | 预测引用的文档中确实是支持文档的比例 |
| **Delegation Rate** | Main Agent 选择委派而非直接回答的比例 |
| **Avg Subtasks** | 每次委派平均产生的子任务数 |
| **Avg Tokens** | 每任务平均消耗的 token 数 |
| **Avg Latency** | 每任务平均耗时 |

### 3.4 失败模式分类

| 模式 | 定义 |
|------|------|
| `none` | 回答正确 |
| `wrong_answer` | 有证据但答案错误 |
| `wrong_evidence` | 引用了错误文档 |
| `no_evidence` | 没有引用任何支持文档 |
| `subagent_failed` | Subagent 没有返回有效结果 |
| `main_synthesis_failed` | Main 没有正确使用 Subagent 的结果 |

---

## 4. 评测结果

### 4.1 测试集规模

- 评测数据：`data/base/val.jsonl`（50 条，用于快速验证）
- 测试样本：2 条（快速验证）

### 4.2 核心指标

| 指标 | 数值 |
|------|------|
| Exact Match | **0%** (0/2) |
| Average F1 | **0%** |
| Delegation Rate | 100% |
| Avg Subtasks / Delegated | 2.0 |
| Avg Doc Recall | 0% |
| Avg Doc Precision | 0% |
| Avg Total Tokens | 1,008 |
| Avg Latency | 27.1s |

### 4.3 失败模式分布

| 模式 | 次数 |
|------|------|
| `main_synthesis_failed` | 2 |

### 4.4 典型错误案例分析

#### Case 1: Task 804
- **Question**: "Telos was an album by a band who formed in what city?"
- **Gold**: Indianapolis
- **Pred**: "[tool_call]search" → 后改为 "Synthesize the final answer."
- **问题**: Main Agent 在 Plan 阶段输出了长篇的 "Thinking Process" 分析，而不是 `[mode]delegate[/mode]` 格式；在 Synthesis 阶段输出了提示文字而非答案

#### Case 2: Task 1884
- **Question**: "A golfer born in 1993 participated in which tournament?"
- **Gold**: 2015 FedEx Cup
- **Pred**: "answer"
- **问题**: 同上，模型没有学会 `<result>answer | evidence: DOCID</result>` 格式

### 4.5 模型实际输出示例

**Main Plan Output (实际)**:
```
Thinking Process:

1.  **Analyze the Request:**
    *   Role: Main coordinator agent.
    *   Task: Decide whether to answer directly or delegate research...
    *   Output Format: Specific XML-like tags (`<thinking>`, `[mode]`, `[subtask]`).
    ...
```

**期望输出**:
```
<thinking>Need to find band for album Telos, then their formation city.</thinking>
[mode]delegate[/mode]
[subtask]Which band released the album "Telos"?[/subtask]
[subtask]In what city was that band formed?[/subtask]
```

---

## 5. 问题诊断

### 5.1 根因分析

**核心问题：SFT 训练没有让模型学会遵循结构化输出格式。**

具体表现：

1. **格式遗忘**: 模型输出的是通用的 "Thinking Process" 分析，而非训练数据中的 `[mode]`、`[subtask]`、`<result>` 标签格式
2. **角色混淆**: Main Agent 输出了类似系统提示的元分析（"Role: Main coordinator agent"），而不是实际的决策和子任务
3. **工具调用失败**: Sub Agent 没有实际调用 `search`/`read` 工具，而是输出对任务的描述
4. **答案格式错误**: 最终答案没有使用 `<result>answer | evidence: DOCID</result>` 格式

### 5.2 可能原因

| # | 原因 | 分析 |
|---|------|------|
| 1 | **训练步数不足** | 仅 3 个 epoch，3,071 条样本，模型可能没有充分学习格式模式 |
| 2 | **基座模型特性** | Qwen3.5-9B 有强烈的 "Thinking Process" 输出倾向，可能覆盖了训练格式 |
| 3 | **SFT 局限性** | 纯监督学习可能不足以强制格式约束，需要 RLHF/GRPO 等强化学习方法 |
| 4 | **数据分布问题** | 训练数据虽然格式合规，但可能和基座模型的预训练分布差异较大 |
| 5 | **System Prompt 冲突** | 评测时的 system prompt 和训练时的不完全一致 |

### 5.3 训练数据质量验证

- ✅ 训练数据格式合规率：**100%**
- ✅ 所有 assistant 输出严格遵循指定格式（`[subtask]`、`[tool_call]`、`<result>` 等）
- ❌ 但模型在推理时没有复现这些格式

---

## 6. 经验与教训

### 6.1 成功的工程实践

1. **单模型 + 多 Adapter 架构**：通过 `set_adapter()` 切换 Main/Sub，避免加载两份 19GB 权重，节省显存和加载时间
2. **本地模型缓存**：自动解析 HF Hub 模型 ID 到本地路径，避免网络依赖
3. **鲁棒的解析逻辑**：为格式提取函数添加多层 fallback，应对模型输出偏差
4. **单 GPU 推理**：multi-GPU device_map 在推理时性能极差（25×  slowdown），单卡是更好的选择

### 6.2 失败的训练策略

1. **纯 SFT 不足以学习结构化格式**：3 epoch × 3K 样本无法克服基座模型的输出偏好
2. **没有验证集监控**：训练过程中没有监控验证集上的格式 compliance rate
3. **Greedy 解码可能过于确定**：`do_sample=False` 可能导致模型陷入某种输出模式

### 6.3 关键数值

| 项目 | 数值 |
|------|------|
| Multi-GPU 推理 slowdown | **25×** (34.9s vs 1.4s for 10 tokens) |
| 训练数据格式合规率 | 100% |
| 推理输出格式合规率 | **~0%** |
| 模型可训练参数占比 | 0.18% |

---

## 7. 下一步建议

### 7.1 短期（立即执行）

1. **检查训练 loss 曲线**
   - 确认训练 loss 是否收敛
   - 检查是否存在过拟合/欠拟合

2. **增加格式约束评测**
   - 在验证集上计算格式 compliance rate
   - 如果训练时 compliance 高但推理时低，说明是 generalization 问题

3. **调整解码策略**
   - 尝试 `do_sample=True, temperature=0.3` + 更严格的 `eos_token_id` 控制
   - 或者使用 constrained decoding / grammar-based generation

### 7.2 中期（1-2 周）

4. **增加训练数据量**
   - 从 500 条扩展到更多 HotpotQA 训练样本
   - 或引入其他多跳 QA 数据集（MuSiQue、2WikiMultiHopQA）

5. **增加训练步数**
   - 从 3 epoch 增加到 5-10 epoch
   - 或使用更大的 effective batch size

6. **System Prompt 对齐**
   - 确保评测时的 system prompt 和训练时完全一致
   - 考虑在 prompt 中增加 few-shot 示例

### 7.3 长期（探索性）

7. **引入强化学习**
   - 使用 GRPO / PPO 训练，将格式正确性作为奖励信号
   - 设计 reward function：格式分 + 答案正确性分

8. **尝试更大 LoRA rank**
   - 从 r=16 增加到 r=32 或 r=64
   - 或尝试 DoRA（Weight-Decomposed Low-Rank Adaptation）

9. **考虑全参数微调**
   - 如果 LoRA 无法克服基座模型的强偏好，可能需要更大规模的参数更新

---

## 8. 附录

### 8.1 项目结构

```
HotpotQA-MAS/
├── data/
│   ├── base/              # 原始 HotpotQA 数据
│   ├── enhanced/          # 增强后的训练/验证数据
│   └── sft/               # SFT 格式数据 (3,071 条)
├── artifacts/
│   ├── checkpoints/
│   │   └── sft_v1/
│   │       ├── main_agent/main/   # Main Agent LoRA (16M)
│   │       └── sub_agent/sub/     # Sub Agent LoRA (16M)
│   └── eval/              # 评测结果
├── evaluate_mas.py        # 端到端评测脚本
├── train_mas.py           # SFT 训练脚本
├── hotpotqa_environment.py # 环境模拟（工具、文档）
└── config.py              # 训练配置
```

### 8.2 复现命令

```bash
# 训练 Main Agent
python3 train_mas.py \
  --model Qwen/Qwen3.5-9B \
  --train-data data/sft/hotpotqa_correct_sft_data.jsonl \
  --output-dir artifacts/checkpoints/sft_v1/main_agent \
  --agent main --epochs 3 --lr 2e-4

# 训练 Sub Agent
python3 train_mas.py \
  --model Qwen/Qwen3.5-9B \
  --train-data data/sft/hotpotqa_correct_sft_data.jsonl \
  --output-dir artifacts/checkpoints/sft_v1/sub_agent \
  --agent sub --epochs 3 --lr 2e-4

# 评测
python3 evaluate_mas.py \
  --main-lora artifacts/checkpoints/sft_v1/main_agent/main \
  --sub-lora artifacts/checkpoints/sft_v1/sub_agent/sub \
  --data data/base/val.jsonl \
  --tasks 2 \
  --output artifacts/eval/results.json
```

### 8.3 相关配置

- Base Model: `Qwen/Qwen3.5-9B` (~19GB in bfloat16)
- GPU: NVIDIA A100-PCIE-40GB
- Python: 3.11
- Key packages: transformers==4.51.3, peft==0.19.1, torch==2.10.0+cu128

---

*报告生成时间: 2025-06-11*  
*版本: sft_v1*
