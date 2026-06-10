# HotpotQA MAS 阶段性报告

## 结论摘要

当前项目已经从数学题 GRPO 迁移到更接近 M-GRPO 目标的 HotpotQA multi-agent setting，并跑通了完整训练闭环：

```text
Main -> subtask
Sub -> search/read multi-step
Sub -> evidence summary
Main -> final answer
Reward -> GRPO / preference / staged training
```

阶段性结论是：

```text
MAS 训练链路有效，但当前 HotpotQA local-context 环境下，MAS 还没有超过 direct Main GRPO。
```

最强 direct Main GRPO：

```text
answer_f1 = 0.422
evidence  = 0.575
reward    = 0.510
```

当前最稳 MAS 候选：

```text
Stage 2 MAS:
answer_f1 = 0.340
evidence  = 0.480
reward    = 0.434

Joint MAS:
answer_f1 = 0.337
evidence  = 0.455
reward    = 0.427
```

因此，当前 MAS 是一个有效的机制复现 prototype，但还不是性能最优路线。

## 已完成工作

### 1. 放弃数学主环境

数学题环境的问题是 reward 太窄，Sub agent 没有真实分工，Main 很容易直接学答案格式。我们将数学保留为 smoke/baseline，把主线迁移到 HotpotQA。

### 2. Direct Main HotpotQA 基线

Direct Main 自己执行：

```text
search/read/answer
```

结果证明 HotpotQA 环境上的 RL 是有信号的：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Direct Main SFT | 0.384 | 0.565 | 0.482 | 0.445 | 0.541 |
| Direct Main GRPO 150x3 | 0.422 | 0.575 | 0.510 | 0.500 | 0.568 |

这说明问题不是 RL 完全无效，而是 MAS 架构需要更合适的任务压力。

### 3. MAS 架构跑通

实现了：

```text
generate_hotpotqa_mas_sft_data.py
grpo_hotpotqa_mas.py
analyze_hotpotqa_mas_results.py
analyze_hotpotqa_sub_oracle.py
train_hotpotqa_sub_preferences.py
run_hotpotqa_eval_suite.py
```

MAS agent 分工：

```text
Main:
  question -> [subtask]...[/subtask]

Sub:
  search/read
  <result>answer clue | evidence: DOCID,...</result>

Main:
  question + sub result -> final answer
```

### 4. Sub 能力拆解

Sub-only oracle eval 证明：

```text
Sub summary 可以通过 summary-reward GRPO 提升。
Sub retrieval 更适合 action preference learning。
```

关键结果：

| Sub checkpoint | support_read_recall | answer_f1 | evidence | reward |
|---|---:|---:|---:|---:|
| MAS SFT v2 Sub | 0.400 | 0.202 | 0.412 | 0.324 |
| Summary-GRPO Sub | 0.400 | 0.323 | 0.438 | 0.414 |
| Preference from SFT Sub | 0.525 | 0.272 | 0.537 | 0.398 |

结论：

```text
summary 和 retrieval 是两种不同能力。
summary-reward GRPO 提升总结。
preference pair 直接提升 read gold docs 的能力。
```

### 5. Stage 2 Main-only

冻结 retrieval-strong Sub，只训练 Main：

| Model | answer_f1 | evidence | reward | best_answer_f1 |
|---|---:|---:|---:|---:|
| MAS SFT Main + Preference Sub | 0.208 | 0.500 | 0.346 | 0.275 |
| Stage 2 Main 20x1 + Preference Sub | 0.299 | 0.475 | 0.404 | 0.401 |
| Stage 2 Main 50x2 + Preference Sub | 0.330 | 0.575 | 0.446 | 0.403 |

这说明 Main evidence integration 可以被 RL 提升。

### 6. Joint GRPO

从 Stage 2 best 出发做保守 joint：

```text
tasks=30
iterations=1
lr=3e-6
reward_threshold=0.35
```

单切片上曾经明显提升，但多 offset 后优势不稳定：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Stage 2 MAS task-weighted | 0.340 | 0.480 | 0.434 | 0.449 | 0.527 |
| Joint MAS task-weighted | 0.337 | 0.455 | 0.427 | 0.448 | 0.524 |

结论：

```text
Joint 是候选 best，但没有稳定超过 Stage 2。
```

### 7. Oracle Main eval

给 Main oracle Sub result：

```text
<result>gold answer | evidence: gold_doc_ids</result>
```

所有 Main 都达到：

```text
answer_f1 = 1.000
evidence  = 1.000
reward    = 0.900
```

这说明 Main 在“Sub 直接给出干净答案”时不是瓶颈。真实瓶颈是：

```text
Sub result/noisy evidence -> Main final answer
```

## 当前问题

### 1. 当前 HotpotQA local context 对 direct Main 太友好

每题只有约 10 篇文档。Direct Main 只需要 search/read 几步，就能完成大部分任务。MAS 的通信成本反而可能成为负担。

### 2. validation 太小

当前 val 只有 50 条，offset 40 只能评 10 条。单切片结果波动明显，必须扩大 validation。

### 3. MAS 分工压力不够

当前任务里，Main 自己做 search/read 并不难。要体现 multi-agent 优势，需要：

```text
更多文档
更多 distractors
更长 context
更多 hop
更明确的子任务分工
```

## 下一阶段目标

下一阶段不是继续盲目调 GRPO，而是增强环境，让 MAS 分工变得必要。

目标：

```text
1. 扩大 HotpotQA train/val。
2. 每题加入跨样本 distractor docs。
3. 让每题 context 从 10 docs 增加到 30/50 docs。
4. 保持 support_doc_ids 正确重映射。
5. 重新跑 direct Main vs MAS。
```

成功标准：

```text
Direct Main 在 enhanced 环境上下降；
Sub preference 对 retrieval 仍然有效；
MAS staged training 相比 direct baseline 的差距缩小，最好反超。
```

## 当前建议路线

```text
Stage A: 生成 enhanced HotpotQA
  train=500
  val=150
  docs_per_task=30 or 50

Stage B: 重新生成 MAS SFT
  使用 enhanced train

Stage C: 先评估现有 checkpoint zero-shot transfer
  direct Main
  Stage2 MAS
  Joint MAS

Stage D: 再决定是否重新 SFT/RL
```

如果 zero-shot 下 direct Main 大幅下降，而 MAS 的 Sub retrieval 相对保留，这条路线就值得继续。

## 已完成：Enhanced HotpotQA 环境

新增：

```text
prepare_hotpotqa_enhanced_data.py
```

增强方式：

```text
1. 保留原任务的 support docs。
2. 保留原任务自带 distractors。
3. 从其他 HotpotQA 样本中混入额外 distractor docs。
4. shuffle 所有 docs。
5. 重新分配 doc_id。
6. 重映射 support_doc_ids。
```

正式生成：

```bash
python prepare_hotpotqa_enhanced_data.py ^
  --output-dir .\hotpotqa_data_enhanced ^
  --train-size 500 ^
  --val-size 150 ^
  --docs-per-task 30 ^
  --pool-multiplier 4 ^
  --seed 2026
```

生成结果：

```text
train = 500
val   = 150
docs_per_task = 30.0
support_docs_per_task = 2.0
```

这比原环境更难：

```text
原环境:
  val = 50
  docs_per_task ≈ 10

enhanced:
  val = 150
  docs_per_task = 30
```

## Enhanced Zero-Shot Smoke

用旧 checkpoint 直接评估 enhanced val 前 20 条、每题 1 sample：

| Model | tasks | samples | answer_f1 | evidence | reward |
|---|---:|---:|---:|---:|---:|
| Direct Main GRPO 150x3 | 20 | 1 | 0.299 | 0.500 | 0.409 |
| MAS Stage 2 | 20 | 1 | 0.359 | 0.375 | 0.426 |

这只是 smoke，不是最终结论，但信号很好：

```text
1. Enhanced 环境确实让 direct Main 下降。
2. MAS Stage 2 在这个小切片上 reward 略高。
3. MAS evidence 低于 direct，但 answer_f1 更高，说明 Main/Sub 协同可能在更难环境中有机会。
```

下一步正式实验：

```text
1. Enhanced multi-offset eval:
   offsets = 0, 20, 40, 60, 80
   tasks = 20
   samples = 2

2. 比较：
   Direct Main SFT
   Direct Main GRPO
   MAS Stage 2
   MAS Joint

3. 如果 MAS 在 enhanced eval 上接近或超过 direct：
   重新生成 enhanced MAS SFT 数据
   在 enhanced train 上做 staged training

4. 如果 direct 仍明显更强：
   继续增加 docs_per_task 到 50，或引入 harder task construction。
```
