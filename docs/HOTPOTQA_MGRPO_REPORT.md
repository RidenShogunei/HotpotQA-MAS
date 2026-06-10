# HotpotQA M-GRPO 实验报告

## 1. 背景

本项目最初使用数学题环境验证 Main/Sub 双 agent 的 GRPO 训练链路，但数学题并不适合作为 M-GRPO 的主要环境：

- 任务缺少真实的多步检索和环境交互。
- Main/Sub 分工不清晰，Sub 容易退化成计算器或格式执行器。
- SFT 已经能学会大部分格式，RL 提供的新信号有限。

随后尝试了本地 synthetic tool-use 和 mini research 环境。它们验证了工具调用、环境执行、reward 和 best checkpoint 的基础链路，但 synthetic research 环境过于模板化，SFT 很快达到满分，仍然没有足够的 RL 空间。

因此当前主线切到真实多跳问答环境 HotpotQA，并以 Agent-R1 legacy 分支中的 HotpotQA/multihop QA 方案作为参考，实现了一个本地轻量 HotpotQA search/read 环境。

## 2. 当前环境

新增的 HotpotQA 环境位于：

- `prepare_hotpotqa_data.py`
- `hotpotqa_environment.py`
- `generate_hotpotqa_sft_data.py`
- `analyze_hotpotqa_results.py`
- `grpo_hotpotqa.py`

环境结构：

```text
HotpotQA question
  -> Main 生成 search/read tool_call
  -> Env 在该题的 distractor context 中执行 search/read
  -> Main 基于多步 research history 生成 final answer
  -> reward = answer_f1 + evidence + tool_valid
```

工具：

```text
search(query) -> 返回候选 doc_id/title，不直接泄露完整答案
read(doc_id) -> 返回该 HotpotQA context 文档正文
```

Reward：

```text
total = 0.7 * answer_f1 + 0.2 * evidence + 0.1 * tool_valid
```

其中：

- `answer_f1`：最终答案与 gold answer 的 token F1。
- `evidence`：输出 evidence doc_id 命中 supporting docs 的比例。
- `tool_valid`：是否生成合法 tool_call。

## 3. 关键修复

实验过程中修复了几个影响结果可信度的问题：

1. `SharedModel.load_sft_weights()` 没有正确加载 PEFT 保存的 LoRA key。
   - 现已兼容裸 PEFT、default adapter、命名 adapter 三种 key 形式。
   - 修复前 GRPO 实际可能没有从 SFT checkpoint 起步。

2. `save_lora()` 会保存成嵌套 adapter 目录。
   - 已修成 `best/main` 和 `best/sub` 可直接加载。

3. HotpotQA context 较长导致 SFT 训练中 assistant label 被截断，引发 `loss=nan`。
   - `sft_trainer.py` 新增 `--max-length`。
   - 对没有可训练 assistant token 的样本会跳过。

4. GRPO checkpoint 选择从 `reward` 优先改成 `answer_f1` 优先。
   - 更符合最终目标。
   - 避免 evidence 分数把总 reward 撑高但答案没提升。

## 4. 实验设置

数据准备：

```bash
python prepare_hotpotqa_data.py --train-size 200 --val-size 50 --output-dir .\hotpotqa_data
```

SFT 数据：

```bash
python generate_hotpotqa_sft_data.py ^
  --train-jsonl .\hotpotqa_data\train.jsonl ^
  --output .\hotpotqa_sft_data.jsonl ^
  --answer-fraction 0.25
```

SFT 训练：

```bash
python sft_trainer.py ^
  --data-path .\hotpotqa_sft_data.jsonl ^
  --save-dir .\hotpotqa_sft_checkpoints ^
  --epochs 1 ^
  --lr 2e-4 ^
  --max-length 2048
```

SFT 训练结果：

```text
Main loss = 0.1206
Sub  loss = 1.1157
```

GRPO 训练：

```bash
python grpo_hotpotqa.py ^
  --train-jsonl .\hotpotqa_data\train.jsonl ^
  --val-jsonl .\hotpotqa_data\val.jsonl ^
  --tasks 100 ^
  --val-tasks 30 ^
  --iterations 3 ^
  --group-size 2 ^
  --eval-samples 2 ^
  --sft-dir .\hotpotqa_sft_checkpoints ^
  --save-dir .\hotpotqa_grpo_100x3_answerbest ^
  --max-response-len 120 ^
  --research-steps 3 ^
  --reward-threshold 0.3 ^
  --best-metric answer_f1
```

## 5. 实验结果

GRPO 训练内部 validation：

```text
init:
  reward      = 0.492
  answer_f1   = 0.400
  best_answer = 0.497

iter1:
  reward      = 0.540
  answer_f1   = 0.454
  best_answer = 0.577

iter2:
  reward      = 0.501
  answer_f1   = 0.400
  best_answer = 0.491

iter3:
  reward      = 0.562
  answer_f1   = 0.481
  best_answer = 0.537
```

独立评估设置：

```text
val tasks = 30
samples   = 2
research_steps = 3
```

独立评估结果：

| Model | tool_valid | answer_f1 | evidence | reward | best_reward | best_answer_f1 |
|---|---:|---:|---:|---:|---:|---:|
| SFT baseline | 1.000 | 0.415 | 0.600 | 0.511 | 0.566 | 0.476 |
| GRPO 100x3 answer-best | 1.000 | 0.487 | 0.650 | 0.571 | 0.627 | 0.557 |

提升：

```text
answer_f1      +0.072
evidence       +0.050
reward         +0.060
best_reward    +0.061
best_answer_f1 +0.081
```

## 6. 结论

当前实验已经证明：

```text
在真实 HotpotQA multi-hop 环境上，
answer_f1 优先的 GRPO 能稳定超过 SFT baseline。
```

这比之前的数学题、SQL tool-use、synthetic research 环境更接近 M-GRPO 的目标：

- 有真实多步检索。
- 有环境 observation。
- 有可验证 final answer。
- Main/Sub 分工更清晰。
- SFT 不再直接满分，RL 有实际提升空间。

目前最有价值的 checkpoint：

```text
hotpotqa_grpo_100x3_answerbest/best/main
hotpotqa_grpo_100x3_answerbest/best/sub
```

## 7. 当前问题

仍然存在几个需要注意的问题：

1. 训练仍然更像 best-of-N + filtered SFT，而不是完整严格的 policy-gradient GRPO。
2. Sub agent 目前主要学习工具执行 observation，Main 的提升更明显。
3. validation 仍有采样方差，因此需要多 seed 或更大 val set 验证。
4. evidence reward 只按 supporting doc_id 命中，没有做到 sentence-level citation。
5. 当前 HotpotQA search 使用题目局部 context，不是全 Wikipedia/KILT 检索。

## 8. 下一步建议

优先级最高：

```bash
python grpo_hotpotqa.py ^
  --tasks 150 ^
  --val-tasks 50 ^
  --iterations 3 ^
  --group-size 2 ^
  --eval-samples 2 ^
  --sft-dir .\hotpotqa_sft_checkpoints ^
  --save-dir .\hotpotqa_grpo_150x3_answerbest ^
  --max-response-len 120 ^
  --research-steps 3 ^
  --reward-threshold 0.3 ^
  --best-metric answer_f1
```

然后做同口径评估：

```bash
python analyze_hotpotqa_results.py ^
  --val-jsonl .\hotpotqa_data\val.jsonl ^
  --tasks 50 ^
  --samples 2 ^
  --lora .\hotpotqa_grpo_150x3_answerbest\best\main ^
  --max-tokens 120 ^
  --research-steps 3
```

如果 150x3 仍稳定超过 SFT，则下一步可以：

- 加 multi-seed。
- 加 sentence-level supporting fact reward。
- 接 Agent-R1 legacy 的 KILT/Wikipedia search server。
- 将当前 local-context HotpotQA 迁移为更接近真实 deep research 的 open-corpus setting。

## 9. 150x3 稳定性验证更新

在 100x3 之后，进一步扩大到 150 train / 50 val / 3 iter，并继续使用 `answer_f1` 作为 best checkpoint 指标：

```bash
python grpo_hotpotqa.py ^
  --train-jsonl .\hotpotqa_data\train.jsonl ^
  --val-jsonl .\hotpotqa_data\val.jsonl ^
  --tasks 150 ^
  --val-tasks 50 ^
  --iterations 3 ^
  --group-size 2 ^
  --eval-samples 2 ^
  --sft-dir .\hotpotqa_sft_checkpoints ^
  --save-dir .\hotpotqa_grpo_150x3_answerbest ^
  --max-response-len 120 ^
  --research-steps 3 ^
  --reward-threshold 0.3 ^
  --best-metric answer_f1
```

训练内部 validation：

```text
init:
  reward      = 0.462
  answer_f1   = 0.362
  best_answer = 0.462

iter1:
  reward      = 0.470
  answer_f1   = 0.372
  best_answer = 0.450

iter2:
  reward      = 0.548
  answer_f1   = 0.470
  best_answer = 0.523

iter3:
  reward      = 0.530
  answer_f1   = 0.446
  best_answer = 0.495
```

独立评估设置：

```text
val tasks = 50
samples   = 2
research_steps = 3
```

独立评估结果：

| Model | tool_valid | answer_f1 | evidence | reward | best_reward | best_answer_f1 |
|---|---:|---:|---:|---:|---:|---:|
| SFT baseline | 1.000 | 0.381 | 0.535 | 0.473 | 0.554 | 0.477 |
| GRPO 150x3 answer-best | 1.000 | 0.469 | 0.600 | 0.548 | 0.572 | 0.498 |

提升：

```text
answer_f1      +0.088
evidence       +0.065
reward         +0.075
best_reward    +0.018
best_answer_f1 +0.021
```

这说明 100x3 的提升不是单次偶然；在更大的 50 条 validation 上，GRPO 仍然稳定超过 SFT baseline。

当前最有价值 checkpoint 更新为：

```text
hotpotqa_grpo_150x3_answerbest/best/main
hotpotqa_grpo_150x3_answerbest/best/sub
```

新的下一步建议：

- 继续扩大到 200 train / 50 val / 3 iter，确认增益是否继续保持。
- 加 multi-seed，验证结果不是某个采样 seed 的偶然。
- 加 sentence-level supporting fact reward，让 evidence 不只是 doc_id 命中。
- 接 Agent-R1 legacy 的 KILT/Wikipedia search server，逐步迁移到 open-corpus deep research。

## 10. Main-only 消融实验

为了判断当前提升主要来自 Main agent，还是来自 Main/Sub 联合训练，新增了 `grpo_hotpotqa.py` 的训练开关：

```bash
--train-main / --no-train-main
--train-sub / --no-train-sub
```

消融设置：

```bash
python grpo_hotpotqa.py ^
  --train-jsonl .\hotpotqa_data\train.jsonl ^
  --val-jsonl .\hotpotqa_data\val.jsonl ^
  --tasks 100 ^
  --val-tasks 30 ^
  --iterations 3 ^
  --group-size 2 ^
  --eval-samples 2 ^
  --sft-dir .\hotpotqa_sft_checkpoints ^
  --save-dir .\hotpotqa_grpo_100x3_mainonly ^
  --max-response-len 120 ^
  --research-steps 3 ^
  --reward-threshold 0.3 ^
  --best-metric answer_f1 ^
  --no-train-sub
```

训练内部 validation：

```text
init:
  reward      = 0.480
  answer_f1   = 0.378
  best_answer = 0.457

iter1:
  reward      = 0.489
  answer_f1   = 0.394
  best_answer = 0.456

iter2:
  reward      = 0.440
  answer_f1   = 0.333
  best_answer = 0.424

iter3:
  reward      = 0.554
  answer_f1   = 0.480
  best_answer = 0.591
```

独立评估设置：

```text
val tasks = 20
samples   = 2
research_steps = 3
```

同口径结果：

| Model | tool_valid | answer_f1 | evidence | reward | best_reward | best_answer_f1 |
|---|---:|---:|---:|---:|---:|---:|
| SFT baseline | 1.000 | 0.420 | 0.525 | 0.499 | 0.594 | 0.549 |
| GRPO main-only 100x3 | 1.000 | 0.530 | 0.675 | 0.606 | 0.637 | 0.567 |
| GRPO main+sub 100x3 | 1.000 | 0.562 | 0.713 | 0.636 | 0.664 | 0.599 |

结论：

```text
main-only 已经显著超过 SFT，说明当前收益主要来自 Main agent 学会更好的 search/read/answer 策略。
full main+sub 又进一步超过 main-only，说明 Sub 联训在当前设置下仍有小幅正贡献。
```

不过需要注意，当前 Sub agent 主要学习工具 observation，而实际 search/read 仍由环境执行。因此这个正贡献可能来自更稳定的 Sub adapter/共享模型更新，而不是论文意义上强 Sub executor 能力。后续应继续做：

- `--no-train-main --train-sub` 的 sub-only 消融。
- 固定 Sub 为纯环境执行器的 baseline。
- 多 seed 消融，避免单次采样影响判断。

## 11. MAS Sub-researcher 升级尝试

为了让 Sub agent 更接近 M-GRPO 论文中的 multi-turn executor / researcher，新增了一条 MAS 版本链路：

- `generate_hotpotqa_mas_sft_data.py`
- `grpo_hotpotqa_mas.py`
- `analyze_hotpotqa_mas_results.py`

新结构：

```text
Main:
  question -> [subtask]...[/subtask]

Sub:
  subtask -> search/read 多步工具调用
  research history -> <result>answer clue | evidence: DOCID, DOCID</result>

Main:
  question + sub result -> final <result>answer | evidence: ...</result>
```

这和旧 HotpotQA GRPO 的区别是：

```text
旧版本：
  Main 直接做 search/read/answer
  Sub 主要拟合工具 observation

MAS 版本：
  Main 只委派和汇总
  Sub 自己多步 search/read 并总结 evidence
```

### MAS SFT

第一次 MAS SFT 使用 `answer_fraction=0.25`，起点较弱：

```text
tasks=5
answer_f1=0.000
evidence=0.300
reward=0.160
```

随后改为 `answer_fraction=1.0`，生成 v2 数据：

```bash
python generate_hotpotqa_mas_sft_data.py ^
  --train-jsonl .\hotpotqa_data\train.jsonl ^
  --output .\hotpotqa_mas_sft_data_v2.jsonl ^
  --answer-fraction 1.0
```

数据规模：

```text
total = 1200
main  = 400
sub   = 800
```

SFT 训练：

```text
Main loss = 0.0859
Sub  loss = 0.1182
```

MAS SFT v2 独立评估：

```text
tasks=10
tool_valid=1.000
answer_f1=0.380
evidence=0.400
reward=0.446
```

### MAS GRPO Smoke

配置：

```bash
python grpo_hotpotqa_mas.py ^
  --tasks 20 ^
  --val-tasks 10 ^
  --iterations 2 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --sft-dir .\hotpotqa_mas_sft_checkpoints_v2 ^
  --save-dir .\hotpotqa_mas_grpo_20x2_v2 ^
  --max-response-len 120 ^
  --sub-steps 3 ^
  --reward-threshold 0.3 ^
  --best-metric answer_f1
```

训练内部 validation：

```text
init:
  reward    = 0.388
  answer_f1 = 0.240
  evidence  = 0.600

iter1:
  reward    = 0.498
  answer_f1 = 0.440
  evidence  = 0.450

iter2:
  reward    = 0.428
  answer_f1 = 0.340
  evidence  = 0.450
```

独立评估同一 10 条 val：

| Model | tool_valid | answer_f1 | evidence | reward |
|---|---:|---:|---:|---:|
| MAS SFT v2 | 1.000 | 0.380 | 0.400 | 0.446 |
| MAS GRPO 20x2 best | 1.000 | 0.280 | 0.500 | 0.396 |

结论：

```text
MAS Sub-researcher 链路已经跑通，但当前 GRPO smoke 尚未稳定超过 MAS SFT。
训练内部出现过 answer_f1 提升，但独立评估没有复现。
```

这说明 Sub 升级方向是正确的工程方向，但还需要进一步调 SFT、reward 和 checkpoint selection，不能直接认为 MAS 版本已经优于旧版 main-direct GRPO。

建议下一步：

- 对 MAS 版本使用更大的 val set 和 `eval_samples=2` 降低方差。
- 增加 Sub summary 的专门 reward：answer clue F1、support doc hit、是否被 Main 使用。
- 降低或分离 Main/Sub 更新阈值，避免 Sub summary 还不稳时反向影响 Main。
- 先跑 MAS main-only / sub-only 消融，确认不稳定来自哪一侧。

## 12. MAS Sub-only GRPO 消融

为了确认 Sub researcher 是否能在冻结 Main 的情况下独立变强，新增了 MAS GRPO 的 adapter 开关：

```bash
--train-main / --no-train-main
--train-sub  / --no-train-sub
```

本轮实验冻结 Main，只训练 Sub：

```bash
python grpo_hotpotqa_mas.py ^
  --tasks 50 ^
  --val-tasks 20 ^
  --iterations 2 ^
  --group-size 2 ^
  --eval-samples 2 ^
  --sft-dir .\hotpotqa_mas_sft_checkpoints_v2 ^
  --save-dir .\hotpotqa_mas_subonly_50x2 ^
  --max-response-len 120 ^
  --sub-steps 3 ^
  --reward-threshold 0.3 ^
  --best-metric sub_reward ^
  --no-train-main ^
  --train-sub
```

训练内部 validation：

```text
init:
  reward        = 0.351
  answer_f1     = 0.269
  evidence      = 0.312
  sub_reward    = 0.351
  sub_evidence  = 0.312

iter1:
  train reward  = 0.545
  train updates = main 0, sub 27
  val reward    = 0.397
  answer_f1     = 0.310
  evidence      = 0.400
  sub_reward    = 0.397
  saved best

iter2:
  train reward  = 0.558
  train updates = main 0, sub 29
  val reward    = 0.389
  answer_f1     = 0.284
  evidence      = 0.450
  sub_reward    = 0.389
```

独立评估使用同一组 20 条 validation、每题 2 samples，Main 均固定为 MAS SFT v2：

| Model | Main | Sub | tool_valid | answer_f1 | evidence | reward | best_reward | best_answer_f1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MAS SFT v2 | SFT v2 | SFT v2 | 1.000 | 0.202 | 0.375 | 0.317 | 0.400 | 0.286 |
| MAS Sub-only GRPO 50x2 | SFT v2 | GRPO best | 1.000 | 0.260 | 0.412 | 0.364 | 0.440 | 0.350 |

结论：

```text
Sub-only GRPO 在同一 Main、同一评估集上确实带来小幅提升：
answer_f1 +0.058
evidence  +0.037
reward    +0.047
best_answer_f1 +0.064
```

这说明 Sub researcher 不是完全无效；它可以通过 RL 学到更好的 search/read/summary 行为。但提升幅度还小，而且绝对效果仍低于旧版 direct Main GRPO。当前 MAS 的主要问题不是工具合法率，而是任务分解、Sub summary、Main 最终整合之间的信用分配还不够干净。

下一步不应该直接扩大联合训练，而应该先把 Sub 的训练信号隔离得更干净：

- 做 Sub-only oracle evaluator：给 Sub oracle subtask，只评估 Sub summary 的 answer clue F1 和 supporting doc hit。
- 用该 evaluator 选择 `best/sub`，避免 Main final answer 的噪声污染 Sub checkpoint selection。
- 然后再做 Stage 2：冻结 Sub best，只训 Main planner/answerer。
- 最后才做低学习率 joint GRPO。

## 13. Sub-only Oracle Evaluator

为了把 Sub researcher 的能力从 Main planning / Main final answer 中隔离出来，新增：

```text
analyze_hotpotqa_sub_oracle.py
```

评估方式：

```text
oracle subtask -> Sub search/read multi-step -> Sub summary
```

这里不再调用 Main。指标只看 Sub 自己：

- `tool_valid`: 每个 sample 是否至少有一个合法工具调用。
- `action_valid`: 每一步 search/read 是否可解析、可执行。
- `support_read_recall`: Sub 实际 read 到 gold supporting docs 的比例。
- `answer_f1`: Sub summary 里的 answer clue F1。
- `evidence`: Sub 输出/工具轨迹里命中的 supporting docs。
- `reward`: HotpotQAEnvironment.reward(task, Sub trajectory + Sub summary)。

同一组 20 条 validation、每题 2 samples、`sub_steps=3`：

| Sub checkpoint | tool_valid | action_valid | support_read_recall | answer_f1 | evidence | reward | best_support_read_recall | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAS SFT v2 Sub | 1.000 | 1.000 | 0.400 | 0.202 | 0.412 | 0.324 | 0.500 | 0.306 | 0.419 |
| MAS Sub-only GRPO 50x2 best | 1.000 | 1.000 | 0.400 | 0.323 | 0.438 | 0.414 | 0.550 | 0.414 | 0.505 |

结论：

```text
在 oracle subtask 条件下，Sub-only GRPO 明显提升了 Sub summary/answer clue：
answer_f1 +0.121
reward    +0.090
best_answer_f1 +0.108
best_reward    +0.086
```

但平均 `support_read_recall` 没有提升，仍为 0.400。这说明当前 Sub RL 的主要收益来自“更会总结/更会把答案写进 result”，而不是稳定读到更多 gold supporting docs。`best_support_read_recall` 从 0.500 到 0.550，说明 sampling 下偶尔能找到更好的 doc，但还没有变成稳定策略。

因此现在的瓶颈更精确了：

```text
已解决/较好：
  tool_call 格式和执行合法率
  Sub summary 的 answer clue 有 RL 提升

仍然不足：
  Sub 的文档选择/read 策略没有稳定提升
  Main planner/final answer 仍会给整条 MAS 链路增加额外噪声
```

下一阶段建议：

1. Stage 1b：优化 Sub retrieval reward，让 read 到 supporting docs 的行为更直接地拿到奖励。
2. Stage 2：冻结当前 Sub GRPO best，只训练 Main planner/answerer。
3. Stage 3：在 Sub retrieval 稳定后，再做低学习率 joint GRPO。

## 14. Stage 1b: Sub Retrieval Reward

为了解决上一节发现的问题，`grpo_hotpotqa_mas.py` 新增了 retrieval-oriented Sub reward：

```text
--sub-reward-mode summary    # 原逻辑：Sub summary reward
--sub-reward-mode retrieval  # 0.8 * support_read_recall + 0.2 * action_valid
--sub-reward-mode mixed      # 0.5 * summary + 0.4 * retrieval + 0.1 * action_valid

--best-metric sub_retrieval
--best-metric sub_train_reward
```

新增训练/验证指标：

- `sub_retrieval_reward`: Sub 实际 read 到 gold supporting docs 的比例。
- `action_valid`: 每一步工具调用是否合法、可执行。
- `sub_train_reward`: 当前 `sub_reward_mode` 下用于训练 Sub 的 reward。

先尝试了较大的 retrieval-only 实验：

```bash
python grpo_hotpotqa_mas.py ^
  --tasks 50 ^
  --val-tasks 20 ^
  --iterations 2 ^
  --group-size 2 ^
  --eval-samples 2 ^
  --sft-dir .\hotpotqa_mas_sft_checkpoints_v2 ^
  --save-dir .\hotpotqa_mas_subretrieval_50x2 ^
  --sub-reward-mode retrieval ^
  --best-metric sub_retrieval ^
  --no-train-main ^
  --train-sub
```

该配置 rollout 太慢，超过 1 小时仍未完成第一个 iteration，因此停止，保留已写出的初始 `best/`。随后改跑较小实验：

```bash
python grpo_hotpotqa_mas.py ^
  --tasks 20 ^
  --val-tasks 10 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --sft-dir .\hotpotqa_mas_sft_checkpoints_v2 ^
  --save-dir .\hotpotqa_mas_subretrieval_20x1 ^
  --sub-reward-mode retrieval ^
  --best-metric sub_retrieval ^
  --reward-threshold 0.2 ^
  --no-train-main ^
  --train-sub
```

训练内部结果：

```text
init val:
  sub_retrieval = 0.400
  sub_train     = 0.520
  sub_reward    = 0.266
  answer_f1     = 0.122

iter1 train:
  sub_retrieval = 0.350
  sub_train     = 0.480
  updates sub   = 20

iter1 val:
  sub_retrieval = 0.350
  sub_train     = 0.480
  sub_reward    = 0.250
  answer_f1     = 0.100
```

内部 validation 没有超过 init，所以 `best/` 没更新。为了观察训练后权重本身，独立评估了 `sub_step_1`：

| Sub checkpoint | support_read_recall | answer_f1 | evidence | reward | best_support_read_recall | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAS SFT v2 Sub | 0.400 | 0.202 | 0.412 | 0.324 | 0.500 | 0.306 | 0.419 |
| MAS Sub-only GRPO 50x2 best | 0.400 | 0.323 | 0.438 | 0.414 | 0.550 | 0.414 | 0.505 |
| MAS Sub-retrieval 20x1 step1 | 0.412 | 0.278 | 0.438 | 0.382 | 0.525 | 0.387 | 0.481 |

结论：

```text
retrieval-only reward 对 support_read_recall 有轻微正向信号：
0.400 -> 0.412

但它牺牲了 summary answer_f1：
0.323 -> 0.278，相比上一轮 Sub-only GRPO best 更低
```

这说明单纯奖励 read 到 gold docs 不够，甚至可能让 Sub 偏向“读对文档”但不稳定总结答案。当前更合理的方向是使用 `mixed` reward，而不是 pure retrieval：

```text
sub_train_reward = 0.5 * summary_reward + 0.4 * retrieval_reward + 0.1 * action_valid
```

下一步建议跑：

```bash
python grpo_hotpotqa_mas.py ^
  --tasks 20 ^
  --val-tasks 10 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --sft-dir .\hotpotqa_mas_sft_checkpoints_v2 ^
  --save-dir .\hotpotqa_mas_submixed_20x1 ^
  --sub-reward-mode mixed ^
  --best-metric sub_train_reward ^
  --reward-threshold 0.25 ^
  --no-train-main ^
  --train-sub
```

如果 mixed reward 同时保住 answer_f1 并提升 support_read_recall，再扩大到 50x2。

## 15. Stage 1c: Mixed Sub Reward

按上一节建议，继续跑 mixed reward：

```bash
python grpo_hotpotqa_mas.py ^
  --tasks 20 ^
  --val-tasks 10 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --sft-dir .\hotpotqa_mas_sft_checkpoints_v2 ^
  --save-dir .\hotpotqa_mas_submixed_20x1 ^
  --sub-reward-mode mixed ^
  --best-metric sub_train_reward ^
  --reward-threshold 0.25 ^
  --no-train-main ^
  --train-sub
```

训练内部结果：

```text
init val:
  sub_reward    = 0.318
  sub_train     = 0.419
  sub_retrieval = 0.400
  answer_f1     = 0.183
  evidence      = 0.450

iter1 train:
  sub_reward    = 0.458
  sub_train     = 0.499
  sub_retrieval = 0.425
  answer_f1     = 0.383
  updates sub   = 16

iter1 val:
  sub_reward    = 0.318
  sub_train     = 0.359
  sub_retrieval = 0.250
  answer_f1     = 0.240
  evidence      = 0.250
```

内部 validation 没超过 init，因此 `best/` 仍是初始 SFT 权重。随后对 `sub_step_1` 做 Sub oracle 独立评估：

| Sub checkpoint | support_read_recall | answer_f1 | evidence | reward | best_support_read_recall | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAS SFT v2 Sub | 0.400 | 0.202 | 0.412 | 0.324 | 0.500 | 0.306 | 0.419 |
| MAS Sub-only GRPO 50x2 best | 0.400 | 0.323 | 0.438 | 0.414 | 0.550 | 0.414 | 0.505 |
| MAS Sub-retrieval 20x1 step1 | 0.412 | 0.278 | 0.438 | 0.382 | 0.525 | 0.387 | 0.481 |
| MAS Sub-mixed 20x1 step1 | 0.350 | 0.230 | 0.362 | 0.333 | 0.425 | 0.262 | 0.368 |

结论：

```text
mixed 20x1 没有成功。它既没有保住 retrieval，也没有超过 summary-only GRPO。
```

当前观察：

- `summary` reward 的 50x2 checkpoint 是目前最好的 Sub checkpoint。
- `retrieval` reward 有轻微 retrieval 信号，但会牺牲 summary。
- `mixed` 在 20x1 下没有改善，可能是训练样本少、validation 方差高，也可能是 candidate selection 仍不够对齐。

更重要的问题是：现在的 GRPO 近似实现是“从 group 中挑 best 然后加权 SFT”，而不是完整 pairwise/group advantage 更新。对于 retrieval 这种稀疏信号，group size=2 太小，随机样本里经常没有更好的 retrieval 行为可学。

下一步不建议继续盲目调 reward 权重，而应该改 Sub action 学习方式：

```text
1. 对每个 task 显式构造 positive read actions：
   read(gold_doc_id)

2. rollout 产生 negative/read-wrong actions：
   read(non_gold_doc_id)

3. 用 preference / contrastive SFT 更新 Sub action：
   同一 history 下提高 gold read，降低 wrong read

4. summary 仍用当前 summary reward 训练
```

也就是说，Sub retrieval 更像“action preference learning”，不适合只靠最终 scalar reward 从很小 group 里碰运气。

## 16. Sub Action Preference Learning

为了直接优化 retrieval action，新增：

```text
train_hotpotqa_sub_preferences.py
```

训练思想：

```text
同一个 Sub action prompt 下：
  chosen   = search(question) 或 read(gold_doc_id)
  rejected = read(non_gold_doc_id)

loss:
  -log sigmoid(beta * (logp(chosen) - logp(rejected)))
  + small chosen SFT loss
```

这不是完整 DPO，因为没有单独 reference model；它是一个轻量 action preference / contrastive SFT 近似，目标很窄：提高 gold read 相对 wrong read 的概率。

### Preference From MAS SFT

训练：

```bash
python train_hotpotqa_sub_preferences.py ^
  --tasks 100 ^
  --max-pairs 250 ^
  --epochs 1 ^
  --sub-lora .\hotpotqa_mas_sft_checkpoints_v2\sub_agent\sub ^
  --save-dir .\hotpotqa_sub_pref_100x250\sub ^
  --lr 1e-5 ^
  --beta 2.0 ^
  --sft-weight 0.05
```

训练结果：

```text
pairs = 250
loss  = 0.0087
margin = 3.8807
```

Sub oracle eval：

| Sub checkpoint | support_read_recall | answer_f1 | evidence | reward | best_support_read_recall | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAS SFT v2 Sub | 0.400 | 0.202 | 0.412 | 0.324 | 0.500 | 0.306 | 0.419 |
| Preference from SFT | 0.525 | 0.272 | 0.537 | 0.398 | 0.625 | 0.348 | 0.468 |

结论：

```text
action preference learning 明显提升 retrieval：
support_read_recall 0.400 -> 0.525
evidence            0.412 -> 0.537
```

这验证了上一节判断：Sub retrieval 的主要问题不是 reward 权重，而是小 group scalar GRPO 很难采到并强化正确 read action。显式 preference pair 更直接有效。

### Preference From Summary-GRPO Best

为了尽量保住 summary 能力，又从当前最好的 Sub summary checkpoint 出发继续 preference：

```bash
python train_hotpotqa_sub_preferences.py ^
  --tasks 100 ^
  --max-pairs 250 ^
  --epochs 1 ^
  --sub-lora .\hotpotqa_mas_subonly_50x2\best\sub ^
  --save-dir .\hotpotqa_sub_pref_from_summary_100x250\sub ^
  --lr 5e-6 ^
  --beta 2.0 ^
  --sft-weight 0.05
```

训练结果：

```text
pairs = 250
loss  = 0.0185
margin = 3.0973
```

Sub oracle eval：

| Sub checkpoint | support_read_recall | answer_f1 | evidence | reward | best_support_read_recall | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAS Sub-only GRPO 50x2 best | 0.400 | 0.323 | 0.438 | 0.414 | 0.550 | 0.414 | 0.505 |
| Preference from Summary-GRPO | 0.450 | 0.298 | 0.450 | 0.399 | 0.575 | 0.377 | 0.479 |

结论：

```text
从 summary-best 出发做 preference：
  retrieval 有提升：0.400 -> 0.450
  summary 有小幅回落：0.323 -> 0.298
```

这比 pure retrieval/mixed scalar reward 更可控。当前最有价值的发现是：

```text
Sub retrieval 可以通过 action preference 明确提升。
Sub summary 可以通过 summary-reward GRPO 提升。
两种能力存在轻微 trade-off，需要 staged training 或混合数据保持。
```

下一步建议：

1. 用 preference from SFT 作为 retrieval-strong Sub，跑完整 MAS eval，看 Main 是否能利用更强 retrieval。
2. 用 preference from summary-best 作为 balanced Sub，跑完整 MAS eval。
3. 如果 retrieval-strong 的 MAS answer 反而不好，说明 Main/summary 整合跟不上；下一步冻结 Sub，训练 Main。
4. 如果 balanced Sub 更好，则进入 `summary GRPO -> preference -> summary replay` 的循环训练。

## 17. Full MAS Eval With Preference Sub

将新的 preference Sub checkpoint 接回完整 MAS 链路，Main 固定为 MAS SFT v2：

```text
Main:
  .\hotpotqa_mas_sft_checkpoints_v2\main_agent\main

Sub candidates:
  .\hotpotqa_mas_subonly_50x2\best\sub
  .\hotpotqa_sub_pref_100x250\sub
  .\hotpotqa_sub_pref_from_summary_100x250\sub
```

统一评估设置：

```bash
python analyze_hotpotqa_mas_results.py ^
  --val-jsonl .\hotpotqa_data\val.jsonl ^
  --tasks 20 ^
  --samples 2 ^
  --main-lora .\hotpotqa_mas_sft_checkpoints_v2\main_agent\main ^
  --sub-lora <SUB> ^
  --max-tokens 120 ^
  --sub-steps 3
```

结果：

| Sub checkpoint | tool_valid | answer_f1 | evidence | reward | best_reward | best_answer_f1 |
|---|---:|---:|---:|---:|---:|---:|
| MAS Sub-only GRPO 50x2 best | 1.000 | 0.201 | 0.350 | 0.311 | 0.343 | 0.233 |
| Preference from SFT | 1.000 | 0.208 | 0.500 | 0.346 | 0.408 | 0.275 |
| Preference from Summary-GRPO | 1.000 | 0.216 | 0.375 | 0.326 | 0.348 | 0.233 |

结论：

```text
Preference from SFT Sub 在完整 MAS 中显著提升 evidence：
0.350 -> 0.500

但 answer_f1 只小幅变化：
0.201 -> 0.208
```

这说明 Main 确实能从 preference Sub 获得更多 supporting evidence，但当前 Main answerer 还不会稳定把这些 evidence 转成正确答案。换句话说：

```text
Sub retrieval 瓶颈已经被 preference learning 部分打开；
新的瓶颈转移到了 Main 对 Sub result/evidence 的整合。
```

因此下一阶段不应该继续只训 Sub，而应该进入 Stage 2：

```text
冻结 retrieval-strong Sub：
  .\hotpotqa_sub_pref_100x250\sub

训练 Main：
  plan prompt
  final answer from Sub summary/evidence

best metric:
  full MAS answer_f1 或 reward
```

预期如果 Main 学会利用更强 evidence，完整 MAS 的 answer_f1 才会真正上来。

## 18. Stage 2: Freeze Retrieval-Strong Sub, Train Main

为了验证“新瓶颈转移到 Main evidence integration”，扩展了 `grpo_hotpotqa_mas.py` / `grpo_v4.py`：

```text
--main-lora <path>
--sub-lora <path>
```

这样可以从不同 checkpoint 分别加载 Main/Sub，而不需要复制目录。

Stage 2 配置：

```text
Main init:
  .\hotpotqa_mas_sft_checkpoints_v2\main_agent\main

Frozen Sub:
  .\hotpotqa_sub_pref_100x250\sub

Train:
  Main only
```

训练命令：

```bash
python grpo_hotpotqa_mas.py ^
  --tasks 20 ^
  --val-tasks 10 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --sft-dir .\hotpotqa_mas_sft_checkpoints_v2 ^
  --main-lora .\hotpotqa_mas_sft_checkpoints_v2\main_agent\main ^
  --sub-lora .\hotpotqa_sub_pref_100x250\sub ^
  --save-dir .\hotpotqa_mas_stage2_main_prefsub_20x1 ^
  --best-metric reward ^
  --reward-threshold 0.25 ^
  --train-main ^
  --no-train-sub
```

训练内部 validation：

```text
init val:
  reward        = 0.358
  answer_f1     = 0.240
  evidence      = 0.450
  sub_retrieval = 0.450

iter1 train:
  reward        = 0.588
  answer_f1     = 0.533
  evidence      = 0.575
  sub_retrieval = 0.550
  updates main  = 12
  updates sub   = 0

iter1 val:
  reward        = 0.389
  answer_f1     = 0.256
  evidence      = 0.550
  sub_retrieval = 0.500
  saved best
```

独立完整 MAS eval，20 条 validation、每题 2 samples：

| Main | Sub | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---|---:|---:|---:|---:|---:|
| MAS SFT v2 | Preference from SFT | 0.208 | 0.500 | 0.346 | 0.275 | 0.408 |
| Stage 2 Main 20x1 | Preference from SFT | 0.299 | 0.475 | 0.404 | 0.401 | 0.491 |

结论：

```text
冻结 retrieval-strong Sub 后训练 Main 是有效的：
answer_f1      0.208 -> 0.299
reward         0.346 -> 0.404
best_answer_f1 0.275 -> 0.401
```

这验证了上一节判断：Preference Sub 已经提供了更强 evidence，Main 经过 RL 后能更好地把 evidence 转成最终答案。虽然 evidence 从 0.500 小幅降到 0.475，但整体 answer/reward 明显提升，说明 MAS 的当前主瓶颈确实是 Main evidence integration。

当前最强 MAS checkpoint：

```text
Main:
  .\hotpotqa_mas_stage2_main_prefsub_20x1\best\main

Sub:
  .\hotpotqa_mas_stage2_main_prefsub_20x1\best\sub
```

下一步建议扩大 Stage 2，而不是马上 joint training：

```text
tasks=50
val-tasks=20
iterations=2
group-size=2
best-metric=reward 或 answer_f1
train-main only
```

如果扩大后仍然提升，再进入轻量 joint GRPO。

## 19. Expanded Stage 2: Main-only 50x2 With Preference Sub

按上一节建议扩大 Stage 2：

```bash
python grpo_hotpotqa_mas.py ^
  --tasks 50 ^
  --val-tasks 20 ^
  --iterations 2 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --sft-dir .\hotpotqa_mas_sft_checkpoints_v2 ^
  --main-lora .\hotpotqa_mas_sft_checkpoints_v2\main_agent\main ^
  --sub-lora .\hotpotqa_sub_pref_100x250\sub ^
  --save-dir .\hotpotqa_mas_stage2_main_prefsub_50x2 ^
  --best-metric reward ^
  --reward-threshold 0.25 ^
  --train-main ^
  --no-train-sub
```

该实验耗时超过 1 小时；命令调用本身超时后训练进程继续运行，并最终正常写出：

```text
best/
main_step_1/
sub_step_1/
main_step_2/
sub_step_2/
```

独立完整 MAS eval，20 条 validation、每题 2 samples：

| Main | Sub | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---|---:|---:|---:|---:|---:|
| MAS SFT v2 | Preference from SFT | 0.208 | 0.500 | 0.346 | 0.275 | 0.408 |
| Stage 2 Main 20x1 | Preference from SFT | 0.299 | 0.475 | 0.404 | 0.401 | 0.491 |
| Stage 2 Main 50x2 | Preference from SFT | 0.330 | 0.575 | 0.446 | 0.403 | 0.517 |

结论：

```text
扩大 Stage 2 继续提升：
answer_f1 0.299 -> 0.330
evidence  0.475 -> 0.575
reward    0.404 -> 0.446
```

这说明“冻结 retrieval-strong Sub，只训练 Main evidence integration”不是偶然 smoke 信号，而是目前最稳定的 MAS 提升路径。

当前最强 MAS checkpoint 更新为：

```text
Main:
  .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main

Sub:
  .\hotpotqa_mas_stage2_main_prefsub_50x2\best\sub
```

下一步可以进入轻量 joint GRPO，但建议非常保守：

```text
init:
  Stage 2 Main 50x2 best
  Preference Sub

train:
  Main + Sub

settings:
  tasks=30 or 50
  iterations=1
  lr <= 3e-6
  reward_threshold >= 0.35
  best_metric=reward or answer_f1
```

如果 joint GRPO 低于 Stage 2 best，应回退到 Stage 2 Main-only 路线，继续扩大 Main-only 而不是强行联合训练。

## 20. Conservative Joint GRPO From Stage 2 Best

在 Stage 2 50x2 best 基础上，尝试保守 joint GRPO：

```bash
python grpo_hotpotqa_mas.py ^
  --tasks 30 ^
  --val-tasks 20 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --main-lora .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main ^
  --sub-lora .\hotpotqa_mas_stage2_main_prefsub_50x2\best\sub ^
  --save-dir .\hotpotqa_mas_joint_from_stage2_30x1 ^
  --lr 3e-6 ^
  --reward-threshold 0.35 ^
  --best-metric reward ^
  --train-main ^
  --train-sub
```

训练内部 validation：

```text
init val:
  reward        = 0.348
  answer_f1     = 0.219
  evidence      = 0.475
  sub_retrieval = 0.450

iter1 train:
  reward        = 0.520
  answer_f1     = 0.467
  evidence      = 0.467
  updates main  = 14
  updates sub   = 14

iter1 val:
  reward        = 0.460
  answer_f1     = 0.378
  evidence      = 0.475
  saved best
```

独立完整 MAS eval，20 条 validation、每题 2 samples：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Stage 2 Main 50x2 + Preference Sub | 0.330 | 0.575 | 0.446 | 0.403 | 0.517 |
| Conservative Joint 30x1 | 0.424 | 0.525 | 0.502 | 0.580 | 0.631 |

结论：

```text
保守 joint GRPO 成功超过 Stage 2：
answer_f1      0.330 -> 0.424
reward         0.446 -> 0.502
best_answer_f1 0.403 -> 0.580
best_reward    0.517 -> 0.631
```

这是目前最强 MAS checkpoint：

```text
Main:
  .\hotpotqa_mas_joint_from_stage2_30x1\best\main

Sub:
  .\hotpotqa_mas_joint_from_stage2_30x1\best\sub
```

同时，对 joint 后的 Sub 单独做 oracle eval：

| Sub checkpoint | support_read_recall | answer_f1 | evidence | reward | best_support_read_recall | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Preference from SFT | 0.525 | 0.272 | 0.537 | 0.398 | 0.625 | 0.348 | 0.468 |
| Joint 30x1 Sub | 0.425 | 0.224 | 0.438 | 0.344 | 0.450 | 0.324 | 0.422 |

这说明 joint 的提升并不是因为 Sub 单独 oracle 能力继续增强；相反，Sub 单独指标有所下降。完整 MAS 提升更可能来自：

```text
Main 和 Sub 的分布协同变好；
Main 更适配 joint 后 Sub 的输出；
完整轨迹 reward 强化了 answer synthesis。
```

因此下一步要避免无限 joint，把当前 checkpoint 视为新 best，并做两件事：

1. 用更大 validation 或不同 seed 复核 `0.502 reward / 0.424 answer_f1` 是否稳定。
2. 如果稳定，再尝试 50x1 joint；如果不稳定，保留 30x1 joint best，不继续破坏 Sub retrieval。

## 21. Larger Validation Recheck

为了复核 joint 30x1 的提升是否只来自 20 条 validation 的偶然性，改用 30 条 validation、每题 2 samples，对 Stage 2 best 和 joint best 做同设置评估。

说明：最初尝试了 50 条 validation，但单次评估超过 30 分钟且工具调用超时后输出没有保留。因此改为 30 条 validation，并给足超时时间。

评估命令：

```bash
python analyze_hotpotqa_mas_results.py ^
  --val-jsonl .\hotpotqa_data\val.jsonl ^
  --tasks 30 ^
  --samples 2 ^
  --main-lora <MAIN> ^
  --sub-lora <SUB> ^
  --max-tokens 120 ^
  --sub-steps 3
```

结果：

| Model | tasks | samples | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage 2 Main 50x2 + Preference Sub | 30 | 2 | 0.192 | 0.417 | 0.318 | 0.307 | 0.418 |
| Conservative Joint 30x1 | 30 | 2 | 0.283 | 0.475 | 0.393 | 0.395 | 0.496 |

结论：

```text
joint best 在更大的 30 条 validation 上仍然显著优于 Stage 2 best：
answer_f1      +0.091
evidence       +0.058
reward         +0.075
best_answer_f1 +0.088
best_reward    +0.078
```

但也要注意：30 条 validation 的绝对分数低于 20 条 validation：

```text
joint 20-task eval:
  answer_f1 = 0.424
  reward    = 0.502

joint 30-task eval:
  answer_f1 = 0.283
  reward    = 0.393
```

这说明 20 条评估偏乐观，HotpotQA validation 切片/采样方差仍然明显。更可靠的结论应该是相对比较：

```text
在同一评估切片下，joint 30x1 > Stage 2 50x2。
```

下一步建议：

1. 保留 `hotpotqa_mas_joint_from_stage2_30x1` 作为当前 best。
2. 增加 evaluator 的 `--seed` / `--offset`，做多切片评估，避免只看前 N 条。
3. 暂不继续更大 joint，先把 evaluation protocol 固化。

已完成 evaluator 固化：

```text
analyze_hotpotqa_mas_results.py:
  --seed
  --offset

analyze_hotpotqa_sub_oracle.py:
  --seed
  --offset
```

后续可以用不同 offset 做 validation slices：

```bash
--offset 0  --tasks 20
--offset 20 --tasks 20
--offset 40 --tasks 20
```

并固定 `--seed` 来降低采样不可复现问题。

## 22. Multi-Offset Evaluation Suite

新增：

```text
run_hotpotqa_eval_suite.py
```

用途：

```text
1. 跑完整 MAS 多切片评估
2. 跑 Sub oracle 多切片评估
3. 每条结果即时写入 results.jsonl
4. 自动生成 summary.md
5. 同时输出 macro average 和 task-weighted average
```

本轮设置：

```bash
python run_hotpotqa_eval_suite.py ^
  --suite mas ^
  --offsets 0 20 40 ^
  --tasks 20 ^
  --samples 2 ^
  --out-dir .\hotpotqa_eval_suite_mas_offsets ^
  --seed 123

python run_hotpotqa_eval_suite.py ^
  --suite sub ^
  --offsets 0 20 40 ^
  --tasks 20 ^
  --samples 2 ^
  --out-dir .\hotpotqa_eval_suite_sub_offsets ^
  --seed 123
```

注意：当前 validation 文件只够 offset 40 取到 10 条，因此 offset 40 的 `tasks=10`。这会让普通 macro average 给这 10 条过高权重，所以更应该看 task-weighted average。

### Full MAS Multi-Offset

逐切片结果：

| Model | offset | tasks | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage 2 Main 50x2 | 0 | 20 | 0.428 | 0.575 | 0.515 | 0.553 | 0.612 |
| Stage 2 Main 50x2 | 20 | 20 | 0.290 | 0.463 | 0.395 | 0.418 | 0.503 |
| Stage 2 Main 50x2 | 40 | 10 | 0.266 | 0.325 | 0.351 | 0.304 | 0.402 |
| Joint 30x1 | 0 | 20 | 0.333 | 0.500 | 0.433 | 0.428 | 0.524 |
| Joint 30x1 | 20 | 20 | 0.310 | 0.425 | 0.402 | 0.462 | 0.518 |
| Joint 30x1 | 40 | 10 | 0.398 | 0.425 | 0.463 | 0.462 | 0.533 |

Task-weighted average：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Stage 2 Main 50x2 | 0.340 | 0.480 | 0.434 | 0.449 | 0.527 |
| Joint 30x1 | 0.337 | 0.455 | 0.427 | 0.448 | 0.524 |

结论修正：

```text
在多 offset、task-weighted 评估下，Joint 30x1 没有稳定超过 Stage 2。
二者基本打平，Stage 2 略高。
```

这说明之前 20/30 条单切片上看到的 joint 明显提升，确实存在切片/采样方差。当前不能宣称 joint 已经稳定胜出。

### Sub Oracle Multi-Offset

Task-weighted average：

| Sub | support_read_recall | answer_f1 | evidence | reward | best_support_read_recall | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Preference from SFT | 0.400 | 0.276 | 0.410 | 0.375 | 0.510 | 0.391 | 0.477 |
| Stage 2 Sub | 0.400 | 0.276 | 0.410 | 0.375 | 0.510 | 0.391 | 0.477 |
| Joint 30x1 Sub | 0.440 | 0.309 | 0.455 | 0.407 | 0.550 | 0.434 | 0.513 |

这里 Stage 2 Sub 和 Preference Sub 完全一致是合理的：Stage 2 冻结了 Sub，所以保存出来的 Sub 本质上还是 preference checkpoint。

Sub oracle 结论：

```text
多切片下，Joint Sub 没有被破坏，反而略高于 Preference/Stage2 Sub。
```

这和单次 20 条 Sub oracle 的结论不同，进一步说明评估必须多切片固定 seed。

当前最稳妥结论：

```text
Stage 2 和 Joint 是当前两个候选 best。
Full MAS task-weighted 上 Stage 2 略高；
Sub oracle 上 Joint 略高。
```

下一步不应继续训练，而应补两个评估：

1. 扩大 validation 数据量，避免 offset 40 只有 10 条。
2. 做 Main answerer oracle-sub-result eval，判断 Stage2/Joint 的差异到底来自 Main 整合能力还是 Sub 输出分布。

## 23. Final Convergence Experiments

为了停止无边界试验，补齐两个收敛判断实验：

1. Main answerer oracle-sub-result eval。
2. Direct Main baseline vs current MAS candidates。

### 23.1 Main Oracle-Sub-Result Eval

新增：

```text
analyze_hotpotqa_main_oracle_answer.py
```

评估方式：

```text
Question + oracle Sub result -> Main final answer
```

其中 oracle Sub result 直接包含：

```text
<result>gold answer | evidence: gold_doc_ids</result>
```

比较：

```text
MAS SFT v2 Main
Stage 2 Main 50x2
Joint 30x1 Main
```

多 offset 结果全部为：

```text
answer_f1 = 1.000
evidence  = 1.000
reward    = 0.900
```

结论：

```text
这个 oracle eval 太容易，因为 Sub result 已经显式包含 gold answer。
它只能说明：当 Sub 给出干净答案线索时，Main answerer 不是瓶颈。
```

也就是说，当前 MAS 的难点不是 Main “看见正确答案还不会抄”，而是：

```text
真实 Sub result 往往不够干净；
Main 需要从 noisy/incomplete Sub summary 中恢复答案；
Main/Sub 输出分布需要协同。
```

后续如果继续做 Main oracle eval，应改成 harder setting：

```text
只给 supporting doc text / evidence ids，不直接给 gold answer。
```

### 23.2 Direct Main Baseline vs MAS

用同一份 `hotpotqa_data/val.jsonl` 50 条 validation、每题 2 samples，评估 direct Main：

```bash
python analyze_hotpotqa_results.py ^
  --val-jsonl .\hotpotqa_data\val.jsonl ^
  --tasks 50 ^
  --samples 2 ^
  --lora .\hotpotqa_sft_checkpoints\main_agent\main ^
  --max-tokens 120 ^
  --research-steps 3

python analyze_hotpotqa_results.py ^
  --val-jsonl .\hotpotqa_data\val.jsonl ^
  --tasks 50 ^
  --samples 2 ^
  --lora .\hotpotqa_grpo_150x3_answerbest\best\main ^
  --max-tokens 120 ^
  --research-steps 3
```

结果：

| Model | tasks | samples | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct Main SFT | 50 | 2 | 0.384 | 0.565 | 0.482 | 0.445 | 0.541 |
| Direct Main GRPO 150x3 | 50 | 2 | 0.422 | 0.575 | 0.510 | 0.500 | 0.568 |

对照当前 MAS 多 offset task-weighted：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Stage 2 MAS | 0.340 | 0.480 | 0.434 | 0.449 | 0.527 |
| Joint MAS | 0.337 | 0.455 | 0.427 | 0.448 | 0.524 |
| Direct Main GRPO 150x3 | 0.422 | 0.575 | 0.510 | 0.500 | 0.568 |

结论：

```text
当前 MAS 还没有超过 direct Main GRPO。
```

但 MAS 的实验不是无效：

```text
Sub action preference 能提升 retrieval。
Main-only Stage 2 能提升 evidence integration。
Conservative joint 在单切片上有效，但多切片后只和 Stage2 打平。
```

阶段性判断：

```text
我们已经复现出 M-GRPO-style MAS 的完整训练闭环和有效组件；
但在这个 HotpotQA local-context 环境里，当前 MAS 架构还没有体现出超过 direct single-agent tool-use 的优势。
```

这给出了清晰的下一步方向：

1. 不再继续盲目扩大 MAS joint。
2. 先修 evaluation/data：扩 validation，避免只有 50 条。
3. 改 MAS 任务结构，让 Main/Sub 分工更有必要。
   当前 HotpotQA local context 对 direct Main 太友好，单 agent 直接 search/read/answer 已经很强。
4. 如果目标是证明 MAS 优势，应引入更长上下文、更多文档、更多子问题、或者 multi-agent specialization 更明显的任务。
