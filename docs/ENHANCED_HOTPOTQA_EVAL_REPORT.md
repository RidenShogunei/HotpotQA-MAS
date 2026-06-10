# Enhanced HotpotQA Evaluation Report

## Setup

正式评估使用 enhanced HotpotQA：

```text
val_jsonl = .\hotpotqa_data_enhanced\val.jsonl
offsets = 0, 20, 40, 60, 80
tasks_per_offset = 20
samples = 2
seed = 123
docs_per_task = 30
```

## Direct Main

结果文件：

```text
.\hotpotqa_direct_eval_enhanced_offsets\summary.md
```

Task-weighted average：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Direct Main SFT | 0.301 | 0.455 | 0.402 | 0.364 | 0.449 |
| Direct Main GRPO 150x3 | 0.372 | 0.515 | 0.463 | 0.410 | 0.491 |

Direct GRPO 在 enhanced 环境上仍有迁移收益：

```text
answer_f1 +0.071
reward    +0.061
```

## MAS

结果文件：

```text
.\hotpotqa_mas_eval_enhanced_offsets\summary.md
```

Task-weighted average：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| MAS Stage 2 | 0.313 | 0.372 | 0.394 | 0.400 | 0.462 |
| MAS Joint 30x1 | 0.286 | 0.372 | 0.375 | 0.366 | 0.436 |

MAS 内部结论：

```text
Stage 2 > Joint。
```

## Direct vs MAS

核心对照：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Direct Main GRPO 150x3 | 0.372 | 0.515 | 0.463 | 0.410 | 0.491 |
| MAS Stage 2 | 0.313 | 0.372 | 0.394 | 0.400 | 0.462 |
| MAS Joint 30x1 | 0.286 | 0.372 | 0.375 | 0.366 | 0.436 |

正式结论：

```text
Enhanced 30-doc 环境确实让任务更难，但当前旧 MAS checkpoint 仍没有超过 direct Main GRPO。
```

相比原环境，direct 和 MAS 的 reward gap 略有缩小：

```text
原环境:
  Direct GRPO reward ~= 0.510
  MAS Stage2 reward  ~= 0.434
  gap ~= 0.076

Enhanced:
  Direct GRPO reward = 0.463
  MAS Stage2 reward  = 0.394
  gap = 0.069
```

方向是对的，但 30 docs 还不足以自然产生 MAS 优势。

## Enhanced Sub Oracle

结果文件：

```text
.\hotpotqa_sub_eval_enhanced_offsets\summary.md
```

Task-weighted average：

| Sub | support_read_recall | answer_f1 | evidence | reward | best_support_read_recall | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Preference / Stage2 Sub | 0.378 | 0.285 | 0.378 | 0.375 | 0.415 | 0.356 | 0.430 |
| Joint Sub | 0.355 | 0.275 | 0.357 | 0.364 | 0.385 | 0.332 | 0.410 |

Sub oracle 结论：

```text
30-doc enhanced 环境下，Sub retrieval 下降明显。
Preference/Stage2 Sub 仍优于 Joint Sub。
```

这定位了 MAS 在 enhanced 上输给 direct 的主要原因：

```text
Sub 在更多 distractors 下没有足够强的 retrieval 能力；
Joint 会轻微损伤 Sub retrieval；
Main 即便能整合，也拿不到足够好的 Sub evidence。
```

## Next Step

现在不应该直接继续用旧 checkpoint 做 joint。下一步应该在 enhanced 环境上重新训练 Sub retrieval：

```text
1. 用 enhanced train 生成新的 preference pairs。
2. 从现有 Preference Sub 出发，训练 enhanced preference Sub。
3. 用 enhanced Sub oracle 验证 support_read_recall 是否超过 0.378。
4. 冻结 enhanced Sub，再训练 Main。
```

推荐下一条实验：

```bash
python train_hotpotqa_sub_preferences.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --tasks 300 ^
  --max-pairs 900 ^
  --epochs 1 ^
  --sub-lora .\hotpotqa_sub_pref_100x250\sub ^
  --save-dir .\hotpotqa_sub_pref_enhanced_300x900\sub ^
  --lr 5e-6 ^
  --beta 2.0 ^
  --sft-weight 0.05
```

成功标准：

```text
Enhanced Sub oracle:
support_read_recall > 0.378
reward > 0.375
```

## Enhanced Sub Preference Training

按上述路线，使用 enhanced train 重新训练 Sub retrieval：

```bash
python train_hotpotqa_sub_preferences.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --tasks 300 ^
  --max-pairs 900 ^
  --epochs 1 ^
  --sub-lora .\hotpotqa_sub_pref_100x250\sub ^
  --save-dir .\hotpotqa_sub_pref_enhanced_300x900\sub ^
  --lr 5e-6 ^
  --beta 2.0 ^
  --sft-weight 0.05 ^
  --max-length 1536
```

训练结果：

```text
tasks = 300
pairs = 900
epoch = 1
loss = 0.0037
margin = 5.3887
```

## Enhanced Sub Oracle After Preference

评估：

```bash
python analyze_hotpotqa_sub_oracle.py ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --offset 0 ^
  --tasks 100 ^
  --samples 2 ^
  --sub-lora .\hotpotqa_sub_pref_enhanced_300x900\sub ^
  --max-tokens 120 ^
  --sub-steps 3 ^
  --seed 123
```

结果：

| Sub | support_read_recall | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|
| Previous Preference / Stage2 Sub | 0.378 | 0.285 | 0.378 | 0.375 | 0.356 | 0.430 |
| Enhanced Preference Sub | 0.502 | 0.440 | 0.502 | 0.508 | 0.516 | 0.566 |

结论：

```text
Enhanced preference training 成功修复了 Sub retrieval：
support_read_recall 0.378 -> 0.502
reward              0.375 -> 0.508
```

这说明 MAS 在 enhanced 上的瓶颈定位正确：旧 Sub 不是不会工具调用，而是在更多 distractors 下 retrieval 不够强。

## Full MAS With Enhanced Sub

使用：

```text
Main:
  .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main

Sub:
  .\hotpotqa_sub_pref_enhanced_300x900\sub
```

评估 enhanced val 前 100 条、每题 2 samples：

```bash
python analyze_hotpotqa_mas_results.py ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --offset 0 ^
  --tasks 100 ^
  --samples 2 ^
  --main-lora .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main ^
  --sub-lora .\hotpotqa_sub_pref_enhanced_300x900\sub ^
  --max-tokens 120 ^
  --sub-steps 3 ^
  --seed 123
```

结果：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Old MAS Stage 2, enhanced multi-offset | 0.313 | 0.372 | 0.394 | 0.400 | 0.462 |
| MAS Stage 2 Main + Enhanced Sub, 100 tasks | 0.407 | 0.512 | 0.488 | 0.500 | 0.560 |

同一 enhanced 100 题上，对比 direct GRPO：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Direct Main GRPO 150x3 | 0.365 | 0.515 | 0.458 | 0.401 | 0.485 |
| MAS Stage 2 Main + Enhanced Sub | 0.407 | 0.512 | 0.488 | 0.500 | 0.560 |

关键结论：

```text
在 enhanced 100-task eval 上，MAS 首次超过 direct Main GRPO：
answer_f1 0.365 -> 0.407
reward    0.458 -> 0.488
best_answer_f1 0.401 -> 0.500
best_reward    0.485 -> 0.560
```

这是当前最重要的阶段性进展。它说明：

```text
1. 增强环境让 multi-agent 分工更有价值。
2. Sub retrieval 经过 enhanced preference training 后成为有效组件。
3. 冻结 Stage 2 Main + 强化后的 Sub，可以超过 direct Main GRPO。
```

当前最强 enhanced MAS checkpoint：

```text
Main:
  .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main

Sub:
  .\hotpotqa_sub_pref_enhanced_300x900\sub
```

下一步建议：

```text
1. 对 MAS Stage2+EnhancedSub 做完整 offsets 0,20,40,60,80 多切片评估。
2. 如果仍然超过 direct GRPO，再冻结 EnhancedSub 训练 Main。
3. 最后谨慎做 enhanced joint GRPO。
```

## Multi-Offset Eval: Stage2 Main + Enhanced Sub

对当前最强 enhanced MAS 组合做正式多 offset 评估：

```bash
python run_hotpotqa_eval_suite.py ^
  --suite mas ^
  --model-names stage2_main_enhanced_sub ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --offsets 0 20 40 60 80 ^
  --tasks 20 ^
  --samples 2 ^
  --out-dir .\hotpotqa_mas_eval_enhanced_sub_offsets ^
  --max-tokens 120 ^
  --sub-steps 3 ^
  --seed 123
```

逐切片结果：

| offset | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.456 | 0.537 | 0.527 | 0.574 | 0.617 |
| 20 | 0.399 | 0.450 | 0.469 | 0.483 | 0.538 |
| 40 | 0.447 | 0.512 | 0.515 | 0.520 | 0.569 |
| 60 | 0.392 | 0.537 | 0.482 | 0.528 | 0.585 |
| 80 | 0.296 | 0.487 | 0.405 | 0.377 | 0.469 |

Task-weighted average：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Direct Main GRPO 150x3 | 0.372 | 0.515 | 0.463 | 0.410 | 0.491 |
| MAS Stage2 Main + Enhanced Sub | 0.398 | 0.505 | 0.480 | 0.496 | 0.556 |

正式结论：

```text
MAS Stage2 Main + Enhanced Sub 在 enhanced 多 offset 评估上稳定超过 direct Main GRPO。
```

提升：

```text
answer_f1      +0.026
reward         +0.017
best_answer_f1 +0.086
best_reward    +0.065
```

这修正了前面的阶段判断：

```text
旧 MAS checkpoint 没有超过 direct；
但在 enhanced train 上重新训练 Sub retrieval 后，MAS 开始超过 direct。
```

关键原因：

```text
增强环境让 direct Main 的单体 search/read 压力变大；
Enhanced Sub preference 显著提升 retrieval；
Stage2 Main 已经具备利用 Sub evidence 的能力；
两者组合后形成了真正的 MAS 优势。
```

当前 best checkpoint：

```text
Main:
  .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main

Sub:
  .\hotpotqa_sub_pref_enhanced_300x900\sub
```

下一步：

```text
1. 冻结 Enhanced Sub，在 enhanced train 上继续训练 Main。
2. 只在 Main-only 继续提升稳定后，再尝试 enhanced joint。
3. 不建议直接 joint，因为之前 joint 容易损伤 Sub retrieval。
```

## Enhanced Main-only Follow-up

尝试冻结 Enhanced Sub，继续训练 Main。

第一版：

```bash
python grpo_hotpotqa_mas.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --tasks 50 ^
  --val-tasks 40 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --main-lora .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main ^
  --sub-lora .\hotpotqa_sub_pref_enhanced_300x900\sub ^
  --save-dir .\hotpotqa_mas_enhanced_mainonly_50x1 ^
  --lr 3e-6 ^
  --reward-threshold 0.35 ^
  --train-main ^
  --no-train-sub
```

结果：

```text
init val reward = 0.467
iter val reward = 0.421
```

该配置过拟合训练切片，validation 下降。

随后改为更保守配置：

```bash
python grpo_hotpotqa_mas.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --tasks 50 ^
  --val-tasks 40 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --main-lora .\hotpotqa_mas_stage2_main_prefsub_50x2\best\main ^
  --sub-lora .\hotpotqa_sub_pref_enhanced_300x900\sub ^
  --save-dir .\hotpotqa_mas_enhanced_mainonly_conservative_50x1 ^
  --lr 1e-6 ^
  --reward-threshold 0.5 ^
  --train-main ^
  --no-train-sub
```

内部 validation：

```text
init val reward = 0.415
iter val reward = 0.433
saved best
```

外部 enhanced 100-task eval：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Stage2 Main + Enhanced Sub | 0.407 | 0.512 | 0.488 | 0.500 | 0.560 |
| Conservative Main-only 50x1 | 0.409 | 0.512 | 0.489 | 0.500 | 0.560 |

结论：

```text
保守 Main-only 没有破坏当前 best，但提升极小：
answer_f1 +0.002
reward    +0.001
```

这说明当前 Main-only 路线已经接近饱和。当前 best 可以更新为 conservative Main-only，但实际差异可以视作打平。

当前推荐 checkpoint：

```text
Main:
  .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\main

Sub:
  .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\sub
```

下一步建议：

```text
1. 不继续扩大 Main-only。
2. 如果继续训练，应优先提升 Sub retrieval 或做 carefully constrained joint。
3. 更可靠的下一步是提高 docs_per_task 到 50，验证 MAS 优势是否进一步扩大。
```

## Conservative Enhanced Joint Attempt

为了验证 joint 是否能在 staged best 基础上继续提升，尝试从当前 best 初始化做保守 joint：

```bash
python grpo_hotpotqa_mas.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --tasks 50 ^
  --val-tasks 40 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --main-lora .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\main ^
  --sub-lora .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\sub ^
  --save-dir .\hotpotqa_mas_enhanced_joint_conservative_50x1 ^
  --lr 1e-6 ^
  --reward-threshold 0.55 ^
  --train-main ^
  --train-sub
```

内部 validation：

```text
init val reward = 0.481
iter val reward = 0.404
updates main = 22
updates sub  = 22
```

因为 validation 下降，`best/` 没有更新，仍是训练前 staged best。

外部 enhanced 100-task eval 对 `main_step_1/sub_step_1`：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Staged best | 0.409 | 0.512 | 0.489 | 0.500 | 0.560 |
| Joint step1 | 0.359 | 0.500 | 0.451 | 0.440 | 0.514 |

Sub oracle 对 `sub_step_1`：

| Sub | support_read_recall | answer_f1 | evidence | reward | best_support_read_recall | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Enhanced Preference Sub | 0.502 | 0.440 | 0.502 | 0.508 | 0.525 | 0.516 | 0.566 |
| Joint step1 Sub | 0.480 | 0.384 | 0.482 | 0.465 | 0.515 | 0.441 | 0.510 |

结论：

```text
当前 conservative enhanced joint 失败。
它同时降低完整 MAS reward 和 Sub oracle reward。
```

这再次说明，当前阶段的最优策略不是 joint，而是 staged：

```text
Sub preference retrieval training
-> freeze Sub
-> Main-only conservative tuning
```

当前 best 仍然是：

```text
Main:
  .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\main

Sub:
  .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\sub
```

如果后续还要尝试 joint，需要改变 joint 机制，而不是简单继续降低 lr：

```text
1. 冻结 Sub action layers / 只训 Sub summary。
2. 对 Sub action 混入 preference replay，防止 retrieval 遗忘。
3. 对 Main/Sub 使用不同 reward threshold。
4. 或者使用真正 group advantage / DPO-style objective，而不是 best-of-group weighted SFT。
```

## Joint With Enhanced Sub Reward

为了解决 joint 中 Sub reward 过于接近 Main reward 的问题，`grpo_hotpotqa_mas.py` 新增：

```text
--sub-reward-mode enhanced
```

Enhanced Sub reward：

```text
sub_train_reward =
  0.40 * support_read_recall
+ 0.25 * sub_summary_answer_f1
+ 0.15 * sub_evidence_recall
+ 0.10 * read_precision
+ 0.05 * action_valid
+ 0.05 * no_duplicate_read
```

同时，在 Main/Sub joint 且 `sub_reward_mode=enhanced` 时，candidate selection 改为混合目标：

```text
0.55 * main_reward + 0.45 * sub_train_reward
```

训练命令：

```bash
python grpo_hotpotqa_mas.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --tasks 50 ^
  --val-tasks 40 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --main-lora .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\main ^
  --sub-lora .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\sub ^
  --save-dir .\hotpotqa_mas_enhanced_joint_subreward_50x1 ^
  --lr 1e-6 ^
  --reward-threshold 0.45 ^
  --best-metric reward ^
  --sub-reward-mode enhanced ^
  --train-main ^
  --train-sub
```

内部 validation：

```text
init val:
  reward        = 0.391
  answer_f1     = 0.301
  evidence      = 0.400
  sub_train     = 0.437
  sub_retrieval = 0.400
  sub_precision = 0.425

iter val:
  reward        = 0.478
  answer_f1     = 0.418
  evidence      = 0.425
  sub_train     = 0.481
  sub_retrieval = 0.425
  sub_precision = 0.438
  saved best
```

外部 enhanced 100-task eval：

| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Staged best | 0.409 | 0.512 | 0.489 | 0.500 | 0.560 |
| Joint enhanced-reward best | 0.361 | 0.490 | 0.451 | 0.446 | 0.517 |

Sub oracle：

| Sub | support_read_recall | answer_f1 | evidence | reward | best_support_read_recall | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Enhanced Preference Sub | 0.502 | 0.440 | 0.502 | 0.508 | 0.525 | 0.516 | 0.566 |
| Joint enhanced-reward Sub | 0.495 | 0.373 | 0.495 | 0.460 | 0.535 | 0.436 | 0.511 |

结论：

```text
Enhanced Sub reward 比旧 joint 更好地保护了 retrieval：
support_read_recall 0.502 -> 0.495
```

但它仍然损伤了：

```text
Sub summary answer_f1: 0.440 -> 0.373
Full MAS reward:       0.489 -> 0.451
```

所以当前 joint 的主要问题已经从“retrieval 被严重破坏”变成：

```text
Sub summary / Main-Sub output distribution 被 joint 更新扰动。
```

最终判断：

```text
仅靠更合理的 scalar Sub reward 还不够。
当前 best-of-group weighted SFT 形式的 joint 仍低于 staged training。
```

后续如果继续改 joint，建议加入 replay/约束：

```text
1. Sub action preference replay，保护 retrieval。
2. Sub summary SFT replay，保护 summary format 和 answer clue。
3. Main answer replay，保护 Main 使用 Sub result 的能力。
4. Main/Sub 分别使用不同 update threshold。
5. 或实现真正 GRPO advantage，而不是只做 best sample 加权 SFT。
```

## Dynamic MAS Prototype

为了支持动态分派，新增第一版动态 MAS 组件：

```text
generate_hotpotqa_dynamic_mas_sft_data.py
analyze_hotpotqa_dynamic_mas_results.py
```

目标结构：

```text
Main plan:
  [mode]direct[/mode]
or:
  [mode]delegate[/mode]
  [subtask]...[/subtask]
  [subtask]...[/subtask]

Sub:
  每个 subtask 独立运行 search/read/summary
  多个 Sub instance 共享同一个 Sub LoRA

Main answer:
  汇总多个 Sub results
```

当前实现特性：

```text
1. 支持 direct/delegate 解析。
2. 支持最多 max_subagents 个 subtask。
3. 多个 Sub instance 共享 Sub adapter，但 history 独立。
4. 评估指标新增 direct_rate 和 avg_subtasks。
5. 如果 Main 没有生成 subtask，会 fallback 到 1 个默认 research subtask。
```

Smoke：

```bash
python generate_hotpotqa_dynamic_mas_sft_data.py ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --output .\hotpotqa_dynamic_mas_sft_data_smoke.jsonl ^
  --limit 5 ^
  --max-subtasks 2

python analyze_hotpotqa_dynamic_mas_results.py ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --tasks 2 ^
  --samples 1 ^
  --main-lora .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\main ^
  --sub-lora .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\sub ^
  --max-subagents 2
```

Smoke 结果：

```text
direct_rate = 0.000
avg_subtasks = 1.000
tool_valid = 1.000
```

因为现有 Main 没有接受过 dynamic plan SFT，所以 fallback 到 1 个默认 subtask 是预期行为。下一步需要：

```text
1. 生成正式 dynamic MAS SFT 数据。
2. 训练 dynamic Main adapter。
3. 再评估 direct_rate / avg_subtasks 是否真的动态化。
4. 最后接 dynamic GRPO。
```
## Dynamic MAS SFT Continuation

New artifacts:
```text
hotpotqa_dynamic_mas_sft_data.jsonl
hotpotqa_dynamic_mas_sft_continued_500x1/
hotpotqa_dynamic_mas_mainonly_sft_500x1/
```

Training data:
```text
500 enhanced HotpotQA train tasks
3000 SFT samples total
1000 Main samples
2000 Sub samples
max_subtasks = 2
direct_fraction = 0.0
```

Continuation runs:
```text
joint dynamic SFT:
  start main = hotpotqa_mas_enhanced_mainonly_conservative_50x1/best/main
  start sub  = hotpotqa_mas_enhanced_mainonly_conservative_50x1/best/sub
  epochs     = 1
  lr         = 5e-5
  main loss  = 0.1322
  sub loss   = 0.0557

main-only dynamic SFT:
  start main = hotpotqa_mas_enhanced_mainonly_conservative_50x1/best/main
  sub frozen = hotpotqa_mas_enhanced_mainonly_conservative_50x1/best/sub
  epochs     = 1
  lr         = 5e-5
  main loss  = 0.1295
```

Dynamic protocol eval, val offset 0, 20 hard tasks, 2 samples:
| Model | direct_rate | avg_subtasks | tool_valid | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Staged best, dynamic fallback | 0.000 | 1.000 | 1.000 | 0.315 | 0.400 | 0.400 | 0.422 | 0.496 |
| Joint dynamic SFT | 0.000 | 1.850 | 1.000 | 0.347 | 0.475 | 0.438 | 0.417 | 0.502 |
| Main-only dynamic SFT + frozen best Sub | 0.000 | 1.825 | 1.000 | 0.293 | 0.487 | 0.402 | 0.404 | 0.493 |

Fixed MAS protocol eval, same 20 hard tasks, 2 samples:
| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| Staged best | 0.456 | 0.537 | 0.527 | 0.574 | 0.617 |
| Joint dynamic SFT | 0.376 | 0.537 | 0.470 | 0.397 | 0.493 |
| Main-only dynamic SFT + frozen best Sub | 0.359 | 0.450 | 0.441 | 0.450 | 0.520 |

Interpretation:
```text
1. Dynamic SFT successfully teaches Main to emit multiple subtasks:
   avg_subtasks improves from 1.000 fallback to about 1.8.
2. Joint dynamic SFT is the best current dynamic-protocol checkpoint on this small eval.
3. However, both dynamic SFT variants underperform staged best under the old fixed MAS protocol.
4. Main-only dynamic SFT is not sufficient. Sub input distribution changes when Main emits focused per-document subtasks.
5. Training Sub helps dynamic protocol reward, but it also perturbs the fixed-protocol summary distribution.
```

Current decision:
```text
Keep staged best as the production/baseline checkpoint.
Use joint dynamic SFT only as a prototype checkpoint for dynamic MAS research.
The next real improvement should be dynamic SFT with replay/mixture:
  - preserve old fixed MAS plan/answer samples
  - preserve enhanced Sub preference/SFT samples
  - add dynamic multi-subtask samples
Then evaluate dynamic and fixed protocols together before any dynamic GRPO.
```

## Dynamic Mixture SFT and Reward Fix

Follow-up implementation:
```text
generate_hotpotqa_dynamic_mixture_sft_data.py
hotpotqa_dynamic_mixture_sft_data_300_v3.jsonl
```

Two important fixes were added:
```text
1. Dynamic Main planning now receives an explicit document catalog:
   Question + Available documents: Dxx: title

2. HotpotQAEnvironment.reward() now scores the last <result> block.
   MAS rollouts contain intermediate Sub <result> blocks before Main's final answer,
   so scoring the first <result> can accidentally evaluate the Sub summary instead of Main final.
```

Why the document catalog matters:
```text
Earlier dynamic Main was asked to emit focused subtasks such as:
  Find evidence from document D16 (...)

But the plan prompt only contained the question, not the local document list.
On validation, Main therefore guessed Dxx/title IDs. This made dynamic routing unstable.
```

Mixture v3 training:
```text
300 enhanced HotpotQA train tasks
4200 SFT samples
1200 Main samples
3000 Sub samples
fixed protocol replay + dynamic focused subtasks
epochs = 1
lr = 5e-5
max_length = 1536
main loss = 0.0298
sub loss = 0.0721
```

20-task hard validation, offset 0, samples 2:
| Model / Protocol | direct_rate | avg_subtasks | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Staged best / fixed MAS | - | - | 0.456 | 0.537 | 0.527 | 0.574 | 0.617 |
| Dynamic mixture v3 / dynamic MAS | 0.000 | 1.800 | 0.224 | 0.675 | 0.392 | 0.248 | 0.418 |
| Dynamic mixture v3 / fixed MAS | - | - | 0.388 | 0.450 | 0.462 | 0.400 | 0.475 |

Current interpretation:
```text
The dynamic system is no longer mainly failing at evidence selection:
  evidence = 0.675 under dynamic MAS.

The remaining bottleneck is final answer synthesis:
  answer_f1 = 0.224 under dynamic MAS.

So dynamic multi-Sub routing has partially worked:
  Main can produce ~1.8 subtasks and select better evidence when given document titles.

But Main has not learned to synthesize multiple focused Sub results into one clean final answer.
```

Next recommended experiment:
```text
Do not start dynamic GRPO yet.
First build a Main-answer-only continuation set where:
  - Sub results contain focused evidence snippets, not full answer strings.
  - Main is trained to produce one clean <result>answer | evidence: ...</result>.
  - Sub adapter is frozen.

Then evaluate:
  1. dynamic evidence stays high
  2. dynamic answer_f1 rises
  3. fixed MAS degradation remains bounded
```

## Dynamic Synthesis Main-Only SFT

Implemented:
```text
generate_hotpotqa_dynamic_synthesis_sft_data.py
hotpotqa_dynamic_synthesis_sft_data_500.jsonl
```

Purpose:
```text
Freeze Sub and train only Main's final answer synthesis step.
Input contains multiple focused Sub results with evidence snippets.
Target is one clean final:
  <result>answer | evidence: Dxx, Dyy</result>
```

Training:
```text
start main = hotpotqa_dynamic_mixture_sft_300x1_v3/main_agent
sub frozen = hotpotqa_dynamic_mixture_sft_300x1_v3/sub_agent
samples    = 500 Main-only synthesis samples
epochs     = 1
lr         = 3e-5
max_length = 1536
loss       = 0.2391
```

20-task hard validation, offset 0, samples 2:
| Model / Protocol | direct_rate | avg_subtasks | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dynamic mixture v3 / dynamic MAS | 0.000 | 1.800 | 0.224 | 0.675 | 0.392 | 0.248 | 0.418 |
| Synthesis 300x1 / dynamic MAS | 0.000 | 1.850 | 0.254 | 0.662 | 0.410 | 0.346 | 0.488 |
| Synthesis 500x1 / dynamic MAS | 0.000 | 1.825 | 0.316 | 0.700 | 0.461 | 0.418 | 0.542 |
| Synthesis 500x1 / fixed MAS | - | - | 0.388 | 0.450 | 0.462 | 0.400 | 0.475 |

Interpretation:
```text
Main-only synthesis SFT works.
It improves dynamic answer_f1 from 0.224 to 0.316 and dynamic reward from 0.392 to 0.461,
while keeping high evidence recall around 0.700.

The result is still below staged best fixed MAS on the same 20-task slice:
  staged best fixed reward = 0.527
  synthesis 500 dynamic reward = 0.461

But the dynamic line now has a concrete improvement path:
  planner/evidence is mostly working;
  final synthesis is improving with targeted Main-only SFT.
```

Next step:
```text
Run a larger validation sweep for synthesis 500x1 before GRPO.
If it holds across offsets, use it as the starting Main for Main-only dynamic GRPO.
Keep Sub frozen for the next RL step.
```

## Dynamic Multi-Offset Validation

Ran a 5-slice validation sweep:
```text
offsets = 0, 20, 40, 60, 80
tasks per offset = 10
samples = 2
val split = hotpotqa_data_enhanced/val.jsonl
```

Dynamic MAS averages:
| Model | direct_rate | avg_subtasks | answer_f1 | evidence | reward | best_answer_f1 | best_reward | tool_valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dynamic_mixture_v3 | 0.000 | 1.840 | 0.318 | 0.675 | 0.458 | 0.392 | 0.516 | 1.000 |
| dynamic_synthesis_500x1 | 0.000 | 1.830 | 0.344 | 0.655 | 0.472 | 0.394 | 0.516 | 1.000 |

Current fixed MAS baseline on the same slices:
| Model | answer_f1 | evidence | reward | best_answer_f1 | best_reward | tool_valid |
|---|---:|---:|---:|---:|---:|---:|
| fixed_staged_best | 0.413 | 0.495 | 0.488 | 0.524 | 0.575 | 1.000 |

Per-slice observation:
```text
Synthesis 500x1 improves the dynamic average:
  reward    0.458 -> 0.472
  answer_f1 0.318 -> 0.344

But it is not uniformly better across offsets.
The largest regression is offset 40:
  dynamic_mixture_v3 reward       = 0.414
  dynamic_synthesis_500x1 reward  = 0.331

Compared with fixed MAS, dynamic still has:
  higher evidence recall: 0.655 vs 0.495
  lower answer_f1:        0.344 vs 0.413
  lower reward:           0.472 vs 0.488
```

Decision:
```text
Synthesis 500x1 is a real improvement over dynamic_mixture_v3, but not stable enough
to treat as a solved starting point for joint GRPO.

The next RL step should still be Main-only if we proceed, with Sub frozen.
Before that, inspect offset 40 failures and improve synthesis robustness.
```

## Offset-40 Failure Trace And Sub Evidence Replay

Added diagnostic tooling:
```text
analyze_hotpotqa_dynamic_failures.py
generate_hotpotqa_dynamic_sub_evidence_sft_data.py
```

The failure tracer separates:
```text
plan_support_recall
read_support_recall
sub_summary_evidence_recall
sub_summary_answer_f1
final_answer_f1
final_evidence
final_reward
duplicate_reads
```

Offset 40 trace for synthesis 500x1 with the original dynamic mixture Sub:
| metric | value |
|---|---:|
| plan_support_recall | 0.550 |
| read_support_recall | 0.700 |
| sub_summary_evidence_recall | 0.650 |
| sub_summary_answer_f1 | 0.388 |
| final_answer_f1 | 0.475 |
| final_evidence | 0.700 |
| final_reward | 0.573 |
| duplicate_reads | 0.000 |

Main finding:
```text
The hard failures are not pure retrieval failures.
Several examples read both gold documents but still choose the wrong comparative/multi-hop answer.
This points to local Sub answer guessing and final Main synthesis as the main bottlenecks.
```

Tried Sub-only evidence-summary continuation:
```text
checkpoint = hotpotqa_dynamic_sub_evidence_500x1/sub_agent
base Sub   = hotpotqa_dynamic_mixture_sft_300x1_v3/sub_agent
data       = 500 tasks, 1000 Sub evidence-summary samples
lr         = 3e-5
epochs     = 1
```

Result on offset 40:
```text
answer_f1 = 0.046
evidence  = 0.625
reward    = 0.257
```

Interpretation:
```text
Pure Sub evidence-summary SFT causes catastrophic forgetting.
The Sub starts sounding like a summarizer but loses the action/read/answer behavior needed by the current dynamic rollout.
```

Then added action replay into the Sub evidence data:
```text
checkpoint = hotpotqa_dynamic_sub_evidence_replay_500x1/sub_agent
data       = 4500 Sub samples
contents   = fixed Sub action replay + focused Sub action replay + evidence-summary samples
lr         = 2e-5
epochs     = 1
```

Same offset 40, samples 2:
| Sub checkpoint | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| original dynamic mixture Sub | 0.137 | 0.675 | 0.331 | 0.140 | 0.338 |
| sub evidence replay 500x1 | 0.205 | 0.650 | 0.374 | 0.303 | 0.442 |

But multi-offset validation shows the replay Sub is not globally better:
| offset | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.012 | 0.725 | 0.254 | 0.022 | 0.260 |
| 20 | 0.104 | 0.675 | 0.308 | 0.108 | 0.316 |
| 40 | 0.205 | 0.650 | 0.374 | 0.303 | 0.442 |
| 60 | 0.010 | 0.750 | 0.257 | 0.012 | 0.275 |
| 80 | 0.108 | 0.650 | 0.306 | 0.206 | 0.374 |
| average | 0.088 | 0.690 | 0.300 | 0.130 | 0.333 |

Decision:
```text
Do not replace the current Sub with the evidence-replay Sub.
The replay version partially fixes offset 40, but collapses answer_f1 across other offsets.

The next useful direction is not more Sub-summary SFT in isolation.
The better path is a verifier/synthesis-style Main objective:
  - keep the current dynamic mixture Sub frozen;
  - expose Main to conflicting/partial Sub outputs;
  - train Main to ground the final answer in evidence, not blindly copy a Sub guess.
```

## Main Verifier/Synthesis SFT

Implemented:
```text
generate_hotpotqa_dynamic_verifier_sft_data.py
```

Purpose:
```text
Freeze the current dynamic mixture Sub.
Continue training only Main from synthesis 500x1.
Expose Main to noisy Sub results:
  - gold evidence-only sub results
  - wrong-first conflicting sub result
  - extra distractor sub result
  - partial gold evidence plus distractor
Target remains:
  <result>gold_answer | evidence: gold_doc_ids</result>
```

Training:
```text
data       = hotpotqa_dynamic_verifier_sft_data_500.jsonl
samples    = 500 tasks x 3 variants = 1500 Main-only samples
start Main = hotpotqa_dynamic_synthesis_mainonly_500x1/main_agent
Sub        = frozen hotpotqa_dynamic_mixture_sft_300x1_v3/sub_agent
epochs     = 1
lr         = 2e-5
max_length = 1536
loss       = 0.0845
```

Hard offset-40 slice, samples 2:
| Main checkpoint | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---|---:|---:|---:|---:|---:|
| synthesis 500x1 | 0.137 | 0.675 | 0.331 | 0.140 | 0.338 |
| verifier 500x1 | 0.312 | 0.675 | 0.454 | 0.500 | 0.600 |

Multi-offset validation:
| offset | answer_f1 | evidence | reward | best_answer_f1 | best_reward |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.400 | 0.750 | 0.530 | 0.400 | 0.540 |
| 20 | 0.317 | 0.450 | 0.412 | 0.417 | 0.482 |
| 40 | 0.312 | 0.675 | 0.454 | 0.500 | 0.600 |
| 60 | 0.220 | 0.600 | 0.374 | 0.407 | 0.525 |
| 80 | 0.400 | 0.500 | 0.480 | 0.550 | 0.605 |
| average | 0.330 | 0.595 | 0.450 | 0.455 | 0.550 |

Compared with previous dynamic synthesis 500x1 average:
```text
answer_f1      0.344 -> 0.330
evidence       0.655 -> 0.595
reward         0.472 -> 0.450
best_answer_f1 0.394 -> 0.455
best_reward    0.516 -> 0.550
```

Interpretation:
```text
Verifier SFT is not a drop-in replacement for synthesis 500x1.
It improves the hard offset-40 slice and improves best-of-samples metrics,
but the average single-sample reward is worse.

This suggests the model has learned a useful verifier mode, but it is unstable:
some samples are better, some are worse.
The next step should use verifier as a selection/reranking signal or distill best-of outputs,
not simply replace the deployed Main checkpoint.
```

## Advantage-Based MAS GRPO Patch

Problem found:
```text
The older grpo_hotpotqa.py and grpo_hotpotqa_mas.py scripts imported grpo_v4.py,
but grpo_v4.py was no longer present after project cleanup.

More importantly, the MAS "GRPO" loop was actually winner-only reward-filtered SFT:
  sample group candidates
  choose the best candidate
  SFT-update only that candidate
```

Implemented:
```text
grpo_v4.py
grpo_hotpotqa_mas.py --objective {best_of, advantage}
```

The new `advantage` objective computes group-relative advantages:
```text
main_advantage = normalize(candidate.final_reward within same task group)
sub_advantage  = normalize(candidate.sub_train_reward within same task group)
```

Then it updates all group candidates:
```text
positive advantage -> increase log-prob of that trajectory
negative advantage -> decrease log-prob of that trajectory
```

This is still not full paper-level M-GRPO:
```text
no old-policy ratio
no PPO/GRPO clipping on probability ratio
no explicit KL penalty
no per-turn critic
```

But it fixes the most important flaw in the previous lightweight trainer:
```text
bad samples are no longer silently discarded;
the group now provides relative positive and negative learning signals.
```

Smoke command:
```bash
python grpo_hotpotqa_mas.py ^
  --base-model .\models\qwen\Qwen2___5-1___5B-Instruct ^
  --train-jsonl .\hotpotqa_data_enhanced\train.jsonl ^
  --val-jsonl .\hotpotqa_data_enhanced\val.jsonl ^
  --tasks 1 ^
  --val-tasks 1 ^
  --iterations 1 ^
  --group-size 2 ^
  --eval-samples 1 ^
  --main-lora .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\main ^
  --sub-lora .\hotpotqa_mas_enhanced_mainonly_conservative_50x1\best\sub ^
  --save-dir .\hotpotqa_mas_advantage_smoke ^
  --max-response-len 80 ^
  --sub-steps 1 ^
  --lr 1e-7 ^
  --reward-threshold 0.45 ^
  --best-metric reward ^
  --sub-reward-mode enhanced ^
  --objective advantage ^
  --advantage-clip 1.0 ^
  --min-advantage 0.01 ^
  --train-main ^
  --train-sub
```

Smoke result:
```text
init val reward = 0.200
iter train reward = 0.300
updates main = 2
updates sub = 2
final val reward = 0.200
status = runs end-to-end
```

Decision:
```text
This patch restores the lightweight MAS RL script and makes the update closer to
real group-relative optimization.

The next real experiment should compare:
  1. old best_of objective
  2. new advantage objective

Use the same staged-best fixed MAS checkpoint, small LR, held-out validation,
and keep checkpoints selected only by validation metrics.
```

## Plancraft Bench Integration

Motivation:
```text
HotpotQA is useful for controlled evidence/reasoning diagnostics, but fixed MAS
is naturally strong there because most tasks need a small fixed number of support
documents.

Plancraft is a better intermediate benchmark for planning/tool-use/delegation:
  - deterministic environment
  - clear success reward
  - text-only interface
  - variable crafting complexity
  - official oracle planner/subplans
```

Implemented:
```text
plancraft_environment.py
analyze_plancraft_results.py
patch_plancraft_windows.py
requirements.txt += plancraft
```

Windows compatibility:
```text
Plancraft 0.4.9 has a Windows import bug in environment/recipes.py:
  tag_file.split("/") does not strip backslash paths

On Windows this can fail with:
  KeyError: 'acacia_logs'

Run once:
  python patch_plancraft_windows.py
```

Current wrapper:
```text
PlancraftBenchEpisode:
  reset()
  oracle_subplans()
  step(action)
  result()

Evaluator policies:
  oracle      = execute Plancraft's official oracle subplans
  impossible  = always emit the impossible action
```

MAS evaluator:
```text
analyze_plancraft_mas_results.py

Each Plancraft step:
  Sub receives current objective/inventory/history and suggests one low-level action.
  Main receives the same state plus Sub advice and outputs one executable action.
  Env executes the Main action and records success/validity/steps.
```

Smoke checks:
```bash
python analyze_plancraft_results.py ^
  --split val.small.easy ^
  --tasks 5 ^
  --policy oracle ^
  --out-dir .\plancraft_eval_oracle_smoke
```

Result:
```text
tasks = 5
success_rate = 1.000
reward = 1.000
avg_steps = 2.000
invalid_action_rate = 0.000
```

Larger oracle sanity check:
```bash
python analyze_plancraft_results.py ^
  --split val.small ^
  --tasks 20 ^
  --policy oracle ^
  --out-dir .\plancraft_eval_oracle_20
```

Result:
```text
tasks = 20
success_rate = 1.000
reward = 1.000
avg_steps = 2.250
invalid_action_rate = 0.000
```

Interpretation:
```text
Plancraft is now connected as a benchmark and the official oracle path can
drive the environment to success.

Zero-shot MAS smoke using the current HotpotQA fixed MAS checkpoint:
  tasks = 1
  success_rate = 0.000
  invalid_action_rate = 1.000

This is expected: the HotpotQA LoRA has never been trained on Plancraft slot/action
syntax, so it cannot be used as a meaningful Plancraft policy yet.

The next step is Plancraft-specific SFT before RL:
  - generate Main/Sub samples from oracle subplans
  - train action syntax and simple crafting behavior
  - then compare fixed MAS vs dynamic MAS on:
  - success_rate
  - valid_action_rate
  - avg_steps
  - impossible accuracy
  - subplan/delegation usage
```

## Plancraft MAS SFT Data

Question:
```text
Does Plancraft already provide oracle SFT data?
```

Answer:
```text
It provides benchmark examples and oracle planners/subplans, but not our
Main/Sub JSONL format for sft_trainer.py.

The package has a PlancraftDialogueDataset loader for oracle_trajectories/*
directories, but those trajectory files are not bundled in the installed package,
and that format is single-dialogue oriented rather than our category=main/sub
adapter split.
```

Implemented:
```text
generate_plancraft_mas_sft_data.py
```

Data construction:
```text
For each Plancraft train example:
  1. Reset official PlancraftGymWrapper.
  2. Ask the official oracle planner for subplans.
  3. For each low-level move/smelt/impossible action:
       Sub sample:
         observation + history -> next action advice
       Main sample:
         observation + history + Sub advice -> executable action
  4. Step the environment with the oracle action and continue.
```

Smoke data:
```text
5 train examples -> 30 samples
main = 15
sub  = 15
```

Plancraft SFT50 data:
```text
50 train examples -> 634 samples
main = 317
sub  = 317
```

Tried larger data:
```text
300 train examples -> 4036 samples
main = 2018
sub  = 2018
```

But 300-example SFT exceeded one hour on the RTX 4060 Laptop GPU and did not
reach the end-of-epoch save point, so it was stopped. The current trainer only
saves after each adapter finishes an epoch, so larger Plancraft SFT needs either
smaller shards or mid-epoch checkpointing.

SFT50 training:
```text
base_model = Qwen2.5-1.5B-Instruct local
data       = plancraft_mas_sft_data_50.jsonl
epochs     = 1
lr         = 2e-4
max_length = 1536
save_dir   = plancraft_mas_sft_50x1
```

Training result:
```text
Main prepared samples = 236 / 317
Main loss = 0.0015
Sub prepared samples = 235 / 317
Sub loss = 0.2258
```

The prepared sample count is lower than raw sample count because long history
prompts can be truncated so aggressively that no assistant target tokens remain.
This needs a shorter history window or larger max_length for larger runs.

Important evaluator bug fixed:
```text
analyze_plancraft_mas_results.py originally reused HotpotQA generate_one(),
which prepended <thinking> to every generation.

Plancraft SFT targets are action-only strings, so the evaluator now uses
generate_action() with no forced prefix.
```

SFT50 easy validation smoke:
```text
split = val.small.easy
tasks = 5
max_steps = 10
checkpoint = plancraft_mas_sft_50x1
```

Result:
```text
success_rate = 0.400
reward = 0.400
avg_steps = 6.800
invalid_action_rate = 0.280
```

Compared with zero-shot HotpotQA MAS checkpoint:
```text
success_rate = 0.000
invalid_action_rate = 1.000
```

Interpretation:
```text
The Plancraft SFT adapter is learning the action syntax and can solve simple
held-out crafting tasks.

Remaining failures are mostly:
  - invalid slots such as [A4]
  - repeated actions after the source slot has changed
  - weak state tracking after each environment transition

The next useful step is to improve data/windowing and train a larger
Plancraft-specific SFT checkpoint before any RL comparison.
```

## Plancraft Advantage GRPO Baseline

Added:
```text
grpo_plancraft_mas.py
```

This is the first Plancraft-specific GRPO script, separated from the older math
and HotpotQA training loops.

Training structure:
```text
shared Qwen base model
main LoRA adapter initialized from plancraft_mas_sft_50x1/main_agent
sub LoRA adapter initialized from plancraft_mas_sft_50x1/sub_agent
```

Rollout:
```text
Sub observes Plancraft state + history -> action advice
Main observes Plancraft state + history + Sub advice -> executable action
Environment executes Main action
Episode reward = success + valid_action bonus - step penalty
Group-relative normalized advantage is applied to both adapters
```

Validation:
```text
The script saves:
  save_dir/best/main
  save_dir/best/sub

Best checkpoint is selected by validation best_success_rate by default.
The evaluator can now use --eval-samples N, reporting both average and
best-of-N validation scores to reduce single-rollout noise.
```

Smoke check:
```text
train = 1
val = 1
iterations = 1
group_size = 2
max_steps = 3
eval_samples = 2
lr = 1e-7
```

Result:
```text
val:init success = 1.000
train success = 0.000
updates main/sub = 0 / 0
val success = 0.500
val best_success = 1.000
```

Interpretation:
```text
The training loop and checkpoint saving work. On a one-task smoke, both group
samples can receive the same reward, giving zero normalized advantage and no
update. This is expected for too-small groups/tasks and is not enough to judge
GRPO quality.
```

Small GRPO run:
```text
train = 5
val = 5
iterations = 1
group_size = 2
max_steps = 8
eval_samples = 2
lr = 1e-7
save_dir = plancraft_mas_grpo_adv_5x1
```

Result:
```text
val:init success = 0.400
val:init best_success = 0.400

train success = 0.200
train reward = 0.297
train valid_rate = 0.825
updates main/sub = 10 / 10

val success = 0.200
val best_success = 0.200
val valid_rate = 0.912
```

Interpretation:
```text
The corrected GRPO loop can now produce real Main/Sub updates, so we finally
have a meaningful SFT-vs-GRPO comparison path.

However, the first small GRPO update made held-out validation worse. The current
best checkpoint therefore remains the initialization checkpoint copied from
Plancraft SFT50, while the updated adapters are saved separately as step_1.

This suggests the immediate bottleneck is not just "can we run RL"; it is
rollout/reward stability. With only group_size=2 and sparse success reward, the
advantage signal is high variance. The next improvement should be either:
  - larger group_size / more validation samples,
  - denser trajectory-level reward for subgoal progress,
  - or a stronger Plancraft SFT base before RL.
```

## Plancraft M-GRPO-Style Reward Split

Changed:
```text
grpo_plancraft_mas.py
```

The previous Plancraft GRPO used one trajectory-level reward for both Main and
Sub. This has now been changed to a lightweight M-GRPO-style credit assignment:

```text
Main reward =
  main_success_weight * episode_success
+ main_valid_weight   * main_action_valid_rate
+ main_oracle_weight  * main_oracle_action_match_rate
- step_penalty        * steps

Sub reward =
  sub_global_weight    * episode_success
+ sub_valid_weight     * sub_advice_valid_rate
+ sub_oracle_weight    * sub_oracle_action_match_rate
+ sub_agreement_weight * sub_main_action_agreement_rate
- step_penalty         * steps
```

Implementation details:
```text
For each step:
  - read official Plancraft oracle next action from current state
  - check whether Sub advice parses as a valid Plancraft action
  - check whether Main action parses as a valid Plancraft action
  - check exact normalized match against oracle next action
  - check whether Sub and Main produced the same normalized action

For each rollout group:
  - normalize Main rewards into main_advantage
  - normalize Sub rewards into sub_advantage
  - update Main with main_advantage
  - update Sub with sub_advantage
```

Smoke:
```text
train = 1
val = 1
iterations = 1
group_size = 2
max_steps = 3
eval_samples = 2
```

Result:
```text
val:init success = 1.000
val:init main_reward = 1.380
val:init sub_reward = 1.480
main_oracle = 1.000
sub_oracle = 1.000

train success = 0.000
updates main/sub = 0 / 0
```

Interpretation:
```text
The split-reward code path works. The one-task group still had no useful
advantage difference, so it produced no updates.
```

Small M-GRPO-style run:
```text
train = 5
val = 5
iterations = 1
group_size = 2
max_steps = 8
eval_samples = 2
lr = 1e-7
save_dir = plancraft_mas_grpo_mgrpo_5x1
```

Result:
```text
val:init success = 0.400
val:init main_reward = 0.599
val:init sub_reward = 0.792
val:init main_oracle = 0.350
val:init sub_oracle = 0.350

train success = 0.200
train main_reward = 0.322
train sub_reward = 0.537
train valid_rate = 0.850
train main_oracle = 0.100
train sub_oracle = 0.100
train sub/main agreement = 1.000
updates main/sub = 26 / 26

val success = 0.200
val main_reward = 0.327
val sub_reward = 0.540
val valid_rate = 0.838
val main_oracle = 0.100
val sub_oracle = 0.100
```

Interpretation:
```text
The M-GRPO-style reward split produces real independent Main/Sub advantages and
more updates than the shared-reward version.

But validation still drops from 0.400 to 0.200 on this small setting, so this is
not yet an improvement over SFT. The best checkpoint remains the initialized SFT
checkpoint.

The important diagnostic is sub/main agreement = 1.000. Sub is not yet behaving
like an independent helper; it mostly mirrors the executable action Main will
take. That means the current Sub reward is too shallow. Exact oracle match and
format validity are not enough to teach useful decomposition.

Next reward improvement should move from action-level exact match to state-level
progress:
  - reward creating/intermediate target ingredients,
  - reward reducing missing recipe requirements,
  - penalize repeated no-op moves,
  - reward Sub advice that would improve state even if Main does not copy it.
```

## Structured Sub Interface

Changed:
```text
generate_plancraft_mas_sft_data.py
analyze_plancraft_mas_results.py
grpo_plancraft_mas.py
```

Motivation:
```text
The previous Sub interface asked Sub to output exactly the next low-level action.
That made Sub collapse into an action mirror of Main.

The new structured interface gives Sub a separate communication role:
  <subgoal>...</subgoal>
  <reason>...</reason>
  <action>...</action>

Main still outputs only one executable Plancraft action.
```

SFT data:
```text
python generate_plancraft_mas_sft_data.py ^
  --split train ^
  --limit 50 ^
  --max-steps 30 ^
  --structured-sub ^
  --output .\plancraft_mas_structured_sft_data_50.jsonl
```

Generated:
```text
634 samples
main = 317
sub = 317
```

Training:
```text
base_model = Qwen2.5-1.5B-Instruct local
init main = plancraft_mas_sft_50x1/main_agent
init sub  = plancraft_mas_sft_50x1/sub_agent
data      = plancraft_mas_structured_sft_data_50.jsonl
epochs    = 1
lr        = 1e-4
max_len   = 2048
save_dir  = plancraft_mas_structured_sft_50x1
```

Training result:
```text
Main prepared samples = 271 / 317
Main loss = 0.0000
Sub prepared samples = 273 / 317
Sub loss = 0.1997
```

Structured SFT easy5 evaluation:
```text
split = val.small.easy
tasks = 5
max_steps = 10
max_tokens = 120
structured_sub = true
```

Result:
```text
success_rate = 0.200
reward = 0.200
avg_steps = 8.400
invalid_action_rate = 0.040
```

Comparison:
```text
Old action-only Plancraft SFT50 easy5:
  success_rate = 0.400
  invalid_action_rate = 0.280

Structured Plancraft SFT50 easy5:
  success_rate = 0.200
  invalid_action_rate = 0.040
```

Interpretation:
```text
The structured interface greatly improves action validity, but reduces solved
tasks on this tiny easy5 slice. Main can parse structured advice, but the new
interface likely needs more SFT data before it matches the old action-only
policy's success rate.
```

Structured M-GRPO-style run:
```text
init = plancraft_mas_structured_sft_50x1
train = 5
val = 5
iterations = 1
group_size = 2
max_steps = 8
max_response_len = 120
eval_samples = 2
lr = 1e-7
structured_sub = true
save_dir = plancraft_mas_grpo_structured_mgrpo_5x1
```

Trainer result:
```text
val:init success = 0.000
val:init valid_rate = 0.973
val:init main_oracle = 0.058
val:init sub_oracle = 0.058

train success = 0.400
train valid_rate = 1.000
train main_oracle = 0.250
train sub_oracle = 0.250
updates main/sub = 48 / 48

val success = 0.200
val valid_rate = 0.963
val main_oracle = 0.163
val sub_oracle = 0.163
best checkpoint saved
```

Unified evaluator on structured GRPO best:
```text
success_rate = 0.200
reward = 0.200
avg_steps = 8.400
invalid_action_rate = 0.060
```

Interpretation:
```text
This is the first setting where structured M-GRPO improves over its own sampled
initial validation inside the trainer and saves a new best checkpoint.

But under the deterministic external evaluator, structured GRPO best is still
tied with structured SFT50 and below old action-only SFT50. The structured
interface is promising for validity and credit assignment, but it is not yet
strong enough on success.

The next practical step is not another 5x1 GRPO run. It is to scale structured
SFT data first, because Main/Sub need more exposure to the new communication
protocol before RL can reliably improve it.
```

## Structured Short-History SFT200

Changed:
```text
generate_plancraft_mas_sft_data.py
```

Added:
```text
--history-steps N
```

Reason:
```text
Structured Sub outputs are longer than action-only Sub outputs. With long
history, SFT samples can be truncated so far that no assistant target tokens
remain. The generator now keeps only the most recent N history steps, defaulting
to 3.
```

Data:
```text
python generate_plancraft_mas_sft_data.py ^
  --split train ^
  --limit 200 ^
  --max-steps 30 ^
  --history-steps 3 ^
  --structured-sub ^
  --output .\plancraft_mas_structured_short_sft_data_200.jsonl
```

Generated:
```text
2860 samples
main = 1430
sub = 1430
```

Prepared-sample check:
```text
max_length = 1536 -> main 1383 / 1430, sub 1387 / 1430
max_length = 2048 -> main 1430 / 1430, sub 1430 / 1430
max_length = 3072 -> main 1430 / 1430, sub 1430 / 1430
```

Training:
```text
base_model = Qwen2.5-1.5B-Instruct local
init main = plancraft_mas_structured_sft_50x1/main_agent
init sub  = plancraft_mas_structured_sft_50x1/sub_agent
data      = plancraft_mas_structured_short_sft_data_200.jsonl
epochs    = 1
lr        = 8e-5
max_len   = 2048
save_dir  = plancraft_mas_structured_short_sft_200x1
```

Training result:
```text
Main prepared samples = 1430 / 1430
Main loss = 0.0000
Sub prepared samples = 1430 / 1430
Sub loss = 0.0221
```

Easy5 evaluation:
```text
split = val.small.easy
tasks = 5
max_steps = 10
max_tokens = 120
structured_sub = true
```

Result:
```text
success_rate = 0.400
reward = 0.400
avg_steps = 4.600
invalid_action_rate = 0.070
```

Comparison:
```text
Old action-only SFT50:
  success_rate = 0.400
  avg_steps = 6.800
  invalid_action_rate = 0.280

Structured SFT50:
  success_rate = 0.200
  avg_steps = 8.400
  invalid_action_rate = 0.040

Structured short-history SFT200:
  success_rate = 0.400
  avg_steps = 4.600
  invalid_action_rate = 0.070
```

Easy20 evaluation:
```text
success_rate = 0.500
reward = 0.500
efficiency = 0.212
avg_steps = 5.050
invalid_action_rate = 0.077
```

Interpretation:
```text
This is the best Plancraft structured checkpoint so far.

The structured interface has recovered action-only SFT50's easy5 success rate
while using fewer steps and far fewer invalid actions. On a broader easy20
slice, it solves half the tasks with low invalid-action rate.

The main fix was not RL; it was giving structured SFT enough data and preserving
all assistant targets with short history + max_length 2048.
```

Structured SFT200 -> M-GRPO 5x1:
```text
init = plancraft_mas_structured_short_sft_200x1
train = 5
val = 5
iterations = 1
group_size = 2
lr = 5e-8
advantage_clip = 0.5
eval_samples = 2
structured_sub = true
```

Result:
```text
val:init success = 0.500
val:init best_success = 0.600
val:init valid_rate = 1.000
val:init main_oracle = 0.487
val:init sub_oracle = 0.487

train success = 0.600
train valid_rate = 0.971
train main_oracle = 0.467
train sub_oracle = 0.400
updates main/sub = 29 / 29

val success = 0.400
val best_success = 0.400
val valid_rate = 1.000
val main_oracle = 0.379
val sub_oracle = 0.379
```

Interpretation:
```text
With a stronger structured SFT base, GRPO still performs real updates but hurts
held-out validation on this small run. That reinforces the current bottleneck:
the action-level reward split is not enough. The next GRPO improvement should
use state-progress reward, not more tiny action-match tuning.
```

## State-Progress Reward for Plancraft GRPO

Changed:
```text
grpo_plancraft_mas.py
```

Added state-progress reward:
```text
oracle_steps_before = len(official_oracle_subplans(current_state))
execute Main action
oracle_steps_after = len(official_oracle_subplans(next_state))
oracle_progress = clamp((before - after) / before, -1.0, 1.0)
```

Reward now includes:
```text
Main reward =
  success
+ valid action reward
+ oracle exact action match
+ main_progress_weight * oracle_progress
- step penalty

Sub reward =
  partial global success
+ valid structured action reward
+ oracle exact action match
+ sub_progress_weight * oracle_progress
+ sub/main agreement
- step penalty
```

Why this matters:
```text
Exact oracle action match is too brittle. In Plancraft, several actions can be
equivalent or at least non-harmful. The new progress term rewards actions that
reduce the remaining official plan length, so it is closer to state-level credit
assignment than pure action imitation.
```

Smoke:
```text
init = plancraft_mas_structured_short_sft_200x1
train = 1
val = 1
group_size = 2
max_steps = 5
eval_samples = 2
structured_sub = true
```

Result:
```text
val:init success = 0.500
val:init best_success = 1.000
val:init progress = 0.396

train success = 0.000
train progress = 0.327
updates main/sub = 9 / 9

val success = 1.000
val progress = 0.750
```

Interpretation:
```text
The new reward path works. Correct validation trajectories have much higher
progress, so the metric is meaningful.
```

Progress-GRPO 5x1:
```text
init = plancraft_mas_structured_short_sft_200x1
train = 5
val = 5
iterations = 1
group_size = 2
lr = 5e-8
advantage_clip = 0.5
eval_samples = 2
structured_sub = true
```

Result:
```text
val:init success = 0.500
val:init best_success = 0.600
val:init progress = 0.671

train success = 0.600
train valid_rate = 1.000
train progress = 0.800
updates main/sub = 23 / 23

val success = 0.300
val best_success = 0.400
val progress = 0.442
```

Interpretation:
```text
State-progress reward makes the training rollouts look genuinely better:
validity is perfect, success is high, and oracle progress is high.

But held-out validation still drops after a small on-policy update. This means
the reward is more informative, but the current GRPO setting is still too
high-variance/overfit for 5 training tasks.

The next RL-side experiment should reduce update aggressiveness and improve
selection stability:
  - larger validation slice,
  - group_size >= 4,
  - more train tasks,
  - lower or zero update when candidate advantages come from a tiny reward gap,
  - optional SFT replay mixed into GRPO updates.

Current best checkpoint remains:
  plancraft_mas_structured_short_sft_200x1
```

## Formal Progress-GRPO 20x1 Group-4 Experiment

Changed:
```text
grpo_plancraft_mas.py
```

Added:
```text
--reward-gap-threshold
```

Behavior:
```text
For Main and Sub separately:
  reward_gap = max(group rewards) - min(group rewards)

If reward_gap < threshold:
  set all advantages for that agent to zero
  skip its update for that group
```

This prevents GRPO from amplifying nearly identical candidate rewards into large
normalized advantages.

Experiment:
```text
init = plancraft_mas_structured_short_sft_200x1
train_tasks = 20
val_tasks = 20
iterations = 1
group_size = 4
max_steps = 8
max_response_len = 120
lr = 3e-8
advantage_clip = 0.5
reward_gap_threshold = 0.02
eval_samples = 2
structured_sub = true
save_dir = plancraft_mas_grpo_progress_20x1_g4
```

Trainer result:
```text
val:init success = 0.200
val:init best_success = 0.300
val:init progress = 0.420

train success = 0.400
train valid_rate = 0.972
train progress = 0.693
updates main/sub = 407 / 400

val success = 0.150
val best_success = 0.250
val progress = 0.458
```

Because trainer evaluation samples at temperature 0.8 and max_steps 8, its
absolute scores are not directly comparable to the external evaluator used for
the SFT200 report. The saved step checkpoint was therefore evaluated with the
same external protocol:

```text
split = val.small.easy
tasks = 20
max_steps = 10
max_tokens = 120
structured_sub = true
```

Fair comparison:
```text
Structured SFT200:
  success_rate = 0.500
  efficiency = 0.212
  avg_steps = 5.050
  invalid_action_rate = 0.077

Progress-GRPO step_1:
  success_rate = 0.450
  efficiency = 0.200
  avg_steps = 5.250
  invalid_action_rate = 0.083
```

Interpretation:
```text
Increasing the rollout scale from 5 tasks/group 2 to 20 tasks/group 4 makes the
GRPO result much closer to the SFT baseline. However, it still does not improve
held-out success, efficiency, or validity.

The result does not support "RL cannot improve the model." It shows that one
iteration over 80 sampled training trajectories is still insufficient, and that
the current per-step optimizer update may be too aggressive: about 400 updates
were applied from only 20 training tasks.

The next controlled experiment should change update mechanics before simply
adding iterations:
  - accumulate candidate losses before optimizer.step(),
  - average updates per group/episode,
  - or mix SFT replay with GRPO.

Current selected checkpoint remains the structured SFT200 initialization.
```

## Group-Batched GRPO with SFT Replay

Changed:
```text
grpo_v4.py
grpo_plancraft_mas.py
```

Previous update behavior:
```text
For every candidate trajectory step:
  backward
  clip gradients
  optimizer.step

The 20-task/group-4 experiment applied about 400 optimizer steps per adapter.
```

New update behavior:
```text
For each rollout group and adapter:
  collect all eligible trajectory steps
  normalize rollout weights by the number of collected steps
  backward all weighted losses
  optionally backward an averaged SFT replay loss
  clip gradients once
  optimizer.step once
```

New CLI options:
```text
--max-train-length
--sft-replay-path
--sft-replay-per-group
--sft-replay-weight
```

Smoke:
```text
train = 1
val = 1
group_size = 2
sft_replay_per_group = 1
sft_replay_weight = 0.1
```

Result:
```text
replay samples loaded:
  main = 1430
  sub = 1430

optimizer steps:
  main = 1
  sub = 1
```

Formal comparison:
```text
init = plancraft_mas_structured_short_sft_200x1
train_tasks = 20
val_tasks = 20
iterations = 1
group_size = 4
lr = 3e-8
advantage_clip = 0.5
reward_gap_threshold = 0.02
sft_replay_per_group = 1
sft_replay_weight = 0.1
max_train_length = 2048
structured_sub = true
save_dir = plancraft_mas_grpo_batch_replay_20x1_g4
```

Trainer result:
```text
val:init success = 0.200
val:init best_success = 0.300

train success = 0.450
train valid_rate = 0.988
train progress = 0.742
optimizer steps main/sub = 20 / 20

val success = 0.125
val best_success = 0.250
```

As before, the trainer uses high-temperature rollout sampling, so final
comparison uses the common external easy20 evaluator.

External easy20:
```text
Structured SFT200:
  success_rate = 0.500
  efficiency = 0.212
  avg_steps = 5.050
  invalid_action_rate = 0.077

Previous progress-GRPO with per-step optimizer updates:
  success_rate = 0.450
  efficiency = 0.200
  avg_steps = 5.250
  invalid_action_rate = 0.083

Group-batched progress-GRPO + SFT replay:
  success_rate = 0.550
  efficiency = 0.225
  avg_steps = 4.800
  invalid_action_rate = 0.034
```

Interpretation:
```text
This is the first Plancraft GRPO checkpoint that improves over the structured
SFT baseline under the same external evaluation protocol.

The result supports the hypothesis that the previous bottleneck was update
mechanics, not only reward design:
  - per-step optimizer updates overfit and destabilized the policy,
  - group-level gradient accumulation made updates much less aggressive,
  - light SFT replay protected the learned structured communication/action
    protocol.

The gain is still preliminary: it is +1 solved task on a 20-task slice and one
random seed. It should be validated on more tasks/seeds before claiming a robust
RL improvement.

Current experimental best checkpoint:
  plancraft_mas_grpo_batch_replay_20x1_g4/main_step_1
  plancraft_mas_grpo_batch_replay_20x1_g4/sub_step_1
```

## Full Easy100 SFT vs GRPO Validation

To test whether the easy20 improvement was robust, both checkpoints were
evaluated on the complete 100-task `val.small.easy` split with identical
settings:

```text
tasks = 100
max_steps = 10
max_tokens = 120
structured_sub = true
seed = 123
```

Results:
```text
Structured SFT200:
  success_rate = 0.400
  solved = 40 / 100
  efficiency = 0.182
  avg_steps = 5.570
  invalid_action_rate = 0.097

Group-batched GRPO + SFT replay:
  success_rate = 0.380
  solved = 38 / 100
  efficiency = 0.169
  avg_steps = 5.680
  invalid_action_rate = 0.078
```

Paired task analysis:
```text
solved by both = 31
solved by neither = 53
GRPO-only solves = 7
SFT-only solves = 9
success-rate difference = -0.020
paired bootstrap 95% CI = [-0.100, +0.060]
exact McNemar p = 0.804
```

Interpretation:
```text
The easy20 +0.05 result does not reproduce on the complete easy100 split.
Therefore, the current evidence does not prove that GRPO improves task success
over SFT.

GRPO does consistently improve action validity:
  invalid_action_rate falls from 0.097 to 0.078.

But that validity improvement does not yet translate into more completed tasks.
The 100-task result suggests that the current RL update changes which tasks are
solved rather than improving overall capability: it gains 7 tasks but loses 9.

The correct conclusion is:
  - batch updates and SFT replay are more stable than per-step updates,
  - GRPO improves output/action discipline,
  - success-rate improvement remains unproven.

The next experiment should train on substantially more RL tasks, rather than
only evaluate on more tasks. A suitable next scale is 50-100 train tasks,
group_size 4, with easy100 held out for final evaluation.
```

## Independent 50-Task Batch GRPO Training

Changed:
```text
grpo_plancraft_mas.py
```

Added:
```text
--train-offset
--val-offset
```

The structured SFT200 data was generated from the first 200 train examples.
This experiment therefore used 50 later examples beginning at train offset 200,
avoiding direct reuse of the SFT task slice.

Training:
```text
init = plancraft_mas_structured_short_sft_200x1
train_split = train
train_offset = 200
train_tasks = 50
val_tasks = 20
iterations = 1
group_size = 4
sampled training trajectories = 200
lr = 3e-8
advantage_clip = 0.5
reward_gap_threshold = 0.02
sft_replay_per_group = 1
sft_replay_weight = 0.1
group-batched optimizer updates = true
structured_sub = true
save_dir = plancraft_mas_grpo_batch_replay_50x1_g4_offset200
```

Training result:
```text
val:init success = 0.200
val:init best_success = 0.300

train success = 0.220
train valid_rate = 0.992
train progress = 0.688
optimizer steps main/sub = 50 / 50

val success = 0.225
val best_success = 0.350
val progress = 0.447
```

The internal high-temperature validation improved and saved a new best
checkpoint.

Final external easy100:
```text
Structured SFT200:
  success_rate = 0.400
  solved = 40 / 100
  efficiency = 0.182
  avg_steps = 5.570
  invalid_action_rate = 0.097

50-task batch GRPO best:
  success_rate = 0.380
  solved = 38 / 100
  efficiency = 0.162
  avg_steps = 5.850
  invalid_action_rate = 0.092
```

Paired analysis:
```text
solved by both = 31
solved by neither = 53
GRPO-only solves = 7
SFT-only solves = 9
success-rate difference = -0.020
paired bootstrap 95% CI = [-0.100, +0.060]
exact McNemar p = 0.804
```

Interpretation:
```text
Scaling RL training from 20 to 50 independent tasks does not produce a
measurable success-rate improvement on easy100.

This rules out the simplest explanation that the negative result was only due
to using five or twenty RL tasks. The current GRPO reward and trajectory update
still improve discipline/validity more reliably than task-solving capability.

The internal high-temperature validation selected a checkpoint that did not
improve the external evaluation. Checkpoint selection and training objective are
therefore misaligned with the final metric.

Before spending substantially more compute on 100+ tasks or multiple
iterations, the next priority should be:
  - deterministic/common-protocol validation during training,
  - selecting checkpoints by external-style success rather than sampled
    best-of-N,
  - and diagnosing the seven gained versus nine lost tasks to find which
    behaviors the reward promotes or suppresses.
```

## Aligned Low-Temperature Validation

Changed:
```text
grpo_v4.py
grpo_plancraft_mas.py
```

Training rollout and validation generation are now separated:
```text
rollout:
  temperature = 0.8
  top_p = 0.95
  max_steps = 8

validation:
  temperature = 0.2
  top_p = 0.9
  repetition_penalty = 1.05
  max_steps = 10
  eval_samples = 1
  fixed eval seed = 123
  checkpoint metric = success_rate
```

New CLI options:
```text
--rollout-temperature
--eval-temperature
--eval-top-p
--eval-repetition-penalty
--eval-max-steps
--eval-seed
```

Alignment sanity check:
```text
checkpoint = structured SFT200
val.small.easy tasks = 20
iterations = 0
```

Result:
```text
aligned trainer validation success = 0.550
```

This is much closer to the external evaluator than the old high-temperature
trainer result of 0.200.

The independent 50-task GRPO experiment was then repeated with aligned
checkpoint selection:

```text
train_offset = 200
train_tasks = 50
group_size = 4
sampled rollouts = 200
batch updates + SFT replay
best metric = low-temperature single-sample success_rate
```

Result:
```text
val:init success = 0.550

train success = 0.200
train valid_rate = 0.997
train progress = 0.593
optimizer steps main/sub = 50 / 50

val after RL success = 0.250
val after RL invalid_rate = 0.073
val after RL progress = 0.252
```

Interpretation:
```text
Aligned validation fixes checkpoint selection: the degraded RL checkpoint is
rejected, and best/ remains the SFT200 initialization.

It also exposes the deeper issue more clearly. The current one-iteration GRPO
update substantially damages low-temperature held-out performance, even though
training validity remains high.

Therefore:
  - the previous high-temperature best-of-N selector was indeed misleading,
  - but checkpoint selection was not the only problem,
  - the current GRPO objective/update still overfits rollout behavior and hurts
    generalization.

The next experiment should not increase the number of tasks again with the same
objective. It should reduce the effective RL update strength, for example:
  - lower LR by 3-10x,
  - reduce advantage_clip,
  - increase SFT replay weight/count,
  - or update Main only while freezing Sub as a controlled ablation.
```

## Low-Strength Main-Only vs Joint GRPO Ablation

Changed:
```text
grpo_plancraft_mas.py
```

Added:
```text
--train-main / --no-train-main
--train-sub / --no-train-sub
```

A frozen adapter still participates in rollout generation, but receives neither
GRPO loss nor SFT replay updates.

Common settings:
```text
init = structured SFT200
train_offset = 200
train_tasks = 20
val_tasks = 20
iterations = 1
group_size = 4
lr = 3e-9
advantage_clip = 0.2
reward_gap_threshold = 0.02
sft_replay_per_group = 4
sft_replay_weight = 0.3
aligned low-temperature validation = true
```

Main-only:
```text
train_main = true
train_sub = false

val:init success = 0.550
train success = 0.350
optimizer steps main/sub = 20 / 0
val success after update = 0.250
val invalid_action_rate = 0.062
```

Joint:
```text
train_main = true
train_sub = true

val:init success = 0.550
train success = 0.300
optimizer steps main/sub = 20 / 20
val success after update = 0.250
val invalid_action_rate = 0.056
```

Interpretation:
```text
Main-only and joint training produce essentially the same held-out degradation.
Freezing Sub does not protect validation success.

This localizes the primary failure to the Main policy update rather than to
joint Main/Sub credit assignment. Even with:
  - 10x lower learning rate,
  - smaller advantage clipping,
  - four replay samples per group,
  - and stronger replay weighting,

the Main update moves the policy away from the strong SFT solution.

Both runs are rejected by aligned validation, so best/ remains the SFT200
initialization. Running easy100 on the rejected step checkpoints is unnecessary
for checkpoint selection.

The next technical priority is to replace the weighted-SFT surrogate with a
proper policy-ratio GRPO objective using stored rollout log probabilities and a
reference/old policy. The current negative-weight cross-entropy update is only a
rough approximation and is now the most likely algorithmic bottleneck.
```
