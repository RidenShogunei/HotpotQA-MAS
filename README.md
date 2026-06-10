# HotpotQA Multi-Agent RL

Standalone research project for training and evaluating Main/Sub agent systems
on local HotpotQA environments.

This repository was extracted from a larger multi-benchmark workspace. It has
no runtime dependency on that repository and contains only the HotpotQA line.

## Research Architecture

The current target is a dynamic hierarchical MAS:

```text
Question + document catalog
        |
        v
Main coordinator
  |-- direct answer
  `-- delegate 1..N focused research subtasks
              |
              v
        shared Sub policy
        search -> read -> summarize
              |
              v
Main synthesis -> final answer + evidence
```

Main and Sub use a shared base model with separate LoRA adapters. Multiple Sub
instances share the same Sub adapter.

## Repository Layout

```text
data/base/       Original local HotpotQA benchmark
data/enhanced/   Harder 30-document benchmark
data/sft/        Current dynamic MAS SFT datasets
docs/            Experiment reports
artifacts/       Local checkpoints and evaluation output (ignored by Git)
```

Current training path:

```text
hotpotqa_environment.py
generate_hotpotqa_mas_sft_data.py
generate_hotpotqa_dynamic_mas_sft_data.py
generate_hotpotqa_dynamic_mixture_sft_data.py
generate_hotpotqa_dynamic_synthesis_sft_data.py
generate_hotpotqa_dynamic_verifier_sft_data.py
sft_trainer.py
grpo_hotpotqa_mas.py
analyze_hotpotqa_dynamic_mas_results.py
run_hotpotqa_dynamic_eval_suite.py
```

Baseline and diagnostic scripts are retained separately:

```text
grpo_hotpotqa.py
analyze_hotpotqa_results.py
analyze_hotpotqa_mas_results.py
analyze_hotpotqa_sub_oracle.py
analyze_hotpotqa_main_oracle_answer.py
analyze_hotpotqa_dynamic_failures.py
train_hotpotqa_sub_preferences.py
```

## Setup

Python 3.10+ and a CUDA-capable PyTorch installation are recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The default model identifier is `Qwen/Qwen3.5-9B`. Every training and
evaluation script accepts `--base-model`, so a local model directory or
another compatible causal LM can be used.

## Data

The repository includes prepared benchmark splits:

```text
data/base/train.jsonl
data/base/val.jsonl
data/enhanced/train.jsonl
data/enhanced/val.jsonl
```

The tracked SFT datasets are the current staged dynamic path:

```text
data/sft/hotpotqa_dynamic_mixture_sft_data_300_v3.jsonl
data/sft/hotpotqa_dynamic_synthesis_sft_data_500.jsonl
data/sft/hotpotqa_dynamic_verifier_sft_data_500.jsonl
```

Older direct/fixed/dynamic-v1 SFT files are intentionally not tracked. They can
be regenerated with the corresponding `generate_*.py` scripts when a historical
baseline needs to be reproduced.

To regenerate them:

```powershell
python prepare_hotpotqa_data.py --output-dir .\data\base
python prepare_hotpotqa_enhanced_data.py --output-dir .\data\enhanced
```

## Dynamic MAS SFT

Generate mixed routing, Sub research, and Main synthesis supervision:

```powershell
python generate_hotpotqa_dynamic_mixture_sft_data.py `
  --train-jsonl .\data\enhanced\train.jsonl `
  --output .\data\sft\dynamic_mixture.jsonl `
  --limit 300 `
  --max-subtasks 2
```

Train both adapters:

```powershell
python sft_trainer.py `
  --data-path .\data\sft\dynamic_mixture.jsonl `
  --save-dir .\artifacts\checkpoints\dynamic_sft `
  --base-model Qwen/Qwen3.5-9B `
  --epochs 1 `
  --max-length 1536
```

## Evaluation

```powershell
python analyze_hotpotqa_dynamic_mas_results.py `
  --base-model Qwen/Qwen3.5-9B `
  --main-lora .\artifacts\checkpoints\dynamic_sft\main_agent `
  --sub-lora .\artifacts\checkpoints\dynamic_sft\sub_agent `
  --val-jsonl .\data\enhanced\val.jsonl `
  --tasks 20 `
  --samples 2 `
  --max-subagents 2
```

Primary metrics are answer F1, evidence accuracy, total reward, valid tool
rate, direct rate, and average delegated subtask count.

## GRPO

The local shared-model implementation is:

```text
grpo_hotpotqa_mas.py
```

`grpo_hotpotqa.py` is retained only as the direct/single-trajectory baseline.
The former TRL single-policy prototype was removed because it did not implement
the Main/Sub architecture and duplicated the direct baseline.

Do not start joint GRPO from an unstable SFT checkpoint. The recommended
experimental order is:

```text
dynamic SFT
-> held-out routing/Sub/synthesis evaluation
-> Main-only or Sub-only RL ablation
-> joint RL
```

## Current Findings

- Fixed MAS can outperform direct Main in the enhanced environment when the
  Sub retrieval policy is strong.
- Dynamic routing can learn to emit multiple focused subtasks.
- Existing dynamic checkpoints tend to over-delegate and Main synthesis
  remains a bottleneck.
- Historical reward-filtered updates are not equivalent to strict GRPO.

See [docs/ENHANCED_HOTPOTQA_EVAL_REPORT.md](docs/ENHANCED_HOTPOTQA_EVAL_REPORT.md)
and [docs/HOTPOTQA_MGRPO_REPORT.md](docs/HOTPOTQA_MGRPO_REPORT.md) for details.
