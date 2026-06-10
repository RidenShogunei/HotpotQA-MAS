# HotpotQA Dynamic MAS

Standalone project for hierarchical Main/Sub training on HotpotQA.

```text
Question + document catalog
        -> Main chooses direct or delegates 1..N subtasks
        -> shared Sub policy searches, reads, and summarizes
        -> Main synthesizes the final answer and evidence
```

## Structure

```text
data/base/       10-document local benchmark
data/enhanced/   30-document benchmark
data/sft/        Current staged SFT datasets
docs/            Historical experiment reports
artifacts/       Checkpoints and evaluation output, ignored by Git
```

The supported workflow has five entry points:

| Task | Command |
|---|---|
| Prepare data | `prepare_hotpotqa_data.py` |
| Generate SFT | `generate_hotpotqa_dynamic_mixture_sft_data.py` |
| Train SFT | `sft_trainer.py` |
| Train GRPO | `grpo_hotpotqa_mas.py` |
| Evaluate | `analyze_hotpotqa_dynamic_mas_results.py` |

`analyze_hotpotqa_dynamic_failures.py` is the only additional diagnostic
entry point.

The other Python files are imported protocol/model helpers rather than
separate workflows.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The default model is `Qwen/Qwen3.5-9B`. Pass `--base-model` to use a local
model directory or another compatible causal LM.

## Prepare Data

Both benchmark variants use the same command:

```powershell
python prepare_hotpotqa_data.py --mode base
python prepare_hotpotqa_data.py --mode enhanced `
  --train-size 500 --val-size 150 --docs-per-task 30
```

Prepared splits are already included under `data/`.

## Generate SFT

One generator supports the three current training stages:

```powershell
# Joint routing/research/synthesis mixture
python generate_hotpotqa_dynamic_mixture_sft_data.py `
  --stage mixture `
  --train-jsonl .\data\enhanced\train.jsonl `
  --output .\data\sft\dynamic_mixture.jsonl `
  --limit 300 --max-subtasks 2

# Main-only evidence synthesis
python generate_hotpotqa_dynamic_mixture_sft_data.py `
  --stage synthesis `
  --output .\data\sft\dynamic_synthesis.jsonl `
  --limit 500

# Main-only noisy-result verification
python generate_hotpotqa_dynamic_mixture_sft_data.py `
  --stage verifier `
  --output .\data\sft\dynamic_verifier.jsonl `
  --limit 500 --samples-per-task 3
```

The repository tracks the corresponding experiment datasets:

```text
data/sft/hotpotqa_dynamic_mixture_sft_data_300_v3.jsonl
data/sft/hotpotqa_dynamic_synthesis_sft_data_500.jsonl
data/sft/hotpotqa_dynamic_verifier_sft_data_500.jsonl
```

## Train SFT

```powershell
python sft_trainer.py `
  --data-path .\data\sft\hotpotqa_dynamic_mixture_sft_data_300_v3.jsonl `
  --save-dir .\artifacts\checkpoints\dynamic_sft `
  --epochs 1 --max-length 1536
```

Use `--no-train-main`, `--no-train-sub`, `--main-lora`, and `--sub-lora` for
staged continuation or ablation.

## Evaluate

```powershell
python analyze_hotpotqa_dynamic_mas_results.py `
  --main-lora .\artifacts\checkpoints\dynamic_sft\main_agent `
  --sub-lora .\artifacts\checkpoints\dynamic_sft\sub_agent `
  --val-jsonl .\data\enhanced\val.jsonl `
  --tasks 20 --samples 2 --max-subagents 2
```

Metrics include answer F1, evidence accuracy, total reward, valid tool rate,
direct rate, and average delegated subtask count.

## Train GRPO

```powershell
python grpo_hotpotqa_mas.py `
  --main-lora .\artifacts\checkpoints\dynamic_sft\main_agent `
  --sub-lora .\artifacts\checkpoints\dynamic_sft\sub_agent `
  --train-jsonl .\data\enhanced\train.jsonl `
  --val-jsonl .\data\enhanced\val.jsonl `
  --save-dir .\artifacts\checkpoints\dynamic_grpo
```

Run Main-only and Sub-only ablations before joint training. Do not select a
checkpoint using training reward alone.

Important: the current GRPO trainer still uses the older fixed
`Main -> one Sub -> Main answer` rollout. Dynamic `direct/delegate 1..N`
behavior is currently implemented in SFT generation and evaluation, but not
yet in the GRPO rollout. Porting GRPO to that dynamic protocol is the next
architecture task.

## Current Findings

- Harder 30-document contexts make Sub research useful.
- Dynamic routing learns focused subtasks but tends to over-delegate.
- Main synthesis remains the largest bottleneck.
- Joint RL has not yet shown stable held-out improvement over staged SFT.

Detailed experiment history remains in `docs/`.
