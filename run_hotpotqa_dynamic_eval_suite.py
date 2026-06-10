"""Run multi-slice dynamic HotpotQA MAS evaluation."""

import argparse
import json
from pathlib import Path

import torch

import analyze_hotpotqa_dynamic_mas_results as dynamic_eval
from hotpotqa_environment import HotpotQAEnvironment


def write_jsonl(path: Path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_tasks(path: str, offset: int, tasks: int):
    env = HotpotQAEnvironment.from_jsonl(path, limit=offset + tasks)
    return env.tasks[offset : offset + tasks]


def avg(rows, key):
    return sum(row[key] for row in rows) / max(len(rows), 1)


def weighted_avg(rows, key):
    weight = sum(row["tasks"] for row in rows)
    return sum(row[key] * row["tasks"] for row in rows) / max(weight, 1)


def write_markdown(path: Path, rows):
    metric_keys = [
        "direct_rate",
        "avg_subtasks",
        "answer_f1",
        "evidence",
        "reward",
        "best_answer_f1",
        "best_reward",
        "tool_valid",
    ]
    lines = ["# Dynamic HotpotQA MAS Evaluation Suite", ""]
    lines.append("## Per Slice")
    lines.append("")
    header = ["model", "offset", "tasks", "samples"] + metric_keys
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        values = [row["model"], str(row["offset"]), str(row["tasks"]), str(row["samples"])]
        values.extend(f"{row[key]:.3f}" for key in metric_keys)
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Averages")
    lines.append("")
    lines.append("| model | " + " | ".join(metric_keys) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(metric_keys)) + "|")
    for model in sorted(set(row["model"] for row in rows)):
        model_rows = [row for row in rows if row["model"] == model]
        values = [model]
        values.extend(f"{avg(model_rows, key):.3f}" for key in metric_keys)
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Task-Weighted Averages")
    lines.append("")
    lines.append("| model | " + " | ".join(metric_keys) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(metric_keys)) + "|")
    for model in sorted(set(row["model"] for row in rows)):
        model_rows = [row for row in rows if row["model"] == model]
        values = [model]
        values.extend(f"{weighted_avg(model_rows, key):.3f}" for key in metric_keys)
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run dynamic HotpotQA MAS multi-offset evaluation.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--name", default="dynamic_mas")
    parser.add_argument("--main-lora", required=True)
    parser.add_argument("--sub-lora", required=True)
    parser.add_argument("--val-jsonl", default="./data/enhanced/val.jsonl")
    parser.add_argument("--out-dir", default="./artifacts/eval/dynamic_suite")
    parser.add_argument("--offsets", type=int, nargs="+", default=[0, 20, 40])
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--sub-steps", type=int, default=3)
    parser.add_argument("--max-subagents", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "results.jsonl"
    if out_jsonl.exists():
        out_jsonl.unlink()

    rows = []
    print(f"[suite:dynamic] loading {args.name}", flush=True)
    model, tokenizer = dynamic_eval.load_model(args.base_model, args.main_lora, args.sub_lora, device)
    for offset in args.offsets:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        tasks = load_tasks(args.val_jsonl, offset, args.tasks)
        print(f"[suite:dynamic] model={args.name} offset={offset} tasks={len(tasks)}", flush=True)
        metrics = dynamic_eval.evaluate(
            model,
            tokenizer,
            tasks,
            device,
            args.samples,
            args.max_tokens,
            args.sub_steps,
            args.max_subagents,
        )
        row = {
            "suite": "dynamic_mas",
            "model": args.name,
            "offset": offset,
            "tasks": len(tasks),
            "samples": args.samples,
            **metrics,
        }
        write_jsonl(out_jsonl, row)
        rows.append(row)

    write_markdown(out_dir / "summary.md", rows)
    print(f"[suite:dynamic] wrote {out_jsonl}", flush=True)
    print(f"[suite:dynamic] wrote {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()

