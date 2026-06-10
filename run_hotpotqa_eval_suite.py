"""Run multi-slice HotpotQA MAS/Sub evaluation and save durable reports."""

import argparse
import json
from pathlib import Path

import torch

import analyze_hotpotqa_mas_results as mas_eval
import analyze_hotpotqa_sub_oracle as sub_eval
from hotpotqa_environment import HotpotQAEnvironment


MAS_MODELS = [
    {
        "name": "fixed_staged_best",
        "main_lora": "./hotpotqa_mas_enhanced_mainonly_conservative_50x1/best/main",
        "sub_lora": "./hotpotqa_mas_enhanced_mainonly_conservative_50x1/best/sub",
    },
    {
        "name": "stage2_main_enhanced_sub",
        "main_lora": "./hotpotqa_mas_enhanced_mainonly_conservative_50x1/best/main",
        "sub_lora": "./hotpotqa_sub_pref_enhanced_300x900/sub",
    },
]


SUB_MODELS = [
    {
        "name": "preference_from_sft",
        "sub_lora": "./hotpotqa_sub_pref_100x250/sub",
    },
    {
        "name": "stage2_sub",
        "sub_lora": "./hotpotqa_mas_stage2_main_prefsub_50x2/best/sub",
    },
    {
        "name": "joint_30x1_sub",
        "sub_lora": "./hotpotqa_mas_joint_from_stage2_30x1/best/sub",
    },
]


def write_jsonl(path: Path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_tasks(path: str, offset: int, tasks: int):
    env = HotpotQAEnvironment.from_jsonl(path, limit=offset + tasks)
    return env.tasks[offset:offset + tasks]


def avg(rows, key):
    return sum(row[key] for row in rows) / max(len(rows), 1)


def weighted_avg(rows, key):
    weight = sum(row["tasks"] for row in rows)
    return sum(row[key] * row["tasks"] for row in rows) / max(weight, 1)


def append_table(lines, title, rows, metric_keys):
    lines.append(f"## {title}")
    lines.append("")
    header = ["model", "offset", "tasks", "samples"] + metric_keys
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        values = [
            row["model"],
            str(row["offset"]),
            str(row["tasks"]),
            str(row["samples"]),
        ]
        values.extend(f"{row[key]:.3f}" for key in metric_keys)
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")

    lines.append(f"## {title} Averages")
    lines.append("")
    lines.append("| model | " + " | ".join(metric_keys) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(metric_keys)) + "|")
    for model in sorted(set(row["model"] for row in rows)):
        model_rows = [row for row in rows if row["model"] == model]
        values = [model]
        values.extend(f"{avg(model_rows, key):.3f}" for key in metric_keys)
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")

    lines.append(f"## {title} Task-Weighted Averages")
    lines.append("")
    lines.append("| model | " + " | ".join(metric_keys) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(metric_keys)) + "|")
    for model in sorted(set(row["model"] for row in rows)):
        model_rows = [row for row in rows if row["model"] == model]
        values = [model]
        values.extend(f"{weighted_avg(model_rows, key):.3f}" for key in metric_keys)
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")


def run_mas(args, out_jsonl: Path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = []
    specs = [spec for spec in MAS_MODELS if not args.model_names or spec["name"] in args.model_names]
    for spec in specs:
        print(f"[suite:mas] loading {spec['name']}", flush=True)
        model, tokenizer = mas_eval.load_model(args.base_model, spec["main_lora"], spec["sub_lora"], device)
        for offset in args.offsets:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            tasks = load_tasks(args.val_jsonl, offset, args.tasks)
            print(f"[suite:mas] model={spec['name']} offset={offset} tasks={len(tasks)}", flush=True)
            metrics = mas_eval.evaluate(model, tokenizer, tasks, device, args.samples, args.max_tokens, args.sub_steps)
            row = {
                "suite": "mas",
                "model": spec["name"],
                "offset": offset,
                "tasks": len(tasks),
                "samples": args.samples,
                **metrics,
            }
            write_jsonl(out_jsonl, row)
            rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def run_sub(args, out_jsonl: Path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows = []
    specs = [spec for spec in SUB_MODELS if not args.model_names or spec["name"] in args.model_names]
    for spec in specs:
        print(f"[suite:sub] loading {spec['name']}", flush=True)
        model, tokenizer = sub_eval.load_model(args.base_model, spec["sub_lora"], device)
        for offset in args.offsets:
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            tasks = load_tasks(args.val_jsonl, offset, args.tasks)
            print(f"[suite:sub] model={spec['name']} offset={offset} tasks={len(tasks)}", flush=True)
            metrics = sub_eval.evaluate(
                model,
                tokenizer,
                tasks,
                device,
                args.samples,
                args.max_tokens,
                args.sub_steps,
                args.temperature,
            )
            row = {
                "suite": "sub_oracle",
                "model": spec["name"],
                "offset": offset,
                "tasks": len(tasks),
                "samples": args.samples,
                **metrics,
            }
            write_jsonl(out_jsonl, row)
            rows.append(row)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


def write_markdown(path: Path, mas_rows, sub_rows):
    lines = ["# HotpotQA MAS Evaluation Suite", ""]
    if mas_rows:
        append_table(
            lines,
            "Full MAS",
            mas_rows,
            ["answer_f1", "evidence", "reward", "best_answer_f1", "best_reward", "tool_valid"],
        )
    if sub_rows:
        append_table(
            lines,
            "Sub Oracle",
            sub_rows,
            [
                "support_read_recall",
                "answer_f1",
                "evidence",
                "reward",
                "best_support_read_recall",
                "best_answer_f1",
                "best_reward",
            ],
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Run HotpotQA multi-slice evaluation suite.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--val-jsonl", default="./data/base/val.jsonl")
    parser.add_argument("--out-dir", default="./artifacts/eval/mas_suite")
    parser.add_argument("--suite", choices=["all", "mas", "sub"], default="all")
    parser.add_argument("--model-names", nargs="*", default=[])
    parser.add_argument("--offsets", type=int, nargs="+", default=[0, 20, 40])
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--sub-steps", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.4)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "results.jsonl"
    if out_jsonl.exists():
        out_jsonl.unlink()

    mas_rows = run_mas(args, out_jsonl) if args.suite in ("all", "mas") else []
    sub_rows = run_sub(args, out_jsonl) if args.suite in ("all", "sub") else []
    write_markdown(out_dir / "summary.md", mas_rows, sub_rows)
    print(f"[suite] wrote {out_jsonl}", flush=True)
    print(f"[suite] wrote {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()

