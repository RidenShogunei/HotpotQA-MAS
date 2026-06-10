"""Trace dynamic HotpotQA MAS rollouts and decompose failure modes."""

import argparse
import json
import random
import re
from pathlib import Path

import torch

from analyze_hotpotqa_dynamic_mas_results import (
    DYNAMIC_MAIN_ANSWER_SYSTEM,
    DYNAMIC_MAIN_PLAN_SYSTEM,
    doc_catalog,
    extract_mode,
    extract_subtasks,
    run_sub_instance,
    sub_results_text,
)
from analyze_hotpotqa_dynamic_mas_results import build_prompt, generate_one, load_model
from hotpotqa_environment import HotpotQAEnvironment


def doc_ids(text: str):
    return sorted(set(re.findall(r"\bD\d{2}\b", text)))


def result_answer(text: str):
    result = HotpotQAEnvironment.extract_result(text)
    return re.split(r"\|\s*evidence\s*:", result, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def summarize_trace(task, plan_raw, sub_results, answer_raw, reward):
    gold_docs = set(task.support_doc_ids)
    plan_docs = set(doc_ids(plan_raw))
    read_docs = set()
    sub_summary_docs = set()
    sub_answer_scores = []
    duplicate_reads = 0
    total_reads = 0

    for sub_result in sub_results:
        seen_reads = []
        for call in sub_result["tool_calls"]:
            parsed = HotpotQAEnvironment.parse_tool_call(call)
            if parsed and parsed[0] == "read":
                total_reads += 1
                seen_reads.append(parsed[1])
                read_docs.add(parsed[1])
        duplicate_reads += max(len(seen_reads) - len(set(seen_reads)), 0)
        sub_summary_docs.update(doc_ids(sub_result["summary"]))
        sub_answer_scores.append(
            HotpotQAEnvironment.token_f1(result_answer(sub_result["summary"]), task.answer)
        )

    return {
        "task_id": task.task_id,
        "question": task.question,
        "gold_answer": task.answer,
        "gold_docs": task.support_doc_ids,
        "plan_docs": sorted(plan_docs),
        "read_docs": sorted(read_docs),
        "sub_summary_docs": sorted(sub_summary_docs),
        "plan_support_recall": len(plan_docs & gold_docs) / max(len(gold_docs), 1),
        "read_support_recall": len(read_docs & gold_docs) / max(len(gold_docs), 1),
        "sub_summary_evidence_recall": len(sub_summary_docs & gold_docs) / max(len(gold_docs), 1),
        "sub_summary_answer_f1": sum(sub_answer_scores) / max(len(sub_answer_scores), 1),
        "final_answer": result_answer(answer_raw),
        "final_answer_f1": reward["answer_f1"],
        "final_evidence": reward["evidence"],
        "final_reward": reward["total"],
        "duplicate_reads": duplicate_reads,
        "total_reads": total_reads,
        "plan_raw": plan_raw,
        "sub_results": sub_results,
        "answer_raw": answer_raw,
    }


def average(rows, key):
    return sum(row[key] for row in rows) / max(len(rows), 1)


def write_summary(path: Path, rows):
    keys = [
        "plan_support_recall",
        "read_support_recall",
        "sub_summary_evidence_recall",
        "sub_summary_answer_f1",
        "final_answer_f1",
        "final_evidence",
        "final_reward",
        "duplicate_reads",
    ]
    lines = ["# Dynamic Failure Trace Summary", ""]
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    for key in keys:
        lines.append(f"| {key} | {average(rows, key):.3f} |")
    lines.append("")
    lines.append("## Worst Final Rewards")
    lines.append("")
    lines.append("| task_id | final_reward | final_answer_f1 | final_evidence | read_support_recall | final_answer | gold_answer |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for row in sorted(rows, key=lambda item: item["final_reward"])[:10]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task_id"],
                    f"{row['final_reward']:.3f}",
                    f"{row['final_answer_f1']:.3f}",
                    f"{row['final_evidence']:.3f}",
                    f"{row['read_support_recall']:.3f}",
                    row["final_answer"].replace("|", "/"),
                    row["gold_answer"].replace("|", "/"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Trace dynamic MAS failures.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--main-lora", required=True)
    parser.add_argument("--sub-lora", required=True)
    parser.add_argument("--val-jsonl", default="./data/enhanced/val.jsonl")
    parser.add_argument("--out-dir", default="./artifacts/eval/dynamic_failures")
    parser.add_argument("--offset", type=int, default=40)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--sub-steps", type=int, default=3)
    parser.add_argument("--max-subagents", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = HotpotQAEnvironment.from_jsonl(args.val_jsonl, limit=args.offset + args.tasks)
    tasks = env.tasks[args.offset : args.offset + args.tasks]
    model, tokenizer = load_model(args.base_model, args.main_lora, args.sub_lora, device)
    rows = []

    for task in tasks:
        for sample_idx in range(args.samples):
            plan_prompt = build_prompt(
                tokenizer,
                DYNAMIC_MAIN_PLAN_SYSTEM,
                f"Question: {task.question}\nAvailable documents:\n{doc_catalog(task)}",
            )
            plan_raw = generate_one(model, tokenizer, "main", plan_prompt, device, args.max_tokens)
            mode = extract_mode(plan_raw)
            subtasks = extract_subtasks(plan_raw, args.max_subagents)
            if mode == "delegate" and not subtasks:
                subtasks = [f"Find the supporting documents and answer for: {task.question}"]

            sub_results = []
            if mode != "direct":
                sub_results = [
                    run_sub_instance(model, tokenizer, task, device, subtask, args.max_tokens, args.sub_steps)
                    for subtask in subtasks
                ]

            answer_prompt = build_prompt(
                tokenizer,
                DYNAMIC_MAIN_ANSWER_SYSTEM,
                f"Question: {task.question}\nSub results:\n{sub_results_text(sub_results)}",
            )
            answer_raw = generate_one(model, tokenizer, "main", answer_prompt, device, args.max_tokens)
            combined = plan_raw + "".join("".join(r["tool_calls"]) + r["summary"] for r in sub_results) + answer_raw
            reward = HotpotQAEnvironment.reward(task, combined)
            row = summarize_trace(task, plan_raw, sub_results, answer_raw, reward)
            row["sample_idx"] = sample_idx
            rows.append(row)
            print(
                f"[trace] task={task.task_id} sample={sample_idx} "
                f"reward={row['final_reward']:.3f} answer_f1={row['final_answer_f1']:.3f} "
                f"read_recall={row['read_support_recall']:.3f}",
                flush=True,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "trace.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_summary(out_dir / "summary.md", rows)
    print(f"[trace] wrote {out_dir / 'trace.jsonl'}", flush=True)
    print(f"[trace] wrote {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()

