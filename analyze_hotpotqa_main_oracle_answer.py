"""Evaluate MAS Main answerers with oracle Sub results."""

import argparse
import json
import random
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_hotpotqa_mas_sft_data import MAIN_ANSWER_SYSTEM, oracle_subtask
from hotpotqa_environment import HotpotQAEnvironment


def load_model(base_model: str, main_lora: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, main_lora, adapter_name="main")
    model.set_adapter("main")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, task):
    evidence = ", ".join(task.support_doc_ids)
    sub_result = f"<result>{task.answer} | evidence: {evidence}</result>"
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": MAIN_ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {task.question}\n"
                    f"Subtask: {oracle_subtask(task)}\n"
                    f"Sub result: {sub_result}"
                ),
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def truncate_result(text: str) -> str:
    end = text.find("</result>")
    if end >= 0:
        return text[:end + len("</result>")]
    return text


def generate_one(model, tokenizer, prompt: str, device: str, max_tokens: int, temperature: float):
    prefix = "<thinking>"
    inputs = tokenizer(prompt + prefix, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return truncate_result(prefix + generated).strip()


def evaluate(model, tokenizer, tasks, device: str, samples: int, max_tokens: int, temperature: float):
    rewards, answers, evidences = [], [], []
    best_reward_total = 0.0
    best_answer_total = 0.0
    for task in tasks:
        prompt = build_prompt(tokenizer, task)
        task_best_reward = 0.0
        task_best_answer = 0.0
        for _ in range(samples):
            answer = generate_one(model, tokenizer, prompt, device, max_tokens, temperature)
            reward = HotpotQAEnvironment.reward(task, answer)
            rewards.append(reward["total"])
            answers.append(reward["answer_f1"])
            evidences.append(reward["evidence"])
            task_best_reward = max(task_best_reward, reward["total"])
            task_best_answer = max(task_best_answer, reward["answer_f1"])
        best_reward_total += task_best_reward
        best_answer_total += task_best_answer
    total = max(len(rewards), 1)
    return {
        "reward": sum(rewards) / total,
        "answer_f1": sum(answers) / total,
        "evidence": sum(evidences) / total,
        "best_reward": best_reward_total / max(len(tasks), 1),
        "best_answer_f1": best_answer_total / max(len(tasks), 1),
    }


def load_tasks(path: str, offset: int, tasks: int):
    env = HotpotQAEnvironment.from_jsonl(path, limit=offset + tasks)
    return env.tasks[offset:offset + tasks]


def write_jsonl(path: Path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def weighted(rows, key):
    denom = sum(row["tasks"] for row in rows)
    return sum(row[key] * row["tasks"] for row in rows) / max(denom, 1)


def write_summary(path: Path, rows):
    keys = ["answer_f1", "evidence", "reward", "best_answer_f1", "best_reward"]
    lines = ["# HotpotQA Main Oracle Answer Evaluation", ""]
    lines.append("| model | offset | tasks | samples | " + " | ".join(keys) + " |")
    lines.append("|---|---:|---:|---:|" + "|".join(["---:"] * len(keys)) + "|")
    for row in rows:
        values = [row["model"], str(row["offset"]), str(row["tasks"]), str(row["samples"])]
        values.extend(f"{row[key]:.3f}" for key in keys)
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Task-Weighted Averages")
    lines.append("")
    lines.append("| model | " + " | ".join(keys) + " |")
    lines.append("|---|" + "|".join(["---:"] * len(keys)) + "|")
    for model in sorted(set(row["model"] for row in rows)):
        model_rows = [row for row in rows if row["model"] == model]
        values = [model]
        values.extend(f"{weighted(model_rows, key):.3f}" for key in keys)
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Main with oracle Sub result.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--name", default="main")
    parser.add_argument("--main-lora", required=True)
    parser.add_argument("--val-jsonl", default="./data/base/val.jsonl")
    parser.add_argument("--out-dir", default="./artifacts/eval/main_oracle")
    parser.add_argument("--offsets", type=int, nargs="+", default=[0, 20, 40])
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.4)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "results.jsonl"
    if out_jsonl.exists():
        out_jsonl.unlink()
    rows = []
    print(f"[main-oracle] loading {args.name}", flush=True)
    model, tokenizer = load_model(args.base_model, args.main_lora, device)
    for offset in args.offsets:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        tasks = load_tasks(args.val_jsonl, offset, args.tasks)
        print(f"[main-oracle] model={args.name} offset={offset} tasks={len(tasks)}", flush=True)
        metrics = evaluate(model, tokenizer, tasks, device, args.samples, args.max_tokens, args.temperature)
        row = {
            "model": args.name,
            "offset": offset,
            "tasks": len(tasks),
            "samples": args.samples,
            **metrics,
        }
        write_jsonl(out_jsonl, row)
        rows.append(row)
    write_summary(out_dir / "summary.md", rows)
    print(f"[main-oracle] wrote {out_jsonl}", flush=True)
    print(f"[main-oracle] wrote {out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()

