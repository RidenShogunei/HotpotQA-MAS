"""Evaluate a HotpotQA MAS Sub checkpoint with oracle subtasks.

This isolates the Sub researcher from Main planning and final synthesis noise:
Sub receives the oracle research request, performs search/read actions, then
summarizes evidence. Metrics are computed from Sub outputs only.
"""

import argparse
import random
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_hotpotqa_mas_sft_data import SUB_ACTION_SYSTEM, SUB_SUMMARY_SYSTEM, oracle_subtask
from hotpotqa_environment import HotpotQAEnvironment


def load_model(base_model: str, sub_lora: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, sub_lora, adapter_name="sub")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, system: str, user: str):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


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
    text = prefix + generated
    for stop in ("[/tool_call]", "</result>"):
        end = text.find(stop)
        if end >= 0:
            return text[:end + len(stop)].strip()
    return text.strip()


def history_text(history):
    if not history:
        return "No observations yet."
    lines = []
    for idx, (tool_call, observation) in enumerate(history, 1):
        lines.append(f"Step {idx} tool call: {tool_call}")
        lines.append(f"Step {idx} observation: {observation}")
    return "\n".join(lines)


def extract_tool_call(text: str) -> str:
    match = re.search(r"\[tool_call\].*?\[/tool_call\]", text, re.DOTALL)
    return match.group(0) if match else text


def read_doc_id(tool_call: str):
    parsed = HotpotQAEnvironment.parse_tool_call(tool_call)
    if parsed is None:
        return None
    tool, arg = parsed
    return arg if tool == "read" else None


def evaluate(model, tokenizer, tasks, device: str, samples: int, max_tokens: int, sub_steps: int, temperature: float):
    rewards, answers, evidences = [], [], []
    action_valids, any_valids, read_recalls = [], [], []
    best_reward_total = 0.0
    best_answer_total = 0.0
    best_read_total = 0.0

    for task in tasks:
        task_best_reward = 0.0
        task_best_answer = 0.0
        task_best_read = 0.0
        subtask = oracle_subtask(task)
        gold_docs = set(task.support_doc_ids)

        for _ in range(samples):
            history = []
            tool_calls = []
            valid_actions = 0
            read_docs = set()

            for _step in range(sub_steps):
                action_prompt = build_prompt(
                    tokenizer,
                    SUB_ACTION_SYSTEM,
                    f"Subtask: {subtask}\nResearch history:\n{history_text(history)}",
                )
                action_raw = generate_one(model, tokenizer, action_prompt, device, max_tokens, temperature)
                tool_call = extract_tool_call(action_raw)
                ok, observation = HotpotQAEnvironment.execute_tool(task, tool_call)
                if ok:
                    valid_actions += 1
                    doc_id = read_doc_id(tool_call)
                    if doc_id:
                        read_docs.add(doc_id)
                else:
                    observation = "Tool execution failed"
                tool_calls.append(tool_call)
                history.append((tool_call, observation))

            summary_prompt = build_prompt(
                tokenizer,
                SUB_SUMMARY_SYSTEM,
                f"Subtask: {subtask}\nResearch history:\n{history_text(history)}",
            )
            sub_summary = generate_one(model, tokenizer, summary_prompt, device, max_tokens, temperature)
            reward = HotpotQAEnvironment.reward(task, "".join(tool_calls) + sub_summary)
            read_recall = len(read_docs & gold_docs) / max(len(gold_docs), 1)

            rewards.append(reward["total"])
            answers.append(reward["answer_f1"])
            evidences.append(reward["evidence"])
            action_valids.append(valid_actions / max(sub_steps, 1))
            any_valids.append(1.0 if valid_actions else 0.0)
            read_recalls.append(read_recall)
            task_best_reward = max(task_best_reward, reward["total"])
            task_best_answer = max(task_best_answer, reward["answer_f1"])
            task_best_read = max(task_best_read, read_recall)

        best_reward_total += task_best_reward
        best_answer_total += task_best_answer
        best_read_total += task_best_read

    total = max(len(rewards), 1)
    task_total = max(len(tasks), 1)
    return {
        "reward": sum(rewards) / total,
        "answer_f1": sum(answers) / total,
        "evidence": sum(evidences) / total,
        "action_valid": sum(action_valids) / total,
        "tool_valid": sum(any_valids) / total,
        "support_read_recall": sum(read_recalls) / total,
        "best_reward": best_reward_total / task_total,
        "best_answer_f1": best_answer_total / task_total,
        "best_support_read_recall": best_read_total / task_total,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA MAS Sub with oracle subtasks.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--sub-lora", default="./hotpotqa_mas_sft_checkpoints_v2/sub_agent/sub")
    parser.add_argument("--val-jsonl", default="./data/base/val.jsonl")
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--sub-steps", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.4)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = HotpotQAEnvironment.from_jsonl(args.val_jsonl, limit=args.offset + args.tasks)
    tasks = env.tasks[args.offset:args.offset + args.tasks]
    model, tokenizer = load_model(args.base_model, args.sub_lora, device)
    metrics = evaluate(
        model,
        tokenizer,
        tasks,
        device,
        args.samples,
        args.max_tokens,
        args.sub_steps,
        args.temperature,
    )
    print("[hotpotqa-sub-oracle-eval]")
    print(f"sub_lora={args.sub_lora}")
    print(f"tasks={len(tasks)} offset={args.offset} seed={args.seed} samples={args.samples} sub_steps={args.sub_steps}")
    print(f"tool_valid={metrics['tool_valid']:.3f}")
    print(f"action_valid={metrics['action_valid']:.3f}")
    print(f"support_read_recall={metrics['support_read_recall']:.3f}")
    print(f"answer_f1={metrics['answer_f1']:.3f}")
    print(f"evidence={metrics['evidence']:.3f}")
    print(f"reward={metrics['reward']:.3f}")
    print(f"best_support_read_recall={metrics['best_support_read_recall']:.3f}")
    print(f"best_answer_f1={metrics['best_answer_f1']:.3f}")
    print(f"best_reward={metrics['best_reward']:.3f}")


if __name__ == "__main__":
    main()

