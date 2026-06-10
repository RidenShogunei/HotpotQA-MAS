"""Evaluate dynamic HotpotQA MAS checkpoints.

Main can choose direct answering or delegate 1..N subtasks. Multiple Sub
instances share the same Sub LoRA and run independently.
"""

import argparse
import random
import re

import torch

from analyze_hotpotqa_mas_results import (
    build_prompt,
    extract_tool_call,
    generate_one,
    history_text,
    load_model,
)
from generate_hotpotqa_mas_sft_data import MAIN_ANSWER_SYSTEM, SUB_ACTION_SYSTEM, SUB_SUMMARY_SYSTEM
from hotpotqa_environment import HotpotQAEnvironment


def doc_catalog(task) -> str:
    return "\n".join(f"{doc.doc_id}: {doc.title}" for doc in task.docs)


DYNAMIC_MAIN_PLAN_SYSTEM = (
    "You are the main coordinator agent. Decide whether to answer directly or delegate research.\n"
    "If the question can be answered directly from known evidence, output:\n"
    "<thinking>brief reason</thinking>\n"
    "[mode]direct[/mode]\n"
    "If research is needed, output:\n"
    "<thinking>brief delegation plan</thinking>\n"
    "[mode]delegate[/mode]\n"
    "[subtask]concrete research request 1[/subtask]\n"
    "Optionally add more [subtask]...[/subtask] blocks, up to the allowed maximum.\n"
    "Stop after the final [/mode] or [/subtask]."
)

DYNAMIC_DIRECT_ANSWER_SYSTEM = (
    "You are the main answer agent. Answer the question directly.\n"
    "Output exactly this format:\n"
    "<thinking>brief answer reasoning</thinking>\n"
    "<result>answer | evidence: DOCID, DOCID</result>\n"
    "Stop after </result>."
)

DYNAMIC_MAIN_ANSWER_SYSTEM = (
    "You are the main coordinator agent. Use all sub agent research results to answer.\n"
    "Output exactly this format:\n"
    "<thinking>brief synthesis across sub results</thinking>\n"
    "<result>answer | evidence: DOCID, DOCID</result>\n"
    "Stop after </result>."
)


def extract_mode(text: str) -> str:
    match = re.search(r"\[mode\]\s*(direct|delegate)\s*\[/mode\]", text, re.IGNORECASE)
    return match.group(1).lower() if match else "delegate"


def extract_subtasks(text: str, max_subagents: int):
    tasks = [m.strip() for m in re.findall(r"\[subtask\]\s*(.*?)\s*\[/subtask\]", text, re.DOTALL)]
    return [task for task in tasks if task][:max_subagents]


def sub_results_text(sub_results):
    if not sub_results:
        return "No sub results."
    lines = []
    for idx, result in enumerate(sub_results, 1):
        lines.append(f"Subtask {idx}: {result['subtask']}")
        lines.append(f"Sub result {idx}: {result['summary']}")
    return "\n".join(lines)


def run_sub_instance(model, tokenizer, task, device: str, subtask: str, max_tokens: int, sub_steps: int):
    history = []
    tool_calls = []
    ok_any = False
    for _step in range(sub_steps):
        action_prompt = build_prompt(
            tokenizer,
            SUB_ACTION_SYSTEM,
            f"Subtask: {subtask}\nResearch history:\n{history_text(history)}",
        )
        action_raw = generate_one(model, tokenizer, "sub", action_prompt, device, max_tokens)
        tool_call = extract_tool_call(action_raw)
        ok, observation = HotpotQAEnvironment.execute_tool(task, tool_call)
        if not ok:
            observation = "Tool execution failed"
        ok_any = ok_any or ok
        tool_calls.append(tool_call)
        history.append((tool_call, observation))

    summary_prompt = build_prompt(
        tokenizer,
        SUB_SUMMARY_SYSTEM,
        f"Subtask: {subtask}\nResearch history:\n{history_text(history)}",
    )
    summary = generate_one(model, tokenizer, "sub", summary_prompt, device, max_tokens)
    return {"subtask": subtask, "summary": summary, "tool_calls": tool_calls, "tool_valid": 1.0 if ok_any else 0.0}


def evaluate(model, tokenizer, tasks, device: str, samples: int, max_tokens: int, sub_steps: int, max_subagents: int):
    rewards, answers, evidences, valids = [], [], [], []
    direct_counts, subtask_counts = [], []
    best_reward_total = 0.0
    best_answer_total = 0.0

    for task in tasks:
        task_best_reward = 0.0
        task_best_answer = 0.0
        for _ in range(samples):
            plan_prompt = build_prompt(
                tokenizer,
                DYNAMIC_MAIN_PLAN_SYSTEM,
                f"Question: {task.question}\nAvailable documents:\n{doc_catalog(task)}",
            )
            plan_raw = generate_one(model, tokenizer, "main", plan_prompt, device, max_tokens)
            mode = extract_mode(plan_raw)
            subtasks = extract_subtasks(plan_raw, max_subagents)
            if mode == "delegate" and not subtasks:
                subtasks = [f"Find the supporting documents and answer for: {task.question}"]

            if mode == "direct":
                answer_prompt = build_prompt(tokenizer, DYNAMIC_DIRECT_ANSWER_SYSTEM, f"Question: {task.question}")
                answer_raw = generate_one(model, tokenizer, "main", answer_prompt, device, max_tokens)
                combined = plan_raw + answer_raw
                valid = 0.0
                subtask_count = 0
            else:
                sub_results = [
                    run_sub_instance(model, tokenizer, task, device, subtask, max_tokens, sub_steps)
                    for subtask in subtasks
                ]
                answer_prompt = build_prompt(
                    tokenizer,
                    DYNAMIC_MAIN_ANSWER_SYSTEM,
                    f"Question: {task.question}\nSub results:\n{sub_results_text(sub_results)}",
                )
                answer_raw = generate_one(model, tokenizer, "main", answer_prompt, device, max_tokens)
                combined = (
                    plan_raw
                    + "".join("".join(result["tool_calls"]) + result["summary"] for result in sub_results)
                    + answer_raw
                )
                valid = max((result["tool_valid"] for result in sub_results), default=0.0)
                subtask_count = len(sub_results)

            reward = HotpotQAEnvironment.reward(task, combined)
            rewards.append(reward["total"])
            answers.append(reward["answer_f1"])
            evidences.append(reward["evidence"])
            valids.append(valid)
            direct_counts.append(1.0 if mode == "direct" else 0.0)
            subtask_counts.append(subtask_count)
            task_best_reward = max(task_best_reward, reward["total"])
            task_best_answer = max(task_best_answer, reward["answer_f1"])

        best_reward_total += task_best_reward
        best_answer_total += task_best_answer

    total = max(len(rewards), 1)
    return {
        "reward": sum(rewards) / total,
        "answer_f1": sum(answers) / total,
        "evidence": sum(evidences) / total,
        "tool_valid": sum(valids) / total,
        "direct_rate": sum(direct_counts) / total,
        "avg_subtasks": sum(subtask_counts) / total,
        "best_reward": best_reward_total / max(len(tasks), 1),
        "best_answer_f1": best_answer_total / max(len(tasks), 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dynamic HotpotQA MAS checkpoints.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--main-lora", required=True)
    parser.add_argument("--sub-lora", required=True)
    parser.add_argument("--val-jsonl", default="./data/enhanced/val.jsonl")
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--sub-steps", type=int, default=3)
    parser.add_argument("--max-subagents", type=int, default=3)
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
    model, tokenizer = load_model(args.base_model, args.main_lora, args.sub_lora, device)
    metrics = evaluate(
        model,
        tokenizer,
        tasks,
        device,
        args.samples,
        args.max_tokens,
        args.sub_steps,
        args.max_subagents,
    )
    print("[hotpotqa-dynamic-mas-eval]")
    print(f"main_lora={args.main_lora}")
    print(f"sub_lora={args.sub_lora}")
    print(f"tasks={len(tasks)} offset={args.offset} seed={args.seed} samples={args.samples}")
    print(f"direct_rate={metrics['direct_rate']:.3f}")
    print(f"avg_subtasks={metrics['avg_subtasks']:.3f}")
    print(f"tool_valid={metrics['tool_valid']:.3f}")
    print(f"answer_f1={metrics['answer_f1']:.3f}")
    print(f"evidence={metrics['evidence']:.3f}")
    print(f"reward={metrics['reward']:.3f}")
    print(f"best_reward={metrics['best_reward']:.3f}")
    print(f"best_answer_f1={metrics['best_answer_f1']:.3f}")


if __name__ == "__main__":
    main()

