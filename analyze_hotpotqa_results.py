"""Evaluate Main adapters on local HotpotQA multi-hop tool trajectories."""

import argparse
import random
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_hotpotqa_sft_data import ANSWER_SYSTEM, MAIN_SYSTEM
from hotpotqa_environment import HotpotQAEnvironment, HotpotTask


def load_model(base_model: str, lora_path: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path, adapter_name="default")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def history_text(history):
    if not history:
        return "No observations yet."
    lines = []
    for idx, (tool_call, observation) in enumerate(history, 1):
        lines.append(f"Step {idx} tool call: {tool_call}")
        lines.append(f"Step {idx} observation: {observation}")
    return "\n".join(lines)


def build_action_prompt(tokenizer, task: HotpotTask, history):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": MAIN_SYSTEM},
            {"role": "user", "content": f"Question: {task.question}\nResearch history:\n{history_text(history)}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_answer_prompt(tokenizer, task: HotpotTask, history):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Question: {task.question}\nResearch history:\n{history_text(history)}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_tool_call_block(text: str) -> str:
    match = re.search(r"\[tool_call\].*?\[/tool_call\]", text, re.DOTALL)
    return match.group(0) if match else text


def truncate_result(text: str) -> str:
    end = text.find("</result>")
    if end >= 0:
        return text[:end + len("</result>")]
    return text


def generate_one(model, tokenizer, prompt: str, device: str, max_tokens: int, use_prefix: bool) -> str:
    prefix = "<thinking>" if use_prefix else ""
    inputs = tokenizer(prompt + prefix, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.4,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return truncate_result(prefix + generated).strip()


def evaluate(model, tokenizer, tasks, device: str, samples: int, max_tokens: int, use_prefix: bool, research_steps: int):
    rewards, answer_f1s, evidences, valids = [], [], [], []
    best_total = 0.0
    best_answer_total = 0.0
    for task in tasks:
        task_best = 0.0
        task_best_answer = 0.0
        for _ in range(samples):
            history = []
            tool_calls = []
            for _step in range(research_steps):
                prompt = build_action_prompt(tokenizer, task, history)
                action = generate_one(model, tokenizer, prompt, device, max_tokens, use_prefix)
                tool_call = extract_tool_call_block(action)
                ok, observation = HotpotQAEnvironment.execute_tool(task, tool_call)
                if not ok:
                    observation = "Tool execution failed"
                tool_calls.append(tool_call)
                history.append((tool_call, observation))
            answer_prompt = build_answer_prompt(tokenizer, task, history)
            answer = generate_one(model, tokenizer, answer_prompt, device, max_tokens, use_prefix)
            reward = HotpotQAEnvironment.reward(task, "".join(tool_calls) + answer)
            rewards.append(reward["total"])
            answer_f1s.append(reward["answer_f1"])
            evidences.append(reward["evidence"])
            valids.append(reward["tool_valid"])
            task_best = max(task_best, reward["total"])
            task_best_answer = max(task_best_answer, reward["answer_f1"])
        best_total += task_best
        best_answer_total += task_best_answer
    total = max(len(rewards), 1)
    return {
        "reward": sum(rewards) / total,
        "answer_f1": sum(answer_f1s) / total,
        "evidence": sum(evidences) / total,
        "tool_valid": sum(valids) / total,
        "best_reward": best_total / max(len(tasks), 1),
        "best_answer_f1": best_answer_total / max(len(tasks), 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA adapters.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--lora", default="")
    parser.add_argument("--val-jsonl", default="./data/base/val.jsonl")
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--research-steps", type=int, default=3)
    parser.add_argument("--raw", action="store_true")
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
    model, tokenizer = load_model(args.base_model, args.lora, device)
    metrics = evaluate(
        model,
        tokenizer,
        tasks,
        device=device,
        samples=args.samples,
        max_tokens=args.max_tokens,
        use_prefix=not args.raw,
        research_steps=args.research_steps,
    )
    print("[hotpotqa-eval]")
    print(f"lora={args.lora or '(base)'}")
    print(f"tasks={len(tasks)} offset={args.offset} seed={args.seed} samples={args.samples}")
    print(f"tool_valid={metrics['tool_valid']:.3f}")
    print(f"answer_f1={metrics['answer_f1']:.3f}")
    print(f"evidence={metrics['evidence']:.3f}")
    print(f"reward={metrics['reward']:.3f}")
    print(f"best_reward={metrics['best_reward']:.3f}")
    print(f"best_answer_f1={metrics['best_answer_f1']:.3f}")


if __name__ == "__main__":
    main()

