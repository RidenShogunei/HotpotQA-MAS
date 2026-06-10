"""Evaluate HotpotQA MAS Main/Sub checkpoints."""

import argparse
import random
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_hotpotqa_mas_sft_data import (
    MAIN_ANSWER_SYSTEM,
    MAIN_PLAN_SYSTEM,
    SUB_ACTION_SYSTEM,
    SUB_SUMMARY_SYSTEM,
)
from hotpotqa_environment import HotpotQAEnvironment


def load_model(base_model: str, main_lora: str, sub_lora: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    if main_lora:
        model = PeftModel.from_pretrained(model, main_lora, adapter_name="main")
    if sub_lora:
        if isinstance(model, PeftModel):
            model.load_adapter(sub_lora, adapter_name="sub")
        else:
            model = PeftModel.from_pretrained(model, sub_lora, adapter_name="sub")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def set_adapter(model, name: str):
    if hasattr(model, "set_adapter"):
        model.set_adapter(name)


def generate_one(model, tokenizer, adapter: str, prompt: str, device: str, max_tokens: int):
    set_adapter(model, adapter)
    prefix = "<thinking>"
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
    text = prefix + generated
    end = text.find("</result>")
    if end >= 0:
        text = text[:end + len("</result>")]
    return text.strip()


def history_text(history):
    if not history:
        return "No observations yet."
    lines = []
    for idx, (tool_call, observation) in enumerate(history, 1):
        lines.append(f"Step {idx} tool call: {tool_call}")
        lines.append(f"Step {idx} observation: {observation}")
    return "\n".join(lines)


def extract_block(text: str, tag: str) -> str:
    match = re.search(rf"\[{tag}\]\s*(.*?)\s*\[/{tag}\]", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def extract_tool_call(text: str) -> str:
    match = re.search(r"\[tool_call\].*?\[/tool_call\]", text, re.DOTALL)
    return match.group(0) if match else text


def build_prompt(tokenizer, system: str, user: str):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def evaluate(model, tokenizer, tasks, device: str, samples: int, max_tokens: int, sub_steps: int):
    rewards, answers, evidences, valids = [], [], [], []
    best_reward_total = 0.0
    best_answer_total = 0.0
    for task in tasks:
        task_best_reward = 0.0
        task_best_answer = 0.0
        for _ in range(samples):
            plan_prompt = build_prompt(tokenizer, MAIN_PLAN_SYSTEM, f"Question: {task.question}")
            plan_raw = generate_one(model, tokenizer, "main", plan_prompt, device, max_tokens)
            subtask = extract_block(plan_raw, "subtask")

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
            sub_summary = generate_one(model, tokenizer, "sub", summary_prompt, device, max_tokens)

            answer_prompt = build_prompt(
                tokenizer,
                MAIN_ANSWER_SYSTEM,
                f"Question: {task.question}\nSubtask: {subtask}\nSub result: {sub_summary}",
            )
            answer_raw = generate_one(model, tokenizer, "main", answer_prompt, device, max_tokens)
            reward = HotpotQAEnvironment.reward(task, "".join(tool_calls) + sub_summary + answer_raw)
            rewards.append(reward["total"])
            answers.append(reward["answer_f1"])
            evidences.append(reward["evidence"])
            valids.append(1.0 if ok_any else 0.0)
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
        "best_reward": best_reward_total / max(len(tasks), 1),
        "best_answer_f1": best_answer_total / max(len(tasks), 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HotpotQA MAS checkpoints.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--main-lora", default="./hotpotqa_mas_sft_checkpoints/main_agent/main")
    parser.add_argument("--sub-lora", default="./hotpotqa_mas_sft_checkpoints/sub_agent/sub")
    parser.add_argument("--val-jsonl", default="./data/base/val.jsonl")
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--sub-steps", type=int, default=3)
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
    metrics = evaluate(model, tokenizer, tasks, device, args.samples, args.max_tokens, args.sub_steps)
    print("[hotpotqa-mas-eval]")
    print(f"main_lora={args.main_lora}")
    print(f"sub_lora={args.sub_lora}")
    print(f"tasks={len(tasks)} offset={args.offset} seed={args.seed} samples={args.samples}")
    print(f"tool_valid={metrics['tool_valid']:.3f}")
    print(f"answer_f1={metrics['answer_f1']:.3f}")
    print(f"evidence={metrics['evidence']:.3f}")
    print(f"reward={metrics['reward']:.3f}")
    print(f"best_reward={metrics['best_reward']:.3f}")
    print(f"best_answer_f1={metrics['best_answer_f1']:.3f}")


if __name__ == "__main__":
    main()

