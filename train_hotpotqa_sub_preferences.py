"""Train the HotpotQA MAS Sub adapter with action preference pairs.

This targets retrieval directly: for the same Sub action prompt, prefer
search(question) or read(gold_doc_id) over wrong read actions.
"""

import argparse
import os
import random
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_hotpotqa_mas_sft_data import SUB_ACTION_SYSTEM, history_text, oracle_subtask
from hotpotqa_environment import HotpotQAEnvironment, HotpotTask


def build_prompt(tokenizer, subtask: str, history):
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SUB_ACTION_SYSTEM},
            {"role": "user", "content": f"Subtask: {subtask}\nResearch history:\n{history_text(history)}"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def action_text(call: str, reason: str) -> str:
    return f"<thinking>{reason}</thinking>[tool_call]{call}[/tool_call]"


def non_gold_doc(task: HotpotTask, rng: random.Random) -> str:
    gold = set(task.support_doc_ids)
    candidates = [doc.doc_id for doc in task.docs if doc.doc_id not in gold]
    if not candidates:
        candidates = [doc.doc_id for doc in task.docs]
    return rng.choice(candidates)


def build_pairs(tasks, tokenizer, max_pairs: int, seed: int):
    rng = random.Random(seed)
    pairs = []
    shuffled = list(tasks)
    rng.shuffle(shuffled)
    for task in shuffled:
        subtask = oracle_subtask(task)

        prompt = build_prompt(tokenizer, subtask, [])
        wrong_doc = non_gold_doc(task, rng)
        pairs.append({
            "prompt": prompt,
            "chosen": action_text(f'search("{task.question}")', "Search for candidate evidence pages."),
            "rejected": action_text(f'read("{wrong_doc}")', "Read a page before searching."),
        })

        search_call = f'search("{task.question}")'
        ok, search_obs = HotpotQAEnvironment.execute_tool(task, f"[tool_call]{search_call}[/tool_call]")
        history = [(search_call, search_obs if ok else "Tool execution failed")]
        for gold_doc in task.support_doc_ids:
            wrong_doc = non_gold_doc(task, rng)
            prompt = build_prompt(tokenizer, subtask, history)
            pairs.append({
                "prompt": prompt,
                "chosen": action_text(f'read("{gold_doc}")', "Read a supporting page."),
                "rejected": action_text(f'read("{wrong_doc}")', "Read a distractor page."),
            })
            ok, obs = HotpotQAEnvironment.execute_tool(task, f"[tool_call]read(\"{gold_doc}\")[/tool_call]")
            history.append((f'read("{gold_doc}")', obs if ok else "Tool execution failed"))

        if max_pairs and len(pairs) >= max_pairs:
            return pairs[:max_pairs]
    return pairs[:max_pairs] if max_pairs else pairs


def load_model(base_model: str, sub_lora: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(model, sub_lora, adapter_name="sub", is_trainable=True)
    model.set_adapter("sub")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.train()
    return model, tokenizer


def response_logprob(model, tokenizer, prompt: str, response: str, device: str, max_length: int):
    text = prompt + response + (tokenizer.eos_token or "")
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = min(len(prompt_ids), enc["input_ids"].shape[1])

    labels = enc["input_ids"].clone()
    labels[:, :prompt_len] = -100
    outputs = model(**enc)
    shift_logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    mask = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~mask, 0)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * mask).sum() / mask.sum().clamp_min(1)


def save_adapter(model, tokenizer, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    try:
        model.save_pretrained(output_dir, selected_adapters=["sub"])
    except TypeError:
        model.save_pretrained(output_dir, adapter_name="sub")
    nested = Path(output_dir) / "sub"
    if nested.is_dir() and (nested / "adapter_config.json").exists():
        for item in nested.iterdir():
            target = Path(output_dir) / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        nested.rmdir()
    tokenizer.save_pretrained(output_dir)


def train(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.base_model, args.sub_lora, device)
    env = HotpotQAEnvironment.from_jsonl(args.train_jsonl, limit=args.tasks)
    pairs = build_pairs(env.tasks, tokenizer, args.max_pairs, args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("[hotpotqa-sub-pref]")
    print(f"tasks={len(env.tasks)} pairs={len(pairs)} epochs={args.epochs}")
    print(f"sub_lora={args.sub_lora}")
    print(f"save_dir={args.save_dir}")

    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(pairs)
        total_loss = 0.0
        total_margin = 0.0
        steps = 0
        for pair in pairs:
            chosen_lp = response_logprob(model, tokenizer, pair["prompt"], pair["chosen"], device, args.max_length)
            rejected_lp = response_logprob(model, tokenizer, pair["prompt"], pair["rejected"], device, args.max_length)
            margin = chosen_lp - rejected_lp
            pref_loss = -torch.nn.functional.logsigmoid(args.beta * margin)
            sft_loss = -chosen_lp
            loss = pref_loss + args.sft_weight * sft_loss
            if not torch.isfinite(loss):
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            total_margin += margin.detach().item()
            steps += 1
        print(
            f"epoch {epoch + 1}/{args.epochs} "
            f"loss={total_loss / max(steps, 1):.4f} "
            f"margin={total_margin / max(steps, 1):.4f}"
        )

    save_adapter(model, tokenizer, args.save_dir)
    print("[OK] saved Sub preference adapter")


def parse_args():
    parser = argparse.ArgumentParser(description="Train HotpotQA Sub action preferences.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--sub-lora", required=True)
    parser.add_argument("--train-jsonl", default="./data/base/train.jsonl")
    parser.add_argument("--save-dir", default="./artifacts/checkpoints/sub_preferences")
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--sft-weight", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

