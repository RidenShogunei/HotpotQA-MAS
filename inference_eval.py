"""Inference evaluation for trained Main/Sub LoRA adapters on HotpotQA."""

import argparse
import json
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import TrainingConfig


def load_model_with_adapter(base_model_path: str, adapter_path: str, device: str = "cuda:4"):
    """Load base model with LoRA adapter."""
    print(f"[system] Loading base model from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use same device_map as training (GPUs 4,5,6)
    num_layers = 32
    device_map = {
        "model.embed_tokens": 4,
        "model.rotary_emb": 4,
        "model.norm": 6,
        "lm_head": 6,
    }
    for i in range(num_layers):
        if i < num_layers // 3:
            device_map[f"model.layers.{i}"] = 4
        elif i < 2 * num_layers // 3:
            device_map[f"model.layers.{i}"] = 5
        else:
            device_map[f"model.layers.{i}"] = 6

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map=device_map,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    print(f"[system] Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path, adapter_name="test")
    model.set_adapter("test")
    model.eval()

    return model, tokenizer


def generate_answer(model, tokenizer, question: str, context: str = "", max_new_tokens: int = 256) -> str:
    """Generate answer for a HotpotQA question."""
    if context:
        prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    # Move to embed_tokens device
    embed_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(embed_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def normalize_answer(s: str) -> str:
    """Normalize answer for EM/F1 comparison."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return int(pred_tokens == gold_tokens)

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_on_hotpotqa(model, tokenizer, data_path: str, max_samples: int = None, max_new_tokens: int = 256):
    """Evaluate on HotpotQA dev set (jsonl format)."""
    print(f"[system] Loading HotpotQA data from {data_path}...")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))

    if max_samples:
        data = data[:max_samples]

    print(f"[system] Evaluating on {len(data)} samples...")
    results = []
    correct = 0
    total_f1 = 0.0

    for i, item in enumerate(data):
        question = item["question"]
        gold_answer = item.get("answer", "")
        # Build context from support docs
        context = ""
        if "docs" in item:
            support_ids = item.get("support_doc_ids", [])
            context_sents = []
            for doc in item["docs"]:
                if doc.get("doc_id") in support_ids:
                    context_sents.extend(doc.get("sentences", []))
            context = " ".join(context_sents[:20])  # Limit context

        pred_answer = generate_answer(model, tokenizer, question, context, max_new_tokens)

        em = exact_match(pred_answer, gold_answer)
        f1 = f1_score(pred_answer, gold_answer)
        correct += int(em)
        total_f1 += f1

        results.append({
            "question": question,
            "gold": gold_answer,
            "pred": pred_answer,
            "em": em,
            "f1": f1,
        })

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(data)} | EM: {correct/(i+1)*100:.1f}% | F1: {total_f1/(i+1)*100:.1f}%")

    em_score = correct / len(data) * 100
    f1_score_avg = total_f1 / len(data) * 100

    print(f"\n{'='*60}")
    print(f"Evaluation Results ({len(data)} samples)")
    print(f"{'='*60}")
    print(f"Exact Match: {em_score:.2f}%")
    print(f"F1 Score:    {f1_score_avg:.2f}%")

    return {
        "em": em_score,
        "f1": f1_score_avg,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter on HotpotQA.")
    parser.add_argument("--adapter-path", required=True, help="Path to LoRA adapter checkpoint.")
    parser.add_argument("--data-path", default="data/base/val.jsonl",
                        help="Path to HotpotQA dev data (jsonl)."),
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Max samples to evaluate (for quick test).")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default=None, help="Path to save detailed results JSON.")
    args = parser.parse_args()

    model, tokenizer = load_model_with_adapter(args.base_model, args.adapter_path)

    eval_results = evaluate_on_hotpotqa(
        model, tokenizer, args.data_path,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        print(f"[system] Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
