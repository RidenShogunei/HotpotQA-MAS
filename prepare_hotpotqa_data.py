"""Prepare small HotpotQA JSONL splits for local agent RL experiments."""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def _context_to_docs(context):
    titles = context["title"]
    sentences = context["sentences"]
    docs = []
    for idx, (title, sents) in enumerate(zip(titles, sentences)):
        docs.append({
            "doc_id": f"D{idx:02d}",
            "title": title,
            "text": " ".join(sents),
            "sentences": sents,
        })
    return docs


def _support_doc_ids(context, supporting_facts):
    title_to_doc = {title: f"D{idx:02d}" for idx, title in enumerate(context["title"])}
    ids = []
    for title in supporting_facts["title"]:
        doc_id = title_to_doc.get(title)
        if doc_id and doc_id not in ids:
            ids.append(doc_id)
    return ids


def _convert(example, idx):
    docs = _context_to_docs(example["context"])
    return {
        "task_id": str(idx),
        "question": example["question"],
        "answer": example["answer"],
        "type": example.get("type", ""),
        "level": example.get("level", ""),
        "support_doc_ids": _support_doc_ids(example["context"], example["supporting_facts"]),
        "support_titles": example["supporting_facts"]["title"],
        "docs": docs,
    }


def write_split(dataset, path: Path, n: int, seed: int):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    rows = [_convert(dataset[i], i) for i in indices[:n]]
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare local HotpotQA JSONL files.")
    parser.add_argument("--output-dir", default="data/base")
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="distractor")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[hotpotqa] loading hotpotqa/hotpot_qa config={args.config}")
    dataset = load_dataset("hotpotqa/hotpot_qa", args.config, trust_remote_code=True)
    train_n = write_split(dataset["train"], out / "train.jsonl", args.train_size, args.seed)
    val_n = write_split(dataset["validation"], out / "val.jsonl", args.val_size, args.seed + 1)
    print(f"[hotpotqa] wrote train={train_n} val={val_n} to {out}")


if __name__ == "__main__":
    main()

