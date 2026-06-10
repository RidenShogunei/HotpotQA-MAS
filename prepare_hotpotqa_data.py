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


def _enhanced_context_to_docs(context):
    docs = []
    for idx, (title, sents) in enumerate(zip(context["title"], context["sentences"])):
        docs.append(
            {
                "source_doc_id": f"D{idx:02d}",
                "title": title,
                "text": " ".join(sents),
                "sentences": sents,
            }
        )
    return docs


def _enhanced_base(example, idx):
    titles = list(dict.fromkeys(example["supporting_facts"]["title"]))
    return {
        "task_id": str(idx),
        "question": example["question"],
        "answer": example["answer"],
        "type": example.get("type", ""),
        "level": example.get("level", ""),
        "support_titles": titles,
        "docs": _enhanced_context_to_docs(example["context"]),
    }


def _build_enhanced_row(base_rows, row_idx: int, docs_per_task: int, rng: random.Random):
    row = base_rows[row_idx]
    support_titles = set(row["support_titles"])
    selected = []
    seen = set()

    for is_support in (True, False):
        for doc in row["docs"]:
            if (doc["title"] in support_titles) != is_support:
                continue
            key = (doc["title"], doc["text"])
            if key not in seen:
                selected.append({**doc, "is_support": is_support, "source_task_id": row["task_id"]})
                seen.add(key)

    indices = list(range(len(base_rows)))
    rng.shuffle(indices)
    for other_idx in indices:
        if len(selected) >= docs_per_task:
            break
        if other_idx == row_idx:
            continue
        docs = list(base_rows[other_idx]["docs"])
        rng.shuffle(docs)
        for doc in docs:
            if len(selected) >= docs_per_task:
                break
            key = (doc["title"], doc["text"])
            if key in seen:
                continue
            selected.append({**doc, "is_support": False, "source_task_id": base_rows[other_idx]["task_id"]})
            seen.add(key)

    rng.shuffle(selected)
    docs = []
    support_doc_ids = []
    selected_support_titles = []
    for idx, doc in enumerate(selected):
        doc_id = f"D{idx:02d}"
        docs.append(
            {
                "doc_id": doc_id,
                "title": doc["title"],
                "text": doc["text"],
                "sentences": doc.get("sentences", []),
                "source_task_id": doc["source_task_id"],
                "source_doc_id": doc.get("source_doc_id", ""),
            }
        )
        if doc["is_support"]:
            support_doc_ids.append(doc_id)
            selected_support_titles.append(doc["title"])

    return {
        "task_id": row["task_id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "support_doc_ids": support_doc_ids,
        "support_titles": selected_support_titles,
        "docs": docs,
        "enhanced": {"docs_per_task": len(docs), "source": "hotpotqa_cross_distractors"},
    }


def write_enhanced_split(dataset, split_name, path, n, seed, docs_per_task, pool_multiplier):
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    pool_n = min(len(indices), max(n * pool_multiplier, n + 100))
    base_rows = [_enhanced_base(dataset[idx], idx) for idx in indices[:pool_n]]
    rows = [_build_enhanced_row(base_rows, idx, docs_per_task, rng) for idx in range(n)]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[enhanced] {split_name}: wrote {len(rows)} rows to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare local HotpotQA JSONL files.")
    parser.add_argument("--mode", choices=["base", "enhanced"], default="enhanced")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-size", type=int, default=200)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="distractor")
    parser.add_argument("--docs-per-task", type=int, default=30)
    parser.add_argument("--pool-multiplier", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir or f"data/{args.mode}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[hotpotqa] loading hotpotqa/hotpot_qa config={args.config}")
    dataset = load_dataset("hotpotqa/hotpot_qa", args.config, trust_remote_code=True)
    if args.mode == "enhanced":
        write_enhanced_split(
            dataset["train"], "train", out / "train.jsonl", args.train_size,
            args.seed, args.docs_per_task, args.pool_multiplier,
        )
        write_enhanced_split(
            dataset["validation"], "val", out / "val.jsonl", args.val_size,
            args.seed + 1, args.docs_per_task, args.pool_multiplier,
        )
    else:
        train_n = write_split(dataset["train"], out / "train.jsonl", args.train_size, args.seed)
        val_n = write_split(dataset["validation"], out / "val.jsonl", args.val_size, args.seed + 1)
        print(f"[hotpotqa] wrote train={train_n} val={val_n} to {out}")


if __name__ == "__main__":
    main()

