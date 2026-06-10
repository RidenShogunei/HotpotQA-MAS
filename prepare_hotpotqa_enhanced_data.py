"""Prepare harder HotpotQA splits with extra cross-example distractor documents."""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def context_to_docs(context):
    docs = []
    for idx, (title, sents) in enumerate(zip(context["title"], context["sentences"])):
        docs.append({
            "source_doc_id": f"D{idx:02d}",
            "title": title,
            "text": " ".join(sents),
            "sentences": sents,
        })
    return docs


def support_titles(example):
    titles = []
    for title in example["supporting_facts"]["title"]:
        if title not in titles:
            titles.append(title)
    return titles


def convert_base(example, idx):
    docs = context_to_docs(example["context"])
    supports = support_titles(example)
    return {
        "task_id": str(idx),
        "question": example["question"],
        "answer": example["answer"],
        "type": example.get("type", ""),
        "level": example.get("level", ""),
        "support_titles": supports,
        "docs": docs,
    }


def make_doc_key(doc):
    return (doc["title"], doc["text"])


def build_enhanced_row(base_rows, row_idx: int, docs_per_task: int, rng: random.Random):
    row = base_rows[row_idx]
    support_title_set = set(row["support_titles"])
    selected = []
    seen = set()

    for doc in row["docs"]:
        if doc["title"] in support_title_set:
            key = make_doc_key(doc)
            if key not in seen:
                selected.append({**doc, "is_support": True, "source_task_id": row["task_id"]})
                seen.add(key)

    for doc in row["docs"]:
        key = make_doc_key(doc)
        if key not in seen:
            selected.append({**doc, "is_support": False, "source_task_id": row["task_id"]})
            seen.add(key)

    candidate_indices = list(range(len(base_rows)))
    rng.shuffle(candidate_indices)
    for other_idx in candidate_indices:
        if len(selected) >= docs_per_task:
            break
        if other_idx == row_idx:
            continue
        other = base_rows[other_idx]
        other_docs = list(other["docs"])
        rng.shuffle(other_docs)
        for doc in other_docs:
            if len(selected) >= docs_per_task:
                break
            key = make_doc_key(doc)
            if key in seen:
                continue
            selected.append({**doc, "is_support": False, "source_task_id": other["task_id"]})
            seen.add(key)

    rng.shuffle(selected)
    docs = []
    support_doc_ids = []
    support_titles = []
    for idx, doc in enumerate(selected):
        doc_id = f"D{idx:02d}"
        docs.append({
            "doc_id": doc_id,
            "title": doc["title"],
            "text": doc["text"],
            "sentences": doc.get("sentences", []),
            "source_task_id": doc["source_task_id"],
            "source_doc_id": doc.get("source_doc_id", ""),
        })
        if doc["is_support"]:
            support_doc_ids.append(doc_id)
            support_titles.append(doc["title"])

    return {
        "task_id": row["task_id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "support_doc_ids": support_doc_ids,
        "support_titles": support_titles,
        "docs": docs,
        "enhanced": {
            "docs_per_task": len(docs),
            "source": "hotpotqa_cross_distractors",
        },
    }


def write_split(dataset, split_name: str, path: Path, n: int, seed: int, docs_per_task: int, pool_multiplier: int):
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    pool_n = min(len(indices), max(n * pool_multiplier, n + 100))
    pool_indices = indices[:pool_n]
    base_rows = [convert_base(dataset[i], i) for i in pool_indices]
    rows = [build_enhanced_row(base_rows, i, docs_per_task, rng) for i in range(n)]
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[enhanced] {split_name}: wrote {len(rows)} rows to {path}")
    print(f"[enhanced] {split_name}: avg_docs={sum(len(r['docs']) for r in rows)/max(len(rows),1):.1f}")
    print(f"[enhanced] {split_name}: avg_support_docs={sum(len(r['support_doc_ids']) for r in rows)/max(len(rows),1):.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare enhanced HotpotQA local-context splits.")
    parser.add_argument("--output-dir", default="data/enhanced")
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=150)
    parser.add_argument("--docs-per-task", type=int, default=30)
    parser.add_argument("--pool-multiplier", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--config", default="distractor")
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[enhanced] loading hotpotqa/hotpot_qa config={args.config}")
    dataset = load_dataset("hotpotqa/hotpot_qa", args.config, trust_remote_code=True)
    write_split(
        dataset["train"],
        "train",
        out / "train.jsonl",
        args.train_size,
        args.seed,
        args.docs_per_task,
        args.pool_multiplier,
    )
    write_split(
        dataset["validation"],
        "val",
        out / "val.jsonl",
        args.val_size,
        args.seed + 1,
        args.docs_per_task,
        args.pool_multiplier,
    )


if __name__ == "__main__":
    main()

