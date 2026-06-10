"""Generate Main-only synthesis SFT data for dynamic HotpotQA MAS."""

import argparse
import json
from pathlib import Path

from generate_hotpotqa_dynamic_mas_sft_data import DYNAMIC_MAIN_ANSWER_SYSTEM, oracle_subtasks
from hotpotqa_environment import HotpotQAEnvironment, HotpotTask


def evidence_snippet(task: HotpotTask, doc_id: str, max_chars: int):
    for doc in task.docs:
        if doc.doc_id == doc_id:
            text = doc.text.replace("\n", " ").strip()
            return text[:max_chars]
    return ""


def focused_sub_results_text(task: HotpotTask, max_subtasks: int, max_snippet_chars: int):
    lines = []
    subtasks = oracle_subtasks(task, max_subtasks)
    for idx, (subtask, doc_id, title) in enumerate(zip(subtasks, task.support_doc_ids, task.support_titles), 1):
        snippet = evidence_snippet(task, doc_id, max_snippet_chars)
        lines.append(f"Subtask {idx}: {subtask}")
        lines.append(
            "Sub result "
            f"{idx}: <result>evidence clue from {title}: {snippet} | evidence: {doc_id}</result>"
        )
    return "\n".join(lines)


def build_synthesis_sample(task: HotpotTask, max_subtasks: int, max_snippet_chars: int):
    evidence = ", ".join(task.support_doc_ids)
    return {
        "messages": [
            {"role": "system", "content": DYNAMIC_MAIN_ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {task.question}\n"
                    f"Sub results:\n{focused_sub_results_text(task, max_subtasks, max_snippet_chars)}"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "<thinking>Combine the focused evidence clues into one final answer.</thinking>"
                    f"<result>{task.answer} | evidence: {evidence}</result>"
                ),
            },
        ],
        "category": "main",
        "stage": "answer_dynamic_synthesis",
        "task_type": task.task_type,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Main-only dynamic synthesis SFT data.")
    parser.add_argument("--train-jsonl", default="./data/enhanced/train.jsonl")
    parser.add_argument("--output", default="data/sft/hotpotqa_dynamic_synthesis_sft_data.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-subtasks", type=int, default=2)
    parser.add_argument("--max-snippet-chars", type=int, default=360)
    return parser.parse_args()


def main():
    args = parse_args()
    env = HotpotQAEnvironment.from_jsonl(args.train_jsonl, limit=args.limit)
    samples = [build_synthesis_sample(task, args.max_subtasks, args.max_snippet_chars) for task in env.tasks]

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[hotpotqa-dynamic-synthesis-sft] wrote {len(samples)} samples to {out}")
    print("[hotpotqa-dynamic-synthesis-sft] main={}".format(sum(1 for s in samples if s["category"] == "main")))
    print("[hotpotqa-dynamic-synthesis-sft] sub={}".format(sum(1 for s in samples if s["category"] == "sub")))


if __name__ == "__main__":
    main()

