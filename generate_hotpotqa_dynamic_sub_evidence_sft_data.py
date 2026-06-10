"""Generate Sub-only evidence summary SFT data for dynamic HotpotQA MAS."""

import argparse
import json
from pathlib import Path

from generate_hotpotqa_dynamic_mixture_sft_data import (
    build_focused_sub_action_samples,
    focused_call_plan,
    focused_history,
)
from generate_hotpotqa_dynamic_mas_sft_data import oracle_subtasks
from generate_hotpotqa_mas_sft_data import SUB_SUMMARY_SYSTEM, build_sub_action_samples, history_text
from hotpotqa_environment import HotpotQAEnvironment, HotpotTask


def evidence_snippet(task: HotpotTask, doc_id: str, max_chars: int):
    for doc in task.docs:
        if doc.doc_id == doc_id:
            return doc.text.replace("\n", " ").strip()[:max_chars]
    return ""


def build_sub_evidence_summary_samples(task: HotpotTask, max_subtasks: int, max_snippet_chars: int):
    samples = []
    subtasks = oracle_subtasks(task, max_subtasks)
    for subtask, doc_id, title in zip(subtasks, task.support_doc_ids, task.support_titles):
        calls = focused_call_plan(task, doc_id, title)
        snippet = evidence_snippet(task, doc_id, max_snippet_chars)
        samples.append(
            {
                "messages": [
                    {"role": "system", "content": SUB_SUMMARY_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Subtask: {subtask}\nResearch history:\n{history_text(focused_history(task, calls))}",
                    },
                    {
                        "role": "assistant",
                        "content": (
                            "<thinking>Report the focused evidence without guessing the final answer.</thinking>"
                            f"<result>evidence clue from {title}: {snippet} | evidence: {doc_id}</result>"
                        ),
                    },
                ],
                "category": "sub",
                "stage": "summary_dynamic_evidence",
                "task_type": task.task_type,
            }
        )
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Sub-only dynamic evidence summary SFT data.")
    parser.add_argument("--train-jsonl", default="./data/enhanced/train.jsonl")
    parser.add_argument("--output", default="data/sft/hotpotqa_dynamic_sub_evidence_sft_data.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-subtasks", type=int, default=2)
    parser.add_argument("--max-snippet-chars", type=int, default=360)
    parser.add_argument("--include-action-replay", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    env = HotpotQAEnvironment.from_jsonl(args.train_jsonl, limit=args.limit)
    samples = []
    for task in env.tasks:
        if args.include_action_replay:
            samples.extend(build_sub_action_samples(task))
            samples.extend(build_focused_sub_action_samples(task, args.max_subtasks))
        samples.extend(build_sub_evidence_summary_samples(task, args.max_subtasks, args.max_snippet_chars))

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[hotpotqa-dynamic-sub-evidence-sft] wrote {len(samples)} samples to {out}")
    print("[hotpotqa-dynamic-sub-evidence-sft] main={}".format(sum(1 for s in samples if s["category"] == "main")))
    print("[hotpotqa-dynamic-sub-evidence-sft] sub={}".format(sum(1 for s in samples if s["category"] == "sub")))


if __name__ == "__main__":
    main()

