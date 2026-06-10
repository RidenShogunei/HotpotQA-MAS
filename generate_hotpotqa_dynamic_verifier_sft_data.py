"""Generate Main-only verifier/synthesis SFT data for dynamic HotpotQA MAS."""

import argparse
import json
import random
from pathlib import Path

from generate_hotpotqa_dynamic_mas_sft_data import DYNAMIC_MAIN_ANSWER_SYSTEM, oracle_subtasks
from hotpotqa_environment import HotpotDoc, HotpotQAEnvironment, HotpotTask


def compact(text: str, max_chars: int) -> str:
    return text.replace("\n", " ").strip()[:max_chars]


def evidence_snippet(task: HotpotTask, doc_id: str, max_chars: int) -> str:
    for doc in task.docs:
        if doc.doc_id == doc_id:
            return compact(doc.text, max_chars)
    return ""


def distractor_docs(task: HotpotTask) -> list[HotpotDoc]:
    support = set(task.support_doc_ids)
    return [doc for doc in task.docs if doc.doc_id not in support]


def wrong_answer(task: HotpotTask, rng: random.Random) -> str:
    candidates = [doc.title for doc in distractor_docs(task)]
    candidates.extend(title for title in task.support_titles if title != task.answer)
    candidates = [item for item in candidates if HotpotQAEnvironment.token_f1(item, task.answer) < 0.5]
    return rng.choice(candidates) if candidates else "unknown"


def result_line(answer: str, doc_id: str, title: str, snippet: str, label: str) -> str:
    return f"<result>{answer} because {label} from {title}: {snippet} | evidence: {doc_id}</result>"


def gold_sub_lines(
    task: HotpotTask,
    max_subtasks: int,
    max_snippet_chars: int,
    start_idx: int = 1,
) -> list[str]:
    subtasks = oracle_subtasks(task, max_subtasks)
    lines = []
    for idx, (subtask, doc_id, title) in enumerate(
        zip(subtasks, task.support_doc_ids, task.support_titles),
        start_idx,
    ):
        snippet = evidence_snippet(task, doc_id, max_snippet_chars)
        lines.append(f"Subtask {idx}: {subtask}")
        lines.append(
            f"Sub result {idx}: "
            + result_line(task.answer, doc_id, title, snippet, "supporting evidence")
        )
    return lines


def distractor_result(task: HotpotTask, rng: random.Random, idx: int, max_snippet_chars: int) -> list[str]:
    docs = distractor_docs(task)
    doc = rng.choice(docs) if docs else task.docs[-1]
    answer = wrong_answer(task, rng)
    return [
        f"Subtask {idx}: Check a plausible but possibly irrelevant document for: {task.question}",
        f"Sub result {idx}: "
        + result_line(answer, doc.doc_id, doc.title, compact(doc.text, max_snippet_chars), "distractor clue"),
    ]


def conflicting_result(task: HotpotTask, rng: random.Random, idx: int, max_snippet_chars: int) -> list[str]:
    doc_id = task.support_doc_ids[0]
    title = task.support_titles[0]
    answer = wrong_answer(task, rng)
    return [
        f"Subtask {idx}: Interpret evidence from {doc_id} ({title}) for: {task.question}",
        f"Sub result {idx}: "
        + result_line(answer, doc_id, title, evidence_snippet(task, doc_id, max_snippet_chars), "misread evidence"),
    ]


def build_sub_results(task: HotpotTask, rng: random.Random, variant: str, max_subtasks: int, max_snippet_chars: int):
    gold = gold_sub_lines(task, max_subtasks, max_snippet_chars)
    if variant == "gold_only":
        return "\n".join(gold)
    if variant == "wrong_first":
        lines = conflicting_result(task, rng, 1, max_snippet_chars)
        lines.extend(gold_sub_lines(task, max_subtasks, max_snippet_chars, start_idx=2))
        return "\n".join(lines)
    if variant == "distractor_extra":
        lines = gold_sub_lines(task, max_subtasks, max_snippet_chars)
        lines.extend(distractor_result(task, rng, len(lines) // 2 + 1, max_snippet_chars))
        return "\n".join(lines)
    if variant == "partial_plus_distractor":
        lines = gold_sub_lines(task, max(1, min(max_subtasks, 1)), max_snippet_chars)
        lines.extend(distractor_result(task, rng, 2, max_snippet_chars))
        return "\n".join(lines)
    raise ValueError(f"Unknown verifier variant: {variant}")


def build_verifier_sample(
    task: HotpotTask,
    rng: random.Random,
    variant: str,
    max_subtasks: int,
    max_snippet_chars: int,
):
    evidence = ", ".join(task.support_doc_ids)
    return {
        "messages": [
            {"role": "system", "content": DYNAMIC_MAIN_ANSWER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question: {task.question}\n"
                    f"Sub results:\n{build_sub_results(task, rng, variant, max_subtasks, max_snippet_chars)}"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "<thinking>Verify conflicting sub results against the cited evidence and answer only what is supported.</thinking>"
                    f"<result>{task.answer} | evidence: {evidence}</result>"
                ),
            },
        ],
        "category": "main",
        "stage": f"answer_dynamic_verifier_{variant}",
        "task_type": task.task_type,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Main-only dynamic verifier SFT data.")
    parser.add_argument("--train-jsonl", default="./data/enhanced/train.jsonl")
    parser.add_argument("--output", default="data/sft/hotpotqa_dynamic_verifier_sft_data.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--max-subtasks", type=int, default=2)
    parser.add_argument("--max-snippet-chars", type=int, default=360)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    variants = ["gold_only", "wrong_first", "distractor_extra", "partial_plus_distractor"]
    env = HotpotQAEnvironment.from_jsonl(args.train_jsonl, limit=args.limit)
    samples = []
    for idx, task in enumerate(env.tasks):
        offset = idx % len(variants)
        for sample_idx in range(args.samples_per_task):
            variant = variants[(offset + sample_idx) % len(variants)]
            samples.append(
                build_verifier_sample(task, rng, variant, args.max_subtasks, args.max_snippet_chars)
            )

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[hotpotqa-dynamic-verifier-sft] wrote {len(samples)} samples to {out}")
    print("[hotpotqa-dynamic-verifier-sft] main={}".format(sum(1 for s in samples if s["category"] == "main")))
    print("[hotpotqa-dynamic-verifier-sft] sub={}".format(sum(1 for s in samples if s["category"] == "sub")))


if __name__ == "__main__":
    main()

