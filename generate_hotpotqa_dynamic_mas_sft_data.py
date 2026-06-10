"""Generate SFT data for dynamic HotpotQA MAS routing."""

import argparse
import json
from pathlib import Path

from generate_hotpotqa_mas_sft_data import (
    SUB_ACTION_SYSTEM,
    SUB_SUMMARY_SYSTEM,
    build_sub_action_samples,
    build_sub_summary_sample,
    history_text,
    oracle_calls,
    oracle_research_history,
)
from hotpotqa_environment import HotpotQAEnvironment, HotpotTask


def doc_catalog(task: HotpotTask) -> str:
    return "\n".join(f"{doc.doc_id}: {doc.title}" for doc in task.docs)


DYNAMIC_MAIN_PLAN_SYSTEM = (
    "You are the main coordinator agent. Decide whether to answer directly or delegate research.\n"
    "If no external research is needed, output:\n"
    "<thinking>brief reason</thinking>\n"
    "[mode]direct[/mode]\n"
    "If research is needed, output:\n"
    "<thinking>brief delegation plan</thinking>\n"
    "[mode]delegate[/mode]\n"
    "[subtask]concrete research request 1[/subtask]\n"
    "Optionally add more [subtask]...[/subtask] blocks.\n"
    "Stop after the final [/mode] or [/subtask]."
)

DYNAMIC_MAIN_ANSWER_SYSTEM = (
    "You are the main coordinator agent. Use all sub agent research results to answer.\n"
    "Output exactly this format:\n"
    "<thinking>brief synthesis across sub results</thinking>\n"
    "<result>answer | evidence: DOCID, DOCID</result>\n"
    "Stop after </result>."
)

DYNAMIC_DIRECT_ANSWER_SYSTEM = (
    "You are the main answer agent. Answer directly only when the plan selected direct mode.\n"
    "Output exactly this format:\n"
    "<thinking>brief answer reasoning</thinking>\n"
    "<result>answer | evidence: DOCID, DOCID</result>\n"
    "Stop after </result>."
)


def oracle_subtasks(task: HotpotTask, max_subtasks: int):
    if max_subtasks <= 1 or len(task.support_doc_ids) <= 1:
        return [f"Find the supporting documents and answer for: {task.question}"]
    subtasks = []
    for doc_id, title in zip(task.support_doc_ids, task.support_titles):
        if len(subtasks) >= max_subtasks:
            break
        subtasks.append(f"Find evidence from document {doc_id} ({title}) relevant to: {task.question}")
    return subtasks or [f"Find the supporting documents and answer for: {task.question}"]


def should_direct(task: HotpotTask, direct_fraction: float, idx: int):
    if direct_fraction <= 0:
        return False
    stride = max(int(1 / min(direct_fraction, 1.0)), 1)
    return idx % stride == 0 and task.level == "easy"


def build_main_plan_sample(task: HotpotTask, idx: int, max_subtasks: int, direct_fraction: float):
    if should_direct(task, direct_fraction, idx):
        content = "<thinking>This looks simple enough to answer directly.</thinking>[mode]direct[/mode]"
        stage = "plan_direct"
    else:
        subtasks = oracle_subtasks(task, max_subtasks)
        blocks = "".join(f"[subtask]{subtask}[/subtask]" for subtask in subtasks)
        content = f"<thinking>Delegate focused evidence searches to sub agents.</thinking>[mode]delegate[/mode]{blocks}"
        stage = "plan_delegate"
    return {
        "messages": [
            {"role": "system", "content": DYNAMIC_MAIN_PLAN_SYSTEM},
            {"role": "user", "content": f"Question: {task.question}\nAvailable documents:\n{doc_catalog(task)}"},
            {"role": "assistant", "content": content},
        ],
        "category": "main",
        "stage": stage,
        "task_type": task.task_type,
    }


def sub_results_text(task: HotpotTask, max_subtasks: int):
    subtasks = oracle_subtasks(task, max_subtasks)
    evidence = ", ".join(task.support_doc_ids)
    lines = []
    for idx, subtask in enumerate(subtasks, 1):
        lines.append(f"Subtask {idx}: {subtask}")
        lines.append(f"Sub result {idx}: <result>{task.answer} | evidence: {evidence}</result>")
    return "\n".join(lines)


def build_main_answer_sample(task: HotpotTask, max_subtasks: int):
    evidence = ", ".join(task.support_doc_ids)
    return {
        "messages": [
            {"role": "system", "content": DYNAMIC_MAIN_ANSWER_SYSTEM},
            {
                "role": "user",
                "content": f"Question: {task.question}\nSub results:\n{sub_results_text(task, max_subtasks)}",
            },
            {
                "role": "assistant",
                "content": f"<thinking>Synthesize the sub agent evidence.</thinking><result>{task.answer} | evidence: {evidence}</result>",
            },
        ],
        "category": "main",
        "stage": "answer_dynamic",
        "task_type": task.task_type,
    }


def build_direct_answer_sample(task: HotpotTask):
    evidence = ", ".join(task.support_doc_ids)
    return {
        "messages": [
            {"role": "system", "content": DYNAMIC_DIRECT_ANSWER_SYSTEM},
            {"role": "user", "content": f"Question: {task.question}"},
            {
                "role": "assistant",
                "content": f"<thinking>Answer directly.</thinking><result>{task.answer} | evidence: {evidence}</result>",
            },
        ],
        "category": "main",
        "stage": "answer_direct",
        "task_type": task.task_type,
    }


def build_dynamic_sub_summary_sample(task: HotpotTask):
    evidence = ", ".join(task.support_doc_ids)
    return {
        "messages": [
            {"role": "system", "content": SUB_SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Subtask: Find focused evidence for: {task.question}\n"
                    f"Research history:\n{history_text(oracle_research_history(task))}"
                ),
            },
            {
                "role": "assistant",
                "content": f"<thinking>The read documents support the answer.</thinking><result>{task.answer} | evidence: {evidence}</result>",
            },
        ],
        "category": "sub",
        "stage": "summary_dynamic",
        "task_type": task.task_type,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dynamic HotpotQA MAS SFT data.")
    parser.add_argument("--train-jsonl", default="./data/enhanced/train.jsonl")
    parser.add_argument("--output", default="data/sft/hotpotqa_dynamic_mas_sft_data.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-subtasks", type=int, default=2)
    parser.add_argument("--direct-fraction", type=float, default=0.0)
    parser.add_argument("--answer-fraction", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    env = HotpotQAEnvironment.from_jsonl(args.train_jsonl, limit=args.limit)
    stride = max(int(1 / max(min(args.answer_fraction, 1.0), 1e-6)), 1)
    samples = []
    for idx, task in enumerate(env.tasks):
        samples.append(build_main_plan_sample(task, idx, args.max_subtasks, args.direct_fraction))
        if args.answer_fraction >= 1.0 or idx % stride == 0:
            samples.append(build_main_answer_sample(task, args.max_subtasks))
            if should_direct(task, args.direct_fraction, idx):
                samples.append(build_direct_answer_sample(task))
            samples.append(build_dynamic_sub_summary_sample(task))
        samples.extend(build_sub_action_samples(task))

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[hotpotqa-dynamic-mas-sft] wrote {len(samples)} samples to {out}")
    print(f"[hotpotqa-dynamic-mas-sft] main={sum(1 for s in samples if s['category'] == 'main')}")
    print(f"[hotpotqa-dynamic-mas-sft] sub={sum(1 for s in samples if s['category'] == 'sub')}")


if __name__ == "__main__":
    main()

