"""Generate replay/mixture SFT data for dynamic HotpotQA MAS.

The mixture preserves the old fixed MAS protocol while adding dynamic
multi-subtask routing and focused Sub action examples.
"""

import argparse
import json
from pathlib import Path

from generate_hotpotqa_dynamic_mas_sft_data import (
    build_main_plan_sample as build_dynamic_main_plan_sample,
    DYNAMIC_MAIN_ANSWER_SYSTEM,
    oracle_subtasks,
    should_direct,
)
from generate_hotpotqa_mas_sft_data import (
    SUB_ACTION_SYSTEM,
    SUB_SUMMARY_SYSTEM,
    build_main_answer_sample as build_fixed_main_answer_sample,
    build_main_plan_sample as build_fixed_main_plan_sample,
    build_sub_action_samples as build_fixed_sub_action_samples,
    build_sub_summary_sample as build_fixed_sub_summary_sample,
    history_text,
)
from hotpotqa_environment import HotpotQAEnvironment, HotpotTask


def focused_call_plan(task: HotpotTask, doc_id: str, title: str):
    return [
        f'search("{title}")',
        f'read("{doc_id}")',
    ]


def focused_history(task: HotpotTask, calls):
    history = []
    for call in calls:
        ok, observation = HotpotQAEnvironment.execute_tool(task, f"[tool_call]{call}[/tool_call]")
        history.append((call, observation if ok else "Tool execution failed"))
    return history


def build_focused_sub_action_samples(task: HotpotTask, max_subtasks: int):
    samples = []
    subtasks = oracle_subtasks(task, max_subtasks)
    for subtask, doc_id, title in zip(subtasks, task.support_doc_ids, task.support_titles):
        history = []
        for step, call in enumerate(focused_call_plan(task, doc_id, title), 1):
            thinking = "Search for the focused evidence page." if step == 1 else "Read the focused evidence page."
            samples.append(
                {
                    "messages": [
                        {"role": "system", "content": SUB_ACTION_SYSTEM},
                        {
                            "role": "user",
                            "content": f"Subtask: {subtask}\nResearch history:\n{history_text(history)}",
                        },
                        {"role": "assistant", "content": f"<thinking>{thinking}</thinking>[tool_call]{call}[/tool_call]"},
                    ],
                    "category": "sub",
                    "stage": "action_dynamic_focused",
                    "task_type": task.task_type,
                }
            )
            ok, observation = HotpotQAEnvironment.execute_tool(task, f"[tool_call]{call}[/tool_call]")
            history.append((call, observation if ok else "Tool execution failed"))
    return samples


def build_focused_sub_summary_samples(task: HotpotTask, max_subtasks: int):
    samples = []
    subtasks = oracle_subtasks(task, max_subtasks)
    for subtask, doc_id, title in zip(subtasks, task.support_doc_ids, task.support_titles):
        calls = focused_call_plan(task, doc_id, title)
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
                            f"<thinking>This focused document contains part of the evidence.</thinking>"
                            f"<result>{task.answer} | evidence: {doc_id}</result>"
                        ),
                    },
                ],
                "category": "sub",
                "stage": "summary_dynamic_focused",
                "task_type": task.task_type,
            }
        )
    return samples


def focused_sub_results_text(task: HotpotTask, max_subtasks: int):
    lines = []
    subtasks = oracle_subtasks(task, max_subtasks)
    for idx, (subtask, doc_id) in enumerate(zip(subtasks, task.support_doc_ids), 1):
        lines.append(f"Subtask {idx}: {subtask}")
        lines.append(f"Sub result {idx}: <result>{task.answer} | evidence: {doc_id}</result>")
    return "\n".join(lines)


def build_focused_dynamic_main_answer_sample(task: HotpotTask, max_subtasks: int):
    evidence = ", ".join(task.support_doc_ids)
    return {
        "messages": [
            {"role": "system", "content": DYNAMIC_MAIN_ANSWER_SYSTEM},
            {
                "role": "user",
                "content": f"Question: {task.question}\nSub results:\n{focused_sub_results_text(task, max_subtasks)}",
            },
            {
                "role": "assistant",
                "content": (
                    "<thinking>Synthesize the focused sub agent evidence.</thinking>"
                    f"<result>{task.answer} | evidence: {evidence}</result>"
                ),
            },
        ],
        "category": "main",
        "stage": "answer_dynamic_focused",
        "task_type": task.task_type,
    }


def include_by_fraction(idx: int, fraction: float) -> bool:
    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    stride = max(int(1 / fraction), 1)
    return idx % stride == 0


def build_samples_for_task(task: HotpotTask, idx: int, args):
    samples = []

    if include_by_fraction(idx, args.fixed_fraction):
        samples.append(build_fixed_main_plan_sample(task))
        if include_by_fraction(idx, args.answer_fraction):
            samples.append(build_fixed_main_answer_sample(task))
            samples.append(build_fixed_sub_summary_sample(task))
        samples.extend(build_fixed_sub_action_samples(task))

    if include_by_fraction(idx, args.dynamic_fraction):
        samples.append(build_dynamic_main_plan_sample(task, idx, args.max_subtasks, args.direct_fraction))
        if include_by_fraction(idx, args.answer_fraction):
            samples.append(build_focused_dynamic_main_answer_sample(task, args.max_subtasks))
        samples.extend(build_focused_sub_action_samples(task, args.max_subtasks))
        samples.extend(build_focused_sub_summary_samples(task, args.max_subtasks))
        if should_direct(task, args.direct_fraction, idx):
            from generate_hotpotqa_dynamic_mas_sft_data import build_direct_answer_sample

            samples.append(build_direct_answer_sample(task))

    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Generate replay/mixture dynamic HotpotQA MAS SFT data.")
    parser.add_argument("--train-jsonl", default="./data/enhanced/train.jsonl")
    parser.add_argument("--output", default="data/sft/hotpotqa_dynamic_mixture_sft_data.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-subtasks", type=int, default=2)
    parser.add_argument("--fixed-fraction", type=float, default=1.0)
    parser.add_argument("--dynamic-fraction", type=float, default=1.0)
    parser.add_argument("--answer-fraction", type=float, default=1.0)
    parser.add_argument("--direct-fraction", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    env = HotpotQAEnvironment.from_jsonl(args.train_jsonl, limit=args.limit)
    samples = []
    for idx, task in enumerate(env.tasks):
        samples.extend(build_samples_for_task(task, idx, args))

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    main_count = sum(1 for sample in samples if sample["category"] == "main")
    sub_count = sum(1 for sample in samples if sample["category"] == "sub")
    stage_counts = {}
    for sample in samples:
        stage_counts[sample.get("stage", "?")] = stage_counts.get(sample.get("stage", "?"), 0) + 1
    print(f"[hotpotqa-dynamic-mixture-sft] wrote {len(samples)} samples to {out}")
    print(f"[hotpotqa-dynamic-mixture-sft] main={main_count}")
    print(f"[hotpotqa-dynamic-mixture-sft] sub={sub_count}")
    print(f"[hotpotqa-dynamic-mixture-sft] stages={json.dumps(stage_counts, sort_keys=True)}")


if __name__ == "__main__":
    main()

