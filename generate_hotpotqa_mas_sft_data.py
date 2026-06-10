"""Generate SFT data for HotpotQA Main/Sub multi-agent research."""

import argparse
import json
from pathlib import Path

from hotpotqa_environment import HotpotQAEnvironment, HotpotTask


MAIN_PLAN_SYSTEM = (
    "You are the main coordinator agent. Delegate the research needed to answer the question.\n"
    "Output exactly this format:\n"
    "<thinking>brief delegation plan</thinking>\n"
    "[subtask]a concrete research request for the sub agent[/subtask]\n"
    "Stop after [/subtask]."
)

MAIN_ANSWER_SYSTEM = (
    "You are the main coordinator agent. Use the sub agent's research result to answer.\n"
    "Output exactly this format:\n"
    "<thinking>brief synthesis</thinking>\n"
    "<result>answer | evidence: DOCID, DOCID</result>\n"
    "Stop after </result>."
)

SUB_ACTION_SYSTEM = (
    "You are the sub research agent. Use search/read tools over the local HotpotQA context.\n"
    "Output exactly this format:\n"
    "<thinking>brief reason for the next action</thinking>\n"
    "[tool_call]search(\"query\") or read(\"DOCID\")[/tool_call]\n"
    "Stop after [/tool_call]."
)

SUB_SUMMARY_SYSTEM = (
    "You are the sub research agent. Summarize the evidence found for the main agent.\n"
    "Output exactly this format:\n"
    "<thinking>brief evidence summary</thinking>\n"
    "<result>answer clue | evidence: DOCID, DOCID</result>\n"
    "Stop after </result>."
)


def history_text(history):
    if not history:
        return "No observations yet."
    lines = []
    for idx, (tool_call, observation) in enumerate(history, 1):
        lines.append(f"Step {idx} tool call: [tool_call]{tool_call}[/tool_call]")
        lines.append(f"Step {idx} observation: {observation}")
    return "\n".join(lines)


def oracle_subtask(task: HotpotTask) -> str:
    return f"Find the supporting documents and answer for: {task.question}"


def oracle_calls(task: HotpotTask):
    calls = [f'search("{task.question}")']
    calls.extend(f'read("{doc_id}")' for doc_id in task.support_doc_ids)
    return calls


def oracle_research_history(task: HotpotTask):
    history = []
    for call in oracle_calls(task):
        ok, observation = HotpotQAEnvironment.execute_tool(task, f"[tool_call]{call}[/tool_call]")
        history.append((call, observation if ok else "Tool execution failed"))
    return history


def build_main_plan_sample(task: HotpotTask):
    subtask = oracle_subtask(task)
    return {
        "messages": [
            {"role": "system", "content": MAIN_PLAN_SYSTEM},
            {"role": "user", "content": f"Question: {task.question}"},
            {
                "role": "assistant",
                "content": f"<thinking>Delegate the evidence search to the sub agent.</thinking>[subtask]{subtask}[/subtask]",
            },
        ],
        "category": "main",
        "stage": "plan",
        "task_type": task.task_type,
    }


def build_main_answer_sample(task: HotpotTask):
    subtask = oracle_subtask(task)
    evidence = ", ".join(task.support_doc_ids)
    sub_result = f"<result>{task.answer} | evidence: {evidence}</result>"
    return {
        "messages": [
            {"role": "system", "content": MAIN_ANSWER_SYSTEM},
            {
                "role": "user",
                "content": f"Question: {task.question}\nSubtask: {subtask}\nSub result: {sub_result}",
            },
            {
                "role": "assistant",
                "content": f"<thinking>Use the sub agent evidence to answer.</thinking><result>{task.answer} | evidence: {evidence}</result>",
            },
        ],
        "category": "main",
        "stage": "answer",
        "task_type": task.task_type,
    }


def build_sub_action_samples(task: HotpotTask):
    samples = []
    subtask = oracle_subtask(task)
    history = []
    for step, call in enumerate(oracle_calls(task), 1):
        thinking = "Search for candidate pages." if step == 1 else "Read a supporting page."
        samples.append({
            "messages": [
                {"role": "system", "content": SUB_ACTION_SYSTEM},
                {
                    "role": "user",
                    "content": f"Subtask: {subtask}\nResearch history:\n{history_text(history)}",
                },
                {"role": "assistant", "content": f"<thinking>{thinking}</thinking>[tool_call]{call}[/tool_call]"},
            ],
            "category": "sub",
            "stage": "action",
            "task_type": task.task_type,
        })
        ok, observation = HotpotQAEnvironment.execute_tool(task, f"[tool_call]{call}[/tool_call]")
        history.append((call, observation if ok else "Tool execution failed"))
    return samples


def build_sub_summary_sample(task: HotpotTask):
    subtask = oracle_subtask(task)
    evidence = ", ".join(task.support_doc_ids)
    return {
        "messages": [
            {"role": "system", "content": SUB_SUMMARY_SYSTEM},
            {
                "role": "user",
                "content": f"Subtask: {subtask}\nResearch history:\n{history_text(oracle_research_history(task))}",
            },
            {
                "role": "assistant",
                "content": f"<thinking>The read documents support the answer.</thinking><result>{task.answer} | evidence: {evidence}</result>",
            },
        ],
        "category": "sub",
        "stage": "summary",
        "task_type": task.task_type,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HotpotQA MAS SFT data.")
    parser.add_argument("--train-jsonl", default="./data/base/train.jsonl")
    parser.add_argument("--output", default="data/sft/hotpotqa_mas_sft_data.jsonl")
    parser.add_argument("--answer-fraction", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    env = HotpotQAEnvironment.from_jsonl(args.train_jsonl, limit=args.limit)
    stride = max(int(1 / max(min(args.answer_fraction, 1.0), 1e-6)), 1)
    samples = []
    for idx, task in enumerate(env.tasks):
        samples.append(build_main_plan_sample(task))
        if args.answer_fraction >= 1.0 or idx % stride == 0:
            samples.append(build_main_answer_sample(task))
            samples.append(build_sub_summary_sample(task))
        samples.extend(build_sub_action_samples(task))

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[hotpotqa-mas-sft] wrote {len(samples)} samples to {out}")
    print(f"[hotpotqa-mas-sft] main={sum(1 for s in samples if s['category'] == 'main')}")
    print(f"[hotpotqa-mas-sft] sub={sum(1 for s in samples if s['category'] == 'sub')}")


if __name__ == "__main__":
    main()

