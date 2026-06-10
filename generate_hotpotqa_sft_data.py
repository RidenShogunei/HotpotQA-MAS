"""Generate Main/Sub SFT data from prepared HotpotQA local-context JSONL."""

import argparse
import json
from pathlib import Path

from hotpotqa_environment import HotpotQAEnvironment, HotpotTask


MAIN_SYSTEM = (
    "You are the main multi-hop research agent. Choose the next tool call needed to answer the question.\n"
    "Output exactly this format:\n"
    "<thinking>brief reason for the next action</thinking>\n"
    "[tool_call]search(\"query\") or read(\"DOCID\")[/tool_call]\n"
    "Stop after [/tool_call]."
)

ANSWER_SYSTEM = (
    "You are the main multi-hop research agent. Use the research history to answer.\n"
    "Output exactly this format:\n"
    "<thinking>brief evidence reasoning</thinking>\n"
    "<result>answer | evidence: DOCID, DOCID</result>\n"
    "Stop after </result>."
)

SUB_SYSTEM = (
    "You are the tool execution agent. Execute the given search/read tool call over the local HotpotQA context.\n"
    "Output exactly this format:\n"
    "<thinking>execute tool</thinking>\n"
    "<result>tool observation</result>"
)


def history_text(history):
    if not history:
        return "No observations yet."
    lines = []
    for idx, (tool_call, observation) in enumerate(history, 1):
        lines.append(f"Step {idx} tool call: [tool_call]{tool_call}[/tool_call]")
        lines.append(f"Step {idx} observation: {observation}")
    return "\n".join(lines)


def oracle_calls(task: HotpotTask):
    calls = [f'search("{task.question}")']
    calls.extend(f'read("{doc_id}")' for doc_id in task.support_doc_ids)
    return calls


def build_main_action_samples(task: HotpotTask):
    samples = []
    history = []
    for step, call in enumerate(oracle_calls(task), 1):
        thinking = "Search for candidate pages." if step == 1 else "Read a supporting page."
        samples.append({
            "messages": [
                {"role": "system", "content": MAIN_SYSTEM},
                {"role": "user", "content": f"Question: {task.question}\nResearch history:\n{history_text(history)}"},
                {"role": "assistant", "content": f"<thinking>{thinking}</thinking>[tool_call]{call}[/tool_call]"},
            ],
            "category": "main",
            "task_type": task.task_type,
            "stage": "action",
        })
        ok, observation = HotpotQAEnvironment.execute_tool(task, f"[tool_call]{call}[/tool_call]")
        if not ok:
            observation = "Tool execution failed"
        history.append((call, observation))
    return samples


def build_main_answer_sample(task: HotpotTask):
    history = []
    for call in oracle_calls(task):
        ok, observation = HotpotQAEnvironment.execute_tool(task, f"[tool_call]{call}[/tool_call]")
        if not ok:
            observation = "Tool execution failed"
        history.append((call, observation))
    evidence = ", ".join(task.support_doc_ids)
    return {
        "messages": [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Question: {task.question}\nResearch history:\n{history_text(history)}"},
            {
                "role": "assistant",
                "content": f"<thinking>Use the read evidence to answer concisely.</thinking><result>{task.answer} | evidence: {evidence}</result>",
            },
        ],
        "category": "main",
        "task_type": task.task_type,
        "stage": "answer",
    }


def build_sub_samples(task: HotpotTask):
    samples = []
    seen = set()
    for call in oracle_calls(task):
        if call in seen:
            continue
        seen.add(call)
        ok, observation = HotpotQAEnvironment.execute_tool(task, f"[tool_call]{call}[/tool_call]")
        if not ok:
            continue
        samples.append({
            "messages": [
                {"role": "system", "content": SUB_SYSTEM},
                {"role": "user", "content": f"[tool_call]{call}[/tool_call]"},
                {"role": "assistant", "content": f"<thinking>execute tool</thinking><result>{observation}</result>"},
            ],
            "category": "sub",
            "task_type": task.task_type,
        })
    return samples


def parse_args():
    parser = argparse.ArgumentParser(description="Generate HotpotQA SFT JSONL.")
    parser.add_argument("--train-jsonl", default="./data/base/train.jsonl")
    parser.add_argument("--output", default="data/sft/hotpotqa_sft_data.jsonl")
    parser.add_argument("--answer-fraction", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    env = HotpotQAEnvironment.from_jsonl(args.train_jsonl, limit=args.limit)
    samples = []
    stride = max(int(1 / max(min(args.answer_fraction, 1.0), 1e-6)), 1)
    for idx, task in enumerate(env.tasks):
        samples.extend(build_main_action_samples(task))
        if args.answer_fraction >= 1.0 or idx % stride == 0:
            samples.append(build_main_answer_sample(task))
        samples.extend(build_sub_samples(task))

    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"[hotpotqa-sft] wrote {len(samples)} samples to {out}")
    print(f"[hotpotqa-sft] main={sum(1 for s in samples if s['category'] == 'main')}")
    print(f"[hotpotqa-sft] sub={sum(1 for s in samples if s['category'] == 'sub')}")


if __name__ == "__main__":
    main()

