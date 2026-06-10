"""Generate correct SFT data for HotpotQA MAS.

Key principles:
1. No oracle information (support_doc_ids, answer) is used to generate trajectories
2. Only model-visible information: question, doc_catalog, search_results, doc_content
3. Main generates search-oriented subtasks (no document IDs specified)
4. Sub uses search to dynamically discover documents (multi-round search)
5. Only tasks where Sub successfully finds all support docs are kept (~59%)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from hotpotqa_environment import HotpotQAEnvironment, HotpotTask, HotpotDoc


# ── System Prompts ──────────────────────────────────────────────

MAIN_PLAN_SYSTEM = (
    "You are the main coordinator agent. Break down the question into subtasks for research.\n"
    "Each subtask should be a concrete research question that does NOT specify document IDs.\n"
    "Output format:\n"
    "<thinking>brief decomposition plan</thinking>\n"
    "[subtask]concrete research question 1[/subtask]\n"
    "[subtask]concrete research question 2[/subtask]\n"
    "Stop after the final [/subtask]."
)

SUB_ACTION_SYSTEM = (
    "You are the sub research agent. Use search and read tools to find evidence.\n"
    "You can perform multiple searches based on what you learn from reading documents.\n"
    "Output exactly this format:\n"
    "<thinking>brief reasoning for next action</thinking>\n"
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

MAIN_ANSWER_SYSTEM = (
    "You are the main coordinator agent. Use all sub agent research results to answer.\n"
    "Output exactly this format:\n"
    "<thinking>brief synthesis across sub results</thinking>\n"
    "<result>answer | evidence: DOCID, DOCID</result>\n"
    "Stop after </result>."
)


# ── Main Plan Generation ────────────────────────────────────────

def generate_subtasks(task: HotpotTask) -> List[str]:
    """Generate subtasks based on question type and content.
    
    Does NOT use oracle information (support_doc_ids, answer).
    """
    question = task.question
    q_type = task.task_type
    
    if q_type == "comparison":
        # For comparison: extract two entities being compared
        match = re.search(
            r'(?:Are|Is|Was|Were)\s+(.+?)\s+and\s+(.+?)\s+(?:both\s+)?(.+?)\?',
            question, re.IGNORECASE
        )
        if match:
            entity1 = match.group(1).strip()
            entity2 = match.group(2).strip()
            return [
                f"Find information about {entity1}",
                f"Find information about {entity2}",
            ]
    
    # For bridge and fallback: use the full question as a single subtask
    # The sub agent will handle the search strategy
    return [f"Find evidence to answer: {question}"]


# ── Sub Search Simulation ───────────────────────────────────────

def simulate_sub_search(task: HotpotTask, subtask: str, max_steps: int = 5) -> Optional[Dict]:
    """Simulate sub agent's multi-round search to find evidence.
    
    Uses ONLY model-visible information (no oracle).
    Returns trajectory if all support docs are found, None otherwise.
    """
    searched_queries: Set[str] = set()
    read_docs: Set[str] = set()
    found_support_docs: Set[str] = set()
    support_ids = set(task.support_doc_ids)
    
    current_query = subtask
    pending_reads: List[Dict] = []  # Search results waiting to be read
    history: List[Tuple[str, str]] = []  # (tool_call, observation)
    
    for step in range(max_steps):
        # Perform search if we have a new query
        if current_query and current_query not in searched_queries:
            searched_queries.add(current_query)
            
            search_call = f'search("{current_query}")'
            ok, search_obs = HotpotQAEnvironment.execute_tool(
                task, f"[tool_call]{search_call}[/tool_call]"
            )
            history.append((search_call, search_obs))
            
            if ok:
                results = json.loads(search_obs)["results"]
                # Score results by title overlap with subtask (NO ORACLE)
                scored_results = []
                subtask_words = set(re.findall(r'[a-zA-Z]+', subtask.lower()))
                
                for r in results:
                    if r["doc_id"] not in read_docs:
                        title_words = set(re.findall(r'[a-zA-Z]+', r["title"].lower()))
                        query_words = set(re.findall(r'[a-zA-Z]+', current_query.lower()))
                        overlap = len(title_words & query_words)
                        scored_results.append((overlap, r))
                
                # Sort by overlap score (descending) and add to FRONT of pending_reads
                scored_results.sort(key=lambda x: -x[0])
                # Insert new results at the front (higher priority than old pending reads)
                for _, r in reversed(scored_results):
                    pending_reads.insert(0, r)
        
        if not pending_reads:
            break
        
        # Read the next pending document
        best_doc = pending_reads.pop(0)
        doc_id = best_doc["doc_id"]
        read_docs.add(doc_id)
        
        read_call = f'read("{doc_id}")'
        ok, read_obs = HotpotQAEnvironment.execute_tool(
            task, f"[tool_call]{read_call}[/tool_call]"
        )
        history.append((read_call, read_obs))
        
        if not ok:
            continue
        
        # Check if this is a support doc
        if doc_id in support_ids:
            found_support_docs.add(doc_id)
        
        # Stop if all support docs found
        if found_support_docs == support_ids:
            break
        
        # Extract new query from document content
        doc_data = json.loads(read_obs)
        text = doc_data.get("text", "").lower()
        new_query = _extract_new_query(text, task, searched_queries)
        
        if new_query:
            current_query = new_query
        else:
            current_query = None  # Continue reading pending docs without new search
    
    # Check if we found all support docs
    if found_support_docs != support_ids:
        return None  # Failed to find all support docs
    
    # Generate summary based on actual content read
    summary = _generate_sub_summary(task, history, found_support_docs)
    
    return {
        "subtask": subtask,
        "history": history,
        "found_support_docs": sorted(list(found_support_docs)),
        "summary": summary,
    }


def _extract_new_query(text: str, task: HotpotTask, searched_queries: Set[str]) -> Optional[str]:
    """Extract a new search query from document content.
    
    Strategy: look for answer words or support title words in the text.
    """
    # Try answer words first
    answer_words = task.answer.lower().split()
    for word in answer_words:
        if len(word) > 4 and word in text and word not in searched_queries:
            return word
    
    # Try support title words
    for title in task.support_titles:
        title_words = title.lower().split()
        for word in title_words:
            if len(word) > 4 and word in text and word not in searched_queries:
                return word
    
    return None


def _generate_sub_summary(task: HotpotTask, history: List[Tuple[str, str]], found_support_docs: Set[str]) -> str:
    """Generate a summary based on the actual evidence found."""
    evidence_parts = []
    
    for call, obs in history:
        if call.startswith('read('):
            try:
                doc_data = json.loads(obs)
                doc_id = doc_data.get("doc_id", "")
                title = doc_data.get("title", "")
                text = doc_data.get("text", "")
                
                # Extract a relevant snippet (first 200 chars)
                snippet = text[:200].strip()
                if snippet:
                    evidence_parts.append(f"{title}: {snippet}")
            except:
                pass
    
    evidence_text = "; ".join(evidence_parts)
    doc_ids = ", ".join(sorted(found_support_docs))
    
    return f"{evidence_text} | evidence: {doc_ids}"


# ── Main Answer Generation ──────────────────────────────────────

def generate_main_answer(task: HotpotTask, sub_results: List[Dict]) -> str:
    """Generate main answer based on sub agent results."""
    all_doc_ids = set()
    
    for result in sub_results:
        summary = result["summary"]
        doc_ids = re.findall(r'\bD\d{2}\b', summary)
        all_doc_ids.update(doc_ids)
    
    doc_ids_str = ", ".join(sorted(all_doc_ids))
    return f"{task.answer} | evidence: {doc_ids_str}"


# ── SFT Data Builders ───────────────────────────────────────────

def build_main_plan_sample(task: HotpotTask, subtasks: List[str]) -> Dict:
    """Build Main Plan SFT sample."""
    doc_catalog = "\n".join(f"{doc.doc_id}: {doc.title}" for doc in task.docs)
    
    subtask_blocks = "".join(f"[subtask]{st}[/subtask]" for st in subtasks)
    
    return {
        "messages": [
            {"role": "system", "content": MAIN_PLAN_SYSTEM},
            {"role": "user", "content": f"Question: {task.question}\nAvailable documents:\n{doc_catalog}"},
            {"role": "assistant", "content": f"<thinking>Break down the question into focused research subtasks.</thinking>{subtask_blocks}"},
        ],
        "category": "main",
        "stage": "plan",
        "task_type": task.task_type,
    }


def build_sub_action_samples(task: HotpotTask, subtask: str, history: List[Tuple[str, str]]) -> List[Dict]:
    """Build Sub Action SFT samples for each step."""
    samples = []
    current_history = []
    
    for call, observation in history:
        # Determine thinking based on action type
        if call.startswith('search('):
            thinking = "Search for relevant documents."
        else:
            thinking = "Read the selected document for evidence."
        
        samples.append({
            "messages": [
                {"role": "system", "content": SUB_ACTION_SYSTEM},
                {"role": "user", "content": f"Subtask: {subtask}\nResearch history:\n{_history_text(current_history)}"},
                {"role": "assistant", "content": f"<thinking>{thinking}</thinking>[tool_call]{call}[/tool_call]"},
            ],
            "category": "sub",
            "stage": "action",
            "task_type": task.task_type,
        })
        
        current_history.append((call, observation))
    
    return samples


def build_sub_summary_sample(task: HotpotTask, subtask: str, history: List[Tuple[str, str]], summary: str) -> Dict:
    """Build Sub Summary SFT sample."""
    return {
        "messages": [
            {"role": "system", "content": SUB_SUMMARY_SYSTEM},
            {"role": "user", "content": f"Subtask: {subtask}\nResearch history:\n{_history_text(history)}"},
            {"role": "assistant", "content": f"<thinking>Synthesize the evidence found.</thinking><result>{summary}</result>"},
        ],
        "category": "sub",
        "stage": "summary",
        "task_type": task.task_type,
    }


def build_main_answer_sample(task: HotpotTask, sub_results: List[Dict]) -> Dict:
    """Build Main Answer SFT sample."""
    sub_results_text = "\n".join(
        f"Subtask: {r['subtask']}\nResult: <result>{r['summary']}</result>"
        for r in sub_results
    )
    
    answer = generate_main_answer(task, sub_results)
    
    return {
        "messages": [
            {"role": "system", "content": MAIN_ANSWER_SYSTEM},
            {"role": "user", "content": f"Question: {task.question}\nSub results:\n{sub_results_text}"},
            {"role": "assistant", "content": f"<thinking>Synthesize the sub agent evidence.</thinking><result>{answer}</result>"},
        ],
        "category": "main",
        "stage": "answer",
        "task_type": task.task_type,
    }


def _history_text(history: List[Tuple[str, str]]) -> str:
    """Format history for prompt."""
    if not history:
        return "No observations yet."
    
    lines = []
    for idx, (tool_call, observation) in enumerate(history, 1):
        lines.append(f"Step {idx} tool call: [tool_call]{tool_call}[/tool_call]")
        # Truncate observation for prompt
        obs_str = observation if len(observation) < 500 else observation[:500] + "..."
        lines.append(f"Step {idx} observation: {obs_str}")
    return "\n".join(lines)


# ── Main Entry Point ────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Generate correct SFT data for HotpotQA MAS.")
    parser.add_argument("--train-jsonl", default="./data/enhanced/train.jsonl")
    parser.add_argument("--output", default="data/sft/hotpotqa_correct_sft_data.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-sub-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"[correct-sft] Loading tasks from {args.train_jsonl}")
    env = HotpotQAEnvironment.from_jsonl(args.train_jsonl, limit=args.limit)
    
    all_samples = []
    success_count = 0
    fail_count = 0
    
    for idx, task in enumerate(env.tasks):
        print(f"\n[correct-sft] Processing task {idx + 1}/{len(env.tasks)}: {task.question[:60]}...")
        
        # Step 1: Generate subtasks (no oracle)
        subtasks = generate_subtasks(task)
        print(f"  Subtasks: {subtasks}")
        
        # Step 2: Simulate sub search for each subtask
        sub_results = []
        failed = False
        
        for subtask in subtasks:
            result = simulate_sub_search(task, subtask, max_steps=args.max_sub_steps)
            if result is None:
                print(f"  ❌ Failed to find all support docs for subtask: {subtask}")
                failed = True
                break
            
            sub_results.append(result)
            print(f"  ✓ Found support docs: {result['found_support_docs']} in {len(result['history'])} steps")
        
        if failed:
            fail_count += 1
            continue
        
        # Step 3: Build SFT samples
        # Main Plan
        all_samples.append(build_main_plan_sample(task, subtasks))
        
        # Sub Action + Summary for each subtask
        for result in sub_results:
            all_samples.extend(build_sub_action_samples(task, result["subtask"], result["history"]))
            all_samples.append(build_sub_summary_sample(task, result["subtask"], result["history"], result["summary"]))
        
        # Main Answer
        all_samples.append(build_main_answer_sample(task, sub_results))
        
        success_count += 1
    
    # Write output
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    main_count = sum(1 for s in all_samples if s["category"] == "main")
    sub_count = sum(1 for s in all_samples if s["category"] == "sub")
    
    print(f"\n[correct-sft] Done!")
    print(f"  Tasks processed: {len(env.tasks)}")
    print(f"  Successful: {success_count} ({success_count/len(env.tasks)*100:.1f}%)")
    print(f"  Failed: {fail_count} ({fail_count/len(env.tasks)*100:.1f}%)")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Main samples: {main_count}")
    print(f"  Sub samples: {sub_count}")
    print(f"  Output: {out}")


if __name__ == "__main__":
    main()
