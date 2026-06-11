#!/usr/bin/env python3
"""
Multi-Agent Collaboration Evaluation Script for HotpotQA-MAS
===========================================================

Comprehensive evaluation of trained Main/Sub multi-agent systems on HotpotQA.

Evaluates:
  - Answer quality (EM, F1)
  - Collaboration effectiveness (delegation rate, subtask success)
  - Evidence retrieval accuracy (support doc recall)
  - Token efficiency (per-episode cost)
  - Failure mode analysis (error taxonomy)

Usage:
    # Evaluate with trained LoRA adapters
    python evaluate_mas.py \
        --main-lora artifacts/checkpoints/sft/main_agent \
        --sub-lora artifacts/checkpoints/sft/sub_agent \
        --data data/enhanced/val.jsonl \
        --tasks 100

    # Evaluate base model (no adapters)
    python evaluate_mas.py \
        --base-model Qwen/Qwen3.5-9B \
        --data data/base/val.jsonl \
        --tasks 50

    # Ablation: Main-only (no subagent)
    python evaluate_mas.py \
        --main-lora artifacts/checkpoints/sft/main_agent \
        --no-subagent \
        --data data/enhanced/val.jsonl
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import TrainingConfig
from hotpotqa_environment import HotpotQAEnvironment, HotpotTask, HotpotDoc


# ═══════════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeResult:
    """Result of a single evaluation episode."""
    task_id: str
    question: str
    gold_answer: str
    predicted_answer: str = ""
    
    # Metrics
    exact_match: bool = False
    f1_score: float = 0.0
    
    # Collaboration tracking
    delegated: bool = False
    num_subtasks: int = 0
    subtask_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Evidence tracking
    gold_doc_ids: List[str] = field(default_factory=list)
    predicted_doc_ids: List[str] = field(default_factory=list)
    doc_recall: float = 0.0
    doc_precision: float = 0.0
    
    # Token tracking
    main_tokens: int = 0
    sub_tokens: int = 0
    total_tokens: int = 0
    
    # Timing
    main_latency_ms: float = 0.0
    sub_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    
    # Error analysis
    failure_mode: str = ""  # none, wrong_answer, wrong_evidence, no_evidence, subagent_failed, main_synthesis_failed
    
    # Raw outputs for debugging
    main_plan_output: str = ""
    main_answer_output: str = ""
    sub_outputs: List[str] = field(default_factory=list)


@dataclass
class EvaluationSummary:
    """Aggregated evaluation results."""
    # Overall
    total_tasks: int = 0
    exact_match_count: int = 0
    exact_match_rate: float = 0.0
    avg_f1: float = 0.0
    
    # Collaboration
    delegation_rate: float = 0.0
    avg_subtasks_per_delegated: float = 0.0
    subtask_success_rate: float = 0.0
    
    # Evidence
    avg_doc_recall: float = 0.0
    avg_doc_precision: float = 0.0
    
    # Efficiency
    avg_total_tokens: float = 0.0
    avg_main_tokens: float = 0.0
    avg_sub_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Per-difficulty breakdown
    by_difficulty: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Per-type breakdown
    by_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Failure mode breakdown
    failure_modes: Dict[str, int] = field(default_factory=dict)
    
    # Episode-level results
    episodes: List[EpisodeResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(base_model_path: str, adapter_path: Optional[str] = None,
               device: str = "cuda") -> Tuple[Any, Any]:
    """Load base model with optional LoRA adapter."""
    # Resolve to local path if available
    from config import _find_local_model
    local_path = _find_local_model(base_model_path)
    if local_path != base_model_path:
        print(f"[system] Resolved base model to local path: {local_path}")
        base_model_path = local_path
    
    print(f"[system] Loading base model: {base_model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Use single GPU for inference (avoid multi-GPU communication overhead)
    device_map = {"": 4}  # GPU 4 has enough memory for Qwen3.5-9B + LoRA
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        device_map=device_map,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    
    if adapter_path and os.path.exists(adapter_path):
        print(f"[system] Loading adapter: {adapter_path}")
        # Use a unique adapter name based on path to avoid conflicts
        adapter_name = Path(adapter_path).name
        model = PeftModel.from_pretrained(model, adapter_path, adapter_name=adapter_name)
        model.set_adapter(adapter_name)
    
    model.eval()
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
#  Generation Utilities
# ═══════════════════════════════════════════════════════════════════════════

def generate(model, tokenizer, prompt: str, system: str = "",
             max_new_tokens: int = 256, temperature: float = 0.0) -> Tuple[str, int]:
    """Generate text and return (output, token_count)."""
    if system:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    
    # Move to model device (single GPU)
    embed_device = model.get_input_embeddings().weight.device
    inputs = {k: v.to(embed_device) for k, v in inputs.items()}
    
    input_tokens = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated = tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)
    output_tokens = outputs[0].shape[0] - input_tokens
    
    return generated.strip(), output_tokens


# ═══════════════════════════════════════════════════════════════════════════
#  Parsing Utilities
# ═══════════════════════════════════════════════════════════════════════════

def extract_mode(text: str) -> str:
    """Extract [mode]direct[/mode] or [mode]delegate[/mode].
    Falls back to keyword detection if tags not found."""
    m = re.search(r'\[mode\](\w+)\[/mode\]', text, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    
    # Fallback: detect keywords in text
    lower = text.lower()
    if "delegate" in lower or "subtask" in lower or "research" in lower:
        return "delegate"
    elif "direct" in lower or "answer directly" in lower:
        return "direct"
    return "delegate"  # Default to delegate for complex questions


def extract_subtasks(text: str) -> List[str]:
    """Extract [subtask]...[/subtask] blocks.
    Falls back to bullet points or numbered lists."""
    subtasks = re.findall(r'\[subtask\](.*?)\[/subtask\]', text, re.DOTALL)
    if subtasks:
        return [s.strip() for s in subtasks]
    
    # Fallback: look for bullet points or numbered items that look like subtasks
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        # Match bullet points or numbered items
        if re.match(r'^[\*\-\d\.\)]\s+', line) and len(line) > 10:
            subtasks.append(re.sub(r'^[\*\-\d\.\)\s]+', '', line).strip())
    
    return subtasks


def extract_result(text: str) -> Tuple[str, List[str]]:
    """Extract <result>answer | evidence: DOCID, DOCID</result>.
    Falls back to extracting the last sentence/line as answer."""
    matches = re.findall(r'<result>\s*(.*?)\s*</result>', text, re.DOTALL)
    if matches:
        result = matches[-1].strip()
        parts = re.split(r'\|\s*evidence\s*:', result, maxsplit=1, flags=re.IGNORECASE)
        answer = parts[0].strip()
        doc_ids = []
        if len(parts) > 1:
            doc_ids = sorted(set(re.findall(r'\bD\d{2}\b', parts[1])))
        return answer, doc_ids
    
    # Fallback: extract last non-empty line as answer
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    # Common instruction/prompt words to skip
    skip_prefixes = (
        'thinking', 'process', 'analyze', 'step', 'summary', 'synthesize',
        '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.',
        '*', '-', '•', 'answer', 'final', 'result', 'conclusion',
        'based on', 'according to', 'the answer is', 'therefore',
    )
    
    if lines:
        # Skip lines that look like instructions or formatting
        for line in reversed(lines):
            lower = line.lower()
            if not lower.startswith(skip_prefixes) and len(line) > 2:
                # Also skip if it's just a single word that's likely a section header
                if len(line.split()) > 1 or len(line) > 15:
                    return line, []
        # If all lines are filtered, return the last one anyway
        return lines[-1], []
    
    return "", []


def extract_tool_call(text: str) -> Optional[Tuple[str, str]]:
    """Extract [tool_call]search("query") or read("DOCID")[/tool_call]."""
    return HotpotQAEnvironment.parse_tool_call(text)


# ═══════════════════════════════════════════════════════════════════════════
#  Scoring Utilities
# ═══════════════════════════════════════════════════════════════════════════

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_doc_metrics(pred_docs: List[str], gold_docs: List[str]) -> Tuple[float, float]:
    """Compute recall and precision for document retrieval."""
    pred_set = set(pred_docs)
    gold_set = set(gold_docs)
    
    if not gold_set:
        return 0.0, 0.0
    
    recall = len(pred_set & gold_set) / len(gold_set)
    precision = len(pred_set & gold_set) / len(pred_set) if pred_set else 0.0
    
    return recall, precision


# ═══════════════════════════════════════════════════════════════════════════
#  System Prompts (from generate_correct_sft_data.py)
# ═══════════════════════════════════════════════════════════════════════════

MAIN_PLAN_SYSTEM = (
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

DIRECT_ANSWER_SYSTEM = (
    "You are the main answer agent. Answer directly when the plan selected direct mode.\n"
    "Output exactly this format:\n"
    "<thinking>brief answer reasoning</thinking>\n"
    "<result>answer | evidence: DOCID, DOCID</result>\n"
    "Stop after </result>."
)


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation Core
# ═══════════════════════════════════════════════════════════════════════════

class MASEvaluator:
    """Evaluator for Multi-Agent System on HotpotQA."""
    
    def __init__(self, main_model, main_tokenizer, sub_model, sub_tokenizer,
                 max_sub_steps: int = 5, max_subtasks: int = 3,
                 main_generate_fn=None, sub_generate_fn=None):
        self.main_model = main_model
        self.main_tokenizer = main_tokenizer
        self.sub_model = sub_model
        self.sub_tokenizer = sub_tokenizer
        self.max_sub_steps = max_sub_steps
        self.max_subtasks = max_subtasks
        self._main_generate = main_generate_fn or generate
        self._sub_generate = sub_generate_fn or generate
    
    def evaluate_task(self, task: HotpotTask, no_subagent: bool = False) -> EpisodeResult:
        """Evaluate a single task."""
        result = EpisodeResult(
            task_id=task.task_id,
            question=task.question,
            gold_answer=task.answer,
            gold_doc_ids=task.support_doc_ids,
        )
        
        # Build document catalog for Main
        doc_catalog = "\n".join([
            f"{doc.doc_id}: {doc.title}" for doc in task.docs
        ])
        
        # ── Stage 1: Main decides direct or delegate ──
        plan_prompt = f"""Question: {task.question}

Document catalog:
{doc_catalog}

Should you answer directly or delegate research?"""
        
        start_time = time.time()
        plan_output, plan_tokens = self._main_generate(
            plan_prompt, system=MAIN_PLAN_SYSTEM,
            max_new_tokens=256
        )
        result.main_plan_output = plan_output
        result.main_tokens += plan_tokens
        result.main_latency_ms = (time.time() - start_time) * 1000
        
        mode = extract_mode(plan_output)
        
        if mode == "direct" or no_subagent:
            # Direct answer
            result.delegated = False
            answer_prompt = f"""Question: {task.question}

Document catalog:
{doc_catalog}

Answer directly."""
            
            start_time = time.time()
            answer_output, answer_tokens = self._main_generate(
                answer_prompt, system=DIRECT_ANSWER_SYSTEM,
                max_new_tokens=256
            )
            result.main_answer_output = answer_output
            result.main_tokens += answer_tokens
            result.main_latency_ms += (time.time() - start_time) * 1000
            
            result.predicted_answer, result.predicted_doc_ids = extract_result(answer_output)
            
        else:
            # Delegate to subagent
            result.delegated = True
            subtasks = extract_subtasks(plan_output)[:self.max_subtasks]
            result.num_subtasks = len(subtasks)
            
            # ── Stage 2: Subagent executes each subtask ──
            sub_results = []
            all_sub_tokens = 0
            
            for subtask_desc in subtasks:
                sub_result = self._execute_subtask(task, subtask_desc)
                sub_results.append(sub_result)
                all_sub_tokens += sub_result.get("tokens", 0)
            
            result.subtask_results = sub_results
            result.sub_tokens = all_sub_tokens
            
            # ── Stage 3: Main synthesizes final answer ──
            sub_outputs_text = "\n\n".join([
                f"Subtask: {sr.get('subtask', '')}\nResult: {sr.get('output', '')}"
                for sr in sub_results
            ])
            
            synthesis_prompt = f"""Question: {task.question}

Subagent research results:
{sub_outputs_text}

Synthesize the final answer."""
            
            start_time = time.time()
            answer_output, answer_tokens = self._main_generate(
                synthesis_prompt, system=MAIN_ANSWER_SYSTEM,
                max_new_tokens=256
            )
            result.main_answer_output = answer_output
            result.main_tokens += answer_tokens
            result.main_latency_ms += (time.time() - start_time) * 1000
            
            result.predicted_answer, result.predicted_doc_ids = extract_result(answer_output)
        
        # ── Compute metrics ──
        result.exact_match = exact_match(result.predicted_answer, task.answer)
        result.f1_score = f1_score(result.predicted_answer, task.answer)
        result.doc_recall, result.doc_precision = compute_doc_metrics(
            result.predicted_doc_ids, task.support_doc_ids
        )
        
        result.total_tokens = result.main_tokens + result.sub_tokens
        result.total_latency_ms = result.main_latency_ms + result.sub_latency_ms
        
        # ── Failure mode analysis ──
        result.failure_mode = self._classify_failure(result)
        
        return result
    
    def _execute_subtask(self, task: HotpotTask, subtask_desc: str) -> Dict[str, Any]:
        """Execute a single subtask with the subagent."""
        sub_prompt = f"""Research task: {subtask_desc}

Question: {task.question}

Use search and read tools to find evidence."""
        
        # Simulate subagent with tool use
        conversation = []
        total_tokens = 0
        
        for step in range(self.max_sub_steps):
            start_time = time.time()
            output, tokens = self._sub_generate(
                sub_prompt + "\n\n" + "\n".join(conversation),
                system=SUB_ACTION_SYSTEM,
                max_new_tokens=128
            )
            total_tokens += tokens
            
            conversation.append(f"Step {step + 1}: {output}")
            
            # Check for tool call
            tool_call = extract_tool_call(output)
            if tool_call:
                tool, arg = tool_call
                success, tool_result = HotpotQAEnvironment.execute_tool(task, output)
                conversation.append(f"Tool result: {tool_result}")
                
                # Check if subagent wants to summarize
                if "summary" in output.lower() or "result" in output.lower():
                    break
            else:
                # No tool call, might be summary
                break
        
        # Final summary
        start_time = time.time()
        summary_output, summary_tokens = self._sub_generate(
            sub_prompt + "\n\n" + "\n".join(conversation) + "\n\nSummarize your findings.",
            system=SUB_SUMMARY_SYSTEM,
            max_new_tokens=128
        )
        total_tokens += summary_tokens
        
        return {
            "subtask": subtask_desc,
            "output": summary_output,
            "conversation": conversation,
            "tokens": total_tokens,
        }
    
    def _classify_failure(self, result: EpisodeResult) -> str:
        """Classify the failure mode of an episode."""
        if result.exact_match:
            return "none"
        
        if result.delegated:
            # Check if subagent found any evidence
            has_subagent_output = any(
                sr.get("output", "") for sr in result.subtask_results
            )
            if not has_subagent_output:
                return "subagent_failed"
            
            # Check if main used subagent results correctly
            if result.predicted_answer and result.f1_score < 0.3:
                return "main_synthesis_failed"
        
        # Check evidence quality
        if result.doc_recall == 0:
            return "no_evidence"
        elif result.doc_recall < 0.5:
            return "wrong_evidence"
        
        return "wrong_answer"


# ═══════════════════════════════════════════════════════════════════════════
#  Main Evaluation Loop
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_dataset(
    data_path: str,
    main_model, main_tokenizer,
    sub_model, sub_tokenizer,
    max_tasks: Optional[int] = None,
    no_subagent: bool = False,
    max_sub_steps: int = 5,
    max_subtasks: int = 3,
) -> EvaluationSummary:
    """Evaluate on a dataset."""
    
    # Load tasks
    tasks = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line.strip())
            docs = [
                HotpotDoc(
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    text=doc["text"],
                    sentences=doc.get("sentences", []),
                )
                for doc in raw["docs"]
            ]
            tasks.append(HotpotTask(
                task_id=raw["task_id"],
                question=raw["question"],
                answer=raw["answer"],
                support_doc_ids=raw.get("support_doc_ids", []),
                support_titles=raw.get("support_titles", []),
                docs=docs,
                level=raw.get("level", ""),
                task_type=raw.get("type", ""),
            ))
    
    if max_tasks:
        tasks = tasks[:max_tasks]
    
    print(f"[system] Evaluating on {len(tasks)} tasks...")
    
    # Initialize evaluator
    evaluator = MASEvaluator(
        main_model, main_tokenizer,
        sub_model, sub_tokenizer,
        max_sub_steps=max_sub_steps,
        max_subtasks=max_subtasks,
    )
    
    # Run evaluation
    summary = EvaluationSummary()
    summary.total_tasks = len(tasks)
    
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Task {task.task_id}: {task.question[:60]}...")
        
        result = evaluator.evaluate_task(task, no_subagent=no_subagent)
        summary.episodes.append(result)
        
        # Print result
        status = "✓" if result.exact_match else "✗"
        print(f"  {status} Pred: {result.predicted_answer or '(empty)'} | Gold: {task.answer}")
        print(f"     F1: {result.f1_score:.2f} | EM: {result.exact_match}")
        print(f"     Delegated: {result.delegated} | Subtasks: {result.num_subtasks}")
        print(f"     Docs: R={result.doc_recall:.2f} P={result.doc_precision:.2f}")
        print(f"     Tokens: {result.total_tokens} | Latency: {result.total_latency_ms:.0f}ms")
        
        if result.failure_mode != "none":
            print(f"     Failure: {result.failure_mode}")
    
    # Aggregate results
    _aggregate_results(summary)
    
    return summary


def _aggregate_results(summary: EvaluationSummary):
    """Aggregate episode results into summary statistics."""
    episodes = summary.episodes
    n = len(episodes)
    if n == 0:
        return
    
    # Overall metrics
    summary.exact_match_count = sum(1 for e in episodes if e.exact_match)
    summary.exact_match_rate = summary.exact_match_count / n
    summary.avg_f1 = sum(e.f1_score for e in episodes) / n
    
    # Collaboration metrics
    delegated_episodes = [e for e in episodes if e.delegated]
    summary.delegation_rate = len(delegated_episodes) / n
    summary.avg_subtasks_per_delegated = (
        sum(e.num_subtasks for e in delegated_episodes) / len(delegated_episodes)
        if delegated_episodes else 0
    )
    
    # Evidence metrics
    summary.avg_doc_recall = sum(e.doc_recall for e in episodes) / n
    summary.avg_doc_precision = sum(e.doc_precision for e in episodes) / n
    
    # Efficiency metrics
    summary.avg_total_tokens = sum(e.total_tokens for e in episodes) / n
    summary.avg_main_tokens = sum(e.main_tokens for e in episodes) / n
    summary.avg_sub_tokens = sum(e.sub_tokens for e in episodes) / n
    summary.avg_latency_ms = sum(e.total_latency_ms for e in episodes) / n
    
    # Per-difficulty breakdown
    by_diff = defaultdict(list)
    for e in episodes:
        by_diff[e.task_id].append(e)  # Use level if available
    
    # Per-type breakdown
    by_type = defaultdict(list)
    for e in episodes:
        by_type[e.task_id].append(e)  # Use type if available
    
    # Failure modes
    for e in episodes:
        if e.failure_mode:
            summary.failure_modes[e.failure_mode] = summary.failure_modes.get(e.failure_mode, 0) + 1


# ═══════════════════════════════════════════════════════════════════════════
#  Reporting
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(summary: EvaluationSummary):
    """Print formatted evaluation summary."""
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    print(f"\nOverall Performance:")
    print(f"  Tasks evaluated:     {summary.total_tasks}")
    print(f"  Exact Match:         {summary.exact_match_count}/{summary.total_tasks} ({summary.exact_match_rate*100:.1f}%)")
    print(f"  Average F1:          {summary.avg_f1*100:.1f}%")
    
    print(f"\nCollaboration:")
    print(f"  Delegation rate:     {summary.delegation_rate*100:.1f}%")
    print(f"  Avg subtasks/deleg:  {summary.avg_subtasks_per_delegated:.1f}")
    
    print(f"\nEvidence Retrieval:")
    print(f"  Avg Doc Recall:      {summary.avg_doc_recall*100:.1f}%")
    print(f"  Avg Doc Precision:   {summary.avg_doc_precision*100:.1f}%")
    
    print(f"\nEfficiency:")
    print(f"  Avg Total Tokens:    {summary.avg_total_tokens:.0f}")
    print(f"  Avg Main Tokens:     {summary.avg_main_tokens:.0f}")
    print(f"  Avg Sub Tokens:      {summary.avg_sub_tokens:.0f}")
    print(f"  Avg Latency:         {summary.avg_latency_ms:.0f}ms")
    
    if summary.failure_modes:
        print(f"\nFailure Mode Breakdown:")
        for mode, count in sorted(summary.failure_modes.items(), key=lambda x: -x[1]):
            print(f"  {mode}: {count}")
    
    print("="*70)


def save_results(summary: EvaluationSummary, output_path: str):
    """Save detailed results to JSON."""
    output = {
        "summary": {
            "total_tasks": summary.total_tasks,
            "exact_match_rate": summary.exact_match_rate,
            "avg_f1": summary.avg_f1,
            "delegation_rate": summary.delegation_rate,
            "avg_subtasks_per_delegated": summary.avg_subtasks_per_delegated,
            "avg_doc_recall": summary.avg_doc_recall,
            "avg_doc_precision": summary.avg_doc_precision,
            "avg_total_tokens": summary.avg_total_tokens,
            "avg_main_tokens": summary.avg_main_tokens,
            "avg_sub_tokens": summary.avg_sub_tokens,
            "avg_latency_ms": summary.avg_latency_ms,
            "failure_modes": summary.failure_modes,
        },
        "episodes": [
            {
                "task_id": e.task_id,
                "question": e.question,
                "gold_answer": e.gold_answer,
                "predicted_answer": e.predicted_answer,
                "exact_match": e.exact_match,
                "f1_score": e.f1_score,
                "delegated": e.delegated,
                "num_subtasks": e.num_subtasks,
                "doc_recall": e.doc_recall,
                "doc_precision": e.doc_precision,
                "total_tokens": e.total_tokens,
                "total_latency_ms": e.total_latency_ms,
                "failure_mode": e.failure_mode,
                "main_plan_output": e.main_plan_output,
                "main_answer_output": e.main_answer_output,
                "subtask_results": e.subtask_results,
            }
            for e in summary.episodes
        ]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n[system] Results saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Multi-Agent System on HotpotQA"
    )
    
    # Model configuration
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B",
                        help="Base model path or HF model id")
    parser.add_argument("--main-lora", default=None,
                        help="Path to Main agent LoRA adapter")
    parser.add_argument("--sub-lora", default=None,
                        help="Path to Sub agent LoRA adapter")
    parser.add_argument("--no-subagent", action="store_true",
                        help="Evaluate Main-only (no subagent delegation)")
    
    # Data configuration
    parser.add_argument("--data", default="data/enhanced/val.jsonl",
                        help="Path to evaluation data (jsonl)")
    parser.add_argument("--tasks", type=int, default=None,
                        help="Number of tasks to evaluate (default: all)")
    
    # Evaluation configuration
    parser.add_argument("--max-sub-steps", type=int, default=5,
                        help="Max tool use steps per subtask")
    parser.add_argument("--max-subtasks", type=int, default=3,
                        help="Max subtasks per episode")
    
    # Output
    parser.add_argument("--output", default=None,
                        help="Path to save detailed results JSON")
    parser.add_argument("--device", default="cuda",
                        help="Device for inference")
    
    args = parser.parse_args()
    
    # Load models
    print("="*70)
    print("Loading Models")
    print("="*70)
    
    # Load base model ONCE
    base_model, tokenizer = load_model(args.base_model, None, args.device)
    
    # Load first adapter to create PeftModel, then load second adapter
    from peft import PeftModel
    
    # Load Main adapter first (creates PeftModel wrapper)
    if args.main_lora and os.path.exists(args.main_lora):
        print(f"[system] Loading Main adapter: {args.main_lora}")
        main_adapter_name = Path(args.main_lora).name
        base_model = PeftModel.from_pretrained(base_model, args.main_lora, adapter_name=main_adapter_name)
    else:
        main_adapter_name = None
    
    # Load Sub adapter on same PeftModel
    if not args.no_subagent and args.sub_lora and os.path.exists(args.sub_lora):
        print(f"[system] Loading Sub adapter: {args.sub_lora}")
        sub_adapter_name = Path(args.sub_lora).name
        base_model.load_adapter(args.sub_lora, adapter_name=sub_adapter_name)
    else:
        sub_adapter_name = None
    
    base_model.eval()
    
    # Create wrapper functions that switch adapters
    def main_generate(*args, **kwargs):
        if main_adapter_name:
            base_model.set_adapter(main_adapter_name)
        return generate(base_model, tokenizer, *args, **kwargs)
    
    def sub_generate(*args, **kwargs):
        if sub_adapter_name:
            base_model.set_adapter(sub_adapter_name)
        else:
            base_model.set_adapter(main_adapter_name)
        return generate(base_model, tokenizer, *args, **kwargs)
    
    # Monkey-patch the generate function for evaluator
    import evaluate_mas
    evaluate_mas.generate = main_generate  # Main uses main adapter by default
    
    # Run evaluation
    print("\n" + "="*70)
    print("Starting Evaluation")
    print("="*70)
    
    # Load tasks
    tasks = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line.strip())
            docs = [
                HotpotDoc(
                    doc_id=doc["doc_id"],
                    title=doc["title"],
                    text=doc["text"],
                    sentences=doc.get("sentences", []),
                )
                for doc in raw["docs"]
            ]
            tasks.append(HotpotTask(
                task_id=raw["task_id"],
                question=raw["question"],
                answer=raw["answer"],
                support_doc_ids=raw.get("support_doc_ids", []),
                support_titles=raw.get("support_titles", []),
                docs=docs,
                level=raw.get("level", ""),
                task_type=raw.get("type", ""),
            ))
    
    if args.tasks:
        tasks = tasks[:args.tasks]
    
    print(f"[system] Evaluating on {len(tasks)} tasks...")
    
    # Initialize evaluator with adapter-switching generate
    evaluator = MASEvaluator(
        base_model, tokenizer,
        base_model, tokenizer,
        max_sub_steps=args.max_sub_steps,
        max_subtasks=args.max_subtasks,
        main_generate_fn=main_generate,
        sub_generate_fn=sub_generate,
    )
    
    # Run episodes
    summary = EvaluationSummary()
    summary.total_tasks = len(tasks)
    
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Task {task.task_id}: {task.question[:60]}...")
        
        result = evaluator.evaluate_task(task, no_subagent=args.no_subagent)
        summary.episodes.append(result)
        
        status = "✓" if result.exact_match else "✗"
        print(f"  {status} Pred: {result.predicted_answer or '(empty)'} | Gold: {task.answer}")
        print(f"     F1: {result.f1_score:.2f} | EM: {result.exact_match}")
        print(f"     Delegated: {result.delegated} | Subtasks: {result.num_subtasks}")
        print(f"     Docs: R={result.doc_recall:.2f} P={result.doc_precision:.2f}")
        print(f"     Tokens: {result.total_tokens} | Latency: {result.total_latency_ms:.0f}ms")
        
        if result.failure_mode:
            print(f"     Failure: {result.failure_mode}")
    
    # Aggregate results
    _aggregate_results(summary)
    
    # Print and save results
    print_summary(summary)
    
    if args.output:
        save_results(summary, args.output)
    
    print("\n[system] Evaluation complete!")


if __name__ == "__main__":
    main()
