"""M-GRPO-style HotpotQA training with Main coordinator and Sub researcher.

Supports:
- Dynamic routing: Main can choose direct answer or delegate 1..N subtasks
- True GRPO policy gradient training (--objective grpo)
- Best-of-N SFT fine-tuning (--objective best_of)
- Wandb experiment tracking (--wandb)
- Checkpoint resume (--resume)
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from config import TrainingConfig
from generate_hotpotqa_mas_sft_data import (
    MAIN_ANSWER_SYSTEM,
    MAIN_PLAN_SYSTEM,
    SUB_ACTION_SYSTEM,
    SUB_SUMMARY_SYSTEM,
)
from grpo_v4 import SharedModel
from hotpotqa_environment import HotpotQAEnvironment, HotpotTask
from utils import dry_run_mode, dry_run_warning, on_shutdown, try_tqdm

logger = logging.getLogger("hotpotqa-mas-grpo")


# ── Dynamic prompt templates ────────────────────────────────────

DIRECT_ANSWER_SYSTEM = (
    "You are the main answer agent. Answer directly when the plan selected direct mode.\n"
    "Output exactly this format:\n"
    "<thinking>brief answer reasoning</thinking>\n"
    "<result>answer | evidence: DOCID, DOCID</result>\n"
    "Stop after </result>."
)

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


class HotpotMASGRPOTrainer:
    def __init__(
        self,
        config: TrainingConfig,
        sub_steps: int = 3,
        best_metric: str = "answer_f1",
        train_main: bool = True,
        train_sub: bool = True,
        sub_reward_mode: str = "summary",
        objective: str = "best_of",
        advantage_clip: float = 2.0,
        min_advantage: float = 0.0,
        max_subtasks: int = 3,
        dynamic_routing: bool = True,
        grpo_policy_clip: float = 0.2,
        grpo_kl_beta: float = 0.01,
    ):
        self.config = config
        self.sub_steps = sub_steps
        self.best_metric = best_metric
        self.train_main = train_main
        self.train_sub = train_sub
        self.sub_reward_mode = sub_reward_mode
        self.objective = objective
        self.advantage_clip = advantage_clip
        self.min_advantage = min_advantage
        self.max_subtasks = max_subtasks
        self.dynamic_routing = dynamic_routing
        self.grpo_policy_clip = grpo_policy_clip
        self.grpo_kl_beta = grpo_kl_beta
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model: Optional[SharedModel] = None
        self._wandb = None
        self._setup_logging()

    def _setup_logging(self):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def _init_wandb(self):
        if not self.config.use_wandb:
            return
        try:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    "base_model": self.config.base_model,
                    "lr": self.config.grpo_lr,
                    "group_size": self.config.group_size,
                    "objective": self.objective,
                    "dynamic_routing": self.dynamic_routing,
                    "sub_steps": self.sub_steps,
                    "max_subtasks": self.max_subtasks,
                },
                resume="allow",
            )
            self._wandb = wandb
        except ImportError:
            logger.warning("wandb not installed; install with: pip install wandb")

    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        for key, val in metrics.items():
            k = f"{prefix}{key}" if prefix else key
            logger.info(f"  {k}: {val:.4f}")
            if self._wandb:
                self._wandb.log({k: val}, step=step)

    # ── Parsing helpers ──────────────────────────────────────────

    @staticmethod
    def extract_block(text: str, tag: str) -> str:
        match = re.search(rf"\[{tag}\]\s*(.*?)\s*\[/{tag}\]", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    @staticmethod
    def extract_blocks(text: str, tag: str) -> List[str]:
        return [m.strip() for m in re.findall(rf"\[{tag}\]\s*(.*?)\s*\[/{tag}\]", text, re.DOTALL)]

    @staticmethod
    def extract_tool_call(text: str) -> str:
        match = re.search(r"\[tool_call\].*?\[/tool_call\]", text, re.DOTALL)
        return match.group(0) if match else ""

    @staticmethod
    def truncate_result(text: str) -> str:
        end = text.find("</result>")
        if end >= 0:
            return text[: end + len("</result>")]
        return text

    @staticmethod
    def history_text(history):
        if not history:
            return "No observations yet."
        lines = []
        for idx, (tool_call, observation) in enumerate(history, 1):
            lines.append(f"Step {idx} tool call: {tool_call}")
            lines.append(f"Step {idx} observation: {observation}")
        return "\n".join(lines)

    # ── Prompt builders ──────────────────────────────────────────

    def _build_prompt(self, system: str, user: str) -> str:
        return self.model.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )

    def build_main_plan_prompt(self, task: HotpotTask) -> str:
        sys = DYNAMIC_MAIN_PLAN_SYSTEM if self.dynamic_routing else MAIN_PLAN_SYSTEM
        return self._build_prompt(sys, f"Question: {task.question}")

    def build_direct_answer_prompt(self, task: HotpotTask) -> str:
        return self._build_prompt(DIRECT_ANSWER_SYSTEM, f"Question: {task.question}")

    def build_sub_action_prompt(self, subtask: str, history) -> str:
        return self._build_prompt(
            SUB_ACTION_SYSTEM,
            f"Subtask: {subtask}\nResearch history:\n{self.history_text(history)}",
        )

    def build_sub_summary_prompt(self, subtask: str, history) -> str:
        return self._build_prompt(
            SUB_SUMMARY_SYSTEM,
            f"Subtask: {subtask}\nResearch history:\n{self.history_text(history)}",
        )

    def build_main_answer_prompt(self, task: HotpotTask, sub_results: List[Tuple[str, str]]) -> str:
        if not sub_results:
            return self._build_prompt(DIRECT_ANSWER_SYSTEM, f"Question: {task.question}")
        sys = DYNAMIC_MAIN_ANSWER_SYSTEM if self.dynamic_routing else MAIN_ANSWER_SYSTEM
        parts = [f"Question: {task.question}"]
        for i, (subtask, result) in enumerate(sub_results, 1):
            parts.append(f"Sub research {i}:\n  Subtask: {subtask}\n  Result: {result}")
        return self._build_prompt(sys, "\n".join(parts))

    # ── Sub agent rollout ────────────────────────────────────────

    def _run_sub_research(self, subtask: str) -> Dict[str, Any]:
        """Run sub agent: multi-step action -> summary. Returns research dict."""
        history = []
        tool_calls = []
        valid_actions = 0
        read_docs = set()
        read_sequence = []
        action_steps = []

        for _ in range(self.sub_steps):
            prompt = self.build_sub_action_prompt(subtask, history)
            raw = self.model.generate_one(
                SharedModel.SUB_ADAPTER, prompt,
                max_tokens=self.config.max_response_len,
                response_prefix="<thinking>",
            )
            tool_call = self.extract_tool_call(raw)
            ok, observation = HotpotQAEnvironment.execute_tool(tool_call)
            if ok:
                valid_actions += 1
                doc_id = HotpotQAEnvironment.parse_tool_call(tool_call)
                if doc_id:
                    _, doc_id_str = doc_id
                    if doc_id_str:
                        read_docs.add(doc_id_str)
                        read_sequence.append(doc_id_str)
            else:
                observation = "Tool execution failed"
            history.append((tool_call, observation))
            action_steps.append({"prompt": prompt, "raw": raw, "tool_call": tool_call, "ok": ok})

        summary_prompt = self.build_sub_summary_prompt(subtask, history)
        summary_raw = self.model.generate_one(
            SharedModel.SUB_ADAPTER, summary_prompt,
            max_tokens=self.config.max_response_len,
            response_prefix="<thinking>",
        )
        summary_raw = self.truncate_result(summary_raw)

        return {
            "subtask": subtask,
            "action_steps": action_steps,
            "summary_prompt": summary_prompt,
            "summary_raw": summary_raw,
            "read_docs": read_docs,
            "read_sequence": read_sequence,
            "valid_actions": valid_actions,
            "all_tool_calls": "".join(tc for tc in [s.get("tool_call", "") for s in action_steps]),
        }

    # ── Candidate generation ─────────────────────────────────────

    def generate_candidate(self, task: HotpotTask) -> Dict[str, Any]:
        plan_prompt = self.build_main_plan_prompt(task)
        plan_raw = self.model.generate_one(
            SharedModel.MAIN_ADAPTER, plan_prompt,
            max_tokens=self.config.max_response_len,
            response_prefix="<thinking>",
        )

        if self.dynamic_routing:
            mode = self.extract_block(plan_raw, "mode")
            if mode == "direct":
                return self._generate_direct_candidate(task, plan_prompt, plan_raw)
            # delegate: extract subtasks
            subtasks = self.extract_blocks(plan_raw, "subtask")
            return self._generate_delegate_candidate(task, plan_prompt, plan_raw, subtasks)

        # Legacy: fixed Main -> one Sub -> Main answer
        subtask = self.extract_block(plan_raw, "subtask")
        return self._generate_delegate_candidate(task, plan_prompt, plan_raw, [subtask])

    def _generate_direct_candidate(self, task: HotpotTask, plan_prompt: str, plan_raw: str) -> Dict[str, Any]:
        answer_prompt = self.build_direct_answer_prompt(task)
        answer_raw = self.model.generate_one(
            SharedModel.MAIN_ADAPTER, answer_prompt,
            max_tokens=self.config.max_response_len,
            response_prefix="<thinking>",
        )
        answer_raw = self.truncate_result(answer_raw)
        main_reward = HotpotQAEnvironment.reward(task, answer_raw)

        return {
            "plan_prompt": plan_prompt,
            "plan_raw": plan_raw,
            "mode": "direct",
            "subtasks": [],
            "sub_researches": [],
            "answer_prompt": answer_prompt,
            "answer_raw": answer_raw,
            "raw": answer_raw,
            "reward": main_reward["total"],
            "answer_f1": main_reward["answer_f1"],
            "evidence": main_reward["evidence"],
            "tool_valid": 1.0,
            "sub_reward": 0.0,
            "sub_train_reward": 0.0,
            "sub_answer_f1": 0.0,
            "sub_evidence": 0.0,
            "sub_retrieval_reward": 0.0,
            "sub_read_precision": 0.0,
            "no_duplicate_read": 1.0,
            "action_valid": 1.0,
            "num_subtasks": 0,
        }

    def _generate_delegate_candidate(
        self, task: HotpotTask, plan_prompt: str, plan_raw: str, subtasks: List[str]
    ) -> Dict[str, Any]:
        subtasks = subtasks[: self.max_subtasks] if subtasks else ["Find supporting documents and answer."]

        # Run sub research for each subtask
        sub_researches = [self._run_sub_research(subtask) for subtask in subtasks]

        # Synthesize with main
        sub_results = [(r["subtask"], r["summary_raw"]) for r in sub_researches]
        answer_prompt = self.build_main_answer_prompt(task, sub_results)
        answer_raw = self.model.generate_one(
            SharedModel.MAIN_ADAPTER, answer_prompt,
            max_tokens=self.config.max_response_len,
            response_prefix="<thinking>",
        )
        answer_raw = self.truncate_result(answer_raw)

        # Compute rewards
        all_tool_text = "".join(r["all_tool_calls"] for r in sub_researches)
        all_summaries = " ".join(r["summary_raw"] for r in sub_researches)
        combined = all_tool_text + all_summaries + answer_raw
        main_reward = HotpotQAEnvironment.reward(task, combined)

        # Sub reward (best sub)
        best_sub_reward = {"total": 0.0, "answer_f1": 0.0, "evidence": 0.0}
        google_docs = set(task.support_doc_ids)
        total_retrieval = 0.0
        total_precision = 0.0
        total_no_dup = 0.0
        total_action_valid = 0.0
        num_subs = max(len(sub_researches), 1)

        for research in sub_researches:
            sr = HotpotQAEnvironment.reward(task, research["all_tool_calls"] + research["summary_raw"])
            if sr["total"] > best_sub_reward["total"]:
                best_sub_reward = sr
            r_docs = research["read_docs"]
            total_retrieval += len(r_docs & google_docs) / max(len(google_docs), 1)
            total_precision += len(r_docs & google_docs) / max(len(r_docs), 1) if r_docs else 0.0
            total_no_dup += (
                len(set(research["read_sequence"])) / max(len(research["read_sequence"]), 1)
                if research["read_sequence"] else 1.0
            )
            total_action_valid += research["valid_actions"] / max(self.sub_steps, 1)

        sub_retrieval_reward = total_retrieval / num_subs
        sub_read_precision = total_precision / num_subs
        no_duplicate_read = total_no_dup / num_subs
        action_valid = total_action_valid / num_subs

        sub_train_reward = self.build_sub_train_reward(
            best_sub_reward, sub_retrieval_reward, action_valid,
            sub_read_precision, no_duplicate_read,
        )

        return {
            "plan_prompt": plan_prompt,
            "plan_raw": plan_raw,
            "mode": "delegate",
            "subtasks": subtasks,
            "sub_researches": sub_researches,
            "answer_prompt": answer_prompt,
            "answer_raw": answer_raw,
            "raw": combined,
            "reward": main_reward["total"],
            "answer_f1": main_reward["answer_f1"],
            "evidence": main_reward["evidence"],
            "tool_valid": 1.0 if any(r["action_steps"] for r in sub_researches) else 0.0,
            "sub_reward": best_sub_reward["total"],
            "sub_train_reward": sub_train_reward,
            "sub_answer_f1": best_sub_reward["answer_f1"],
            "sub_evidence": best_sub_reward["evidence"],
            "sub_retrieval_reward": sub_retrieval_reward,
            "sub_read_precision": sub_read_precision,
            "no_duplicate_read": no_duplicate_read,
            "action_valid": action_valid,
            "num_subtasks": len(subtasks),
        }

    # ── Reward / scoring ────────────────────────────────────────

    def validation_score(self, metrics):
        mapping = {
            "answer_f1": "answer_f1", "reward": "reward", "best_reward": "best_reward",
            "sub_reward": "sub_reward", "sub_evidence": "sub_evidence",
            "sub_retrieval": "sub_retrieval_reward", "sub_train_reward": "sub_train_reward",
        }
        return metrics.get(mapping.get(self.best_metric, self.best_metric), 0.0)

    def candidate_key(self, cand):
        if self.train_main and self.train_sub and self.sub_reward_mode == "enhanced":
            return (
                0.55 * cand["reward"] + 0.45 * cand["sub_train_reward"],
                cand["answer_f1"], cand["sub_retrieval_reward"],
                cand["sub_read_precision"], cand["tool_valid"],
            )
        return (
            cand["sub_train_reward"] if self.train_sub and not self.train_main else cand["answer_f1"],
            cand["reward"], cand["sub_retrieval_reward"],
            cand["sub_evidence"], cand["tool_valid"],
        )

    def build_sub_train_reward(
        self, summary_reward: Dict, retrieval_reward: float, action_valid: float,
        read_precision: float = 0.0, no_duplicate_read: float = 1.0,
    ) -> float:
        if self.sub_reward_mode == "summary":
            return summary_reward["total"]
        if self.sub_reward_mode == "retrieval":
            return 0.8 * retrieval_reward + 0.2 * action_valid
        if self.sub_reward_mode == "mixed":
            return 0.5 * summary_reward["total"] + 0.4 * retrieval_reward + 0.1 * action_valid
        if self.sub_reward_mode == "enhanced":
            return (
                0.40 * retrieval_reward + 0.25 * summary_reward["answer_f1"]
                + 0.15 * summary_reward["evidence"] + 0.10 * read_precision
                + 0.05 * action_valid + 0.05 * no_duplicate_read
            )
        raise ValueError(f"Unknown sub_reward_mode: {self.sub_reward_mode}")

    # ── Group / episode runners ──────────────────────────────────

    def run_episode(self, task: HotpotTask) -> List[Dict[str, Any]]:
        """Generate group_size candidates, return sorted list (best first)."""
        candidates = [self.generate_candidate(task) for _ in range(self.config.group_size)]
        candidates.sort(key=self.candidate_key, reverse=True)
        return candidates

    @staticmethod
    def group_advantages(candidates, reward_key: str, clip: float):
        values = [cand.get(reward_key, 0.0) for cand in candidates]
        mean = sum(values) / max(len(values), 1)
        var = sum((v - mean) ** 2 for v in values) / max(len(values), 1)
        std = max(math.sqrt(var), 1e-6)
        for cand, value in zip(candidates, values):
            adv = (value - mean) / std
            cand[f"{reward_key}_advantage"] = max(min(adv, clip), -clip)
        return candidates

    # ── Training updates ─────────────────────────────────────────

    def apply_best_of_update(self, best: Dict[str, Any]):
        main_updates, sub_updates = 0, 0
        if self.train_main and best["reward"] >= self.config.reward_threshold:
            self.model.sft_step(SharedModel.MAIN_ADAPTER, best["plan_prompt"], best["plan_raw"],
                                weight=best["reward"])
            self.model.sft_step(SharedModel.MAIN_ADAPTER, best["answer_prompt"], best["answer_raw"],
                                weight=best["reward"])
            main_updates += 1

        if self.train_sub and best.get("sub_train_reward", 0) >= self.config.reward_threshold:
            for research in best.get("sub_researches", []):
                for step in research.get("action_steps", []):
                    self.model.sft_step(SharedModel.SUB_ADAPTER, step["prompt"], step["tool_call"],
                                        weight=best.get("sub_train_reward", 0.0))
                self.model.sft_step(SharedModel.SUB_ADAPTER, research["summary_prompt"],
                                    research["summary_raw"], weight=best.get("sub_train_reward", 0.0))
            sub_updates += 1
        return main_updates, sub_updates

    def apply_grpo_update(self, candidates: List[Dict[str, Any]], task: HotpotTask):
        """True GRPO policy-gradient update for all candidates."""
        main_updates, sub_updates = 0, 0
        grpo_info = {"main_policy_loss": 0.0, "main_kl": 0.0, "main_ratio": 0.0,
                      "sub_policy_loss": 0.0, "sub_kl": 0.0, "sub_ratio": 0.0}
        main_count = sub_count = 0

        for cand in candidates:
            main_adv = cand.get("reward_advantage", 0.0)
            sub_adv = cand.get("sub_train_reward_advantage", 0.0)

            # Compute reference logprobs ONCE per candidate
            main_ref = self.model.response_token_logprobs(
                self.model.reference_adapter(SharedModel.MAIN_ADAPTER),
                cand["plan_prompt"], cand["plan_raw"], with_grad=False,
            )
            answer_ref = self.model.response_token_logprobs(
                self.model.reference_adapter(SharedModel.MAIN_ADAPTER),
                cand["answer_prompt"], cand["answer_raw"], with_grad=False,
            )

            if self.train_main and abs(main_adv) >= self.min_advantage:
                info = self.model.grpo_step(
                    SharedModel.MAIN_ADAPTER, cand["plan_prompt"], cand["plan_raw"],
                    reference_logprobs=main_ref, advantage=main_adv,
                    policy_clip=self.grpo_policy_clip, kl_beta=self.grpo_kl_beta,
                    weight=abs(main_adv),
                )
                grpo_info["main_policy_loss"] += info["policy_loss"]
                grpo_info["main_kl"] += info["kl"]
                grpo_info["main_ratio"] += info["ratio"]
                main_count += 1

                info_a = self.model.grpo_step(
                    SharedModel.MAIN_ADAPTER, cand["answer_prompt"], cand["answer_raw"],
                    reference_logprobs=answer_ref, advantage=main_adv,
                    policy_clip=self.grpo_policy_clip, kl_beta=self.grpo_kl_beta,
                    weight=abs(main_adv),
                )
                grpo_info["main_policy_loss"] += info_a["policy_loss"]
                grpo_info["main_kl"] += info_a["kl"]
                main_count += 1

            if self.train_sub and abs(sub_adv) >= self.min_advantage:
                for research in cand.get("sub_researches", []):
                    sub_ref = self.model.response_token_logprobs(
                        self.model.reference_adapter(SharedModel.SUB_ADAPTER),
                        research["summary_prompt"], research["summary_raw"], with_grad=False,
                    )
                    for step in research.get("action_steps", []):
                        info_s = self.model.grpo_step(
                            SharedModel.SUB_ADAPTER, step["prompt"], step["tool_call"],
                            reference_logprobs=sub_ref, advantage=sub_adv,
                            policy_clip=self.grpo_policy_clip, kl_beta=self.grpo_kl_beta,
                            weight=abs(sub_adv),
                        )
                        grpo_info["sub_policy_loss"] += info_s["policy_loss"]
                        grpo_info["sub_kl"] += info_s["kl"]
                        sub_count += 1
                    info_ss = self.model.grpo_step(
                        SharedModel.SUB_ADAPTER, research["summary_prompt"],
                        research["summary_raw"], reference_logprobs=sub_ref,
                        advantage=sub_adv,
                        policy_clip=self.grpo_policy_clip, kl_beta=self.grpo_kl_beta,
                        weight=abs(sub_adv),
                    )
                    grpo_info["sub_policy_loss"] += info_ss["policy_loss"]
                    grpo_info["sub_kl"] += info_ss["kl"]
                    sub_count += 1

        mc = max(main_count, 1)
        sc = max(sub_count, 1)
        grpo_info["main_policy_loss"] /= mc
        grpo_info["main_kl"] /= mc
        grpo_info["main_ratio"] /= mc
        grpo_info["sub_policy_loss"] /= sc
        grpo_info["sub_kl"] /= sc
        grpo_info["sub_ratio"] /= sc
        grpo_info["main_updates"] = mc
        grpo_info["sub_updates"] = sc
        return grpo_info

    # ── Evaluation ───────────────────────────────────────────────

    def evaluate(self, tasks: List[HotpotTask], samples: int = 1) -> Dict[str, float]:
        if not tasks:
            return {k: 0.0 for k in [
                "reward", "answer_f1", "evidence", "tool_valid",
                "sub_reward", "sub_train_reward", "sub_evidence", "sub_retrieval_reward",
                "sub_read_precision", "no_duplicate_read", "action_valid",
                "best_reward", "best_answer_f1", "direct_rate", "avg_subtasks",
            ]}

        self.model.model.eval()
        all_metrics: List[Dict] = []
        best_reward_total = 0.0
        best_answer_total = 0.0

        for task in tasks:
            task_best_reward = 0.0
            task_best_answer = 0.0
            for _ in range(samples):
                cand = self.generate_candidate(task)
                cand["direct"] = cand.get("mode") == "direct"
                all_metrics.append(cand)
                task_best_reward = max(task_best_reward, cand["reward"])
                task_best_answer = max(task_best_answer, cand["answer_f1"])
            best_reward_total += task_best_reward
            best_answer_total += task_best_answer

        self.model.model.train()
        total = max(len(all_metrics), 1)
        result = {
            "reward": sum(m["reward"] for m in all_metrics) / total,
            "answer_f1": sum(m["answer_f1"] for m in all_metrics) / total,
            "evidence": sum(m["evidence"] for m in all_metrics) / total,
            "tool_valid": sum(m["tool_valid"] for m in all_metrics) / total,
            "sub_reward": sum(m["sub_reward"] for m in all_metrics) / total,
            "sub_train_reward": sum(m["sub_train_reward"] for m in all_metrics) / total,
            "sub_evidence": sum(m["sub_evidence"] for m in all_metrics) / total,
            "sub_retrieval_reward": sum(m["sub_retrieval_reward"] for m in all_metrics) / total,
            "sub_read_precision": sum(m["sub_read_precision"] for m in all_metrics) / total,
            "no_duplicate_read": sum(m["no_duplicate_read"] for m in all_metrics) / total,
            "action_valid": sum(m["action_valid"] for m in all_metrics) / total,
            "best_reward": best_reward_total / len(tasks),
            "best_answer_f1": best_answer_total / len(tasks),
            "direct_rate": sum(1 for m in all_metrics if m.get("direct")) / total,
            "avg_subtasks": sum(m.get("num_subtasks", 0) for m in all_metrics) / total,
        }
        return result

    # ── Main training loop ───────────────────────────────────────

    def train(self, train_tasks: List[HotpotTask], val_tasks: List[HotpotTask],
              iterations: int, eval_samples: int):
        logger.info(f"[hotpotqa-mas-grpo] train={len(train_tasks)} val={len(val_tasks)} iter={iterations}")
        logger.info(f"[hotpotqa-mas-grpo] lr={self.config.grpo_lr} group={self.config.group_size} "
                    f"threshold={self.config.reward_threshold} accum={self.config.gradient_accumulation_steps}")
        logger.info(f"[hotpotqa-mas-grpo] train_main={self.train_main} train_sub={self.train_sub} "
                    f"best_metric={self.best_metric} sub_reward_mode={self.sub_reward_mode} "
                    f"objective={self.objective} dynamic_routing={self.dynamic_routing} "
                    f"max_subtasks={self.max_subtasks}")

        self.model = SharedModel(self.config.base_model, self.config)
        self.model.load_sft_weights()

        if self.config.resume_from_checkpoint:
            if os.path.exists(self.config.resume_from_checkpoint):
                logger.info(f"Resuming from checkpoint: {self.config.resume_from_checkpoint}")
                self.model.load_checkpoint(self.config.resume_from_checkpoint)
            else:
                logger.warning(f"Checkpoint not found: {self.config.resume_from_checkpoint}")

        self._init_wandb()
        self.model.model.train()

        # Register graceful shutdown — save checkpoint on Ctrl+C
        on_shutdown(lambda: self.model.save_checkpoint(
            str(self.save_dir / "emergency_checkpoint")
        ))

        logger.info("\n===== HotpotQA MAS Initial Validation =====")
        init = self.evaluate(val_tasks, samples=eval_samples)
        best_val = self.validation_score(init)
        self._log_eval(init, "val/init")
        self._log_metrics(init, step=0, prefix="val/init/")

        for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
            self.model.save_lora(name, str(self.save_dir / "best" / name))

        global_step = 0
        for it in range(iterations):
            logger.info(f"\n===== HotpotQA MAS Iter {it + 1}/{iterations} =====")
            rewards, answers, evidences, valids = [], [], [], []
            sub_rewards, sub_train_rewards, sub_retrievals = [], [], []
            sub_precisions, direct_flags, num_subtasks_list = [], [], []
            main_updates, sub_updates = 0, 0
            grpo_info = {}

            for task in try_tqdm(train_tasks, desc=f"Iter {it + 1}/{iterations}", unit="task"):
                candidates = self.run_episode(task)
                best = candidates[0]

                rewards.append(best["reward"])
                answers.append(best["answer_f1"])
                evidences.append(best["evidence"])
                valids.append(best["tool_valid"])
                sub_rewards.append(best["sub_reward"])
                sub_train_rewards.append(best["sub_train_reward"])
                sub_retrievals.append(best["sub_retrieval_reward"])
                sub_precisions.append(best["sub_read_precision"])
                direct_flags.append(1.0 if best.get("mode") == "direct" else 0.0)
                num_subtasks_list.append(best.get("num_subtasks", 0))

                if self.objective == "grpo":
                    self.group_advantages(candidates, "reward", self.advantage_clip)
                    self.group_advantages(candidates, "sub_train_reward", self.advantage_clip)
                    info = self.apply_grpo_update(candidates, task)
                    grpo_info = info
                    main_updates += info["main_updates"]
                    sub_updates += info["sub_updates"]
                else:
                    mu, su = self.apply_best_of_update(best)
                    main_updates += mu
                    sub_updates += su
                global_step += 1

            n = max(len(rewards), 1)
            train_metrics = {
                "reward": sum(rewards) / n,
                "answer_f1": sum(answers) / n,
                "evidence": sum(evidences) / n,
                "tool_valid": sum(valids) / n,
                "sub_reward": sum(sub_rewards) / n,
                "sub_train_reward": sum(sub_train_rewards) / n,
                "sub_retrieval_reward": sum(sub_retrievals) / n,
                "sub_read_precision": sum(sub_precisions) / n,
                "direct_rate": sum(direct_flags) / n,
                "avg_subtasks": sum(num_subtasks_list) / n,
                "main_updates": float(main_updates),
                "sub_updates": float(sub_updates),
            }
            self._log_metrics(train_metrics, step=global_step, prefix="train/")

            val = self.evaluate(val_tasks, samples=eval_samples)
            self._log_metrics(val, step=global_step, prefix="val/")

            score = self.validation_score(val)
            if score > best_val:
                best_val = score
                logger.info(f"  [best] save best checkpoint ({self.best_metric}={best_val:.4f})")
                for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
                    self.model.save_lora(name, str(self.save_dir / "best" / name))

            # Periodic checkpoint
            if self.config.save_every_steps > 0 and (it + 1) % self.config.save_every_steps == 0:
                ckpt_dir = str(self.save_dir / f"checkpoint_iter_{it + 1}")
                self.model.save_checkpoint(ckpt_dir)
                logger.info(f"  [ckpt] saved checkpoint to {ckpt_dir}")

        logger.info("\n[OK] HotpotQA MAS GRPO complete")
        if self._wandb:
            self._wandb.finish()

    def _log_eval(self, metrics: Dict[str, float], prefix: str):
        logger.info(
            f"  [{prefix}] reward={metrics['reward']:.3f} best={metrics['best_reward']:.3f} "
            f"answer_f1={metrics['answer_f1']:.3f} best_answer={metrics['best_answer_f1']:.3f} "
            f"evidence={metrics['evidence']:.3f} sub_reward={metrics['sub_reward']:.3f} "
            f"sub_train={metrics['sub_train_reward']:.3f} sub_retrieval={metrics['sub_retrieval_reward']:.3f} "
            f"sub_precision={metrics['sub_read_precision']:.3f} no_dup={metrics['no_duplicate_read']:.3f} "
            f"sub_evidence={metrics['sub_evidence']:.3f} action_valid={metrics['action_valid']:.3f} "
            f"tool_valid={metrics['tool_valid']:.3f} direct_rate={metrics.get('direct_rate', 0):.3f} "
            f"avg_subtasks={metrics.get('avg_subtasks', 0):.2f}"
        )


# ── CLI ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train HotpotQA Main/Sub MAS agents.")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--sft-dir", default="./artifacts/checkpoints/mas_sft")
    p.add_argument("--main-lora", default=None)
    p.add_argument("--sub-lora", default=None)
    p.add_argument("--save-dir", default="./artifacts/checkpoints/mas_grpo")
    p.add_argument("--train-jsonl", default="./data/base/train.jsonl")
    p.add_argument("--val-jsonl", default="./data/base/val.jsonl")
    p.add_argument("--tasks", type=int, default=50)
    p.add_argument("--val-tasks", type=int, default=20)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--group-size", type=int, default=2)
    p.add_argument("--eval-samples", type=int, default=1)
    p.add_argument("--max-response-len", type=int, default=120)
    p.add_argument("--sub-steps", type=int, default=3)
    p.add_argument("--max-subtasks", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--reward-threshold", type=float, default=0.3)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--best-metric", choices=[
        "answer_f1", "reward", "best_reward", "sub_reward",
        "sub_evidence", "sub_retrieval", "sub_train_reward",
    ], default="answer_f1")
    p.add_argument("--sub-reward-mode", choices=["summary", "retrieval", "mixed", "enhanced"], default="summary")
    p.add_argument("--objective", choices=["best_of", "grpo"], default="best_of")
    p.add_argument("--dynamic-routing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--advantage-clip", type=float, default=2.0)
    p.add_argument("--min-advantage", type=float, default=0.0)
    p.add_argument("--grpo-policy-clip", type=float, default=0.2)
    p.add_argument("--grpo-kl-beta", type=float, default=0.01)
    p.add_argument("--train-main", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--train-sub", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", default="hotpotqa-mas")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--resume", default=None, help="Path to checkpoint directory to resume from.")
    p.add_argument("--save-every", type=int, default=0, help="Save checkpoint every N iterations (0=disable).")
    return p.parse_args()


def main():
    args = parse_args()

    if dry_run_mode():
        dry_run_warning("Validating data and config without loading model.")
        train_env = HotpotQAEnvironment.from_jsonl(args.train_jsonl, limit=min(args.tasks, 3))
        val_env = HotpotQAEnvironment.from_jsonl(args.val_jsonl, limit=min(args.val_tasks, 3))
        print(f"  Train tasks: {len(train_env.tasks)} (limit={args.tasks})")
        print(f"  Val tasks: {len(val_env.tasks)} (limit={args.val_tasks})")
        print(f"  Config: lr={args.lr} group={args.group_size} iter={args.iterations}")
        print(f"  Objective: {args.objective}  Dynamic routing: {args.dynamic_routing}")
        print("  [dry-run] All checks passed.  Fire up the GPUs.")
        return

    train_env = HotpotQAEnvironment.from_jsonl(args.train_jsonl, limit=args.tasks)
    val_env = HotpotQAEnvironment.from_jsonl(args.val_jsonl, limit=args.val_tasks)

    config = TrainingConfig(
        base_model=args.base_model,
        sft_dir=args.sft_dir,
        main_lora_path=args.main_lora,
        sub_lora_path=args.sub_lora,
        save_dir=args.save_dir,
        group_size=args.group_size,
        max_response_len=args.max_response_len,
        grpo_lr=args.lr,
        reward_threshold=args.reward_threshold,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        resume_from_checkpoint=args.resume,
        save_every_steps=args.save_every,
    )

    HotpotMASGRPOTrainer(
        config,
        sub_steps=args.sub_steps,
        max_subtasks=args.max_subtasks,
        best_metric=args.best_metric,
        train_main=args.train_main,
        train_sub=args.train_sub,
        sub_reward_mode=args.sub_reward_mode,
        objective=args.objective,
        dynamic_routing=args.dynamic_routing,
        advantage_clip=args.advantage_clip,
        min_advantage=args.min_advantage,
        grpo_policy_clip=args.grpo_policy_clip,
        grpo_kl_beta=args.grpo_kl_beta,
    ).train(
        train_env.tasks, val_env.tasks,
        iterations=args.iterations, eval_samples=args.eval_samples,
    )


if __name__ == "__main__":
    main()
