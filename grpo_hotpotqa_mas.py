"""M-GRPO-style HotpotQA training with Main coordinator and Sub researcher."""

import argparse
import re
from pathlib import Path
from typing import List

from generate_hotpotqa_mas_sft_data import (
    MAIN_ANSWER_SYSTEM,
    MAIN_PLAN_SYSTEM,
    SUB_ACTION_SYSTEM,
    SUB_SUMMARY_SYSTEM,
)
from grpo_v4 import CoTrainConfig, SharedModel
from hotpotqa_environment import HotpotQAEnvironment, HotpotTask


class HotpotMASGRPOTrainer:
    def __init__(
        self,
        config: CoTrainConfig,
        sub_steps: int = 3,
        best_metric: str = "answer_f1",
        train_main: bool = True,
        train_sub: bool = True,
        sub_reward_mode: str = "summary",
        objective: str = "best_of",
        advantage_clip: float = 2.0,
        min_advantage: float = 0.0,
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
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    @staticmethod
    def extract_block(text: str, tag: str) -> str:
        match = re.search(rf"\[{tag}\]\s*(.*?)\s*\[/{tag}\]", text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    @staticmethod
    def extract_tool_call(text: str) -> str:
        match = re.search(r"\[tool_call\].*?\[/tool_call\]", text, re.DOTALL)
        return match.group(0) if match else text

    @staticmethod
    def truncate_result(text: str) -> str:
        end = text.find("</result>")
        if end >= 0:
            return text[:end + len("</result>")]
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

    def build_main_plan_prompt(self, task: HotpotTask) -> str:
        return self.model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": MAIN_PLAN_SYSTEM},
                {"role": "user", "content": f"Question: {task.question}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_sub_action_prompt(self, subtask: str, history) -> str:
        return self.model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SUB_ACTION_SYSTEM},
                {"role": "user", "content": f"Subtask: {subtask}\nResearch history:\n{self.history_text(history)}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_sub_summary_prompt(self, subtask: str, history) -> str:
        return self.model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SUB_SUMMARY_SYSTEM},
                {"role": "user", "content": f"Subtask: {subtask}\nResearch history:\n{self.history_text(history)}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_main_answer_prompt(self, task: HotpotTask, subtask: str, sub_result: str) -> str:
        return self.model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": MAIN_ANSWER_SYSTEM},
                {
                    "role": "user",
                    "content": f"Question: {task.question}\nSubtask: {subtask}\nSub result: {sub_result}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def validation_score(self, metrics):
        if self.best_metric == "answer_f1":
            return metrics["answer_f1"]
        if self.best_metric == "reward":
            return metrics["reward"]
        if self.best_metric == "best_reward":
            return metrics["best_reward"]
        if self.best_metric == "sub_reward":
            return metrics["sub_reward"]
        if self.best_metric == "sub_evidence":
            return metrics["sub_evidence"]
        if self.best_metric == "sub_retrieval":
            return metrics["sub_retrieval_reward"]
        if self.best_metric == "sub_train_reward":
            return metrics["sub_train_reward"]
        raise ValueError(f"Unknown best_metric: {self.best_metric}")

    def candidate_key(self, cand):
        if self.train_main and self.train_sub and self.sub_reward_mode == "enhanced":
            return (
                0.55 * cand["reward"] + 0.45 * cand["sub_train_reward"],
                cand["answer_f1"],
                cand["sub_retrieval_reward"],
                cand["sub_read_precision"],
                cand["tool_valid"],
            )
        return (
            cand["sub_train_reward"] if self.train_sub and not self.train_main else cand["answer_f1"],
            cand["reward"],
            cand["sub_retrieval_reward"],
            cand["sub_evidence"],
            cand["tool_valid"],
        )

    @staticmethod
    def read_doc_id(tool_call: str):
        parsed = HotpotQAEnvironment.parse_tool_call(tool_call)
        if parsed is None:
            return None
        tool, arg = parsed
        return arg if tool == "read" else None

    def build_sub_train_reward(
        self,
        summary_reward,
        retrieval_reward: float,
        action_valid: float,
        read_precision: float = 0.0,
        no_duplicate_read: float = 1.0,
    ) -> float:
        if self.sub_reward_mode == "summary":
            return summary_reward["total"]
        if self.sub_reward_mode == "retrieval":
            return 0.8 * retrieval_reward + 0.2 * action_valid
        if self.sub_reward_mode == "mixed":
            return 0.5 * summary_reward["total"] + 0.4 * retrieval_reward + 0.1 * action_valid
        if self.sub_reward_mode == "enhanced":
            return (
                0.40 * retrieval_reward
                + 0.25 * summary_reward["answer_f1"]
                + 0.15 * summary_reward["evidence"]
                + 0.10 * read_precision
                + 0.05 * action_valid
                + 0.05 * no_duplicate_read
            )
        raise ValueError(f"Unknown sub_reward_mode: {self.sub_reward_mode}")

    def generate_candidate(self, task: HotpotTask):
        plan_prompt = self.build_main_plan_prompt(task)
        plan_raw = self.model.generate_one(
            SharedModel.MAIN_ADAPTER,
            plan_prompt,
            max_tokens=self.config.max_response_len,
            response_prefix="<thinking>",
            canonicalizer=None,
        )
        subtask = self.extract_block(plan_raw, "subtask")

        history = []
        sub_action_steps = []
        tool_calls = []
        ok_any = False
        valid_actions = 0
        read_docs = set()
        read_sequence = []
        for _ in range(self.sub_steps):
            prompt = self.build_sub_action_prompt(subtask, history)
            raw = self.model.generate_one(
                SharedModel.SUB_ADAPTER,
                prompt,
                max_tokens=self.config.max_response_len,
                response_prefix="<thinking>",
                canonicalizer=None,
            )
            tool_call = self.extract_tool_call(raw)
            ok, observation = HotpotQAEnvironment.execute_tool(task, tool_call)
            if not ok:
                observation = "Tool execution failed"
            else:
                valid_actions += 1
                doc_id = self.read_doc_id(tool_call)
                if doc_id:
                    read_docs.add(doc_id)
                    read_sequence.append(doc_id)
            ok_any = ok_any or ok
            history.append((tool_call, observation))
            sub_action_steps.append((prompt, tool_call, ok, observation))
            tool_calls.append(tool_call)

        sub_summary_prompt = self.build_sub_summary_prompt(subtask, history)
        sub_summary = self.model.generate_one(
            SharedModel.SUB_ADAPTER,
            sub_summary_prompt,
            max_tokens=self.config.max_response_len,
            response_prefix="<thinking>",
            canonicalizer=None,
        )
        sub_summary = self.truncate_result(sub_summary)

        answer_prompt = self.build_main_answer_prompt(task, subtask, sub_summary)
        answer_raw = self.model.generate_one(
            SharedModel.MAIN_ADAPTER,
            answer_prompt,
            max_tokens=self.config.max_response_len,
            response_prefix="<thinking>",
            canonicalizer=None,
        )
        answer_raw = self.truncate_result(answer_raw)

        combined = "".join(tool_calls) + sub_summary + answer_raw
        main_reward = HotpotQAEnvironment.reward(task, combined)
        sub_reward = HotpotQAEnvironment.reward(task, "".join(tool_calls) + sub_summary)
        gold_docs = set(task.support_doc_ids)
        sub_retrieval_reward = len(read_docs & gold_docs) / max(len(gold_docs), 1)
        sub_read_precision = len(read_docs & gold_docs) / max(len(read_docs), 1) if read_docs else 0.0
        no_duplicate_read = len(set(read_sequence)) / max(len(read_sequence), 1) if read_sequence else 1.0
        action_valid = valid_actions / max(self.sub_steps, 1)
        sub_train_reward = self.build_sub_train_reward(
            sub_reward,
            sub_retrieval_reward,
            action_valid,
            sub_read_precision,
            no_duplicate_read,
        )
        return {
            "plan_prompt": plan_prompt,
            "plan_raw": f"<thinking>Delegate research.</thinking>[subtask]{subtask}[/subtask]",
            "sub_action_steps": sub_action_steps,
            "sub_summary_prompt": sub_summary_prompt,
            "sub_summary": sub_summary,
            "answer_prompt": answer_prompt,
            "answer_raw": answer_raw,
            "raw": combined,
            "reward": main_reward["total"],
            "answer_f1": main_reward["answer_f1"],
            "evidence": main_reward["evidence"],
            "tool_valid": 1.0 if ok_any else 0.0,
            "sub_reward": sub_reward["total"],
            "sub_train_reward": sub_train_reward,
            "sub_answer_f1": sub_reward["answer_f1"],
            "sub_evidence": sub_reward["evidence"],
            "sub_retrieval_reward": sub_retrieval_reward,
            "sub_read_precision": sub_read_precision,
            "no_duplicate_read": no_duplicate_read,
            "action_valid": action_valid,
        }

    def run_episode(self, task: HotpotTask):
        candidates = [self.generate_candidate(task) for _ in range(self.config.group_size)]
        candidates.sort(key=self.candidate_key, reverse=True)
        return candidates[0]

    @staticmethod
    def group_advantages(candidates, reward_key: str, clip: float):
        values = [cand[reward_key] for cand in candidates]
        mean = sum(values) / max(len(values), 1)
        var = sum((value - mean) ** 2 for value in values) / max(len(values), 1)
        std = max(var ** 0.5, 1e-6)
        for cand, value in zip(candidates, values):
            adv = (value - mean) / std
            cand[f"{reward_key}_advantage"] = max(min(adv, clip), -clip)
        return candidates

    def run_group(self, task: HotpotTask):
        candidates = [self.generate_candidate(task) for _ in range(self.config.group_size)]
        candidates = self.group_advantages(candidates, "reward", self.advantage_clip)
        candidates = self.group_advantages(candidates, "sub_train_reward", self.advantage_clip)
        candidates.sort(key=self.candidate_key, reverse=True)
        return candidates

    def apply_best_of_update(self, best):
        main_updates, sub_updates = 0, 0
        if self.train_main and best["reward"] >= self.config.reward_threshold:
            self.model.sft_step(SharedModel.MAIN_ADAPTER, best["plan_prompt"], best["plan_raw"], weight=best["reward"])
            self.model.sft_step(SharedModel.MAIN_ADAPTER, best["answer_prompt"], best["answer_raw"], weight=best["reward"])
            main_updates += 1

        if self.train_sub and best["sub_train_reward"] >= self.config.reward_threshold:
            for prompt, action, _ok, _observation in best["sub_action_steps"]:
                self.model.sft_step(SharedModel.SUB_ADAPTER, prompt, action, weight=best["sub_train_reward"])
            self.model.sft_step(SharedModel.SUB_ADAPTER, best["sub_summary_prompt"], best["sub_summary"], weight=best["sub_train_reward"])
            sub_updates += 1
        return main_updates, sub_updates

    def apply_advantage_update(self, candidates):
        main_updates, sub_updates = 0, 0
        for cand in candidates:
            main_adv = cand["reward_advantage"]
            sub_adv = cand["sub_train_reward_advantage"]
            if self.train_main and abs(main_adv) >= self.min_advantage:
                self.model.sft_step(SharedModel.MAIN_ADAPTER, cand["plan_prompt"], cand["plan_raw"], weight=main_adv)
                self.model.sft_step(SharedModel.MAIN_ADAPTER, cand["answer_prompt"], cand["answer_raw"], weight=main_adv)
                main_updates += 1
            if self.train_sub and abs(sub_adv) >= self.min_advantage:
                for prompt, action, _ok, _observation in cand["sub_action_steps"]:
                    self.model.sft_step(SharedModel.SUB_ADAPTER, prompt, action, weight=sub_adv)
                self.model.sft_step(SharedModel.SUB_ADAPTER, cand["sub_summary_prompt"], cand["sub_summary"], weight=sub_adv)
                sub_updates += 1
        return main_updates, sub_updates

    def evaluate(self, tasks: List[HotpotTask], samples: int = 1):
        if not tasks:
            return {
                "reward": 0.0,
                "answer_f1": 0.0,
                "evidence": 0.0,
                "tool_valid": 0.0,
                "sub_reward": 0.0,
                "sub_train_reward": 0.0,
                "sub_evidence": 0.0,
                "sub_retrieval_reward": 0.0,
                "sub_read_precision": 0.0,
                "no_duplicate_read": 0.0,
                "action_valid": 0.0,
                "best_reward": 0.0,
                "best_answer_f1": 0.0,
            }
        self.model.model.eval()
        rewards, answers, evidences, valids = [], [], [], []
        sub_rewards, sub_train_rewards, sub_evidences, sub_retrievals = [], [], [], []
        sub_precisions, no_duplicate_reads, action_valids = [], [], []
        best_reward_total = 0.0
        best_answer_total = 0.0
        for task in tasks:
            task_best_reward = 0.0
            task_best_answer = 0.0
            for _ in range(samples):
                cand = self.generate_candidate(task)
                rewards.append(cand["reward"])
                answers.append(cand["answer_f1"])
                evidences.append(cand["evidence"])
                valids.append(cand["tool_valid"])
                sub_rewards.append(cand["sub_reward"])
                sub_train_rewards.append(cand["sub_train_reward"])
                sub_evidences.append(cand["sub_evidence"])
                sub_retrievals.append(cand["sub_retrieval_reward"])
                sub_precisions.append(cand["sub_read_precision"])
                no_duplicate_reads.append(cand["no_duplicate_read"])
                action_valids.append(cand["action_valid"])
                task_best_reward = max(task_best_reward, cand["reward"])
                task_best_answer = max(task_best_answer, cand["answer_f1"])
            best_reward_total += task_best_reward
            best_answer_total += task_best_answer
        self.model.model.train()
        total = max(len(rewards), 1)
        return {
            "reward": sum(rewards) / total,
            "answer_f1": sum(answers) / total,
            "evidence": sum(evidences) / total,
            "tool_valid": sum(valids) / total,
            "sub_reward": sum(sub_rewards) / total,
            "sub_train_reward": sum(sub_train_rewards) / total,
            "sub_evidence": sum(sub_evidences) / total,
            "sub_retrieval_reward": sum(sub_retrievals) / total,
            "sub_read_precision": sum(sub_precisions) / total,
            "no_duplicate_read": sum(no_duplicate_reads) / total,
            "action_valid": sum(action_valids) / total,
            "best_reward": best_reward_total / len(tasks),
            "best_answer_f1": best_answer_total / len(tasks),
        }

    def train(self, train_tasks: List[HotpotTask], val_tasks: List[HotpotTask], iterations: int, eval_samples: int):
        print(f"[hotpotqa-mas-grpo] train={len(train_tasks)} val={len(val_tasks)} iter={iterations}")
        print(f"[hotpotqa-mas-grpo] lr={self.config.lr} group={self.config.group_size} threshold={self.config.reward_threshold}")
        print(
            f"[hotpotqa-mas-grpo] train_main={self.train_main} train_sub={self.train_sub} "
            f"best_metric={self.best_metric} sub_reward_mode={self.sub_reward_mode} "
            f"objective={self.objective} advantage_clip={self.advantage_clip} min_advantage={self.min_advantage}"
        )

        self.model = SharedModel(self.config.base_model, self.config)
        self.model.load_sft_weights()
        self.model.model.train()

        print("\n===== HotpotQA MAS Initial Validation =====")
        init = self.evaluate(val_tasks, samples=eval_samples)
        best_val = self.validation_score(init)
        print(
            f"  [val:init] reward={init['reward']:.3f} best={init['best_reward']:.3f} "
            f"answer_f1={init['answer_f1']:.3f} best_answer={init['best_answer_f1']:.3f} "
            f"evidence={init['evidence']:.3f} sub_reward={init['sub_reward']:.3f} "
            f"sub_train={init['sub_train_reward']:.3f} sub_retrieval={init['sub_retrieval_reward']:.3f} "
            f"sub_precision={init['sub_read_precision']:.3f} no_dup={init['no_duplicate_read']:.3f} "
            f"sub_evidence={init['sub_evidence']:.3f} action_valid={init['action_valid']:.3f} "
            f"tool_valid={init['tool_valid']:.3f}"
        )
        for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
            self.model.save_lora(name, str(self.save_dir / "best" / name))

        for it in range(iterations):
            print(f"\n===== HotpotQA MAS Iter {it + 1}/{iterations} =====")
            rewards, answers, evidences, valids = [], [], [], []
            sub_rewards, sub_train_rewards, sub_retrievals, sub_precisions = [], [], [], []
            main_updates, sub_updates = 0, 0
            for task in train_tasks:
                if self.objective == "advantage":
                    candidates = self.run_group(task)
                    best = candidates[0]
                else:
                    candidates = None
                    best = self.run_episode(task)
                rewards.append(best["reward"])
                answers.append(best["answer_f1"])
                evidences.append(best["evidence"])
                valids.append(best["tool_valid"])
                sub_rewards.append(best["sub_reward"])
                sub_train_rewards.append(best["sub_train_reward"])
                sub_retrievals.append(best["sub_retrieval_reward"])
                sub_precisions.append(best["sub_read_precision"])

                if self.objective == "advantage":
                    group_main_updates, group_sub_updates = self.apply_advantage_update(candidates)
                else:
                    group_main_updates, group_sub_updates = self.apply_best_of_update(best)
                main_updates += group_main_updates
                sub_updates += group_sub_updates

            print(
                f"  train reward={sum(rewards)/max(len(rewards),1):.3f} "
                f"answer_f1={sum(answers)/max(len(answers),1):.3f} "
                f"evidence={sum(evidences)/max(len(evidences),1):.3f} "
                f"tool_valid={sum(valids)/max(len(valids),1):.3f} "
                f"sub={sum(sub_rewards)/max(len(sub_rewards),1):.3f} "
                f"sub_train={sum(sub_train_rewards)/max(len(sub_train_rewards),1):.3f} "
                f"sub_retrieval={sum(sub_retrievals)/max(len(sub_retrievals),1):.3f} "
                f"sub_precision={sum(sub_precisions)/max(len(sub_precisions),1):.3f} "
                f"updates main={main_updates} sub={sub_updates}"
            )

            val = self.evaluate(val_tasks, samples=eval_samples)
            print(
                f"  [val] reward={val['reward']:.3f} best={val['best_reward']:.3f} "
                f"answer_f1={val['answer_f1']:.3f} best_answer={val['best_answer_f1']:.3f} "
                f"evidence={val['evidence']:.3f} sub_reward={val['sub_reward']:.3f} "
                f"sub_train={val['sub_train_reward']:.3f} sub_retrieval={val['sub_retrieval_reward']:.3f} "
                f"sub_precision={val['sub_read_precision']:.3f} no_dup={val['no_duplicate_read']:.3f} "
                f"sub_evidence={val['sub_evidence']:.3f} action_valid={val['action_valid']:.3f} "
                f"tool_valid={val['tool_valid']:.3f}"
            )
            score = self.validation_score(val)
            if score > best_val:
                best_val = score
                print(f"  [best] save best checkpoint ({self.best_metric}={best_val:.3f})")
                for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
                    self.model.save_lora(name, str(self.save_dir / "best" / name))

            for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
                self.model.save_lora(name, str(self.save_dir / f"{name}_step_{it + 1}"))

        print("\n[OK] HotpotQA MAS GRPO complete")


def parse_args():
    parser = argparse.ArgumentParser(description="Train HotpotQA Main/Sub MAS agents.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--sft-dir", default="./artifacts/checkpoints/mas_sft")
    parser.add_argument("--main-lora", default=None)
    parser.add_argument("--sub-lora", default=None)
    parser.add_argument("--save-dir", default="./artifacts/checkpoints/mas_grpo")
    parser.add_argument("--train-jsonl", default="./data/base/train.jsonl")
    parser.add_argument("--val-jsonl", default="./data/base/val.jsonl")
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--val-tasks", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--max-response-len", type=int, default=120)
    parser.add_argument("--sub-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--reward-threshold", type=float, default=0.3)
    parser.add_argument(
        "--best-metric",
        choices=[
            "answer_f1",
            "reward",
            "best_reward",
            "sub_reward",
            "sub_evidence",
            "sub_retrieval",
            "sub_train_reward",
        ],
        default="answer_f1",
    )
    parser.add_argument("--sub-reward-mode", choices=["summary", "retrieval", "mixed", "enhanced"], default="summary")
    parser.add_argument("--objective", choices=["best_of", "advantage"], default="best_of")
    parser.add_argument("--advantage-clip", type=float, default=2.0)
    parser.add_argument("--min-advantage", type=float, default=0.0)
    parser.add_argument("--train-main", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-sub", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    train_env = HotpotQAEnvironment.from_jsonl(args.train_jsonl, limit=args.tasks)
    val_env = HotpotQAEnvironment.from_jsonl(args.val_jsonl, limit=args.val_tasks)
    config = CoTrainConfig(
        base_model=args.base_model,
        sft_dir=args.sft_dir,
        main_lora_path=args.main_lora,
        sub_lora_path=args.sub_lora,
        save_dir=args.save_dir,
        group_size=args.group_size,
        max_response_len=args.max_response_len,
        lr=args.lr,
        reward_threshold=args.reward_threshold,
    )
    HotpotMASGRPOTrainer(
        config,
        sub_steps=args.sub_steps,
        best_metric=args.best_metric,
        train_main=args.train_main,
        train_sub=args.train_sub,
        sub_reward_mode=args.sub_reward_mode,
        objective=args.objective,
        advantage_clip=args.advantage_clip,
        min_advantage=args.min_advantage,
    ).train(
        train_env.tasks, val_env.tasks, iterations=args.iterations, eval_samples=args.eval_samples
    )


if __name__ == "__main__":
    main()

