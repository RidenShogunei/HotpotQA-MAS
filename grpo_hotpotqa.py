"""GRPO-style training for local HotpotQA search/read trajectories."""

import argparse
import re
from pathlib import Path
from typing import List

from generate_hotpotqa_sft_data import ANSWER_SYSTEM, MAIN_SYSTEM, SUB_SYSTEM
from grpo_v4 import CoTrainConfig, SharedModel
from hotpotqa_environment import HotpotQAEnvironment, HotpotTask


class HotpotGRPOTrainer:
    def __init__(
        self,
        config: CoTrainConfig,
        research_steps: int = 3,
        best_metric: str = "answer_f1",
        train_main: bool = True,
        train_sub: bool = True,
    ):
        self.config = config
        self.research_steps = research_steps
        self.best_metric = best_metric
        self.train_main = train_main
        self.train_sub = train_sub
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = None

    @staticmethod
    def history_text(history):
        if not history:
            return "No observations yet."
        lines = []
        for idx, (tool_call, observation) in enumerate(history, 1):
            lines.append(f"Step {idx} tool call: {tool_call}")
            lines.append(f"Step {idx} observation: {observation}")
        return "\n".join(lines)

    @staticmethod
    def extract_tool_call_block(text: str) -> str:
        match = re.search(r"\[tool_call\].*?\[/tool_call\]", text, re.DOTALL)
        return match.group(0) if match else text

    @staticmethod
    def truncate_result(text: str) -> str:
        end = text.find("</result>")
        if end >= 0:
            return text[:end + len("</result>")]
        return text

    def build_action_prompt(self, task: HotpotTask, history) -> str:
        return self.model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": MAIN_SYSTEM},
                {"role": "user", "content": f"Question: {task.question}\nResearch history:\n{self.history_text(history)}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_answer_prompt(self, task: HotpotTask, history) -> str:
        return self.model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": f"Question: {task.question}\nResearch history:\n{self.history_text(history)}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def build_sub_prompt(self, tool_call: str) -> str:
        return self.model.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SUB_SYSTEM},
                {"role": "user", "content": tool_call},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def sub_reward(expected_output: str, response: str) -> float:
        expected = HotpotQAEnvironment.normalize(expected_output)
        got = HotpotQAEnvironment.normalize(response)
        return 1.0 if expected[:160] and expected[:160] in got else 0.0

    def candidate_key(self, candidate):
        return (
            candidate["answer_f1"],
            candidate["reward"],
            candidate["evidence"],
            candidate["tool_valid"],
        )

    def validation_score(self, metrics):
        if self.best_metric == "answer_f1":
            return metrics["answer_f1"]
        if self.best_metric == "reward":
            return metrics["reward"]
        if self.best_metric == "best_reward":
            return metrics["best_reward"]
        raise ValueError(f"Unknown best_metric: {self.best_metric}")

    def generate_candidate(self, task: HotpotTask):
        history = []
        action_steps = []
        ok_any = False
        for _ in range(self.research_steps):
            prompt = self.build_action_prompt(task, history)
            raw = self.model.generate_one(
                SharedModel.MAIN_ADAPTER,
                prompt,
                max_tokens=self.config.max_response_len,
                response_prefix="<thinking>",
                canonicalizer=None,
            )
            raw = self.truncate_result(raw)
            tool_call = self.extract_tool_call_block(raw)
            ok, observation = HotpotQAEnvironment.execute_tool(task, tool_call)
            if not ok:
                observation = "Tool execution failed"
            ok_any = ok_any or ok
            history.append((tool_call, observation))
            action_steps.append((prompt, tool_call, ok, observation))

        answer_prompt = self.build_answer_prompt(task, history)
        answer_raw = self.model.generate_one(
            SharedModel.MAIN_ADAPTER,
            answer_prompt,
            max_tokens=self.config.max_response_len,
            response_prefix="<thinking>",
            canonicalizer=None,
        )
        answer_raw = self.truncate_result(answer_raw)
        combined = "".join(step[1] for step in action_steps) + answer_raw
        reward = HotpotQAEnvironment.reward(task, combined)
        return {
            "action_steps": action_steps,
            "answer_prompt": answer_prompt,
            "answer_raw": answer_raw,
            "raw": combined,
            "reward": reward["total"],
            "answer_f1": reward["answer_f1"],
            "evidence": reward["evidence"],
            "tool_valid": reward["tool_valid"],
            "tool_ok": 1.0 if ok_any else 0.0,
        }

    def run_episode(self, task: HotpotTask):
        candidates = [self.generate_candidate(task) for _ in range(self.config.group_size)]
        candidates.sort(key=self.candidate_key, reverse=True)
        best = candidates[0]

        sub_candidates = []
        for _prompt, tool_call, ok, observation in best["action_steps"]:
            if not ok:
                continue
            prompt_s = self.build_sub_prompt(tool_call)
            target = f"<thinking>execute tool</thinking><result>{observation}</result>"
            sub_candidates.append((prompt_s, target, 1.0))
        return best, sub_candidates

    def evaluate(self, tasks: List[HotpotTask], samples: int = 1):
        if not tasks:
            return {"reward": 0.0, "answer_f1": 0.0, "evidence": 0.0, "tool_valid": 0.0, "best_reward": 0.0}

        self.model.model.eval()
        rewards, answers, evidences, valids = [], [], [], []
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
            "best_reward": best_reward_total / len(tasks),
            "best_answer_f1": best_answer_total / len(tasks),
        }

    def train(self, train_tasks: List[HotpotTask], val_tasks: List[HotpotTask], iterations: int, eval_samples: int):
        print(f"[hotpotqa-grpo] train={len(train_tasks)} val={len(val_tasks)} iter={iterations}")
        print(f"[hotpotqa-grpo] lr={self.config.lr} group={self.config.group_size} threshold={self.config.reward_threshold}")
        print(f"[hotpotqa-grpo] train_main={self.train_main} train_sub={self.train_sub} best_metric={self.best_metric}")

        self.model = SharedModel(self.config.base_model, self.config)
        self.model.load_sft_weights()
        self.model.model.train()

        print("\n===== HotpotQA Initial Validation =====")
        init = self.evaluate(val_tasks, samples=eval_samples)
        best_val = self.validation_score(init)
        print(
            f"  [val:init] reward={init['reward']:.3f} best={init['best_reward']:.3f} "
            f"answer_f1={init['answer_f1']:.3f} best_answer={init['best_answer_f1']:.3f} "
            f"evidence={init['evidence']:.3f} tool_valid={init['tool_valid']:.3f}"
        )
        for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
            self.model.save_lora(name, str(self.save_dir / "best" / name))

        for it in range(iterations):
            print(f"\n===== HotpotQA Iter {it + 1}/{iterations} =====")
            rewards, answers, evidences, valids, sub_rewards = [], [], [], [], []
            updates, sub_updates = 0, 0
            for task in train_tasks:
                best, sub_candidates = self.run_episode(task)
                rewards.append(best["reward"])
                answers.append(best["answer_f1"])
                evidences.append(best["evidence"])
                valids.append(best["tool_valid"])

                if self.train_main and best["reward"] >= self.config.reward_threshold:
                    for prompt, action, _ok, _observation in best["action_steps"]:
                        self.model.sft_step(SharedModel.MAIN_ADAPTER, prompt, action, weight=best["reward"])
                    self.model.sft_step(
                        SharedModel.MAIN_ADAPTER,
                        best["answer_prompt"],
                        best["answer_raw"],
                        weight=best["reward"],
                    )
                    updates += 1

                if sub_candidates:
                    prompt_s, resp_s, rew_s = sub_candidates[0]
                    sub_rewards.append(rew_s)
                    if self.train_sub and rew_s >= self.config.reward_threshold:
                        self.model.sft_step(SharedModel.SUB_ADAPTER, prompt_s, resp_s, weight=rew_s)
                        sub_updates += 1

            print(
                f"  train reward={sum(rewards)/max(len(rewards),1):.3f} "
                f"answer_f1={sum(answers)/max(len(answers),1):.3f} "
                f"evidence={sum(evidences)/max(len(evidences),1):.3f} "
                f"tool_valid={sum(valids)/max(len(valids),1):.3f} updates={updates}/{len(train_tasks)} | "
                f"sub={sum(sub_rewards)/max(len(sub_rewards),1):.3f} sub_updates={sub_updates}"
            )

            val = self.evaluate(val_tasks, samples=eval_samples)
            print(
                f"  [val] reward={val['reward']:.3f} best={val['best_reward']:.3f} "
                f"answer_f1={val['answer_f1']:.3f} best_answer={val['best_answer_f1']:.3f} "
                f"evidence={val['evidence']:.3f} tool_valid={val['tool_valid']:.3f}"
            )
            score = self.validation_score(val)
            if score > best_val:
                best_val = score
                print(f"  [best] save best checkpoint ({self.best_metric}={best_val:.3f})")
                for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
                    self.model.save_lora(name, str(self.save_dir / "best" / name))

            for name in [SharedModel.MAIN_ADAPTER, SharedModel.SUB_ADAPTER]:
                self.model.save_lora(name, str(self.save_dir / f"{name}_step_{it + 1}"))

        print("\n[OK] HotpotQA GRPO complete")


def parse_args():
    parser = argparse.ArgumentParser(description="Train HotpotQA Main/Sub agents with GRPO-style updates.")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--sft-dir", default="./artifacts/checkpoints/sft")
    parser.add_argument("--save-dir", default="./artifacts/checkpoints/direct_grpo")
    parser.add_argument("--train-jsonl", default="./data/base/train.jsonl")
    parser.add_argument("--val-jsonl", default="./data/base/val.jsonl")
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--val-tasks", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--max-response-len", type=int, default=160)
    parser.add_argument("--research-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--reward-threshold", type=float, default=0.3)
    parser.add_argument("--best-metric", choices=["answer_f1", "reward", "best_reward"], default="answer_f1")
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
        save_dir=args.save_dir,
        group_size=args.group_size,
        max_response_len=args.max_response_len,
        lr=args.lr,
        reward_threshold=args.reward_threshold,
    )
    HotpotGRPOTrainer(
        config,
        research_steps=args.research_steps,
        best_metric=args.best_metric,
        train_main=args.train_main,
        train_sub=args.train_sub,
    ).train(
        train_env.tasks, val_env.tasks, iterations=args.iterations, eval_samples=args.eval_samples
    )


if __name__ == "__main__":
    main()

