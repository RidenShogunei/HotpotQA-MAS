"""Shared Main/Sub LoRA model utilities for lightweight GRPO-style trainers."""

import os
import shutil
from dataclasses import dataclass

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def clipped_grpo_objective(
    current_logprobs,
    old_logprobs,
    reference_logprobs,
    advantage: float,
    policy_clip: float,
    kl_beta: float,
):
    ratio = torch.exp((current_logprobs - old_logprobs).clamp(min=-20.0, max=20.0))
    clipped_ratio = ratio.clamp(1.0 - policy_clip, 1.0 + policy_clip)
    advantage_tensor = torch.full_like(ratio, float(advantage))
    policy_loss = -torch.minimum(ratio * advantage_tensor, clipped_ratio * advantage_tensor).mean()
    reference_gap = (reference_logprobs - current_logprobs).clamp(min=-20.0, max=20.0)
    kl = (torch.exp(reference_gap) - reference_gap - 1.0).mean()
    return policy_loss + float(kl_beta) * kl, policy_loss, kl


@dataclass
class CoTrainConfig:
    base_model: str = "Qwen/Qwen3.5-9B"
    sft_dir: str = "./artifacts/checkpoints/sft"
    main_lora_path: str | None = None
    sub_lora_path: str | None = None
    save_dir: str = "./artifacts/checkpoints/grpo"
    lr: float = 5e-6
    group_size: int = 2
    reward_threshold: float = 0.3
    max_response_len: int = 160
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_train_length: int = 1536


class SharedModel:
    MAIN_ADAPTER = "main"
    SUB_ADAPTER = "sub"
    REFERENCE_SUFFIX = "_reference"

    def __init__(self, base_model: str, config: CoTrainConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=True,
            device_map={"": config.device},
            low_cpu_mem_usage=True,
        )
        self.optimizers = {}

    def adapter_path(self, adapter_name: str) -> str | None:
        explicit = self.config.main_lora_path if adapter_name == self.MAIN_ADAPTER else self.config.sub_lora_path
        if explicit:
            return explicit
        candidates = [
            os.path.join(self.config.sft_dir, f"{adapter_name}_agent"),
            os.path.join(self.config.sft_dir, adapter_name),
        ]
        for path in candidates:
            if os.path.exists(os.path.join(path, "adapter_config.json")):
                return path
        return None

    def load_sft_weights(self):
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        for adapter_name in [self.MAIN_ADAPTER, self.SUB_ADAPTER]:
            path = self.adapter_path(adapter_name)
            if path:
                if isinstance(self.model, PeftModel):
                    self.model.load_adapter(path, adapter_name=adapter_name, is_trainable=True)
                else:
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        path,
                        adapter_name=adapter_name,
                        is_trainable=True,
                    )
                self.model.load_adapter(
                    path,
                    adapter_name=self.reference_adapter(adapter_name),
                    is_trainable=False,
                )
            elif isinstance(self.model, PeftModel):
                self.model.add_adapter(adapter_name, lora_config)
            else:
                self.model = get_peft_model(self.model, lora_config, adapter_name=adapter_name)
            self.ensure_optimizer(adapter_name)

    @classmethod
    def reference_adapter(cls, adapter_name: str) -> str:
        return f"{adapter_name}{cls.REFERENCE_SUFFIX}"

    def has_adapter(self, adapter_name: str) -> bool:
        return adapter_name in getattr(self.model, "peft_config", {})

    def set_trainable_adapter(self, adapter_name: str):
        self.model.set_adapter(adapter_name)
        needle = f".{adapter_name}."
        for name, param in self.model.named_parameters():
            param.requires_grad = "lora_" in name and needle in name

    def ensure_optimizer(self, adapter_name: str):
        self.set_trainable_adapter(adapter_name)
        params = [param for param in self.model.parameters() if param.requires_grad]
        self.optimizers[adapter_name] = torch.optim.AdamW(params, lr=self.config.lr)

    def generate_one(
        self,
        adapter_name: str,
        prompt: str,
        max_tokens: int,
        response_prefix: str = "",
        canonicalizer=None,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
    ) -> str:
        self.model.eval()
        self.model.set_adapter(adapter_name)
        full_prompt = prompt + (response_prefix or "")
        encoded = self.tokenizer(full_prompt, return_tensors="pt").to(self.config.device)
        with torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, encoded["input_ids"].shape[-1] :]
        text = (response_prefix or "") + self.tokenizer.decode(generated, skip_special_tokens=True)
        self.model.train()
        return canonicalizer(text) if canonicalizer else text

    def sft_backward(self, adapter_name: str, prompt: str, response: str, weight: float = 1.0) -> float:
        if abs(weight) <= 1e-8:
            return 0.0
        self.set_trainable_adapter(adapter_name)
        text = prompt + response + (self.tokenizer.eos_token or "")
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_train_length,
            return_tensors="pt",
        ).to(self.config.device)
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.max_train_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        labels = encoding["input_ids"].clone()
        prompt_len = min(prompt_encoding["input_ids"].shape[-1], labels.shape[-1])
        labels[:, :prompt_len] = -100
        if (labels != -100).sum().item() == 0:
            return 0.0

        outputs = self.model(**encoding)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = ce_loss * float(weight)
        if not torch.isfinite(loss):
            return 0.0
        loss.backward()
        return float(loss.detach().cpu())

    def response_token_logprobs(
        self,
        adapter_name: str,
        prompt: str,
        response: str,
        with_grad: bool = False,
    ):
        self.model.set_adapter(adapter_name)
        text = prompt + response + (self.tokenizer.eos_token or "")
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config.max_train_length,
            return_tensors="pt",
        ).to(self.config.device)
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config.max_train_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        prompt_len = min(prompt_encoding["input_ids"].shape[-1], encoding["input_ids"].shape[-1])
        context = torch.enable_grad() if with_grad else torch.no_grad()
        self.model.eval()
        with context:
            logits = self.model(**encoding).logits[:, :-1, :]
            labels = encoding["input_ids"][:, 1:]
            token_logprobs = torch.log_softmax(logits, dim=-1).gather(
                dim=-1,
                index=labels.unsqueeze(-1),
            ).squeeze(-1)
            positions = torch.arange(1, encoding["input_ids"].shape[-1], device=self.config.device)
            response_mask = positions >= prompt_len
            selected = token_logprobs[0, response_mask]
        return selected if with_grad else selected.detach().cpu()

    def grpo_backward(
        self,
        adapter_name: str,
        prompt: str,
        response: str,
        old_logprobs,
        reference_logprobs,
        advantage: float,
        policy_clip: float = 0.2,
        kl_beta: float = 0.01,
        weight: float = 1.0,
    ) -> dict:
        if abs(advantage) <= 1e-8 or weight <= 0:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "kl": 0.0,
                "ratio": 1.0,
                "clip_fraction": 0.0,
                "tokens": 0,
            }
        self.set_trainable_adapter(adapter_name)
        current = self.response_token_logprobs(adapter_name, prompt, response, with_grad=True)
        old = old_logprobs.to(current.device)
        reference = reference_logprobs.to(current.device)
        token_count = min(current.numel(), old.numel(), reference.numel())
        if token_count == 0:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "kl": 0.0,
                "ratio": 1.0,
                "clip_fraction": 0.0,
                "tokens": 0,
            }
        current = current[:token_count]
        old = old[:token_count]
        reference = reference[:token_count]

        objective, policy_loss, kl = clipped_grpo_objective(
            current,
            old,
            reference,
            advantage=advantage,
            policy_clip=policy_clip,
            kl_beta=kl_beta,
        )
        loss = objective * float(weight)
        if not torch.isfinite(loss):
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "kl": 0.0,
                "ratio": 1.0,
                "clip_fraction": 0.0,
                "tokens": 0,
            }
        loss.backward()
        with torch.no_grad():
            ratio = torch.exp((current - old).clamp(min=-20.0, max=20.0))
            clipped = (ratio < 1.0 - policy_clip) | (ratio > 1.0 + policy_clip)
        return {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "ratio": float(ratio.mean().detach().cpu()),
            "clip_fraction": float(clipped.float().mean().detach().cpu()),
            "tokens": int(token_count),
        }

    def optimizer_zero_grad(self, adapter_name: str):
        self.optimizers[adapter_name].zero_grad(set_to_none=True)

    def optimizer_step(self, adapter_name: str):
        self.set_trainable_adapter(adapter_name)
        optimizer = self.optimizers[adapter_name]
        torch.nn.utils.clip_grad_norm_((p for p in self.model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    def sft_step(self, adapter_name: str, prompt: str, response: str, weight: float = 1.0) -> float:
        self.optimizer_zero_grad(adapter_name)
        loss = self.sft_backward(adapter_name, prompt, response, weight=weight)
        if loss != 0.0:
            self.optimizer_step(adapter_name)
        return loss

    def save_lora(self, adapter_name: str, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir, selected_adapters=[adapter_name])
        self.tokenizer.save_pretrained(output_dir)
        nested_dir = os.path.join(output_dir, adapter_name)
        if os.path.exists(os.path.join(nested_dir, "adapter_config.json")):
            for item in os.listdir(nested_dir):
                src = os.path.join(nested_dir, item)
                dst = os.path.join(output_dir, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)

