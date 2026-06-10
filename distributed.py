"""Lightweight multi-GPU support via PyTorch DDP / FSDP.

Usage:
    # DDP with 4 GPUs
    torchrun --nproc_per_node=4 distributed.py --config ...

    # FSDP (larger models)
    torchrun --nproc_per_node=4 distributed.py --fsdp ...

Designed to wrap SharedModel for multi-GPU inference during rollout
and optionally distribute training.
"""

import argparse
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from config import TrainingConfig
from grpo_v4 import SharedModel


class HotpotDataset(Dataset):
    """Simple dataset wrapping HotpotQA tasks for DataLoader."""

    def __init__(self, tasks: list):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]


def setup_distributed():
    """Initialize NCCL process group. Returns (rank, world_size, local_rank)."""
    if "RANK" not in os.environ:
        return 0, 1, 0  # single GPU

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


class DistributedSharedModel:
    """Wrapper that adds DDP/FSDP to SharedModel for multi-GPU training.

    Only the main process handles generation/rollout. All processes
    participate in gradient computation during training steps.
    """

    def __init__(
        self,
        base_model: str,
        config: TrainingConfig,
        use_fsdp: bool = False,
        local_rank: int = 0,
    ):
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.local_rank = local_rank
        self.is_main = self.rank == 0
        self.use_fsdp = use_fsdp

        if self.is_main:
            self._shared = SharedModel(base_model, config)
        else:
            self._shared = None

        barrier()
        # Sync: all ranks create their own SharedModel
        if not self.is_main:
            self._shared = SharedModel(base_model, config)

        # Load weights on all ranks
        self._shared.load_sft_weights()

        if use_fsdp:
            self._wrap_fsdp()
        else:
            self._wrap_ddp()

    @property
    def model(self):
        return self._ddp_model if hasattr(self, "_ddp_model") else self._shared.model

    @property
    def tokenizer(self):
        return self._shared.tokenizer

    @property
    def config(self):
        return self._shared.config

    def _wrap_ddp(self):
        """Wrap model with DDP, keeping LoRA adapters."""
        # DDP wraps the base model; PEFT handles adapter routing
        self._ddp_model = DDP(
            self._shared.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True,  # PEFT may leave some params unused
        )

    def _wrap_fsdp(self):
        """FSDP wrapping (requires PyTorch >= 2.0)."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import _module_wrap_policy

        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        self._ddp_model = FSDP(
            self._shared.model,
            device_id=self.local_rank,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            auto_wrap_policy=_module_wrap_policy,
        )

    def generate_one(self, adapter_name: str, prompt: str, max_tokens: int, **kwargs) -> str:
        """Only main process generates."""
        if self.is_main:
            return self._shared.generate_one(adapter_name, prompt, max_tokens, **kwargs)
        return ""

    def generate_batch(self, adapter_name: str, prompts: list, max_tokens: int, **kwargs) -> list:
        if self.is_main:
            return self._shared.generate_batch(adapter_name, prompts, max_tokens, **kwargs)
        return [""] * len(prompts)

    def response_token_logprobs(self, adapter_name: str, prompt: str, response: str, with_grad: bool = False):
        return self._shared.response_token_logprobs(adapter_name, prompt, response, with_grad=with_grad)

    def grpo_backward(self, adapter_name: str, prompt: str, response: str, old_logprobs,
                      reference_logprobs, advantage: float, **kwargs) -> dict:
        return self._shared.grpo_backward(
            adapter_name, prompt, response, old_logprobs, reference_logprobs, advantage, **kwargs,
        )

    def sft_backward(self, adapter_name: str, prompt: str, response: str, weight: float = 1.0) -> float:
        return self._shared.sft_backward(adapter_name, prompt, response, weight=weight)

    def optimizer_zero_grad(self, adapter_name: str):
        self._shared.optimizer_zero_grad(adapter_name)

    def optimizer_step(self, adapter_name: str):
        self._shared.optimizer_step(adapter_name)

    def _maybe_optimizer_step(self, adapter_name: str):
        self._shared._maybe_optimizer_step(adapter_name)

    def sft_step(self, adapter_name: str, prompt: str, response: str, weight: float = 1.0) -> float:
        return self._shared.sft_step(adapter_name, prompt, response, weight=weight)

    def grpo_step(self, adapter_name: str, prompt: str, response: str,
                  reference_logprobs, advantage: float, **kwargs) -> dict:
        return self._shared.grpo_step(
            adapter_name, prompt, response, reference_logprobs, advantage, **kwargs,
        )

    def save_checkpoint(self, output_dir: str):
        if self.is_main:
            self._shared.save_checkpoint(output_dir)

    def load_checkpoint(self, checkpoint_dir: str):
        self._shared.load_checkpoint(checkpoint_dir)

    def save_lora(self, adapter_name: str, output_dir: str):
        if self.is_main:
            self._shared.save_lora(adapter_name, output_dir)

    def reference_adapter(self, adapter_name: str) -> str:
        return self._shared.reference_adapter(adapter_name)

    def set_trainable_adapter(self, adapter_name: str):
        self._shared.set_trainable_adapter(adapter_name)

    def train(self):
        self._shared.model.train()

    def eval(self):
        self._shared.model.eval()

    def global_step(self, adapter_name: str) -> int:
        return self._shared.global_step(adapter_name)


def parse_distributed_args():
    p = argparse.ArgumentParser(description="Distributed HotpotQA-MAS trainer.")
    p.add_argument("--fsdp", action="store_true", help="Use FSDP instead of DDP.")
    p.add_argument("--config", default=None, help="Config override (JSON file).")
    return p.parse_known_args()[0]
