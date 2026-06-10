"""Unified configuration for HotpotQA-MAS training pipeline."""

import os
from dataclasses import dataclass

import torch


def _find_local_model(model_id: str = "Qwen/Qwen3.5-9B") -> str:
    """Resolve model path: local cache first, then fall back to HF model id.

    Checks HuggingFace cache and a few common local paths to avoid
    re-downloading 19GB of model weights every time someone reformats
    their laptop.
    """
    org, name = model_id.split("/")
    hf_cache = os.path.expanduser(f"~/.cache/huggingface/hub/models--{org}--{name}")
    local_paths = [
        hf_cache,
        os.path.expanduser(f"~/.cache/tiny-agents/models/{org}/{name}"),
        os.path.expanduser(f"~/.cache/modelscope/hub/{org}/{name}"),
    ]
    for path in local_paths:
        if os.path.exists(os.path.join(path, "config.json")):
            return path
    return model_id  # YOLO — HF will download it


@dataclass
class TrainingConfig:
    """Single source of truth for all training hyperparameters.

    Used by both SFT trainer and GRPO trainer. Fields are grouped by
    training stage; unused fields are simply ignored by each trainer.
    """

    # ── Model ────────────────────────────────────────────────────
    base_model: str = _find_local_model("Qwen/Qwen3.5-9B")
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ── LoRA ─────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # ── SFT stage ────────────────────────────────────────────────
    sft_dir: str = "./artifacts/checkpoints/sft"
    sft_lr: float = 3e-4
    sft_epochs: int = 3
    sft_batch_size: int = 1
    sft_gradient_accumulation_steps: int = 4
    sft_max_length: int = 1024
    sft_use_4bit: bool = False

    # ── GRPO stage ───────────────────────────────────────────────
    main_lora_path: str | None = None
    sub_lora_path: str | None = None
    save_dir: str = "./artifacts/checkpoints/grpo"
    grpo_lr: float = 5e-6
    group_size: int = 2
    reward_threshold: float = 0.3
    max_response_len: int = 160
    max_train_length: int = 1536
    gradient_accumulation_steps: int = 4

    # ── Logging ──────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "hotpotqa-mas"
    wandb_run_name: str | None = None
    log_interval: int = 10

    # ── Checkpointing ────────────────────────────────────────────
    resume_from_checkpoint: str | None = None
    save_every_steps: int = 500
