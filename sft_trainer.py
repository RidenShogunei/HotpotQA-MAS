"""SFT trainer for Main/Sub LoRA adapters."""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class CoTrainConfig:
    base_model: str = "Qwen/Qwen3.5-9B"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    lr: float = 3e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    save_dir: str = "./sft_checkpoints"
    device: str = "cuda:0"
    use_4bit: bool = False
    max_length: int = 1024


def load_sft_data(data_path: str) -> tuple[list[dict], list[dict]]:
    main_samples = []
    sub_samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item["category"] == "main":
                main_samples.append(item)
            else:
                sub_samples.append(item)
    return main_samples, sub_samples


def prepare_training_data(samples: List[Dict], tokenizer, max_length: int = 512) -> List[Dict]:
    encodings = []
    for sample in samples:
        messages = sample["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("SFT sample must end with an assistant message")

        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        text = prompt_text + messages[-1]["content"] + (tokenizer.eos_token or "")

        encoding = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
        prompt_encoding = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        prompt_len = min(prompt_encoding["input_ids"].shape[-1], labels.shape[0])
        labels[:prompt_len] = -100
        if (labels != -100).sum().item() == 0:
            continue

        encodings.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    return encodings


def save_selected_adapter(model, tokenizer, output_dir: str, adapter_name: str):
    os.makedirs(output_dir, exist_ok=True)
    try:
        model.save_pretrained(output_dir, selected_adapters=[adapter_name])
    except TypeError:
        model.save_pretrained(output_dir, adapter_name=adapter_name)
    tokenizer.save_pretrained(output_dir)

    nested_dir = os.path.join(output_dir, adapter_name)
    if os.path.exists(os.path.join(nested_dir, "adapter_config.json")):
        for item in os.listdir(nested_dir):
            src = os.path.join(nested_dir, item)
            dst = os.path.join(output_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)


def set_trainable_adapter(model, adapter_id: str):
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_id)
    needle = f".{adapter_id}."
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = needle in name
        else:
            param.requires_grad = False


def add_or_load_adapter(model, lora_config: LoraConfig, adapter_id: str, lora_path: str | None):
    if lora_path:
        if isinstance(model, PeftModel):
            model.load_adapter(lora_path, adapter_name=adapter_id, is_trainable=True)
            return model
        return PeftModel.from_pretrained(model, lora_path, adapter_name=adapter_id, is_trainable=True)
    if isinstance(model, PeftModel):
        model.add_adapter(adapter_id, lora_config)
        return model
    return get_peft_model(model, lora_config, adapter_name=adapter_id)


def train_lora(model, tokenizer, train_data: List[Dict], config: CoTrainConfig, adapter_name: str, output_dir: str):
    print(f"\n{'=' * 60}")
    print(f"Training {adapter_name} Agent LoRA...")
    print(f"{'=' * 60}")
    print(f"Samples: {len(train_data)}")
    print(f"Epochs: {config.num_epochs}")
    print(f"LR: {config.lr}")

    adapter_id = adapter_name.lower()
    set_trainable_adapter(model, adapter_id)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.lr)
    model.train()

    for epoch in range(config.num_epochs):
        total_loss = 0.0
        num_batches = 0
        indices = torch.randperm(len(train_data)).tolist()

        for i in range(0, len(train_data), config.batch_size):
            batch_indices = indices[i : i + config.batch_size]
            if len(batch_indices) < config.batch_size:
                continue

            input_ids_list = [train_data[idx]["input_ids"] for idx in batch_indices]
            attention_mask_list = [train_data[idx]["attention_mask"] for idx in batch_indices]
            label_ids_list = [train_data[idx]["labels"] for idx in batch_indices]
            max_len = max(ids.shape[0] for ids in input_ids_list)

            padded_input_ids = []
            padded_attention_mask = []
            labels_list = []
            for ids, mask, labels in zip(input_ids_list, attention_mask_list, label_ids_list):
                pad_len = max_len - ids.shape[0]
                padded_input_ids.append(
                    torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
                )
                padded_attention_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)]))
                labels_list.append(torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)]))

            input_ids = torch.stack(padded_input_ids).to(config.device)
            attention_mask = torch.stack(padded_attention_mask).to(config.device)
            labels = torch.stack(labels_list).to(config.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if (shift_labels != -100).sum().item() == 0:
                continue

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if not torch.isfinite(loss):
                continue

            loss.backward()
            total_loss += loss.item()
            num_batches += 1

            if num_batches % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if num_batches % config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{config.num_epochs} - Loss: {avg_loss:.4f}")

    save_selected_adapter(model, tokenizer, output_dir, adapter_id)
    print(f"LoRA weights saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train Main/Sub LoRA adapters from SFT JSONL data.")
    parser.add_argument("--data-path", default=str(Path(__file__).parent / "sft_data.jsonl"))
    parser.add_argument("--save-dir", default="./artifacts/checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--main-lora", default=None, help="Optional Main LoRA path to continue training from.")
    parser.add_argument("--sub-lora", default=None, help="Optional Sub LoRA path to continue training from.")
    parser.add_argument("--train-main", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-sub", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = CoTrainConfig(
        base_model=args.base_model,
        save_dir=args.save_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        max_length=args.max_length,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n[system] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[system] Loading base model...")
    quantization_config = None
    if config.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map={"": config.device},
        low_cpu_mem_usage=True,
    )

    main_samples, sub_samples = load_sft_data(args.data_path)
    print("[system] Loaded SFT data:")
    print(f"  Main Agent: {len(main_samples)} samples")
    print(f"  Sub Agent: {len(sub_samples)} samples")

    print("[system] Preparing training tensors...")
    main_train_data = prepare_training_data(main_samples, tokenizer, max_length=config.max_length)
    sub_train_data = prepare_training_data(sub_samples, tokenizer, max_length=config.max_length)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    if args.train_main:
        print("\n[system] Adding/loading Main Agent LoRA adapter...")
        model = add_or_load_adapter(model, lora_config, "main", args.main_lora)
        model.print_trainable_parameters()
        train_lora(model, tokenizer, main_train_data, config, "Main", os.path.join(config.save_dir, "main_agent"))

    if args.train_sub:
        print("\n[system] Adding/loading Sub Agent LoRA adapter...")
        model = add_or_load_adapter(model, lora_config, "sub", args.sub_lora)
        train_lora(model, tokenizer, sub_train_data, config, "Sub", os.path.join(config.save_dir, "sub_agent"))

    print("\n[system] SFT training complete.")
    print(f"[system] Weights saved to: {config.save_dir}")


if __name__ == "__main__":
    main()

