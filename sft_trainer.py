"""SFT trainer for Main/Sub LoRA adapters.  Uses unified config.TrainingConfig."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import TrainingConfig
from utils import (
    dry_run_mode,
    dry_run_warning,
    save_selected_adapter,
    set_trainable_adapter,
    try_tqdm,
)


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

        encodings.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
    return encodings


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


def train_lora(model, tokenizer, train_data: List[Dict], config: TrainingConfig, adapter_name: str, output_dir: str):
    print(f"\n{'=' * 60}")
    print(f"Training {adapter_name} Agent LoRA...")
    print(f"{'=' * 60}")
    print(f"Samples: {len(train_data)}")
    print(f"Epochs: {config.sft_epochs}")
    print(f"LR: {config.sft_lr}  Batch: {config.sft_batch_size}  Accum: {config.sft_gradient_accumulation_steps}")

    adapter_id = adapter_name.lower()
    set_trainable_adapter(model, adapter_id)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config.sft_lr,
    )
    model.train()

    num_batches_total = len(train_data) // config.sft_batch_size

    for epoch in range(config.sft_epochs):
        total_loss = 0.0
        num_batches = 0
        accum_count = 0
        indices = torch.randperm(len(train_data)).tolist()

        pbar = try_tqdm(
            range(0, len(train_data), config.sft_batch_size),
            desc=f"Epoch {epoch + 1}/{config.sft_epochs} {adapter_name}",
            total=num_batches_total,
        )
        for i in pbar:
            batch_indices = indices[i : i + config.sft_batch_size]
            if len(batch_indices) < config.sft_batch_size:
                continue

            input_ids_list = [train_data[idx]["input_ids"] for idx in batch_indices]
            attention_mask_list = [train_data[idx]["attention_mask"] for idx in batch_indices]
            label_ids_list = [train_data[idx]["labels"] for idx in batch_indices]
            max_len = max(ids.shape[0] for ids in input_ids_list)

            padded_input_ids, padded_attention_mask, labels_list = [], [], []
            for ids, mask, labels in zip(input_ids_list, attention_mask_list, label_ids_list):
                pad_len = max_len - ids.shape[0]
                padded_input_ids.append(
                    torch.cat([ids, torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)])
                )
                padded_attention_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)]))
                labels_list.append(torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)]))

            input_ids = torch.stack(padded_input_ids)
            attention_mask = torch.stack(padded_attention_mask)
            labels = torch.stack(labels_list)
            # Move to the same device as the model's first parameter
            # For device_map models, use the device of the embedding layer
            device = model.get_input_embeddings().weight.device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if (shift_labels != -100).sum().item() == 0:
                continue

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ) / max(config.sft_gradient_accumulation_steps, 1)

            if not torch.isfinite(loss):
                continue

            loss.backward()
            total_loss += loss.item()
            accum_count += 1

            if accum_count % config.sft_gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                num_batches += 1
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(loss=f"{loss.item() * config.sft_gradient_accumulation_steps:.3f}", step=num_batches)

        # remaining accumulated gradients
        if accum_count % config.sft_gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{config.sft_epochs} - Loss: {avg_loss:.4f}")

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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--main-lora", default=None, help="Optional Main LoRA path to continue training from.")
    parser.add_argument("--sub-lora", default=None, help="Optional Sub LoRA path to continue training from.")
    parser.add_argument("--train-main", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--train-sub", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-4bit", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Validate data loading and config without loading the model.")
    args = parser.parse_args()

    if args.dry_run or dry_run_mode():
        dry_run_warning("Testing data loading and config only.")
        main_samples, sub_samples = load_sft_data(args.data_path)
        print(f"  Main Agent: {len(main_samples)} samples")
        print(f"  Sub Agent: {len(sub_samples)} samples")
        print(f"  Config: epochs={args.epochs} lr={args.lr} batch={args.batch_size} ")
        print("  [dry-run] All checks passed.  Ready for real training.")
        return

    config = TrainingConfig(
        base_model=args.base_model,
        sft_dir=args.save_dir,
        sft_lr=args.lr,
        sft_epochs=args.epochs,
        sft_max_length=args.max_length,
        sft_batch_size=args.batch_size,
        sft_gradient_accumulation_steps=args.gradient_accumulation_steps,
        sft_use_4bit=args.use_4bit,
        main_lora_path=args.main_lora,
        sub_lora_path=args.sub_lora,
        device="cuda:4" if torch.cuda.is_available() else "cpu",
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
    if config.sft_use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    # Use GPUs 4,5,6 with manual device_map (avoid 0-3 used by vLLM)
    # CRITICAL: include model.rotary_emb to avoid device mismatch in Qwen3.5
    num_layers = 32
    device_map = {
        "model.embed_tokens": 4,
        "model.rotary_emb": 4,
        "model.norm": 6,
        "lm_head": 6,
    }
    for i in range(num_layers):
        if i < num_layers // 3:
            device_map[f"model.layers.{i}"] = 4
        elif i < 2 * num_layers // 3:
            device_map[f"model.layers.{i}"] = 5
        else:
            device_map[f"model.layers.{i}"] = 6
    print(f"[system] Using device_map across GPUs 4,5,6 ({len(device_map)} entries)")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map=device_map,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    main_samples, sub_samples = load_sft_data(args.data_path)
    print("[system] Loaded SFT data:")
    print(f"  Main Agent: {len(main_samples)} samples")
    print(f"  Sub Agent: {len(sub_samples)} samples")

    print("[system] Preparing training tensors...")
    main_train_data = prepare_training_data(main_samples, tokenizer, max_length=config.sft_max_length)
    sub_train_data = prepare_training_data(sub_samples, tokenizer, max_length=config.sft_max_length)

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
        model = add_or_load_adapter(model, lora_config, "main", config.main_lora_path)
        model.print_trainable_parameters()
        train_lora(model, tokenizer, main_train_data, config, "Main",
                   os.path.join(config.sft_dir, "main_agent"))

    if args.train_sub:
        print("\n[system] Adding/loading Sub Agent LoRA adapter...")
        model = add_or_load_adapter(model, lora_config, "sub", config.sub_lora_path)
        train_lora(model, tokenizer, sub_train_data, config, "Sub",
                   os.path.join(config.sft_dir, "sub_agent"))

    print("\n[system] SFT training complete.")
    print(f"[system] Weights saved to: {config.sft_dir}")


if __name__ == "__main__":
    main()
