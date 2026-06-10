"""Shared utilities for HotpotQA-MAS — because copy-paste is not a build system.

Contains:
- save_lora / set_trainable_adapter (one copy, not three)
- tqdm-integrated training loop helpers
- graceful shutdown with signal handler
- .env loader for HF_TOKEN etc.
"""

import os
import shutil
import signal
import sys
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load .env on import — one less thing to forget
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


# ── HF Token ────────────────────────────────────────────────────

def hf_token() -> Optional[str]:
    """Return HF_TOKEN from env.  Logs a polite reminder if missing.

    Without this, you get that wonderful "401 Unauthorized" error
    three layers deep in a HuggingFace stack trace.  You're welcome.
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        # Not fatal — maybe the model is local or you ran huggingface-cli login
        pass
    return token


# ── LoRA save (one true implementation) ─────────────────────────

def save_selected_adapter(model, tokenizer, output_dir: str, adapter_name: str):
    """Save a single PEFT adapter — used by SFT trainer, GRPO trainer, eval.

    Why is this a free function?  Because the previous three copies
    had drifted subtly apart and nobody noticed.  Now there is One.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        model.save_pretrained(output_dir, selected_adapters=[adapter_name])
    except TypeError:
        model.save_pretrained(output_dir, adapter_name=adapter_name)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    # Flatten nested adapter directory (peft sometimes creates adapter_name/ subdir)
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
    """Set only the named LoRA adapter's params to requires_grad=True.

    Version 5.0: revert to correct needle matching.
    """
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_id)
    needle = f".{adapter_id}."
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = needle in name
        else:
            param.requires_grad = False


# ── Graceful shutdown ───────────────────────────────────────────

_shutdown_callbacks: List[callable] = []
_shutdown_triggered = False


def on_shutdown(callback):
    """Register a callback to run on SIGINT/SIGTERM.

    Use this to save checkpoints instead of rage-quitting.
    """
    _shutdown_callbacks.append(callback)


def _handle_shutdown(signum, frame):
    global _shutdown_triggered
    if _shutdown_triggered:
        print("\n[!] Double interrupt — hard exit.  Hope you saved recently.", file=sys.stderr)
        sys.exit(1)
    _shutdown_triggered = True
    sig_name = signal.Signals(signum).name
    print(f"\n[!] Received {sig_name}.  Attempting graceful shutdown...")
    for cb in _shutdown_callbacks:
        try:
            cb()
        except Exception as exc:
            print(f"  [warn] shutdown callback failed: {exc}", file=sys.stderr)
    print("[OK] Shutdown complete.  Your GPU is free.  Go touch grass.")
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_shutdown)
signal.signal(signal.SIGTERM, _handle_shutdown)


# ── Progress bar wrapper ────────────────────────────────────────

def try_tqdm(iterable, desc: str = "", total: Optional[int] = None, **kwargs):
    """tqdm if available, otherwise a silent passthrough.

    Because nothing says "I don't know if my code is running" like
    a blank terminal for 20 minutes.
    """
    try:
        from tqdm import tqdm as _tqdm

        return _tqdm(iterable, desc=desc, total=total, **kwargs)
    except ImportError:
        return iterable


# ── Dry-run guard ───────────────────────────────────────────────

def dry_run_mode() -> bool:
    """Check if HOTPOTQA_DRY_RUN=1 is set in env."""
    return os.getenv("HOTPOTQA_DRY_RUN", "0") == "1"


def dry_run_warning(msg: str = ""):
    """Print a notice that we're in dry-run mode."""
    banner = "=" * 50
    print(f"\n{banner}")
    print("  DRY RUN MODE — no model loaded, no training done")
    if msg:
        print(f"  {msg}")
    print(f"{banner}\n")
