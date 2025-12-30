# lora_train.py
"""
LoRA training script for BERT (Masked LM) using PEFT.

- Compatible with peft >= 0.18.0 where TaskType.TOKEN_CLS is used for Masked LM.
- Trains only LoRA adapters (small). Saves adapter weights + tokenizer to out_dir.
- Expects a plain text corpus file with one (or many) documents per line.

Usage example:
    python lora_train.py \
        --corpus ../DAPT_Scraped_corpus/wiki_videogame_corpus.txt \
        --base_model bert-base-uncased \
        --out_dir ../TrainedModels/qbert-lora \
        --epochs 2 \
        --batch_size 8

LoRA training script for BERT (Masked LM) using PEFT.
Compatible avec un entraînement *from scratch* ou *continue training* depuis un checkpoint LoRA existant.
"""

import argparse
import json
from pathlib import Path
import torch
from typing import Tuple

from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from huggingface_hub import hf_hub_download

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# optionally support safetensors
try:
    from safetensors.torch import load_file as safetensors_load
    HAS_SAFETENSORS = True
except Exception:
    HAS_SAFETENSORS = False


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA training for BERT (Masked LM)")
    parser.add_argument("--corpus", type=str, required=True,
                        help="Path to text corpus file.")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased",
                        help="Either a HF model name, or an existing LoRA checkpoint dir (e.g. TrainedModels/qbert-lora/checkpoint-1146).")
    parser.add_argument("--out_dir", type=str, default="TrainedModels/qbert-lora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def _count_trainable_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def robust_load_adapter_state(model: torch.nn.Module, adapter_path: Path):
    """
    Load adapter weights into model. Supports safetensors or torch .bin files.
    We try to load the state dict and then call load_state_dict on the peft model.
    """
    adapter_file_safetensors = adapter_path / "adapter_model.safetensors"
    adapter_file_bin = adapter_path / "adapter_model.bin"
    adapter_state = None

    if adapter_file_safetensors.exists():
        if not HAS_SAFETENSORS:
            raise RuntimeError("adapter_model.safetensors found but safetensors is not installed. pip install safetensors")
        print("[info] Loading adapter weights (safetensors)...")
        adapter_state = safetensors_load(str(adapter_file_safetensors))
    elif adapter_file_bin.exists():
        print("[info] Loading adapter weights (torch .bin)...")
        adapter_state = torch.load(str(adapter_file_bin), map_location="cpu")
    else:
        # Maybe saved with another name in checkpoint root
        # try any file that contains "adapter" in name
        for f in adapter_path.iterdir():
            if "adapter" in f.name and f.is_file():
                if f.suffix == ".safetensors":
                    if not HAS_SAFETENSORS:
                        raise RuntimeError("Found safetensors but not installed.")
                    adapter_state = safetensors_load(str(f))
                    break
                else:
                    adapter_state = torch.load(str(f), map_location="cpu")
                    break

    if adapter_state is None:
        raise FileNotFoundError(f"No adapter model file found in {adapter_path} (looked for adapter_model.safetensors/adapter_model.bin).")

    # adapter_state might be a dict of numpy/tensors; ensure keys map
    # Load into model with strict=False to avoid exact key mismatches
    model.load_state_dict(adapter_state, strict=False)
    print("[info] Adapter weights loaded (strict=False).")


def attach_peft_and_maybe_load(base_model_path: str, checkpoint_path: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    """
    Robust loader:
    - If checkpoint_path contains adapter_config.json -> resume mode:
        * read adapter_config to get backbone name (base_model_name_or_path)
        * load backbone model
        * create peft model via get_peft_model using config from adapter_config (fallback to provided args)
        * load adapter weights from adapter_model.safetensors or .bin
    - Else: start fresh from base_model_path and attach new LoRA config
    Returns: model, tokenizer
    """
    cp = Path(checkpoint_path)
    is_checkpoint = cp.exists() and any(cp.iterdir())

    if is_checkpoint and (cp / "adapter_config.json").exists():
        print(f"[info] Resuming from LoRA checkpoint directory: {checkpoint_path}")

        # read adapter config to infer base model and lora params
        cfg = json.load(open(cp / "adapter_config.json", "r", encoding="utf-8"))
        backbone = cfg.get("base_model_name_or_path", None) or cfg.get("pretrained_model_name_or_path", None)

        if backbone is None:
            # fallback: if base_model_path provided, use it
            backbone = base_model_path

        # load tokenizer & backbone
        # Tokenizer might be saved in checkpoint dir; prefer it
        tokenizer_dir = cp if (cp / "tokenizer_config.json").exists() else backbone
        tokenizer = BertTokenizerFast.from_pretrained(str(tokenizer_dir))

        base_model = BertForMaskedLM.from_pretrained(backbone)

        # Reconstruct LoraConfig from adapter_config.json where possible
        # adapter_config.json keys vary; we map known fields
        # Use defaults if missing
        lora_cfg = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=cfg.get("r", lora_r),
            lora_alpha=cfg.get("lora_alpha", lora_alpha),
            lora_dropout=cfg.get("lora_dropout", lora_dropout),
            bias=cfg.get("bias", "none"),
            target_modules=cfg.get("target_modules", ["query", "value", "key", "dense"]),
        )

        # Attach PEFT structure (creates the adapter modules)
        model = get_peft_model(base_model, lora_cfg)

        # Load adapter weights
        try:
            robust_load_adapter_state(model, cp)
        except Exception as e:
            print(f"[warning] Failed to load adapter weights via robust loader: {e}")
            # fallback: try PeftModel.from_pretrained
            try:
                print("[info] Fallback to PeftModel.from_pretrained(...)")
                model = PeftModel.from_pretrained(base_model, checkpoint_path)
            except Exception as e2:
                raise RuntimeError(f"Unable to load LoRA checkpoint: {e2}") from e

    else:
        # fresh training from base_model_path
        print(f"[info] Starting NEW LoRA training from base model: {base_model_path}")
        tokenizer = BertTokenizerFast.from_pretrained(base_model_path)
        base_model = BertForMaskedLM.from_pretrained(base_model_path)

        lora_cfg = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["query", "value", "key", "dense"],
        )

        model = get_peft_model(base_model, lora_cfg)

    # Ensure only adapter params are trainable (and set requires_grad correctly)
    # First freeze all params
    for p in model.parameters():
        p.requires_grad = False

    # Then enable gradient for parameters that contain 'lora' OR typical adapter names
    enabled = 0
    for name, p in model.named_parameters():
        lname = name.lower()
        if "lora" in lname or "adapter" in lname or "alpha" in lname:
            p.requires_grad = True
            enabled += p.numel()

    # As fallback: if enabled == 0 but model has attribute 'get_peft_config', try PeftModel.print_trainable_parameters
    if enabled == 0:
        try:
            # If it's a PeftModel with adapters attached, call its helper
            model.print_trainable_parameters()
            # Some PEFT versions manage requires_grad internally; trust print output
        except Exception:
            pass

    trainable, total = _count_trainable_params(model)
    print(f"[info] trainable params: {trainable} || all params: {total} || trainable%: {100*trainable/total:.6f}")

    return model, tokenizer


def train_lora(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")

    model, tokenizer = attach_peft_and_maybe_load(args.base_model, args.base_model, args.lora_r, args.lora_alpha, args.lora_dropout)

    # Move model to device
    model.to(device)

    # Print trainable params (again)
    try:
        model.print_trainable_parameters()
    except:
        pass

    # Load dataset
    ds = load_dataset("text", data_files={"train": args.corpus})

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length
        )

    tokenized = ds["train"].map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    per_device_batch = args.batch_size
    if device.type == "cpu" and per_device_batch > 4:
        per_device_batch = 4

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=per_device_batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=100,
        save_steps=500,
        max_grad_norm = 1.0,
        save_total_limit=args.save_total_limit,
        fp16=(args.fp16 and torch.cuda.is_available()),
        remove_unused_columns=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("[info] Starting training…")
    trainer.train()

    # Save adapters and tokenizer
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[info] Saving LoRA adapters + tokenizer to {args.out_dir}")
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("[success] Training finished.")


if __name__ == "__main__":
    args = parse_args()
    train_lora(args)