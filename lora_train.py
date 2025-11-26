# lora_train.py
"""
LoRA training script for BERT (Masked LM) using PEFT.

- Compatible with peft >= 0.18.0 where TaskType.TOKEN_CLS is used for Masked LM.
- Trains only LoRA adapters (small). Saves adapter weights + tokenizer to out_dir.
- Expects a plain text corpus file with one (or many) documents per line.

Usage example:
    python lora_train.py \
        --corpus DAPT_Scraped_corpus/wiki_videogame_corpus.txt \
        --base_model bert-base-uncased \
        --out_dir TrainedModels/qbert-lora \
        --epochs 2 \
        --batch_size 8
"""

import argparse
import os
from pathlib import Path
import torch

from datasets import load_dataset
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

from peft import LoraConfig, get_peft_model, TaskType

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA training for BERT (Masked LM)")
    parser.add_argument("--corpus", type=str, required=True, default="DAPT_Scraped_corpus/wiki_videogame_corpus.txt",
                        help="Path to a text file (one document per line) or folder accepted by datasets.load_dataset('text').")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased", help="Base model name or path.")
    parser.add_argument("--out_dir", type=str, default="TrainedModels/qbert-lora", help="Where to save LoRA adapters.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank r.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 (if available).")
    return parser.parse_args()

def train_lora(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Device: {device}")

    # Tokenizer & model (masked LM)
    print(f"[info] Loading tokenizer and model from {args.base_model} ...")
    tokenizer = BertTokenizerFast.from_pretrained(args.base_model, use_fast=True)
    model = BertForMaskedLM.from_pretrained(args.base_model)

    # LoRA config
    # For BERT MLM, use TaskType.TOKEN_CLS (treated as per-token classification)
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["query", "value", "dense"]  # common targets for BERT; adjust if errors
    )

    # apply LoRA on the model
    model = get_peft_model(model, peft_config)

    # show trainable parameters summary
    print("\n[info] Trainable parameters (LoRA adapters):")
    try:
        model.print_trainable_parameters()
    except Exception:
        # fallback printing
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f" Trainable params: {n_trainable} / {n_total} ({100 * n_trainable / n_total:.4f}%)")

    # Dataset loading
    print(f"[info] Loading dataset from {args.corpus} ...")
    # load_dataset('text', data_files={'train': file_or_pattern})
    ds = load_dataset("text", data_files={"train": args.corpus})

    # Tokenization function
    max_length = args.max_length
    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    print("[info] Tokenizing dataset (this can take a while)...")
    tokenized = ds["train"].map(tokenize_fn, batched=True, remove_columns=["text"])

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # Training arguments
    # If GPU not available, lower batch size to avoid OOM
    per_device_batch = args.batch_size
    if device.type == "cpu" and per_device_batch > 4:
        print("[warning] Running on CPU - reducing batch size to 4 for stability.")
        per_device_batch = 4

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=per_device_batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=100,
        save_steps=500,
        save_total_limit=args.save_total_limit,
        fp16=(args.fp16 and torch.cuda.is_available()),
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("[info] Starting training ...")
    trainer.train()

    # Save only LoRA adapters and tokenizer (small)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print(f"[info] Saving LoRA adapters + tokenizer to {args.out_dir} ...")
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print("[success] LoRA training complete. Adapters saved.")

if __name__ == "__main__":
    args = parse_args()
    train_lora(args)
