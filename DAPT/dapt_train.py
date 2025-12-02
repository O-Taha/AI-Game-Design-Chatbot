"""
Domain-Adaptive Pretraining (DAPT)
Entrée : corpus.txt (texte brut, une phrase/ligne)
Sortie : ./TrainedModels/qbert-dapt

Entraînement normal
python dapt_train.py \
    --corpus DAPT/wiki.txt \
    --base_model bert-base-uncased \
    --out_dir TrainedModels/qbert-dapt

Reprendre depuis un checkpoint
python dapt_train.py \
    --resume_checkpoint TrainedModels/qbert-dapt/checkpoint-1500

"""

from pathlib import Path
from transformers import (
    BertTokenizer, BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments
)
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
import os
import json


def plot_loss(log_history, out_dir):
    """Génère un graphique propre de la loss et learning rate"""
    steps = []
    losses = []
    lrs = []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])
        if "learning_rate" in entry:
            lrs.append(entry["learning_rate"])

    if not steps:
        print("⚠️ Impossible de générer le plot : aucune loss trouvée dans log_history")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(steps[:len(losses)], losses, label="Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    if lrs:
        plt.figure(figsize=(10, 6))
        plt.plot(steps[:len(lrs)], lrs, label="Learning Rate")
        plt.xlabel("Steps")
        plt.ylabel("LR")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(out_dir, "lr_curve.png"))
        plt.close()

    print("📈 Courbes de loss et LR sauvegardées.")


def train_dapt(
    corpus_file="DAPT/DAPT_Scraped_corpus/wiki_videogame_corpus.txt",
    base_model="bert-base-uncased",
    out_dir="TrainedModels/qbert-dapt",
    resume_checkpoint=None,
    epochs=3,
    per_device_batch_size=4,
    lr=2e-5,
    max_seq_length=256,
):
    print("\n📚 Chargement du tokenizer et du modèle...")

    tokenizer = BertTokenizer.from_pretrained(
        resume_checkpoint if resume_checkpoint else base_model
    )
    model = BertForMaskedLM.from_pretrained(
        resume_checkpoint if resume_checkpoint else base_model
    )

    print("📖 Chargement du corpus...")
    ds = load_dataset("text", data_files={"train": corpus_file})

    # --- Tokenisation simple et propre ---
    print("✏️ Tokenisation...")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
        )

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # --- Data collator ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        report_to="none",
        fp16=torch.cuda.is_available(),
        weight_decay=0.01,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_accumulation_steps=1,
        push_to_hub=False,
        load_best_model_at_end=False,
    )

    print(f"🚀 Entraînement sur {'GPU' if torch.cuda.is_available() else 'CPU'}...")
    if resume_checkpoint:
        print(f"🔄 Reprise depuis le checkpoint : {resume_checkpoint}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # --- Sauvegarde ---
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"\n✅ Modèle final sauvegardé dans {out_dir}")

    # --- Plot de la loss ---
    plot_loss(trainer.state.log_history, out_dir)

    # Sauvegarde des logs pour inspection
    with open(os.path.join(out_dir, "training_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print("📁 training_log.json sauvegardé.")


if __name__ == "__main__":
    train_dapt()
