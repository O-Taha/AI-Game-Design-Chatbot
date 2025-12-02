    """
    Domain-Adaptive PreTraining (DAPT) for Q*bert
    Entrée : corpus.txt (une phrase/ligne)
    Sortie : ./TrainedModels/qbert-dapt
    """

    from pathlib import Path
    from transformers import (
        BertTokenizer, BertForMaskedLM,
        DataCollatorForLanguageModeling,
        Trainer, TrainingArguments
    )
    from datasets import load_dataset
    from tqdm.auto import tqdm
    import torch

    def train_dapt(
        corpus_file="DAPT/DAPT_Scraped_corpus/wiki_videogame_corpus.txt",
        base_model="bert-base-uncased",  # "bert-large-uncased" si bon GPU
        out_dir="TrainedModels/qbert-dapt",
        epochs=2,
        per_device_batch_size=4,
        lr=2e-5,
        max_seq_length=256
    ):
        print("📚 Chargement du tokenizer et du modèle...")
        tokenizer = BertTokenizer.from_pretrained(base_model)
        model = BertForMaskedLM.from_pretrained(base_model)

        print("📖 Chargement du corpus...")
        ds = load_dataset("text", data_files={"train": corpus_file})

        # --- Tokenisation avec barre de progression ---
        print("✏️ Tokenisation du corpus (cela peut prendre un peu de temps)...")

        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_seq_length
            )

        # Application manuelle de tqdm sur la tokenisation
        num_rows = ds["train"].num_rows
        chunk_size = max(1, num_rows // 100)  # pour avoir une barre fluide

        tokenized_batches = []
        for i in tqdm(range(0, num_rows, chunk_size), desc="🔠 Tokenisation"):
            batch = ds["train"][i:i + chunk_size]
            tokenized_batches.append(tokenize_fn(batch))

        # Fusion des batches
        from datasets import Dataset
        tokenized = Dataset.from_dict({
            key: sum([b[key] for b in tokenized_batches], [])
            for key in tokenized_batches[0].keys()
        })

        # --- Data collator ---
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        # --- Arguments d'entraînement ---
        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=per_device_batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            save_total_limit=2,
            logging_steps=200,
            report_to="none",
            fp16=torch.cuda.is_available(),
            weight_decay=0.01,
            dataloader_pin_memory=False,
            push_to_hub=False
        )

        print(f"🚀 Entraînement lancé sur {'GPU' if torch.cuda.is_available() else 'CPU'}...")

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=data_collator
        )

        trainer.train()

        # --- Sauvegarde du modèle ---
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        trainer.save_model(out_dir)
        tokenizer.save_pretrained(out_dir)
        print(f"\n✅ Modèle sauvegardé dans {out_dir}")

    if __name__ == "__main__":
        train_dapt()
