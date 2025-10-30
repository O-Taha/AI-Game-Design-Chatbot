import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from text_embedding import MODEL_NAME
from tqdm import tqdm
import pandas as pd
import os

class GameMechanicDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=128):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = self.tokenizer(
            row["input_text"],
            row["target_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        label = torch.tensor(row["label"], dtype=torch.long)
        return {key: val.squeeze(0) for key, val in inputs.items()}, label


pre_trained_model = MODEL_NAME
def fine_tune_bert(dataset_path="query_mechanics_dataset.csv", model_name=pre_trained_model, epochs=2, batch_size=2, lr=2e-5):
    """
    Entraîne BERT à prédire si une paire (problème, mécanique) correspond.

    - Le modèle est ensuite sauvegardé dans ./fine_tuned_model/
    - Compatible avec text_embedding() via AutoModel.from_pretrained("fine_tuned_model")
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    dataset = GameMechanicDataset(dataset_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss/len(dataloader):.4f}")

    os.makedirs("fine_tuned_model", exist_ok=True)
    model.save_pretrained("fine_tuned_model")
    tokenizer.save_pretrained("fine_tuned_model")
    print("✅ Fine-tuned model saved to ./fine_tuned_model/")


if __name__ == "__main__":
    fine_tune_bert()