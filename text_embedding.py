from transformers import (
    AutoTokenizer,
    AutoModel,
    BertTokenizer,
    BertForMaskedLM
)
from typing import Union
from pathlib import Path
import torch
import numpy as np

# Modèle utilisé
FINE_TUNED_MODEL = "TrainedModels/qbert-ranked"
BASE_MODEL = "bert-base-uncased"

# -----------------------------
# Chargement du modèle d'embedding (BertModel)
# -----------------------------
if Path(FINE_TUNED_MODEL).exists():
    tokenizer_embed = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL)
    model_embed = AutoModel.from_pretrained(FINE_TUNED_MODEL)
    print(f"[Embedding] Loaded merged model: {FINE_TUNED_MODEL}")
else:
    tokenizer_embed = AutoTokenizer.from_pretrained(BASE_MODEL)
    model_embed = AutoModel.from_pretrained(BASE_MODEL)
    print(f"[Embedding] WARNING: merged model not found, fallback to base model.")

model_embed.eval()


# ------------------------------------------------------
# 1) Fonction d'embedding robuste (CLS + MaxPool → 2048)
# ------------------------------------------------------
def text_embedding(text: str, normalize: bool = True) -> np.ndarray:
    """
    Génère un embedding 2048D : concaténation (CLS | max_pooling).
    Compatible avec le modèle mergé.
    """
    inputs = tokenizer_embed(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model_embed(**inputs)

    last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden)

    cls_vec = last_hidden[:, 0, :]                  # (1, 768)
    maxpool_vec = torch.max(last_hidden, dim=1).values  # (1, 768)

    embedding = torch.cat([cls_vec, maxpool_vec], dim=1)[0]  # (2048,)

    embedding = embedding.cpu().numpy().astype("float32")

    if normalize:
        n = np.linalg.norm(embedding)
        if n > 0:
            embedding = embedding / n

    return embedding


# ------------------------------------------------------
# 2) Similarité cosinus (embeddings ou textes)
# ------------------------------------------------------
def cosine_similarity(a: Union[np.ndarray, str],
                      b: Union[np.ndarray, str]) -> float:
    if isinstance(a, str):
        a = text_embedding(a, normalize=False)
    if isinstance(b, str):
        b = text_embedding(b, normalize=False)

    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    if a_norm == 0 or b_norm == 0:
        return 0.0

    return float(np.dot(a / a_norm, b / b_norm))


# ------------------------------------------------------
# 3) Fonction masked LM SEPARÉE (pour compléter [MASK])
# ------------------------------------------------------
def bert_predict_mask(text, model_path=FINE_TUNED_MODEL, top_k=10):
    """
    Prédiction de tokens pour [MASK].
    Utilise BertForMaskedLM, séparé du modèle embed.
    """

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path)

    inputs = tokenizer(text, return_tensors="pt")

    mask_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        logits = model(**inputs).logits

    mask_logits = logits[0, mask_index, :]
    topk = torch.topk(mask_logits, top_k, dim=1).indices[0].tolist()

    tokens = [tokenizer.decode([t]).strip() for t in topk]
    print(f"{text} → {tokens}")
    return tokens
