from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertForMaskedLM
from typing import Union
from tqdm import tqdm
import torch
import numpy as np
from pathlib import Path

MODEL_NAME = "bert-base-uncased"
FINE_TUNED_MODEL = "TrainedModels/qbert-lora"

# Charger tokenizer / model de façon robuste : si dossier introuvable -> fallback sur MODEL_NAME
if Path(FINE_TUNED_MODEL).exists():
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL)
    model = AutoModel.from_pretrained(FINE_TUNED_MODEL)
else:
    # fallback
    print(f"[text_embedding] Warning: model path {FINE_TUNED_MODEL} not found. Falling back to {MODEL_NAME}.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    
def text_embedding(text: str, normalize: bool = True) -> np.ndarray:
    """
    Génère un embedding pour un texte donné à partir du modèle HuggingFace chargé.

    Paramètres :
        text (str) :
            Le texte à encoder.
        normalize (bool) :
            Si True, retourne l'embedding L2-normalisé (vecteur unitaire).
            Si False, retourne l'embedding brut (non normalisé).
    Retour :
        np.ndarray :
            Embedding 1D (taille 2048) en dtype float32.
    Exemple d'appel :
        e = text_embedding("My platformer's levels are too boring")
    Exemple de sortie (forme) :
        array([ 0.012345, -0.23456, ..., 0.98765], dtype=float32)  # shape (2048,)
    """
    # Tokenisation et conversion en tenseurs
    inputs = tokenizer(text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # (1, seq_len, hidden_size)
    # CLS token
    cls_embedding = last_hidden_state[:, 0, :]  # (1, hidden_size)
    # Max pooling
    max_pooling = torch.max(last_hidden_state, dim=1).values  # (1, hidden_size)
    # Concaténer (CLS | max_pooling) -> (1, 2048)
    sentence_embedding = torch.cat([cls_embedding, max_pooling], dim=1)  # (1, 2048)
    emb = sentence_embedding[0]  # (2048,)
    # Optionnel : normaliser L2
    if normalize:
        norm = torch.norm(emb, p=2)
        if norm > 0:
            emb = emb / norm

    return emb.cpu().numpy().astype('float32')


def cosine_similarity(a: Union[np.ndarray, str],
                        b: Union[np.ndarray, str]) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs ou deux textes.

    Paramètres :
        a, b (np.ndarray | str) :
            - Si np.ndarray : vecteurs 1D (shape (2048,)).
            - Si str : textes (les embeddings seront générés automatiquement).
            Note : si tu passes des textes, les embeddings seront calculés en brut
            (non normalisés) puis normalisés pour la similarité — cela évite
            d'altérer la sortie du modèle si tu souhaites garder les vecteurs bruts.
    Retour :
        float :
            Similarité cosinus (valeur entre -1 et 1).
    Exemple d'appel (avec embeddings) :
        e1 = text_embedding("texte1", normalize=False)
        e2 = text_embedding("texte2", normalize=False)
        sim = cosine_similarity(e1, e2)
    Exemple d'appel (avec textes) :
        sim = cosine_similarity("texte1", "texte2")
    Exemple de sortie :
        0.7845234
    """
    # Si l'un des arguments est une string, on calcule l'embedding brut (normalize=False)
    if isinstance(a, str):
        a_vec = text_embedding(a, normalize=False)
    else:
        a_vec = np.asarray(a, dtype='float32')

    if isinstance(b, str):
        b_vec = text_embedding(b, normalize=False)
    else:
        b_vec = np.asarray(b, dtype='float32')

    # Normalisation L2 (pour la similarité cosinus)
    a_norm = np.linalg.norm(a_vec)
    b_norm = np.linalg.norm(b_vec)

    if a_norm == 0 or b_norm == 0:
        # Cas défensif : si vecteur nul, renvoyer 0.0
        return 0.0

    a_unit = a_vec / a_norm
    b_unit = b_vec / b_norm

    return float(np.dot(a_unit, b_unit))


def bert_predict_mask(text, model_path="TrainedModels/qbert-lora", top_k=10):
    """
    Predicte les top-k complétions pour un masque [MASK] dans une phrase donnée.
    """

    # Charger tokenizer + modèle
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path)

    # Tokenisation
    inputs = tokenizer(text, return_tensors="pt")

    # Trouver l'index du masque
    mask_token_index = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    # Forward
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    # Obtenir les scores du masque
    mask_token_logits = logits[0, mask_token_index, :]

    # Top-k
    top_k_tokens = torch.topk(mask_token_logits, top_k, dim=1).indices[0].tolist()
    tokens = [tokenizer.decode([token]).strip() for token in top_k_tokens]
    print(text, tokens, "\n")
    return tokens



if __name__ == "__main__":
    x1_pt = text_embedding("My platformer's levels are too flat and the level design isn't interesting")
    x2_pt = text_embedding("A Charge Attack is a combat action where the player holds down an input button to build up power, then releases it to unleash a more powerful version of an attack. The design is defined by the trade-off between the time spent charging (a window of vulnerability) and the increased damage, area of effect, or special properties (like breaking a guard) of the resulting attack. Problems Solved One-note combat rhythm If all attacks are instantaneous button presses, combat can devolve into a repetitive, button-mashing rhythm. Charge attacks solve this by introducing a new timing element. They create a risk-reward cadence of finding a safe moment to charge and timing the release, adding a layer of deliberate pacing to the fight.")
    x3_pt = text_embedding("A Double Jump is a core platforming mechanic that grants the player the ability to perform a second jump while in mid-air. This action defies real-world physics to provide the player with significantly enhanced vertical and horizontal control during aerial traversal. It serves as a foundational tool for creating more complex and demanding platforming challenges. Problems Solved One-dimensional level design With only a single jump, level design is often restricted to simple, linear paths on a single plane. The double jump solves this by opening up the vertical axis. It allows designers to create more complex, layered environments with platforms and secrets that require precise aerial control to reach, adding significant depth to exploration. ")

    # Dot-product
    dot12 = np.dot(x1_pt, x2_pt)
    dot13 = np.dot(x1_pt, x3_pt)


    # Exemple 1 : obtenir un embedding (déjà normalisé)
    e = text_embedding("My platformer's levels are too boring")  # shape (2048,)

    # Exemple 2 : similarité entre deux embeddings (on peut fournir np.ndarray)
    e1 = text_embedding("My platformer's levels are too boring", normalize=True)
    e2 = text_embedding("A Double Jump is a core platforming mechanic that grants the player the ability to perform a second jump while in mid-air. This action defies real-world physics to provide the player with significantly enhanced vertical and horizontal control during aerial traversal. It serves as a foundational tool for creating more complex and demanding platforming challenges. Problems Solved One-dimensional level design With only a single jump, level design is often restricted to simple, linear paths on a single plane. The double jump solves this by opening up the vertical axis. It allows designers to create more complex, layered environments with platforms and secrets that require precise aerial control to reach, adding significant depth to exploration.", normalize=True)
    sim = cosine_similarity(e1, e2)  # float entre -1 et 1
    print("sim high (embeddings) =", sim)
    e2 = text_embedding("A Charge Attack is a combat action where the player holds down an input button to build up power, then releases it to unleash a more powerful version of an attack. The design is defined by the trade-off between the time spent charging (a window of vulnerability) and the increased damage, area of effect, or special properties (like breaking a guard) of the resulting attack. Problems Solved One-note combat rhythm If all attacks are instantaneous button presses, combat can devolve into a repetitive, button-mashing rhythm. Charge attacks solve this by introducing a new timing element. They create a risk-reward cadence of finding a safe moment to charge and timing the release, adding a layer of deliberate pacing to the fight.", normalize=True)
    sim = cosine_similarity(e1, e2)  # float entre -1 et 1
    print("sim low (embeddings) =", sim)

    bert_predict_mask("In this platformer, the player can [MASK] walls.")
    bert_predict_mask("A metroidvania usually features a large interconnected [MASK].")
    bert_predict_mask("Hitstop is a short freeze when a [MASK] connects.")
    bert_predict_mask("Punch Out is a [MASK] game where you fight against Mike Tyson", "bert-base-uncased")
    bert_predict_mask("Punch Out is a [MASK] game where you fight against Mike Tyson")





