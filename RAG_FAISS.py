import faiss
import numpy as np
import pickle
from tqdm import tqdm
import os
from pathlib import Path
from typing import List, Tuple
import hashlib

from text_embedding import text_embedding, MODEL_NAME, FINE_TUNED_MODEL

# --- gestion des dossiers ---
os.makedirs('RAGIndex', exist_ok=True)

def compute_model_signature(model_dir: str) -> str:
    """
    Calcule une signature (sha1) du contenu d'un dossier modèle pour détecter changements.
    """
    p = Path(model_dir)
    if not p.exists():
        return ""
    sha = hashlib.sha1()
    # parcourir tous les fichiers
    for f in sorted([x for x in p.rglob("*") if x.is_file()]):
        try:
            # inclure nom relatif, taille, date modif
            rel = str(f.relative_to(p)).encode()
            sha.update(rel)
            st = f.stat()
            sha.update(str(st.st_size).encode())
            sha.update(str(int(st.st_mtime)).encode())
        except Exception:
            continue
    return sha.hexdigest()

def write_model_signature(index_file: str, model_dir: str):
    sig = compute_model_signature(model_dir)
    try:
        with open(index_file + ".modelsig", "w") as f:
            f.write(sig)
    except Exception as e:
        print(f"[RAG_FAISS] Warning: could not write model signature: {e}")

def read_model_signature(index_file: str) -> str:
    try:
        with open(index_file + ".modelsig", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""

def needs_rebuild(index_file: str = 'RAGIndex/mechdex.faiss', model_dir: str = FINE_TUNED_MODEL) -> bool:
    """
    Vérifie s’il faut reconstruire l’index : index manquant ou modèle modifié.
    """
    idx = Path(index_file)
    if not idx.exists():
        return True
    saved = read_model_signature(index_file)
    current = compute_model_signature(model_dir)
    if saved != current:
        print(f"[RAG_FAISS] Model signature changed (old={saved}, new={current}) → rebuild needed")
        return True
    return False


def build_faiss_index(mechanics_data, index_file='RAGIndex/mechdex.faiss'):
    """
    Construit un index FAISS à partir des descriptions de mécaniques de jeu
    pour permettre un retrieval efficace basé sur la similarité cosinus.

    Chaque mécanique est transformée en vecteur d'embedding via la fonction
    `text_embedding`. L'index utilise le produit scalaire interne (IP) qui 
    équivaut à la similarité cosinus si les vecteurs sont normalisés.

    Paramètres :
        mechanics_data (list[dict]) : liste de mécaniques, chaque dict
            doit contenir au minimum 'long_description' et 'solved_problems'.
        index_file (str) : chemin de sauvegarde pour l'index FAISS.

    Retour :
        tuple : (index, metadata)
            - index (faiss.IndexFlatIP) : l'index FAISS prêt pour le retrieval
            - metadata (list[dict]) : liste de mécaniques correspondantes aux vecteurs dans le même ordre que dans l'index

    Exemple d'appel :
        index, metadata = build_faiss_index(mechanics)

    Exemple d'output :
        index -> <faiss.swigfaiss_avx2.IndexFlatIP object at 0x7f...>
        metadata -> [
            OrderedDict([('name', 'Double Jump'), ...]),
            OrderedDict([('name', 'Combo Input'), ...]),
            ...
        ]
    """

    # Récupère la première embedding pour adapter la dimension au modèle
    first_text = mechanics_data[0]['long_description'] + " " + mechanics_data[0]['solved_problems']
    first_emb = text_embedding(first_text)
    dim = len(first_emb)
    print(f"Detected embedding dimension: {dim}")
    
    # Création d'un index FAISS utilisant l'Inner Product
    # (équivalent à cosine similarity si vecteurs normalisés)
    index = faiss.IndexFlatIP(dim)
    
    vectors = []   # liste pour stocker les embeddings
    metadata = []  # liste pour stocker les dictionnaires de mécaniques

    # Boucle sur toutes les mécaniques avec barre de progression
    for mech in tqdm(mechanics_data, desc="Generating embeddings"):
        # Texte combiné pour l'embedding : description + problèmes résolus
        text = mech['long_description'] + " " + mech['solved_problems']
        
        # Génération de l'embedding (vecteur 2048)
        emb = text_embedding(text)
        
        vectors.append(emb)
        metadata.append(mech)
    
    # Conversion en numpy array 2D (n_méca, dim)
    vectors = np.vstack(vectors)
    
    # Ajout des vecteurs à l'index FAISS
    index.add(vectors)
    
    # Sauvegarde de l'index sur disque
    faiss.write_index(index, index_file)
    print("📁 Current working directory:", os.getcwd())
    print("💾 Saving FAISS index to:", os.path.abspath(index_file), "\n")
    
    # Sauvegarde séparée des métadonnées
    with open(index_file + '.meta', 'wb') as f:
        pickle.dump(metadata, f)
    
    write_model_signature(index_file, FINE_TUNED_MODEL)

    return index, metadata


def retrieve(query: str,
            index_file: str = 'RAGIndex/mechdex.faiss',
            top_k: int | None = 2) -> List[Tuple[dict, float]]:
    """
    Recherche les top_k voisins pour une query dans l'index FAISS.

    - index_file : chemin vers l'index FAISS (ex: 'RAGIndex/mechdex.faiss')
    - top_k : nombre de voisins à retourner. Si None -> retourne tous les vecteurs.

    Retour :
        list de tuples (metadata_item, score)
    """
    # charger index + meta
    index_path = Path(index_file)
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    index = faiss.read_index(str(index_path))
    with open(str(index_path) + '.meta', 'rb') as f:
        metadata = pickle.load(f)

    # Debug prints (utile pour comprendre pourquoi top_k "ne marche pas")
    print(f"[retrieve] index file: {index_file}")
    print(f"[retrieve] index type: {type(index)}")
    try:
        print(f"[retrieve] index.ntotal = {index.ntotal}, index.d = {index.d}")
    except Exception:
        print("[retrieve] could not read index.ntotal/index.d (index type special)")

    # calcul embedding de la query (assure numpy float32, vector shape)
    q = text_embedding(query)  # retourne np.ndarray (float32)
    if not isinstance(q, np.ndarray):
        q = np.asarray(q, dtype='float32')
    else:
        q = q.astype('float32')

    if q.ndim == 1:
        query_vec = q.reshape(1, -1)
    elif q.ndim == 2 and q.shape[0] == 1:
        query_vec = q
    else:
        raise ValueError(f"Query embedding has unexpected shape {q.shape}")

    print(f"[retrieve] query_vec.shape = {query_vec.shape}, dtype = {query_vec.dtype}")

    # déterminer top_k effectif
    if top_k is None:
        # si metadata disponible, utiliser sa longueur
        try:
            top_k_effective = len(metadata)
        except Exception:
            top_k_effective = 100  # fallback
    else:
        top_k_effective = int(top_k)

    # clamp top_k_effective <= index.ntotal (si lisible)
    if hasattr(index, 'ntotal'):
        if top_k_effective > index.ntotal:
            print(f"[retrieve] requested top_k={top_k_effective} > index.ntotal={index.ntotal}, clamping")
            top_k_effective = index.ntotal

    print(f"[retrieve] using top_k = {top_k_effective}")

    # Si index est un IVF-like, augmenter nprobe pour rechercher davantage de cellules
    # (utile si tu as IndexIVFFlat ou IndexIVFPQ)
    if isinstance(index, faiss.IndexIVF):
        # safe set nprobe
        nlist = index.nlist if hasattr(index, 'nlist') else None
        nprobe = min(10, nlist) if nlist else 10
        print(f"[retrieve] index is IVF, setting nprobe = {nprobe}")
        index.nprobe = nprobe

    # Effectuer la recherche (FAISS attend float32)
    try:
        D, I = index.search(query_vec, top_k_effective)
    except AssertionError as ae:
        # dimension mismatch -> raise user friendly message
        raise RuntimeError(f"FAISS dimension mismatch or error during search: {ae}")

    # I contient -1 pour voisins inexistants ; filtrer ceux-ci
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx == -1:
            continue
        results.append((metadata[idx], float(score)))

    print(f"[retrieve] returning {len(results)} results")
    return results