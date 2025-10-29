import faiss
import numpy as np
import pickle
from tqdm import tqdm
import os, fnmatch
from pathlib import Path
from typing import List, Tuple

from text_embedding import *

os.makedirs('RAGIndex', exist_ok=True)

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
    
    return index, metadata


def retrieve(query, index_file='RAGIndex/mechdex.faiss', top_k=5):
    # Charger index et metadata
    index = faiss.read_index(index_file)
    with open(index_file + '.meta', 'rb') as f:
        metadata = pickle.load(f)
    
    # Embedding de la query
    query_vec = text_embedding(query).reshape(1, -1)
    
    # Recherche
    D, I = index.search(query_vec, top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        results.append((metadata[idx], float(score)))
    return results
