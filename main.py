from mechdex_repo_retrival import *
from RAG_FAISS import *
from text_embedding import *
from yaml_parsing import *
import os.path
from pathlib import Path

if __name__ == "__main__":
    index_found = any(Path('.').rglob("mechdex.faiss"))
    if is_new_commit() or not index_found:
        print("[Action] New commit detected or missing index, rebuilding FAISS index...\n")
        mechanics = load_mechanics('.')
        build_faiss_index(mechanics)
    else:
        print("[Info] No new commit, using existing index.\n")
    
    query =  """I'm developping a sort of punch out clone where you're in first person view and can punch as well as dodge left or right, but i feel like the game isn't original enough"""

    try:
        # Tentative de recherche dans l'index existant
        results = retrieve(query, top_k = 5)

    except Exception as e:
        # Si une erreur survient (ex: dimensions incompatibles, fichier manquant, etc.)
        print(f"\n[Info] Retrieval failed: {e}\n")
        print("[Action] Rebuilding FAISS index due to model or dimension change...\n")
        
        # Suppression sécurisée des anciens fichiers d’index
        for file in Path('.').rglob("mechdex.faiss*"):
            file.unlink(missing_ok=True)
        
        mechanics = load_mechanics('.')
        build_faiss_index(mechanics)
        results = retrieve(query, index_file='RAGIndex/mechdex.faiss', top_k=10)
        
    # Tri des résultats par score décroissant
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

    # Affichage des résultats
    print("\n=== Retrieved Mechanics (sorted by similarity score) ===\n")
    for mech, score in results_sorted:
        print(f"🧩 {mech['name']} ({mech['category']}) - Score: {score:.4f}")
        print(f"   → Solved Problems: {mech['solved_problems']}\n")

        #Ajouter un dictionnaire de définitions au query
        #Reentrainer le modèle avec des définitions spécifiquement