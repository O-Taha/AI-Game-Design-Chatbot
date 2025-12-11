from yaml_parsing import load_mechanics
from text_embedding import FINE_TUNED_MODEL
from pathlib import Path
from mechdex_repo_retrival import is_new_commit
from RAG_FAISS import needs_rebuild, build_faiss_index, retrieve

def safe_rebuild_index(index_file='RAGIndex/mechdex.faiss'):
    print("[Action] Removing old FAISS index files...\n")
    for file in Path('RAGIndex').glob("mechdex.faiss*"):
        try:
            file.unlink(missing_ok=True)
        except Exception:
            pass
    print("[Action] Rebuilding FAISS index...\n")
    mechanics = load_mechanics('.')
    build_faiss_index(mechanics, index_file=index_file)
    print("[Info] FAISS index rebuilt.\n")

if __name__ == "__main__":
    index_file = 'RAGIndex/mechdex.faiss'

    # Si index absent, modèle modifié ou nouvelles mécaniques → rebuild
    if needs_rebuild(index_file=index_file, model_dir=FINE_TUNED_MODEL) or is_new_commit():
        print("[Action] Index missing or model changed → rebuilding index...\n")
        safe_rebuild_index(index_file=index_file)
    else:
        print("[Info] Existing index valid, using it.\n")

    # Exemple de query
    query = "this refers to video game design" + """I'm developping a sort of punch out clone where you're in first person view and can punch as well as dodge left or right, but i feel like the game isn't original enough"""

    try:
        results = retrieve(query, index_file=index_file, top_k=5)
    except Exception as e:
        print(f"\n[Warning] Retrieval failed: {e}\n")
        print("[Action] Error during retrieval → rebuilding index...\n")
        safe_rebuild_index(index_file=index_file)
        results = retrieve(query, index_file=index_file, top_k=5)

    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

    print("\n=== Retrieved Mechanics (sorted by similarity score) ===\n")
    for mech, score in results_sorted:
        print(f"🧩 {mech['name']} ({mech['category']}) - Score: {score:.4f}")
        print(f"   → Solved Problems: {mech['solved_problems']}\n")
