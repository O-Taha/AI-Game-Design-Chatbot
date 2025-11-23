import wikipedia
import re
import json
import os

# Configuration de la langue
wikipedia.set_lang("en")

# Pages à scraper
pages = [
    "Glossary of video game terms",
    "List of video game genres"
]

# Dossier de sortie
output_dir = "DAPT/DAPT_Scraped_corpus"
os.makedirs(output_dir, exist_ok=True)

def scrape_wiki_page(title):
    """Scrape une page Wikipedia et renvoie une liste (term, definition)."""
    try:
        page = wikipedia.page(title)
        content = page.content
    except Exception as e:
        print(f"⚠️ Erreur lors du scraping de {title}: {e}")
        return []

    # Nettoyage basique
    content = re.sub(r"\[\d+\]", "", content)
    lines = [l.strip() for l in content.split("\n") if l.strip()]

    entries = []
    current_term = None
    current_def = []

    for line in lines:
        # Ignorer les titres de section (A, B, C, etc.)
        if re.fullmatch(r"[A-Z]$", line):
            continue

        # Détection d’un nouveau terme
        if (
            len(line) < 80
            and not line.endswith(".")
            and not line.startswith("See also")
            and not re.match(r"^[0-9•-]", line)
        ):
            # Sauvegarder le précédent
            if current_term and current_def:
                entries.append({
                    "term": current_term.strip(),
                    "definition": " ".join(current_def).strip()
                })
                current_def = []
            current_term = line
        else:
            if current_term:
                current_def.append(line)

    # Dernière entrée
    if current_term and current_def:
        entries.append({
            "term": current_term.strip(),
            "definition": " ".join(current_def).strip()
        })

    print(f"✅ {len(entries)} entrées extraites depuis « {title} »")
    return entries


# Fusion de toutes les pages
all_entries = []
for page_title in pages:
    all_entries.extend(scrape_wiki_page(page_title))

# Nettoyage final
cleaned = []
seen = set()
for e in all_entries:
    term = e["term"].strip()
    if term not in seen and len(e["definition"]) > 15:
        seen.add(term)
        cleaned.append(e)

# Sauvegarde JSON et TXT
json_path = os.path.join(output_dir, "wiki_videogame_corpus.json")
txt_path = os.path.join(output_dir, "wiki_videogame_corpus.txt")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

with open(txt_path, "w", encoding="utf-8") as f:
    for e in cleaned:
        f.write(f"{e['term']}:\n    {e['definition']}\n\n")

print(f"✅ Corpus fusionné : {len(cleaned)} termes sauvegardés dans {output_dir}/")
