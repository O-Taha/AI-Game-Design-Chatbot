import wikipedia
import re
import json
import os

# Configuration de la langue
wikipedia.set_lang("en")

# Récupération du texte complet via l'API
page = wikipedia.page("Glossary of video game terms")
content = page.content

# Dossier de sortie
os.makedirs("DAPT/DAPT_Scraped_corpus", exist_ok=True)

# Nettoyage du texte : suppression des références [1], [2], etc.
content = re.sub(r"\[\d+\]", "", content)

# Découpage en lignes
lines = [l.strip() for l in content.split("\n") if l.strip()]

entries = []
current_term = None
current_def = []

for line in lines:
    # Ignorer les titres de section (A, B, C, etc.)
    if re.fullmatch(r"[A-Z]$", line):
        continue

    # Si la ligne semble être un nouveau terme (souvent sans point et courte)
    if (
        len(line) < 80  # les termes sont courts
        and not line.endswith(".")  # pas une phrase complète
        and not line.startswith("See also")  # filtrer les liens divers
        and not re.match(r"^[0-9•-]", line)  # éviter les listes numérotées
    ):
        # Sauvegarder le terme précédent s'il existe
        if current_term and current_def:
            entries.append({
                "term": current_term.strip(),
                "definition": " ".join(current_def).strip()
            })
            current_def = []

        current_term = line

    else:
        # Ligne appartenant à la définition en cours
        if current_term:
            current_def.append(line)

# Dernière entrée
if current_term and current_def:
    entries.append({
        "term": current_term.strip(),
        "definition": " ".join(current_def).strip()
    })

# Nettoyage final : suppression des doublons et trimming
cleaned = []
seen = set()
for e in entries:
    if e["term"] not in seen and len(e["definition"]) > 15:
        seen.add(e["term"])
        cleaned.append(e)

# Sauvegarde JSON et TXT
with open("DAPT/DAPT_Scraped_corpus/wiki_videogame_corpus.json", "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

with open("DAPT/DAPT_Scraped_corpus/wiki_videogame_corpus.txt", "w", encoding="utf-8") as f:
    for e in cleaned:
        f.write(f"{e['term']}:\n    {e['definition']}\n\n")

print(f"✅ {len(cleaned)} termes extraits et sauvegardés dans DAPT/DAPT_Scraped_corpus/")
