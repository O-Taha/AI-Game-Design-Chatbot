import wikipedia
import re
import json
import os
from bs4 import BeautifulSoup

# Configuration de la langue
wikipedia.set_lang("en")

# Pages à scraper
pages = [
    "Glossary of video game terms",
    "List of video game genres",
    "Video game development",
    "Game design document",
    "Video game design",
    "Outline of video games",
    "Game design",
    "Level (video games)",
    "Video game genre",

    "List of beat 'em ups",
    "List of fighting games",
    "List of first-person shooters",
    "List of light-gun games",
    "List of platform game series",
    "List of third-person shooters",

    "List of party video games",
    "List of puzzle video games",
    "List of maze video games",
    "List of Tetris variants",
    "List of quiz arcade games",

    "List of role-playing video games",
    "List of massively multiplayer online role-playing games",
    "List of MUDs",
    "List of roguelikes",

    "List of business simulation video games",
    "List of city-building video games",
    "List of racing video games",
    "List of god video games",
    "List of simulation video games",
    "List of space flight simulation games"
]

# Dossier de sortie
output_dir = "DAPT/DAPT_Scraped_corpus"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------------------------
#     EXTRACTION DES TABLEAUX WIKIPÉDIA
# ------------------------------------------------------------------------------

def extract_wiki_tables(html):
    """Extrait tous les tableaux Wikipedia sous forme (term, definition)."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    entries = []

    for table in tables:
        # Lire toutes les lignes
        rows = table.find_all("tr")
        if not rows:
            continue

        # Récupérer les noms des colonnes
        headers = [th.get_text(" ", strip=True) for th in rows[0].find_all("th")]

        for row in rows[1:]:
            cells = row.find_all(["td", "th"])
            if not cells or len(cells) < 2:
                continue

            values = [c.get_text(" ", strip=True) for c in cells]

            term = values[0]
            pairs = [f"{headers[i]}: {values[i]}" for i in range(1, min(len(headers), len(values)))]

            definition = " | ".join(pairs)

            if len(term) > 0 and len(definition) > 5:
                entries.append({
                    "term": term,
                    "definition": definition
                })

    return entries



# ------------------------------------------------------------------------------
#     EXTRACTION TEXTE (déjà présente dans ton script)
# ------------------------------------------------------------------------------

def extract_text_entries(content):
    content = re.sub(r"\[\d+\]", "", content)
    lines = [l.strip() for l in content.split("\n") if l.strip()]

    entries = []
    current_term = None
    current_def = []

    for line in lines:
        if re.fullmatch(r"[A-Z]$", line):
            continue

        if (
            len(line) < 80
            and not line.endswith(".")
            and not line.startswith("See also")
            and not re.match(r"^[0-9•-]", line)
        ):
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

    if current_term and current_def:
        entries.append({
            "term": current_term.strip(),
            "definition": " ".join(current_def).strip()
        })

    return entries



# ------------------------------------------------------------------------------
#     SCRAPER PRINCIPAL
# ------------------------------------------------------------------------------

def scrape_wiki_page(title):
    try:
        page = wikipedia.page(title)
        content = page.content
        html = page.html()
    except Exception as e:
        print(f"⚠️ Erreur lors du scraping de {title}: {e}")
        return []

    text_entries = extract_text_entries(content)
    table_entries = extract_wiki_tables(html)

    total = len(text_entries) + len(table_entries)
    print(f"✅ {total} entrées extraites depuis « {title} » (texte + tableaux)")

    return text_entries + table_entries



# ------------------------------------------------------------------------------
#     PIPELINE COMPLET
# ------------------------------------------------------------------------------

all_entries = []
for page_title in pages:
    all_entries.extend(scrape_wiki_page(page_title))

# Déduplication + nettoyage
cleaned = []
seen = set()

for e in all_entries:
    term = e["term"].strip()
    if term not in seen and len(e["definition"]) > 15:
        seen.add(term)
        cleaned.append(e)

# Sauvegarde
json_path = os.path.join(output_dir, "wiki_videogame_corpus_2.json")
txt_path = os.path.join(output_dir, "wiki_videogame_corpus_2.txt")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(cleaned, f, indent=2, ensure_ascii=False)

with open(txt_path, "w", encoding="utf-8") as f:
    for e in cleaned:
        f.write(f"{e['term']}:\n    {e['definition']}\n\n")

print(f"🎉 Corpus créé : {len(cleaned)} entrées")