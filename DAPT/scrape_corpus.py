import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import re
import time
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCES = [
    "https://gameontology.com/index.php/Main_Page",
    "http://motivationalpatterns.com/doku.php?id=start",
    "http://virt10.itu.chalmers.se/index.php/Main_Page",
]

MAX_DEPTH = 2           # profondeur d'exploration
CRAWL_DELAY = 0.3       # délai entre les requêtes (secondes)
OUTPUT_FILE = "DAPT/DAPT_Scraped_corpus/videogame_corpus.txt"

# --- LISTES DE FILTRAGE ---
EXCLUDE_PATTERNS = [
    "Special:", "User:", "Talk:", "File:", "Help:", "Category:",
    "action=", "oldid=", "diff=", "printable=", "feed=", "curid=",
]
EXCLUDE_EXTENSIONS = (
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".mp3", ".wav", ".ogg",
    ".mp4", ".avi", ".mov", ".pdf", ".zip", ".rar", ".tar", ".gz",
    ".exe", ".bin", ".xml"
)

# --- VARIABLES GLOBALES ---
visited = set()
corpus = []

# --- FONCTIONS ---
def clean_url(url):
    """Supprime les ancres et les paramètres inutiles."""
    url, _ = urldefrag(url)
    url = re.sub(r"&oldid=\d+", "", url)
    return url.strip("/")

def clean_text(text):
    """Nettoie le texte brut : espaces, numéros de référence, etc."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[\d+\]", "", text)
    return text.strip()

def is_excluded(url):
    """Filtre les liens inutiles (fichiers, paramètres, sections)."""
    if any(url.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
        return True
    if any(ex in url for ex in EXCLUDE_PATTERNS):
        return True
    return False

def scrape_page(url):
    """Télécharge et nettoie une page web."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form", "aside"]):
            tag.decompose()
        return clean_text(soup.get_text(separator=" ", strip=True)), soup
    except Exception as e:
        print(f"❌ Erreur sur {url}: {e}")
        return "", None

def crawl(urls):
    """Explore récursivement les liens internes à partir d’une liste d’URLs."""
    global visited, corpus
    next_depth_urls = set()

    with tqdm(total=len(urls), desc="Exploration", ncols=100) as pbar:
        for url in urls:
            url = clean_url(url)
            if url in visited or is_excluded(url):
                pbar.update(1)
                continue
            visited.add(url)

            text, soup = scrape_page(url)
            if text:
                corpus.append(text)

            if soup:
                base = urlparse(url).netloc
                for a in soup.find_all("a", href=True):
                    link = urljoin(url, a["href"])
                    parsed = urlparse(link)
                    if parsed.netloc == base and not is_excluded(link):
                        next_depth_urls.add(clean_url(link))

            pbar.update(1)
            time.sleep(CRAWL_DELAY)

    return list(next_depth_urls)

def fetch_corpus(sources):
    """Lance le scraping complet avec gestion des niveaux de profondeur."""
    urls = sources
    for depth in range(MAX_DEPTH + 1):
        print(f"\n🕸️ Niveau {depth}/{MAX_DEPTH} — {len(urls)} URL à explorer")
        urls = crawl(urls)
        if not urls:
            break

    # Nettoyage global du corpus
    clean_corpus = [clean_text(text) for text in corpus if len(text) > 100]

    # Sauvegarde finale
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(clean_corpus))

    print(f"\n✅ Corpus enregistré dans '{OUTPUT_FILE}' ({len(clean_corpus)} pages uniques)")

# --- POINT D’ENTRÉE ---
if __name__ == "__main__":
    fetch_corpus(SOURCES)
