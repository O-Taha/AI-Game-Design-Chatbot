import json, jsonlines, random, re
from pathlib import Path
import requests
import httplib2
from bs4 import BeautifulSoup, SoupStrainer
import textdistance

METACRITIC_URL = "https://www.metacritic.com"

random.seed(42)

INFILE = Path("all_mechanics.jsonl")
OUTFILE = Path("training_triplets.jsonl")
SAMPLES_PER_MECHANIC = 20  # ajustable
NEGATIVES_COUNT = 5


# Templates to create natural language "problem" anchors.
TEMPLATES = [
	"Players are not enjoying my game.",
	"Players are not enjoying this in my game.",
	"How can I improve my game?",
	"How can I improve this game?",
	"I am designing a game and I can't find a solution for this.",
	"I'm designing a game and I can't find a solution for this.",
	"I am designing my game and I can't find a solution for this.",
	"I'm designing my game and I can't find a solution for this.",
	"How can I fix my game?",
	"How can I fix this game?",
	"My game isn't fun.",
	"This game isn't fun.",
	"I feel like something is missing from my game.",
	"I feel like something is missing from this game.",
	"Players are not playing right.",
	"Testers are not playing right.",
	"I am developping a game.",
	"I am developping my game.",
	"How can I improve my game.",
	"How can I improve this game.",
	"My game is missing something.",
	"This game is missing something.",
	"Can you fix my game?",
	"Can you fix this game?",
	"Can you improve my game?",
	"Can you improve this game?",
	"Can you help me?",
	"Can you tell me what's missing from my game?",
	"Can you tell me what's missing from this game?",
	"What's missing from my game?",
	"What's missing from this game?",
	"Please fix.",
	"How to fix this issue with my game?",
	"How can we improve this aspect of my game?",
	"I am designing a game.",
	"I'm designing a game.",
	"I am designing this game.",
	"I'm designing this game.",
	"What can I try?",
	"How to address this in game design?",
	"Curious about this.",
	"How should I go about this?",
	"I am currently working on a game.",
	"I'm currently working on a game.",
	"I am currently working on my game.",
	"I'm currently working on my game.",
	"I'm having trouble designing this game.",
	"I'm having trouble designing my game.",
	"I am having trouble designing this game.",
	"I am having trouble designing my game."
]


def build_anchors(m):
	"""
		Turns a mechanic into a set of related believable user-queries
	"""
	short = m["short_description"]
	category = m["category"]
	longd = m["long_description"]
	solved = m["solved_problems"]
	examples = m["examples"]
	
	templates_subset = TEMPLATES.copy()
	query_variants = []

	# Always add a few direct problem rewrites from short/long/solved
	if short != "" and random.getrandbits(1):
		query_variants.append(f"I'm looking for a mechanic that does this : {short}")
	if solved != "":
		# try to extract the description part if the field contains a python dict string
		# attempt to isolate 'description' value
		pattern = r"'description':\s*'([^.]*)\."
		desc = re.search(pattern, solved)
		if desc:
			desc = desc.group(1) + '.'
			#print(desc)
			query_variants.append(f"{desc} What mechanic would solve this?")
			query_variants.append(f"{desc} How to design around that?")
		pattern = r"'title'\s*:\s*'([^']+)'"
		problem_title = re.search(pattern, solved).group(1)
		#print(problem_title)
		if problem_title:
			query_variants.append(f"How can I fix {problem_title}?")
	if examples != "":
		examples = examples.split(";")
		genres = []
		for example in examples:
			pattern = r"'title'\s*:\s*'([^']+)'"
			game_title = re.search(pattern, example)
			if game_title:
				game_title = game_title.group(1)
				genre = get_game_genre(game_title)
				if genre:  # Vérifier si un genre a été obtenu
					genres.append(genre)
	anchors = set()
	for v in query_variants:
		anchors.add(v)
		anchors.add(
			v.replace("How to", "How do we").replace(
				"How do we make", "How can we make"
			)
		)
		anchors.add(v + " (design problem)")
		anchors.add(f"I am making a game with {category} mechanics. " + v)

		# remove duplicates
		unique_genres = list(dict.fromkeys(genres))
		for genre in unique_genres:
			anchors.add(f"I am making a {genre} game. " + v)
			anchors.add(f"The genre of my game is {genre}. " + v)

	while len(anchors) < SAMPLES_PER_MECHANIC and templates_subset:
		chosen_template = random.choice(templates_subset)
		for v in query_variants:
			anchors.add(chosen_template + " " + v)
		templates_subset.remove(chosen_template)
		
	return list(anchors)


def search_metacritic_game(name):
	found_game = look_up_best_result(name)

	if found_game:
		return found_game
	else:
		print("No match found")


def look_up_best_result(name):
	query = name.replace(" ", "%20")
	url = f"{METACRITIC_URL}/search/{query}/?category=13&page=1"
	print("Searching:", url)

	params = {"search_term": name}
	headers = {"User-Agent": "Mozilla/5.0"}

	r = requests.get(url, params=params, headers=headers)
	html_code = r.text

	pattern = r'<a href="/game/(.*?)/" data-testid="search-result-item"'
	game_result_urls = re.findall(pattern, html_code)

	if not game_result_urls:  # Absolutely no result...
		return None

	best_match = None
	max_similarity = 0

	for game_result in game_result_urls:
		similarity = calculate_similarity(game_result, name)
		if similarity > max_similarity:
			max_similarity = similarity
			best_match = game_result

	return best_match if max_similarity > 0.5 else None


def calculate_similarity(game_title_in_url, initially_searched_game):
	# Strip titles of any arbitrarily different element
	found_game = game_title_in_url.replace("-", " ").replace(":", " ").lower()
	searched_game = initially_searched_game.replace("-", " ").replace(":", " ").lower()

	similarity = textdistance.overlap.normalized_similarity(found_game, searched_game)
	#print(f"Similarity between '{found_game}' and '{searched_game}': {similarity}")
	return similarity


def get_game_genre(name):
	game_url_name = search_metacritic_game(name)
	if not game_url_name:
		return None
	game_page_path = "/game/" + game_url_name

	params = {"search_term": name}
	headers = {"User-Agent": "Mozilla/5.0"}

	r = requests.get(METACRITIC_URL + game_page_path, params=params, headers=headers)
	html_code = r.text

	pattern = r'"genre":"(.*?)"'
	match = re.search(pattern, html_code)

	if match:
		genre = match.group(1)
		print(game_url_name, genre)  
		return genre
	else:
		print("No match found", r.status_code)


# Read mechanics
mechanics = []
with open(INFILE, "r", encoding="utf-8") as f:
	reader = jsonlines.Reader(f)
	for mechanic in reader:
		if mechanic:
			mechanics.append(mechanic)

all_mechanics_names = [m["name"] for m in mechanics]
mechanics_per_category = {}
for m in mechanics:
	mechanics_per_category.setdefault(m["category"], []).append(m["name"])

dataset = []
for m in mechanics:
	mechanic_name = m["name"]
	anchors = build_anchors(m)
	if anchors is None:
		continue
	# limit anchors to SAMPLES_PER_MECHANIC (if too many)
	chosen_anchors = random.sample(anchors, SAMPLES_PER_MECHANIC)

	for anchor in chosen_anchors[:SAMPLES_PER_MECHANIC]:
		# Select negatives trying to avoid same category
		all_mechanics_in_same_category = mechanics_per_category.get(m["category"], [])
		negatives_pool = [name for name in all_mechanics_names if name != mechanic_name and name not in all_mechanics_in_same_category] 
		if len(negatives_pool) < NEGATIVES_COUNT: # Can't avoid same category
			negatives_pool = [t for t in all_mechanics_names if t != mechanic_name]
		
		negatives = random.sample(negatives_pool, NEGATIVES_COUNT)
		sample = {"anchor": anchor, "positive": mechanic_name, "negatives": negatives}
		dataset.append(sample)

# write jsonl
with OUTFILE.open("w", encoding="utf-8") as out:
	for s in dataset:
		out.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Wrote {len(dataset)} samples to {OUTFILE}")
