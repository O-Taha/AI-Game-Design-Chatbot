# generate_training_pairs.py
import json, random, re
from pathlib import Path

random.seed(42)

INFILE = Path("all_mechanics.json")
OUTFILE = Path("training_triplets.jsonl")
SAMPLES_PER_MECHANIC = 30  # ajustable
NEGATIVES_COUNT = 5


def safe_load_line(line):
	# chaque ligne est un JSON dict ; strip trailing commas/whitespace
	line = line.strip()
	if not line:
		return None
	try:
		return json.loads(line)
	except Exception:
		# fallback: try to fix common single-quotes in python-style repr -> replace single quotes by double quotes
		# This is a heuristic; preferred input is valid JSON per line.
		try:
			fixed = line.replace("'", '"')
			return json.loads(fixed)
		except Exception:
			return None


def extract_text_field(obj, key):
	try:
		v = obj.get(key, "")
		return str(v)
	except:
		return ""
	if isinstance(v, str):
		return v
	try:
		return str(v)
	except:
		return ""


# Templates to create natural language "problem" anchors.
TEMPLATES = [
	"Players are not enjoying this.",
	"How can I improve my game?",
	"I am designing a game and I can't find a solution for this",
	"How can I fix my game?",
	"My game isn't fun.",
	"I feel like something is missing from my game",
	"Players are not playing right.",
	"I am developping a game.",
	"How can I improve my game.",
	"My game is missing something.",
	"Can you fix my game?",
	"Can you improve my game?",
	"Can you help me?",
	"Can you tell me what's missing from my game?",
	"What's missing from my game?",
	"Please fix",
	"How to fix this issue with my game",
	"how can we improve this aspect of my game?",
	"I am designing a game.",
	"What can I try?",
	"How to address this in game design?",
]


def build_anchors(m):
	"""
	Turns a mechanic into a set of related queries
	"""
	short = extract_text_field(m, "short_description").strip().rstrip(".")
	category = extract_text_field(m, "category").strip()
	longd = extract_text_field(m, "long_description").strip()
	solved = extract_text_field(m, "solved_problems").strip()
	examples = extract_text_field(m, "examples")

	query_variants = []
	# Always add a few direct problem rewrites from short/long/solved
	if short != "":
		query_variants.append(f"I'm looking for a mechanic that does this : {short}.")
	else:
		return None
	if solved != "":
		# try to extract the description part if the field contains a python dict string
		# attempt to isolate 'description' value
		desc = extract_text_field(safe_load_line(solved), "description")
		desc = desc.split(".")[0] + "."
		if desc and desc != ".":
			query_variants.append(f"{desc} What mechanic would solve this?")
			query_variants.append(f"{desc} How to design around that?")
		problem_title = extract_text_field(safe_load_line(solved), "title")
		if problem_title:
			query_variants.append(f"How can I fix {problem_title}?")
	# if longd != "":
	#	 # pick first sentence
	#	 first_sent = longd.split(".")[0]
	#	 if first_sent:
	#		 query_variants.append(f"{first_sent}. What can designers do to solve this?")
	# create slight paraphrases
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
		anchors.add(random.choice(TEMPLATES) + " " + v)
	return list(anchors)


# Read mechanics
mechanics = []
with INFILE.open("r", encoding="utf-8") as f:
	for line in f:
		obj = safe_load_line(line)
		if obj:
			# unify the term name field (we'll use "name" or "term")
			term = obj.get("name") or obj.get("term")
			if not term:
				continue
			mechanics.append(
				{
					"term": term,
					"category": obj.get("category", "").strip(),
					"short_description": extract_text_field(obj, "short_description"),
					"long_description": extract_text_field(obj, "long_description"),
					"solved_problems": extract_text_field(obj, "solved_problems"),
					"examples": extract_text_field(obj, "examples"),
				}
			)

all_terms = [m["term"] for m in mechanics]
cat_map = {}
for m in mechanics:
	cat_map.setdefault(m["category"], []).append(m["term"])

dataset = []
for m in mechanics:
	term = m["term"]
	anchors = build_anchors(m)
	if anchors is None:
		continue
	# limit anchors to SAMPLES_PER_MECHANIC (if too many)
	chosen_anchors = (
		anchors[:SAMPLES_PER_MECHANIC]
		if len(anchors) >= SAMPLES_PER_MECHANIC
		else anchors
		+ [
			random.choice(anchors)
			for _ in range(max(0, SAMPLES_PER_MECHANIC - len(anchors)))
		]
	)

	for anchor in chosen_anchors[:SAMPLES_PER_MECHANIC]:
		# select negatives trying to avoid same category
		same_cat = cat_map.get(m["category"], [])
		pool = [t for t in all_terms if t != term and t not in same_cat]
		if len(pool) < NEGATIVES_COUNT:
			pool = [t for t in all_terms if t != term]
		negatives = random.sample(pool, NEGATIVES_COUNT)
		sample = {"anchor": anchor, "positive": term, "negatives": negatives}
		dataset.append(sample)

# write jsonl
with OUTFILE.open("w", encoding="utf-8") as out:
	for s in dataset:
		out.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Wrote {len(dataset)} samples to {OUTFILE}")
