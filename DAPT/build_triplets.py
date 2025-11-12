"""
Transforme les mécaniques chargées avec load_mechanics() en triplets:
(anchor_text, positive_text, negative_text)
Anchor: problème résolu (solved_problems)
Positive: mécanique correspondante (name + long_description)
Negative: mécanique aléatoire différente
Sauvegarde en TSV ou JSONL utilisable par sentence-transformers.
"""

import random
import argparse
import json
from yaml_parsing import load_mechanics
from pathlib import Path

def build_triplets(mechanics_dir="../mechanics", out_file="triplets.jsonl", num_negatives=1):
    mechanics = load_mechanics(mechanics_dir)
    # construire mapping, filter
    items = [m for m in mechanics if m.get("solved_problems") and m.get("long_description")]
    triplets = []
    for m in items:
        anchor = m["solved_problems"]
        positive = f"{m['name']} - {m['long_description']}"
        # choose negatives
        negs = random.sample([x for x in items if x["name"] != m["name"]], k=min(num_negatives, max(0, len(items)-1)))
        for neg in negs:
            negative = f"{neg['name']} - {neg['long_description']}"
            triplets.append({"anchor": anchor, "positive": positive, "negative": negative})
    Path(out_file).write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in triplets), encoding="utf-8")
    print(f"✅ Wrote {len(triplets)} triplets to {out_file}")
    return out_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="triplets.jsonl")
    parser.add_argument("--mechanics", default="../mechanics")
    parser.add_argument("--neg", type=int, default=1)
    args = parser.parse_args()
    build_triplets(args.mechanics, args.out, args.neg)
