import random
import csv
from pathlib import Path
from yaml_parsing import load_mechanics

def build_training_dataset(mechanics_dir="./mechanics", output_file="query_mechanics_dataset.csv", num_negatives=2):
    """
    Génère un dataset pour fine-tuner BERT sur la compréhension des mécaniques de jeu.
    
    Chaque entrée est une paire (input_text, target_text, label) :
        - label = 1 pour une correspondance réelle entre un "problème" et une "mécanique"
        - label = 0 pour une paire aléatoire inadéquate

    Exemple d'entrée positive :
        (One-dimensional level design	Double Jump - Grants a second mid-air jump.	1)
    
    Exemple d'entrée négative :
        (One-dimensional level design	Crafting - Allows players to create items.	0)

    Paramètres :
        mechanics_dir (str): dossier racine des mécaniques YAML.
        output_file (str): chemin du fichier CSV de sortie.
        num_negatives (int): nombre de paires négatives générées par mécanique.
    """
    mechanics = load_mechanics(mechanics_dir)
    print(f"Loaded {len(mechanics)} mechanics from {mechanics_dir}")

    dataset = []
    for mech in mechanics:
        problem = mech.get("solved_problems", "").strip()
        desc = mech.get("long_description", "").strip()
        name = mech.get("name", "Unnamed")
        if not problem or not desc:
            continue

        # --- Paire positive ---
        input_text = problem
        target_text = f"{name} - {desc}"
        dataset.append((input_text, target_text, 1))

        # --- Paires négatives ---
        negatives = random.sample(mechanics, k=min(num_negatives, len(mechanics)-1))
        for neg in negatives:
            if neg["name"] == name:
                continue
            neg_text = f"{neg['name']} - {neg['long_description']}"
            dataset.append((input_text, neg_text, 0))

    # --- Sauvegarde CSV ---
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input_text", "target_text", "label"])
        writer.writerows(dataset)

    print(f"✅ Dataset saved to {output_file} ({len(dataset)} pairs total)")
    return output_file


if __name__ == "__main__":
    build_training_dataset()