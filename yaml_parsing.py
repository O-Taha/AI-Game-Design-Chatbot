from pathlib import Path
import pprint
import faiss
from collections import OrderedDict
import numpy as np
import yaml

def read_yaml_file(file_path):
    """
    Lit un fichier YAML et retourne son contenu sous forme de dictionnaire Python.

    Paramètres :
        file_path (str ou Path) : Chemin du fichier YAML à lire.
    Retour :
        dict : contenu YAML chargé.
    Exemple d'appel :
        data = read_yaml_file("mechanics/Exploration/Me/mechanic.yaml")
    Exemple de sortie (simplifié) :
        {
            'mechanic': {
                'name': 'Double Jump',
                'short_description': 'Permet de sauter une deuxième fois en l’air.',
                'long_description': 'Le Double Jump est une mécanique centrale ...',
                'examples': ['Super Mario', 'Celeste'],
                'solved_problems': 'One-dimensional level design'
            }
        }
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return yaml.safe_load(f)


def process_field(value):
    """
    Formate un champ du YAML pour qu'il soit une chaîne de caractères lisible.

    - Si c'est une liste, elle est jointe par "; ".
    - Si c'est une chaîne, les retours à la ligne sont supprimés et les espaces extrêmes sont enlevés.
    - Sinon, convertit l'objet en chaîne.
    Paramètres :
        value : valeur à traiter (str, list, autre)
    Retour :
        str : valeur formatée
    Exemple d'appel :
        s = process_field(['Mario', 'Luigi'])
    Exemple de sortie :
        "Mario; Luigi"
    Exemple d'appel :
        s = process_field("Bonjour\nmonde ")
    Exemple de sortie :
        "Bonjour monde"
    """
    if isinstance(value, list):
        return '; '.join(str(item) for item in value)
    elif isinstance(value, str):
        return value.replace('\n', ' ').strip()
    return str(value)


def load_mechanics(mechanics_dir='.'):
    """
    Parcourt récursivement tous les fichiers `mechanic.yaml` dans le dossier donné,
    extrait les informations de chaque mécanique, et retourne une liste de dictionnaires.

    Paramètres :
        mechanics_dir (str ou Path) : dossier racine contenant les sous-dossiers de mécaniques.
    Retour :
        list[dict] : chaque dictionnaire contient les champs suivants :
            - name
            - symbol
            - category
            - short_description
            - long_description
            - examples
            - solved_problems
    Exemple d'appel :
        mechanics = load_mechanics("./mechanics")
    Exemple de sortie (simplifiée pour la première mécanique) :
        [{
            'name': 'Double Jump',
            'symbol': 'Dj',
            'category': 'Exploration',
            'short_description': 'Permet de sauter une deuxième fois en l’air.',
            'long_description': 'Le Double Jump est une mécanique centrale ...',
            'examples': 'Super Mario; Celeste',
            'solved_problems': 'One-dimensional level design'
        }, 
        ...]
    """
    mechanics_data = []
    for yaml_file in Path(mechanics_dir).rglob('mechanic.yaml'):
        try:
            data = read_yaml_file(yaml_file)
            if 'mechanic' in data:
                mechanic = data['mechanic']
                symbol = yaml_file.parent.name
                category = yaml_file.parent.parent.name

                mech_dict = OrderedDict([
                    ('name', process_field(mechanic.get('name', ''))),
                    ('symbol', symbol),
                    ('category', category),
                    ('short_description', process_field(mechanic.get('short_description', ''))),
                    ('long_description', process_field(mechanic.get('long_description', ''))),
                    ('examples', process_field(mechanic.get('examples', []))),
                    ('solved_problems', process_field(mechanic.get('solved_problems', '')))
                ])

                mechanics_data.append(mech_dict)
        except Exception as e:
            print(f"Error processing {yaml_file}: {e}")
    return mechanics_data


if __name__ == "__main__":
    mechanics = load_mechanics("./mechanics")
    print(f"{len(mechanics)} mécaniques trouvées.\n")
    for mech in mechanics[:2]:
        pprint.pprint(mech)
