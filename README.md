# AI Game Design Chatbot
## Description
This project is an AI helper chatbot for [Mechadex](https://mechadex.github.io/) \
By running main.py, you're prompted to give a description of the *game design* promblem/mechanic you're encountering/looking for.

**This tool is perfect for problem-solving or brainstorming**

The model used was trained on queries such as :
>- **anchor**: "How can I fix unclear progression paths?" -> "Skill Trees"
>- **anchor**: "I'm working on the online mode of my game and players end up against opponents who are too strong for them." -> "Matchmaking"
>- **anchor**: "The genre of my game is Action RPG. Traditional game dialogue is a one-way street where the player just listens to an NPC's exposition, which can be passive and boring. What mechanic would solve this?" -> "Branching Dialogue"

## Instructions
Link to the [model used: qbert-ranked](https://drive.google.com/drive/folders/1WtwyxcCjWkQ42nAEmi7ZxbUx9F4M61J3?usp=sharing)
1. Clone the [Mechadex repository](https://github.com/Mechadex/mechanics) for local access to mechanics
2. Download and put the whole [```qbert-ranked/```](https://drive.google.com/drive/folders/1WtwyxcCjWkQ42nAEmi7ZxbUx9F4M61J3?usp=sharing)  directory inside ```TrainedModels/```, your local tree should look like this :
```
AI-Game-Design-Chatbot/         
├── DAPT/                       
├── LoRA/                       
├── mechanics/     ---------->  mechanics/
├── RAGIndex/                   ├── .github/
├── TrainedModels/              ├── Actions/                         
│   └── qbert-ranked/           ├── AI/                              
├── generate_training_json.py   └── ...                                    
├── main.py                                             
├── mechdex_repo_retrieval.py                                               
├── RAG_FAISS.py                                                
├── README.md                                               
├── requirements.txt                                                
├── text_embedding.py                                               
└── yaml_parsing.py                                             
```

3. Run ```python main.py``` inside your repo folder
4. Let it load for ~10 sec (Building embeddings, loading mechanics...)
5. Ask your question and press 'enter' !

### Dependencies
Main dependencies depend on your usage of this repo. : (see ```requirements.txt```)

| Dependency           | Version     | Usage                         |
|---------------------|-------------|-------------------------------|
| huggingface_hub     | 0.35.3      | For Inference                 |
| transformers        | 4.57.0      | For Inference                 |
| torch               | 2.9.1       | For Inference                 |
| tqdm                | 4.67.1      | For Inference                 |
| sentence_transformers| 5.2.0      | For Inference                 |
| faiss_cpu           | 1.12.0      | For Inference                 |
| PyYAML              | 6.0.3       | For Inference                 |
| PyYAML              | 5.4.1       | For Inference                 |
| numpy               | 2.4.0       | For Inference                 |
| matplotlib          | 3.10.8      | For Inference                 |
| safetensors          | 0.7.0       | For Inference                 |
| peft                | 0.18.0      | For Dataset/Fine-Tuning/Contributing |
| beautifulsoup4       | 4.14.3      | For Dataset/Fine-Tuning/Contributing |
| jsonlines           | 4.0.0       | For Dataset/Fine-Tuning/Contributing |
| datasets            | 4.3.0       | For Dataset/Fine-Tuning/Contributing |
| httplib2            | 0.20.2      | For Dataset/Fine-Tuning/Contributing |
| lunr                | 0.8.0       | For Dataset/Fine-Tuning/Contributing |
| Requests            | 2.32.5      | For Dataset/Fine-Tuning/Contributing |
| textdistance        | 4.6.3       | For Dataset/Fine-Tuning/Contributing |
| wikipedia           | 1.4.0       | For Dataset/Fine-Tuning/Contributing |


## Project Pipeline
Here's the project pipeline, straight from my class project presentation :

![Project Pipeline](https://raw.githubusercontent.com/O-Taha/AI-Game-Design-Chatbot/refs/heads/main/Pr%C3%A9sentation%20Projet%20IA%20-%20Game%20Design%20Chatbot.jpg)

```mechdev_repo_retrieval.py```\
Fetches game design mechanics from Mechadex repository.

```yaml_parsing.py```\
Parses game design mechanics defined in YAML files into a python-friendly object.

```text_embedding.py```\
Encodes game design problems, mechanics, and queries into dense semantic vectors using the Q*bert model.
Provides the shared embedding space used for similarity search and ranking.

```rag_faiss.py```\
Performs semantic retrieval over embedded game design mechanics using a FAISS vector index.
Selects the most relevant mechanics given a designer’s query before final ranking and explanation.

## Credits
- Thanks to Google for bert-base-uncased
- A huge thanks to @DarkWolfX2244 & @legibleguy for creating and contributing to Mechadex, and allowing me to use it for this project.