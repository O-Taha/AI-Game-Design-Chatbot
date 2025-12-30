from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import jsonlines

# =========================
# CONFIG
# =========================

MODEL_PATH = "../../TrainedModels/qbert-merged"
INFILE = "training_triplets.jsonl"
OUTDIR = "./TrainedModels/qbert-ranked"

BATCH_SIZE = 16
EPOCHS = 7
LR = 2e-5
WARMUP_STEPS = 500
STEPS_PER_EPOCH = 100
TRIPLET_MARGIN = 0.3

# =========================
# 1. Charger le modèle MERGED
# =========================

model = SentenceTransformer(MODEL_PATH)

# =========================
# 2. Charger le dataset JSONL
# =========================

train_examples = []

with open(INFILE, "r", encoding="utf-8") as f:
    reader = jsonlines.Reader(f)

    for idx, item in enumerate(reader, 1):
        try:
            anchor = item["anchor"]
            positive = item["positive"]
            negatives = item["negatives"]
        except KeyError as e:
            raise KeyError(f"Clé manquante ligne {idx}: {e}")

        for neg in negatives:
            train_examples.append(
                InputExample(texts=[anchor, positive, neg])
            )

print(f"✔ Loaded {len(train_examples)} triplets")

# =========================
# 3. DataLoader
# =========================

train_dataloader = DataLoader(
    train_examples,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# =========================
# 4. Triplet Loss
# =========================

train_loss = losses.TripletLoss(
    model=model,
    distance_metric=losses.TripletDistanceMetric.COSINE,
    triplet_margin=TRIPLET_MARGIN
)

# =========================
# 5. Entraînement
# =========================

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS,
    steps_per_epoch=STEPS_PER_EPOCH,
    optimizer_params={"lr": LR},
    show_progress_bar=True,
    output_path=OUTDIR
)

print("✅ Training finished")
