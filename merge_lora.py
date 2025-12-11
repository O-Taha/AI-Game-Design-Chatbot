"""
merge_lora.py
-------------
Merge a PEFT LoRA adapter into a base BERT model.

Input:
    - BASE_MODEL: the base checkpoint (bert-base-uncased)
    - LORA_PATH: directory containing adapter_model.safetensors, adapter_config.json
Output:
    - MERGED_PATH: clean BERT model with LoRA weights merged permanently
"""

from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import torch

BASE_MODEL = "bert-base-uncased"
LORA_PATH = "TrainedModels/qbert-lora"
MERGED_PATH = "TrainedModels/qbert-merged"

print("\n=== MERGE LORA → BERT ===\n")
print("Base model:     ", BASE_MODEL)
print("LoRA adapter:   ", LORA_PATH)
print("Output folder:  ", MERGED_PATH, "\n")

# Load base model
print("[1/4] Loading base model...")
base_model = AutoModel.from_pretrained(BASE_MODEL)

# Load lora adapter into base
print("[2/4] Loading LoRA adapter...")
lora_model = PeftModel.from_pretrained(base_model, LORA_PATH)

# Merge LoRA weights
print("[3/4] Merging LoRA weights into base model...")
merged_model = lora_model.merge_and_unload()

# Save
print("[4/4] Saving merged model...")
merged_model.save_pretrained(MERGED_PATH)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
tokenizer.save_pretrained(MERGED_PATH)

print("\n=== DONE! ===")
print("Merged model saved to:", MERGED_PATH)
print("This model no longer uses PEFT and cannot throw 'labels' errors.\n")
