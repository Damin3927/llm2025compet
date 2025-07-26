from datasets import load_dataset
import json

with open("keys.json", "r") as f:
    keys = json.load(f)

hf_token = keys["HF_TOKEN"]

# This will now work correctly with your prefix structure
dataset = load_dataset("neko-llm/SFT_OpenMathReasoning", token=hf_token)

# Access splits
train_data = dataset["train"]      # ✅ Works
val_data = dataset["validation"]   # ✅ Works
test_data = dataset["test"]        # ✅ Works

print(f"Train: {len(train_data)} examples")
print(f"Validation: {len(val_data)} examples")
print(f"Test: {len(test_data)} examples")