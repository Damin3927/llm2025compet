from datasets import load_dataset
import json

with open("keys.json", "r") as f:
    keys = json.load(f)

hf_token = keys["llm"]

# This will now work correctly with your prefix structure
dataset = load_dataset("neko-llm/SFT_OpenMathReasoning", token=hf_token)

# show splits
print(dataset.keys())

# print the datasize of each split
for split in dataset.keys():
    print(f"{split}: {len(dataset[split])} examples")