import os
import json

with open("chempile_qa_pairs.json", "r") as f:
    data = json.load(f)

new_items = []
for item in data:
    if "error" in item:
        continue
    new_items.append(item)

with open("chempile_qa_pairs_clean.json", "w") as f:
    json.dump(new_items, f, indent=2, ensure_ascii=False)