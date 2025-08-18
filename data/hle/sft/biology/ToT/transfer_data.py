import os
import json

data_path = "../results/selected_data/ToT_biology/ToT__500_samples.json"
res_path = "../results/sft_data/ToT_biology/"
if not os.path.exists(res_path):
    os.makedirs(res_path)

with open(data_path, "r") as f:
    data = json.load(f)

new_data = []
for index, item in enumerate(data):
    new_data.append({
        "id": f"ToT_{index}",
        "question": item["question"],
        "output": f"<think>{item['output']['reasoning']}</think>{item['answer']}",
        "answer": item["answer"]
    })

with open(os.path.join(res_path, "ToT_biology_500_samples.json"), "w") as f:
    json.dump(new_data, f, indent=4)