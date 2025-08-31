import json
import os

INPUT_JSONL = "adversarial_benign_v2_cot_answer.jsonl"
basename = os.path.basename(INPUT_JSONL).split('.jsonl')[0]
OUTPUT_JSONL = str(basename) + "_reid.jsonl"

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

print(f"読み込み: {len(data)} 件")

new_data = []
for i, item in enumerate(data, start=1):
    item["id"] = str(i)
    new_data.append(item)

with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for item in new_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"保存完了: {OUTPUT_JSONL}, 件数 = {len(new_data)}")