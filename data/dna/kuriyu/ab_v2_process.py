import json
import os
import re

# === 設定 ===
INPUT_JSON = "adversarial_benign_v2.jsonl"
basename = os.path.basename(INPUT_JSON).split('.jsonl')[0]
COT_JSONL = str(basename) + "_qwen3-32B.jsonl"
VALID_JSONL = str(basename) + "_filtered.jsonl"
OUTPUT_JSONL = str(basename) + "_cot_answer.jsonl"

valid_ids = set()
with open(VALID_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if "id" in item:
                valid_ids.add(item["id"])
        except Exception:
            continue

print(f"VALID 判定済み: {len(valid_ids)} 件")

done_ids = set()
if os.path.exists(OUTPUT_JSONL):
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done_ids.add(json.loads(line)["id"])
            except:
                continue
print(f"再開処理: すでに {len(done_ids)} 件完了済み")

think_pattern = re.compile(r"<think>(.*?)</think>(.*)", re.DOTALL)

def is_valid_cot_and_answer(output_text: str) -> bool:
    if not isinstance(output_text, str):
        return False
    m = think_pattern.match(output_text.strip())
    if not m:
        return False
    think_part = (m.group(1) or "").strip()
    answer_part = (m.group(2) or "").strip()
    if not think_part:
        return False
    if not answer_part:
        return False
    return True

# === マージ処理 ===
kept, skipped_not_valid, skipped_bad_cot = 0, 0, 0

with open(COT_JSONL, "r", encoding="utf-8") as fin:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception as e:
            print(f"JSON 読み込み失敗 → スキップ: {e}")
            continue

        _id = item.get("id")
        if _id is None:
            continue

        if _id in done_ids:
            continue

        if _id not in valid_ids:
            skipped_not_valid += 1
            continue

        output_text = item.get("output", "")
        if not is_valid_cot_and_answer(output_text):
            skipped_bad_cot += 1
            continue

        try:
            with open(OUTPUT_JSONL, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                fout.flush()
                os.fsync(fout.fileno())
            done_ids.add(_id)
            kept += 1
            print(f"保持 → ID {_id} 保存")
        except Exception as e:
            print(f"保存失敗 → ID {_id}: {e}")

print(f"完了: 保存 {kept} 件 / VALID外スキップ {skipped_not_valid} 件 / CoT不備スキップ {skipped_bad_cot} 件")