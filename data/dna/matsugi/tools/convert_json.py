import json

# 入力ファイルと出力ファイルのパス
input_path = "adversarial_and_vanilla_dpo.json"
output_path = "DPO_adversarial_and_vanilla.json"

# ファイル読み込み
with open(input_path, "r", encoding="utf-8") as infile:
    data = json.load(infile)  # リストとして読み込む

# フォーマット変換
converted_data = []
for entry in data:
    new_entry = {
        "id": entry["id"],
        "question": entry["question"],
        "preffered_output": entry["safe_cot"] + entry["answer"],
        "non_preffered_output": entry["unsafe_cot"] + entry["answer"]
    }
    converted_data.append(new_entry)

# ファイル書き出し
with open(output_path, "w", encoding="utf-8") as outfile:
    json.dump(converted_data, outfile, indent=2, ensure_ascii=False)
