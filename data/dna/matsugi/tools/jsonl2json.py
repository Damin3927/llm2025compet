import json

# 入力ファイル（jsonl）と出力ファイル（json）のパス
input_file = 'vanilla_harmful_train_part_07_qwen3-32B.jsonl'
output_file = 'vanilla_harmful_train_part_07_qwen3-32B.json'

# jsonlを読み込んで各行をJSONとしてパース
data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:  # 空行を無視
            data.append(json.loads(line))

# JSON配列として保存
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"変換が完了しました: {output_file}")