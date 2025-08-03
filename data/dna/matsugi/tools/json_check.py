import json
import re

def is_valid_format(item):
    # 必須キーがすべて存在するか
    required_keys = {"id", "question", "output", "answer"}
    if not isinstance(item, dict):
        return False
    if not required_keys.issubset(item.keys()):
        return False

    # output に <think>...</think> が含まれているか
    output = item["output"]
    if not isinstance(output, str):
        return False
    if not re.match(r'^<think>.*?</think>.*', output, re.DOTALL):
        return False

    return True

def validate_json_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("❌ ルート要素はリストではありません。")
        return False

    all_valid = True
    for idx, item in enumerate(data):
        if not is_valid_format(item):
            print(f"❌ 無効な形式: インデックス {idx}, データ: {item}")
            all_valid = False

    if all_valid:
        print("✅ すべての要素が正しい形式です。")
    return all_valid

# 例：使用方法
validate_json_file("vanilla_benign_train_part_07_qwen3-32B.json")
