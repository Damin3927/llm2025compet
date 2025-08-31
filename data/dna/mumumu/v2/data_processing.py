import json

# 入力ファイルと出力ファイルのパス
input_path = "selected_cot_1200.jsonl"
output_path = "selected_cot_1200_extracted.jsonl"

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
       data = [json.loads(line) for line in f]

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





def main():
    new_id = 0
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)

            question = data.get("question", "")
            output = data.get("output", "")
            answer = data.get("answer", "")

            flag = True
            
            if output == answer:
                flag = False
            
            if not flag: # thinkが含まれていない場合はスキップ
                continue

            # 加工後の新しい辞書を作成
            new_data = {
                "id": new_id,
                "question": question,
                "output": output,
                "answer": answer,
            }

            # 新しいファイルに1行ずつ書き出し
            json.dump(new_data, f_out, ensure_ascii=False)
            f_out.write("\n")
            new_id += 1
            
    validate_json_file("vanilla_with_cot_vllm_cot_extracted.jsonl")    
    


if __name__ == "__main__":
    main()
