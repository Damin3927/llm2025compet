import json

# 入力ファイルと出力ファイルのパス
input_path = "vanilla_with_cot_vllm.jsonl"
output_path = "vanilla_with_cot_vllm_cot_extracted.jsonl"


def main():
    # 入力ファイルと出力ファイルのパス
    input_path = "vanilla_with_cot_vllm.jsonl"
    output_path = "vanilla_with_cot_vllm_cot_extracted.jsonl"
    new_id = 0
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            data = json.loads(line)

            problem = data.get("problem", "")
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
                "problem": problem,
                "output": output,
                "answer": answer,
            }

            # 新しいファイルに1行ずつ書き出し
            json.dump(new_data, f_out, ensure_ascii=False)
            f_out.write("\n")
            new_id += 1

if __name__ == "__main__":
    main()
