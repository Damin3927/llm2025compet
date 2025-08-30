import os
import json
from vllm import LLM, SamplingParams
import tqdm

def build_prompt(answer_text):
    """
    LLMに安全性を判定させるためのプロンプトを構築します。
    """
    return (
        "You are a safety classifier. "
        "If the following answer contains any harmful, hateful, or unsafe content, respond with 'YES'. "
        "If it is safe, respond with 'NO'.\n"
        f"Answer:\n{answer_text}\n"
        "Is this answer harmful? (YES or NO):"
    )

if __name__ == "__main__":
    INPUT_JSON = "vanilla_benign_train_qwen3-32B.json"
    OUTPUT_JSON = "DNA_SFT_wj_Vanilla_benign.json"

    # LLMモデルの準備
    llm = LLM(
        model="Qwen/Qwen3-32B",
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.98,
        max_model_len= 8192,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=5,
        stop=["\n"]
    )

    # 入力JSONファイルの読み込み
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # プロンプトのリストを作成
    prompts = [build_prompt(item.get("answer", "")) for item in data]
    
    # LLMで一括して有害性を判定
    print("LLMによる有害性の判定を開始します...")
    outputs = llm.generate(prompts, sampling_params)
    print("判定が完了しました。")

    safe_data = []
    new_id_counter = 1 

    # 判定結果を確認し、「NO」（安全）と判定されたデータのみを抽出
    for i, output in tqdm.tqdm(enumerate(outputs), total=len(outputs), desc="安全なデータをフィルタリング中"):
        judgement = output.outputs[0].text.strip().upper()
        if "NO" in judgement:
            # 元のデータを取得
            item = data[i]
            # idを新しい連番に更新
            item['id'] = new_id_counter
            safe_data.append(item)
            new_id_counter += 1 

    # 抽出された安全なデータをJSON配列形式でファイルに書き出す
    print(f"{len(safe_data)}件の安全なデータが見つかりました。ファイルに書き出します...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(safe_data, f, indent=2, ensure_ascii=False)

    print(f"処理が完了しました。出力ファイル: {OUTPUT_JSON}")
