import os
import sys
import json
import requests
import argparse
import time
import re
from datasets import load_dataset

# --- 設定 ---
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "deepseek/deepseek-r1:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
# ★★★ streaming=False に変更 ★★★
DATASET_NAME = "neko-llm/HLE_RL_OlympiadBench"
DATASET_SPLIT = "train"
JSONL_OUTPUT_FILE = "output.jsonl"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

# --- プロンプトテンプレート ---
# (前回と同じ内容)
INITIAL_PROMPT_TEMPLATE = """
You are a brilliant mathematician. Your task is to solve the following math problem step-by-step.
**You MUST conduct your entire thinking process and reasoning in English only.**
First, provide your entire thought process wrapped in <think> and </think> tags.
After the thought process, state the final answer, wrapping the final numerical result by LaTeX expression in $\\boxed{}$.

Please follow these steps within your thinking process:
1.  Analyze the problem and clarify the definitions of the terms used.
2.  List all the constraints of the problem.
3.  Devise a strategy to find the solution.
4.  Execute the strategy with logical reasoning and step-by-step calculations. Explain why each step is necessary.
5.  If there are any key insights or logical leaps that help narrow down the candidates, explain them in detail.
6.  Derive the final conclusion that leads to the answer.

Here is an example of the required format:
Question: What is 1 + 1?
Answer: <think>I need to calculate the sum of 1 and 1. This is a simple addition operation. The result of adding 1 and 1 is 2.</think>1 + 1 is $\\boxed{2}$.

---

Now, solve the following problem.

Question: 
"""

SELF_CORRECTION_PROMPT_TEMPLATE = """
You are a meticulous editor. Your task is to review and finalize the following draft solution.
Ensure the final output strictly follows these three rules:
1.  **Language:** The entire thinking process inside the `<think>`...`</think>` tags MUST be in English only. Remove any non-English characters (e.g., Chinese, Japanese).
2.  **CoT Tags:** The thinking process must be correctly enclosed in one set of `<think>` and `</think>` tags.
3.  **Answer Box:** The final numerical answer must be enclosed in `\\boxed{}`.

If the draft is already perfect, output it as is. If it needs corrections, provide the corrected, final version of the entire response.

### Draft Solution to Review:
"""

def normalize_answer(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    return re.sub(r'[\s(){}\\$]', '', s)

def generate_cot_for_questions(start_id, num_to_process):
    if not API_KEY:
        print("エラー: 環境変数 'OPENROUTER_API_KEY' が設定されていません。", file=sys.stderr)
        return

    print(f"スクリプトを開始します... (開始ID: {start_id}, 問題数: {num_to_process})")

    # ★★★ 1. 処理済みのIDを先に読み込む ★★★
    processed_ids = set()
    if os.path.exists(JSONL_OUTPUT_FILE):
        print(f"'{JSONL_OUTPUT_FILE}'を読み込み、処理済みのIDをチェックします...")
        with open(JSONL_OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 各行をJSONとして解釈し、'id'キーの値を取得
                    processed_ids.add(json.loads(line)['id'])
                except (json.JSONDecodeError, KeyError):
                    # 形式が正しくない行は無視
                    continue
        print(f"完了。{len(processed_ids)}件の処理済みIDが見つかりました。")

    try:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=False)
        id_list = list(dataset['id'])
        start_index = id_list.index(start_id)
        end_index = start_index + num_to_process
        questions_to_process = dataset.select(range(start_index, min(end_index, len(dataset))))
        print(f"{len(questions_to_process)}問の問題を取得しました。")
    except (ValueError, Exception) as e:
        print(f"エラー: データセットの読み込みまたはIDの検索に失敗しました。詳細: {e}", file=sys.stderr)
        return

    with open(JSONL_OUTPUT_FILE, "a", encoding="utf-8") as jsonl_file:
        processed_count = 0
        for i, item in enumerate(questions_to_process):
            current_id = item.get('id')
            
            # ★★★ 2. IDが処理済みセットに含まれているかチェック ★★★
            if current_id in processed_ids:
                print(f"スキップ (ID: {current_id}): 既に'{JSONL_OUTPUT_FILE}'に存在します。")
                continue

            question_text = item.get('question', '')
            original_answer = item.get('answer', '')
            original_solution = item.get('solution', '')

            if original_answer == 'NO_SOLUTION_FOUND':
                print(f"スキップ (ID: {current_id}): answerがNO_SOLUTION_FOUNDです。")
                continue

            print(f"処理中 {i+1}/{len(questions_to_process)} (ID: {current_id})...")

            for attempt in range(MAX_RETRIES):
                try:
                    # (API呼び出しとデータ処理のロジックは変更なし)
                    print(f"  (ID: {current_id}) ステップ1: 初期生成中...")
                    initial_prompt = f"{INITIAL_PROMPT_TEMPLATE}{question_text}\n\nAnswer:"
                    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
                    payload1 = { "model": MODEL_NAME, "messages": [{"role": "user", "content": initial_prompt}], "include_reasoning": True }
                    response1 = requests.post(API_URL, headers=headers, data=json.dumps(payload1), timeout=180)
                    response1.raise_for_status()
                    initial_response_data = response1.json()
                    initial_output = initial_response_data['choices'][0]['message']['content']

                    print(f"  (ID: {current_id}) ステップ2: 自己修正中...")
                    correction_prompt = f"{SELF_CORRECTION_PROMPT_TEMPLATE}\n{initial_output}\n\n### Final, Corrected Solution:"
                    payload2 = { "model": MODEL_NAME, "messages": [{"role": "user", "content": correction_prompt}], "include_reasoning": True }
                    response2 = requests.post(API_URL, headers=headers, data=json.dumps(payload2), timeout=180)
                    response2.raise_for_status()
                    final_response_data = response2.json()

                    reasoning_text = final_response_data['choices'][0]['message'].get('reasoning', "No reasoning field found.")
                    CoT = f"<think>\n{reasoning_text}\n</think>"
                    final_content_text = final_response_data['choices'][0]['message'].get('content', "No content field found.")
                    match = re.search(r'\\boxed\{(.*?)\}', final_content_text, re.DOTALL)
                    answer_CoT_value = match.group(1) if match else "EXTRACTION_FAILED"
                    is_correct = normalize_answer(original_answer) == normalize_answer(answer_CoT_value)

                    output_record = {
                        "id": current_id, "question": question_text, "output": CoT,
                        "answer": original_answer, "solution": original_solution,
                        "answer_CoT": answer_CoT_value, "is_correct": is_correct
                    }
                    
                    jsonl_file.write(json.dumps(output_record, ensure_ascii=False) + '\n')
                    jsonl_file.flush()
                    print(f"成功 (ID: {current_id}): 結果を'{JSONL_OUTPUT_FILE}'に書き込みました。(正解: {is_correct})")
                    processed_count += 1
                    
                    break

                except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
                    print(f"リトライ {attempt + 1}/{MAX_RETRIES}: エラーが発生しました ({type(e).__name__})。", file=sys.stderr)
                    if attempt < MAX_RETRIES - 1:
                        print(f"{RETRY_DELAY_SECONDS}秒後に再試行します...", file=sys.stderr)
                        time.sleep(RETRY_DELAY_SECONDS)
                    else:
                        print(f"エラー (ID: {current_id}): {MAX_RETRIES}回のリトライ後も失敗しました。", file=sys.stderr)
                
                except (KeyError, IndexError) as e:
                    print(f"エラー (ID: {current_id}): レスポンス解析エラー: {e}", file=sys.stderr)
                    break

    print(f"\n処理が完了しました。新たに{processed_count}件の問題を'{JSONL_OUTPUT_FILE}'に出力しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hugging Faceデータセットから数学問題のCoTを生成し、JSONL形式で出力します。")
    parser.add_argument("--start_id", type=int, default=0, help="処理を開始するデータセットの'id'列の値。")
    parser.add_argument("--num_questions", type=int, default=3, help="処理する問題の数。")
    args = parser.parse_args()
    generate_cot_for_questions(args.start_id, args.num_questions)