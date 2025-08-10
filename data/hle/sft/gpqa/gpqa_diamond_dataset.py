import os
import pandas as pd
import requests
from tqdm import tqdm
import time
import json
import re
import datasets
from huggingface_hub import upload_file

# --- OpenRouter API Settings ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "deepseek/deepseek-r1-0528:free"
YOUR_SITE_URL = "http://localhost"

# --- App Name ---
APP_NAME = "GPQA add CoT Dataset"

# --- Hugging Face Upload Settings: Repository Name ---
OUTPUT_DATASET_ID = "neko-llm/HLE_SFT_GPQA_Diamond"


def generate_prompt_for_gpqa_dataset(question: str, explanation: str, subdomain: str) -> str:
  """
  プロンプトを生成する関数

  - GPQA Dataset のデータを受け取り、フォーマットに従ってプロンプトを生成する。
  """

  return f"""
    You are an expert in {subdomain}.

    ## Rules
    - Using the information from question/explanation, please generate a reasoning process (Reasoning/CoT) and answer.
      - question: {question}
      - explanation: {explanation}

    ## Output Format
    - Please output the reasoning process (Reasoning/CoT) and answer in the following format:
    ```json
    {{
        "reasoning": "reasoning process (Reasoning/CoT) content",
        "answer": "answer content"
    }}
    ```
  """

def generate_judge_prompt(generated_answer: str, correct_answer: str, question: str, subdomain: str) -> str:
    """
    LLM判定用のプロンプトを生成する関数

    Args:
        generated_answer: 生成された回答
        correct_answer: 正解
        question: 元の質問
        subdomain: 分野

    Returns:
        str: 判定用プロンプト
    """
    return f"""
You are an expert judge for evaluating the correctness of answers to scientific questions in {subdomain}.
Decide if the generated answer is semantically equivalent to the reference answer.

## Inputs
[Question]
{question}

[Generated Answer]
{generated_answer}

[Reference Answer (Correct)]
{correct_answer}

## What to do (follow silently)
1) Extract the final asserted answer from each text:
   - Ignore explanations, apologies, or prefaces.
   - If multiple values appear, use the final, explicitly stated conclusion (e.g., after "Answer:", or the last value/choice).
2) Normalize notation for BOTH answers before comparing:
   - Trim spaces, punctuation, and surrounding quotes.
   - Scientific notation: treat these as equivalent forms:
     10^-4, 10^{{-4}}, 1e-4, 1×10^-4, 1·10^-4, 0.0001
   - Minus sign: treat "-" and "−" as identical.
   - Multiplication sign: treat "×", "·", "*" as identical.
   - LaTeX vs plain text: interpret \(10^{{-4}}\) == 10^-4, \frac{{1}}{{2}} == 0.5, etc.
   - Units:
     * Be case-insensitive where standard (eV == electronvolt).
     * Accept SI prefixes & micro symbol variants (μ == u).
     * If units are convertible and dimensionally the same (e.g., eV vs J), convert conceptually and compare values.
     * If the generated answer omits an unambiguous unit but the numeric value matches within tolerance, treat as equivalent.
   - Percent/fraction equivalence: 10% == 0.10.
   - Thousands separators and locale decimals: ignore commas, treat "." as decimal point.
3) Compare using these criteria:
   - Multiple choice: if the question lists options, map letters (A/B/C/…) to their option text and compare to the reference.
   - Exact strings (non-numeric): compare case-insensitively after normalization; accept common synonyms (True/Yes, False/No).
   - Numbers:
     * Consider correct if values match exactly AFTER normalization OR
       |generated - reference| ≤ max(1e-12, 0.01×|reference|)  (≈1% relative tolerance).
     * Also accept rounding consistent with the significant figures of the reference.
   - Chemical formulas: compare structural/stoichiometric equivalence; ignore whitespace and typical state annotations.
4) Mark INCORRECT if there is a sign error, wrong order of magnitude, wrong (non-convertible) units/dimensions, or a different option than the reference.

## Output Format
Respond with EXACTLY one token:
- "CORRECT"  (semantically equivalent after normalization)
- "INCORRECT" (otherwise)

Only output "CORRECT" or "INCORRECT".
"""



def judge_answer_by_llm(generated_answer, correct_answer: str, question: str = "", subdomain: str = "") -> bool:
    """
    OpenRouter経由でLLMに回答の正当性を判定させる関数

    Args:
        generated_answer: 生成された回答（str または list）
        correct_answer: 正解
        question: 元の質問（判定の参考用、オプション）

    Returns:
        bool: 回答が正しい場合True、間違いの場合False
    """
    # generated_answerがリスト型の場合は文字列に変換
    if isinstance(generated_answer, list):
        generated_answer = str(generated_answer)
    elif not isinstance(generated_answer, str):
        generated_answer = str(generated_answer)

    if not OPENROUTER_API_KEY:
        print("⚠️ OPENROUTER_API_KEY が設定されていません。単純な文字列比較を使用します。")
        return generated_answer.strip().lower() == correct_answer.strip().lower()

    # 判定用プロンプトを生成
    judge_prompt = generate_judge_prompt(generated_answer, correct_answer, question, subdomain)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": f"{APP_NAME} - Answer Judge",
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0.1,  # 判定の一貫性のため低い温度を使用
        "max_tokens": 10     # "CORRECT" または "INCORRECT" のみ期待
    }

    last_error = ""

    for attempt in range(3):  # 3回まで再試行
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            json_response = response.json()

            if 'choices' in json_response and len(json_response['choices']) > 0:
                content = json_response['choices'][0]['message']['content'].strip().upper()

                print(f"🔍 Judge API Response: {content}")

                if "CORRECT" in content:
                    return True
                elif "INCORRECT" in content:
                    return False
                else:
                    print(f"⚠️ 予期しない判定結果: {content}")
                    # フォールバック: 単純な文字列比較
                    return generated_answer.strip().lower() == correct_answer.strip().lower()
            else:
                last_error = f"❌ Judge API response missing valid content. Response: {json_response}"
                print(last_error)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [402, 429]:
                try:
                    error_details = e.response.json().get('error', {}).get('message', '')
                except json.JSONDecodeError:
                    error_details = e.response.text
                print(f"❌ Judge API credit/rate limit error ({e.response.status_code}): {error_details}")
                # フォールバック: 単純な文字列比較
                return generated_answer.strip().lower() == correct_answer.strip().lower()
            else:
                last_error = f"❌ Judge API HTTP Error: {e}"
        except Exception as e:
            last_error = f"❌ Judge API unknown error: {e}"

        time.sleep(1)  # 再試行前に1秒待機

    print(f"❌ Judge API failed after 3 attempts: {last_error}")
    print("⚠️ フォールバックとして単純な文字列比較を使用します。")
    return generated_answer.strip().lower() == correct_answer.strip().lower()

def extract_final_answer_from_cot(cot_response: str) -> str:
    """
    CoTレスポンスから最終回答を抽出する関数
    JSON形式のレスポンスから"answer"フィールドを抽出する

    Args:
        cot_response: CoT生成APIからのレスポンス（JSON形式）

    Returns:
        str: 抽出された最終回答
    """
    # まずJSON形式での解析を試行
    parsed_json = parse_json_response(cot_response)
    if parsed_json and 'answer' in parsed_json:
        return parsed_json['answer'].strip()

    # JSONでない場合のフォールバック処理
    # <think>タグがある場合、</think>の後の内容を最終回答とする
    think_end_pattern = r'</think>\s*(.*?)$'
    match = re.search(think_end_pattern, cot_response, re.DOTALL | re.IGNORECASE)
    if match:
        final_answer = match.group(1).strip()
        if final_answer:
            return final_answer

    # <think>タグがない場合、最後の行または段落を最終回答とする
    lines = cot_response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('-'):
            return line

    # フォールバック: 全体をそのまま返す
    return cot_response.strip()


def generate_cot_with_openrouter(prompt: str) -> tuple[str, str]:
    """
    OpenRouter API を使って CoT を生成する関数

    Returns:
        tuple[status, response]: (成功/失敗, API応答またはエラーメッセージ)
    """
    if not OPENROUTER_API_KEY:
        return "failure", "❌ OPENROUTER_API_KEY environment variable not set."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": YOUR_SITE_URL,
        "X-Title": APP_NAME,
        "Content-Type": "application/json"
    }

    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }

    last_error = ""

    for _attempt in range(3):  # 3回まで再試行
        try:
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=120)
            response.raise_for_status()
            json_response = response.json()
            # print(f"🔍 OpenRouter API レスポンス: {json_response}")

            if 'choices' in json_response and len(json_response['choices']) > 0:
                content = json_response['choices'][0]['message']['content']

                # reasoningキーが存在するかチェック（DeepSeek R1特有）
                reasoning = ""
                if 'reasoning' in json_response['choices'][0]:
                    reasoning = json_response['choices'][0]['reasoning']
                else:
                    print("⚠️ reasoning フィールドが見つかりませんでした。contentのみを使用します。")

                print(f"🔍 OpenRouter API Content: {content}")
                if reasoning:
                    print(f"🔍 OpenRouter API Reasoning: {reasoning}")

                # contentのみを返す（統一された戻り値）
                return "success", content.strip()
            else:
                last_error = f"❌ API response missing valid content. Response: {json_response}"
                print(f"❌ API response missing valid content. Response:: {json_response}")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code in [402, 429]:
                try:
                    error_details = e.response.json().get('error', {}).get('message', '')
                except json.JSONDecodeError:
                    error_details = e.response.text
                final_error_message = f"❌ Possible credit exhaustion or rate limit ({e.response.status_code}): {error_details}"
                return "failure", final_error_message
            else:
                last_error = f"❌ HTTP Error: {e}"
        except Exception as e:
            last_error = f"❌ An unknown error occurred: {e}"

        time.sleep(1)  # 再試行前に1秒待機

    return "failure", last_error


def parse_json_response(response_text: str) -> dict:
    """
    API レスポンスから JSON を抽出・パースする関数

    Args:
        response_text: APIレスポンステキスト

    Returns:
        dict: パースされた JSON データ、失敗時は空の辞書
    """
    if not response_text or not response_text.strip():
        print("⚠️ Empty response text provided to JSON parser")
        return {}

    # 複数のJSON抽出パターンを試行
    json_patterns = [
        # ```json ... ``` の形式
        r'```json\s*(.*?)\s*```',
        # ``` ... ``` の形式（jsonキーワードなし）
        r'```\s*(.*?)\s*```',
        # { ... } の形式を直接探す
        r'(\{.*?\})',
    ]

    for pattern in json_patterns:
        try:
            json_match = re.search(pattern, response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                parsed = json.loads(json_str)

                # 期待するキーが含まれているかチェック
                if isinstance(parsed, dict) and ('reasoning' in parsed or 'answer' in parsed):
                    return parsed

        except (json.JSONDecodeError, AttributeError) as e:
            continue  # 次のパターンを試行

    # 最後の手段: 全体をJSONとして試行
    try:
        parsed = json.loads(response_text.strip())
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # JSON解析に完全に失敗した場合
    print(f"⚠️ すべてのJSON解析パターンが失敗しました。レスポンス: {response_text[:200]}...")
    return {}


def generate_readme_content(df: pd.DataFrame, csv_filename: str, parquet_filename: str, jsonl_filename: str, total_count: int, success_count: int) -> str:
    """
    README.md の内容を生成する関数

    Args:
        df: データセットのDataFrame
        csv_filename: CSVファイル名
        parquet_filename: Parquetファイル名
        jsonl_filename: JSONLファイル名
        total_count: 総処理数
        success_count: 成功数

    Returns:
        str: README.mdの内容
    """
    # データの列名を取得
    columns = list(df.columns) if not df.empty else []

    # サンプルデータの取得（最初の1件）
    sample_data = df.iloc[0].to_dict() if not df.empty else {}

    readme_content = f"""# HLE SFT GPQA Diamond Dataset

## 概要

このデータセットは、GPQA (Graduate-level Google-proof Q&A) Diamond データセットを基に、Chain of Thought (CoT) 推論を追加して生成されたSupervised Fine-Tuning (SFT) 用のデータセットです。

専門的な科学分野（物理学、化学、生物学）における高度な質問に対して、段階的な推論プロセスを含む回答を提供します。

## データセット統計

- **総問題数**: {total_count:,} 問
- **成功生成数**: {success_count:,} 問
- **成功率**: {success_count/total_count*100:.1f}%

## ファイル形式

このデータセットは以下の3つの形式で提供されています：

### 1. CSV形式 (`{csv_filename}`)
- 一般的な表形式データ
- Excel やスプレッドシートソフトで開けます
- Pandas で簡単に読み込み可能

### 2. Parquet形式 (`{parquet_filename}`)
- 高効率なカラム型データ形式
- 大規模データの高速読み込みに最適
- Apache Arrow エコシステムで推奨

### 3. JSONL形式 (`{jsonl_filename}`)
- JSON Lines 形式（1行1レコード）
- ストリーミング処理に適している
- 機械学習フレームワークで広く対応

## データ構造

各レコードには以下のフィールドが含まれています：

| フィールド名 | 型 | 説明 |
|-------------|----|----|"""

    # 列情報を動的に追加
    for col in columns:
        if col == 'id':
            readme_content += f"\n| `{col}` | int | 問題の一意識別子 |"
        elif col == 'question':
            readme_content += f"\n| `{col}` | str | 元の質問文 |"
        elif col == 'output':
            readme_content += f"\n| `{col}` | str | CoT推論過程を含む完全な回答（&lt;think&gt;...&lt;/think&gt;形式） |"
        elif col == 'answer':
            readme_content += f"\n| `{col}` | str | 最終的な正解 |"
        else:
            readme_content += f"\n| `{col}` | str | {col} |"

    readme_content += f"""

## サンプルデータ

```json
{json.dumps(sample_data, ensure_ascii=False, indent=2)}
```

## 使用方法

### Python (Pandas)

```python
import pandas as pd

# CSV形式で読み込み
df = pd.read_csv('{csv_filename}')

# Parquet形式で読み込み（推奨）
df = pd.read_parquet('{parquet_filename}')

# JSONL形式で読み込み
df = pd.read_json('{jsonl_filename}', lines=True)

print(f"データセットサイズ: {{len(df)}} 問")
print(df.head())
```

### Hugging Face Datasets

```python
from datasets import load_dataset

# このリポジトリから直接読み込み
dataset = load_dataset("neko-llm/HLE_SFT_GPQA_Diamond")

# 特定のファイル形式を指定
dataset = load_dataset("neko-llm/HLE_SFT_GPQA_Diamond", data_files="{parquet_filename}")
```

### PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class GPQADataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {{
            'question': row['question'],
            'output': row['output'],
            'answer': row['answer']
        }}

# データ読み込み
df = pd.read_parquet('{parquet_filename}')
dataset = GPQADataset(df)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## 生成方法

このデータセットは以下の手順で生成されました：

1. **元データ**: [GPQA Diamond Dataset](https://huggingface.co/datasets/Idavidrein/gpqa) の train split を使用
2. **CoT生成**: DeepSeek-R1-0528:free モデルを使用して推論過程を生成
3. **フォーマット**: `<think>推論過程</think>最終回答` の形式で構造化
4. **品質管理**: API呼び出しの成功/失敗を記録し、品質を担保

### 生成に使用したモデル
- **モデル**: `{MODEL_NAME}`
- **API**: OpenRouter API
- **生成方式**: JSON構造化プロンプトによる推論生成
- **出力形式**: `{{"reasoning": "推論過程", "answer": "回答"}}`のJSON形式

## ライセンス

このデータセットは元のGPQAデータセットのライセンスに従います。学術研究目的での使用を推奨します。

## 引用

このデータセットを使用する場合は、以下を引用してください：

```bibtex
@dataset{{hle_sft_gpqa_diamond,
  title={{HLE SFT GPQA Diamond Dataset with Chain of Thought}},
  author={{neko-llm}},
  year={{2024}},
  url={{https://huggingface.co/datasets/neko-llm/HLE_SFT_GPQA_Diamond}}
}}
```

元のGPQAデータセットの引用：
```bibtex
@article{{rein2023gpqa,
  title={{GPQA: A Graduate-Level Google-Proof Q&A Benchmark}},
  author={{Rein, David and Hou, Betty Li and Stickland, Asa Cooper and Petty, Jackson and Pang, Richard Yuanzhe and Dirani, Julien and Michael, Julian and Bowman, Samuel R}},
  journal={{arXiv preprint arXiv:2311.12022}},
  year={{2023}}
}}
```

## 問い合わせ

データセットに関する質問や問題がある場合は、[Issues](https://huggingface.co/datasets/neko-llm/HLE_SFT_GPQA_Diamond/discussions) でお知らせください。
"""

    return readme_content


def main():
    """
    GPQA Dataset の CoT (Reasoning) 生成 & フォーマット処理を行う Main Script。

    NOTE:
    関数の機能:
    1. GPQA Dataset を読み込む。
    2. GPQA Dataset を処理して、作成するデータセットのフォーマットに変換する。
    3. 作成したデータセットを local に保存する。
    4. 作成したデータセットを Hugging Face にアップロードする。
    """
    print("🚀 GPQA Dataset CoT 生成処理を開始します...")

    # 出力ファイル設定
    output_filename = "gpqa_diamond_cot_dataset.csv"

    # --- Resume Logic ---
    processed_ids = []
    if os.path.exists(output_filename):
        print(f"📄 既存の結果ファイルを発見: '{output_filename}'")
        try:
            existing_df = pd.read_csv(output_filename)
            if 'id' in existing_df.columns:
                processed_ids = existing_df['id'].dropna().tolist()
                print(f"✅ {len(processed_ids)} 件の処理済み問題を発見。これらはスキップされます。")
        except pd.errors.EmptyDataError:
            print("⚠️ 既存の結果ファイルが空です。新規で開始します。")
        except Exception as e:
            print(f"⚠️ 既存ファイルの読み込みに失敗。新規で開始します。エラー: {e}")

    # --- GPQA Dataset Loading ---
    print("📥 GPQA Dataset (gpqa_diamond) を読み込み中...")
    try:
        # GPQA の gpqa_diamond subset を読み込み（trainデータのみ）
        diamond_dataset = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", split='train')
        print(f"✅ データセット読み込み完了。{len(diamond_dataset)} 件の問題を発見。")

        # サンプルデータの確認
        print("📋 サンプルデータ:")
        sample = diamond_dataset[0]
        for key in sample.keys():
            print(f"  {key}: {str(sample[key])[:100]}...")

    except Exception as e:
        print(f"❌ データセット読み込みエラー: {e}")
        return

    # --- Main Processing Loop ---
    batch_results = []
    total_problems = len(diamond_dataset)

    # 統計情報の初期化
    total_processed = 0
    total_generated = 0
    total_judged_correct = 0
    total_api_failures = 0

    print(f"\n🚀 {total_problems} 件の問題に対してCoT生成を開始...")
    print(f"💾 結果は '{output_filename}' に5件ずつバッチで保存されます。")
    print(f"🔍 回答の正当性判定を有効化しています。正しい回答のみが保存されます。")

    for i, item in enumerate(tqdm(diamond_dataset, desc="CoT生成中")):
        # 既に処理済みの場合はスキップ
        item_id = i + 1  # IDを1から開始
        if item_id in processed_ids:
            continue

        total_processed += 1
        print(f"\n--- 問題 {i + 1}/{total_problems} を処理中 ---")

        # プロンプト生成
        prompt = generate_prompt_for_gpqa_dataset(
            question=item['Question'],
            explanation=item['Explanation'],
            subdomain=item['Subdomain']
        )
        print(f"🔍 プロンプト: {prompt[:200]}...")

        # OpenRouter API でCoT生成
        status, content = generate_cot_with_openrouter(prompt)
        print(f"🔍 API ステータス: {status}")

        if status == "success":
            total_generated += 1
            print(f"🔍 API レスポンス(Content): {content[:200]}...")

            # JSON形式のレスポンスをパース
            parsed_json = parse_json_response(content)

            if parsed_json and 'reasoning' in parsed_json and 'answer' in parsed_json:
                # JSON形式の場合
                reasoning = parsed_json['reasoning']
                generated_answer = parsed_json['answer']
                print(f"🔍 JSON解析成功:")
                print(f"   推論過程: {reasoning[:100]}...")
                print(f"   生成回答: {generated_answer}")

                # 回答の正当性を判定
                print(f"⚖️ 回答の正当性を判定中...")
                is_correct = judge_answer_by_llm(
                    generated_answer=generated_answer,
                    correct_answer=item['Correct Answer'],
                    question=item['Question'],
                    subdomain=item['Subdomain']
                )

                if is_correct:
                    total_judged_correct += 1
                    print(f"✅ 判定結果: 正解! データを保存します。")

                    # 正解の場合のみ結果を保存
                    batch_results.append({
                        "id": item_id,
                        "question": item['Question'],
                        # 推論過程(Reasoning/CoT)を<think>...</think>タグで囲み、最終回答を後に記載
                        "output": f"<think>{reasoning}</think>{item['Correct Answer']}",
                        "answer": item['Correct Answer'],
                        "generated_answer": generated_answer,
                        "judgment_status": "correct"
                    })
                else:
                    print(f"❌ 判定結果: 不正解。このデータは保存されません。")
                    print(f"   生成回答: {generated_answer}")
                    print(f"   正解: {item['Correct Answer']}")
            else:
                # JSON解析失敗時のフォールバック処理
                print(f"⚠️ JSON解析失敗、フォールバック処理を実行...")
                generated_answer = extract_final_answer_from_cot(content)
                print(f"🔍 フォールバックで抽出された最終回答: {generated_answer}")

                # 回答の正当性を判定
                print(f"⚖️ 回答の正当性を判定中...")
                is_correct = judge_answer_by_llm(
                    generated_answer=generated_answer,
                    correct_answer=item['Correct Answer'],
                    question=item['Question']
                )

                if is_correct:
                    total_judged_correct += 1
                    print(f"✅ 判定結果: 正解! データを保存します。")

                    # 正解の場合のみ結果を保存（フォールバック処理）
                    batch_results.append({
                        "id": item_id,
                        "question": item['Question'],
                        # 推論過程(Reasoning/CoT)を生成して、<think>...</think> タグで囲み、最終的な回答(answer)を </think> タグの後に記載する。
                        "output": f"<think>{content}</think>{item['Correct Answer']}",
                        "answer": item['Correct Answer'],
                        "generated_answer": generated_answer
                    })
                else:
                    print(f"❌ 判定結果: 不正解。このデータは保存されません。")
                    print(f"   生成回答: {generated_answer}")
                    print(f"   正解: {item['Correct Answer']}")
        else:
            total_api_failures += 1
            print(f"❌ CoT生成に失敗: {content}")

        # 5件ごとまたは最後の処理でCSVに保存（正解データのみ）
        if len(batch_results) >= 5 or (i + 1) == total_problems:
            if batch_results:  # 保存するデータがある場合のみ
                print(f"💾 {len(batch_results)} 件の正解データをCSVに保存中...")
                temp_df = pd.DataFrame(batch_results)
                temp_df.to_csv(
                    output_filename,
                    mode='a',
                    header=not os.path.exists(output_filename) or os.path.getsize(output_filename) == 0,
                    index=False,
                    encoding='utf-8-sig'
                )
                batch_results.clear()

        # 進捗統計を表示
        if (i + 1) % 10 == 0:
            success_rate = (total_judged_correct / total_processed * 100) if total_processed > 0 else 0
            print(f"📊 進捗統計 ({i + 1}/{total_problems}):")
            print(f"   処理済み: {total_processed}, 生成成功: {total_generated}, 判定正解: {total_judged_correct}")
            print(f"   正解率: {success_rate:.1f}%, API失敗: {total_api_failures}")

        # API レート制限を考慮して少し待機
        time.sleep(0.5)

    # 最終統計情報を表示
    final_success_rate = (total_judged_correct / total_processed * 100) if total_processed > 0 else 0
    generation_success_rate = (total_generated / total_processed * 100) if total_processed > 0 else 0

    print(f"\n✅ 全ての処理が完了しました。最終結果は '{output_filename}' に保存されています。")
    print(f"\n📊 最終統計情報:")
    print(f"   総処理数: {total_processed}")
    print(f"   CoT生成成功: {total_generated} ({generation_success_rate:.1f}%)")
    print(f"   判定正解数: {total_judged_correct} ({final_success_rate:.1f}%)")
    print(f"   API失敗数: {total_api_failures}")
    print(f"   保存データ数: {total_judged_correct} (正解データのみ)")

    # 品質レポート
    if total_generated > 0:
        judgment_accuracy = (total_judged_correct / total_generated * 100)
        print(f"   生成品質: {judgment_accuracy:.1f}% (生成成功データ中の正解率)")

    if total_judged_correct == 0:
        print("⚠️ 正解と判定されたデータがありません。データセットは作成されていません。")

    # --- 複数フォーマットでの保存とHugging Face アップロード ---
    try:
        print("\n🔄 最終データの処理とアップロードを開始...")
        final_df: pd.DataFrame = pd.read_csv(output_filename)

        # 統計情報の計算
        # 実際に保存されたデータ数（全て正解データ）
        success_count = len(final_df)
        total_count = total_processed  # 実際に処理した総数

        print(f"📊 最終データセット統計:")
        print(f"   保存されたレコード数: {success_count}")
        print(f"   処理対象総数: {total_count}")
        print(f"   データセット品質: 100% (正解データのみ)")

        # ベースファイル名（拡張子なし）を取得
        base_filename = os.path.splitext(output_filename)[0]

        # --- 2. Parquet形式での保存 ---
        parquet_filename = f"{base_filename}.parquet"
        print(f"💾 Parquet形式で保存中: {parquet_filename}")
        final_df.to_parquet(parquet_filename, index=False)

        # --- 3. JSONL形式での保存 ---
        jsonl_filename = f"{base_filename}.jsonl"
        print(f"💾 JSONL形式で保存中: {jsonl_filename}")
        final_df.to_json(jsonl_filename, orient='records', lines=True, force_ascii=False)

        # --- Hugging Face Datasetsに3つのファイル形式でアップロード ---
        print(f"📤 {OUTPUT_DATASET_ID} に3つのファイル形式でアップロード中...")

        # 1. CSV ファイルをアップロード
        print("  📄 CSV ファイルをアップロード中...")
        upload_file(
            path_or_fileobj=output_filename,
            path_in_repo=output_filename,
            repo_id=OUTPUT_DATASET_ID,
            repo_type="dataset"
        )

        # 2. Parquet ファイルをアップロード
        print("  📊 Parquet ファイルをアップロード中...")
        upload_file(
            path_or_fileobj=parquet_filename,
            path_in_repo=parquet_filename,
            repo_id=OUTPUT_DATASET_ID,
            repo_type="dataset"
        )

        # 3. JSONL ファイルをアップロード
        print("  📝 JSONL ファイルをアップロード中...")
        upload_file(
            path_or_fileobj=jsonl_filename,
            path_in_repo=jsonl_filename,
            repo_id=OUTPUT_DATASET_ID,
            repo_type="dataset"
        )

        print(f"✅ 3つのファイル形式でのアップロード完了: {OUTPUT_DATASET_ID}")

        # --- README.md の生成とアップロード ---
        print("📝 README.md を生成中...")
        readme_content = generate_readme_content(final_df, output_filename, parquet_filename, jsonl_filename, total_count, success_count)
        readme_filename = "README.md"

        with open(readme_filename, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        print("📤 README.md をアップロード中...")
        upload_file(
            path_or_fileobj=readme_filename,
            path_in_repo=readme_filename,
            repo_id=OUTPUT_DATASET_ID,
            repo_type="dataset"
        )
        print("✅ README.md のアップロード完了")

        # --- 統計情報の表示 ---
        print(f"\n📊 処理統計:")
        print(f"  - 総処理数: {total_count}")
        print(f"  - 成功数: {success_count}")
        print(f"  - 成功率: {success_count/total_count*100:.1f}%")

        print(f"\n📁 保存されたファイル:")
        print(f"  - CSV: {output_filename}")
        print(f"  - Parquet: {parquet_filename}")
        print(f"  - JSONL: {jsonl_filename}")
        print(f"  - README: {readme_filename}")

        print(f"\n🌐 Hugging Face Datasetsアップロード先:")
        print(f"  - Repository: {OUTPUT_DATASET_ID}")
        print(f"    ├─ {output_filename}")
        print(f"    ├─ {parquet_filename}")
        print(f"    ├─ {jsonl_filename}")
        print(f"    └─ {readme_filename}")

    except Exception as e:
        print(f"❌ データ処理またはHugging Face Hubへのアップロードに失敗: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Processing complete ---")

if __name__ == "__main__":
    main()
