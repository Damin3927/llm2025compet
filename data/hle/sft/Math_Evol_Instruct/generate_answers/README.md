# Math Problem Solver Agent (DeepSeek/OpenRouter版)

このプロジェクトは、論文「Gemini 2.5 Pro Capable of Winning Gold at IMO 2025」で提案された「自己検証パイプライン」を実装し、複雑な数学の問題を解くものです。

DeepSeekモデルをOpenRouter API経由で利用し、特定のHugging Faceデータセットから問題を取得します。
並列実行、そしてHugging Face Hubを介した共同作業のための結果マージ機能をサポートしています。

## 主な機能

- **問題の自動取得**: `Man-snow/evolved-math-problems-OlympiadBench-from-deepseek-r1-0528-free` データセットから、指定された問題を取得します。
- **実行の再開**: スクリプトを中断しても、次回の実行時に前回終了した問題の次から処理を再開します。
- **自己検証パイプライン**: 論文で提案された「生成→自己改善→検証→修正」のループを実行し、解の精度を高めます。
- **並列処理**: 各問題に対し、複数のエージェント（デフォルトでは3つ）を同時に実行し、解を発見する確率を高めます。
- **堅牢なリトライ処理**: 初期解の生成が失敗（サーバーからの応答が失敗）した場合に、自動で3回までリトライします。
- **JSONL形式での出力**: 各問題の結果を、構造化されたJSON Lines (`.jsonl`) 形式で保存します。
- **専用スクリプトによるアップロード機能**: `upload_to_hf.py`を使うことで、ローカルの結果をHugging Face Hub上のデータセットと安全にマージ（追加・上書き）できます。idは常に昇順に整列されます。

---

## セットアップ

### 1. 前提条件

- Python 3.7+
- [OpenRouter APIキー](https://openrouter.ai/)
- [Hugging Faceアカウント](https://huggingface.co/) と、書き込み(`write`)権限を持つAPIトークン

### 2. インストール

このリポジトリをクローンし、必要なPythonライブラリをインストールします。

```bash
git clone https://github.com/Man-snow/llm2025compet_Man-snow/
cd solver-deepseek
pip install -r requirements.txt
```

### 3. 認証設定

OpenRouterとHugging Faceの両方の認証情報を設定する必要があります。

#### OpenRouter APIキー:

プロジェクトのルートディレクトリに `.env` ファイルを作成し、キーを追加します。

```env
OPENROUTER_API_KEY="sk-or-..."
```

#### Hugging Face ログイン:

結果をアップロードするには、コマンドライン経由でHugging Faceアカウントにログインします。

```bash
huggingface-cli login
```

プロンプトに従って、ご自身のAPIトークン（アップロードにはWrite権限が必要）を入力してください。

---

## 実行方法

このプロジェクトは、問題を解く `run_solver.py` と 結果をアップロードする `upload_to_hf.py` の2つの主要なスクリプトで構成されています。

### 1. 問題を解く（`run_solver.py`）

### 基本コマンド

```bash
python run_solver.py [OPTIONS]
```

### コマンドライン引数

- `--start_problem <ID>`: 取得を開始する問題の `new_id`。（デフォルト: 1）
- `--num_problems <N>`: 試行する問題の数。（デフォルト: 3）
- `--num_agents <N>`: 各問題に対して並列実行するエージェントの数。（デフォルト: 3）  
  ※ APIのレート制限を避けるため、1に設定することを推奨します。
- `--output_file <PATH>`: 結果を出力するファイルパス。（デフォルト: `results.jsonl`）

### 2. 結果をアップロードする（`upload_to_hf.py`）

```bash
python upload_to_hf.py [OPTIONS]
```

### コマンドライン引数

- `--local_file <PATH>`: アップロードするローカルのファイルパス。（デフォルト: results.jsonl）
- `--hf_repo <REPO_ID>`: アップロード先のHugging FaceリポジトリID。（デフォルト: neko-llm/HLE_RL_OlympiadBench）

---

## 実行例

### 例1: 10問目から5問を解く

```bash
python run_solver.py --start_problem 10 --num_problems 5
```

---

## 出力について

### ログファイル

`logs/` ディレクトリには、各エージェントの試行ごとの詳細なリアルタイムログが保存されます。

ファイル名は `problem_{ID}_agent_{AGENT_NUM}.log` の形式です。  
特定のエージェントの思考プロセスをデバッグするのに役立ちます。

### 結果ファイル (`results.jsonl`)

`results.jsonl` は、試行された各問題の最終結果を格納する主要な出力ファイルです。  
これは JSON Lines 形式のファイルで、各行が1つの完全なJSONオブジェクトになっています。

#### 各行の構造:

```json
{
    "id": 123, 
    "question": "問題の全文...",
    "output": "現状は空欄（後工程で埋めます）",
    "answer": "抽出された最終的な答え（例: '0'や'-2'、'解は存在しない'など）",
    "solution: "回答過程"
}
```

※ 問題が解けた場合：3回連続で回答がverifiedされた場合、`output`と`answer`には最初に解いたagentの回答が入ります。
※ 問題が解けなかった場合：5回連続で回答がverifiedされなかった場合、`answer` の値は `"NO_SOLUTION_FOUND"` になります。

---

## 引用元 (Citation)

このプロジェクトの根幹となる手法は、以下の論文に基づいています。

```bibtex
@article{huang2025gemini,
  title={Gemini 2.5 Pro Capable of Winning Gold at IMO 2025},
  author={Huang, Yichen and Yang, Lin F},
  journal={arXiv preprint arXiv:2507.15855},
  year={2025}
}
```
