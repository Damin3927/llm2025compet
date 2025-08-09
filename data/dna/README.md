# 有害性分類スクリプト

## 概要

このスクリプトは、Hugging Faceの `neko-llm/DNA_DPO_hh-rlhf` データセットの各質問項目に対して、大規模言語モデル（LLM）を用いて3段階の有害性カテゴリ分類を行うものです。

OpenRouter APIを経由して指定したLLMに問い合わせを行い、`data/dna/dna_hierarchy.json` に定義された分類基準に基づいて、各データに `risk_area`, `type_of_harm`, `specific_harm` の3つの情報を付与し、結果をJSONL形式で出力します。

コマンドライン引数による処理範囲の指定に対応しており、大規模なデータセットの分類作業を複数人で分担・並列実行することが可能です。

-----

## 実行方法

### 1\. 準備

#### **ディレクトリ構成**

スクリプトの実行には、プロジェクトルートから見て以下のファイル・ディレクトリが必要です。（このタスクに関係のないファイルは省略しています）

```
.
├── data/
│   └── dna/
│       ├── classify_script.py      # 実行するスクリプト本体
│       ├── dna_hierarchy.json    # 分類基準を定義したJSON
│       ├── logs/                 # 実行ログがこのディレクトリに保存されます
│       └── requirements.txt        # 依存ライブラリ
└── .env                          # APIキーを保存するファイル
```

#### **依存ライブラリのインストール**

`data/dna/` ディレクトリにある `requirements.txt` を使って、pipでライブラリをインストールします。

**インストールコマンド** (プロジェクトのルートディレクトリで実行):

```bash
pip install -r data/dna/requirements.txt
```

#### **APIキーの設定**

プロジェクトのルートに `.env` ファイルを作成し、OpenRouterのAPIキーを記述します。

**`.env`**:

```
OPENROUTER_API_KEY="ここにあなたのAPIキーを貼り付け"
```

### 2\. 実行

プロジェクトのルートディレクトリから、以下のコマンド形式でスクリプトを実行します。

```bash
python data/dna/classify_script.py --start_index <開始インデックス> --end_index <終了インデックス> [オプション]
```

#### **コマンドライン引数**

| 引数 | 説明 | 必須/任意 | デフォルト値 |
| :--- | :--- | :--- | :--- |
| `--start_index` | 処理を開始するデータセットのインデックス。 | **必須** | - |
| `--end_index` | 処理を終了するインデックス（この値は含まれない）。 | **必須** | - |
| `--output_file` | 出力ファイル名。指定しない場合は自動生成されます。 | 任意 | `classified_{start}-{end-1}.jsonl` |
| `--model_name` | 使用するOpenRouterのモデルID。 | 任意 | `meta-llama/llama-3-8b-instruct` |
| `--hierarchy_file` | 分類基準を定義したJSONファイルのパス。 | 任意 | `data/dna/dna_hierarchy.json` |
| `--log_dir` | ログファイルを保存するディレクトリ。 | 任意 | `data/dna/logs` |
| `--api_key` | OpenRouter APIキー。環境変数より優先されます。 | 任意 | - |

#### **実行例**

**例1：最初の10件をテスト実行する**
（推奨モデル: `meta-llama/llama-3.1-405b-instruct`）

```bash
python data/dna/classify_script.py \
    --start_index 0 \
    --end_index 10 \
    --model_name "meta-llama/llama-3.1-405b-instruct" \
    --output_file "test_run_0-9.jsonl"
```

**例2：本番の分燗作業（1000〜1999件目を処理）**

```bash
python data/dna/classify_script.py --start_index 1000 --end_index 2000
# 出力ファイル: classified_1000-1999.jsonl
# ログは data/dna/logs/ に保存されます
```