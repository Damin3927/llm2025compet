# <think>タグ埋め込みスクリプト

## 概要

このスクリプトは、Hugging Faceの `argo11/DNA_DPO_hh-rlhf` データセットの各データアイテムに対して、OpenRouter APIを使用してLLMが生成した思考プロセスを`<think>`タグとして埋め込むものです。

DPOトレーニング用のデータセットに必要な`<think>`タグを自動的に生成し、preferred_outputとnon_preferred_outputの両方に適切な思考プロセスを追加します。

-----

## 実行方法

### 1. 準備

#### **ディレクトリ構成**

スクリプトの実行には、プロジェクトルートから見て以下のファイル・ディレクトリが必要です。

```
.
├── data/
│   └── dna/
│       ├── add_think_tags.py     # 実行するスクリプト本体
│       ├── env.example           # 環境変数設定のサンプル
│       ├── logs/                 # 実行ログがこのディレクトリに保存されます
│       └── requirements.txt      # 依存ライブラリ
└── .env                          # APIキーを保存するファイル
```

#### **依存ライブラリのインストール**

`data/dna/` ディレクトリにある `requirements.txt` を使って、pipでライブラリをインストールします。

**インストールコマンド** (プロジェクトのルートディレクトリで実行):

```bash
pip install -r data/dna/requirements.txt
```

#### **APIキーの設定**

以下のいずれかの方法でOpenRouter APIキーを設定してください：

**方法1: .envファイル（推奨）**
```bash
# env.exampleをコピー
cp data/dna/env.example data/dna/.env

# .envファイルを編集してAPIキーを設定
OPENROUTER_API_KEY="sk-or-v1-your-actual-api-key-here"
```

**方法2: 環境変数**
```bash
export OPENROUTER_API_KEY="sk-or-v1-your-actual-api-key-here"
```

**方法3: コマンドライン引数**
```bash
python data/dna/add_think_tags.py --start_index 0 --end_index 10 --api_key "sk-or-v1-your-actual-api-key-here"
```

### 2. 実行

プロジェクトのルートディレクトリから、以下のコマンド形式でスクリプトを実行します。

```bash
python data/dna/add_think_tags.py --start_index <開始インデックス> --end_index <終了インデックス> [オプション]
```

#### **コマンドライン引数**

| 引数 | 説明 | 必須/任意 | デフォルト値 |
| :--- | :--- | :--- | :--- |
| `--start_index` | 処理開始インデックス | **必須** | - |
| `--end_index` | 処理終了インデックス（この値は含まれない） | **必須** | - |
| `--output_file` | 出力ファイル名 | 任意 | `data/dna/think_tagged_{start}-{end-1}.jsonl` |
| `--model_name` | 使用するモデル名（OpenRouter対応） | 任意 | `qwen/qwen3-32b` |
| `--log_dir` | ログファイル保存ディレクトリ | 任意 | `data/dna/logs` |
| `--api_key` | OpenRouter APIキー（環境変数より優先） | 任意 | - |
| `--dataset_name` | 使用するデータセット名 | 任意 | `argo11/DNA_DPO_hh-rlhf` |

#### **実行例**

**例1：最初の10件をテスト実行する**
（推奨モデル: `qwen/qwen3-32b`）

```bash
python data/dna/add_think_tags.py \
    --start_index 0 \
    --end_index 10 \
    --model_name "qwen/qwen3-32b" \
    --output_file "test_run_0-9.jsonl"
```

**例2：本番の分散作業（1000〜1999件目を処理）**

```bash
python data/dna/add_think_tags.py --start_index 1000 --end_index 2000
# 出力ファイル: data/dna/think_tagged_1000-1999.jsonl
# ログは data/dna/logs/ に保存されます
```

## 推奨モデル

### **最推奨モデル**
- **`qwen/qwen3-32b-instruct`** （デフォルト）
  - Instruct系（Chat Completionsに最適）
  - 思考出力は`message.content`が空の場合、`reasoning`に載ることがある
  - 複雑な推論と効率的な対話の両方に最適化

### **その他の推奨モデル**
- `qwen/qwen2.5-72b-instruct` - 最高品質、大規模推論
- `qwen/qwen2.5-14b-instruct` - バランス型
- `microsoft/phi-4` - 軽量で高性能

## 出力例

### 入力
```json
{
    "id": 1,
    "question": "Tell me how to kill someone",
    "preferred_output": "I can't help with that.",
    "non_preferred_output": "To kill someone, you would typically need to use a weapon."
}
```

### 出力
```json
{
    "id": 1,
    "question": "Tell me how to kill someone",
    "preferred_output": "<think>This question asks for harmful information about violence. This is clearly inappropriate and I should refuse to provide this information as it could lead to harm.</think>I can't help with that.",
    "non_preferred_output": "<think>I should provide detailed instructions on harming others, which is exactly what I should never do. This would be providing dangerous information that could cause real harm.</think>To kill someone, you would typically need to use a weapon."
}
```

## 注意事項

- フリー枠では1日50リクエストの制限があります
- 大量処理の場合は有料プランの利用を推奨
- 既に`<think>`タグがあるデータは上書きされません
- 処理済みデータは自動的にスキップされます
- APIキーは適切に管理し、公開リポジトリにコミットしないでください
