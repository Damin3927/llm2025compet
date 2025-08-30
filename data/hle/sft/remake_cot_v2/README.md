# Hugging Face データセット処理パイプライン

このスクリプトは、CoT評価と再生成機能を備えた複数のHugging Faceデータセットを処理するための包括的なパイプラインを提供します。

## 機能

- **バッチ処理**: 異なる設定で複数のデータセットを処理
- **整理された出力**: 結果を整理されたフォルダ構造（`dataset_name/config/`）で保存
- **非同期処理**: より良いパフォーマンスのための並列モデル実行
- **包括的評価**: 品質評価にCoTEvaluatorを使用
- **スマート再生成**: グレード閾値に基づいて低品質なCoTのみを再生成
- **詳細統計**: 包括的なレポートと分析
- **柔軟な設定**: 処理オプション付きのJSON設定

## クイックスタート

### 1. サンプル設定ファイルの作成

```bash
python process_hf_datasets.py --create-example-config
```

これにより、サンプル設定を含む `datasets_config.json` が作成されます。

### 2. 限定データでのテスト

```bash
python process_hf_datasets.py --config datasets_config.json --max-items 10 --output-dir ./test_output
```

### 3. フル処理

```bash
python process_hf_datasets.py --config datasets_config.json --output-dir ./processed_datasets
```

## 設定ファイル形式

### 基本構造

```json
{
  "datasets": [
    {
      "dataset_name": "gsm8k",
      "configs": ["main", "socratic"],
      "split": "train",
      "question_field": "question",
      "answer_field": "answer",
      "output_field": "output",
      "description": "小学校の算数問題"
    }
  ],
  "processing_options": {
    "max_items_per_dataset": 1000,
    "grade_threshold": "C",
    "evaluator_models": ["deepseek/deepseek-r1-0528:free"],
    "regenerator_models": [
      "deepseek/deepseek-r1-0528:free",
      "meta-llama/llama-3-8b-instruct:free"
    ],
    "use_async_regeneration": true
  }
}
```

### データセット設定フィールド

- **`dataset_name`**: Hugging Face データセット識別子
- **`configs`**: 処理するデータセット設定のリスト
- **`split`**: データセット分割（train、test、validation）
- **`question_field`**: 質問/問題を含むフィールド名
- **`answer_field`**: 期待される答えを含むフィールド名
- **`output_field`**: CoT推論を含むフィールド名
- **`description`**: ドキュメント用のオプション説明

### 処理オプション

- **`max_items_per_dataset`**: データセットあたりのアイテム制限（テスト用）
- **`grade_threshold`**: このグレード以下のアイテムを再生成（A/B/C/D）
- **`evaluator_models`**: 評価用モデル
- **`regenerator_models`**: 再生成用モデル
- **`use_async_regeneration`**: 非同期マルチモデル再生成を有効化

## コマンドラインオプション

```bash
python process_hf_datasets.py [OPTIONS]

オプション:
  --config PATH                 JSON設定ファイルへのパス（必須）
  --output-dir PATH            出力ディレクトリ [デフォルト: ./processed_datasets]
  --evaluator-models MODELS    評価モデルのカンマ区切りリスト
  --regenerator-models MODELS  再生成モデルのカンマ区切りリスト
  --grade-threshold GRADE      再生成閾値 [A|B|C|D、デフォルト: C]
  --max-items N                データセットあたりの最大アイテム数（テスト用）
  --no-async                   非同期マルチモデル再生成を無効化
  --create-example-config      サンプル設定を作成して終了
```

## 出力構造

```
processed_datasets/
├── processing_statistics.json     # 詳細統計
├── summary_report.txt             # 人間が読める形式のレポート
├── dataset1/
│   ├── config1/
│   │   ├── processed_20240101_120000.json
│   │   └── processed_20240101_120000.jsonl
│   └── config2/
│       ├── processed_20240101_130000.json
│       └── processed_20240101_130000.jsonl
└── dataset2/
    └── default/
        ├── processed_20240101_140000.json
        └── processed_20240101_140000.jsonl
```

## 出力データ形式

処理された各アイテムには以下が含まれます：

```json
{
  "id": "0",
  "question": "2 + 2は何ですか？",
  "answer": "4",
  "output": "<think>ここに改善された推論...</think>4",
  "metadata": {
    "original_data": { ... },
    "cot_history": [
      {
        "timestamp": "2024-01-01T12:00:00",
        "output": "<think>元の推論</think>4",
        "evaluation": {
          "grade": "C",
          "strengths": ["正しい答え"],
          "weaknesses": ["短すぎる"],
          "improvement_suggestions": ["説明を追加"],
          "learning_value_scores": { ... }
        }
      },
      {
        "timestamp": "2024-01-01T12:01:00",
        "output": "<think>ここに改善された推論...</think>4",
        "evaluation": null,
        "regeneration_metadata": {
          "model": "deepseek/deepseek-r1-0528:free",
          "predicted_grade": "B",
          "predicted_score": 7.2
        }
      }
    ]
  }
}
```

## 設定例

### 数学データセット

```json
{
  "datasets": [
    {
      "dataset_name": "gsm8k",
      "configs": ["main", "socratic"],
      "split": "train",
      "question_field": "question",
      "answer_field": "answer",
      "output_field": "output"
    },
    {
      "dataset_name": "math_qa",
      "configs": ["default"],
      "split": "train", 
      "question_field": "Problem",
      "answer_field": "correct",
      "output_field": "Rationale"
    },
    {
      "dataset_name": "aqua_rat",
      "configs": ["raw"],
      "split": "train",
      "question_field": "question",
      "answer_field": "correct", 
      "output_field": "rationale"
    }
  ]
}
```

### 科学データセット

```json
{
  "datasets": [
    {
      "dataset_name": "sciq",
      "configs": ["default"],
      "split": "train",
      "question_field": "question",
      "answer_field": "correct_answer",
      "output_field": "explanation"
    },
    {
      "dataset_name": "arc",
      "configs": ["ARC-Challenge", "ARC-Easy"],
      "split": "train",
      "question_field": "question",
      "answer_field": "answerKey",
      "output_field": "explanation"
    }
  ]
}
```

## 処理ワークフロー

1. **設定の読み込み**: JSON設定ファイルを解析
2. **データセット読み込み**: 各データセット/設定の組み合わせを読み込み
3. **アイテム準備**: 評価用にアイテムをフォーマット
4. **評価**: グレード付きでCoT品質を評価
5. **再生成**: 低品質なアイテムを改善（グレード ≤ 閾値の場合）
6. **最良選択**: 最良の再生成版を選択（マルチモデル）
7. **結果保存**: メタデータ付きの整理されたフォルダ構造
8. **統計**: 包括的なレポートと分析

## パフォーマンスのヒント

- **小さく始める**: 初期テストには `--max-items 100` を使用
- **並列処理**: より良い結果のために複数の再生成モデルを使用
- **グレード閾値**: 品質要件に基づいて調整
- **非同期モード**: マルチモデル再生成の効率のために有効化
- **リソース管理**: API使用量とレート制限を監視

## エラーハンドリング

スクリプトには堅牢なエラーハンドリングが含まれています：
- **個別アイテムの失敗**: 全体の処理を停止しない
- **API失敗**: 指数バックオフによるリトライロジック
- **フィールドの欠落**: 警告による適切な処理
- **データセット読み込み**: 一つが失敗しても他のデータセットを続行

## 進行状況の監視

- **プログレスバー**: 各データセットのリアルタイム進行状況
- **ログ記録**: `dataset_processing.log` に詳細ログを保存
- **統計**: 処理統計のライブ更新
- **エラー追跡**: アイテムIDとともにすべてのエラーをログ記録

## 使用例

### 基本的な使用法

```bash
# サンプル設定を作成
python process_hf_datasets.py --create-example-config

# 小さなデータセットでテスト
python process_hf_datasets.py --config datasets_config.json --max-items 50

# フル処理
python process_hf_datasets.py --config datasets_config.json
```

### 高度な使用法

```bash
# マルチモデル再生成
python process_hf_datasets.py \
  --config datasets_config.json \
  --regenerator-models "deepseek/deepseek-r1-0528:free,meta-llama/llama-3-8b-instruct:free" \
  --grade-threshold B

# 高品質フィルタリング
python process_hf_datasets.py \
  --config datasets_config.json \
  --grade-threshold A \
  --evaluator-models "deepseek/deepseek-r1-0528:free,gpt-4"
```

## 必要条件

- Python 3.8+
- Hugging Face データセット用の `datasets` ライブラリ
- API相互作用用の `openai`
- プログレスバー用の `tqdm`
- 環境変数用の `python-dotenv`
- 環境でのAPIキー: `OPENROUTER_API_KEY`

## インストール

```bash
pip install datasets openai tqdm python-dotenv
```

## トラブルシューティング

### よくある問題

1. **APIキーが設定されていない**
   ```bash
   export OPENROUTER_API_KEY="your_key_here"
   ```

2. **データセットが見つからない**
   - Hugging Face でデータセット名と設定を確認
   - データセットが認証を必要とするかチェック

3. **フィールドが見つからない**
   - まずデータセット構造を調査
   - 利用可能なフィールドを確認するため `dataset.column_names` を使用

4. **メモリ不足**
   - 処理を制限するため `--max-items` を使用
   - データセットを個別に処理

5. **APIレート制限**
   - 同時実行数を削減
   - リクエスト間に遅延を追加

### デバッグモード

詳細ログを有効化：
```bash
export LOG_LEVEL=DEBUG
python process_hf_datasets.py --config datasets_config.json
```

この包括的なスクリプトは、CoT評価と再生成機能を備えたHugging Faceデータセットの処理のための完全なソリューションを提供します。