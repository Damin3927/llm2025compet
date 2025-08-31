# Hugging Faceデータセットアップロードスクリプト

このスクリプトは、thinkタグ付きのDNA_DPO_hh-rlhfデータセットをHugging Faceにアップロードするために使用されます。

## 概要

`upload_to_hf.py`は、以下の機能を提供します：

- 複数のthinkタグ付きJSONLファイルのマージ
- データの検証（必須フィールド、thinkタグの存在確認）
- Hugging Faceデータセットへのアップロード
- ドライラン機能（実際のアップロード前の検証）

## 前提条件

### 1. 必要なライブラリのインストール

```bash
pip install -r requirements.txt
```

または個別にインストール：

```bash
pip install datasets huggingface-hub python-dotenv
```

### 2. Hugging Face APIキーの設定

`.env`ファイルに以下を追加：

```bash
HUGGINGFACE_API_KEY="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"
```

または環境変数として設定：

```bash
export HUGGINGFACE_API_KEY="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"
```

### 3. Hugging Faceへのアクセス権限

データセット `neko-llm/dna_dpo_hh-rlhf` への書き込み権限が必要です。

## 使用方法

### 基本的な使用方法

```bash
# すべてのthink_taggedファイルをアップロード
python upload_to_hf.py --files "think_tagged_*.jsonl"

# 特定のファイルを指定してアップロード
python upload_to_hf.py --files "think_tagged_0-9999.jsonl" "think_tagged_10000-19999.jsonl"

# カスタムコミットメッセージを指定
python upload_to_hf.py --files "think_tagged_*.jsonl" --commit-message "Add think tags for DPO training"
```

### オプション

- `--files`: アップロードするファイルパターン（必須、glob対応）
- `--dataset-name`: アップロード先のデータセット名（デフォルト: neko-llm/dna_dpo_hh-rlhf）
- `--commit-message`: コミットメッセージ（省略時は自動生成）
- `--api-key`: Hugging Face APIキー（環境変数より優先）
- `--log-level`: ログレベル（DEBUG, INFO, WARNING, ERROR）
- `--dry-run`: 実際のアップロードは行わず、検証のみ実行

### ドライラン（検証のみ）

アップロード前にデータの検証を行いたい場合：

```bash
python upload_to_hf.py --files "think_tagged_*.jsonl" --dry-run
```

これにより以下が実行されます：
- ファイルの読み込みとマージ
- データの検証
- 統計情報の表示
- エラーの確認

## データ形式

アップロードするデータは以下の形式である必要があります：

```json
{
  "id": 0,
  "question": "質問文",
  "preferred_output": "<think>思考プロセス</think>好ましい応答",
  "non_preferred_output": "好ましくない応答"
}
```

### 必須フィールド

- `question`: 質問文
- `preferred_output`: 好ましい応答（thinkタグ必須）
- `non_preferred_output`: 好ましくない応答

### thinkタグの要件

- `preferred_output`には必ず`<think>...</think>`タグが含まれている必要があります
- thinkタグの内容は20文字以上である必要があります

## 処理フロー

1. **ファイル読み込み**: 指定されたファイルパターンにマッチするファイルを読み込み
2. **データマージ**: 複数ファイルのデータをマージし、IDでソート
3. **データ検証**: 必須フィールドとthinkタグの存在確認
4. **データセット作成**: Hugging Faceデータセット形式に変換
5. **アップロード**: Hugging Face Hubにプッシュ

## エラーハンドリング

スクリプトは以下のエラーを適切に処理します：

- ファイルが見つからない場合
- JSON解析エラー
- データ検証エラー
- Hugging Face APIエラー
- ネットワークエラー

## ログ出力

スクリプトは詳細なログを出力します：

- ファイル読み込み状況
- データ検証結果
- 統計情報
- エラー詳細

## 注意事項

1. **大量データ**: 49,000件以上のデータをアップロードする場合、時間がかかる可能性があります
2. **API制限**: Hugging Face APIの制限に注意してください
3. **バックアップ**: アップロード前にデータのバックアップを取ることを推奨します
4. **権限確認**: データセットへの書き込み権限があることを確認してください

## トラブルシューティング

### よくある問題

1. **APIキーエラー**
   ```
   エラー: Hugging Face APIキーが設定されていません
   ```
   → `.env`ファイルまたは環境変数でAPIキーを設定してください

2. **ライブラリエラー**
   ```
   エラー: 必要なライブラリがインストールされていません
   ```
   → `pip install -r requirements.txt`でインストールしてください

3. **権限エラー**
   ```
   エラー: データセットへの書き込み権限がありません
   ```
   → Hugging Faceでデータセットへの書き込み権限を確認してください

4. **ファイルが見つからない**
   ```
   アップロード対象のファイルが見つかりません
   ```
   → ファイルパスとパターンを確認してください

## サンプル実行

### 1. 依存関係のインストール

```bash
cd data/dna
pip install -r requirements.txt
```

### 2. ドライランで検証

```bash
python upload_to_hf.py --files "think_tagged_*.jsonl" --dry-run
```

### 3. 実際のアップロード

```bash
python upload_to_hf.py --files "think_tagged_*.jsonl"
```

### 4. 特定のファイル範囲のみアップロード

```bash
python upload_to_hf.py --files "think_tagged_0-9999.jsonl" "think_tagged_10000-19999.jsonl"
```

## 成功時の出力

アップロードが成功すると、以下のような出力が表示されます：

```
✅ アップロード完了!
データセット: https://huggingface.co/datasets/neko-llm/dna_dpo_hh-rlhf
```

## 関連リンク

- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Hugging Face Hub](https://huggingface.co/docs/hub/)
- [DNA_DPO_hh-rlhf データセット](https://huggingface.co/datasets/neko-llm/dna_dpo_hh-rlhf)
