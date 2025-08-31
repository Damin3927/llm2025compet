# DNA DPO データセット処理ツール

このディレクトリには、DNA（Do Not Answer）DPOデータセットの処理・生成・分類・アップロードを行うPythonスクリプトが含まれています。

## スクリプト一覧

### 1. `classify_script.py` - 有害性分類スクリプト
- **目的**: データセットの3段階有害性分類（リスク領域 → 害の種類 → 具体的な害）
- **入力**: `neko-llm/DNA_DPO_hh-rlhf` データセット
- **出力**: 分類結果（JSONL形式）
- **特徴**: wandbによる可視化・統計機能付き

```bash
python classify_script.py --start_index 0 --end_index 100 --output_file classified_0-99.jsonl
```

### 2. `add_think_tags.py` - Thinkタグ埋め込みスクリプト
- **目的**: DPOデータセットに`<think>`タグを自動生成・埋め込み
- **入力**: 既存データセット
- **出力**: Thinkタグ付きデータ（JSONL形式）
- **特徴**: OpenRouter API使用、リトライ機能、wandb連携

```bash
python add_think_tags.py --start_index 0 --end_index 50 --output_file think_tagged_0-49.jsonl
```

### 3. `dna_dpo_v2_generator.py` - DPO V2データセット生成スクリプト
- **目的**: 特定の欠陥カテゴリを修正する高品質DPOデータセット生成
- **入力**: 既存データセット
- **出力**: 改善されたDPOデータセット（JSONL形式）
- **特徴**: 並列処理対応、llama-3.1-70b-instruct使用

```bash
python dna_dpo_v2_generator.py --test  # テストモード
python dna_dpo_v2_generator.py         # 本格実行
```

### 4. `upload_to_hf.py` - Hugging Faceアップロードスクリプト
- **目的**: Thinkタグ付きデータセットをHugging Faceにアップロード
- **入力**: Thinkタグ付きJSONLファイル
- **出力**: Hugging Faceデータセット
- **特徴**: データ検証、統計情報、ドライラン機能

```bash
python upload_to_hf.py --files "think_tagged_*.jsonl" --dataset-name "your-dataset-name"
```

## 共通設定

### 環境変数
`.env`ファイルに以下を設定してください：

```bash
OPENROUTER_API_KEY=your_api_key_here
HUGGINGFACE_API_KEY=your_hf_api_key_here
```

### 依存関係
```bash
pip install -r requirements.txt
```

## 処理フロー

1. **分類**: `classify_script.py`でデータセットを有害性分類
2. **Thinkタグ追加**: `add_think_tags.py`で推論プロセスを埋め込み
3. **品質向上**: `dna_dpo_v2_generator.py`で特定カテゴリを改善
4. **アップロード**: `upload_to_hf.py`でHugging Faceに公開

## 注意事項

- 大量のデータ処理時はwandbによる進捗監視を推奨
- API制限に注意（OpenRouter、Hugging Face）
- 処理前のデータバックアップを推奨
- ログファイルは`logs/`ディレクトリに保存されます
