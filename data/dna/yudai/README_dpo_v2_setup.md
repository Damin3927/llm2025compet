# DPOデータセットV2生成スクリプト セットアップ手順

## 前提条件

- Python 3.8以上
- 必要なPythonパッケージがインストール済み（requirements.txt参照）
- OpenRouter APIキー

## セットアップ手順

### 1. 環境変数の設定

`.env`ファイルを作成し、OpenRouter APIキーを設定してください：

```bash
# data/dna/.env
OPENROUTER_API_KEY="your_actual_openrouter_api_key_here"
```

### 2. 依存関係のインストール

```bash
cd data/dna
pip install -r requirements.txt
```

### 3. ログディレクトリの作成

```bash
mkdir -p data/dna/logs
```

## 使用方法

### 基本的な実行

```bash
cd data/dna
python dna_dpo_v2_generator.py
```

### 並列処理の制御

```bash
# デフォルト（並列処理、自動ワーカー数決定）
python dna_dpo_v2_generator.py

# シーケンシャル処理（並列処理を無効化）
python dna_dpo_v2_generator.py --sequential

# ワーカー数を指定（最大8）
python dna_dpo_v2_generator.py --workers 4

# テストモード（小規模データで動作確認）
python dna_dpo_v2_generator.py --test

# 組み合わせ例
python dna_dpo_v2_generator.py --workers 6 --test
```

### 実行時の動作

1. **データセットロード**: Hugging Faceから`neko-llm/dna_dpo_hh-rlhf`をロード
2. **フィルタリング**: 指定された欠陥カテゴリのデータのみを抽出
3. **並列処理**: 複数ワーカーでAPI呼び出しを並列実行
4. **進捗監視**: 50件ごとの中間保存とサンプルデータ表示
5. **データ保存**: `data/dna/processed/dataset_v2_mvp.jsonl`に結果を保存

### 処理対象カテゴリ

#### 欠陥A（HLE評価リスク）
- `Causing material harm by disseminating misinformation e.g. in medicine or law`

#### 欠陥B（リグレッション・リスク）
- `Compromise privacy by leaking or inferring private information (person/individual)`
- `Nudging or advising users to perform unethical or unsafe actions`

## 出力ファイル

### リアルタイム出力（各回答生成ごと）
- **リアルタイム累積ファイル**: `data/dna/processed/realtime_results.jsonl`
  - 全ての結果を順次追加（追記形式）
  - 処理の進行状況をリアルタイムで確認可能
  - ファイル数が増えない（追記のみ）

### 中間保存ファイル
- **中間結果ファイル**: `data/dna/processed/intermediate_results_50.jsonl` など
  - 50件ごとにバッチ保存
  - 処理の途中経過を確認可能

### 最終出力ファイル
- **メインファイル**: `data/dna/processed/dataset_v2_mvp.jsonl`
  - 全ての処理完了後の最終結果

### その他
- **ログファイル**: `data/dna/logs/dpo_v2_generation.log`
- **W&B連携**: プロジェクト「dna-dpo-v2-generation」に結果を記録

## エラーハンドリング

- API呼び出し失敗時は最大3回のリトライ
- 失敗した行はスキップして処理を継続
- 詳細なログ出力で問題の特定が可能

## 並列処理の利点

- **処理速度の大幅向上**: 複数ワーカーでAPI呼び出しを並列実行
- **リソース効率**: CPUコアを最大限活用
- **進捗監視**: 50件ごとの中間保存で処理状況を確認
- **柔軟な制御**: ワーカー数や処理モードを自由に設定

## ワーカー数指定 vs シーケンシャル処理の違い

### ワーカー数指定（並列処理）
```bash
python dna_dpo_v2_generator.py --workers 4
```
- **動作**: 4つのスレッドで同時にAPI呼び出しを実行
- **利点**: 
  - 処理速度が大幅に向上（約4倍）
  - 複数のAPIリクエストを並行処理
  - ネットワーク待機時間を効率的に活用
- **使用場面**: 大量データの処理、高速化が必要な場合

### シーケンシャル処理
```bash
python dna_dpo_v2_generator.py --sequential
```
- **動作**: 1つのスレッドで順次処理
- **利点**: 
  - メモリ使用量が少ない
  - デバッグが容易
  - 安定性が高い
- **使用場面**: 
  - 小規模データの処理
  - デバッグやテスト
  - リソースが限られている環境

### 処理速度の比較例
- **シーケンシャル**: 1,000件 × 3秒 = 50分
- **4ワーカー並列**: 1,000件 × 3秒 ÷ 4 = 12.5分
- **8ワーカー並列**: 1,000件 × 3秒 ÷ 8 = 6.25分

## 監視方法

### 1. リアルタイムログ監視

```bash
# 別ターミナルでログファイルを監視
cd /home/argo/llm2025compet/data/dna
tail -f logs/dpo_v2_generation.log
```

### 2. プロセス状況の監視

```bash
# プロセス一覧を確認
ps aux | grep "dna_dpo_v2_generator"

# リアルタイムでプロセス状況を監視（5秒間隔）
watch -n 5 'ps aux | grep "dna_dpo_v2_generator"'

# プロセスの詳細情報を確認
top -p $(pgrep -f "dna_dpo_v2_generator")
```

### 3. リアルタイム結果ファイルの監視

```bash
# リアルタイム結果ファイルの監視
cd /home/argo/llm2025compet/data/dna/processed
tail -f realtime_results.jsonl

# ファイルサイズと行数の変化を監視
watch -n 5 'echo "ファイルサイズ: $(ls -lh realtime_results.jsonl 2>/dev/null | awk "{print \$5}" || echo "未作成"), 行数: $(wc -l < realtime_results.jsonl 2>/dev/null || echo 0)"'

# 最新の結果を確認
tail -5 realtime_results.jsonl
```

### 4. 中間ファイルの監視

```bash
# 中間保存ファイルの確認
cd /home/argo/llm2025compet/data/dna/processed
ls -la intermediate_*.jsonl

# 最新の中間ファイルの内容を確認
ls -t intermediate_*.jsonl | head -1 | xargs tail -20

# ファイルサイズの変化を監視
watch -n 10 'ls -lh intermediate_*.jsonl 2>/dev/null || echo "まだ中間ファイルが生成されていません"'
```

## 注意事項

- OpenRouter APIキーが正しく設定されていることを確認
- 大量のデータを処理するため、十分な時間を確保
- ネットワーク接続が安定していることを確認
- 並列処理時は適切なワーカー数を設定（推奨: CPUコア数の半分〜同数）
- メモリ使用量が増加するため、十分なRAMを確保
