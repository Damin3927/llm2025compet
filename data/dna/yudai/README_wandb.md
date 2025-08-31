# 有害性分類スクリプト - wandb統合版

## 概要

このスクリプトは、neko-llm/DNA_DPO_hh-rlhfデータセットの質問に対して3段階の有害性分類を行い、wandbを使って結果を可視化・分析できるようにしたものです。

## 新機能

### wandbによる可視化
- **リアルタイムメトリクス追跡**: 処理件数、成功率、エラー率をリアルタイムで記録
- **分布分析**: リスク領域、害の種類の分布を円グラフ・棒グラフで可視化
- **クロス集計**: リスク領域と害の種類の関係をヒートマップで表示
- **成功率推移**: 処理の進行に伴う成功率の変化を時系列グラフで表示
- **結果テーブル**: 分類結果と統計サマリーをテーブル形式で記録

## セットアップ

### 1. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 2. wandbの設定
```bash
# wandbにログイン
wandb login

# プロジェクトの作成（初回のみ）
wandb init
```

### 3. 環境変数の設定
```bash
export OPENROUTER_API_KEY="your_api_key_here"
```

## 使用方法

### 基本的な実行（wandb有効）
```bash
python classify_script.py --start_index 0 --end_index 100
```

### カスタムプロジェクト名
```bash
python classify_script.py --start_index 0 --end_index 100 --wandb_project "my-harm-analysis"
```

### カスタム実行名
```bash
python classify_script.py --start_index 0 --end_index 100 --wandb_run_name "experiment_001"
```

### wandbを無効化
```bash
python classify_script.py --start_index 0 --end_index 100 --no_wandb
```

## wandbで確認できる内容

### 1. メトリクス
- **total_processed**: 処理済み件数
- **success_rate**: 分類成功率
- **error_rate**: 分類エラー率
- **risk_area/{カテゴリ}**: 各リスク領域の件数
- **type_of_harm/{カテゴリ}**: 各害の種類の件数

### 2. 可視化
- **risk_area_distribution**: リスク領域の分布（円グラフ）
- **type_of_harm_distribution**: 害の種類の分布（棒グラフ）
- **cross_tabulation**: リスク領域と害の種類のクロス集計（ヒートマップ）
- **success_rate_trend**: 成功率の推移（時系列グラフ）

### 3. テーブル
- **classification_results**: 分類結果のサンプル（最初の100件）
- **summary_stats**: 統計サマリー

## 分析例

### 問題の割合の確認
- wandbのダッシュボードで`success_rate`と`error_rate`を確認
- エラー率が高い場合は、モデルの性能やプロンプトの改善が必要

### 偏りの確認
- `risk_area_distribution`でリスク領域の分布を確認
- `type_of_harm_distribution`で害の種類の分布を確認
- 特定のカテゴリに偏りがある場合は、データセットの特性や分類器の偏りを分析

### 時系列分析
- `success_rate_trend`で処理の進行に伴う性能変化を確認
- 特定の範囲でエラーが集中している場合は、データの特性を調査

## カスタマイズ

### 可視化の追加
`WandbVisualizer.create_visualizations()`メソッドに新しい可視化を追加できます。

### メトリクスの追加
`WandbVisualizer.log_metrics()`メソッドに新しいメトリクスを追加できます。

### ログ頻度の調整
現在は50件ごとにメトリクスを記録していますが、この値は変更可能です。

## 注意事項

- wandbの使用にはインターネット接続が必要です
- 大量のデータを処理する場合は、wandbのストレージ制限に注意してください
- 可視化の生成にはmatplotlibとseabornが必要です
- 処理中にwandbの接続が切れた場合、ローカルファイルには結果が保存されます
