# SFT パイプライン

このフォルダには、教師ありファインチューニング（SFT）用に処理するためのスクリプトとツールが含まれています。パイプラインには、フィルタリング、処理、Hugging Faceへのデータセットアップロードが含まれます。

## 📁 ディレクトリ構造

```
data/hle/sft/
├── OpenMathReasoningFiltering.py          # LLMベースフィルタリングスクリプト
├── OpenMathReasoningFiltering_bylabel.py  # ラベルベースフィルタリング（高速）
├── generateFromSeed.py                    # シード問題からソリューション生成
├── upload_data.py                         # データセットをParquetに変換し、jsonとParquetをHFにアップロード
├── difficulty_scorer.py                   # 多指標難易度評価ツール
├── length_selector.py                     # 回答長ベース半ガウシアン分布データ選択ツール
├── run_filter.sh                          # LLMフィルタリング用SLURMスクリプト
├── run_label_filter.sh                    # ラベルフィルタリング用SLURMスクリプト
├── run_length_selector.sh                 # データ選択用SLURMスクリプト
├── universal_length_selector.sh           # ユニバーサルデータ選択スクリプト
├── keys.json.example                      # APIキーテンプレート
├── results/                               # 処理済みデータの出力ディレクトリ
```

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# 仮想環境の作成とアクティベート
python -m venv hfenv
source hfenv/bin/activate  # Linux/Mac
# または
hfenv\Scripts\activate     # Windows

# 依存関係のインストール
pip install torch transformers datasets huggingface-hub tqdm pandas pyarrow
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124
```

### 2. APIキーの設定

```bash
# サンプルファイルをコピー
cp keys.json.example keys.json

# トークンで編集
{
  "llm": "your_huggingface_token_here"
}
```

## 🔄 SFTデータ処理パイプライン

SFTパイプラインは以下の段階的なワークフローに従います：

### ステップ1: データ選択・フィルタリング
**目的**: 品質の高いシードデータを選択して、効果的な学習データセットを構築

**オプション A: 高速データ選択（推奨）**
```bash
# 回答長ベースでの高速選択
python length_selector.py \
    --input "dataset-name" \
    --total_samples 10000 \
    --output "selected_seeds.json"
```
- ✅ 高速処理（大規模データセット対応）
- ✅ 半ガウシアン分布で長い回答を優先選択
- ✅ ストリーミング処理対応

**オプション B: 高精度難易度ベース選択**
```bash
# 多指標難易度評価による精密選択
python difficulty_scorer.py \
    --input "dataset-name" \
    --output "difficulty_scores.json" \
    --max_samples 50000
```
- ✅ 複数LLMによる総合難易度評価
- ✅ 対数確率・正解率・IRT分析
- ⚠️ 処理時間が長い（小〜中規模データ向け）

**オプション C: ラベルベース高速フィルタリング**
```bash
# 事前計算ラベルによる超高速フィルタリング
python OpenMathReasoningFiltering_bylabel.py \
    --filter-by-pass-rate 0.1 \
    --save-per-iteration 10000
```
- ✅ 最高速（LLM推論不要）
- ✅ 大規模データセット対応
- ⚠️ 事前ラベルが必要

### ステップ2: データ拡張（オプション）
**目的**: 選択されたシードデータから新しい学習例を生成

```bash
# シード問題から新規ソリューション生成
python generateFromSeed.py \
    --model Qwen/Qwen3-32B \
    --input_file selected_seeds.json \
    --output_file expanded_solutions.json \
    --max_tokens 4096
```
- 📝 多様なソリューションパターン生成
- 🎯 学習データの多様性向上
- ⚠️ 大規模モデル推奨（品質確保のため）

### ステップ3: データセットアップロード
**目的**: 処理済みデータをHugging Face Hubにアップロードして共有  
**準備**: 違うsplitに入れる予定のデータを別々のフォルダに入れる

```bash
# JSON→Parquet変換 & HFアップロード
python upload_data.py \
    --dataset_path ./results/processed_dataset \
    --repo_id your-username/sft-dataset-name
```
- 🔄 JSON→Parquet自動変換
- 📤 データセット設定ファイル自動生成
- 🔒 プライベート/パブリックリポジトリ対応

### 推奨ワークフロー例

**小規模・高品質データセット（<10K件）**
```bash
# 1. 難易度ベース精密選択
python difficulty_scorer.py --input "dataset" --max_samples 5000
# 2. データ拡張
python generateFromSeed.py --input_file scored_data.json
# 3. アップロード
python upload_data.py --dataset_path ./results
```

**大規模・効率重視データセット（>100K件）**
```bash
# 1. 長さベース高速選択
python length_selector.py --input "dataset" --total_samples 50000
# 2. 直接アップロード（拡張スキップ）
python upload_data.py --dataset_path ./results
```

**超大規模・ラベル利用可能（>1M件）**
```bash
# 1. ラベルベース超高速フィルタリング
python OpenMathReasoningFiltering_bylabel.py --filter-by-pass-rate 0.05
# 2. アップロード
python upload_data.py --dataset_path ./results
```


## 📋 スクリプト使用ガイド

### 🏷️ ラベルベースフィルタリング（推奨）

**スクリプト:** `OpenMathReasoningFiltering_bylabel.py`  
**目的:** 事前計算されたラベルを使用した高速フィルタリング、LLM推論不要

```bash
# 基本使用法 - パス率閾値でフィルタリング
python OpenMathReasoningFiltering_bylabel.py \
    --filter-by-pass-rate 0.1 \
    --save-per-iteration 10000

# 大規模データセット用のカスタムバッチサイズ
python OpenMathReasoningFiltering_bylabel.py \
    --filter-by-pass-rate 0.05 \
    --save-per-iteration 50000 \
    --batch-size 5000
```

**主要パラメータ:**
- `--filter-by-pass-rate`: パス率閾値（0.0-1.0）。低いほど難しい問題
- `--save-per-iteration`: メモリ問題を避けるためN例ごとに結果を保存
- `--batch-size`: メモリ管理用の処理バッチサイズ

**出力:** `results/filtered_dataset/`内のフィルタリング済みJSONファイル

---

### 🤖 LLMベースフィルタリング（少量データの高精度フィルタリング）

**スクリプト:** `OpenMathReasoningFiltering.py`  
**目的:** カスタムフィルタリング基準のためのLLM推論使用   
**注意:**　推論速度がかなり遅い（Qwen3-32Bで５０s/件）から大量データ（＞５０００件）では使えない

```bash
# Qwenモデルでの基本LLMフィルタリング
python OpenMathReasoningFiltering.py \
    --inference-model Qwen/Qwen3-32B \
    --judgement-model Qwen/Qwen3-32B \
    --inference-batch-size 8 \
    --judgement-batch-size 8 \
    --start-from-percentage 0.0 \
    --end-at-percentage 1.0

# テンソル並列化によるマルチGPUセットアップ
python OpenMathReasoningFiltering.py \
    --inference-model Qwen/Qwen3-32B \
    --inference-tp 2 \
    --inference-batch-size 4 \
    --vllm-batch-size 32 \
    --judgement-model Qwen/Qwen3-32B \
    --judgement-tp 2
```

**主要パラメータ:**
- `--inference-model`: ソリューション生成用モデル
- `--judgement-model`: ソリューション評価用モデル
- `--inference-tp`: テンソル並列化サイズ（2GPUの場合は2）
- `--vllm-batch-size`: vLLM処理用バッチサイズ（大きいほど効率的）
- `--start-from-percentage/--end-at-percentage`: データセットの一部を処理

**出力:** 指定ディレクトリ内の推論と判定結果

---

### 🌱 シード問題からの生成

**スクリプト:** `generateFromSeed.py`  
**目的:** シード問題から新しいソリューションを生成
**注意:** 
1. 生成の質とフォーマットを確保するため、Qwen3-32Bなど大規模モデルがおすすめ。小さいモデル（Qwen3-8B）でエラーが出る報告がある
2. 推論速度がかなり遅い（Qwen3-32Bで５０s/件）から大量データ生成（＞５０００件）では使えない

```bash
# Qwenモデルを使用したソリューション生成
python generateFromSeed.py \
    --model Qwen/Qwen3-32B \
    --input_file seed_problems.json \
    --output_file generated_solutions.json \
    --max_tokens 4096 \
    --temperature 0.3
```

**主要パラメータ:**
- `--model`: 生成に使用するモデル
- `--input_file`: シード問題を含むJSONファイル
- `--output_file`: 生成されたソリューションの保存先
- `--max_tokens`: 生成あたりの最大トークン数
- `--temperature`: サンプリング温度（0.0 = 決定論的、1.0 = ランダム）

---

### 🎯 難易度評価ツール

**スクリプト:** `difficulty_scorer.py`  
**目的:** 質問回答データの難易度を複数の指標で評価し、スコア化

**機能:**
- 複数の小型LLM（≤8B）による評価
- 3つの指標を使用:
  1. 金回答の平均対数確率
  2. アンサンブル正解率  
  3. IRT難易度パラメータ β
- ストリーミング処理でメモリ効率的
- GPU/CPU両対応、OOM対策済み
- プライベートデータセット対応

```bash
# 基本的な使用
python difficulty_scorer.py \
    --input "dataset-name" \
    --output "difficulty_scores.json" \
    --max_samples 10000

# HuggingFaceプライベートデータセット
python difficulty_scorer.py \
    --input "neko-llm/SFT_OpenMathReasoning" \
    --dataset_spec "cot" \
    --question_field "problem" \
    --answer_field "generated_solution" \
    --max_samples 50000 \
    --max_sequence_length 1024 \
    --use_float32 \
    --disable_flash_attention \
    --output "math_difficulty_scores.json"

# SLURMでの実行
sbatch run_difficulty_scorer.sh
```

**主要パラメータ:**
- `--primary_model`: ログ確率計算用メインモデル（デフォルト: microsoft/Phi-4-mini-reasoning）
- `--ensemble_models`: 追加評価モデルリスト
- `--max_sequence_length`: 入力シーケンス最大長（OOM対策）
- `--max_input_length`: 生成時入力最大長（OOM対策）
- `--disable_flash_attention`: Flash Attentionエラー対策

**出力形式:**
```json
[
  {
    "id": "item_1",
    "avg_logprob": -2.1,
    "ensemble_acc": 0.75,
    "irt_beta": 0.3,
    "difficulty_z": 1.2
  }
]
```

---

### 📊 データ選択ツール（長さベース）

**スクリプト:** `length_selector.py`  
**目的:** 快速にSFT用のデータをフィルタリング  
**機能:**
- 回答長に基づく半ガウシアン分布データ選択（最長回答最多、最短回答最少）
- 動的ビン作成（実データ分布に基づく）
- 半ガウシアン分布による重み付け
- オープンエンド方式（外れ値も含む）
- ストリーミング処理対応
- リザーバーサンプリング

```bash
# 基本的な使用
python length_selector.py \
    --input "dataset-name" \
    --total_samples 5000 \
    --output "selected_data.json"

# 詳細設定
python length_selector.py \
    --input "neko-llm/SFT_OpenMathReasoning" \
    --dataset_spec "cot" \
    --answer_field "generated_solution" \
    --total_samples 10000 \
    --num_bins 8 \
    --curve_sharpness 3.0 \
    --sample_size_for_stats 2000 \
    --shuffle \
    --output "math_selected.json"

# SLURMでの実行
sbatch run_length_selector.sh
# または
./universal_length_selector.sh "dataset-name" 5000
```

**主要パラメータ:**
- `--curve_sharpness`: 分布の鋭さ（1.0=緩やか、2.0=標準、4.0=鋭い）
- `--num_bins`: ビン数（デフォルト: 6）
- `--sample_size_for_stats`: ビン作成用サンプル数（デフォルト: 1000）
- `--shuffle`: データセットをランダム順で処理

**分布例（6ビン）:**
```
Bin 0 (shortest): 100 samples (10%)   ← 最少
Bin 1: 200 samples (20%)
Bin 2: 300 samples (30%)
Bin 3: 400 samples (40%)
Bin 4: 500 samples (50%)
Bin 5 (longest): 600 samples (60%)    ← 最多
```

---

### 📤 Hugging Faceへのアップロード

**スクリプト:** `upload_data.py`
**目的:** JSONファイルをParquetに変換し、JSONとParquet両方をHugging Face Hubにアップロード  
**説明:** Parquet形式が`datasets.load_dataset()`で直接使える。JSON/JSONLファイルが大規模トレーニングする時に使いやすい。  
**機能:**
- 各JSONファイルを個別のParquetファイルに変換（メモリ効率的）
- splitプレフィックス付きファイル名で区別可能
- 互換性のため元のJSONファイルを保持
- YAML メタデータで各splitの設定を自動生成
- プライベートデータセット対応
- 既存ファイルのスキップ機能


```bash
# 基本アップロード（データセットカード自動生成）
python upload_data.py \
    --dataset_path ./results/filtered_dataset \
    --repo_id your-username/dataset-name

# 既存のREADMEがある場合はoriginal_README.logから読み込み
# データセットカードは常に生成される
```

**期待される入力構造:**
```
dataset_path/
├── split_1(ネームは自由)/
│   ├── file1.json
│   ├── file2.json
│   └── ...
├── split_2/
│   ├── file1.json
│   └── ...
└── split_3/
│   └── file1.json
└── original_readme.log (任意, md形式でhuggingfaceで使いたいreadmeを中に、このファイルがないとreadmeを自動生成)
```

**出力構造:**
```
dataset_path/
├── train/ (元のJSONファイル)
├── validation/ (元のJSONファイル)  
├── test/ (元のJSONファイル)
├── data/
│   ├── train_file1.parquet    # splitプレフィックス付き
│   ├── train_file2.parquet    # splitプレフィックス付き
│   ├── validation_file1.parquet
│   └── test_file1.parquet
└── README.md (YAML メタデータ付き)
```

**自動生成するYAML メタデータ例:**
```yaml
---
configs:
- config_name: cot
  data_files:
    - "data/cot_file1.parquet"
    - "data/cot_file2.parquet"
- config_name: genselect
  data_files: "data/genselect_file1.parquet"
---
```

**データセットの読み込み方法:**
```python
from datasets import load_dataset

# 特定のsplitを読み込む
train_data = load_dataset("your-username/dataset-name", "cot", "split"="train")
val_data = load_dataset("your-username/dataset-name", "genselect", "split"="train")

# または手動でdata_filesを指定
dataset = load_dataset(
    "parquet",
    data_files={
        "train": "data/train_*.parquet",
        "validation": "data/validation_*.parquet",
    }
)

# 個別ファイルの読み込み
import pandas as pd
df = pd.read_parquet("data/train_file1.parquet")
```

---

## 🖥️ SLURMスクリプト

### HPC/クラスター使用用

**ラベルベースフィルタリング:**
```bash
sbatch run_label_filter.sh
```

**LLMベースフィルタリング:**
```bash
sbatch run_filter.sh
```

**難易度評価:**
```bash
sbatch run_difficulty_scorer.sh
```

**データ選択:**
```bash
sbatch run_length_selector.sh
# または
./universal_length_selector.sh "dataset-name" 5000
```

各スクリプトには以下が含まれます:
- GPUリソース割り当て
- 環境セットアップ
- CUDA設定
- メモリ最適化設定

## 📊 データスキーマ

処理されたデータセットは以下のスキーマに従います (OpenMathを例に):

```json
{
  "problem": "string",           // 数学問題文
  "answer": "string",           // 正解
  "generated_solution": "string", // ステップバイステップソリューション
  "problem_type": "string",     // 問題タイプ識別子
  "pass_rate_72b_tir": "float", // 難易度メトリック（0.0-1.0）
  "dataset": "string"           // ソースデータセット識別子
}
```

## 🔧 設定のヒント

### 大規模データセット用
- 初期処理にはラベルベースフィルタリングを使用
- 適切な`--save-per-iteration`値を設定
- メモリ使用量を制御するため`--batch-size`を使用

### マルチGPUシステム用
- テンソル並列化を使用（`--inference-tp 2`で2GPU）
- より良いスループットのため`--vllm-batch-size`を増加
- `nvidia-smi`でGPUメモリを監視

### メモリ最適化用
- 個別のParquetファイルを使用（マージしない）
- データセットをチャンクで処理
- 利用可能なRAMに基づいて適切なバッチサイズを設定

### アップロード用
- プライベートリポジトリではYAMLメタデータを使用
- `original_README.log`で既存READMEを保持
- 既存Parquetファイルは自動スキップされる
- splitプレフィックスでファイルを整理

## 🐛 トラブルシューティング

### 一般的な問題

**メモリ不足:**
```bash
# バッチサイズを削減
--inference-batch-size 1 --vllm-batch-size 16

# テンソル並列化を使用
--inference-tp 2
```

**CUDAエラー:**
```bash
# 環境変数を設定
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_FLASH_ATTENTION=1
```

**アップロード失敗:**
```bash
# Hugging Faceトークンを確認
cat keys.json

# リポジトリ権限を確認
# リポジトリへの書き込み権限があることを確認
```

## 📝 ワークフロー例

### 完全フィルタリングワークフロー

```bash
# 1. ラベルでデータセットをフィルタリング（高速）
python OpenMathReasoningFiltering_bylabel.py \
    --filter-by-pass-rate 0.1 \
    --save-per-iteration 10000

# 2. Hugging Faceに変換してアップロード
python upload_data.py \
    --dataset_path ./results/filtered_dataset \
    --repo_id your-username/openmath-filtered

# 3. 追加ソリューション生成（オプション）
python generateFromSeed.py \
    --model Qwen/Qwen3-8B \
    --input_file ./results/filtered_dataset/train/hard_problems.json \
    --output_file ./results/generated_solutions.json
```

### カスタムLLMフィルタリングワークフロー

```bash
# 1. カスタム基準で前半を処理
python OpenMathReasoningFiltering.py \
    --inference-model Qwen/Qwen3-32B \
    --start-from-percentage 0.0 \
    --end-at-percentage 0.5

# 2. 並列で後半を処理
python OpenMathReasoningFiltering.py \
    --inference-model Qwen/Qwen3-32B \
    --start-from-percentage 0.5 \
    --end-at-percentage 1.0

# 3. 結果を結合してアップロード
python upload_data.py \
    --dataset_path ./results/combined_dataset \
    --repo_id your-username/custom-filtered
```

## 📋 要件

- Python 3.8+
- CUDAサポート付きPyTorch
- 効率的な推論用vLLM
- Hugging Face Hubアカウント
- 十分なGPUメモリ（8GB+推奨）
- 大規模データセット用: 32GB+ RAM推奨

## 📞 サポート

問題や質問については:
1. 上記のトラブルシューティングセクションを確認
2. `--help`でスクリプトパラメータを確認
3. 処理中のリソース使用量を監視
4. 適切な環境設定を確認
5. Slackで`@Junyu Liu`に連絡
