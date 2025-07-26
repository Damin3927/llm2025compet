# OpenMathReasoning SFT パイプライン

このフォルダには、OpenMathReasoningデータセットを教師ありファインチューニング（SFT）用に処理するためのスクリプトとツールが含まれています。パイプラインには、フィルタリング、処理、Hugging Faceへのデータセットアップロードが含まれます。

## 📁 ディレクトリ構造

```
data/hle/sft/
├── OpenMathReasoningFiltering.py          # メインLLMベースフィルタリングスクリプト
├── OpenMathReasoningFiltering_bylabel.py  # ラベルベースフィルタリング（高速）
├── generateFromSeed.py                    # シード問題からソリューション生成
├── upload_data.py                         # データセットをParquetに変換してHFにアップロード
├── run_filter.sh                          # LLMフィルタリング用SLURMスクリプト
├── run_label_filter.sh                    # ラベルフィルタリング用SLURMスクリプト
├── run_filter_dart_math.sh               # DART-Math用SLURMスクリプト
├── keys.json                              # APIキー設定
├── keys.json.example                      # APIキーテンプレート
├── results/                               # 処理済みデータの出力ディレクトリ
└── Mixture-of-Thoughts/                   # MoT処理スクリプト
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

### 🤖 LLMベースフィルタリング

**スクリプト:** `OpenMathReasoningFiltering.py`  
**目的:** カスタムフィルタリング基準のためのLLM推論使用

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
**目的:** シード数学問題から新しいソリューションを生成

```bash
# Qwenモデルを使用したソリューション生成
python generateFromSeed.py \
    --model Qwen/Qwen3-8B \
    --input_file seed_problems.json \
    --output_file generated_solutions.json \
    --max_tokens 1024 \
    --temperature 0.7
```

**主要パラメータ:**
- `--model`: 生成に使用するモデル
- `--input_file`: シード問題を含むJSONファイル
- `--output_file`: 生成されたソリューションの保存先
- `--max_tokens`: 生成あたりの最大トークン数
- `--temperature`: サンプリング温度（0.0 = 決定論的、1.0 = ランダム）

---

### 📤 Hugging Faceへのアップロード

**スクリプト:** `upload_data.py`  
**目的:** JSONファイルをParquetに変換してHugging Face Hubにアップロード

```bash
# 自動Parquet変換による基本アップロード
python upload_data.py \
    --dataset_path ./results/filtered_dataset \
    --repo_id your-username/dataset-name

# データセットカード生成付き
python upload_data.py \
    --dataset_path ./results/filtered_dataset \
    --repo_id your-username/dataset-name \
    --create_dataset_card
```

**期待される入力構造:**
```
dataset_path/
├── train/
│   ├── file1.json
│   ├── file2.json
│   └── ...
├── validation/
│   ├── file1.json
│   └── ...
└── test/
    └── file1.json
```

**出力構造:**
```
dataset_path/
├── train/ (元のJSONファイル)
├── validation/ (元のJSONファイル)  
├── test/ (元のJSONファイル)
├── data/
│   ├── train/ (変換されたParquetファイル)
│   ├── validation/ (変換されたParquetファイル)
│   └── test/ (変換されたParquetファイル)
└── README.md (オプション)
```

**機能:**
- 各JSONファイルを個別のParquetファイルに変換（メモリ効率的）
- 互換性のため元のJSONファイルを保持
- フォルダ構造に基づいてデータセット分割を自動作成
- ファイル統計付きデータセットカードを生成

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

**DART-Mathフィルタリング:**
```bash
sbatch run_filter_dart_math.sh
```

各スクリプトには以下が含まれます:
- GPUリソース割り当て
- 環境セットアップ
- CUDA設定
- メモリ最適化設定

## 📊 データスキーマ

処理されたデータセットは以下のスキーマに従います:

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
    --repo_id your-username/openmath-filtered \
    --create_dataset_card

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
