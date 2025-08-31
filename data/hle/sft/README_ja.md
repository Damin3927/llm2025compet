# SFTデータ処理パイプライン

このリポジトリには、教師あり微調整（SFT）用の多様なデータセットを処理するための包括的なツールとスクリプトが含まれています。パイプラインは、数学、科学、医学、法律、歴史、化学、一般推論など複数のドメインをカバーしています。

## 📁 ディレクトリ構造

```
data/hle/sft/
├── 🧮 MixtureOfThoughts/     # Mixture of Thoughtsデータセット処理
├── 📐 OpenMath/              # OpenMath推論とフィルタリング
├── 🧬 biology/               # 生物学Tree of Thoughtsデータ
├── ⚗️ chemistry/             # ChemPile化学Q&A抽出
├── 💭 general/               # 一般推論（StrategyQA、MedCalc）
├── 📚 humanity/              # 人文科学データ（歴史、法律）
├── 🏥 medical/               # 医療推論（ReasonMD）
├── 📊 results/               # 処理済み出力ディレクトリ
├── 🔧 コアスクリプト/         # メイン処理ツール
│   ├── OpenMathReasoningFiltering.py      # LLMベースフィルタリング
│   ├── OpenMathReasoningFiltering_bylabel.py  # ラベルベースフィルタリング
│   ├── generateFromSeed.py               # シードからの解答生成
│   ├── upload_data.py                    # HuggingFaceアップロードツール
│   ├── difficulty_scorer.py              # 多指標難易度評価
│   ├── length_selector.py                # 長さベースデータ選択
│   └── merge_datasets.py                 # データセット結合ユーティリティ
└── 🚀 Bashスクリプト/        # 自動化スクリプト
    ├── run_filter.sh                      # LLMフィルタリングSLURMスクリプト
    ├── run_label_filter.sh                # ラベルフィルタリングSLURMスクリプト
    ├── run_length_selector.sh             # データ選択SLURMスクリプト
    └── run_upload_data.sh                 # アップロードSLURMスクリプト
```

## 🎯 データセットカバレッジ

### 数学と科学
- **MixtureOfThoughts**: コード、数学、科学にわたる348Kサンプル（明示的な推論トレース付き）
- **OpenMath**: 難易度フィルタリング付き390万の数学問題（CoTとGenSelect分割）
- **Biology**: 生物学的質問のためのTree of Thoughts推論
- **Chemistry**: クリーニング済み化学Q&AペアのChemPileデータセット

### 人文科学と医学
- **History**: LLM生成推論チェーン付き歴史Q&A
- **Law**: CoT_Legal_Issues_And_Lawsから難易度（1-5スケール）でフィルタリングされた法律の質問
- **Medical/ReasonMD**: VLLM抽出された簡潔な回答を含む医療推論
- **General**: 一般推論と医療計算のためのStrategyQAとMedCalc

## 🚀 クイックスタートガイド

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

### 2. APIキー設定

```bash
# キーテンプレートをコピーして編集
cp keys.json.example keys.json

# Hugging Faceトークンを追加
{
  "llm": "your_huggingface_token_here"
}
```

## 🔄 完全な処理ワークフロー

### ステップ1: データ選択とフィルタリング

データセットのサイズと要件に基づいて、3つのアプローチから選択：

#### オプションA: 高速な長さベース選択（大規模データセット推奨）
```bash
python length_selector.py \
    --input "dataset-name" \
    --total_samples 10000 \
    --output "selected_seeds.json"
```

#### オプションB: 精密な難易度ベース選択（小〜中規模データセット用）
```bash
python difficulty_scorer.py \
    --input "dataset-name" \
    --output "difficulty_scores.json" \
    --max_samples 50000
```

#### オプションC: ラベルベース超高速フィルタリング（事前ラベル付きデータ用）
```bash
python OpenMathReasoningFiltering_bylabel.py \
    --filter-by-pass-rate 0.1 \
    --save-per-iteration 10000
```

### ステップ2: ドメイン固有の処理

ドメイン要件に従ってデータを処理：

#### 数学（MixtureOfThoughts）
```bash
cd MixtureOfThoughts
python process_mot_dataset.py
# 出力: processed_mot_data/ にMoT_code.json、MoT_math.json、MoT_science.json
```

#### OpenMath推論
```bash
cd OpenMath
python OpenMathReasoningFiltering.py \
    --inference-model Qwen/Qwen3-8B \
    --filter-by-pass-rate 0.3
```

#### 医療（ReasonMD）
```bash
cd medical/reasonMD
# ステップ1: データ選択
python reasonmd_selector.py --target_samples 1000
# ステップ2: VLLMで変換
python convert_reasonmd.py
```

#### 法律
```bash
cd humanity/law
python format_law.py --difficulty 4 --num_samples 200
```

#### 歴史
```bash
cd humanity/history
python convert_history.py --input input.json --output output.json
```

#### 化学
```bash
cd chemistry/chempile
bash run_chempile_extraction.sh
python remove_error_items.py
```

### ステップ3: データ拡張（オプション）
```bash
# シード問題から新しい解答を生成
python generateFromSeed.py \
    --model Qwen/Qwen3-32B \
    --input_file selected_seeds.json \
    --output_file expanded_solutions.json \
    --max_tokens 4096
```

### ステップ4: Hugging Faceへのデータセットアップロード
```bash
# Parquetに変換してアップロード
python upload_data.py \
    --dataset_path ./results/processed_dataset \
    --repo_id your-username/sft-dataset-name \
    --create_dataset_card
```

## 📊 統一出力形式

すべての処理済みデータセットは一貫したスキーマに従います：

```json
{
  "id": "{dataset}_{index}",
  "question": "元の質問または問題",
  "output": "<think>ステップバイステップの推論プロセス</think>\n最終回答",
  "answer": "簡潔な最終回答"
}
```

### フィールドの説明
- **id**: データセット接頭辞付きの一意の識別子
- **question**: データセットからの元の質問/問題
- **output**: 結合された推論（`<think>`タグ内）と回答
- **answer**: 検証用のスタンドアロン最終回答

## 🛠️ コア処理スクリプト

### 📊 難易度スコアラー（`difficulty_scorer.py`）
複数の指標を使用して質問の難易度を評価：
- ゴールド回答の平均対数確率
- 複数モデル間のアンサンブル精度
- IRT難易度パラメータβ

```bash
python difficulty_scorer.py \
    --input "dataset-name" \
    --output "scores.json" \
    --max_samples 10000
```

### 📏 長さセレクター（`length_selector.py`）
回答長分布に基づく高速データ選択：
- 半ガウス分布（長い回答を優先）
- 実データに基づく動的ビニング
- 大規模データセット用のストリーミング処理

```bash
python length_selector.py \
    --input "dataset-name" \
    --total_samples 5000 \
    --num_bins 6 \
    --curve_sharpness 3.0
```

### 🌱 シードジェネレーター（`generateFromSeed.py`）
シード問題から新しい解答を生成：
```bash
python generateFromSeed.py \
    --model Qwen/Qwen3-32B \
    --input_file seeds.json \
    --output_file generated.json
```

### 📤 アップロードツール（`upload_data.py`）
JSONをParquetに変換してHugging Faceにアップロード：
```bash
python upload_data.py \
    --dataset_path ./results \
    --repo_id username/dataset \
    --create_dataset_card
```

## 🖥️ HPC用SLURMスクリプト

クラスター/HPC環境用：

```bash
# ラベルベースフィルタリング
sbatch run_label_filter.sh

# LLMベースフィルタリング
sbatch run_filter.sh

# 長さベース選択
sbatch run_length_selector.sh

# Hugging Faceへアップロード
sbatch run_upload_data.sh
```

## 📋 ワークフロー例

### 完全なマルチドメインSFTデータセット
```bash
# 1. 数学を処理
cd MixtureOfThoughts && python process_mot_dataset.py
cd ../OpenMath && python OpenMathReasoningFiltering_bylabel.py

# 2. 科学を処理
cd ../biology && python ToT/transfer_data.py
cd ../chemistry && bash chempile/run_chempile_extraction.sh

# 3. 人文科学を処理
cd ../humanity/law && python format_law.py
cd ../history && python convert_history.py

# 4. 医療を処理
cd ../../medical/reasonMD && python reasonmd_selector.py && python convert_reasonmd.py

# 5. 結合してアップロード
cd ../..
python merge_datasets.py --input_dirs results/* --output merged_dataset
python upload_data.py --dataset_path merged_dataset --repo_id username/complete-sft
```

### クイックドメイン固有データセット
```bash
# 数学のみの場合
cd OpenMath
python OpenMathReasoningFiltering_bylabel.py --filter-by-pass-rate 0.1
cd ..
python upload_data.py --dataset_path OpenMath/results --repo_id username/math-sft
```

## 🔧 設定のヒント

### 大規模データセット（>100Kサンプル）用
- ラベルベースまたは長さベースフィルタリングを使用
- 適切な`--save-per-iteration`値を設定
- パーセンテージ範囲で並列処理

### マルチGPUシステム用
- テンソル並列を使用: `--inference-tp 2`
- バッチサイズを増加: `--vllm-batch-size 32`
- `nvidia-smi`でGPUメモリを監視

### メモリ最適化用
- データセットをチャンクで処理
- 利用可能な場所でストリーミングを使用
- Parquetファイルは分離したまま（マージしない）

## 🐛 トラブルシューティング

### メモリ不足
```bash
# バッチサイズを削減
--inference-batch-size 1 --vllm-batch-size 16

# テンソル並列を使用
--inference-tp 2
```

### CUDAエラー
```bash
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_FLASH_ATTENTION=1
```

### アップロード失敗
```bash
# HFトークンを確認
cat keys.json

# リポジトリ権限を確認
# リポジトリへの書き込みアクセスを確保
```

## 📚 データセットソース

- **MixtureOfThoughts**: `open-r1/Mixture-of-Thoughts`
- **OpenMath**: `nvidia/OpenMathReasoning`
- **Law**: `moremilk/CoT_Legal_Issues_And_Laws`
- **ReasonMD**: `lingshu-medical-mllm/ReasonMed`、`neko-llm/CoT_Medicine`
- **カスタム**: 歴史、生物学ToT、化学ChemPile

## 📋 要件

- Python 3.8+
- CUDAサポート付きPyTorch
- 効率的な推論用vLLM
- Hugging Face Hubアカウント
- 8GB+ VRAMのGPU（推奨）
- 大規模データセット用32GB+ RAM

## 📞 サポート

問題や質問については：
1. 上記のトラブルシューティングセクションを確認
2. スクリプトヘルプを確認: `python script.py --help`
3. 処理中のリソース使用量を監視
4. 環境設定を確認
5. Slackで@Junyu Liuに連絡

## 📝 ライセンス

このパイプラインは研究および教育目的で提供されています。データ使用権については、個々のデータセットライセンスを参照してください。