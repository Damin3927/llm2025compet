# DNA チーム スクリプト集（yudai 配下・統合版）

このディレクトリは DNA チームの各種スクリプトとドキュメントを `data/dna/yudai` に集約したものです。GitHub 追跡対象はスクリプトと最小限の設定ファイルに限定し、生成物・大容量データは `.gitignore` により無視します。

## 機能一覧

- 分類（有害性3段階）: `classify_script.py`
- 思考タグ生成（<think>埋め込み）: `add_think_tags.py`, `add_think_tags_offline.py`
- DPO データ生成（V2）: `dna_dpo_v2_generator.py`
- アップロード（Hugging Face）: `upload_to_hf.py`
- 可視化（W&B）: 分類スクリプトの W&B 統合オプション
- ユーティリティ: `dna_scripts/`, `scripts/`

## セットアップ

### 1. 依存関係のインストール

プロジェクトルートで次を実行してください。

```bash
pip install -r data/dna/yudai/requirements.txt
```

### 2. 環境変数の設定

`env.example` をコピーし、OpenRouter と（必要に応じて）Hugging Face のトークンを設定します。

```bash
cp data/dna/yudai/env.example data/dna/yudai/.env
# 必要に応じて追加
# HUGGINGFACE_API_KEY="hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
```

スクリプトによっては `.env` の自動読み込みを行わないものがあります。その場合は CLI 引数や環境変数で上書きしてください。

### 3. ディレクトリ構成（抜粋）

```
data/dna/yudai/
├── add_think_tags.py
├── add_think_tags_offline.py
├── classify_script.py
├── dna_dpo_v2_generator.py
├── dna_hierarchy.json
├── upload_to_hf.py
├── dna_scripts/
├── scripts/
├── requirements.txt
└── .gitignore   # 生成物・大容量を無視
```

## 使い方

### A. 有害性分類（3段階）

```bash
python data/dna/yudai/classify_script.py \
  --start_index 0 \
  --end_index 10 \
  --model_name "meta-llama/llama-3.1-405b-instruct" \
  --hierarchy_file data/dna/yudai/dna_hierarchy.json \
  --log_dir data/dna/yudai/logs \
  --output_file data/dna/yudai/classified_0-9.jsonl
```

主な引数:
- `--hierarchy_file`: `data/dna/yudai/dna_hierarchy.json` を指定
- `--log_dir`: `data/dna/yudai/logs` を推奨

### B. <think> タグ埋め込み

```bash
python data/dna/yudai/add_think_tags.py \
  --start_index 0 \
  --end_index 10 \
  --model_name "qwen/qwen3-32b" \
  --log_dir data/dna/yudai/logs \
  --output_file data/dna/yudai/think_tagged_0-9.jsonl
```

環境変数は `.env`（`data/dna/yudai/.env`）または `--api_key` で指定可能です。

### C. DPO データ生成（V2）

```bash
python data/dna/yudai/dna_dpo_v2_generator.py --workers 4
```

出力（例）:
- リアルタイム: `data/dna/yudai/processed/realtime_results.jsonl`
- 中間: `data/dna/yudai/processed/intermediate_results_50.jsonl`
- 最終: `data/dna/yudai/processed/dataset_v2_mvp.jsonl`

### D. Hugging Face へアップロード

```bash
python data/dna/yudai/upload_to_hf.py --files "data/dna/yudai/think_tagged_*.jsonl" \
  --dataset-name neko-llm/dna_dpo_hh-rlhf
```

事前に `HUGGINGFACE_API_KEY` を `.env` または環境変数で設定してください。

### E. W&B 連携（任意）

```bash
wandb login
python data/dna/yudai/classify_script.py --start_index 0 --end_index 100 --wandb_project dna-harm
```

## 注意事項

- 本ディレクトリは yudai チーム向けに隔離されています。他ディレクトリに変更が波及しないように運用してください。
- 生成物（.jsonl など）や大容量ファイルは `.gitignore` で無視されます。
- 既存のドキュメント（個別 README 群）の内容は本 README に統合済みです。
