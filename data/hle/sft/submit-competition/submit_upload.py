"""
HLE の SFT データセットを 松尾研コンペの提出用 Hugging Face にアップロードするための Script。
"""

import os
from tqdm import tqdm
import json
import datasets
from huggingface_hub import upload_file, create_repo, HfApi

# --- Hugging Face Upload Settings: Repository Name ---
OUTPUT_DATASET_ORG = "weblab-llm-competition-2025-bridge"

# チーム ねこ🐱 Prefix
PREFIX_NAME = "neko-prelim-"

# --- Target Dataset ID ---
TARGET_DATASET_LIST = [
  # id: データセットID, config: データセットのSubset名, split: データセットのSplit名
  # 学習Configを参照: https://www.notion.so/Config-2479dd6b4cc28002b151f788920d51b4
  {"id": "neko-llm/HLE_SFT_OlymMATH",  "split": "train"},
  {"id": "neko-llm/HLE_SFT_medical",  "split": "train"},
  {"id": "neko-llm/HLE_SFT_PHYBench",  "split": "train"},
  {"id": "neko-llm/HLE_SFT_OlympiadBench",  "split": "train"},
  {"id": "neko-llm/HLE_SFT_biology",  "split": "train"},
  {"id": "neko-llm/HLE_SFT_OpenMathReasoning", "config": "cot", "split": "train"},
  {"id": "neko-llm/HLE_SFT_MixtureOfThoughts", "config": "MoT_science", "split": "train"},
  {"id": "neko-llm/HLE_SFT_PhysReason",  "split": "train"},
  {"id": "neko-llm/HLE_SFT_GPQA_Diamond",  "split": "train"},
  {"id": "neko-llm/HLE_SFT_OpenThoughts-114k",  "split": "train"},
  {"id": "neko-llm/HLE_SFT_general", "config": "medcal", "split": "train"},
  {"id": "neko-llm/HLE_RL_Olympiadbench-v2",  "split": "train"},
  {"id": "neko-llm/HLE_SFT_LIMO",  "split": "train"},
  {"id": "neko-llm/HLE_SFT_LIMO-v2",  "split": "train"},
]

# 要件定義
# 1. 1つずつデータセットを読み込む。
# 2. データセットを指定の名前のフォーマットでHugging Faceにアップロードする。
  # SUBMIT_DATASET_FORMAT/{データセット名}
  # ただし、neko-llm/ というPrefixは削る。

# Read Dataset -> Upload Dataset
for dataset_id in TARGET_DATASET_LIST:

  print(f"🔍 Reading dataset: {dataset_id['id']}")

  # データセットの読み込み: データセット名、config、splitを指定して読み込む。
  if dataset_id.get('config'):
    dataset = datasets.load_dataset(dataset_id['id'], dataset_id['config'], split=dataset_id['split'])
  else:
    dataset = datasets.load_dataset(dataset_id['id'], split=dataset_id['split'])

  print(f"✅ Dataset read successfully: {dataset_id['id']}")
  print(f"Dataset size: {len(dataset)}")

  # データセット名からneko-llm/ Prefixを削る
  dataset_name = dataset_id['id'].replace('neko-llm/', '')

  # JSONLファイル名を作成
  jsonl_filename = f"{PREFIX_NAME}{dataset_name}.jsonl"

  # 完全なリポジトリIDを作成
  full_repo_id = f"{OUTPUT_DATASET_ORG}/{jsonl_filename.replace('.jsonl', '')}"

  print(f"💾 Saving dataset to JSONL: {jsonl_filename}")

  # データセットをJSONL形式で保存
  with open(jsonl_filename, 'w', encoding='utf-8') as f:
    for item in tqdm(dataset, desc="Saving to JSONL"):
      json_line = json.dumps(item, ensure_ascii=False)
      f.write(json_line + '\n')

  print(f"✅ JSONL file saved: {jsonl_filename}")
  print(f"📤 Uploading to Hugging Face: {full_repo_id}")

  try:
    # リポジトリが存在するかチェック
    api = HfApi()
    try:
      api.repo_info(repo_id=full_repo_id, repo_type="dataset")
      print(f"📁 Repository exists: {full_repo_id}")
    except Exception:
      print(f"🆕 Creating new repository: {full_repo_id}")
      create_repo(
        repo_id=full_repo_id,
        repo_type="dataset",
        private=False
      )
      print(f"✅ Repository created: {full_repo_id}")

    # JSONLファイルをHugging Faceにアップロード
    upload_file(
      path_or_fileobj=jsonl_filename,
      path_in_repo=jsonl_filename,
      repo_id=full_repo_id,
      repo_type="dataset"
    )

    print(f"✅ Successfully uploaded: {jsonl_filename}")
    print(f"🔗 Dataset URL: https://huggingface.co/datasets/{full_repo_id}")

  except Exception as e:
    print(f"❌ Failed to upload {dataset_id['id']}: {str(e)}")
    print(f"🔍 Error details: {type(e).__name__}")
    continue

  # アップロード後、ローカルファイルを削除（オプション）
  try:
    os.remove(jsonl_filename)
    print(f"🗑️ Cleaned up local file: {jsonl_filename}")
  except Exception as e:
    print(f"⚠️ Could not remove local file {jsonl_filename}: {str(e)}")

  print("-" * 80)

print("🎉 All datasets processing completed!")
