#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi
import argparse
from dotenv import load_dotenv
import os
import glob

load_dotenv()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='データセット名（phybench, physreason）')
    args = parser.parse_args()
    
    # 最新のgeneratedファイルを探す
    generated_files = sorted(glob.glob(f"data/{args.dataset}/generated/generated_cot_*.jsonl"))
    if not generated_files:
        raise FileNotFoundError(f"No generated files found for {args.dataset}")
    file_path = generated_files[-1]  # 最新のファイル
    
    # repo_id決定
    dataset_map = {
        'phybench': 'neko-llm/HLE_SFT_PHYBench',
        'physreason': 'neko-llm/HLE_SFT_PhysReason'
    }
    repo_id = dataset_map[args.dataset]
    
    # JSONLを読み込み
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    
    # Parquetに変換
    df = pd.DataFrame(data)
    parquet_path = Path(file_path).with_suffix('.parquet')
    df.to_parquet(parquet_path)
    
    # HFにアップロード
    hf_token = os.getenv("HF_TOKEN")
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo=parquet_path.name,
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    # README.mdもアップロード
    readme_path = Path(f"data/{args.dataset}/README.md")
    if readme_path.exists():
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset"
        )
    
    print(f"アップロード完了: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()