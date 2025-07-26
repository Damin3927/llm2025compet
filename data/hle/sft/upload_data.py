# upload data to huggingface

import os
import argparse
import json
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="./data/filtered_dataset")
parser.add_argument("--repo_id", type=str, default="neko-llm/SFT_OpenMathReasoning")
args = parser.parse_args()


with open("keys.json", "r") as f:
    keys = json.load(f)
hf_token = keys["llm"]

api = HfApi(token=hf_token)
api.upload_folder(
    folder_path=args.dataset_path,
    repo_id=args.repo_id,
    repo_type="dataset",
)

print(f"Uploaded data {args.dataset_path} to Hugging Face: {args.repo_id}")