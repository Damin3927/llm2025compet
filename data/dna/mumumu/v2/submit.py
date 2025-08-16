from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, HfFolder

# ローカルの JSONL をロード
upload_file = "selected_cot_1200_extracted.jsonl"
dataset = load_dataset("json", data_files={"train": upload_file})

# push_to_hub（datasetをそのままアップロード）
dataset.push_to_hub("neko-llm/dna_vanilla_harmful_v2")