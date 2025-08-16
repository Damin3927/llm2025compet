from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, HfFolder

# ローカルの JSONL をロード
dataset = load_dataset("json", data_files={"train": "vanilla_with_cot_vllm_cot_extracted.jsonl"})

# push_to_hub（datasetをそのままアップロード）
dataset.push_to_hub("neko-llm/dna_vanilla_harmful")