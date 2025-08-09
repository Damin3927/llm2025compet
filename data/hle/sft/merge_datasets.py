"""
aim: concatenate all the datasets into a single file
input: a json file containing a list of dictionaries, each dictionary contains the dataset name and the config, split to be included in the final file
output: a json file containing the concatenated dataset
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict, Any


def load_config(config_path: str) -> List[Dict[str, Any]]:
    """Load the configuration file containing dataset specifications."""
    with open(config_path, 'r') as f:
        return json.load(f)


def merge_datasets(config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge datasets according to configuration.
    The config should be the format:
    [
        {
            "dataset_name": "sft_dataset_v1",
            "config": config_name,
            "split": split_name
        }, ...
    ]
    """
    merged_data = []
    
    for dataset_config in config:
        dataset_name = dataset_config['dataset_name']
        config_name = dataset_config.get('config', None)
        split = dataset_config.get('split', 'train')
        
        print(f"Loading {dataset_name} (config: {config_name}, split: {split})")
        
        try:
            # Load dataset
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split=split)
            else:
                dataset = load_dataset(dataset_name, split=split)
            
            # Convert to list of dictionaries
            dataset_data = list(dataset)
            merged_data.extend(dataset_data)
            
            print(f"Added {len(dataset_data)} examples from {dataset_name}")
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue
    
    return merged_data


def save_merged_dataset(data: List[Dict[str, Any]], output_path: str):
    """Save the merged dataset to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved merged dataset with {len(data)} examples to {output_path}")


def main(hf_key: str):
    os.environ["HF_TOKEN"] = hf_key
    parser = argparse.ArgumentParser(description="Merge multiple datasets into a single JSON file")
    parser.add_argument("config", help="Path to configuration JSON file")
    parser.add_argument("output", help="Path to output JSON file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Merge datasets
    merged_data = merge_datasets(config)
    
    # Save merged dataset
    save_merged_dataset(merged_data, args.output)


if __name__ == "__main__":
    with open("./keys.json", "r") as f:
        keys = json.load(f)
    main(keys)
