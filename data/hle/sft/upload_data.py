"""
Upload generated data to Hugging Face

This script processes JSON data files organized in subfolders, converts them to Parquet format,
and uploads them as a structured dataset to Hugging Face Hub.

Structure:
- Each subfolder in dataset_path becomes a dataset split
- JSON files in each subfolder are combined and converted to Parquet
- Both JSON and Parquet files are uploaded to maintain compatibility
"""

import os
import argparse
import json
import pandas as pd
from pathlib import Path
from huggingface_hub import HfApi
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def convert_json_to_parquet(json_file_path, output_dir):
    """
    Convert a single JSON file to Parquet format.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.warning(f"Skipping {json_file_path}: not a list format")
            return False
        
        if not data:
            logger.warning(f"Skipping {json_file_path}: empty file")
            return False
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create output filename (replace .json with .parquet)
        json_filename = Path(json_file_path).stem
        parquet_file = os.path.join(output_dir, f"{json_filename}.parquet")
        
        # Save as Parquet
        df.to_parquet(parquet_file, index=False)
        
        logger.info(f"Converted {json_file_path} -> {parquet_file} ({len(df)} examples)")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {json_file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to convert {json_file_path} to Parquet: {e}")
        return False

def process_dataset_folder(dataset_path):
    """
    Process the dataset folder to convert JSON files to Parquet format.
    Each JSON file gets its own corresponding Parquet file.
    Each subfolder becomes a dataset split.
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Create data subfolder for Parquet files
    data_folder = dataset_path / "data"
    data_folder.mkdir(exist_ok=True)
    
    # Get all subfolders (potential splits)
    subfolders = [f for f in dataset_path.iterdir() if f.is_dir() and f.name != "data"]
    
    if not subfolders:
        raise ValueError(f"No subfolders found in {dataset_path}")
    
    logger.info(f"Found {len(subfolders)} subfolders to process as splits")
    
    processed_splits = []
    
    for subfolder in subfolders:
        split_name = subfolder.name
        logger.info(f"Processing split: {split_name}")
        
        # Create corresponding data subfolder for this split
        split_data_folder = data_folder / split_name
        split_data_folder.mkdir(exist_ok=True)
        
        # Get all JSON files in this subfolder
        json_files = list(subfolder.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {subfolder}")
            continue
        
        logger.info(f"Found {len(json_files)} JSON files in {split_name}")
        
        # Convert each JSON file to Parquet individually
        successful_conversions = 0
        for json_file in tqdm(json_files, desc=f"Converting {split_name} files"):
            success = convert_json_to_parquet(json_file, split_data_folder)
            if success:
                successful_conversions += 1
        
        if successful_conversions > 0:
            processed_splits.append(split_name)
            logger.info(f"Successfully converted {successful_conversions}/{len(json_files)} files for split '{split_name}'")
        else:
            logger.warning(f"No files were successfully converted for split '{split_name}'")
    
    if not processed_splits:
        raise ValueError("No data was successfully processed")
    
    logger.info(f"Successfully processed splits: {processed_splits}")
    return processed_splits

def create_dataset_card(dataset_path, splits, repo_id):
    """
    Create a basic dataset card if one doesn't exist.
    """
    readme_path = dataset_path / "README.md"
    
    if readme_path.exists():
        logger.info("README.md already exists, skipping dataset card creation")
        return
    
    # Count files for each split
    data_folder = dataset_path / "data"
    split_info = []
    
    for split in splits:
        json_folder = dataset_path / split
        parquet_folder = data_folder / split
        
        json_count = len(list(json_folder.glob("*.json"))) if json_folder.exists() else 0
        parquet_count = len(list(parquet_folder.glob("*.parquet"))) if parquet_folder.exists() else 0
        
        split_info.append(f"- **{split}**: {json_count} JSON files, {parquet_count} Parquet files")
    
    dataset_card_content = f"""# {repo_id.split('/')[-1]}

## Dataset Description

This dataset contains processed data organized into the following splits:

{chr(10).join(split_info)}

## Dataset Structure

Each split is available in both JSON and Parquet formats:
- **JSON files**: Original format in respective subfolders (`{split}/`)
- **Parquet files**: Optimized format in data subfolders (`data/{split}/`)

Each JSON file has a corresponding Parquet file with the same base name for efficient processing of large datasets.

## Usage

```python
from datasets import load_dataset

# Load the dataset (will automatically use Parquet files)
dataset = load_dataset("{repo_id}")

# Access specific splits
{chr(10).join([f"{split}_data = dataset['{split}']" for split in splits])}

# Or load individual files if needed
import pandas as pd
df = pd.read_parquet("data/{splits[0] if splits else 'split'}/filename.parquet")
```

## File Structure

```
{repo_id.split('/')[-1]}/
├── {splits[0] if splits else 'split'}/
│   ├── file1.json
│   ├── file2.json
│   └── ...
├── data/
│   └── {splits[0] if splits else 'split'}/
│       ├── file1.parquet
│       ├── file2.parquet
│       └── ...
└── README.md
```

## Data Format

Each example contains structured data as dictionaries with various fields depending on the specific dataset content.

## Benefits

- **Memory Efficient**: Individual Parquet files allow selective loading
- **Fast Processing**: Parquet format optimized for analytical workloads
- **Flexibility**: Can load entire splits or individual files as needed
- **Compatibility**: Original JSON files preserved alongside Parquet versions
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(dataset_card_content)
    
    logger.info(f"Created dataset card at {readme_path}")

def upload_to_huggingface(dataset_path, repo_id, hf_token):
    """
    Upload the dataset folder to Hugging Face Hub.
    """
    try:
        api = HfApi(token=hf_token)
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
            logger.info(f"Repository {repo_id} is ready")
        except Exception as e:
            logger.warning(f"Could not create/access repository: {e}")
        
        # Upload the entire folder
        logger.info(f"Uploading {dataset_path} to {repo_id}")
        api.upload_folder(
            folder_path=str(dataset_path),
            repo_id=repo_id,
            repo_type="dataset",
        )
        
        logger.info(f"Successfully uploaded dataset to https://huggingface.co/datasets/{repo_id}")
        
    except Exception as e:
        logger.error(f"Failed to upload to Hugging Face: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON data to Parquet and upload to Hugging Face"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True,
        help="Path to the folder containing data subfolders"
    )
    parser.add_argument(
        "--repo_id", 
        type=str, 
        required=True,
        help="Hugging Face repository ID (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--create_dataset_card",
        action="store_true",
        help="Create a basic dataset card if README.md doesn't exist"
    )
    
    args = parser.parse_args()
    
    # Load Hugging Face token
    try:
        with open("keys.json", "r") as f:
            keys = json.load(f)
        hf_token = keys["llm"]
    except FileNotFoundError:
        logger.error("keys.json file not found. Please create it with your HF token.")
        raise
    except KeyError:
        logger.error("'llm' key not found in keys.json")
        raise
    
    dataset_path = Path(args.dataset_path)
    
    try:
        # Process JSON files and convert to Parquet
        logger.info("Starting dataset processing...")
        processed_splits = process_dataset_folder(dataset_path)
        
        # Create dataset card if requested
        if args.create_dataset_card:
            create_dataset_card(dataset_path, processed_splits, args.repo_id)
        
        # Upload to Hugging Face
        upload_to_huggingface(dataset_path, args.repo_id, hf_token)
        
        logger.info("Dataset processing and upload completed successfully!")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise

if __name__ == "__main__":
    main()