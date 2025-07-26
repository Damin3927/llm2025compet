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
import sys

# Setup logging to ensure output goes to terminal
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)





def process_dataset_folder(dataset_path):
    """
    Process the dataset folder to convert JSON files to Parquet format.
    Each JSON file gets its own Parquet file with split name as prefix.
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
        
        # Get all JSON files in this subfolder
        json_files = list(subfolder.glob("*.json"))
        
        if not json_files:
            logger.warning(f"No JSON files found in {subfolder}")
            continue
        
        logger.info(f"Found {len(json_files)} JSON files in {split_name}")
        
        # Convert each JSON file to Parquet individually with split prefix
        successful_conversions = 0
        for json_file in tqdm(json_files, desc=f"Converting {split_name} files"):
            success = convert_json_to_parquet_with_prefix(json_file, data_folder, split_name)
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

def convert_json_to_parquet_with_prefix(json_file_path, output_dir, split_name):
    """
    Convert a single JSON file to Parquet format with split name as prefix.
    """
    # Create output filename with split prefix (e.g., "train_file1.parquet")
    json_filename = Path(json_file_path).stem
    parquet_filename = f"{split_name}_{json_filename}.parquet"
    parquet_file = os.path.join(output_dir, parquet_filename)
    if os.path.exists(parquet_file):
        logger.info(f"Skipping {json_file_path}: {parquet_file} already exists")
        return True

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

def create_dataset_card(dataset_path, splits, repo_id):
    """
    Create a basic dataset card if one doesn't exist.
    """
    readme_path = dataset_path / "README.md"
    original_readme_path = dataset_path / "original_README.log"
    original_readme = None
    if original_readme_path.exists():
        with open(original_readme_path, 'r', encoding='utf-8') as f:
            original_readme = f.read()
    else:
        logger.info("No original README.md found, creating a new one")
    
    # Count files for each split
    data_folder = dataset_path / "data"
    split_info = []
    
    for split in splits:
        json_folder = dataset_path / split
        parquet_files = list(data_folder.glob(f"{split}_*.parquet"))
        
        json_count = len(list(json_folder.glob("*.json"))) if json_folder.exists() else 0
        parquet_count = len(parquet_files)
        
        split_info.append(f"- **{split}**: {json_count} JSON files, {parquet_count} Parquet files")
    
    # Create YAML metadata for custom splits
    data_files_config = {}
    for split in splits:
        data_folder = dataset_path / "data"
        split_files = list(data_folder.glob(f"{split}_*.parquet"))
        if split_files:
            # Use relative paths for the data_files configuration
            file_paths = [f"data/{f.name}" for f in split_files]
            data_files_config[split] = file_paths
    
    # Create YAML front matter with separate config for each split
    yaml_metadata = "---\nconfigs:\n"
    
    for split, files in data_files_config.items():
        yaml_metadata += f"- config_name: {split}\n"
        if len(files) == 1:
            # Single file: use string format
            yaml_metadata += f"  data_files: \"{files[0]}\"\n"
        else:
            # Multiple files: use array format
            yaml_metadata += f"  data_files:\n"
            for file_path in files:
                yaml_metadata += f"    - \"{file_path}\"\n"
    
    yaml_metadata += "---\n\n"


    if not original_readme:
        original_readme = f"""
## Dataset Description

This dataset contains processed data organized into the following splits:

{chr(10).join(split_info)}

## Dataset Structure

Each split is available in both JSON and Parquet formats:
- **JSON files**: Original format in respective subfolders (`{split}/`)
- **Parquet files**: Optimized format with split prefixes (`data/{split}_*.parquet`)

Each JSON file has a corresponding Parquet file with split name as prefix for efficient processing of large datasets.

## Usage

```python
from datasets import load_dataset

# Load specific split configurations
{chr(10).join([f'{split}_data = load_dataset("{repo_id}", "{split}")' for split in splits])}

# Or specify data_files manually for custom loading
dataset = load_dataset(
    "parquet",
    data_files={{
{chr(10).join([f'        "{split}": "data/{split}_*.parquet",' for split in splits])}
    }}
)

# Load individual files if needed
import pandas as pd
df = pd.read_parquet("data/{splits[0] if splits else 'split'}_filename.parquet")

# Load all files for a specific split
from pathlib import Path
split_files = list(Path("data").glob("{splits[0] if splits else 'split'}_*.parquet"))
for file in split_files:
    df = pd.read_parquet(file)
    # Process df...
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

    dataset_card_content = f"{yaml_metadata}# {repo_id.split('/')[-1]}\n{original_readme}"

    
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
    
    # Add verbose logging at startup
    logger.info("="*50)
    logger.info("Starting upload_data.py script")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Repository ID: {args.repo_id}")
    logger.info(f"Create dataset card: {args.create_dataset_card}")
    logger.info("="*50)
    
    # Load Hugging Face token
    try:
        with open("keys.json", "r") as f:
            keys = json.load(f)
        hf_token = keys["llm"]
        logger.info("Successfully loaded Hugging Face token")
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
        
        # Create dataset card
        create_dataset_card(dataset_path, processed_splits, args.repo_id)

        # Upload to Hugging Face
        upload_to_huggingface(dataset_path, args.repo_id, hf_token)
        
        logger.info("Dataset processing and upload completed successfully!")
        
    except Exception as e:
        logger.error(f"Process failed: {e}")
        raise

if __name__ == "__main__":
    main()